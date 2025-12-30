import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import MLFlowLogger
import hydra
from omegaconf import DictConfig
from pathlib import Path

from .data.dataset import SMSDataModule
from .models.lstm_classifier import SMSLSTMClassifier


@hydra.main(config_path="configs", config_name="base", version_base=None)
def train(config: DictConfig):
    data_module = SMSDataModule(data_dir=config.paths.data, **config.data)
    data_module.setup()

    vocab_size = len(data_module.vocab)
    config.model.vocab_size = vocab_size

    model = SMSLSTMClassifier(vocab_size=vocab_size, **config.model)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(dict(config.model))
        mlflow.log_param("vocab_size", vocab_size)

        mlflow_logger = MLFlowLogger(
            experiment_name=config.mlflow.experiment_name,
            tracking_uri=config.mlflow.tracking_uri,
        )

        checkpoint = ModelCheckpoint(
            dirpath=str(Path("outputs/models")),
            filename="sms-lstm-{epoch:02d}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
        )

        trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            logger=mlflow_logger,
            callbacks=[checkpoint, RichProgressBar()],
            log_every_n_steps=10,
        )

        trainer.fit(model, data_module)
        trainer.test(model, data_module)

        mlflow.pytorch.log_model(model, "lstm_model")


if __name__ == "__main__":
    train()
