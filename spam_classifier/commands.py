import mlflow
import fire
import git
from pathlib import Path
from hydra import compose, initialize
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger

from lightning.pytorch.callbacks import EarlyStopping
from .download import download_sms_data
from .data.dataset import SMSDataModule
from .models.lstm_classifier import SMSLSTMClassifier


def train(config_name: str = "config", overrides: list[str] | None = None):
    if overrides is None:
        overrides = []

    with initialize(config_path="../configs", version_base=None, job_name="sms_train"):
        cfg = compose(config_name=config_name, overrides=overrides)

    config = OmegaConf.to_container(cfg, resolve=True)
    print("Config loaded")
    print(f"MLflow tracking URI: {config['mlflow']['tracking_uri']}")

    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Create'{experiment_name}' с ID: {experiment_id}")
    else:
        print(f"'{experiment_name}' exist (ID: {experiment.experiment_id})")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run started: {run_id}")

        Path("outputs/models/lstm").mkdir(parents=True, exist_ok=True)

        data_dir = Path(config["paths"]["data"])
        download_sms_data(data_dir)
        data_module = SMSDataModule(
            data_dir=data_dir,
            batch_size=config["data"]["batch_size"],
        )
        data_module.setup()
        print(
            f"Dataset sizes: train={len(data_module.train_dataset)}, val={len(data_module.val_dataset)}"
        )
        print(f"Vocab size: {len(data_module.vocab)}")

        model = SMSLSTMClassifier(
            vocab_size=len(data_module.vocab),
            **config["model"],
        )

        hparams = {
            "vocab_size": len(data_module.vocab),
            "train_size": len(data_module.train_dataset),
            "val_size": len(data_module.val_dataset),
            **config["model"],
            **config["trainer"],
            **config["data"],
        }
        mlflow.log_params(hparams)

        try:
            repo = git.Repo(search_parent_directories=True)
            mlflow.log_param("git_commit", repo.head.object.hexsha)
            mlflow.log_param("git_branch", repo.active_branch.name)
            print(f"GIT: {repo.active_branch.name} @ {repo.head.object.hexsha[:8]}")
        except Exception as e:
            print(f"GIT info not available: {e}")
            mlflow.log_param("git_commit", "no_git")

        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_id=run_id,
        )
        print(f"MLflow Logger created with run_id: {run_id}")

        checkpoint_callback = ModelCheckpoint(
            dirpath="outputs/models/lstm",
            filename="lstm-{epoch:02d}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=3,
        )

        trainer = pl.Trainer(
            max_epochs=config["trainer"]["max_epochs"],
            accelerator="cpu",
            logger=mlf_logger,
            callbacks=[
                checkpoint_callback,
                RichProgressBar(),
                EarlyStopping(monitor="val_f1", mode="max", patience=8, verbose=True),
            ],
            log_every_n_steps=10,
        )

        print("Starting training...")
        trainer.fit(model, datamodule=data_module)

        print("Running test...")
        test_results = trainer.test(model, datamodule=data_module)

        if test_results:
            for metric_name, metric_value in test_results[0].items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

        if checkpoint_callback.best_model_path:
            mlflow.log_artifact(
                checkpoint_callback.best_model_path, artifact_path="models"
            )
            print(f"Best model logged: {checkpoint_callback.best_model_path}")

        print("Training finished")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"MLflow Run ID: {run_id}")
        print(f"View results: mlflow ui  (or open {tracking_uri})")


def download():
    download_sms_data(Path("data"))


if __name__ == "__main__":
    fire.Fire({"train": train, "download": download})
