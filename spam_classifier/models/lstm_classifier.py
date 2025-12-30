import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score


class SMSLSTMClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.4,
        bidirectional: bool = True,
        lr: float = 5e-4,
        pos_weight: float = 4.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(lstm_output_dim, 1)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        self.train_acc = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        mask = (x != 0).float().unsqueeze(-1)
        masked_out = lstm_out * mask
        summed = masked_out.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths

        dropped = self.dropout(pooled)
        logits = self.classifier(dropped).squeeze(-1)
        return logits

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)

    def on_train_epoch_end(self):
        self.train_acc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.val_acc.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        self.test_acc.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )
