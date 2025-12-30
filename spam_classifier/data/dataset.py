import re
from collections import Counter, OrderedDict
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def simple_tokenizer(text: str, max_length: int = 100):
    text = text.lower()
    text = re.sub(r"\b(?:\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,})\b", " <phone> ", text)
    text = re.sub(r"£|\$|€", " <money> ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()[:max_length]

    return tokens


class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length: int = 100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = simple_tokenizer(self.texts[idx], self.max_length)
        encoded = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        padded = encoded + [self.vocab["<pad>"]] * (self.max_length - len(encoded))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


class SMSDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

        self.train_dataset: SMSDataset | None = None
        self.val_dataset: SMSDataset | None = None
        self.test_dataset: SMSDataset | None = None
        self.vocab: dict | None = None

    def setup(self, stage=None):
        data_path = self.data_dir / "raw" / "sms.tsv"
        df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])
        df["label"] = (df["label"] == "spam").astype(int)

        print(f"{len(df)} SMS ({df['label'].mean():.1%} spam)")

        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            df["text"],
            df["label"],
            test_size=0.36,
            random_state=42,
            stratify=df["label"],
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels,
        )

        word_counts = Counter()
        for text in train_texts:
            word_counts.update(simple_tokenizer(text))

        self.vocab = OrderedDict({"<pad>": 0, "<unk>": 1})
        self.vocab.update(
            {
                word: idx + 2
                for idx, (word, _) in enumerate(word_counts.most_common(10000))
            }
        )

        print(f"Vocab size: {len(self.vocab)} (top-10k train)")

        self.train_dataset = SMSDataset(
            train_texts.tolist(), train_labels.tolist(), self.vocab
        )
        self.val_dataset = SMSDataset(
            val_texts.tolist(), val_labels.tolist(), self.vocab
        )
        self.test_dataset = SMSDataset(
            test_texts.tolist(), test_labels.tolist(), self.vocab
        )

        print(
            f"Datasets: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)


if __name__ == "__main__":
    dm = SMSDataModule("data", batch_size=32)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
