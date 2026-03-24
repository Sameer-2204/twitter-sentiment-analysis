"""
Train a DistilBERT sentiment classifier with HuggingFace Transformers.

This script:
1. Loads the train / validation CSV splits.
2. Tokenizes tweets with ``DistilBertTokenizer``.
3. Wraps examples in a PyTorch ``Dataset`` and ``DataLoader``.
4. Fine-tunes ``DistilBertForSequenceClassification`` with AdamW.
5. Tracks train loss, validation loss, and validation accuracy.
6. Saves the best model, tokenizer, and training history.

Usage
-----
    py -3 -m scripts.train_distilbert
    py -3 -m scripts.train_distilbert --epochs 4 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg  # noqa: E402
from scripts.training_config import TrainingConfig  # noqa: E402
from scripts.training_utils import (  # noqa: E402
    encode_label_series,
    load_and_prepare_data,
    save_training_history,
    set_all_seeds,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MODEL_NAME = "distilbert-base-uncased"


class TweetDataset(Dataset):
    """PyTorch dataset for tokenized tweet classification examples."""

    def __init__(
        self,
        texts,
        labels,
        tokenizer: DistilBertTokenizer,
        max_length: int,
    ) -> None:
        """Store tweet texts, labels, and tokenization settings."""
        self.texts = texts.astype(str).tolist()
        self.labels = labels.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return tokenized inputs and numeric label for one example."""
        encoding = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def create_dataloaders(
    tokenizer: DistilBertTokenizer,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, int]:
    """Load raw data and wrap it in train / validation data loaders."""
    X_train, y_train, X_valid, y_valid = load_and_prepare_data(
        model_type="transformer",
        config=config,
    )
    if X_valid is None or y_valid is None:
        raise ValueError("Validation split is required for DistilBERT training.")

    encoded_y_train, encoded_y_valid, label_mapping = encode_label_series(
        y_train=y_train,
        y_valid=y_valid,
    )
    num_classes = len(label_mapping)
    if num_classes != len(config.sentiment_labels):
        logger.warning(
            (
                "Config expects %d classes but data contains %d classes. "
                "Using the data-derived class count for DistilBERT training."
            ),
            len(config.sentiment_labels),
            num_classes,
        )
    logger.info("Detected %d target classes for DistilBERT.", num_classes)

    train_dataset = TweetDataset(
        texts=X_train,
        labels=pd.Series(encoded_y_train),
        tokenizer=tokenizer,
        max_length=config.max_sequence_length,
    )
    valid_dataset = TweetDataset(
        texts=X_valid,
        labels=pd.Series(encoded_y_valid),
        tokenizer=tokenizer,
        max_length=config.max_sequence_length,
    )

    train_batch_size = config.batch_size
    valid_batch_size = max(config.batch_size * 2, config.batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
    )

    logger.info(
        (
            "Created dataloaders with %d training batches and %d validation "
            "batches (batch_size=%d/%d)."
        ),
        len(train_loader),
        len(valid_loader),
        train_batch_size,
        valid_batch_size,
    )
    return train_loader, valid_loader, num_classes


def build_model_and_tokenizer(
    num_labels: int,
    device: torch.device,
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizer]:
    """Load the DistilBERT model and tokenizer."""
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )
    model.to(device)
    logger.info("Loaded %s on device=%s", MODEL_NAME, device)
    return model, tokenizer


def evaluate(
    model: DistilBertForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
    epoch_index: int,
    num_epochs: int,
) -> tuple[float, float]:
    """Run one validation epoch and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch_index}/{num_epochs} [valid]",
        leave=False,
    )

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(total_correct / max(total_examples, 1)):.4f}",
            )

    average_loss = total_loss / max(len(data_loader), 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def train(
    config: Optional[TrainingConfig] = None,
) -> DistilBertForSequenceClassification:
    """Run end-to-end DistilBERT training and save all artefacts."""
    config = config or TrainingConfig(
        batch_size=cfg.DistilBERT.BATCH_SIZE,
        epochs=cfg.DistilBERT.EPOCHS,
        learning_rate=cfg.DistilBERT.LEARNING_RATE,
    )
    config.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("DISTILBERT TRAINING - START")
    logger.info("=" * 60)
    logger.info("Using device: %s", device)
    set_all_seeds(config.random_seed)

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    train_loader, valid_loader, num_classes = create_dataloaders(
        tokenizer=tokenizer,
        config=config,
    )
    model, tokenizer = build_model_and_tokenizer(
        num_labels=num_classes,
        device=device,
    )

    total_steps = len(train_loader) * config.epochs
    warmup_steps = max(1, math.ceil(total_steps * 0.1))

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        "Optimizer and scheduler ready (total_steps=%d, warmup_steps=%d).",
        total_steps,
        warmup_steps,
    )

    history: Dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        epoch_index = epoch + 1
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch_index}/{config.epochs} [train]",
            leave=False,
        )

        for step, batch in enumerate(progress_bar, start=1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            average_loss = running_loss / step
            progress_bar.set_postfix(loss=f"{average_loss:.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=valid_loader,
            device=device,
            epoch_index=epoch_index,
            num_epochs=config.epochs,
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_accuracy=%.4f",
            epoch_index,
            config.epochs,
            train_loss,
            val_loss,
            val_accuracy,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(config.distilbert_model_dir)
            logger.info(
                "Saved new best model checkpoint to %s",
                config.distilbert_model_dir,
            )

    tokenizer.save_pretrained(config.distilbert_tokenizer_dir)
    logger.info("Tokenizer saved to %s", config.distilbert_tokenizer_dir)
    save_training_history(history, config.reports_dir / "distilbert_history.json")

    model = DistilBertForSequenceClassification.from_pretrained(
        config.distilbert_model_dir
    )
    model.to(device)

    logger.info("=" * 60)
    logger.info("DISTILBERT TRAINING - DONE")
    logger.info("=" * 60)
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DistilBERT training."""
    parser = argparse.ArgumentParser(
        description="Train DistilBERT for Twitter sentiment classification.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.DistilBERT.EPOCHS,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.DistilBERT.BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=cfg.DistilBERT.LEARNING_RATE,
        help="Learning rate for AdamW.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config=TrainingConfig(
            random_seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    )
