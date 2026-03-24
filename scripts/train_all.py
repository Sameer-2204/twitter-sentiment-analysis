"""
Unified training runner for all Twitter sentiment analysis models.

This script can train one model or all available models, collect a small
set of comparable validation metrics, print a summary table, and persist
the results to ``reports/training_summary.json``.

Usage
-----
    python scripts/train_all.py --model lstm --epochs 30
    python scripts/train_all.py --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg  # noqa: E402
from scripts.training_config import TrainingConfig  # noqa: E402
from scripts.training_utils import load_and_prepare_data  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MODEL_CHOICES = ["logistic", "lstm", "bilstm", "cnn", "distilbert", "all"]


@dataclass
class TrainingResult:
    """Summary metrics for one training run."""

    model: str
    train_acc: Optional[float]
    val_acc: Optional[float]
    val_f1: Optional[float]
    time_taken_seconds: float
    status: str = "success"
    error: Optional[str] = None


def build_config(
    model_name: str,
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
) -> TrainingConfig:
    """Create a model-specific training config with CLI overrides applied."""
    if model_name == "lstm":
        return TrainingConfig(
            epochs=epochs if epochs is not None else cfg.LSTM.EPOCHS,
            batch_size=batch_size if batch_size is not None else cfg.LSTM.BATCH_SIZE,
            learning_rate=(
                learning_rate
                if learning_rate is not None
                else cfg.LSTM.LEARNING_RATE
            ),
        )
    if model_name == "bilstm":
        return TrainingConfig(
            epochs=epochs if epochs is not None else cfg.BiLSTM.EPOCHS,
            batch_size=(
                batch_size if batch_size is not None else cfg.BiLSTM.BATCH_SIZE
            ),
            learning_rate=(
                learning_rate
                if learning_rate is not None
                else cfg.BiLSTM.LEARNING_RATE
            ),
        )
    if model_name == "cnn":
        return TrainingConfig(
            epochs=epochs if epochs is not None else cfg.CNN.EPOCHS,
            batch_size=batch_size if batch_size is not None else cfg.CNN.BATCH_SIZE,
            learning_rate=(
                learning_rate
                if learning_rate is not None
                else cfg.CNN.LEARNING_RATE
            ),
        )
    if model_name == "distilbert":
        return TrainingConfig(
            epochs=epochs if epochs is not None else cfg.DistilBERT.EPOCHS,
            batch_size=(
                batch_size if batch_size is not None else cfg.DistilBERT.BATCH_SIZE
            ),
            learning_rate=(
                learning_rate
                if learning_rate is not None
                else cfg.DistilBERT.LEARNING_RATE
            ),
        )
    return TrainingConfig()


def load_pickle(path: Path):
    """Load a pickled artifact from disk."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def vectorize_with_tokenizer(
    tokenizer,
    train_texts,
    valid_texts,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert text collections into padded integer sequences."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_texts.astype(str).tolist()),
        maxlen=max_length,
        padding="post",
        truncating="post",
    )
    X_valid = pad_sequences(
        tokenizer.texts_to_sequences(valid_texts.astype(str).tolist()),
        maxlen=max_length,
        padding="post",
        truncating="post",
    )
    return X_train, X_valid


def evaluate_logistic(model, config: TrainingConfig) -> tuple[float, float, float]:
    """Evaluate the logistic regression pipeline."""
    X_train, y_train, X_valid, y_valid = load_and_prepare_data(
        model_type="sklearn",
        config=config,
    )
    if X_valid is None or y_valid is None:
        raise ValueError("Validation data is required for summary evaluation.")

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    return (
        accuracy_score(y_train, train_pred),
        accuracy_score(y_valid, valid_pred),
        f1_score(y_valid, valid_pred, average="macro"),
    )


def evaluate_keras(
    model,
    config: TrainingConfig,
    tokenizer_path: Path,
) -> tuple[float, float, float]:
    """Evaluate a Keras text classifier using its saved tokenizer."""
    X_train_text, y_train, X_valid_text, y_valid = load_and_prepare_data(
        model_type="keras",
        config=config,
    )
    if X_valid_text is None or y_valid is None:
        raise ValueError("Validation data is required for summary evaluation.")

    tokenizer = load_pickle(tokenizer_path)
    X_train, X_valid = vectorize_with_tokenizer(
        tokenizer=tokenizer,
        train_texts=X_train_text,
        valid_texts=X_valid_text,
        max_length=config.max_sequence_length,
    )

    train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    valid_pred = np.argmax(model.predict(X_valid, verbose=0), axis=1)
    y_train_true = np.argmax(y_train, axis=1)
    y_valid_true = np.argmax(y_valid, axis=1)

    return (
        accuracy_score(y_train_true, train_pred),
        accuracy_score(y_valid_true, valid_pred),
        f1_score(y_valid_true, valid_pred, average="macro"),
    )


def predict_distilbert_labels(model, data_loader, device):
    """Collect labels and predictions from a DistilBERT model."""
    import torch

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            predictions = torch.argmax(outputs.logits, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return y_true, y_pred


def evaluate_distilbert(
    model,
    config: TrainingConfig,
) -> tuple[float, float, float]:
    """Evaluate DistilBERT on the train and validation splits."""
    import torch
    from transformers import DistilBertTokenizer

    from scripts.train_distilbert import TweetDataset

    X_train, y_train, X_valid, y_valid = load_and_prepare_data(
        model_type="transformer",
        config=config,
    )
    if X_valid is None or y_valid is None:
        raise ValueError("Validation data is required for summary evaluation.")

    tokenizer = DistilBertTokenizer.from_pretrained(config.distilbert_tokenizer_dir)
    train_dataset = TweetDataset(
        texts=X_train,
        labels=y_train,
        tokenizer=tokenizer,
        max_length=config.max_sequence_length,
    )
    valid_dataset = TweetDataset(
        texts=X_valid,
        labels=y_valid,
        tokenizer=tokenizer,
        max_length=config.max_sequence_length,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=max(config.batch_size * 2, config.batch_size),
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_train_true, y_train_pred = predict_distilbert_labels(model, train_loader, device)
    y_valid_true, y_valid_pred = predict_distilbert_labels(model, valid_loader, device)

    return (
        accuracy_score(y_train_true, y_train_pred),
        accuracy_score(y_valid_true, y_valid_pred),
        f1_score(y_valid_true, y_valid_pred, average="macro"),
    )


def format_metric(value: Optional[float]) -> str:
    """Format a summary metric for console output."""
    return f"{value:.4f}" if value is not None else "N/A"


def format_seconds(seconds: float) -> str:
    """Format elapsed time for console output."""
    return f"{seconds:.1f}s"


def print_summary_table(results: list[TrainingResult]) -> None:
    """Print a compact summary table for all completed runs."""
    headers = ["Model", "Train Acc", "Val Acc", "Val F1", "Time Taken"]
    rows = [
        [
            result.model,
            format_metric(result.train_acc),
            format_metric(result.val_acc),
            format_metric(result.val_f1),
            format_seconds(result.time_taken_seconds),
        ]
        for result in results
    ]

    widths = [
        max(len(str(row[index])) for row in [headers] + rows)
        for index in range(len(headers))
    ]
    line = "-+-".join("-" * width for width in widths)

    header_row = " | ".join(
        header.ljust(width) for header, width in zip(headers, widths)
    )
    print("\nTraining Summary")
    print(header_row)
    print(line)
    for row in rows:
        print(" | ".join(str(value).ljust(width) for value, width in zip(row, widths)))


def save_summary(results: list[TrainingResult], save_path: Path) -> Path:
    """Persist the training summary as JSON."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_unix": time.time(),
        "results": [asdict(result) for result in results],
    }
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Training summary saved to %s", save_path)
    return save_path


def run_with_timing(
    model_name: str,
    train_fn: Callable[[], object],
    evaluate_fn: Callable[[object], tuple[float, float, float]],
) -> TrainingResult:
    """Train one model, evaluate it, and capture elapsed time."""
    logger.info("=" * 60)
    logger.info("RUNNING MODEL: %s", model_name)
    logger.info("=" * 60)
    start = time.perf_counter()

    model = train_fn()
    train_acc, val_acc, val_f1 = evaluate_fn(model)
    elapsed = time.perf_counter() - start

    logger.info(
        "%s finished in %.1f seconds | train_acc=%.4f | val_acc=%.4f | val_f1=%.4f",
        model_name,
        elapsed,
        train_acc,
        val_acc,
        val_f1,
    )
    return TrainingResult(
        model=model_name,
        train_acc=float(train_acc),
        val_acc=float(val_acc),
        val_f1=float(val_f1),
        time_taken_seconds=float(elapsed),
    )


def run_model(
    model_name: str,
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    glove_path: Optional[Path] = None,
) -> TrainingResult:
    """Dispatch training and evaluation for one selected model."""
    config = build_config(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    if model_name == "logistic":
        from scripts.train_logistic_regression import train as train_logistic_model

        return run_with_timing(
            model_name="logistic",
            train_fn=lambda: train_logistic_model(config=config),
            evaluate_fn=lambda model: evaluate_logistic(model, config),
        )
    if model_name == "lstm":
        from scripts.train_lstm import train as train_lstm_model

        return run_with_timing(
            model_name="lstm",
            train_fn=lambda: train_lstm_model(config=config, glove_path=glove_path),
            evaluate_fn=lambda model: evaluate_keras(
                model=model,
                config=config,
                tokenizer_path=config.model_dir / "tokenizer.pkl",
            ),
        )
    if model_name == "bilstm":
        from scripts.train_bilstm import train as train_bilstm_model

        return run_with_timing(
            model_name="bilstm",
            train_fn=lambda: train_bilstm_model(config=config, glove_path=glove_path),
            evaluate_fn=lambda model: evaluate_keras(
                model=model,
                config=config,
                tokenizer_path=config.model_dir / "bilstm_tokenizer.pkl",
            ),
        )
    if model_name == "cnn":
        from scripts.train_cnn import train as train_cnn_model

        return run_with_timing(
            model_name="cnn",
            train_fn=lambda: train_cnn_model(config=config, glove_path=glove_path),
            evaluate_fn=lambda model: evaluate_keras(
                model=model,
                config=config,
                tokenizer_path=config.model_dir / "cnn_tokenizer.pkl",
            ),
        )
    if model_name == "distilbert":
        from scripts.train_distilbert import train as train_distilbert_model

        return run_with_timing(
            model_name="distilbert",
            train_fn=lambda: train_distilbert_model(config=config),
            evaluate_fn=lambda model: evaluate_distilbert(
                model=model,
                config=config,
            ),
        )
    raise ValueError(f"Unsupported model choice: {model_name}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the unified training runner."""
    parser = argparse.ArgumentParser(
        description="Train one or all sentiment analysis models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=MODEL_CHOICES,
        help="Model to train (default: all).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for neural models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional batch-size override for neural models.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Optional learning-rate override for neural models.",
    )
    parser.add_argument(
        "--glove-path",
        type=Path,
        default=None,
        help="Optional explicit path to glove.6B.200d.txt for Keras models.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for unified model training."""
    args = parse_args()
    models_to_run = (
        ["logistic", "lstm", "bilstm", "cnn", "distilbert"]
        if args.model == "all"
        else [args.model]
    )

    results: list[TrainingResult] = []
    for model_name in models_to_run:
        start = time.perf_counter()
        try:
            result = run_model(
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                glove_path=args.glove_path,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.exception("Training failed for model=%s", model_name)
            result = TrainingResult(
                model=model_name,
                train_acc=None,
                val_acc=None,
                val_f1=None,
                time_taken_seconds=elapsed,
                status="failed",
                error=str(exc),
            )
            if args.model != "all":
                raise
        results.append(result)

    print_summary_table(results)
    save_summary(results, TrainingConfig().reports_dir / "training_summary.json")


if __name__ == "__main__":
    main()
