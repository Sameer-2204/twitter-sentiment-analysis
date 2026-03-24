"""
training_utils.py — Shared helpers for every training script.

Provides:
* ``set_all_seeds``          — deterministic seeding across libraries
* ``load_and_prepare_data``  — model-agnostic data loading
* ``save_training_history``  — serialise a Keras / dict history to JSON
* ``log_metrics``            — structured metric logging
* ``plot_training_curves``   — side-by-side loss & accuracy curves

Usage
-----
    from scripts.training_utils import (
        set_all_seeds,
        load_and_prepare_data,
        save_training_history,
        log_metrics,
        plot_training_curves,
    )
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg                          # noqa: E402
from scripts.training_config import TrainingConfig      # noqa: E402

# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ═════════════════════════════════════════════════════════════
# 1.  Reproducibility
# ═════════════════════════════════════════════════════════════

def set_all_seeds(seed: int = 42) -> None:
    """Seed every random-number generator for full reproducibility.

    Seeds:
    * ``random`` (stdlib)
    * ``numpy``
    * ``tensorflow`` (if importable)
    * ``torch``      (if importable)
    * ``PYTHONHASHSEED`` environment variable

    Parameters
    ----------
    seed : int, default 42
        The seed value to use across all libraries.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── TensorFlow ───────────────────────────────────────────
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Optionally make GPU ops deterministic (may hurt perf)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        logger.info("TensorFlow seed set to %d (deterministic ops ON).", seed)
    except ImportError:
        logger.debug("TensorFlow not installed — skipping TF seed.")

    # ── PyTorch ──────────────────────────────────────────────
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False      # type: ignore[attr-defined]
        logger.info("PyTorch seed set to %d.", seed)
    except ImportError:
        logger.debug("PyTorch not installed — skipping Torch seed.")

    logger.info(
        "All seeds set to %d  (random, numpy, PYTHONHASHSEED%s%s).",
        seed,
        ", tensorflow" if "tensorflow" in sys.modules else "",
        ", torch" if "torch" in sys.modules else "",
    )


# ═════════════════════════════════════════════════════════════
# 2.  Data Loading & Preparation
# ═════════════════════════════════════════════════════════════

ModelType = Literal["sklearn", "keras", "transformer"]


def encode_label_series(
    y_train: pd.Series,
    y_valid: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[int, int]]:
    """Map labels to contiguous integer IDs suitable for neural training.

    Parameters
    ----------
    y_train : pd.Series
        Training labels.
    y_valid : pd.Series, optional
        Validation labels.

    Returns
    -------
    tuple
        ``(encoded_train, encoded_valid, mapping)``
        where *mapping* is ``{original_label: encoded_index}``.
    """
    all_labels = pd.Series(y_train.astype(int).tolist())
    if y_valid is not None:
        all_labels = pd.concat([all_labels, pd.Series(y_valid.astype(int).tolist())])

    unique_labels = sorted(all_labels.unique().tolist())
    mapping = {int(label): index for index, label in enumerate(unique_labels)}

    encoded_train = y_train.astype(int).map(mapping).to_numpy()
    encoded_valid = (
        y_valid.astype(int).map(mapping).to_numpy()
        if y_valid is not None
        else None
    )

    if unique_labels != list(range(len(unique_labels))):
        logger.warning(
            "Non-contiguous labels detected. Applying label mapping: %s",
            mapping,
        )

    return encoded_train, encoded_valid, mapping


def load_and_prepare_data(
    model_type: ModelType = "sklearn",
    config: Optional[TrainingConfig] = None,
    text_col: str = "text",
    label_col: str = "label",
) -> Tuple[Any, Any, Any, Any]:
    """Load train / validation CSVs and return arrays ready for *model_type*.

    Parameters
    ----------
    model_type : {"sklearn", "keras", "transformer"}
        Determines the output format:

        * ``"sklearn"``      → ``(X_train, y_train, X_valid, y_valid)``
          where X is a Pandas Series of raw text.
        * ``"keras"``        → same, but ``y`` is one-hot encoded
          (``np.ndarray``).
        * ``"transformer"``  → same as ``"sklearn"``.
    config : TrainingConfig, optional
        Defaults to ``TrainingConfig()``.
    text_col : str
        Name of the text column in the CSV.
    label_col : str
        Name of the label column in the CSV.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_valid, y_valid)``

    Raises
    ------
    FileNotFoundError
        If either the training or validation CSV is missing.
    ValueError
        If *model_type* is unrecognised.
    """
    config = config or TrainingConfig()
    valid_types = ("sklearn", "keras", "transformer")
    if model_type not in valid_types:
        raise ValueError(
            f"Unknown model_type={model_type!r}. Choose from {valid_types}."
        )

    # ── Load CSVs ────────────────────────────────────────────
    logger.info("Loading training data from %s …", config.train_csv)
    if not config.train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {config.train_csv}")

    train_df = pd.read_csv(config.train_csv)
    logger.info("  → %d training samples loaded.", len(train_df))

    X_train: Union[pd.Series, np.ndarray] = train_df[text_col]
    y_train: Union[pd.Series, np.ndarray] = train_df[label_col]

    X_valid: Union[pd.Series, np.ndarray, None] = None
    y_valid: Union[pd.Series, np.ndarray, None] = None

    if config.valid_csv.exists():
        logger.info("Loading validation data from %s …", config.valid_csv)
        valid_df = pd.read_csv(config.valid_csv)
        logger.info("  → %d validation samples loaded.", len(valid_df))
        X_valid = valid_df[text_col]
        y_valid = valid_df[label_col]
    else:
        logger.warning(
            "Validation CSV not found at %s — returning None for valid split.",
            config.valid_csv,
        )

    # ── Handle NaN / missing text ────────────────────────────
    for name, series in [("X_train", X_train), ("X_valid", X_valid)]:
        if series is not None and series.isna().any():
            n_nan = int(series.isna().sum())
            logger.warning("Dropping %d NaN rows from %s.", n_nan, name)

    if X_train.isna().any():
        mask = ~X_train.isna()
        X_train, y_train = X_train[mask], y_train[mask]
    if X_valid is not None and X_valid.isna().any():
        mask = ~X_valid.isna()
        X_valid, y_valid = X_valid[mask], y_valid[mask]  # type: ignore[index]

    # ── Encode labels for Keras (one-hot) ────────────────────
    if model_type == "keras":
        from tensorflow.keras.utils import to_categorical  # type: ignore[import]

        encoded_y_train, encoded_y_valid, label_mapping = encode_label_series(
            y_train=y_train,
            y_valid=y_valid if y_valid is not None else None,
        )
        num_classes = len(label_mapping)
        if num_classes != len(config.sentiment_labels):
            logger.warning(
                (
                    "Config expects %d classes but data contains %d classes. "
                    "Using the data-derived class count for Keras training."
                ),
                len(config.sentiment_labels),
                num_classes,
            )

        y_train = to_categorical(encoded_y_train, num_classes=num_classes)
        if y_valid is not None:
            y_valid = to_categorical(encoded_y_valid, num_classes=num_classes)
        logger.info("Labels one-hot encoded for Keras (%d classes).", num_classes)

    logger.info(
        "Data ready for model_type=%r  |  train=%d  valid=%s",
        model_type,
        len(X_train),
        len(X_valid) if X_valid is not None else "N/A",
    )

    return X_train, y_train, X_valid, y_valid


# ═════════════════════════════════════════════════════════════
# 3.  Training History Serialisation
# ═════════════════════════════════════════════════════════════

def save_training_history(
    history: Union[Dict[str, List[float]], Any],
    save_path: Union[str, Path],
) -> Path:
    """Serialise a Keras ``History`` object (or plain dict) to JSON.

    Parameters
    ----------
    history : dict | keras.callbacks.History
        Either a plain ``{metric: [values]}`` dict or a Keras History
        whose ``.history`` attribute will be extracted.
    save_path : str | Path
        Destination JSON file path.

    Returns
    -------
    Path
        The resolved path the file was written to.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle Keras History object
    if hasattr(history, "history"):
        history = history.history  # type: ignore[union-attr]

    # Convert numpy types to native Python for JSON serialisation
    serialisable: Dict[str, List[float]] = {}
    for key, values in history.items():
        serialisable[key] = [
            float(v) if isinstance(v, (np.floating, np.integer)) else v
            for v in values
        ]

    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    logger.info("Training history saved → %s", save_path)
    return save_path


# ═════════════════════════════════════════════════════════════
# 4.  Metric Logging
# ═════════════════════════════════════════════════════════════

def log_metrics(
    metrics: Dict[str, float],
    prefix: str = "",
    level: int = logging.INFO,
) -> None:
    """Log a dictionary of evaluation metrics in a structured format.

    Parameters
    ----------
    metrics : dict[str, float]
        Keys are metric names (e.g. ``"accuracy"``, ``"f1_macro"``);
        values are numeric scores.
    prefix : str, optional
        Prefix for the log line (e.g. ``"[Validation]"``).
    level : int, optional
        Logging level (default ``INFO``).

    Example
    -------
    >>> log_metrics({"accuracy": 0.87, "f1_macro": 0.85}, prefix="[Valid]")
    """
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    line = "  |  ".join(parts)
    if prefix:
        line = f"{prefix}  {line}"
    logger.log(level, line)


# ═════════════════════════════════════════════════════════════
# 5.  Training Curve Visualisation
# ═════════════════════════════════════════════════════════════

def plot_training_curves(
    history: Union[Dict[str, List[float]], Any],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """Plot side-by-side loss and accuracy curves.

    Parameters
    ----------
    history : dict | keras.callbacks.History
        Training history with at least ``"loss"`` /
        ``"val_loss"`` and ``"accuracy"`` / ``"val_accuracy"`` keys.
    save_path : str | Path, optional
        If provided, save the figure to this path.
    title : str
        Figure super-title.
    figsize : tuple[int, int]
        Figure dimensions in inches.
    """
    # Handle Keras History object
    if hasattr(history, "history"):
        history = history.history  # type: ignore[union-attr]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # ── Loss ─────────────────────────────────────────────────
    ax_loss = axes[0]
    if "loss" in history:
        ax_loss.plot(history["loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax_loss.plot(history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # ── Accuracy ─────────────────────────────────────────────
    ax_acc = axes[1]
    # Keras uses "accuracy", some configs use "acc"
    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"

    if acc_key in history:
        ax_acc.plot(history[acc_key], label="Train Accuracy", linewidth=2)
    if val_acc_key in history:
        ax_acc.plot(
            history[val_acc_key],
            label="Val Accuracy",
            linewidth=2,
            linestyle="--",
        )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves saved → %s", save_path)

    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── set_all_seeds ──")
    set_all_seeds(42)

    print("\n── load_and_prepare_data ──")
    try:
        X_tr, y_tr, X_val, y_val = load_and_prepare_data("sklearn")
        print(f"  X_train: {len(X_tr)}   X_valid: {len(X_val) if X_val is not None else 'N/A'}")
    except FileNotFoundError as exc:
        print(f"  (skipped — {exc})")

    print("\n── save_training_history ──")
    dummy_history = {"loss": [0.9, 0.7, 0.5], "accuracy": [0.6, 0.75, 0.82]}
    out = save_training_history(dummy_history, Path("reports") / "dummy_history.json")
    print(f"  saved to {out}")

    print("\n── log_metrics ──")
    log_metrics({"accuracy": 0.87, "f1_macro": 0.85}, prefix="[Test]")

    print("\n── plot_training_curves ──")
    plot_training_curves(
        {
            "loss": [0.9, 0.7, 0.5],
            "val_loss": [1.0, 0.8, 0.6],
            "accuracy": [0.6, 0.75, 0.82],
            "val_accuracy": [0.55, 0.70, 0.78],
        },
        save_path=Path("reports") / "dummy_curves.png",
    )
    print("  Done.")
