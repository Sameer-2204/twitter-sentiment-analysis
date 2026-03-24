"""
training_config.py — Centralised training configuration dataclass.

Provides a single ``TrainingConfig`` dataclass that aggregates every
hyper-parameter and filesystem path required by the training scripts.
Default values are sourced from ``configs.config`` to maintain a single
source of truth.

Usage
-----
    from scripts.training_config import TrainingConfig

    config = TrainingConfig()          # all defaults
    config = TrainingConfig(epochs=30) # override one field
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ── project imports ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Centralised, immutable-by-convention training configuration.

    All default values are derived from the project-level
    ``configs.config.cfg`` so that there is exactly one source of truth.

    Attributes
    ----------
    random_seed : int
        Global random seed for reproducibility.
    max_sequence_length : int
        Maximum token-sequence length for deep-learning models.
    batch_size : int
        Default mini-batch size.
    epochs : int
        Maximum number of training epochs.
    learning_rate : float
        Default optimiser learning rate.
    embedding_dim : int
        Dimensionality of word embeddings.
    max_vocab_size : int
        Maximum vocabulary size for tokenisers / TF-IDF.
    early_stopping_patience : int
        Number of epochs with no improvement before stopping.
    sentiment_labels : Dict[str, int]
        Mapping of label name → integer index.
    sentiment_labels_inv : Dict[int, str]
        Reverse mapping of integer index → label name.
    model_dir : Path
        Directory for saving trained model artefacts.
    reports_dir : Path
        Directory for saving classification reports.
    log_dir : Path
        Directory for training log files.
    lr_model_path : Path
        Save path for the Logistic Regression pipeline pickle.
    tfidf_path : Path
        Save path for the TF-IDF vectoriser pickle.
    lstm_model_path : Path
        Save path for the LSTM model.
    bilstm_model_path : Path
        Save path for the BiLSTM model.
    cnn_model_path : Path
        Save path for the CNN model.
    distilbert_model_dir : Path
        Save directory for the DistilBERT model.
    distilbert_tokenizer_dir : Path
        Save directory for the DistilBERT tokeniser.
    train_csv : Path
        Path to the training CSV.
    valid_csv : Path
        Path to the validation CSV.
    test_csv : Path
        Path to the test CSV.
    """

    # ── reproducibility ──────────────────────────────────────
    random_seed: int = cfg.SEED                     # 42

    # ── sequence / tokeniser ─────────────────────────────────
    max_sequence_length: int = cfg.MAX_SEQ_LEN      # 128
    max_vocab_size: int = 50_000

    # ── training loop ────────────────────────────────────────
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5

    # ── embeddings ───────────────────────────────────────────
    embedding_dim: int = cfg.EMB_DIM                # 200

    # ── label mappings ───────────────────────────────────────
    sentiment_labels: Dict[str, int] = field(
        default_factory=lambda: dict(cfg.LABELS),   # {"Negative": 0, ...}
    )
    sentiment_labels_inv: Dict[int, str] = field(
        default_factory=lambda: dict(cfg.LABELS_INV),
    )

    # ── directories ──────────────────────────────────────────
    model_dir: Path = cfg.PATHS.MODEL_DIR
    reports_dir: Path = cfg.PATHS.DATA_DIR.parent / "reports"
    log_dir: Path = cfg.PATHS.LOG_DIR

    # ── model save paths ─────────────────────────────────────
    lr_model_path: Path = cfg.PATHS.MODEL_DIR / "logistic_regression.pkl"
    tfidf_path: Path = cfg.PATHS.MODEL_DIR / "tfidf_vectorizer.pkl"
    lstm_model_path: Path = cfg.PATHS.MODEL_DIR / "lstm_model.h5"
    bilstm_model_path: Path = cfg.PATHS.MODEL_DIR / "bilstm_model.h5"
    cnn_model_path: Path = cfg.PATHS.MODEL_DIR / "cnn_model.h5"
    distilbert_model_dir: Path = cfg.PATHS.MODEL_DIR / "distilbert_model"
    distilbert_tokenizer_dir: Path = cfg.PATHS.MODEL_DIR / "distilbert_tokenizer"

    # ── data paths ───────────────────────────────────────────
    train_csv: Path = cfg.PATHS.TRAIN_CSV
    valid_csv: Path = cfg.PATHS.VALID_CSV
    test_csv: Path = cfg.PATHS.TEST_CSV

    # ── helpers ──────────────────────────────────────────────
    def ensure_dirs(self) -> None:
        """Create output directories if they don't already exist."""
        for d in (self.model_dir, self.reports_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        """Auto-create output directories on instantiation."""
        self.ensure_dirs()


# ─────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = TrainingConfig()
    print(config)
