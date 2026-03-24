"""
Centralised configuration for the Twitter Sentiment Analysis project.

Usage
-----
    from configs.config import cfg

    print(cfg.DATA_DIR)
    print(cfg.LSTM.EPOCHS)
"""

from pathlib import Path


# ──────────────────────────────────────────────
# Project root (two levels up from this file)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
class Paths:
    DATA_DIR        = PROJECT_ROOT / "data"
    RAW_DATA_DIR    = DATA_DIR / "raw"
    PROCESSED_DIR   = DATA_DIR / "processed"
    MODEL_DIR       = PROJECT_ROOT / "models"
    NOTEBOOK_DIR    = PROJECT_ROOT / "notebooks"
    LOG_DIR         = PROJECT_ROOT / "logs"

    TRAIN_CSV       = DATA_DIR / "train_data.csv"
    VALID_CSV       = DATA_DIR / "valid_data.csv"
    TEST_CSV        = DATA_DIR / "test_data.csv"


# ──────────────────────────────────────────────
# Global settings
# ──────────────────────────────────────────────
RANDOM_SEED: int = 42
NUM_CLASSES: int = 3
MAX_VOCAB_SIZE: int = 20_000
MAX_SEQUENCE_LENGTH: int = 128
EMBEDDING_DIM: int = 200


# ──────────────────────────────────────────────
# Sentiment label mappings
# ──────────────────────────────────────────────
LABEL_TO_INDEX: dict[str, int] = {
    "Negative": 0,
    "Neutral":  1,
    "Positive": 2,
}
INDEX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INDEX.items()}

# Original Twitter dataset encoding  (0 → Negative, 2 → Neutral, 4 → Positive)
RAW_LABEL_MAP: dict[int, str] = {
    0: "Negative",
    2: "Neutral",
    4: "Positive",
}


# ──────────────────────────────────────────────
# Model hyper-parameters
# ──────────────────────────────────────────────
class LogisticRegressionCfg:
    """Logistic Regression + TF-IDF."""
    MAX_FEATURES: int     = 50_000
    NGRAM_RANGE: tuple    = (1, 2)
    C: float              = 1.0
    MAX_ITER: int         = 1_000
    SOLVER: str           = "lbfgs"


class LSTMCfg:
    """Vanilla LSTM."""
    EMBEDDING_DIM: int    = 200
    HIDDEN_UNITS: int     = 128
    DROPOUT: float        = 0.3
    RECURRENT_DROPOUT: float = 0.2
    EPOCHS: int           = 10
    BATCH_SIZE: int       = 64
    LEARNING_RATE: float  = 1e-3


class BiLSTMCfg:
    """Bidirectional LSTM."""
    EMBEDDING_DIM: int    = 200
    HIDDEN_UNITS: int     = 128
    DROPOUT: float        = 0.3
    RECURRENT_DROPOUT: float = 0.2
    EPOCHS: int           = 10
    BATCH_SIZE: int       = 64
    LEARNING_RATE: float  = 1e-3


class CNNCfg:
    """1-D Convolutional network for text."""
    EMBEDDING_DIM: int    = 200
    NUM_FILTERS: int      = 128
    KERNEL_SIZES: list    = [3, 4, 5]
    DROPOUT: float        = 0.5
    EPOCHS: int           = 10
    BATCH_SIZE: int       = 64
    LEARNING_RATE: float  = 1e-3


class DistilBERTCfg:
    """DistilBERT fine-tuning."""
    MODEL_NAME: str       = "distilbert-base-uncased"
    MAX_LENGTH: int       = 128
    EPOCHS: int           = 3
    BATCH_SIZE: int       = 16
    LEARNING_RATE: float  = 2e-5
    WARMUP_STEPS: int     = 500
    WEIGHT_DECAY: float   = 0.01
    GRADIENT_ACCUMULATION_STEPS: int = 2


# ──────────────────────────────────────────────
# Convenience namespace
# ──────────────────────────────────────────────
class cfg:
    """Single entry-point: ``from configs.config import cfg``."""
    PATHS            = Paths
    SEED             = RANDOM_SEED
    NUM_CLASSES      = NUM_CLASSES
    MAX_VOCAB        = MAX_VOCAB_SIZE
    MAX_SEQ_LEN      = MAX_SEQUENCE_LENGTH
    EMB_DIM          = EMBEDDING_DIM
    LABELS           = LABEL_TO_INDEX
    LABELS_INV       = INDEX_TO_LABEL
    RAW_LABELS       = RAW_LABEL_MAP

    LOGREG           = LogisticRegressionCfg
    LSTM             = LSTMCfg
    BiLSTM           = BiLSTMCfg
    CNN              = CNNCfg
    DistilBERT       = DistilBERTCfg
