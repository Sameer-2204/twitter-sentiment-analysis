"""
Train an LSTM sentiment classifier with frozen GloVe embeddings.

This script:
1. Loads the train / validation CSV splits.
2. Fits a Keras tokenizer with a capped vocabulary.
3. Builds an embedding matrix from ``glove.6B.200d.txt``.
4. Trains an LSTM classifier with standard callbacks.
5. Saves the trained model, tokenizer, and history artefacts.

Usage
-----
    py -3 -m scripts.train_lstm
    py -3 -m scripts.train_lstm --glove-path data/raw/glove.6B.200d.txt
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    SpatialDropout1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg  # noqa: E402
from scripts.training_config import TrainingConfig  # noqa: E402
from scripts.training_utils import (  # noqa: E402
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

DEFAULT_GLOVE_FILENAME = "glove.6B.200d.txt"


def resolve_glove_path(glove_path: Optional[Path] = None) -> Path:
    """Resolve the GloVe embedding file path from explicit or common locations."""
    if glove_path is not None:
        candidate = glove_path.expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"GloVe file not found: {candidate}")

    search_roots = (
        Path.cwd(),
        cfg.PATHS.DATA_DIR,
        cfg.PATHS.RAW_DATA_DIR,
        cfg.PATHS.PROCESSED_DIR,
        cfg.PATHS.MODEL_DIR.parent / "embeddings",
    )
    candidates = [root / DEFAULT_GLOVE_FILENAME for root in search_roots]

    for candidate in candidates:
        if candidate.exists():
            logger.info("Using GloVe embeddings from %s", candidate)
            return candidate.resolve()

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Unable to locate glove.6B.200d.txt. Checked:\n"
        f"{searched}\n"
        "Pass --glove-path to specify the file explicitly."
    )


def build_tokenizer(
    texts: Iterable[str],
    max_vocab_size: int,
) -> Tokenizer:
    """Fit and return a Keras tokenizer."""
    tokenizer = Tokenizer(
        num_words=max_vocab_size,
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts(list(texts))
    logger.info(
        "Tokenizer fit complete with %d raw tokens (cap=%d).",
        len(tokenizer.word_index),
        max_vocab_size,
    )
    return tokenizer


def vectorize_texts(
    tokenizer: Tokenizer,
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw text into padded integer sequences."""
    X_train = pad_sequences(
        tokenizer.texts_to_sequences(list(train_texts)),
        maxlen=max_length,
        padding="post",
        truncating="post",
    )
    X_valid = pad_sequences(
        tokenizer.texts_to_sequences(list(valid_texts)),
        maxlen=max_length,
        padding="post",
        truncating="post",
    )
    logger.info(
        "Text vectorized to padded sequences with shape train=%s valid=%s.",
        X_train.shape,
        X_valid.shape,
    )
    return X_train, X_valid


def load_glove_embeddings(glove_path: Path) -> Dict[str, np.ndarray]:
    """Load word vectors from a GloVe text file."""
    embeddings_index: Dict[str, np.ndarray] = {}
    logger.info("Loading GloVe embeddings from %s", glove_path)

    with glove_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = np.asarray(parts[1:], dtype="float32")
            embeddings_index[word] = vector

    logger.info("Loaded %d embedding vectors.", len(embeddings_index))
    return embeddings_index


def build_embedding_matrix(
    tokenizer: Tokenizer,
    embeddings_index: Dict[str, np.ndarray],
    max_vocab_size: int,
    embedding_dim: int,
) -> np.ndarray:
    """Create an embedding matrix aligned with the tokenizer word index."""
    vocab_size = min(max_vocab_size, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    matched_tokens = 0

    for word, index in tokenizer.word_index.items():
        if index >= max_vocab_size:
            continue
        vector = embeddings_index.get(word)
        if vector is not None and vector.shape[0] == embedding_dim:
            embedding_matrix[index] = vector
            matched_tokens += 1

    logger.info(
        "Embedding matrix created with shape %s (%d matched tokens).",
        embedding_matrix.shape,
        matched_tokens,
    )
    return embedding_matrix


def build_lstm_model(
    vocab_size: int,
    max_length: int,
    embedding_dim: int,
    embedding_matrix: np.ndarray,
    num_classes: int,
    learning_rate: float,
) -> Sequential:
    """Build and compile the LSTM model."""
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=max_length,
                trainable=False,
            ),
            SpatialDropout1D(0.3),
            LSTM(128, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Compiled LSTM model.")
    return model


def build_callbacks(model_path: Path) -> list:
    """Create the callback set used during training."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def save_tokenizer(tokenizer: Tokenizer, save_path: Path) -> Path:
    """Persist the fitted tokenizer to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as handle:
        pickle.dump(tokenizer, handle)
    logger.info("Tokenizer saved to %s", save_path)
    return save_path


def train(
    config: Optional[TrainingConfig] = None,
    glove_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
) -> Sequential:
    """Run end-to-end LSTM training and save all artefacts."""
    config = config or TrainingConfig()
    config.ensure_dirs()
    glove_path = resolve_glove_path(glove_path)
    tokenizer_path = tokenizer_path or (config.model_dir / "tokenizer.pkl")
    history_path = history_path or (config.reports_dir / "lstm_history.json")

    logger.info("=" * 60)
    logger.info("LSTM TRAINING - START")
    logger.info("=" * 60)
    set_all_seeds(config.random_seed)

    X_train_text, y_train, X_valid_text, y_valid = load_and_prepare_data(
        model_type="keras",
        config=config,
    )
    if X_valid_text is None or y_valid is None:
        raise ValueError("Validation split is required for callback monitoring.")

    tokenizer = build_tokenizer(X_train_text.astype(str), config.max_vocab_size)
    X_train, X_valid = vectorize_texts(
        tokenizer=tokenizer,
        train_texts=X_train_text.astype(str),
        valid_texts=X_valid_text.astype(str),
        max_length=config.max_sequence_length,
    )

    embeddings_index = load_glove_embeddings(glove_path)
    embedding_matrix = build_embedding_matrix(
        tokenizer=tokenizer,
        embeddings_index=embeddings_index,
        max_vocab_size=config.max_vocab_size,
        embedding_dim=config.embedding_dim,
    )
    vocab_size = embedding_matrix.shape[0]

    model = build_lstm_model(
        vocab_size=vocab_size,
        max_length=config.max_sequence_length,
        embedding_dim=config.embedding_dim,
        embedding_matrix=embedding_matrix,
        num_classes=int(y_train.shape[1]),
        learning_rate=config.learning_rate,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=build_callbacks(config.lstm_model_path),
        verbose=1,
    )

    model.save(config.lstm_model_path)
    logger.info("Model saved to %s", config.lstm_model_path)
    save_tokenizer(tokenizer, tokenizer_path)
    save_training_history(history, history_path)

    logger.info("=" * 60)
    logger.info("LSTM TRAINING - DONE")
    logger.info("=" * 60)
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LSTM training."""
    parser = argparse.ArgumentParser(
        description="Train an LSTM model with frozen GloVe embeddings.",
    )
    parser.add_argument(
        "--glove-path",
        type=Path,
        default=None,
        help="Optional explicit path to glove.6B.200d.txt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.LSTM.EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.LSTM.BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.SEED,
        help="Random seed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config=TrainingConfig(
            random_seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=cfg.LSTM.LEARNING_RATE,
        ),
        glove_path=args.glove_path,
    )
