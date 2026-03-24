"""
Train a Bidirectional LSTM sentiment classifier with frozen GloVe embeddings.

Usage
-----
    py -3 -m scripts.train_bilstm
    py -3 -m scripts.train_bilstm --glove-path data/raw/glove.6B.200d.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    SpatialDropout1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg  # noqa: E402
from scripts.train_lstm import (  # noqa: E402
    build_callbacks,
    build_embedding_matrix,
    build_tokenizer,
    load_glove_embeddings,
    resolve_glove_path,
    save_tokenizer,
    vectorize_texts,
)
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


def build_bilstm_model(
    vocab_size: int,
    max_length: int,
    embedding_dim: int,
    embedding_matrix,
    num_classes: int,
    learning_rate: float,
) -> Sequential:
    """Build and compile the BiLSTM model."""
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
            Bidirectional(LSTM(128, return_sequences=True)),
            GlobalMaxPooling1D(),
            BatchNormalization(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Compiled BiLSTM model.")
    return model


def train(
    config: Optional[TrainingConfig] = None,
    glove_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
) -> Sequential:
    """Run end-to-end BiLSTM training and save all artefacts."""
    config = config or TrainingConfig()
    config.ensure_dirs()
    glove_path = resolve_glove_path(glove_path)
    tokenizer_path = tokenizer_path or (config.model_dir / "bilstm_tokenizer.pkl")
    history_path = history_path or (config.reports_dir / "bilstm_history.json")

    logger.info("=" * 60)
    logger.info("BILSTM TRAINING - START")
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

    model = build_bilstm_model(
        vocab_size=embedding_matrix.shape[0],
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
        callbacks=build_callbacks(config.bilstm_model_path),
        verbose=1,
    )

    model.save(config.bilstm_model_path)
    logger.info("Model saved to %s", config.bilstm_model_path)
    save_tokenizer(tokenizer, tokenizer_path)
    save_training_history(history, history_path)

    logger.info("=" * 60)
    logger.info("BILSTM TRAINING - DONE")
    logger.info("=" * 60)
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for BiLSTM training."""
    parser = argparse.ArgumentParser(
        description="Train a BiLSTM model with frozen GloVe embeddings.",
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
        default=cfg.BiLSTM.EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.BiLSTM.BATCH_SIZE,
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
            learning_rate=cfg.BiLSTM.LEARNING_RATE,
        ),
        glove_path=args.glove_path,
    )
