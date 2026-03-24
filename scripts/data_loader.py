"""
data_loader.py — Data loading, splitting, and class-imbalance handling.

Provides helpers to:
* Load raw train / valid / test CSVs.
* Create a stratified 70-15-15 split when a test set is missing.
* Inspect and handle class imbalance (class weights + SMOTE).
* Orchestrate the full load → clean → split → save pipeline.

Usage
-----
    from scripts.data_loader import run_data_pipeline, load_processed_data

    # First time — builds processed CSVs
    run_data_pipeline()

    # Later — load the processed splits
    train, valid, test = load_processed_data()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ----- project imports -----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg  # noqa: E402

# ──────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ──────────────────────────────────────────────
# 1. Loading raw data
# ──────────────────────────────────────────────

def load_raw_data(
    train_path: Optional[Path] = None,
    valid_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """Load raw CSV splits into a dictionary.

    Parameters
    ----------
    train_path, valid_path, test_path : Path, optional
        Override default paths from ``cfg.PATHS``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"train"``, ``"valid"``, and optionally ``"test"``.

    Raises
    ------
    FileNotFoundError
        If the training CSV does not exist.
    """
    train_path = train_path or cfg.PATHS.TRAIN_CSV
    valid_path = valid_path or cfg.PATHS.VALID_CSV
    test_path = test_path or cfg.PATHS.TEST_CSV

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    data: dict[str, pd.DataFrame] = {}
    data["train"] = pd.read_csv(train_path)
    logger.info("Loaded train  — %d rows from %s", len(data["train"]), train_path)

    if valid_path.exists():
        data["valid"] = pd.read_csv(valid_path)
        logger.info("Loaded valid  — %d rows from %s", len(data["valid"]), valid_path)

    if test_path.exists():
        data["test"] = pd.read_csv(test_path)
        logger.info("Loaded test   — %d rows from %s", len(data["test"]), test_path)
    else:
        logger.warning("No test CSV found at %s — will create one via stratified split.", test_path)

    return data


# ──────────────────────────────────────────────
# 2. Stratified splitting
# ──────────────────────────────────────────────

def create_stratified_splits(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = cfg.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train / valid / test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset.
    label_col : str
        Column used for stratification.
    train_ratio, valid_ratio, test_ratio : float
        Must sum to 1.0.
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, valid, test)`` DataFrames.

    Raises
    ------
    ValueError
        If ratios don't sum to 1.0 (within tolerance).
    """
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    # First split: train vs (valid + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(valid_ratio + test_ratio),
        stratify=df[label_col],
        random_state=random_seed,
    )

    # Second split: valid vs test
    relative_test = test_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df[label_col],
        random_state=random_seed,
    )

    logger.info(
        "Stratified split %d rows → train=%d, valid=%d, test=%d",
        len(df),
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ──────────────────────────────────────────────
# 3. Class distribution & imbalance handling
# ──────────────────────────────────────────────

def check_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label",
    imbalance_threshold: float = 3.0,
) -> dict[str, int]:
    """Log class counts and warn if the dataset is imbalanced.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a label column.
    label_col : str
        Column name.
    imbalance_threshold : float
        Maximum ratio between the largest and smallest class before
        a warning is emitted (default ``3.0``).

    Returns
    -------
    dict[str, int]
        Mapping of label → count.
    """
    counts = df[label_col].value_counts().to_dict()
    logger.info("Class distribution: %s", counts)

    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = max_count / min_count if min_count > 0 else float("inf")

    if ratio > imbalance_threshold:
        logger.warning(
            "⚠ Imbalanced classes detected (ratio %.1f:1). "
            "Consider using class weights or SMOTE.",
            ratio,
        )
    else:
        logger.info("Class balance ratio: %.2f:1 — within acceptable range.", ratio)

    return counts


def get_class_weights(
    labels: np.ndarray | pd.Series,
) -> dict[int, float]:
    """Compute balanced class weights using scikit-learn.

    Parameters
    ----------
    labels : array-like
        Ground-truth labels.

    Returns
    -------
    dict[int, float]
        Mapping ``{class_label: weight}``.
    """
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(labels))
    weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    logger.info("Computed class weights: %s", weight_dict)
    return weight_dict


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_seed: int = cfg.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to balance classes.

    Best used on *already-vectorised* features (e.g. TF-IDF matrix),
    **not** on raw text.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    y : np.ndarray
        Label array.
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X_resampled, y_resampled)``

    Raises
    ------
    ImportError
        If ``imbalanced-learn`` is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE.  "
            "Install it:  pip install imbalanced-learn"
        ) from exc

    smote = SMOTE(random_state=random_seed)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(
        "SMOTE resampling: %d → %d samples.",
        len(y),
        len(y_res),
    )
    return X_res, y_res


# ──────────────────────────────────────────────
# 4. Load processed data
# ──────────────────────────────────────────────

def load_processed_data(
    processed_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved processed CSVs.

    Parameters
    ----------
    processed_dir : Path, optional
        Directory containing ``processed_train.csv``, ``processed_valid.csv``,
        ``processed_test.csv``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, valid, test)`` DataFrames.

    Raises
    ------
    FileNotFoundError
        If any of the three files are missing.
    """
    d = processed_dir or cfg.PATHS.PROCESSED_DIR

    paths = {
        "train": d / "processed_train.csv",
        "valid": d / "processed_valid.csv",
        "test": d / "processed_test.csv",
    }

    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed files: {missing}. Run `run_data_pipeline()` first."
        )

    train = pd.read_csv(paths["train"])
    valid = pd.read_csv(paths["valid"])
    test = pd.read_csv(paths["test"])

    logger.info(
        "Loaded processed data — train=%d, valid=%d, test=%d.",
        len(train),
        len(valid),
        len(test),
    )
    return train, valid, test


# ──────────────────────────────────────────────
# 5. Full pipeline orchestrator
# ──────────────────────────────────────────────

def run_data_pipeline(
    force_split: bool = False,
    text_col: str = "text",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end pipeline: load → clean → split → save.

    Parameters
    ----------
    force_split : bool
        If ``True``, re-create train/valid/test from combined data
        even when separate CSVs exist.
    text_col : str
        Text column name.
    label_col : str
        Label column name.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_clean, valid_clean, test_clean)``
    """
    from scripts.data_cleaning import DataCleaner

    # --- Load ---
    raw = load_raw_data()
    has_test = "test" in raw

    if force_split or not has_test:
        logger.info("Creating stratified 70-15-15 split from combined data …")
        if "valid" in raw:
            combined = pd.concat([raw["train"], raw["valid"]], ignore_index=True)
        else:
            combined = raw["train"].copy()

        if has_test:
            combined = pd.concat([combined, raw["test"]], ignore_index=True)

        train_df, valid_df, test_df = create_stratified_splits(
            combined, label_col=label_col
        )
    else:
        train_df = raw["train"]
        valid_df = raw.get("valid", pd.DataFrame())
        test_df = raw["test"]

    # --- Clean ---
    cleaner = DataCleaner()

    logger.info("Cleaning train split …")
    train_clean = cleaner.clean_dataframe(train_df, text_col=text_col)

    if len(valid_df) > 0:
        logger.info("Cleaning valid split …")
        valid_clean = cleaner.clean_dataframe(valid_df, text_col=text_col)
    else:
        valid_clean = pd.DataFrame(columns=train_clean.columns)

    logger.info("Cleaning test split …")
    test_clean = cleaner.clean_dataframe(test_df, text_col=text_col)

    # --- Distribution check ---
    check_class_distribution(train_clean, label_col=label_col)

    # --- Save ---
    out_dir = cfg.PATHS.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train_clean.to_csv(out_dir / "processed_train.csv", index=False)
    valid_clean.to_csv(out_dir / "processed_valid.csv", index=False)
    test_clean.to_csv(out_dir / "processed_test.csv", index=False)

    logger.info("Saved processed CSVs → %s", out_dir)
    return train_clean, valid_clean, test_clean


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the data loading + cleaning pipeline.")
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Force re-creation of train/valid/test splits.",
    )
    args = parser.parse_args()

    run_data_pipeline(force_split=args.force_split)
