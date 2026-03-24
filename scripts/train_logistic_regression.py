"""
train_logistic_regression.py — Train a TF-IDF + Logistic Regression pipeline.

Pipeline
--------
1. Seed all RNGs for reproducibility.
2. Load train / validation CSVs via ``load_and_prepare_data("sklearn")``.
3. Build a sklearn ``Pipeline([TfidfVectorizer, LogisticRegression])``.
4. Run ``RandomizedSearchCV`` with stratified 5-fold CV over a defined
   hyper-parameter search space.
5. Log best parameters and cross-validation scores.
6. Evaluate on the validation set and log metrics.
7. Save:
   - Full pipeline pickle   → ``models/logistic_regression.pkl``
   - TF-IDF vectoriser      → ``models/tfidf_vectorizer.pkl``
   - Classification report  → ``reports/lr_classification_report.txt``
   - Training history (CV)  → ``reports/lr_cv_results.json``

Usage
-----
    python -m scripts.train_logistic_regression
    python -m scripts.train_logistic_regression --n-iter 30 --cv-folds 5

Requires: scikit-learn, tqdm, joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg                          # noqa: E402
from scripts.training_config import TrainingConfig      # noqa: E402
from scripts.training_utils import (                    # noqa: E402
    load_and_prepare_data,
    log_metrics,
    save_training_history,
    set_all_seeds,
)

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
# 1.  Pipeline Builder
# ═════════════════════════════════════════════════════════════

def build_pipeline(
    max_features: int = 50_000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Pipeline:
    """Create a TF-IDF + Logistic Regression sklearn Pipeline.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size for the TF-IDF vectoriser.
    ngram_range : tuple[int, int]
        Unigram + bigram range ``(1, 2)``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        ``[("tfidf", TfidfVectorizer), ("clf", LogisticRegression)]``
    """
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=cfg.SEED,
                    max_iter=1_000,
                    verbose=0,
                ),
            ),
        ]
    )
    logger.info("Built pipeline: %s", pipeline)
    return pipeline


# ═════════════════════════════════════════════════════════════
# 2.  Hyper-Parameter Search Space
# ═════════════════════════════════════════════════════════════

def get_search_space() -> Dict[str, Any]:
    """Return the ``RandomizedSearchCV`` parameter distributions.

    Returns
    -------
    dict[str, Any]
        Keys prefixed with ``clf__`` for the ``LogisticRegression`` step.
    """
    return [
        {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["saga"],
            "clf__max_iter": [500, 1000],
        },
        {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
            "clf__max_iter": [500, 1000],
        },
    ]


# ═════════════════════════════════════════════════════════════
# 3.  Training Routine
# ═════════════════════════════════════════════════════════════

def train(
    config: TrainingConfig | None = None,
    n_iter: int = 20,
    cv_folds: int = 5,
    scoring: str = "f1_macro",
) -> Pipeline:
    """End-to-end Logistic Regression training with hyper-parameter search.

    Parameters
    ----------
    config : TrainingConfig, optional
        Defaults to ``TrainingConfig()``.
    n_iter : int
        Number of random hyper-parameter combinations to try.
    cv_folds : int
        Number of stratified cross-validation folds.
    scoring : str
        Scoring metric for ``RandomizedSearchCV``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The best-estimator pipeline found by the search.
    """
    config = config or TrainingConfig()
    start_time = time.time()

    # ── 1. Reproducibility ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("LOGISTIC REGRESSION TRAINING — START")
    logger.info("=" * 60)
    set_all_seeds(config.random_seed)

    # ── 2. Load data ─────────────────────────────────────────
    logger.info("Loading data …")
    X_train, y_train, X_valid, y_valid = load_and_prepare_data(
        model_type="sklearn",
        config=config,
    )

    logger.info(
        "Data shapes — X_train: %s  y_train: %s",
        X_train.shape if hasattr(X_train, "shape") else len(X_train),
        y_train.shape if hasattr(y_train, "shape") else len(y_train),
    )

    # ── 3. Build pipeline & search space ─────────────────────
    pipeline = build_pipeline(
        max_features=config.max_vocab_size,
        ngram_range=(1, 2),
    )
    param_dist = get_search_space()
    logger.info("Search space: %s", param_dist)

    # ── 4. Randomised search with stratified K-Fold ──────────
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=config.random_seed,
    )

    logger.info(
        "Starting RandomizedSearchCV  (n_iter=%d, cv=%d, scoring=%s) …",
        n_iter,
        cv_folds,
        scoring,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=config.random_seed,
        n_jobs=-1,
        verbose=2,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    # ── 5. Log best parameters & CV scores ───────────────────
    logger.info("─" * 40)
    logger.info("Best parameters: %s", search.best_params_)
    logger.info("Best CV score (%s): %.4f", scoring, search.best_score_)
    logger.info("─" * 40)

    cv_results_df = pd.DataFrame(search.cv_results_)
    top_N = min(5, len(cv_results_df))
    logger.info("Top-%d CV results:", top_N)
    cols = ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    for _, row in cv_results_df.nsmallest(top_N, "rank_test_score")[cols].iterrows():
        logger.info(
            "  rank %d  |  %.4f ± %.4f  |  %s",
            row["rank_test_score"],
            row["mean_test_score"],
            row["std_test_score"],
            row["params"],
        )

    best_pipeline: Pipeline = search.best_estimator_

    # ── 6. Evaluate on validation set ────────────────────────
    if X_valid is not None and y_valid is not None:
        logger.info("Evaluating on validation set …")
        y_pred = best_pipeline.predict(X_valid)

        val_metrics = {
            "accuracy": accuracy_score(y_valid, y_pred),
            "f1_macro": f1_score(y_valid, y_pred, average="macro"),
            "precision_macro": precision_score(y_valid, y_pred, average="macro"),
            "recall_macro": recall_score(y_valid, y_pred, average="macro"),
        }
        log_metrics(val_metrics, prefix="[Validation]")

        # Build the report from the labels actually present in this run.
        observed_labels = sorted(
            set(np.asarray(y_valid).astype(int).tolist())
            | set(np.asarray(y_pred).astype(int).tolist())
        )
        config_labels = sorted(config.sentiment_labels_inv.keys())
        if observed_labels == config_labels:
            label_names = [
                config.sentiment_labels_inv[label]
                for label in observed_labels
            ]
        else:
            logger.warning(
                (
                    "Config label map has %d classes but validation data "
                    "contains %d classes. Using numeric labels in the "
                    "classification report."
                ),
                len(config_labels),
                len(observed_labels),
            )
            label_names = [str(label) for label in observed_labels]
        report_text = classification_report(
            y_valid,
            y_pred,
            labels=observed_labels,
            target_names=label_names,
        )
        logger.info("\nClassification Report:\n%s", report_text)
    else:
        report_text = "No validation set available — report skipped."
        logger.warning(report_text)

    # ── 7. Save artefacts ────────────────────────────────────
    config.ensure_dirs()

    # 7a. Full pipeline pickle
    model_save_path = config.lr_model_path
    with open(model_save_path, "wb") as fh:
        pickle.dump(best_pipeline, fh)
    logger.info("Model pipeline saved → %s", model_save_path)

    # 7b. TF-IDF vectoriser (standalone)
    tfidf_save_path = config.tfidf_path
    with open(tfidf_save_path, "wb") as fh:
        pickle.dump(best_pipeline.named_steps["tfidf"], fh)
    logger.info("TF-IDF vectoriser saved → %s", tfidf_save_path)

    # 7c. Classification report
    report_path = config.reports_dir / "lr_classification_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("Best Parameters\n")
        fh.write("=" * 40 + "\n")
        for k, v in search.best_params_.items():
            fh.write(f"  {k}: {v}\n")
        fh.write(f"\nBest CV Score ({scoring}): {search.best_score_:.4f}\n\n")
        fh.write("Classification Report (Validation)\n")
        fh.write("=" * 40 + "\n")
        fh.write(report_text + "\n")
    logger.info("Classification report saved → %s", report_path)

    # 7d. CV results JSON
    cv_json_path = config.reports_dir / "lr_cv_results.json"
    cv_serialisable = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "cv_results": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in search.cv_results_.items()
            if k != "params"  # dicts aren't JSON-serialisable by default
        },
    }
    with open(cv_json_path, "w", encoding="utf-8") as fh:
        json.dump(cv_serialisable, fh, indent=2, default=str)
    logger.info("CV results saved → %s", cv_json_path)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(
        "LOGISTIC REGRESSION TRAINING — DONE  (%.1f s / %.1f min)",
        elapsed,
        elapsed / 60,
    )
    logger.info("=" * 60)

    return best_pipeline


# ═════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + Logistic Regression pipeline with RandomizedSearchCV.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of random hyper-parameter combinations (default: 20).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="f1_macro",
        help="Scoring metric for RandomizedSearchCV (default: f1_macro).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainingConfig(random_seed=args.seed)
    train(
        config=config,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
    )
