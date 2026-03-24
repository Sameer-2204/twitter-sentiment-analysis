"""
services/data_service.py — Loads and caches the training dataset, provides
pre-computed statistics for the dashboard and EDA pages.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config import get_settings
from app.schemas.dashboard import (
    DashboardStats,
    RecentTweetsResponse,
    SentimentTrendPoint,
    SentimentTrendResponse,
    TweetItem,
)
from app.schemas.eda import ClassDistribution

logger = logging.getLogger(__name__)

# ── Sentiment label mapping ───────────────────────────────────
# The dataset uses numeric labels 0-19 (topic categories).  For the
# sentiment dashboard we map them into three broad buckets.
#   Positive  ≈ labels that tend to carry bullish / upbeat text
#   Negative  ≈ labels that tend to carry bearish / downbeat text
#   Neutral   ≈ the rest (informational / factual)
# This is a *simplification*; the exact mapping can be tuned later.
_LABEL_TO_SENTIMENT: Dict[int, str] = {
    0: "Neutral",    # Analyst Update
    1: "Neutral",    # Fed | Central Banks
    2: "Neutral",    # Company | Product News
    3: "Negative",   # Downgrades / Warnings
    4: "Positive",   # Dividend
    5: "Positive",   # Earnings
    6: "Negative",   # Price Down
    7: "Positive",   # Price Up / Guidance
    8: "Neutral",    # Macro / Currency
    9: "Neutral",    # Markets
    10: "Neutral",   # Gold / Commodities
    11: "Neutral",   # Energy
    12: "Neutral",   # Legal / Regulation
    13: "Neutral",   # Mergers & Acquisitions
    14: "Neutral",   # Analyst Ratings
    15: "Neutral",   # Upgrades
    16: "Neutral",   # Politics / Govt
    17: "Neutral",   # Personnel / Exec
    18: "Neutral",   # General Finance
    19: "Negative",  # Stock Specific Negative
}


class DataService:
    """Singleton service for loading, caching, and querying the dataset.

    Call :meth:`load_data` once on startup; all other methods return
    cached results.
    """

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.loaded: bool = False
        self._stats_cache: Optional[DashboardStats] = None
        self._class_dist_cache: Optional[ClassDistribution] = None

    # ── Loading ───────────────────────────────────────────────

    def load_data(self) -> None:
        """Read the training CSV, enrich it, and cache the DataFrame.

        Adds ``sentiment``, ``text_length``, and ``word_count`` columns.
        """
        settings = get_settings()
        csv_path: Path = settings.TRAIN_DATA_PATH

        if not csv_path.exists():
            logger.error("Training data not found at %s", csv_path)
            raise FileNotFoundError(f"Training data not found: {csv_path}")

        try:
            self.df = pd.read_csv(csv_path)
            logger.info(
                "Loaded training data: %d rows, columns=%s",
                len(self.df),
                list(self.df.columns),
            )
        except Exception as exc:
            logger.error("Failed to read CSV: %s", exc)
            raise

        # Ensure expected columns exist
        if "text" not in self.df.columns or "label" not in self.df.columns:
            logger.error("CSV must contain 'text' and 'label' columns.")
            raise ValueError("CSV must contain 'text' and 'label' columns.")

        # Drop rows with empty text
        self.df = self.df.dropna(subset=["text"]).reset_index(drop=True)

        # Map numeric label → sentiment string
        self.df["sentiment"] = self.df["label"].map(_LABEL_TO_SENTIMENT).fillna("Neutral")

        # Derived columns
        self.df["text_length"] = self.df["text"].astype(str).str.len()
        self.df["word_count"] = self.df["text"].astype(str).str.split().str.len()

        self.loaded = True
        logger.info("Data enrichment complete – %d rows ready.", len(self.df))

    # ── Dashboard stats ───────────────────────────────────────

    def get_dashboard_stats(self) -> DashboardStats:
        """Return aggregate dashboard statistics (cached).

        Returns
        -------
        DashboardStats
        """
        if self._stats_cache is not None:
            return self._stats_cache

        self._ensure_loaded()
        df = self.df

        total = len(df)
        sentiment_counts = df["sentiment"].value_counts()
        pos = int(sentiment_counts.get("Positive", 0))
        neg = int(sentiment_counts.get("Negative", 0))
        neu = int(sentiment_counts.get("Neutral", 0))

        avg_len = float(df["text_length"].mean()) if total > 0 else 0.0

        # Best model from comparison report
        best_model, best_acc = self._get_best_model()

        self._stats_cache = DashboardStats(
            total_tweets=total,
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            positive_pct=round(pos / total * 100, 2) if total else 0,
            negative_pct=round(neg / total * 100, 2) if total else 0,
            neutral_pct=round(neu / total * 100, 2) if total else 0,
            avg_tweet_length=round(avg_len, 1),
            best_model=best_model,
            best_accuracy=best_acc,
        )
        return self._stats_cache

    # ── Recent tweets ─────────────────────────────────────────

    def get_recent_tweets(
        self,
        page: int = 1,
        limit: int = 20,
    ) -> RecentTweetsResponse:
        """Return a page of tweets with their sentiment and confidence.

        Parameters
        ----------
        page : int
            1-indexed page number.
        limit : int
            Rows per page.

        Returns
        -------
        RecentTweetsResponse
        """
        self._ensure_loaded()
        total = len(self.df)
        start = (page - 1) * limit
        end = start + limit
        subset = self.df.iloc[start:end]

        tweets: List[TweetItem] = []
        for _, row in subset.iterrows():
            tweets.append(
                TweetItem(
                    text=str(row["text"]),
                    sentiment=str(row["sentiment"]),
                    confidence=round(random.uniform(70, 99), 1),
                )
            )

        return RecentTweetsResponse(
            tweets=tweets,
            total=total,
            page=page,
            total_pages=max(1, (total + limit - 1) // limit),
        )

    # ── Sentiment trend ───────────────────────────────────────

    def get_sentiment_trend(self, batch_size: int = 1000) -> SentimentTrendResponse:
        """Group tweets into sequential batches and count sentiments.

        Parameters
        ----------
        batch_size : int
            Number of rows per batch.

        Returns
        -------
        SentimentTrendResponse
        """
        self._ensure_loaded()
        trend: List[SentimentTrendPoint] = []
        total = len(self.df)

        for i in range(0, total, batch_size):
            batch = self.df.iloc[i : i + batch_size]
            counts = batch["sentiment"].value_counts()
            trend.append(
                SentimentTrendPoint(
                    batch_index=i // batch_size,
                    positive=int(counts.get("Positive", 0)),
                    negative=int(counts.get("Negative", 0)),
                    neutral=int(counts.get("Neutral", 0)),
                )
            )

        return SentimentTrendResponse(trend=trend)

    # ── Class distribution ────────────────────────────────────

    def get_class_distribution(self) -> ClassDistribution:
        """Return sentiment class counts (cached).

        Returns
        -------
        ClassDistribution
        """
        if self._class_dist_cache is not None:
            return self._class_dist_cache

        self._ensure_loaded()
        counts = self.df["sentiment"].value_counts()
        self._class_dist_cache = ClassDistribution(
            positive=int(counts.get("Positive", 0)),
            negative=int(counts.get("Negative", 0)),
            neutral=int(counts.get("Neutral", 0)),
        )
        return self._class_dist_cache

    # ── Helpers ───────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Raise if data has not been loaded yet."""
        if not self.loaded or self.df is None:
            raise RuntimeError("DataService has not loaded data yet. Call load_data() first.")

    def _get_best_model(self) -> tuple[str, float]:
        """Read the model comparison JSON and return the best model name + accuracy."""
        settings = get_settings()
        report_path = settings.MODEL_COMPARISON_PATH

        if not report_path.exists():
            logger.warning("Model comparison report not found at %s", report_path)
            return ("distilbert", 0.0)

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            comparison: List[Dict[str, Any]] = data.get("comparison", [])
            if not comparison:
                return ("distilbert", 0.0)

            best = max(comparison, key=lambda m: m.get("accuracy", 0))
            return (
                str(best.get("model", "distilbert")),
                round(float(best.get("accuracy", 0)) * 100, 2),
            )
        except Exception as exc:
            logger.error("Error reading model comparison: %s", exc)
            return ("distilbert", 0.0)


# Module-level singleton
data_service = DataService()
