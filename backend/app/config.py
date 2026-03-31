"""
config.py — Application settings for local development.

Uses pydantic-settings to load from environment / .env file.
All paths auto-resolve relative to the backend/ directory.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for the Twitter Sentiment Analysis API (local dev)."""

    # ── App meta ──────────────────────────────────────────────
    APP_NAME: str = "Twitter Sentiment Analysis API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

    # ── Server ────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── CORS — allow everything for local development ─────────
    ALLOWED_ORIGINS: str = "*"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse ALLOWED_ORIGINS into a list.

        - ``"*"``  → ``["*"]``
        - ``"https://a.com,https://b.com"`` → ``["https://a.com", "https://b.com"]``
        """
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # ── Paths (auto-detected from config.py location) ─────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent  # → backend/
    MODELS_DIR: Optional[Path] = Field(default=None)
    DATA_DIR: Optional[Path] = Field(default=None)
    REPORTS_DIR: Optional[Path] = Field(default=None)

    # ── Dataset file paths ────────────────────────────────────
    TRAIN_DATA_PATH: Optional[Path] = Field(default=None)
    VALID_DATA_PATH: Optional[Path] = Field(default=None)

    # ── Model file paths ──────────────────────────────────────
    LOGISTIC_REGRESSION_PATH: Optional[Path] = Field(default=None)
    TFIDF_VECTORIZER_PATH: Optional[Path] = Field(default=None)
    TOKENIZER_PATH: Optional[Path] = Field(default=None)
    LSTM_MODEL_PATH: Optional[Path] = Field(default=None)
    BILSTM_MODEL_PATH: Optional[Path] = Field(default=None)
    CNN_MODEL_PATH: Optional[Path] = Field(default=None)
    DISTILBERT_MODEL_PATH: Optional[Path] = Field(default=None)
    DISTILBERT_TOKENIZER_PATH: Optional[Path] = Field(default=None)
    MODEL_COMPARISON_PATH: Optional[Path] = Field(default=None)

    # ── Sentiment mapping ─────────────────────────────────────
    SENTIMENT_MAP: Dict[int, str] = {
        0: "Negative",
        2: "Neutral",
        4: "Positive",
    }
    SENTIMENT_COLORS: Dict[str, str] = {
        "Positive": "#06d6a0",
        "Negative": "#ef476f",
        "Neutral": "#ffd166",
    }

    # ── Model constants ───────────────────────────────────────
    MAX_SEQUENCE_LENGTH: int = 128
    MODEL_NAMES: List[str] = [
        "logistic_regression",
        "lstm",
        "bilstm",
        "cnn",
        "distilbert",
    ]

    # ── Logging ───────────────────────────────────────────────
    LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def model_post_init(self, __context) -> None:
        """Resolve default paths relative to BASE_DIR (backend/).

        All paths point to sibling directories of backend/ in the
        project root (i.e. ``../models``, ``../data``, ``../reports``).
        """
        project_root = self.BASE_DIR.parent  # twitter_analysis/

        # ── Directory paths ───────────────────────────────────
        if self.MODELS_DIR is None:
            self.MODELS_DIR = project_root / "models"
        if self.DATA_DIR is None:
            self.DATA_DIR = project_root / "data"
        if self.REPORTS_DIR is None:
            self.REPORTS_DIR = project_root / "reports"

        # ── Dataset file paths ────────────────────────────────
        if self.TRAIN_DATA_PATH is None:
            self.TRAIN_DATA_PATH = self.DATA_DIR / "train_data.csv"
        if self.VALID_DATA_PATH is None:
            self.VALID_DATA_PATH = self.DATA_DIR / "valid_data.csv"

        # ── Model file paths ─────────────────────────────────
        if self.LOGISTIC_REGRESSION_PATH is None:
            self.LOGISTIC_REGRESSION_PATH = self.MODELS_DIR / "logistic_regression.pkl"
        if self.TFIDF_VECTORIZER_PATH is None:
            self.TFIDF_VECTORIZER_PATH = self.MODELS_DIR / "tfidf_vectorizer.pkl"
        if self.TOKENIZER_PATH is None:
            self.TOKENIZER_PATH = self.MODELS_DIR / "tokenizer.pkl"
        if self.LSTM_MODEL_PATH is None:
            self.LSTM_MODEL_PATH = self.MODELS_DIR / "lstm_model.h5"
        if self.BILSTM_MODEL_PATH is None:
            self.BILSTM_MODEL_PATH = self.MODELS_DIR / "bilstm_model.h5"
        if self.CNN_MODEL_PATH is None:
            self.CNN_MODEL_PATH = self.MODELS_DIR / "cnn_model.h5"
        if self.DISTILBERT_MODEL_PATH is None:
            self.DISTILBERT_MODEL_PATH = self.MODELS_DIR / "distilbert_model"
        if self.DISTILBERT_TOKENIZER_PATH is None:
            self.DISTILBERT_TOKENIZER_PATH = self.MODELS_DIR / "distilbert_tokenizer"
        if self.MODEL_COMPARISON_PATH is None:
            self.MODEL_COMPARISON_PATH = self.REPORTS_DIR / "model_comparison_report.json"

        # ── Verify paths exist and warn ───────────────────────
        logger = logging.getLogger(__name__)
        for label, path in [
            ("MODELS_DIR", self.MODELS_DIR),
            ("DATA_DIR", self.DATA_DIR),
            ("REPORTS_DIR", self.REPORTS_DIR),
        ]:
            if not path.exists():
                logger.warning("⚠️  %s does not exist: %s", label, path)

    # ── Model path helpers ────────────────────────────────────

    def get_logistic_regression_path(self) -> Path:
        return self.LOGISTIC_REGRESSION_PATH

    def get_tfidf_path(self) -> Path:
        return self.TFIDF_VECTORIZER_PATH

    def get_tokenizer_path(self) -> Path:
        return self.TOKENIZER_PATH

    def get_lstm_path(self) -> Path:
        return self.LSTM_MODEL_PATH

    def get_bilstm_path(self) -> Path:
        return self.BILSTM_MODEL_PATH

    def get_cnn_path(self) -> Path:
        return self.CNN_MODEL_PATH

    def get_distilbert_model_path(self) -> Path:
        return self.DISTILBERT_MODEL_PATH

    def get_distilbert_tokenizer_path(self) -> Path:
        return self.DISTILBERT_TOKENIZER_PATH


def configure_logging(settings: Settings) -> None:
    """Configure root logger based on settings."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT,
        force=True,
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()
