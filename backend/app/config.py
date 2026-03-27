"""
config.py — Application settings loaded from environment variables via Pydantic.

Uses ``functools.lru_cache`` so the settings object is created only once.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Central configuration for the Twitter Sentiment Analysis API."""

    # ── App meta ──────────────────────────────────────────────
    APP_NAME: str = "Twitter Sentiment Analysis API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ALLOWED_ORIGINS: str = "https://*.vercel.app,http://localhost:*"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Split comma-separated origins into a list."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # ── Paths ─────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent  # backend/
    DATA_DIR: Optional[Path] = Field(default=None)
    MODELS_DIR: Optional[Path] = Field(default=None)
    REPORTS_DIR: Optional[Path] = Field(default=None)

    # ── Dataset paths ─────────────────────────────────────────
    TRAIN_DATA_PATH: Optional[Path] = Field(default=None)
    VALID_DATA_PATH: Optional[Path] = Field(default=None)

    # ── Model paths ───────────────────────────────────────────
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

    # ── Deployment / Resource management ──────────────────────
    DEPLOYMENT_TARGET: str = "auto"  # auto | render | hf
    SPACE_ID: Optional[str] = None   # Set automatically on Hugging Face Spaces
    LIGHTWEIGHT_MODE: Optional[bool] = None
    LAZY_LOADING: Optional[bool] = None

    # ── Rate limiting ─────────────────────────────────────────
    RATE_LIMIT_MAX_REQUESTS: int = 30
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def model_post_init(self, __context) -> None:
        """Resolve default paths relative to BASE_DIR after init."""
        is_hf_space = bool(self.SPACE_ID or os.getenv("SPACE_ID"))
        target = (self.DEPLOYMENT_TARGET or "auto").strip().lower()
        if target == "auto":
            target = "hf" if is_hf_space else "render"

        # Deployment-aware defaults:
        # - Render free: lightweight + lazy-loading (fits 512MB)
        # - HF Spaces free CPU: full preload (enough RAM for all 5 models)
        if self.LIGHTWEIGHT_MODE is None:
            self.LIGHTWEIGHT_MODE = target != "hf"
        if self.LAZY_LOADING is None:
            self.LAZY_LOADING = target == "render"

        project_root = self.BASE_DIR.parent  # twitter_analysis/

        def resolve_shared_dir(name: str) -> Path:
            candidates = [
                self.BASE_DIR / name,
                project_root / name,
            ]

            for candidate in candidates:
                if candidate.is_dir():
                    has_real_content = any(
                        child.name != ".gitkeep"
                        for child in candidate.rglob("*")
                    )
                    if has_real_content:
                        return candidate

            for candidate in candidates:
                if candidate.exists():
                    return candidate

            return self.BASE_DIR / name

        if self.DATA_DIR is None:
            self.DATA_DIR = resolve_shared_dir("data")
        if self.MODELS_DIR is None:
            self.MODELS_DIR = resolve_shared_dir("models")
        if self.REPORTS_DIR is None:
            self.REPORTS_DIR = resolve_shared_dir("reports")

        if self.TRAIN_DATA_PATH is None:
            self.TRAIN_DATA_PATH = self.DATA_DIR / "train_data.csv"
        if self.VALID_DATA_PATH is None:
            self.VALID_DATA_PATH = self.DATA_DIR / "valid_data.csv"

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


@lru_cache()
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()
