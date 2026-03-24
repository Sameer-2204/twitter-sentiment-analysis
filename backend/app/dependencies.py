"""
dependencies.py — FastAPI dependency functions for service injection.

Each function returns the module-level singleton so that route handlers
can request services via ``Depends()`` without importing them directly.
"""

from __future__ import annotations

from app.services.data_service import DataService, data_service
from app.services.eda_service import EDAService, eda_service
from app.services.model_service import ModelService, model_service
from app.services.predictor import SentimentPredictor, predictor


def get_predictor() -> SentimentPredictor:
    """Return the global SentimentPredictor singleton."""
    return predictor


def get_data_service() -> DataService:
    """Return the global DataService singleton."""
    return data_service


def get_eda_service() -> EDAService:
    """Return the global EDAService singleton."""
    return eda_service


def get_model_service() -> ModelService:
    """Return the global ModelService singleton."""
    return model_service
