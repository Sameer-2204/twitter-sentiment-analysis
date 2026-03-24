"""
routes/models.py — Endpoints for model comparison and evaluation data.

Mounted at ``/api/models`` in main.py.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Path

from app.schemas.models import (
    ConfusionMatrixResponse,
    ModelComparisonResponse,
    TrainingHistoryResponse,
)
from app.services.model_service import model_service
from app.services.predictor import predictor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Models"])

# ── Allowed model names ───────────────────────────────────────
_ALL_MODELS = {
    "logistic_regression", "lstm", "bilstm", "cnn", "distilbert",
}
_DL_MODELS = {"lstm", "bilstm", "cnn", "distilbert"}


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────


@router.get(
    "/comparison",
    response_model=ModelComparisonResponse,
    summary="Get performance comparison of all 5 models",
    description=(
        "Returns accuracy, precision, recall, F1-score, training time, "
        "and model size for every trained model, along with the best model name."
    ),
)
def get_model_comparison():
    """Return side-by-side metrics for all trained models."""
    try:
        return model_service.get_model_comparison()
    except Exception as exc:
        logger.error("Error in /comparison: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/confusion-matrix/{model_name}",
    response_model=ConfusionMatrixResponse,
    summary="Get confusion matrix for a specific model",
    description=(
        "Returns the confusion matrix (2D array) and class labels for "
        "the specified model. Falls back to a realistic placeholder if "
        "the evaluation report is not available."
    ),
)
def get_confusion_matrix(
    model_name: str = Path(
        ...,
        description="Model key: logistic_regression, lstm, bilstm, cnn, distilbert",
    ),
):
    """Return the confusion matrix for a specific model."""
    if model_name not in _ALL_MODELS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_name}' not found. "
                f"Available models: {sorted(_ALL_MODELS)}"
            ),
        )
    try:
        return model_service.get_confusion_matrix(model_name)
    except Exception as exc:
        logger.error("Error in /confusion-matrix/%s: %s", model_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/training-history/{model_name}",
    response_model=TrainingHistoryResponse,
    summary="Get training loss/accuracy curves for a DL model",
    description=(
        "Returns epoch-level training and validation loss/accuracy curves. "
        "Only available for deep-learning models (LSTM, BiLSTM, CNN, DistilBERT). "
        "Logistic Regression does not have training history."
    ),
)
def get_training_history(
    model_name: str = Path(
        ...,
        description="DL model key: lstm, bilstm, cnn, distilbert",
    ),
):
    """Return training curves for a deep-learning model."""
    if model_name == "logistic_regression":
        raise HTTPException(
            status_code=400,
            detail=(
                "Training history is not available for Logistic Regression. "
                "Only deep-learning models (lstm, bilstm, cnn, distilbert) "
                "have epoch-level training curves."
            ),
        )
    if model_name not in _DL_MODELS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_name}' not found. "
                f"DL models with training history: {sorted(_DL_MODELS)}"
            ),
        )
    try:
        return model_service.get_training_history(model_name)
    except Exception as exc:
        logger.error("Error in /training-history/%s: %s", model_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/available",
    response_model=List[str],
    summary="List currently loaded models",
    description=(
        "Returns the names of all ML models that were successfully loaded "
        "into memory at startup and are available for prediction."
    ),
)
def get_available_models():
    """Return a list of successfully loaded model names."""
    try:
        return predictor.get_available_models()
    except Exception as exc:
        logger.error("Error in /available: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
