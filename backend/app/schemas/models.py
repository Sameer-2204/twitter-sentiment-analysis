"""
schemas/models.py — Pydantic models for the model-performance endpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    """Evaluation metrics for a single model."""

    name: str = Field(..., description="Model name.", json_schema_extra={"example": "distilbert"})
    accuracy: float = Field(..., description="Accuracy percentage.", json_schema_extra={"example": 86.5})
    precision: float = Field(..., description="Precision percentage.", json_schema_extra={"example": 85.2})
    recall: float = Field(..., description="Recall percentage.", json_schema_extra={"example": 84.8})
    f1_score: float = Field(..., description="F1-score percentage.", json_schema_extra={"example": 85.0})
    training_time: str = Field(default="—", description="Training duration string.")
    model_size: str = Field(default="—", description="Model file size.")


class ModelComparisonResponse(BaseModel):
    """Side-by-side comparison of all models."""

    models: List[ModelMetrics] = Field(..., description="List of model metrics.")
    best_model: str = Field(..., description="Name of the best-performing model.")


class ConfusionMatrixResponse(BaseModel):
    """Confusion matrix for a specific model."""

    matrix: List[List[int]] = Field(..., description="2D confusion matrix.")
    labels: List[str] = Field(..., description="Class labels in matrix order.")
    model_name: str = Field(..., description="Model that produced this matrix.")


class TrainingHistoryResponse(BaseModel):
    """Epoch-level training curves for a deep-learning model."""

    epochs: List[int] = Field(..., description="Epoch numbers.")
    train_loss: List[float] = Field(default_factory=list, description="Training loss per epoch.")
    val_loss: List[float] = Field(default_factory=list, description="Validation loss per epoch.")
    train_acc: List[float] = Field(default_factory=list, description="Training accuracy per epoch.")
    val_acc: List[float] = Field(default_factory=list, description="Validation accuracy per epoch.")
    model_name: str = Field(..., description="Model name this history belongs to.")
