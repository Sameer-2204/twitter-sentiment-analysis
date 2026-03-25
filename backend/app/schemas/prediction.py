"""
schemas/prediction.py — Pydantic models for prediction endpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for single-model sentiment prediction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The tweet or text to analyse.",
        json_schema_extra={"example": "I love this product! Best purchase ever."},
    )
    model_name: str = Field(
        default="logistic_regression",
        description="Model to use for prediction.",
        json_schema_extra={"example": "logistic_regression"},
    )


class PredictionResponse(BaseModel):
    """Response from a single-model prediction."""

    label: str = Field(..., description="Predicted sentiment label.", json_schema_extra={"example": "Positive"})
    confidence: float = Field(..., description="Confidence score (0-1).", json_schema_extra={"example": 0.94})
    model_used: str = Field(..., description="Name of the model used.", json_schema_extra={"example": "distilbert"})
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-class probability distribution.",
        json_schema_extra={"example": {"Positive": 0.94, "Negative": 0.03, "Neutral": 0.03}},
    )


class AllModelsRequest(BaseModel):
    """Request body for comparing all models on a single text."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The text to analyse with every model.",
    )


class AllModelsResponse(BaseModel):
    """Response from running all models on a single text."""

    results: List[PredictionResponse] = Field(
        ..., description="Prediction result from each model."
    )
    consensus: str = Field(
        ..., description="Majority sentiment across models.",
        json_schema_extra={"example": "Positive"},
    )
    agreement_count: int = Field(
        ..., description="How many models agree on the consensus.",
        json_schema_extra={"example": 4},
    )


class BatchPredictionResponse(BaseModel):
    """Response from batch CSV prediction."""

    results: List[Dict] = Field(
        ..., description="List of per-row prediction dicts."
    )
    summary: Dict = Field(
        ..., description="Aggregate counts and percentages."
    )
    total_processed: int = Field(
        ..., description="Number of rows processed.",
        json_schema_extra={"example": 150},
    )
