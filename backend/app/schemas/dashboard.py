"""
schemas/dashboard.py — Pydantic models for dashboard endpoints.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DashboardStats(BaseModel):
    """Aggregate statistics for the dashboard overview."""

    total_tweets: int = Field(..., description="Total number of tweets in the dataset.", json_schema_extra={"example": 16000})
    positive_count: int = Field(..., description="Number of positive tweets.", json_schema_extra={"example": 5200})
    negative_count: int = Field(..., description="Number of negative tweets.", json_schema_extra={"example": 5400})
    neutral_count: int = Field(..., description="Number of neutral tweets.", json_schema_extra={"example": 5400})
    positive_pct: float = Field(..., description="Positive percentage.", json_schema_extra={"example": 32.5})
    negative_pct: float = Field(..., description="Negative percentage.", json_schema_extra={"example": 33.75})
    neutral_pct: float = Field(..., description="Neutral percentage.", json_schema_extra={"example": 33.75})
    avg_tweet_length: float = Field(..., description="Average character length of tweets.", json_schema_extra={"example": 112.5})
    best_model: str = Field(default="distilbert", description="Name of the best-performing model.")
    best_accuracy: float = Field(default=0.0, description="Accuracy of the best model.")


class TweetItem(BaseModel):
    """A single tweet with sentiment info."""

    text: str = Field(..., description="Tweet text.")
    sentiment: str = Field(..., description="Sentiment label.")
    confidence: float = Field(..., description="Model confidence (0-100).", json_schema_extra={"example": 87.5})


class RecentTweetsResponse(BaseModel):
    """Paginated list of recent tweets."""

    tweets: List[TweetItem] = Field(..., description="List of tweet items.")
    total: int = Field(..., description="Total tweets available.")
    page: int = Field(default=1, description="Current page number.")
    total_pages: int = Field(default=1, description="Total number of pages.")


class SentimentTrendPoint(BaseModel):
    """One data point in the sentiment trend chart."""

    batch_index: int = Field(..., description="Sequential batch index.")
    positive: int = Field(..., description="Positive count in this batch.")
    negative: int = Field(..., description="Negative count in this batch.")
    neutral: int = Field(..., description="Neutral count in this batch.")


class SentimentTrendResponse(BaseModel):
    """Sentiment counts over sequential batches for a line chart."""

    trend: List[SentimentTrendPoint] = Field(
        ..., description="List of sentiment counts per batch."
    )
