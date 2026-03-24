"""
routes/dashboard.py — Endpoints for the dashboard overview page.

Mounted at ``/api/dashboard`` in main.py.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.schemas.dashboard import (
    DashboardStats,
    RecentTweetsResponse,
    SentimentTrendResponse,
)
from app.services.data_service import data_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dashboard"])


@router.get(
    "/stats",
    response_model=DashboardStats,
    summary="Dashboard statistics",
    description=(
        "Returns aggregate statistics including total tweets, sentiment "
        "counts and percentages, average tweet length, and best model info."
    ),
)
def get_dashboard_stats():
    """Return aggregate dashboard statistics."""
    try:
        return data_service.get_dashboard_stats()
    except RuntimeError as exc:
        logger.error("Error in /stats: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error in /stats: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/recent-tweets",
    response_model=RecentTweetsResponse,
    summary="Recent tweets",
    description=(
        "Returns a paginated list of tweets with their predicted sentiment "
        "and confidence score."
    ),
)
def get_recent_tweets(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(
        20, ge=1, le=100,
        description="Number of tweets per page (max 100)",
    ),
):
    """Return a paginated list of recent tweets."""
    try:
        return data_service.get_recent_tweets(page=page, limit=limit)
    except RuntimeError as exc:
        logger.error("Error in /recent-tweets: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error in /recent-tweets: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/sentiment-trend",
    response_model=SentimentTrendResponse,
    summary="Sentiment trend over time",
    description=(
        "Groups tweets into sequential batches and returns per-batch "
        "sentiment counts for rendering a trend line chart."
    ),
)
def get_sentiment_trend(
    batch_size: int = Query(
        1000, ge=100, le=5000,
        description="Number of tweets per batch",
    ),
):
    """Return sentiment counts over sequential batches."""
    try:
        return data_service.get_sentiment_trend(batch_size=batch_size)
    except RuntimeError as exc:
        logger.error("Error in /sentiment-trend: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error in /sentiment-trend: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
