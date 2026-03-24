"""
routes/eda.py — Endpoints for exploratory data analysis.

Mounted at ``/api/eda`` in main.py.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.schemas.eda import (
    ClassDistribution,
    NgramItem,
    TweetLengthStats,
    WordcloudData,
    WordFrequencyResponse,
)
from app.services.eda_service import eda_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["EDA"])

# ── Allowed sentiment values ──────────────────────────────────
_VALID_SENTIMENTS = {"all", "positive", "negative", "neutral"}


def _validate_sentiment(sentiment: str) -> str:
    """Validate and normalise the sentiment query parameter."""
    s = sentiment.strip().lower()
    if s not in _VALID_SENTIMENTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid sentiment filter '{sentiment}'. "
                f"Must be one of: {', '.join(sorted(_VALID_SENTIMENTS))}."
            ),
        )
    return s


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────


@router.get(
    "/class-distribution",
    response_model=ClassDistribution,
    summary="Sentiment class distribution",
    description="Returns the count of positive, negative, and neutral tweets in the dataset.",
)
def get_class_distribution():
    """Return sentiment class counts."""
    try:
        return eda_service.get_class_distribution()
    except Exception as exc:
        logger.error("Error in /class-distribution: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/word-frequency",
    response_model=WordFrequencyResponse,
    summary="Top word frequencies",
    description="Returns the most frequent words in the dataset, optionally filtered by sentiment.",
)
def get_word_frequency(
    sentiment: str = Query("all", description="Filter: all, positive, negative, neutral"),
    top_n: int = Query(30, ge=5, le=200, description="Number of top words to return"),
):
    """Return the most frequent words."""
    sentiment = _validate_sentiment(sentiment)
    try:
        return eda_service.get_word_frequency(sentiment=sentiment, top_n=top_n)
    except Exception as exc:
        logger.error("Error in /word-frequency: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/bigrams",
    response_model=List[NgramItem],
    summary="Top bigrams",
    description="Returns the most frequent bigrams (two-word pairs), optionally filtered by sentiment.",
)
def get_bigrams(
    sentiment: str = Query("all", description="Filter: all, positive, negative, neutral"),
    top_n: int = Query(20, ge=5, le=100, description="Number of top bigrams to return"),
):
    """Return the most frequent bigrams."""
    sentiment = _validate_sentiment(sentiment)
    try:
        return eda_service.get_bigrams(sentiment=sentiment, top_n=top_n)
    except Exception as exc:
        logger.error("Error in /bigrams: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/trigrams",
    response_model=List[NgramItem],
    summary="Top trigrams",
    description="Returns the most frequent trigrams (three-word phrases), optionally filtered by sentiment.",
)
def get_trigrams(
    sentiment: str = Query("all", description="Filter: all, positive, negative, neutral"),
    top_n: int = Query(20, ge=5, le=100, description="Number of top trigrams to return"),
):
    """Return the most frequent trigrams."""
    sentiment = _validate_sentiment(sentiment)
    try:
        return eda_service.get_trigrams(sentiment=sentiment, top_n=top_n)
    except Exception as exc:
        logger.error("Error in /trigrams: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/tweet-lengths",
    response_model=TweetLengthStats,
    summary="Tweet length statistics",
    description="Returns character-length and word-count distributions for all tweets.",
)
def get_tweet_lengths():
    """Return tweet length statistics."""
    try:
        return eda_service.get_tweet_length_stats()
    except Exception as exc:
        logger.error("Error in /tweet-lengths: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/wordcloud-data",
    response_model=WordcloudData,
    summary="Word cloud data",
    description="Returns a word→frequency mapping for rendering a word cloud, optionally filtered by sentiment.",
)
def get_wordcloud_data(
    sentiment: str = Query("all", description="Filter: all, positive, negative, neutral"),
):
    """Return word cloud data."""
    sentiment = _validate_sentiment(sentiment)
    try:
        return eda_service.get_wordcloud_data(sentiment=sentiment)
    except Exception as exc:
        logger.error("Error in /wordcloud-data: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/hashtags",
    response_model=List[NgramItem],
    summary="Top hashtags",
    description="Returns the most frequently used hashtags extracted from tweets.",
)
def get_hashtags(
    top_n: int = Query(20, ge=5, le=50, description="Number of top hashtags to return"),
):
    """Return the most common hashtags."""
    try:
        return eda_service.get_hashtags(top_n=top_n)
    except Exception as exc:
        logger.error("Error in /hashtags: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/mentions",
    response_model=List[NgramItem],
    summary="Top mentions",
    description="Returns the most frequently mentioned users (@mentions) in the dataset.",
)
def get_mentions(
    top_n: int = Query(20, ge=5, le=50, description="Number of top mentions to return"),
):
    """Return the most common @mentions."""
    try:
        return eda_service.get_mentions(top_n=top_n)
    except Exception as exc:
        logger.error("Error in /mentions: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
