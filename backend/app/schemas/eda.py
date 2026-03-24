"""
schemas/eda.py — Pydantic models for exploratory data analysis endpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ClassDistribution(BaseModel):
    """Counts per sentiment class."""

    positive: int = Field(..., description="Positive tweet count.")
    negative: int = Field(..., description="Negative tweet count.")
    neutral: int = Field(..., description="Neutral tweet count.")


class WordFrequencyItem(BaseModel):
    """A word and its frequency."""

    word: str = Field(..., description="The word or token.")
    count: int = Field(..., description="Frequency count.")


class WordFrequencyResponse(BaseModel):
    """List of most frequent words, optionally filtered by sentiment."""

    words: List[WordFrequencyItem] = Field(..., description="Frequency-sorted word list.")
    sentiment_filter: str = Field(
        default="all",
        description="Sentiment filter applied ('all', 'positive', 'negative', 'neutral').",
    )


class NgramItem(BaseModel):
    """An n-gram and its frequency."""

    ngram: str = Field(..., description="The bigram or trigram string.")
    count: int = Field(..., description="Frequency count.")


class NgramResponse(BaseModel):
    """List of most frequent n-grams."""

    ngrams: List[NgramItem] = Field(..., description="Frequency-sorted n-gram list.")
    sentiment_filter: str = Field(default="all")
    n: int = Field(default=2, description="N-gram size (2=bigram, 3=trigram).")


class TweetLengthStats(BaseModel):
    """Statistics about tweet character and word lengths."""

    char_lengths: List[int] = Field(default_factory=list, description="Character length of each tweet.")
    word_counts: List[int] = Field(default_factory=list, description="Word count of each tweet.")
    avg_length: float = Field(default=0.0, description="Average character length.")
    avg_words: float = Field(default=0.0, description="Average word count.")
    avg_punctuation: float = Field(default=0.0, description="Average punctuation count.")


class WordcloudData(BaseModel):
    """Word-frequency map for rendering a word cloud."""

    words: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of word → count.",
    )
    sentiment_filter: str = Field(default="all")


class HashtagItem(BaseModel):
    """A hashtag and its count."""

    word: str = Field(..., description="Hashtag text (without #).")
    count: int = Field(..., description="Occurrence count.")


class MentionItem(BaseModel):
    """A mention and its count."""

    word: str = Field(..., description="Mentioned user (without @).")
    count: int = Field(..., description="Occurrence count.")
