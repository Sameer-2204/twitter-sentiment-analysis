"""
services/eda_service.py — Computes and caches exploratory data analysis
statistics for the EDA page.

Depends on ``data_service.df`` being loaded before any methods are called.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.schemas.eda import (
    ClassDistribution,
    NgramItem,
    TweetLengthStats,
    WordcloudData,
    WordFrequencyItem,
    WordFrequencyResponse,
)

logger = logging.getLogger(__name__)

# ── NLTK stopwords (embedded to avoid runtime download) ───────
_STOP_WORDS: frozenset = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn",
    "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won",
    "wouldn",
    # Extra Twitter / finance noise words
    "amp", "rt", "https", "http", "co", "via", "like", "get", "one",
    "would", "also", "new", "us", "may", "could", "says", "said",
})

_RE_HASHTAG = re.compile(r"#(\w+)")
_RE_MENTION = re.compile(r"@(\w+)")
_RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")


class EDAService:
    """Computes and caches EDA statistics from the loaded DataFrame.

    Parameters
    ----------
    data_service : DataService
        Reference to the data service singleton whose ``.df`` is used.
    """

    def __init__(self, data_service: Any) -> None:
        self._data = data_service
        self._cache: Dict[str, Any] = {}

    # ────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """Shortcut to the loaded DataFrame (raises if not loaded)."""
        if not self._data.loaded or self._data.df is None:
            raise RuntimeError(
                "DataService has not loaded data yet. Call load_data() first."
            )
        return self._data.df

    def _filter_by_sentiment(self, sentiment: str) -> pd.DataFrame:
        """Return the DataFrame filtered by sentiment, or all rows."""
        if sentiment.lower() == "all":
            return self.df
        # Match case-insensitive against the 'sentiment' column
        mask = self.df["sentiment"].str.lower() == sentiment.lower()
        return self.df[mask]

    @staticmethod
    def _tokenize(texts: pd.Series) -> List[str]:
        """Tokenize a Series of texts into a flat lowercased word list,
        removing stopwords and short tokens."""
        words: List[str] = []
        for text in texts.dropna():
            cleaned = _RE_NON_ALPHA.sub(" ", str(text).lower())
            tokens = cleaned.split()
            words.extend(
                t for t in tokens
                if t not in _STOP_WORDS and len(t) > 2
            )
        return words

    @staticmethod
    def _build_ngrams(tokens: List[str], n: int) -> List[str]:
        """Build n-grams from a token list, skipping stopwords."""
        ngrams: List[str] = []
        for i in range(len(tokens) - n + 1):
            gram_tokens = tokens[i : i + n]
            # Skip if any token in the n-gram is a stopword
            if any(t in _STOP_WORDS for t in gram_tokens):
                continue
            ngrams.append(" ".join(gram_tokens))
        return ngrams

    def _cached(self, key: str):
        """Return cached value or None."""
        return self._cache.get(key)

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = value

    # ────────────────────────────────────────────────────────────
    # 1. Class distribution
    # ────────────────────────────────────────────────────────────

    def get_class_distribution(self) -> ClassDistribution:
        """Count tweets per sentiment class.

        Returns
        -------
        ClassDistribution
            Counts for positive, negative, and neutral.
        """
        cache_key = "class_distribution"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        start = time.time()
        counts = self.df["sentiment"].value_counts()

        result = ClassDistribution(
            positive=int(counts.get("Positive", 0)),
            negative=int(counts.get("Negative", 0)),
            neutral=int(counts.get("Neutral", 0)),
        )

        self._set_cache(cache_key, result)
        logger.info(
            "get_class_distribution computed in %.3f s", time.time() - start
        )
        return result

    # ────────────────────────────────────────────────────────────
    # 2. Word frequency
    # ────────────────────────────────────────────────────────────

    def get_word_frequency(
        self,
        sentiment: str = "all",
        top_n: int = 30,
    ) -> WordFrequencyResponse:
        """Return the most frequent words, optionally filtered by sentiment.

        Parameters
        ----------
        sentiment : str
            ``"all"``, ``"positive"``, ``"negative"``, or ``"neutral"``.
        top_n : int
            Number of top words to return.

        Returns
        -------
        WordFrequencyResponse
        """
        cache_key = f"word_freq_{sentiment.lower()}_{top_n}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        start = time.time()
        subset = self._filter_by_sentiment(sentiment)

        if subset.empty:
            result = WordFrequencyResponse(words=[], sentiment_filter=sentiment)
            self._set_cache(cache_key, result)
            return result

        tokens = self._tokenize(subset["text"])
        counter = Counter(tokens).most_common(top_n)

        result = WordFrequencyResponse(
            words=[WordFrequencyItem(word=w, count=c) for w, c in counter],
            sentiment_filter=sentiment,
        )

        self._set_cache(cache_key, result)
        logger.info(
            "get_word_frequency(sentiment=%s, top_n=%d) computed in %.3f s",
            sentiment, top_n, time.time() - start,
        )
        return result

    # ────────────────────────────────────────────────────────────
    # 3. Bigrams
    # ────────────────────────────────────────────────────────────

    def get_bigrams(
        self,
        sentiment: str = "all",
        top_n: int = 20,
    ) -> List[NgramItem]:
        """Return the most frequent bigrams.

        Parameters
        ----------
        sentiment : str
        top_n : int

        Returns
        -------
        list[NgramItem]
        """
        cache_key = f"bigrams_{sentiment.lower()}_{top_n}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        start = time.time()
        subset = self._filter_by_sentiment(sentiment)

        if subset.empty:
            self._set_cache(cache_key, [])
            return []

        # Tokenize per-tweet and build bigrams per tweet
        all_bigrams: List[str] = []
        for text in subset["text"].dropna():
            cleaned = _RE_NON_ALPHA.sub(" ", str(text).lower())
            tokens = [t for t in cleaned.split() if len(t) > 2]
            all_bigrams.extend(self._build_ngrams(tokens, 2))

        counter = Counter(all_bigrams).most_common(top_n)
        result = [NgramItem(ngram=g, count=c) for g, c in counter]

        self._set_cache(cache_key, result)
        logger.info(
            "get_bigrams(sentiment=%s, top_n=%d) computed in %.3f s",
            sentiment, top_n, time.time() - start,
        )
        return result

    # ────────────────────────────────────────────────────────────
    # 4. Trigrams
    # ────────────────────────────────────────────────────────────

    def get_trigrams(
        self,
        sentiment: str = "all",
        top_n: int = 20,
    ) -> List[NgramItem]:
        """Return the most frequent trigrams.

        Parameters
        ----------
        sentiment : str
        top_n : int

        Returns
        -------
        list[NgramItem]
        """
        cache_key = f"trigrams_{sentiment.lower()}_{top_n}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        start = time.time()
        subset = self._filter_by_sentiment(sentiment)

        if subset.empty:
            self._set_cache(cache_key, [])
            return []

        all_trigrams: List[str] = []
        for text in subset["text"].dropna():
            cleaned = _RE_NON_ALPHA.sub(" ", str(text).lower())
            tokens = [t for t in cleaned.split() if len(t) > 2]
            all_trigrams.extend(self._build_ngrams(tokens, 3))

        counter = Counter(all_trigrams).most_common(top_n)
        result = [NgramItem(ngram=g, count=c) for g, c in counter]

        self._set_cache(cache_key, result)
        logger.info(
            "get_trigrams(sentiment=%s, top_n=%d) computed in %.3f s",
            sentiment, top_n, time.time() - start,
        )
        return result

    # ────────────────────────────────────────────────────────────
    # 5. Tweet length statistics
    # ────────────────────────────────────────────────────────────

    def get_tweet_length_stats(self) -> TweetLengthStats:
        """Compute character-length and word-count distributions.

        Returns
        -------
        TweetLengthStats
            Lists of lengths + per-sentiment averages.
        """
        cache_key = "tweet_length_stats"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        start = time.time()
        df = self.df

        char_lengths: List[int] = df["text_length"].tolist()
        word_counts: List[int] = df["word_count"].tolist()

        # Average punctuation per tweet
        punct_counts = df["text"].astype(str).str.count(r"[^\w\s]")
        avg_punct = float(punct_counts.mean()) if len(punct_counts) > 0 else 0.0

        result = TweetLengthStats(
            char_lengths=char_lengths,
            word_counts=word_counts,
            avg_length=round(float(np.mean(char_lengths)), 1) if char_lengths else 0.0,
            avg_words=round(float(np.mean(word_counts)), 1) if word_counts else 0.0,
            avg_punctuation=round(avg_punct, 2),
        )

        self._set_cache(cache_key, result)
        logger.info(
            "get_tweet_length_stats computed in %.3f s", time.time() - start
        )
        return result

    # ────────────────────────────────────────────────────────────
    # 6. Wordcloud data
    # ────────────────────────────────────────────────────────────

    def get_wordcloud_data(
        self,
        sentiment: str = "all",
    ) -> WordcloudData:
        """Return a word→frequency mapping for rendering a word cloud.

        Parameters
        ----------
        sentiment : str

        Returns
        -------
        WordcloudData
        """
        cache_key = f"wordcloud_{sentiment.lower()}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        start = time.time()
        subset = self._filter_by_sentiment(sentiment)

        if subset.empty:
            result = WordcloudData(words={}, sentiment_filter=sentiment)
            self._set_cache(cache_key, result)
            return result

        tokens = self._tokenize(subset["text"])
        counter = Counter(tokens).most_common(100)

        result = WordcloudData(
            words=dict(counter),
            sentiment_filter=sentiment,
        )

        self._set_cache(cache_key, result)
        logger.info(
            "get_wordcloud_data(sentiment=%s) computed in %.3f s",
            sentiment, time.time() - start,
        )
        return result

    # ────────────────────────────────────────────────────────────
    # 7. Hashtags
    # ────────────────────────────────────────────────────────────

    def get_hashtags(self, top_n: int = 20) -> List[NgramItem]:
        """Extract and count the most common hashtags from tweets.

        Parameters
        ----------
        top_n : int

        Returns
        -------
        list[NgramItem]
        """
        cache_key = f"hashtags_{top_n}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        start = time.time()
        all_tags: List[str] = []
        for text in self.df["text"].dropna():
            all_tags.extend(
                tag.lower() for tag in _RE_HASHTAG.findall(str(text))
            )

        counter = Counter(all_tags).most_common(top_n)
        result = [NgramItem(ngram=f"#{t}", count=c) for t, c in counter]

        self._set_cache(cache_key, result)
        logger.info(
            "get_hashtags(top_n=%d) computed in %.3f s",
            top_n, time.time() - start,
        )
        return result

    # ────────────────────────────────────────────────────────────
    # 8. Mentions
    # ────────────────────────────────────────────────────────────

    def get_mentions(self, top_n: int = 20) -> List[NgramItem]:
        """Extract and count the most common @mentions from tweets.

        Parameters
        ----------
        top_n : int

        Returns
        -------
        list[NgramItem]
        """
        cache_key = f"mentions_{top_n}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        start = time.time()
        all_mentions: List[str] = []
        for text in self.df["text"].dropna():
            all_mentions.extend(
                m.lower() for m in _RE_MENTION.findall(str(text))
            )

        counter = Counter(all_mentions).most_common(top_n)
        result = [NgramItem(ngram=f"@{m}", count=c) for m, c in counter]

        self._set_cache(cache_key, result)
        logger.info(
            "get_mentions(top_n=%d) computed in %.3f s",
            top_n, time.time() - start,
        )
        return result


    # ────────────────────────────────────────────────────────────
    # Precompute (called at startup)
    # ────────────────────────────────────────────────────────────

    def precompute(self) -> None:
        """Pre-cache the most common EDA queries to speed up first requests.

        Called once during the startup event after data has been loaded.
        """
        start = time.time()
        logger.info("Pre-computing EDA statistics …")

        try:
            self.get_class_distribution()
        except Exception as exc:
            logger.error("precompute — class_distribution failed: %s", exc)

        for sentiment in ("all", "positive", "negative", "neutral"):
            try:
                self.get_word_frequency(sentiment=sentiment, top_n=30)
            except Exception as exc:
                logger.error("precompute — word_freq(%s) failed: %s", sentiment, exc)

            try:
                self.get_wordcloud_data(sentiment=sentiment)
            except Exception as exc:
                logger.error("precompute — wordcloud(%s) failed: %s", sentiment, exc)

        try:
            self.get_tweet_length_stats()
        except Exception as exc:
            logger.error("precompute — tweet_length_stats failed: %s", exc)

        try:
            self.get_bigrams()
        except Exception as exc:
            logger.error("precompute — bigrams failed: %s", exc)

        try:
            self.get_hashtags()
        except Exception as exc:
            logger.error("precompute — hashtags failed: %s", exc)

        try:
            self.get_mentions()
        except Exception as exc:
            logger.error("precompute — mentions failed: %s", exc)

        elapsed = time.time() - start
        logger.info(
            "EDA pre-computation complete in %.2f s (%d items cached)",
            elapsed,
            len(self._cache),
        )


# ── Singleton ─────────────────────────────────────────────────
# Import data_service here to avoid circular imports at module level.
from app.services.data_service import data_service  # noqa: E402

eda_service = EDAService(data_service)
