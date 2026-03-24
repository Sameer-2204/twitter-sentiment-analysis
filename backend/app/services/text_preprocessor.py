"""
services/text_preprocessor.py — Lightweight text cleaning for API inference.

Unlike the full ``scripts/data_cleaning.py`` pipeline (which uses spaCy and
lemmatization), this preprocessor is intentionally *fast* so it can run
synchronously during HTTP requests.
"""

from __future__ import annotations

import html
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# ── Pre-compiled regex patterns ───────────────────────────────
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENTION = re.compile(r"@\w+")
_RE_HASHTAG_SYMBOL = re.compile(r"#")
_RE_RT = re.compile(r"\bRT\b", re.IGNORECASE)
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_SPECIAL = re.compile(r"[^a-zA-Z0-9\s]")
_RE_EXTRA_SPACES = re.compile(r"\s{2,}")

# ── Common English contractions ───────────────────────────────
_CONTRACTIONS: dict[str, str] = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "it's": "it is",
    "it'll": "it will",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "doesn't": "does not",
    "didn't": "did not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mustn't": "must not",
    "needn't": "need not",
    "let's": "let us",
    "who's": "who is",
    "what's": "what is",
    "here's": "here is",
    "where's": "where is",
    "when's": "when is",
    "how's": "how is",
}

# Build a single regex from keys (longest-first to avoid partial matches)
_CONTRACTIONS_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _expand_match(match: re.Match) -> str:
    """Return the expansion for a matched contraction."""
    word = match.group(0).lower()
    return _CONTRACTIONS.get(word, word)


class TextPreprocessor:
    """Fast text preprocessor for API inference requests.

    Applies URL removal, mention stripping, contraction expansion, etc.
    Does **not** perform lemmatization or stopword removal so that inference
    latency stays low.
    """

    def clean_text(self, text: str) -> str:
        """Clean a single text string for model inference.

        Parameters
        ----------
        text : str
            Raw input text (tweet or free-form).

        Returns
        -------
        str
            Cleaned text ready for model tokenisation / vectorisation.
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("Received empty or non-string input.")
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Remove URLs
        text = _RE_URL.sub("", text)

        # 3. Remove @mentions
        text = _RE_MENTION.sub("", text)

        # 4. Remove # symbol but keep hashtag word
        text = _RE_HASHTAG_SYMBOL.sub("", text)

        # 5. Remove RT tag
        text = _RE_RT.sub("", text)

        # 6. Unescape HTML entities
        text = html.unescape(text)

        # 7. Remove HTML tags
        text = _RE_HTML_TAG.sub("", text)

        # 8. Expand contractions
        text = _CONTRACTIONS_RE.sub(_expand_match, text)

        # 9. Remove special characters (keep letters, digits, spaces)
        text = _RE_SPECIAL.sub(" ", text)

        # 10. Collapse extra whitespace and strip
        text = _RE_EXTRA_SPACES.sub(" ", text).strip()

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a list of texts.

        Parameters
        ----------
        texts : list[str]
            Raw text strings.

        Returns
        -------
        list[str]
            Cleaned text strings.
        """
        logger.info("Cleaning batch of %d texts.", len(texts))
        return [self.clean_text(t) for t in texts]

    def validate_text(self, text: str) -> bool:
        """Check whether text is non-empty after cleaning.

        Parameters
        ----------
        text : str
            Raw text input.

        Returns
        -------
        bool
            ``True`` if the cleaned text contains at least one token.
        """
        cleaned = self.clean_text(text)
        return len(cleaned.strip()) > 0


# Module-level singleton
text_preprocessor = TextPreprocessor()
