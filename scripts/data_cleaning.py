"""
data_cleaning.py — Production-grade text cleaning pipeline for Twitter sentiment analysis.

Provides a ``DataCleaner`` class that chains every cleaning step in a
configurable, reproducible order.  Designed to be imported by other scripts
or executed standalone to clean a CSV from the command line.

Usage
-----
    from scripts.data_cleaning import DataCleaner

    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_dataframe(df, text_col="text")
"""

from __future__ import annotations

import html
import logging
import re
import string
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import spacy
from tqdm import tqdm

# --------------- optional deps (graceful fallback) ---------------
try:
    import contractions as _contractions_lib

    def _expand(text: str) -> str:
        return _contractions_lib.fix(text)
except ImportError:
    _contractions_lib = None  # type: ignore[assignment]

    def _expand(text: str) -> str:  # type: ignore[misc]
        return text

try:
    import demoji as _demoji_lib


    def _emojis_to_text(text: str) -> str:
        return _demoji_lib.replace_with_desc(text, sep=" ")
except ImportError:
    _demoji_lib = None  # type: ignore[assignment]

    def _emojis_to_text(text: str) -> str:  # type: ignore[misc]
        return text

# -----------------------------------------------------------------

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg  # noqa: E402

# ──────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ──────────────────────────────────────────────
# Pre-compiled regex patterns (module-level for speed)
# ──────────────────────────────────────────────
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENTION = re.compile(r"@\w+")
_RE_HASHTAG_SYMBOL = re.compile(r"#")
_RE_RT = re.compile(r"\bRT\b", re.IGNORECASE)
_RE_DIGITS = re.compile(r"\d+")
_RE_ELONGATED = re.compile(r"(.)\1{2,}")  # 3+ repeated chars → 2
_RE_EXTRA_SPACES = re.compile(r"\s{2,}")

# Negation words to **keep** during stopword removal
_NEGATION_WORDS: frozenset[str] = frozenset(
    {"not", "no", "nor", "never", "neither", "hardly", "barely", "scarcely",
     "doesn", "doesn't", "don", "don't", "didn", "didn't",
     "isn", "isn't", "wasn", "wasn't", "weren", "weren't",
     "won", "won't", "wouldn", "wouldn't", "couldn", "couldn't",
     "shouldn", "shouldn't", "haven", "haven't", "hasn", "hasn't",
     "hadn", "hadn't", "mustn", "mustn't", "needn", "needn't",
     "aren", "aren't"}
)


class DataCleaner:
    """End-to-end text cleaner for tweet data.

    Parameters
    ----------
    spacy_model : str
        Name of the spaCy language model for lemmatization (default: ``en_core_web_sm``).
    remove_stopwords : bool
        Whether to strip stopwords (default ``True``).
    keep_negations : bool
        If ``True``, negation words survive stopword removal (default ``True``).
    handle_emojis : str
        ``"replace"`` converts emojis to text descriptions,
        ``"remove"`` strips them entirely (default ``"replace"``).
    min_token_count : int
        Tweets with fewer tokens after cleaning are dropped (default ``3``).
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        remove_stopwords: bool = True,
        keep_negations: bool = True,
        handle_emojis: str = "replace",
        min_token_count: int = 3,
    ) -> None:
        # ---- spaCy model ----
        try:
            self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])
            logger.info("Loaded spaCy model '%s'.", spacy_model)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Run:  python -m spacy download %s",
                spacy_model,
                spacy_model,
            )
            raise

        # ---- stopwords ----
        self.remove_stopwords = remove_stopwords
        self.keep_negations = keep_negations
        spacy_stops: set[str] = self.nlp.Defaults.stop_words.copy()
        if keep_negations:
            spacy_stops -= _NEGATION_WORDS
        self._stop_words: frozenset[str] = frozenset(spacy_stops)

        # ---- emoji strategy ----
        if handle_emojis not in {"replace", "remove"}:
            raise ValueError(f"handle_emojis must be 'replace' or 'remove', got '{handle_emojis}'")
        self.handle_emojis = handle_emojis

        # ---- minimum length ----
        self.min_token_count = min_token_count

        # ---- punctuation translation table ----
        self._punct_table = str.maketrans("", "", string.punctuation)

        logger.info(
            "DataCleaner initialised (stopwords=%s, keep_negations=%s, "
            "emoji_mode=%s, min_tokens=%d).",
            remove_stopwords,
            keep_negations,
            handle_emojis,
            min_token_count,
        )

    # ──────────────────────────────────────────
    # Individual cleaning steps
    # ──────────────────────────────────────────

    @staticmethod
    def _lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Strip URLs (http, https, www)."""
        return _RE_URL.sub("", text)

    @staticmethod
    def _remove_mentions(text: str) -> str:
        """Strip @mentions."""
        return _RE_MENTION.sub("", text)

    @staticmethod
    def _remove_hashtag_symbols(text: str) -> str:
        """Remove '#' symbol but keep hashtag text."""
        return _RE_HASHTAG_SYMBOL.sub("", text)

    @staticmethod
    def _remove_rt_tags(text: str) -> str:
        """Remove RT (retweet) markers."""
        return _RE_RT.sub("", text)

    @staticmethod
    def _remove_html_entities(text: str) -> str:
        """Unescape HTML entities  (``&amp;`` → ``&``, etc.)."""
        return html.unescape(text)

    @staticmethod
    def _handle_emojis_replace(text: str) -> str:
        """Convert emojis to their text descriptions."""
        return _emojis_to_text(text)

    @staticmethod
    def _handle_emojis_remove(text: str) -> str:
        """Strip all emoji characters."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & maps
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    @staticmethod
    def _expand_contractions(text: str) -> str:
        """Expand contractions (don't → do not)."""
        return _expand(text)

    @staticmethod
    def _normalize_elongated_words(text: str) -> str:
        """Collapse repeated characters (happpyyy → happy)."""
        return _RE_ELONGATED.sub(r"\1\1", text)

    @staticmethod
    def _remove_digits(text: str) -> str:
        """Remove standalone digits."""
        return _RE_DIGITS.sub("", text)

    def _remove_special_characters(self, text: str) -> str:
        """Strip punctuation and non-alphanumeric characters."""
        return text.translate(self._punct_table)

    def _filter_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stopwords while optionally preserving negation words."""
        return [t for t in tokens if t not in self._stop_words]

    def _lemmatize(self, text: str) -> str:
        """Lemmatize using spaCy."""
        doc = self.nlp(text)
        return " ".join(token.lemma_ for token in doc if token.lemma_.strip())

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse multiple spaces into one and strip edges."""
        return _RE_EXTRA_SPACES.sub(" ", text).strip()

    # ──────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """Apply the full cleaning pipeline to a single text string.

        Parameters
        ----------
        text : str
            Raw tweet text.

        Returns
        -------
        str
            Cleaned text.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self._remove_html_entities(text)
        text = self._remove_urls(text)
        text = self._remove_rt_tags(text)
        text = self._remove_mentions(text)
        text = self._remove_hashtag_symbols(text)

        # Emojis — before lowercasing to preserve emoji descriptions
        if self.handle_emojis == "replace":
            text = self._handle_emojis_replace(text)
        else:
            text = self._handle_emojis_remove(text)

        text = self._lowercase(text)
        text = self._expand_contractions(text)
        text = self._normalize_elongated_words(text)
        text = self._remove_digits(text)
        text = self._remove_special_characters(text)

        # Lemmatize
        text = self._lemmatize(text)

        # Stopword removal (on token list)
        if self.remove_stopwords:
            tokens = text.split()
            tokens = self._filter_stopwords(tokens)
            text = " ".join(tokens)

        text = self._collapse_whitespace(text)
        return text

    # ──────────────────────────────────────────
    # DataFrame-level operations
    # ──────────────────────────────────────────

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        drop_duplicates: bool = True,
        drop_short: bool = True,
    ) -> pd.DataFrame:
        """Clean an entire DataFrame of tweets.

        Steps:
        1. Drop rows with null ``text_col``.
        2. Optionally drop duplicate tweets.
        3. Apply :meth:`clean_text` to every row.
        4. Optionally drop tweets with < ``min_token_count`` tokens.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with at least ``text_col``.
        text_col : str
            Name of the text column.
        label_col : str
            Name of the label column (used for logging only).
        drop_duplicates : bool
            Whether to drop duplicate texts.
        drop_short : bool
            Whether to remove short tweets after cleaning.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe (copy — original is not mutated).
        """
        initial_rows = len(df)
        out = df.copy()

        # 1 — Drop nulls
        null_count = out[text_col].isna().sum()
        if null_count > 0:
            out = out.dropna(subset=[text_col])
            logger.info("Dropped %d rows with null '%s'.", null_count, text_col)

        # 2 — Drop duplicates
        if drop_duplicates:
            before = len(out)
            out = out.drop_duplicates(subset=[text_col])
            dupes = before - len(out)
            if dupes > 0:
                logger.info("Dropped %d duplicate tweets.", dupes)

        # 3 — Clean text
        tqdm.pandas(desc="Cleaning tweets")
        out[text_col] = out[text_col].progress_apply(self.clean_text)

        # 4 — Drop short tweets
        if drop_short:
            before = len(out)
            out = out[out[text_col].str.split().str.len() >= self.min_token_count]
            short = before - len(out)
            if short > 0:
                logger.info(
                    "Dropped %d tweets shorter than %d tokens.",
                    short,
                    self.min_token_count,
                )

        out = out.reset_index(drop=True)
        logger.info(
            "Cleaning complete: %d → %d rows (removed %d).",
            initial_rows,
            len(out),
            initial_rows - len(out),
        )
        return out


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
def main() -> None:
    """Clean train and validation CSVs and save to ``data/processed/``."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean tweet CSVs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=cfg.PATHS.TRAIN_CSV,
        help="Path to input CSV (default: train_data.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save cleaned CSV (default: data/processed/<input_name>).",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path: Path = args.output or (cfg.PATHS.PROCESSED_DIR / input_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s …", input_path)
    df = pd.read_csv(input_path)

    cleaner = DataCleaner()
    df_clean = cleaner.clean_dataframe(df)

    df_clean.to_csv(output_path, index=False)
    logger.info("Saved cleaned data → %s", output_path)


if __name__ == "__main__":
    main()
