"""
feature_engineering.py — Feature extraction for Twitter sentiment analysis.

Provides modular featurizers that can be used independently or composed:

* **TfidfFeaturizer** — TF-IDF vectorization (unigrams + bigrams).
* **EmbeddingLoader** — Load pre-trained GloVe / Word2Vec and build an
  embedding matrix for the model vocabulary.
* **DistilBERTTokenizerWrapper** — Hugging Face tokenizer with configurable
  padding / truncation.
* **extract_text_features** — Surface-level features: length, word count,
  punctuation count, uppercase ratio.
* **extract_vader_features** — VADER sentiment scores as numeric features.
* **build_feature_matrix** — Combine TF-IDF + text + VADER into one matrix.

Usage
-----
    from scripts.feature_engineering import (
        TfidfFeaturizer,
        EmbeddingLoader,
        DistilBERTTokenizerWrapper,
        extract_text_features,
        extract_vader_features,
        build_feature_matrix,
    )
"""

from __future__ import annotations

import logging
import pickle
import string
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# ----- project imports -----
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


# ═══════════════════════════════════════════════
# 1. TF-IDF Featurizer
# ═══════════════════════════════════════════════

class TfidfFeaturizer:
    """TF-IDF vectorizer wrapper with save / load support.

    Parameters
    ----------
    max_features : int
        Maximum number of features (default from config).
    ngram_range : tuple[int, int]
        N-gram range (default from config).
    sublinear_tf : bool
        Apply sublinear TF scaling (default ``True``).
    """

    def __init__(
        self,
        max_features: int = cfg.LOGREG.MAX_FEATURES,
        ngram_range: tuple[int, int] = cfg.LOGREG.NGRAM_RANGE,
        sublinear_tf: bool = True,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            dtype=np.float32,
        )
        self._is_fitted: bool = False
        logger.info(
            "TfidfFeaturizer created (max_features=%d, ngram_range=%s).",
            max_features,
            ngram_range,
        )

    def fit(self, texts: pd.Series | list[str]) -> "TfidfFeaturizer":
        """Fit the vectorizer on training texts.

        Parameters
        ----------
        texts : pd.Series | list[str]
            Training corpus.

        Returns
        -------
        TfidfFeaturizer
            Self, for method chaining.
        """
        self.vectorizer.fit(texts)
        self._is_fitted = True
        logger.info(
            "TF-IDF fitted — vocabulary size: %d.",
            len(self.vectorizer.vocabulary_),
        )
        return self

    def transform(self, texts: pd.Series | list[str]) -> np.ndarray:
        """Transform texts into a TF-IDF feature matrix.

        Parameters
        ----------
        texts : pd.Series | list[str]
            Texts to vectorize.

        Returns
        -------
        sparse matrix, shape ``(n_samples, n_features)``

        Raises
        ------
        RuntimeError
            If the vectorizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("TfidfFeaturizer has not been fitted. Call .fit() first.")
        matrix = self.vectorizer.transform(texts)
        logger.info("TF-IDF transformed — shape %s.", matrix.shape)
        return matrix

    def fit_transform(self, texts: pd.Series | list[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: Optional[Path] = None) -> None:
        """Persist the fitted vectorizer to disk.

        Parameters
        ----------
        path : Path, optional
            Defaults to ``models/tfidf_vectorizer.pkl``.
        """
        path = path or cfg.PATHS.MODEL_DIR / "tfidf_vectorizer.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info("Saved TF-IDF vectorizer → %s", path)

    def load(self, path: Optional[Path] = None) -> "TfidfFeaturizer":
        """Load a previously fitted vectorizer from disk.

        Parameters
        ----------
        path : Path, optional
            Defaults to ``models/tfidf_vectorizer.pkl``.

        Returns
        -------
        TfidfFeaturizer
            Self, with restored vectorizer.
        """
        path = path or cfg.PATHS.MODEL_DIR / "tfidf_vectorizer.pkl"
        if not path.exists():
            raise FileNotFoundError(f"TF-IDF vectorizer not found: {path}")
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)  # noqa: S301
        self._is_fitted = True
        logger.info("Loaded TF-IDF vectorizer ← %s", path)
        return self


# ═══════════════════════════════════════════════
# 2. Pre-trained Embedding Loader
# ═══════════════════════════════════════════════

class EmbeddingLoader:
    """Build an embedding matrix from pre-trained GloVe or Word2Vec files.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of embeddings (must match the file).
    """

    def __init__(self, embedding_dim: int = cfg.EMB_DIM) -> None:
        self.embedding_dim = embedding_dim
        self._word_vectors: dict[str, np.ndarray] = {}

    def load_glove(self, glove_path: Union[str, Path]) -> "EmbeddingLoader":
        """Load a GloVe text file.

        Parameters
        ----------
        glove_path : str | Path
            Path to the ``.txt`` GloVe file (e.g. ``glove.6B.100d.txt``).

        Returns
        -------
        EmbeddingLoader
            Self, for chaining.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        glove_path = Path(glove_path)
        if not glove_path.exists():
            raise FileNotFoundError(f"GloVe file not found: {glove_path}")

        logger.info("Loading GloVe embeddings from %s …", glove_path)
        with open(glove_path, encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading GloVe"):
                parts = line.rstrip().split(" ")
                word = parts[0]
                vector = np.asarray(parts[1:], dtype=np.float32)
                if vector.shape[0] == self.embedding_dim:
                    self._word_vectors[word] = vector

        logger.info("GloVe loaded — %d word vectors.", len(self._word_vectors))
        return self

    def load_word2vec(self, w2v_path: Union[str, Path]) -> "EmbeddingLoader":
        """Load a Word2Vec model using Gensim.

        Parameters
        ----------
        w2v_path : str | Path
            Path to a ``.bin`` or ``.txt`` Word2Vec model.

        Returns
        -------
        EmbeddingLoader
            Self, for chaining.

        Raises
        ------
        ImportError
            If ``gensim`` is not installed.
        """
        try:
            from gensim.models import KeyedVectors
        except ImportError as exc:
            raise ImportError(
                "gensim is required for Word2Vec loading.  "
                "Install: pip install gensim"
            ) from exc

        w2v_path = Path(w2v_path)
        is_binary = w2v_path.suffix == ".bin"
        logger.info("Loading Word2Vec from %s (binary=%s) …", w2v_path, is_binary)

        kv = KeyedVectors.load_word2vec_format(str(w2v_path), binary=is_binary)
        self._word_vectors = {word: kv[word] for word in kv.key_to_index}
        self.embedding_dim = kv.vector_size
        logger.info("Word2Vec loaded — %d vectors, dim=%d.", len(self._word_vectors), self.embedding_dim)
        return self

    def build_embedding_matrix(
        self,
        word_index: dict[str, int],
        max_words: Optional[int] = None,
    ) -> np.ndarray:
        """Build an embedding matrix aligned with a Keras-style ``word_index``.

        Parameters
        ----------
        word_index : dict[str, int]
            Token → integer index mapping (e.g. from ``keras.preprocessing.text.Tokenizer``).
        max_words : int, optional
            Cap on vocabulary size (default: all).

        Returns
        -------
        np.ndarray, shape ``(vocab_size, embedding_dim)``
        """
        if not self._word_vectors:
            raise RuntimeError("No embeddings loaded. Call load_glove() or load_word2vec() first.")

        vocab_size = min(len(word_index) + 1, max_words) if max_words else len(word_index) + 1
        matrix = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
        found = 0

        for word, idx in tqdm(word_index.items(), desc="Building embedding matrix"):
            if idx >= vocab_size:
                continue
            vec = self._word_vectors.get(word)
            if vec is not None:
                matrix[idx] = vec
                found += 1

        coverage = found / (vocab_size - 1) * 100 if vocab_size > 1 else 0.0
        logger.info(
            "Embedding matrix built — shape %s, coverage %.1f%% (%d/%d words).",
            matrix.shape,
            coverage,
            found,
            vocab_size - 1,
        )
        return matrix


# ═══════════════════════════════════════════════
# 3. DistilBERT Tokenizer Wrapper
# ═══════════════════════════════════════════════

class DistilBERTTokenizerWrapper:
    """Convenience wrapper around the Hugging Face DistilBERT tokenizer.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier (default from config).
    max_length : int
        Maximum token length for padding / truncation.
    """

    def __init__(
        self,
        model_name: str = cfg.DistilBERT.MODEL_NAME,
        max_length: int = cfg.DistilBERT.MAX_LENGTH,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required.  Install: pip install transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        logger.info(
            "DistilBERT tokenizer loaded (model=%s, max_length=%d).",
            model_name,
            max_length,
        )

    def tokenize(
        self,
        texts: list[str] | pd.Series,
        return_tensors: Optional[str] = "pt",
    ) -> dict[str, "torch.Tensor"]:
        """Tokenize a list of texts with padding and truncation.

        Parameters
        ----------
        texts : list[str] | pd.Series
            Input texts.
        return_tensors : str, optional
            ``"pt"`` for PyTorch, ``"tf"`` for TensorFlow, ``"np"`` for NumPy.

        Returns
        -------
        dict[str, Tensor]
            Keys: ``input_ids``, ``attention_mask``.
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        logger.info(
            "Tokenized %d texts → input_ids shape %s.",
            len(texts),
            tuple(encodings["input_ids"].shape),
        )
        return encodings


# ═══════════════════════════════════════════════
# 4. Surface-level Text Features
# ═══════════════════════════════════════════════

def extract_text_features(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """Compute lightweight surface features from raw/cleaned text.

    Features added:
    * ``char_count`` — total characters
    * ``word_count`` — total whitespace-separated tokens
    * ``punctuation_count`` — number of punctuation characters
    * ``uppercase_ratio`` — fraction of uppercase characters

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_col : str
        Column to extract features from.

    Returns
    -------
    pd.DataFrame
        Copy of the input with four extra columns.
    """
    out = df.copy()
    series = out[text_col].astype(str)

    out["char_count"] = series.str.len()
    out["word_count"] = series.str.split().str.len()
    out["punctuation_count"] = series.apply(
        lambda t: sum(1 for ch in t if ch in string.punctuation)
    )
    out["uppercase_ratio"] = series.apply(
        lambda t: sum(1 for ch in t if ch.isupper()) / max(len(t), 1)
    )

    logger.info("Extracted text features (char_count, word_count, punctuation_count, uppercase_ratio).")
    return out


# ═══════════════════════════════════════════════
# 5. VADER Sentiment Features
# ═══════════════════════════════════════════════

def extract_vader_features(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """Add VADER sentiment scores as numeric features.

    Features added:
    * ``vader_compound``
    * ``vader_pos``
    * ``vader_neg``
    * ``vader_neu``

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_col : str
        Column to analyse.

    Returns
    -------
    pd.DataFrame
        Copy of the input with four extra columns.

    Raises
    ------
    ImportError
        If ``vaderSentiment`` is not installed.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise ImportError(
            "vaderSentiment is required.  Install: pip install vaderSentiment"
        ) from exc

    sia = SentimentIntensityAnalyzer()
    out = df.copy()

    tqdm.pandas(desc="Computing VADER scores")
    scores = out[text_col].astype(str).progress_apply(lambda t: sia.polarity_scores(t))

    out["vader_compound"] = scores.apply(lambda s: s["compound"])
    out["vader_pos"] = scores.apply(lambda s: s["pos"])
    out["vader_neg"] = scores.apply(lambda s: s["neg"])
    out["vader_neu"] = scores.apply(lambda s: s["neu"])

    logger.info("Extracted VADER features (compound, pos, neg, neu).")
    return out


# ═══════════════════════════════════════════════
# 6. Combined Feature Matrix Builder
# ═══════════════════════════════════════════════

def build_feature_matrix(
    df: pd.DataFrame,
    text_col: str = "text",
    tfidf: Optional[TfidfFeaturizer] = None,
    include_text_features: bool = True,
    include_vader: bool = True,
) -> np.ndarray:
    """Build a single feature matrix combining TF-IDF + text + VADER.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    text_col : str
        Text column name.
    tfidf : TfidfFeaturizer, optional
        A **fitted** TF-IDF featurizer.  If ``None``, TF-IDF is skipped.
    include_text_features : bool
        Whether to append surface-level text features.
    include_vader : bool
        Whether to append VADER scores.

    Returns
    -------
    np.ndarray or sparse matrix
        Combined feature matrix.
    """
    parts: list = []

    # TF-IDF
    if tfidf is not None:
        tfidf_matrix = tfidf.transform(df[text_col])
        parts.append(tfidf_matrix)
        logger.info("Added TF-IDF features — shape %s.", tfidf_matrix.shape)

    # Text features
    if include_text_features:
        feat_df = extract_text_features(df, text_col=text_col)
        text_feat_cols = ["char_count", "word_count", "punctuation_count", "uppercase_ratio"]
        text_feats = feat_df[text_feat_cols].values.astype(np.float32)
        parts.append(text_feats)
        logger.info("Added text features — shape %s.", text_feats.shape)

    # VADER features
    if include_vader:
        vader_df = extract_vader_features(df, text_col=text_col)
        vader_cols = ["vader_compound", "vader_pos", "vader_neg", "vader_neu"]
        vader_feats = vader_df[vader_cols].values.astype(np.float32)
        parts.append(vader_feats)
        logger.info("Added VADER features — shape %s.", vader_feats.shape)

    if not parts:
        raise ValueError("No features selected — enable at least one of tfidf / text / vader.")

    # Combine
    if any(issparse(p) for p in parts):
        # Convert dense arrays to sparse for hstacking
        from scipy.sparse import csr_matrix
        sparse_parts = [p if issparse(p) else csr_matrix(p) for p in parts]
        combined = sparse_hstack(sparse_parts, format="csr")
    else:
        combined = np.hstack(parts)

    logger.info("Combined feature matrix — final shape %s.", combined.shape)
    return combined


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main() -> None:
    """Quick demo — fit TF-IDF on processed train data and print shape."""
    from scripts.data_loader import load_processed_data

    train, valid, test = load_processed_data()

    # TF-IDF
    tfidf = TfidfFeaturizer()
    X_train = tfidf.fit_transform(train["text"])
    X_valid = tfidf.transform(valid["text"])
    tfidf.save()

    logger.info("Train TF-IDF shape: %s", X_train.shape)
    logger.info("Valid TF-IDF shape: %s", X_valid.shape)

    # Combined matrix
    X_combined = build_feature_matrix(train, tfidf=tfidf)
    logger.info("Combined feature matrix shape: %s", X_combined.shape)


if __name__ == "__main__":
    main()
