"""
services/predictor.py — Loads ML models and exposes a unified prediction
interface.

Supports two deployment modes controlled by environment variables:

* **LIGHTWEIGHT_MODE=true** (default): Only loads Logistic Regression at
  startup (~15 MB RAM).  Other models are skipped unless lazy-loaded.
* **LAZY_LOADING=true** (default): Heavy models (LSTM, BiLSTM, CNN,
  DistilBERT) are loaded on first prediction request.  The previously
  loaded heavy model is unloaded to free RAM — only one heavy model is
  kept in memory at a time alongside Logistic Regression.

Heavy imports (TensorFlow, PyTorch, Transformers) are done lazily inside
their respective loader methods.
"""

from __future__ import annotations

import gc
from importlib.util import find_spec
import logging
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import get_settings
from app.schemas.prediction import (
    AllModelsResponse,
    BatchPredictionResponse,
    PredictionResponse,
)
from app.services.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

# Names of models that are considered "heavy" (high RAM usage)
_HEAVY_MODELS = {"lstm", "bilstm", "cnn", "distilbert"}


class SentimentPredictor:
    """Loads trained sentiment models and provides prediction methods.

    Call :meth:`load_all_models` once during startup.  With lightweight
    mode enabled only Logistic Regression is loaded eagerly; other models
    are loaded on-demand via :meth:`_ensure_model_loaded`.
    """

    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.tfidf: Any = None
        self.keras_tokenizer: Any = None
        self.bert_tokenizer: Any = None
        self.preprocessor: TextPreprocessor = TextPreprocessor()
        self.loaded: bool = False
        self.load_times: Dict[str, float] = {}
        self.device: str = "cpu"
        self._current_heavy_model: Optional[str] = None  # track which heavy model is in RAM

        # Detect CUDA
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("PyTorch device: %s", self.device)
        except ImportError:
            logger.warning("PyTorch not installed — DistilBERT unavailable.")

    # ──────────────────────────────────────────────────────────
    #  Model loading
    # ──────────────────────────────────────────────────────────

    def load_all_models(self) -> None:
        """Load model artefacts from disk.

        In **lightweight mode** only the TF-IDF vectoriser and Logistic
        Regression are loaded eagerly.  In **full mode** every model is
        loaded (each in its own try/except).
        """
        settings = get_settings()
        total_start = time.time()

        # ── Always load TF-IDF + Logistic Regression ──────────
        self._load_tfidf(settings.TFIDF_VECTORIZER_PATH)
        self._load_pickle_model("logistic_regression", settings.LOGISTIC_REGRESSION_PATH)

        if settings.LIGHTWEIGHT_MODE:
            logger.info(
                "LIGHTWEIGHT MODE: Only Logistic Regression loaded eagerly. "
                "Heavy models will be lazy-loaded on first request."
            )
        else:
            # ── Full mode: load everything ────────────────────
            logger.info("FULL MODE: Loading all models …")
            self._load_keras_tokenizer(settings.TOKENIZER_PATH)
            self._load_keras_model("lstm", settings.LSTM_MODEL_PATH)
            self._load_keras_model("bilstm", settings.BILSTM_MODEL_PATH)
            self._load_keras_model("cnn", settings.CNN_MODEL_PATH)
            self._load_distilbert(
                model_path=settings.DISTILBERT_MODEL_PATH,
                tokenizer_path=settings.DISTILBERT_TOKENIZER_PATH,
            )

        self.loaded = True
        total_elapsed = time.time() - total_start

        loaded = list(self.models.keys())
        failed = [n for n in settings.MODEL_NAMES if n not in self.models]
        logger.info(
            "Model loading complete in %.2f s — loaded: %s | deferred: %s",
            total_elapsed,
            loaded or "none",
            failed or "none",
        )

    # ──────────────────────────────────────────────────────────
    #  Lazy loading helpers
    # ──────────────────────────────────────────────────────────

    def _ensure_model_loaded(self, model_name: str) -> None:
        """Ensure the requested model is in memory, loading it lazily.

        If the model is a heavy model and lazy loading is enabled:
        1. Unload the *previous* heavy model (if different) to free RAM.
        2. Load the requested model.
        Logistic Regression stays in memory permanently.
        """
        if model_name in self.models:
            return  # already loaded

        settings = get_settings()
        if not settings.LAZY_LOADING:
            raise ValueError(
                f"Model '{model_name}' is not loaded and lazy loading is disabled."
            )

        logger.info("Lazy-loading model '%s' on demand …", model_name)

        # Unload previous heavy model to free RAM
        if model_name in _HEAVY_MODELS and self._current_heavy_model:
            prev = self._current_heavy_model
            if prev != model_name and prev in self.models:
                logger.info("Unloading '%s' to free RAM for '%s'.", prev, model_name)
                del self.models[prev]
                self._current_heavy_model = None
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        # Load the requested model
        self._load_single_model(model_name)

        if model_name in _HEAVY_MODELS:
            self._current_heavy_model = model_name

    def _load_single_model(self, model_name: str) -> None:
        """Load a single model by name."""
        settings = get_settings()

        if model_name == "logistic_regression":
            self._load_tfidf(settings.TFIDF_VECTORIZER_PATH)
            self._load_pickle_model("logistic_regression", settings.LOGISTIC_REGRESSION_PATH)

        elif model_name in ("lstm", "bilstm", "cnn"):
            if find_spec("tensorflow") is None:
                raise ValueError(
                    f"Model '{model_name}' requires TensorFlow, which is not installed "
                    "in this Railway deployment."
                )
            # Ensure Keras tokenizer is loaded (shared)
            if self.keras_tokenizer is None:
                self._load_keras_tokenizer(settings.TOKENIZER_PATH)
            path_map = {
                "lstm": settings.LSTM_MODEL_PATH,
                "bilstm": settings.BILSTM_MODEL_PATH,
                "cnn": settings.CNN_MODEL_PATH,
            }
            self._load_keras_model(model_name, path_map[model_name])

        elif model_name == "distilbert":
            if find_spec("torch") is None or find_spec("transformers") is None:
                raise ValueError(
                    "Model 'distilbert' requires PyTorch and Transformers, which are "
                    "not installed in this Railway deployment."
                )
            self._load_distilbert(
                model_path=settings.DISTILBERT_MODEL_PATH,
                tokenizer_path=settings.DISTILBERT_TOKENIZER_PATH,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # ──────────────────────────────────────────────────────────
    #  Prediction
    # ──────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        model_name: str = "distilbert",
    ) -> PredictionResponse:
        """Run a single prediction with the specified model.

        Parameters
        ----------
        text : str
            Raw user input text (tweet or free-form).
        model_name : str
            ``logistic_regression``, ``lstm``, ``bilstm``, ``cnn``, ``distilbert``.

        Returns
        -------
        PredictionResponse

        Raises
        ------
        ValueError
            If ``model_name`` is unknown or cannot be loaded.
        RuntimeError
            If models have not been initialised yet.
        """
        if not self.loaded:
            raise RuntimeError("Models have not been loaded. Call load_all_models() first.")

        # Lazy-load if needed
        self._ensure_model_loaded(model_name)

        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' could not be loaded. "
                f"Available: {list(self.models.keys())}"
            )

        # Pre-process
        cleaned = self.preprocessor.clean_text(text)
        if not cleaned.strip():
            raise ValueError("Text is empty after preprocessing.")

        start = time.time()
        settings = get_settings()

        # Dispatch
        if model_name == "logistic_regression":
            probabilities, prediction_idx = self._predict_logistic(cleaned)
        elif model_name in ("lstm", "bilstm", "cnn"):
            probabilities, prediction_idx = self._predict_keras(
                cleaned, model_name, settings.MAX_SEQUENCE_LENGTH,
            )
        elif model_name == "distilbert":
            probabilities, prediction_idx = self._predict_distilbert(
                cleaned, settings.MAX_SEQUENCE_LENGTH,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        sentiment_map = settings.SENTIMENT_MAP
        prob_dict = self._build_probability_dict(probabilities, sentiment_map)

        label = sentiment_map.get(prediction_idx, "Neutral")
        confidence = float(max(probabilities))

        elapsed = time.time() - start
        logger.info(
            "predict | model=%s | text_len=%d | label=%s | conf=%.2f%% | time=%.3fs",
            model_name, len(text), label, confidence * 100, elapsed,
        )

        return PredictionResponse(
            label=label,
            confidence=round(confidence * 100, 2),
            model_used=model_name,
            probabilities=prob_dict,
        )

    def predict_all_models(self, text: str) -> AllModelsResponse:
        """Run every loaded model on the same text and return consensus."""
        settings = get_settings()
        results: List[PredictionResponse] = []

        # In lightweight mode, only run models that are already loaded
        models_to_run = (
            list(self.models.keys())
            if settings.LIGHTWEIGHT_MODE
            else settings.MODEL_NAMES
        )

        for model_name in models_to_run:
            try:
                result = self.predict(text, model_name)
                results.append(result)
            except Exception as exc:
                logger.error("predict_all_models — '%s' failed: %s", model_name, exc)

        if not results:
            return AllModelsResponse(results=[], consensus="Unknown", agreement_count=0)

        labels = [r.label for r in results]
        label_counts = Counter(labels)
        consensus_label, agreement_count = label_counts.most_common(1)[0]

        logger.info(
            "predict_all_models | consensus=%s | agreement=%d/%d",
            consensus_label, agreement_count, len(results),
        )

        return AllModelsResponse(
            results=results,
            consensus=consensus_label,
            agreement_count=agreement_count,
        )

    def predict_batch(
        self,
        texts: List[str],
        model_name: str = "distilbert",
    ) -> BatchPredictionResponse:
        """Predict sentiment for a batch of texts."""
        results: List[Dict] = []
        summary_counter: Counter = Counter()
        total = len(texts)

        logger.info("predict_batch | model=%s | total=%d texts", model_name, total)

        for i, text in enumerate(texts):
            try:
                pred = self.predict(text, model_name)
                results.append({
                    "text": text,
                    "label": pred.label,
                    "confidence": pred.confidence,
                    "model_used": pred.model_used,
                    "probabilities": pred.probabilities,
                })
                summary_counter[pred.label] += 1
            except Exception as exc:
                logger.error("Batch item %d failed: %s", i, exc)
                results.append({
                    "text": text,
                    "label": "Error",
                    "confidence": 0.0,
                    "model_used": model_name,
                    "probabilities": {},
                })

            if (i + 1) % 50 == 0 or (i + 1) == total:
                logger.info(
                    "Batch progress: %d/%d (%.0f%%)",
                    i + 1, total, (i + 1) / total * 100,
                )

        summary: Dict[str, int] = {
            "Positive": summary_counter.get("Positive", 0),
            "Negative": summary_counter.get("Negative", 0),
            "Neutral": summary_counter.get("Neutral", 0),
        }
        logger.info("predict_batch complete | summary=%s", summary)

        return BatchPredictionResponse(
            results=results, summary=summary, total_processed=total,
        )

    def get_available_models(self) -> List[str]:
        """Return model names currently loaded in memory."""
        return list(self.models.keys())

    # ──────────────────────────────────────────────────────────
    #  Private loaders
    # ──────────────────────────────────────────────────────────

    def _load_pickle_model(self, name: str, path: Path) -> None:
        if not path.exists():
            logger.warning("Skipping %s — file not found: %s", name, path)
            return
        start = time.time()
        try:
            with open(path, "rb") as f:
                self.models[name] = pickle.load(f)
            elapsed = time.time() - start
            self.load_times[name] = elapsed
            logger.info("Loaded %s in %.2f s", name, elapsed)
        except Exception as exc:
            logger.error("Failed to load %s: %s", name, exc)

    def _load_tfidf(self, path: Path) -> None:
        if self.tfidf is not None:
            return
        if not path.exists():
            logger.warning("TF-IDF vectoriser not found: %s", path)
            return
        start = time.time()
        try:
            with open(path, "rb") as f:
                self.tfidf = pickle.load(f)
            elapsed = time.time() - start
            self.load_times["tfidf"] = elapsed
            logger.info("Loaded TF-IDF vectoriser in %.2f s", elapsed)
        except Exception as exc:
            logger.error("Failed to load TF-IDF: %s", exc)

    def _load_keras_tokenizer(self, path: Path) -> None:
        if self.keras_tokenizer is not None:
            return
        if not path.exists():
            logger.warning("Keras tokenizer not found: %s", path)
            return
        start = time.time()
        try:
            with open(path, "rb") as f:
                self.keras_tokenizer = pickle.load(f)
            elapsed = time.time() - start
            self.load_times["keras_tokenizer"] = elapsed
            logger.info("Loaded Keras tokenizer in %.2f s", elapsed)
        except Exception as exc:
            logger.error("Failed to load Keras tokenizer: %s", exc)

    def _load_keras_model(self, name: str, path: Path) -> None:
        if not path.exists():
            logger.warning("Skipping %s — file not found: %s", name, path)
            return
        start = time.time()
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            self.models[name] = tf.keras.models.load_model(str(path))
            elapsed = time.time() - start
            self.load_times[name] = elapsed
            logger.info("Loaded %s in %.2f s", name, elapsed)
        except Exception as exc:
            logger.error("Failed to load Keras model %s: %s", name, exc)

    def _load_distilbert(self, model_path: Path, tokenizer_path: Path) -> None:
        if not model_path.exists():
            logger.warning("DistilBERT model dir not found: %s", model_path)
            return
        if not tokenizer_path.exists():
            logger.warning("DistilBERT tokenizer dir not found: %s", tokenizer_path)
            return
        start = time.time()
        try:
            import torch
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizer,
            )
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(str(tokenizer_path))
            model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
            model.to(self.device)
            model.eval()
            self.models["distilbert"] = model
            elapsed = time.time() - start
            self.load_times["distilbert"] = elapsed
            logger.info("Loaded DistilBERT in %.2f s (device=%s)", elapsed, self.device)
        except Exception as exc:
            logger.error("Failed to load DistilBERT: %s", exc)

    # ──────────────────────────────────────────────────────────
    #  Private prediction methods
    # ──────────────────────────────────────────────────────────

    def _predict_logistic(self, cleaned_text: str) -> tuple[np.ndarray, int]:
        model = self.models["logistic_regression"]
        if self.tfidf is None:
            raise RuntimeError("TF-IDF vectoriser not loaded.")
        features = self.tfidf.transform([cleaned_text])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        return probabilities, prediction

    def _predict_keras(
        self, cleaned_text: str, model_name: str, max_length: int,
    ) -> tuple[np.ndarray, int]:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        model = self.models[model_name]
        if self.keras_tokenizer is None:
            raise RuntimeError("Keras tokenizer not loaded.")
        sequence = self.keras_tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=max_length)
        probabilities = model.predict(padded, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        return probabilities, prediction

    def _predict_distilbert(
        self, cleaned_text: str, max_length: int,
    ) -> tuple[np.ndarray, int]:
        import torch
        model = self.models["distilbert"]
        if self.bert_tokenizer is None:
            raise RuntimeError("DistilBERT tokenizer not loaded.")
        inputs = self.bert_tokenizer(
            cleaned_text, return_tensors="pt",
            max_length=max_length, truncation=True, padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = int(torch.argmax(probabilities).item())
        return probabilities.cpu().numpy(), prediction

    # ──────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_probability_dict(
        probabilities: np.ndarray,
        sentiment_map: Dict[int, str],
    ) -> Dict[str, float]:
        sorted_keys = sorted(sentiment_map.keys())
        n_classes = len(probabilities)

        if n_classes == len(sorted_keys):
            return {
                sentiment_map[sorted_keys[i]]: round(float(probabilities[i]), 4)
                for i in range(n_classes)
            }

        third = n_classes // 3
        neg_prob = float(probabilities[:third].sum())
        neu_prob = float(probabilities[third : 2 * third].sum())
        pos_prob = float(probabilities[2 * third :].sum())
        return {
            "Negative": round(neg_prob, 4),
            "Neutral": round(neu_prob, 4),
            "Positive": round(pos_prob, 4),
        }


# Module-level singleton
predictor = SentimentPredictor()
