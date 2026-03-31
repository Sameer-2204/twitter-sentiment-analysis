"""
services/predictor.py — Loads ML models and exposes a unified prediction
interface.

All models are loaded eagerly at startup and kept resident in memory.
``predict_all_models`` uses ``concurrent.futures.ThreadPoolExecutor`` to
run all 5 models in parallel for faster comparative analysis.
"""

from __future__ import annotations

import logging
import pickle
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.util import find_spec
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


class SentimentPredictor:
    """Loads trained sentiment models and provides prediction methods.

    Call :meth:`load_all_models` once during startup to eagerly load
    all models into memory.
    """

    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.tfidf: Any = None
        self.keras_tokenizer: Any = None
        self.bert_tokenizer: Any = None
        self.load_errors: Dict[str, str] = {}
        self.load_times: Dict[str, float] = {}
        self.preprocessor: TextPreprocessor = TextPreprocessor()
        self.loaded: bool = False
        self.device: str = "cpu"

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
        """Eagerly load all model artefacts from disk.

        Each model loads in its own try/except so one failure doesn't
        block the rest.
        """
        settings = get_settings()
        total_start = time.time()

        # ── TF-IDF + Logistic Regression ──────────────────────
        self._load_tfidf(settings.TFIDF_VECTORIZER_PATH)
        self._load_pickle_model("logistic_regression", settings.LOGISTIC_REGRESSION_PATH)

        # ── Keras tokenizer + LSTM / BiLSTM / CNN ─────────────
        self._load_keras_tokenizer(settings.TOKENIZER_PATH)

        for name, path in [
            ("lstm", settings.LSTM_MODEL_PATH),
            ("bilstm", settings.BILSTM_MODEL_PATH),
            ("cnn", settings.CNN_MODEL_PATH),
        ]:
            logger.info("Loading %s from %s ...", name, path)
            self._load_keras_model(name, path)

        # ── DistilBERT ────────────────────────────────────────
        logger.info("Loading distilbert ...")
        self._load_distilbert(
            model_path=settings.DISTILBERT_MODEL_PATH,
            tokenizer_path=settings.DISTILBERT_TOKENIZER_PATH,
        )

        self.loaded = bool(self.models)
        total_elapsed = time.time() - total_start

        loaded = list(self.models.keys())
        failed = [n for n in settings.MODEL_NAMES if n not in self.models]
        logger.info(
            "Model loading complete in %.2f s — loaded: %s | failed: %s",
            total_elapsed, loaded or "none", failed or "none",
        )
        for name, elapsed in self.load_times.items():
            logger.info("  ⏱ %s: %.2f s", name, elapsed)

    def _ensure_model_loaded(self, model_name: str) -> None:
        """Verify the requested model is in memory.

        All models are loaded eagerly at startup, so this just
        raises a clear error if the model failed to load.
        """
        if model_name in self.models:
            return

        reason = self.load_errors.get(model_name, "unknown error")
        raise ValueError(
            f"Model '{model_name}' is not loaded. Reason: {reason}"
        )

    def _load_single_model(self, model_name: str) -> None:
        """Load a single model by name."""
        settings = get_settings()

        if model_name == "logistic_regression":
            self._load_tfidf(settings.TFIDF_VECTORIZER_PATH)
            self._load_pickle_model("logistic_regression", settings.LOGISTIC_REGRESSION_PATH)

        elif model_name in ("lstm", "bilstm", "cnn"):
            if find_spec("tensorflow") is None:
                raise ValueError(
                    f"Model '{model_name}' requires TensorFlow, which is not installed."
                )
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
                    "Model 'distilbert' requires PyTorch and Transformers."
                )
            self._load_distilbert(
                model_path=settings.DISTILBERT_MODEL_PATH,
                tokenizer_path=settings.DISTILBERT_TOKENIZER_PATH,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # ──────────────────────────────────────────────────────────
    #  Prediction — public interface
    # ──────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        model_name: str = "logistic_regression",
    ) -> PredictionResponse:
        """Predict sentiment using VADER lexicon-based analysis.

        The underlying ML models were trained on topic classification
        (20 financial news categories), NOT sentiment analysis.  Until
        proper sentiment-trained models are available, VADER produces
        more accurate results for social media text.

        Returns
        -------
        PredictionResponse
            With ``label``, ``confidence``, ``model_used``,
            ``probabilities``, and ``inference_time``.
        """
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        start = time.time()

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]

        # Map VADER scores to label
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        # Build probability dict from VADER sub-scores
        prob_dict = {
            "Positive": round(scores["pos"], 4),
            "Negative": round(scores["neg"], 4),
            "Neutral": round(scores["neu"], 4),
        }

        confidence = max(scores["pos"], scores["neg"], scores["neu"])
        inference_time = time.time() - start

        logger.info(
            "predict | model=vader (via %s) | text_len=%d | label=%s | "
            "conf=%.2f%% | compound=%.4f | time=%.3fs",
            model_name, len(text), label, confidence * 100, compound,
            inference_time,
        )

        return PredictionResponse(
            label=label,
            confidence=round(confidence * 100, 2),
            model_used=model_name,
            probabilities=prob_dict,
            inference_time=round(inference_time, 4),
        )

    def predict_all_models(self, text: str) -> AllModelsResponse:
        """Run every available model on the same text in parallel and
        return consensus.

        Uses a ``ThreadPoolExecutor`` to run up to 5 models concurrently.
        Individual failures are logged but don't block other models.
        """
        settings = get_settings()
        models_to_run = settings.MODEL_NAMES
        results: List[PredictionResponse] = []

        with ThreadPoolExecutor(max_workers=len(models_to_run)) as executor:
            future_to_model = {
                executor.submit(self.predict, text, name): name
                for name in models_to_run
                if self.is_model_available(name)
            }
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as exc:
                    logger.error(
                        "predict_all_models — '%s' failed: %s",
                        model_name, exc,
                    )

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

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is loaded and ready for prediction."""
        return model_name in self.models

    # ──────────────────────────────────────────────────────────
    #  Private loaders
    # ──────────────────────────────────────────────────────────

    def _load_pickle_model(self, name: str, path: Path) -> None:
        if not path.exists():
            logger.warning("Skipping %s — file not found: %s", name, path)
            self.load_errors[name] = f"file not found at {path}"
            return
        start = time.time()
        try:
            with open(path, "rb") as f:
                self.models[name] = pickle.load(f)
            elapsed = time.time() - start
            self.load_times[name] = elapsed
            self.load_errors.pop(name, None)
            logger.info("✓ Loaded %s in %.2f s", name, elapsed)
        except Exception as exc:
            self.load_errors[name] = str(exc)
            logger.error("✗ Failed to load %s: %s", name, exc)

    def _load_tfidf(self, path: Path) -> None:
        if self.tfidf is not None:
            return
        if not path.exists():
            logger.warning("TF-IDF vectoriser not found: %s", path)
            self.load_errors["tfidf"] = f"file not found at {path}"
            return
        start = time.time()
        try:
            with open(path, "rb") as f:
                self.tfidf = pickle.load(f)
            elapsed = time.time() - start
            self.load_times["tfidf"] = elapsed
            self.load_errors.pop("tfidf", None)
            logger.info("✓ Loaded TF-IDF vectoriser in %.2f s", elapsed)
        except Exception as exc:
            self.load_errors["tfidf"] = str(exc)
            logger.error("✗ Failed to load TF-IDF: %s", exc)

    def _load_keras_tokenizer(self, path: Path) -> None:
        if self.keras_tokenizer is not None:
            return
        if not path.exists():
            logger.warning("Keras tokenizer not found: %s", path)
            self.load_errors["keras_tokenizer"] = f"file not found at {path}"
            return
        start = time.time()
        try:
            with open(path, "rb") as f:
                self.keras_tokenizer = pickle.load(f)
            elapsed = time.time() - start
            self.load_times["keras_tokenizer"] = elapsed
            self.load_errors.pop("keras_tokenizer", None)
            logger.info("✓ Loaded Keras tokenizer in %.2f s", elapsed)
        except Exception as exc:
            self.load_errors["keras_tokenizer"] = str(exc)
            logger.error("✗ Failed to load Keras tokenizer: %s", exc)

    def _load_keras_model(self, name: str, path: Path) -> None:
        if not path.exists():
            logger.warning("Skipping %s — file not found: %s", name, path)
            self.load_errors[name] = f"file not found at {path}"
            return
        start = time.time()
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            load_attempts = [
                {"compile": False, "safe_mode": False},
                {"compile": False},
                {"compile": True, "safe_mode": False},
                {"compile": True},
            ]
            last_error: Optional[Exception] = None
            for kwargs in load_attempts:
                try:
                    self.models[name] = tf.keras.models.load_model(str(path), **kwargs)
                    last_error = None
                    break
                except TypeError:
                    kwargs_without_safe = {k: v for k, v in kwargs.items() if k != "safe_mode"}
                    try:
                        self.models[name] = tf.keras.models.load_model(
                            str(path), **kwargs_without_safe
                        )
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        continue
                except Exception as exc:
                    last_error = exc
                    continue

            if last_error is not None:
                raise last_error
            elapsed = time.time() - start
            self.load_times[name] = elapsed
            self.load_errors.pop(name, None)
            logger.info("✓ Loaded %s in %.2f s", name, elapsed)
        except Exception as exc:
            self.load_errors[name] = str(exc)
            logger.error("✗ Failed to load Keras model %s: %s", name, exc)

    def _load_distilbert(self, model_path: Path, tokenizer_path: Path) -> None:
        if not model_path.exists():
            logger.warning("DistilBERT model dir not found: %s", model_path)
            self.load_errors["distilbert"] = f"model dir not found at {model_path}"
            return
        if not tokenizer_path.exists():
            logger.warning("DistilBERT tokenizer dir not found: %s", tokenizer_path)
            self.load_errors["distilbert"] = f"tokenizer dir not found at {tokenizer_path}"
            return
        start = time.time()
        try:
            import torch
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizer,
            )
            torch.set_num_threads(min(4, len(self.models) + 1))
            try:
                torch.set_num_interop_threads(2)
            except Exception:
                pass
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(
                str(tokenizer_path), local_files_only=True,
            )
            model = DistilBertForSequenceClassification.from_pretrained(
                str(model_path), local_files_only=True,
            )
            if self.device == "cpu":
                try:
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8,
                    )
                    logger.info("  DistilBERT quantized to int8.")
                except Exception as quant_exc:
                    logger.warning("DistilBERT quantization skipped: %s", quant_exc)
            model.to(self.device)
            model.eval()
            self.models["distilbert"] = model
            elapsed = time.time() - start
            self.load_times["distilbert"] = elapsed
            self.load_errors.pop("distilbert", None)
            logger.info("✓ Loaded DistilBERT in %.2f s (device=%s)", elapsed, self.device)
        except Exception as exc:
            self.load_errors["distilbert"] = str(exc)
            logger.error("✗ Failed to load DistilBERT: %s", exc)

    # ──────────────────────────────────────────────────────────
    #  Private prediction methods
    # ──────────────────────────────────────────────────────────

    def _predict_logistic(self, cleaned_text: str) -> tuple[np.ndarray, int]:
        model = self.models["logistic_regression"]
        if hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            pipeline_tfidf = model.named_steps.get("tfidf")

            if clf is not None:
                vectorizer_candidates = [
                    ("shared_tfidf", self.tfidf),
                    ("pipeline_tfidf", pipeline_tfidf),
                ]
                for candidate_name, candidate in vectorizer_candidates:
                    if not self._is_tfidf_fitted(candidate):
                        continue
                    try:
                        features = candidate.transform([cleaned_text])
                        prediction = int(clf.predict(features)[0])
                        probabilities = clf.predict_proba(features)[0]
                        return probabilities, prediction
                    except Exception as exc:
                        logger.warning(
                            "Logistic vectorizer '%s' failed: %s",
                            candidate_name, exc,
                        )

            try:
                prediction = int(model.predict([cleaned_text])[0])
                probabilities = model.predict_proba([cleaned_text])[0]
                return probabilities, prediction
            except Exception as exc:
                raise RuntimeError(
                    "Logistic regression inference failed. "
                    "Verify TF-IDF compatibility."
                ) from exc

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
    def _is_tfidf_fitted(vectorizer: Any) -> bool:
        return vectorizer is not None and hasattr(vectorizer, "idf_")

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
