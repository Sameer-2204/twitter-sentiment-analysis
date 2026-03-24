"""
Unified prediction utilities for all trained sentiment models.

The ``SentimentPredictor`` class can load and run inference with any of:
* Logistic Regression pipeline
* LSTM
* BiLSTM
* CNN
* DistilBERT

Usage
-----
    from scripts.predict import SentimentPredictor

    predictor = SentimentPredictor(model_name="lstm")
    print(predictor.predict("Markets are looking strong today"))

    all_predictions = predictor.predict_with_all_models("This stock is awful")
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import cfg  # noqa: E402
from scripts.training_config import TrainingConfig  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "logistic": {"kind": "sklearn"},
    "lstm": {"kind": "keras"},
    "bilstm": {"kind": "keras"},
    "cnn": {"kind": "keras"},
    "distilbert": {"kind": "transformers"},
}

_WHITESPACE_RE = re.compile(r"\s+")


class SentimentPredictor:
    """Unified loader and inference wrapper for all sentiment models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialise the predictor and optionally preload a model."""
        self.config = config or TrainingConfig()
        self.default_model_name = model_name if model_name != "all" else None
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._bundles: dict[str, dict[str, Any]] = {}

        if model_name and model_name != "all":
            self.load_model(model_name)

    @property
    def supported_models(self) -> list[str]:
        """Return the list of supported model names."""
        return list(MODEL_REGISTRY.keys())

    def preprocess_texts(self, texts: Iterable[Any]) -> list[str]:
        """Apply a lightweight, inference-safe text normalization pipeline."""
        cleaned: list[str] = []
        for text in texts:
            normalized = "" if text is None else str(text)
            normalized = normalized.replace("\r", " ").replace("\n", " ")
            normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
            cleaned.append(normalized)
        return cleaned

    def _load_pickle(self, path: Path) -> Any:
        """Load a pickled artifact from disk."""
        with path.open("rb") as handle:
            return pickle.load(handle)

    def _resolve_keras_tokenizer_path(self, model_name: str) -> Path:
        """Resolve the tokenizer path for a Keras model with sensible fallbacks."""
        candidates = {
            "lstm": [
                self.config.model_dir / "tokenizer.pkl",
            ],
            "bilstm": [
                self.config.model_dir / "bilstm_tokenizer.pkl",
                self.config.model_dir / "tokenizer.pkl",
            ],
            "cnn": [
                self.config.model_dir / "cnn_tokenizer.pkl",
                self.config.model_dir / "tokenizer.pkl",
            ],
        }
        for candidate in candidates[model_name]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No tokenizer found for model={model_name!r}. "
            f"Checked: {[str(path) for path in candidates[model_name]]}"
        )

    def _resolve_model_artifact(self, model_name: str) -> Path:
        """Resolve the main artifact path for one model."""
        mapping = {
            "logistic": self.config.lr_model_path,
            "lstm": self.config.lstm_model_path,
            "bilstm": self.config.bilstm_model_path,
            "cnn": self.config.cnn_model_path,
            "distilbert": self.config.distilbert_model_dir,
        }
        artifact = mapping[model_name]
        if not artifact.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact}")
        return artifact

    def _load_transformers_bundle(self) -> dict[str, Any]:
        """Load the DistilBERT model and tokenizer."""
        model_dir = self._resolve_model_artifact("distilbert")
        tokenizer_dir = self.config.distilbert_tokenizer_dir
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

        model_kwargs: dict[str, Any] = {"local_files_only": True}
        has_pytorch_weights = any(
            (model_dir / filename).exists()
            for filename in ("pytorch_model.bin", "model.safetensors")
        )
        has_tf_weights = (model_dir / "tf_model.h5").exists()
        if not has_pytorch_weights and has_tf_weights:
            model_kwargs["from_tf"] = True

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            **model_kwargs,
        )
        model.to(self.device)
        model.eval()
        class_labels = self._default_class_labels(int(model.config.num_labels))

        return {
            "kind": "transformers",
            "model": model,
            "tokenizer": tokenizer,
            "artifact_path": model_dir,
            "class_labels": class_labels,
            "display_label_map": self._build_display_label_map(class_labels),
        }

    def load_model(self, model_name: str) -> dict[str, Any]:
        """Load one model bundle and cache it for future predictions."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model {model_name!r}. "
                f"Choose from {self.supported_models}."
            )

        if model_name in self._bundles:
            return self._bundles[model_name]

        logger.info("Loading model bundle for '%s'.", model_name)

        if model_name == "logistic":
            artifact_path = self._resolve_model_artifact(model_name)
            model = self._load_pickle(artifact_path)
            vectorizer_path = self.config.tfidf_path
            vectorizer = None
            classes_ = getattr(model, "classes_", None)
            if classes_ is None:
                raise ValueError("Logistic model does not expose learned class labels.")
            if not isinstance(model, Pipeline):
                if not vectorizer_path.exists():
                    raise FileNotFoundError(
                        "Logistic Regression model requires a standalone TF-IDF "
                        f"vectorizer, but none was found at {vectorizer_path}"
                    )
                vectorizer = self._load_pickle(vectorizer_path)
            class_labels = self._normalize_class_labels(classes_)
            bundle = {
                "kind": "sklearn",
                "model": model,
                "vectorizer": vectorizer,
                "artifact_path": artifact_path,
                "class_labels": class_labels,
                "display_label_map": self._build_display_label_map(class_labels),
            }
        elif model_name in {"lstm", "bilstm", "cnn"}:
            artifact_path = self._resolve_model_artifact(model_name)
            tokenizer_path = self._resolve_keras_tokenizer_path(model_name)
            keras_model = load_model(artifact_path, compile=False)
            output_dim = int(keras_model.output_shape[-1])
            class_labels = self._default_class_labels(output_dim)
            bundle = {
                "kind": "keras",
                "model": keras_model,
                "tokenizer": self._load_pickle(tokenizer_path),
                "artifact_path": artifact_path,
                "tokenizer_path": tokenizer_path,
                "class_labels": class_labels,
                "display_label_map": self._build_display_label_map(class_labels),
            }
        else:
            bundle = self._load_transformers_bundle()

        self._bundles[model_name] = bundle
        return bundle

    def _align_probabilities(
        self,
        probabilities: np.ndarray,
        classes_: Optional[Iterable[int]] = None,
        target_class_labels: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Align probability columns to the project label order."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if classes_ is None or target_class_labels is None:
            return probabilities

        classes = list(classes_)
        normalized_classes = self._normalize_class_labels(classes)
        aligned = np.zeros(
            (probabilities.shape[0], len(target_class_labels)),
            dtype=np.float64,
        )
        class_position_map = {
            int(class_label): position
            for position, class_label in enumerate(target_class_labels)
        }
        for column_index, normalized_index in enumerate(normalized_classes):
            if normalized_index not in class_position_map:
                raise ValueError(
                    f"Normalized class index {normalized_index} was not found in "
                    f"target labels {target_class_labels}."
                )
            aligned[:, class_position_map[normalized_index]] = probabilities[
                :,
                column_index,
            ]
        return aligned

    def _normalize_class_index(
        self,
        class_index: int,
        allow_identity: bool = False,
    ) -> int:
        """Map raw dataset labels such as 0/2/4 onto project label indices."""
        int_index = int(class_index)
        if int_index in self.config.sentiment_labels_inv:
            return int_index

        if int_index in cfg.RAW_LABELS:
            label_name = cfg.RAW_LABELS[int_index]
            return int(self.config.sentiment_labels[label_name])

        if allow_identity:
            return int_index

        raise ValueError(
            f"Unable to map model class index {class_index!r} "
            "to the configured sentiment labels."
        )

    def _normalize_class_labels(self, class_labels: Iterable[int]) -> list[int]:
        """Normalize a full class-label set while avoiding accidental remapping."""
        normalized = [int(class_label) for class_label in class_labels]
        raw_label_keys = sorted(cfg.RAW_LABELS.keys())
        if sorted(normalized) == raw_label_keys:
            return [
                int(self.config.sentiment_labels[cfg.RAW_LABELS[class_label]])
                for class_label in normalized
            ]
        return normalized

    def _default_class_labels(self, num_classes: int) -> list[int]:
        """Return the class labels that should correspond to output columns."""
        configured_labels = sorted(self.config.sentiment_labels_inv)
        if num_classes == len(configured_labels):
            return configured_labels
        return list(range(num_classes))

    def _build_display_label_map(self, class_labels: list[int]) -> dict[int, str]:
        """Create a display label mapping for the given class labels."""
        configured_labels = sorted(self.config.sentiment_labels_inv)
        if class_labels == configured_labels:
            return {
                label: self.config.sentiment_labels_inv[label]
                for label in class_labels
            }
        return {label: str(label) for label in class_labels}

    def get_class_labels(self, model_name: str) -> list[int]:
        """Return the ordered class labels for one model."""
        bundle = self.load_model(model_name)
        return list(bundle["class_labels"])

    def get_display_label_map(self, model_name: str) -> dict[int, str]:
        """Return the display label mapping for one model."""
        bundle = self.load_model(model_name)
        return dict(bundle["display_label_map"])

    def predict_proba_batch(
        self,
        texts: Iterable[Any],
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return class-probability predictions for a batch of texts."""
        selected_model = model_name or self.default_model_name
        if selected_model is None:
            raise ValueError(
                "No model selected. Pass model_name explicitly or initialise "
                "SentimentPredictor(model_name=...)."
            )

        texts_list = self.preprocess_texts(texts)
        if not texts_list:
            bundle = self.load_model(selected_model)
            return np.empty((0, len(bundle["class_labels"])), dtype=np.float64)

        bundle = self.load_model(selected_model)
        kind = bundle["kind"]

        if kind == "sklearn":
            model = bundle["model"]
            if bundle.get("vectorizer") is not None:
                features = bundle["vectorizer"].transform(texts_list)
                raw_probabilities = model.predict_proba(features)
                classes_ = getattr(model, "classes_", None)
            else:
                raw_probabilities = model.predict_proba(texts_list)
                classes_ = getattr(model, "classes_", None)
            probabilities = self._align_probabilities(
                raw_probabilities,
                classes_,
                bundle["class_labels"],
            )
            return probabilities

        if kind == "keras":
            sequences = bundle["tokenizer"].texts_to_sequences(texts_list)
            padded_sequences = pad_sequences(
                sequences,
                maxlen=self.config.max_sequence_length,
                padding="post",
                truncating="post",
            )
            probabilities = bundle["model"].predict(
                padded_sequences,
                batch_size=batch_size,
                verbose=0,
            )
            return np.asarray(probabilities, dtype=np.float64)

        all_probabilities: list[np.ndarray] = []
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]

        for start_index in range(0, len(texts_list), batch_size):
            batch_texts = texts_list[start_index : start_index + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=self.config.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                probabilities = torch.softmax(outputs.logits, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

        return np.vstack(all_probabilities)

    def _format_prediction_output(
        self,
        model_name: str,
        texts: list[str],
        probabilities: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Convert probability matrices into user-facing prediction objects."""
        predicted_indices = np.argmax(probabilities, axis=1)
        confidence_scores = np.max(probabilities, axis=1)
        bundle = self.load_model(model_name)
        class_labels = bundle["class_labels"]
        display_label_map = bundle["display_label_map"]

        results: list[dict[str, Any]] = []
        for index, text in enumerate(texts):
            class_label = int(class_labels[int(predicted_indices[index])])
            probability_map = {
                display_label_map[int(class_labels[class_index])]: float(
                    probabilities[index, class_index]
                )
                for class_index in range(len(class_labels))
            }
            results.append(
                {
                    "model": model_name,
                    "text": text,
                    "label": display_label_map[class_label],
                    "label_index": class_label,
                    "confidence": float(confidence_scores[index]),
                    "probabilities": probability_map,
                }
            )
        return results

    def predict_batch(
        self,
        texts: Iterable[Any],
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Return predictions for a batch of texts."""
        selected_model = model_name or self.default_model_name
        if selected_model is None:
            raise ValueError(
                "No model selected. Pass model_name explicitly or initialise "
                "SentimentPredictor(model_name=...)."
            )

        normalized_texts = self.preprocess_texts(texts)
        probabilities = self.predict_proba_batch(
            normalized_texts,
            model_name=selected_model,
            batch_size=batch_size,
        )
        return self._format_prediction_output(
            model_name=selected_model,
            texts=normalized_texts,
            probabilities=probabilities,
        )

    def predict(
        self,
        text: Any,
        model_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Predict one text and return the sentiment label with confidence."""
        predictions = self.predict_batch([text], model_name=model_name)
        return predictions[0]

    def predict_with_all_models(self, text: Any) -> dict[str, dict[str, Any]]:
        """Run one text through all supported models for side-by-side comparison."""
        return {
            model_name: self.predict(text, model_name=model_name)
            for model_name in self.supported_models
        }

    def get_model_size_mb(self, model_name: str) -> float:
        """Return the total model artifact size in megabytes."""
        if model_name == "distilbert":
            artifact_path = self._resolve_model_artifact(model_name)
            total_bytes = sum(
                file_path.stat().st_size
                for file_path in artifact_path.rglob("*")
                if file_path.is_file()
            )
        elif model_name == "logistic":
            artifact_path = self._resolve_model_artifact(model_name)
            total_bytes = artifact_path.stat().st_size
            if self.config.tfidf_path.exists():
                total_bytes += self.config.tfidf_path.stat().st_size
        else:
            artifact_path = self._resolve_model_artifact(model_name)
            total_bytes = artifact_path.stat().st_size
        return total_bytes / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ad-hoc local prediction."""
    parser = argparse.ArgumentParser(
        description="Run sentiment prediction with one model or all models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Model to use for prediction.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to classify.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for quick manual predictions."""
    args = parse_args()
    predictor = SentimentPredictor(
        model_name=None if args.model == "all" else args.model,
    )

    if args.model == "all":
        output = predictor.predict_with_all_models(args.text)
    else:
        output = predictor.predict(args.text)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
