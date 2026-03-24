"""
Interactive and static visualization pipeline for Twitter sentiment analysis.

This module provides a ``Visualizer`` class that can generate dataset EDA,
text analysis, temporal analysis, advanced plots, and model-performance charts.
Each visualization is saved to ``reports/figures`` as:
* a static PNG built with matplotlib / seaborn
* an HTML artifact built with Plotly when applicable
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from wordcloud import WordCloud

__all__ = ["Visualizer", "FigureArtifact", "to_serializable"]

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore[assignment]

try:
    import umap  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    umap = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.training_config import TrainingConfig  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

sns.set_theme(style="whitegrid")

POSITIVE_COLOR = "#4CAF50"
NEUTRAL_COLOR = "#FFC107"
NEGATIVE_COLOR = "#F44336"
EXTRA_COLORS = [
    "#2196F3",
    "#9C27B0",
    "#009688",
    "#FF5722",
    "#3F51B5",
    "#795548",
    "#607D8B",
    "#8BC34A",
    "#E91E63",
    "#00BCD4",
    "#FF9800",
    "#673AB7",
    "#CDDC39",
    "#FF6F61",
    "#26A69A",
    "#5C6BC0",
    "#EC407A",
]
DEFAULT_COLORS = [NEGATIVE_COLOR, NEUTRAL_COLOR, POSITIVE_COLOR, *EXTRA_COLORS]

HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@(\w+)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
TIMESTAMP_CANDIDATES = [
    "timestamp",
    "created_at",
    "date",
    "datetime",
    "tweet_created_at",
    "time",
]
MODEL_NAMES = ["logistic", "lstm", "bilstm", "cnn", "distilbert"]


@dataclass
class FigureArtifact:
    """Metadata describing one generated chart."""

    name: str
    title: str
    section: str
    png_path: Optional[str]
    html_path: Optional[str]
    description: str
    status: str = "generated"
    notes: Optional[str] = None


def to_serializable(value: Any) -> Any:
    """Convert numpy and pandas objects into JSON-safe values."""
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


class Visualizer:
    """Generate EDA and model-performance plots for the project."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset_path: Optional[Path] = None,
        evaluation_dataset_path: Optional[Path] = None,
        figures_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None,
        max_projection_samples: int = 1500,
        performance_sample_size: int = 300,
    ) -> None:
        self.config = config or TrainingConfig()
        self.config.ensure_dirs()
        self.reports_dir = reports_dir or self.config.reports_dir
        self.figures_dir = figures_dir or (self.reports_dir / "figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.evaluation_dataset_path = (
            Path(evaluation_dataset_path) if evaluation_dataset_path else None
        )
        self.max_projection_samples = max_projection_samples
        self.performance_sample_size = performance_sample_size
        self.plotly_template = "plotly_white"
        self.artifacts: list[FigureArtifact] = []
        self.skipped_items: list[dict[str, Any]] = []
        self._predictor: Optional[Any] = None
        self._vader = (
            SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None
        )
        self._performance_cache: Optional[dict[str, Any]] = None

        self.dataset = self._load_dataset()
        self.evaluation_df, self.evaluation_source = self._load_evaluation_dataset()
        self.labels = sorted(self.dataset["label"].astype(int).unique().tolist())
        self.label_display_map = self._build_label_display_map(self.labels)
        self.dataset["label_name"] = self.dataset["label"].map(self.label_display_map)
        self.evaluation_df["label_name"] = self.evaluation_df["label"].map(
            self.label_display_map
        )
        self.color_map = self._build_color_map(
            [self.label_display_map[label] for label in self.labels]
        )
        self.training_summary = self._load_json_if_exists(
            self.reports_dir / "training_summary.json"
        )
        self.model_report = self._load_json_if_exists(
            self.reports_dir / "model_comparison_report.json"
        )

    @property
    def predictor(self) -> Any:
        """Load the predictor lazily because model imports are expensive."""
        if self._predictor is None:
            from scripts.predict import SentimentPredictor  # noqa: WPS433

            self._predictor = SentimentPredictor(config=self.config)
        return self._predictor

    def _load_json_if_exists(self, path: Path) -> Optional[dict[str, Any]]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_label_display_map(self, labels: list[int]) -> dict[int, str]:
        config_labels = sorted(self.config.sentiment_labels_inv.keys())
        if labels == config_labels:
            return {
                label: self.config.sentiment_labels_inv[label]
                for label in labels
            }
        return {label: str(label) for label in labels}

    def _build_color_map(self, label_names: list[str]) -> dict[str, str]:
        colors = {
            "Positive": POSITIVE_COLOR,
            "Neutral": NEUTRAL_COLOR,
            "Negative": NEGATIVE_COLOR,
            "positive": POSITIVE_COLOR,
            "neutral": NEUTRAL_COLOR,
            "negative": NEGATIVE_COLOR,
        }
        palette = iter(DEFAULT_COLORS)
        for label_name in label_names:
            if label_name not in colors:
                colors[label_name] = next(palette, "#455A64")
        return colors

    def _find_timestamp_column(self, dataframe: pd.DataFrame) -> Optional[str]:
        for column in TIMESTAMP_CANDIDATES:
            if column in dataframe.columns:
                return column
        return None

    def _standardize_text(self, text: Any) -> str:
        value = "" if pd.isna(text) else str(text)
        value = value.replace("\r", " ").replace("\n", " ")
        return WHITESPACE_RE.sub(" ", value).strip()

    def _uppercase_ratio(self, text: str) -> float:
        letters = [character for character in text if character.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for character in letters if character.isupper()) / len(letters)

    def _vader_scores(self, text: str) -> dict[str, float]:
        if self._vader is None:
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
        return self._vader.polarity_scores(text)

    def _prepare_dataframe(self, dataframe: pd.DataFrame, split_name: str) -> pd.DataFrame:
        frame = dataframe.copy()
        required = {"text", "label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Dataset {split_name!r} is missing required columns: {sorted(missing)}"
            )
        frame = frame.dropna(subset=["text", "label"]).reset_index(drop=True)
        frame["split"] = split_name
        frame["text"] = frame["text"].map(self._standardize_text)
        frame["label"] = frame["label"].astype(int)
        frame["char_length"] = frame["text"].str.len()
        frame["word_count"] = frame["text"].str.split().str.len()
        frame["hashtags"] = frame["text"].str.findall(HASHTAG_RE)
        frame["mentions"] = frame["text"].str.findall(MENTION_RE)
        frame["hashtag_count"] = frame["hashtags"].str.len()
        frame["mention_count"] = frame["mentions"].str.len()
        frame["url_count"] = frame["text"].str.count(URL_RE)
        frame["exclamation_count"] = frame["text"].str.count("!")
        frame["question_count"] = frame["text"].str.count(r"\?")
        frame["uppercase_ratio"] = frame["text"].map(self._uppercase_ratio)

        timestamp_column = self._find_timestamp_column(frame)
        if timestamp_column is not None:
            frame["parsed_timestamp"] = pd.to_datetime(
                frame[timestamp_column],
                errors="coerce",
                utc=False,
            )
            frame["hour_of_day"] = frame["parsed_timestamp"].dt.hour
        else:
            frame["parsed_timestamp"] = pd.NaT
            frame["hour_of_day"] = np.nan

        if self._vader is not None:
            vader_df = pd.DataFrame(frame["text"].map(self._vader_scores).tolist())
            vader_df.columns = [f"vader_{column}" for column in vader_df.columns]
            frame = pd.concat([frame, vader_df], axis=1)

        return frame

    def _load_dataset(self) -> pd.DataFrame:
        if self.dataset_path is not None:
            return self._prepare_dataframe(
                pd.read_csv(self.dataset_path),
                self.dataset_path.stem,
            )

        frames: list[pd.DataFrame] = []
        for split_name, path in [
            ("train", self.config.train_csv),
            ("valid", self.config.valid_csv),
            ("test", self.config.test_csv),
        ]:
            if path.exists():
                frames.append(self._prepare_dataframe(pd.read_csv(path), split_name))
        if not frames:
            raise FileNotFoundError("No dataset CSVs found for visualization.")
        return pd.concat(frames, ignore_index=True)

    def _load_evaluation_dataset(self) -> tuple[pd.DataFrame, str]:
        if self.evaluation_dataset_path is not None:
            path = self.evaluation_dataset_path
            return self._prepare_dataframe(pd.read_csv(path), path.stem), path.stem
        for split_name, path in [
            ("test", self.config.test_csv),
            ("valid", self.config.valid_csv),
            ("train", self.config.train_csv),
        ]:
            if path.exists():
                return self._prepare_dataframe(pd.read_csv(path), split_name), split_name
        raise FileNotFoundError("No evaluation dataset CSVs found.")

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return slug or "plot"

    def _relative_path(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        try:
            return str(path.relative_to(self.reports_dir)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

    def _write_plotly_figure(self, figure: go.Figure, html_path: Path) -> None:
        figure.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    def _save_matplotlib(self, figure: plt.Figure, png_path: Path) -> None:
        figure.tight_layout()
        figure.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(figure)

    def _register_artifact(
        self,
        name: str,
        title: str,
        section: str,
        png_path: Optional[Path],
        html_path: Optional[Path],
        description: str,
        notes: Optional[str] = None,
        status: str = "generated",
    ) -> FigureArtifact:
        artifact = FigureArtifact(
            name=name,
            title=title,
            section=section,
            png_path=self._relative_path(png_path),
            html_path=self._relative_path(html_path),
            description=description,
            status=status,
            notes=notes,
        )
        self.artifacts.append(artifact)
        return artifact

    def _register_skip(self, name: str, section: str, reason: str) -> FigureArtifact:
        logger.warning("Skipping %s: %s", name, reason)
        self.skipped_items.append({"name": name, "section": section, "reason": reason})
        return self._register_artifact(
            name=name,
            title=name.replace("_", " ").title(),
            section=section,
            png_path=None,
            html_path=None,
            description=reason,
            notes=reason,
            status="skipped",
        )

    def _safe_call(
        self,
        func: Callable[..., FigureArtifact],
        name: str,
        section: str,
        **kwargs: Any,
    ) -> FigureArtifact:
        """Call *func* and return the artifact, or register a skip on failure."""
        try:
            return func(**kwargs)
        except Exception as exc:
            logger.exception("Failed to generate %s", name)
            return self._register_skip(name=name, section=section, reason=str(exc))

    def _save_dual_figure(
        self,
        name: str,
        title: str,
        section: str,
        description: str,
        plotly_figure: go.Figure,
        matplotlib_builder: Callable[[Path], None],
        notes: Optional[str] = None,
    ) -> FigureArtifact:
        html_path = self.figures_dir / f"{name}.html"
        png_path = self.figures_dir / f"{name}.png"
        self._write_plotly_figure(plotly_figure, html_path)
        matplotlib_builder(png_path)
        return self._register_artifact(
            name=name,
            title=title,
            section=section,
            png_path=png_path,
            html_path=html_path,
            description=description,
            notes=notes,
        )

    def _create_plotly_table(self, dataframe: pd.DataFrame, title: str) -> go.Figure:
        figure = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": list(dataframe.columns),
                        "fill_color": "#263238",
                        "font": {"color": "white"},
                        "align": "left",
                    },
                    cells={
                        "values": [dataframe[column] for column in dataframe.columns],
                        "fill_color": "#FAFAFA",
                        "align": "left",
                    },
                )
            ]
        )
        figure.update_layout(
            title=title,
            template=self.plotly_template,
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        return figure

    def _dataset_statistics_table(self) -> pd.DataFrame:
        rows = [
            ("rows", len(self.dataset)),
            ("unique_labels", self.dataset["label"].nunique()),
            ("avg_char_length", round(float(self.dataset["char_length"].mean()), 2)),
            ("median_char_length", round(float(self.dataset["char_length"].median()), 2)),
            ("avg_word_count", round(float(self.dataset["word_count"].mean()), 2)),
            ("median_word_count", round(float(self.dataset["word_count"].median()), 2)),
            ("avg_hashtag_count", round(float(self.dataset["hashtag_count"].mean()), 2)),
            ("avg_mention_count", round(float(self.dataset["mention_count"].mean()), 2)),
            ("timestamp_rows", int(self.dataset["parsed_timestamp"].notna().sum())),
            ("num_splits_loaded", self.dataset["split"].nunique()),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def _table_matplotlib_builder(self, dataframe: pd.DataFrame) -> Callable[[Path], None]:
        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(8, max(2.5, len(dataframe) * 0.4)))
            axis.axis("off")
            table = axis.table(
                cellText=dataframe.values,
                colLabels=dataframe.columns,
                cellLoc="left",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.4)
            self._save_matplotlib(figure, png_path)

        return builder

    def _top_counter_from_series(
        self,
        series: Iterable[list[str]],
        limit: int = 20,
    ) -> pd.DataFrame:
        counter: Counter[str] = Counter()
        for items in series:
            counter.update(token.lower() for token in items)
        return pd.DataFrame(counter.most_common(limit), columns=["token", "count"])

    def _count_top_ngrams(
        self,
        texts: pd.Series,
        ngram_value: int,
        top_n: int = 20,
    ) -> pd.DataFrame:
        cleaned = texts.dropna().astype(str)
        if cleaned.empty:
            return pd.DataFrame(columns=["ngram", "count"])
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(ngram_value, ngram_value),
            min_df=1,
        )
        try:
            matrix = vectorizer.fit_transform(cleaned)
        except ValueError:
            return pd.DataFrame(columns=["ngram", "count"])
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        grams = np.asarray(vectorizer.get_feature_names_out())
        order = np.argsort(counts)[::-1][:top_n]
        return pd.DataFrame(
            {"ngram": grams[order], "count": counts[order].astype(int)}
        )

    def _wordcloud_plotly_figure(self, image_bytes: bytes, title: str) -> go.Figure:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        figure = go.Figure()
        figure.add_layout_image(
            {
                "source": f"data:image/png;base64,{encoded}",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1,
                "sizex": 1,
                "sizey": 1,
                "sizing": "stretch",
                "layer": "below",
            }
        )
        figure.update_xaxes(visible=False, range=[0, 1])
        figure.update_yaxes(visible=False, range=[0, 1])
        figure.update_layout(
            title=title,
            template=self.plotly_template,
            margin={"l": 10, "r": 10, "t": 60, "b": 10},
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return figure

    def available_models(self) -> list[str]:
        candidates = {
            "logistic": self.config.lr_model_path.exists(),
            "lstm": self.config.lstm_model_path.exists(),
            "bilstm": self.config.bilstm_model_path.exists(),
            "cnn": self.config.cnn_model_path.exists(),
            "distilbert": self.config.distilbert_model_dir.exists()
            and self.config.distilbert_tokenizer_dir.exists(),
        }
        return [model for model, exists in candidates.items() if exists]

    def _resolve_models(self, models: Optional[list[str]]) -> list[str]:
        requested = models or list(MODEL_NAMES)
        available = set(self.available_models())
        resolved = [model for model in requested if model in available]
        for model in requested:
            if model not in available:
                self._register_skip(
                    name=f"{model}_artifacts_missing",
                    section="model_performance",
                    reason=f"Required artifacts for model={model!r} were not found.",
                )
        return resolved

    def _align_probabilities(
        self,
        probabilities: np.ndarray,
        model_labels: list[int],
        target_labels: list[int],
    ) -> np.ndarray:
        aligned = np.zeros((probabilities.shape[0], len(target_labels)), dtype=np.float64)
        target_positions = {label: index for index, label in enumerate(target_labels)}
        for column_index, label in enumerate(model_labels):
            if int(label) not in target_positions:
                continue
            aligned[:, target_positions[int(label)]] = probabilities[:, column_index]
        row_sums = aligned.sum(axis=1, keepdims=True)
        valid_rows = row_sums.squeeze() > 0
        aligned[valid_rows] = aligned[valid_rows] / row_sums[valid_rows]
        return aligned

    def _model_prediction_bundle(
        self,
        models: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
    ) -> dict[str, Any]:
        cache_key = (tuple(sorted(models or [])), max_samples or self.performance_sample_size)
        if self._performance_cache is not None and self._performance_cache.get("cache_key") == cache_key:
            return self._performance_cache

        resolved_models = self._resolve_models(models)
        if not resolved_models:
            raise FileNotFoundError("No trained model artifacts available for performance plots.")

        evaluation_df = self.evaluation_df.copy()
        if max_samples is None:
            max_samples = self.performance_sample_size
        if max_samples and len(evaluation_df) > max_samples:
            evaluation_df = evaluation_df.sample(
                n=max_samples,
                random_state=self.config.random_seed,
            ).sort_index()

        y_true = evaluation_df["label"].astype(int).to_numpy()
        texts = evaluation_df["text"].astype(str).tolist()
        label_indices = sorted(evaluation_df["label"].astype(int).unique().tolist())
        label_names = [self.label_display_map[label] for label in label_indices]

        per_model: dict[str, Any] = {}
        comparison_rows: list[dict[str, Any]] = []
        for model_name in resolved_models:
            logger.info("Collecting performance predictions for %s", model_name)
            start = time.perf_counter()
            raw_probabilities = self.predictor.predict_proba_batch(
                texts,
                model_name=model_name,
            )
            elapsed = time.perf_counter() - start
            probabilities = self._align_probabilities(
                probabilities=raw_probabilities,
                model_labels=self.predictor.get_class_labels(model_name),
                target_labels=label_indices,
            )
            predictions = np.asarray(
                [label_indices[index] for index in np.argmax(probabilities, axis=1)],
                dtype=int,
            )
            confidences = probabilities.max(axis=1)
            summary = {
                "model": model_name,
                "accuracy": accuracy_score(y_true, predictions),
                "precision_macro": precision_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "recall_macro": recall_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "f1_macro": f1_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "avg_prediction_time_ms": (elapsed / max(len(texts), 1)) * 1000,
                "model_size_mb": self.predictor.get_model_size_mb(model_name),
            }
            y_true_bin = label_binarize(y_true, classes=label_indices)
            try:
                summary["roc_auc_ovr_macro"] = roc_auc_score(
                    y_true_bin,
                    probabilities,
                    average="macro",
                    multi_class="ovr",
                )
            except ValueError:
                summary["roc_auc_ovr_macro"] = None

            per_model[model_name] = {
                "summary": summary,
                "y_true": y_true,
                "predictions": predictions,
                "probabilities": probabilities,
                "confidences": confidences,
                "confusion_matrix": confusion_matrix(
                    y_true,
                    predictions,
                    labels=label_indices,
                ),
            }
            comparison_rows.append(summary)

        bundle = {
            "cache_key": cache_key,
            "evaluation_source": self.evaluation_source,
            "sample_size": len(evaluation_df),
            "label_indices": label_indices,
            "label_names": label_names,
            "dataframe": evaluation_df,
            "per_model": per_model,
            "comparison_df": pd.DataFrame(comparison_rows).sort_values(
                by=["f1_macro", "accuracy"],
                ascending=False,
            ),
        }
        self._performance_cache = bundle
        return bundle

    def plot_sentiment_distribution_bar(self) -> FigureArtifact:
        counts = (
            self.dataset["label_name"]
            .value_counts()
            .rename_axis("label_name")
            .reset_index(name="count")
        )
        plotly_figure = px.bar(
            counts,
            x="label_name",
            y="count",
            color="label_name",
            color_discrete_map=self.color_map,
            title="Sentiment Class Distribution",
            template=self.plotly_template,
        )
        plotly_figure.update_layout(showlegend=False, xaxis_title="Sentiment", yaxis_title="Tweets")

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=counts,
                x="label_name",
                y="count",
                hue="label_name",
                dodge=False,
                palette=self.color_map,
                ax=axis,
            )
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
            axis.set_title("Sentiment Class Distribution")
            axis.set_xlabel("Sentiment")
            axis.set_ylabel("Tweets")
            axis.tick_params(axis="x", rotation=45)
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="dataset_sentiment_distribution_bar",
            title="Sentiment Class Distribution",
            section="dataset_overview",
            description="Bar chart showing the number of tweets in each sentiment class.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_sentiment_distribution_donut(self) -> FigureArtifact:
        counts = (
            self.dataset["label_name"]
            .value_counts()
            .rename_axis("label_name")
            .reset_index(name="count")
        )
        plotly_figure = px.pie(
            counts,
            names="label_name",
            values="count",
            hole=0.45,
            color="label_name",
            color_discrete_map=self.color_map,
            title="Sentiment Distribution Donut Chart",
            template=self.plotly_template,
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(8, 8))
            axis.pie(
                counts["count"],
                labels=counts["label_name"],
                colors=[self.color_map[label] for label in counts["label_name"]],
                wedgeprops={"width": 0.45},
                autopct="%1.1f%%",
                startangle=90,
            )
            axis.set_title("Sentiment Distribution Donut Chart")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="dataset_sentiment_distribution_donut",
            title="Sentiment Distribution Donut Chart",
            section="dataset_overview",
            description="Donut chart view of class balance across the dataset.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_tweet_length_distribution(self) -> FigureArtifact:
        plotly_figure = px.histogram(
            self.dataset,
            x="char_length",
            color="label_name",
            nbins=40,
            opacity=0.7,
            marginal="box",
            color_discrete_map=self.color_map,
            title="Tweet Length Distribution by Sentiment",
            template=self.plotly_template,
        )
        plotly_figure.update_layout(
            xaxis_title="Character Length",
            yaxis_title="Count",
            legend_title="Sentiment",
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 6))
            sns.histplot(
                data=self.dataset,
                x="char_length",
                hue="label_name",
                bins=40,
                multiple="layer",
                alpha=0.45,
                palette=self.color_map,
                ax=axis,
            )
            axis.set_title("Tweet Length Distribution by Sentiment")
            axis.set_xlabel("Character Length")
            axis.set_ylabel("Count")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="dataset_tweet_length_distribution",
            title="Tweet Length Distribution by Sentiment",
            section="dataset_overview",
            description="Histogram of tweet character lengths grouped by sentiment.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_word_count_distribution(self) -> FigureArtifact:
        plotly_figure = px.histogram(
            self.dataset,
            x="word_count",
            color="label_name",
            nbins=30,
            opacity=0.7,
            marginal="box",
            color_discrete_map=self.color_map,
            title="Word Count Distribution by Sentiment",
            template=self.plotly_template,
        )
        plotly_figure.update_layout(
            xaxis_title="Word Count",
            yaxis_title="Count",
            legend_title="Sentiment",
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 6))
            sns.histplot(
                data=self.dataset,
                x="word_count",
                hue="label_name",
                bins=30,
                multiple="layer",
                alpha=0.45,
                palette=self.color_map,
                ax=axis,
            )
            axis.set_title("Word Count Distribution by Sentiment")
            axis.set_xlabel("Word Count")
            axis.set_ylabel("Count")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="dataset_word_count_distribution",
            title="Word Count Distribution by Sentiment",
            section="dataset_overview",
            description="Histogram of tweet word counts grouped by sentiment.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_dataset_statistics_table(self) -> FigureArtifact:
        summary_table = self._dataset_statistics_table()
        plotly_figure = self._create_plotly_table(
            summary_table,
            title="Dataset Statistics Summary",
        )
        return self._save_dual_figure(
            name="dataset_statistics_summary",
            title="Dataset Statistics Summary",
            section="dataset_overview",
            description="Tabular summary of overall dataset statistics.",
            plotly_figure=plotly_figure,
            matplotlib_builder=self._table_matplotlib_builder(summary_table),
        )

    def plot_wordcloud_for_label(self, label: int) -> FigureArtifact:
        label_name = self.label_display_map[label]
        subset = self.dataset[self.dataset["label"] == label]
        if subset.empty:
            return self._register_skip(
                name=f"wordcloud_{self._slugify(label_name)}",
                section="text_analysis",
                reason=f"No rows available for label={label_name!r}.",
            )
        text_blob = " ".join(subset["text"].astype(str).tolist()).strip()
        if not text_blob:
            return self._register_skip(
                name=f"wordcloud_{self._slugify(label_name)}",
                section="text_analysis",
                reason=f"Text content is empty for label={label_name!r}.",
            )

        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            collocations=False,
            colormap="viridis",
        ).generate(text_blob)
        buffer = io.BytesIO()
        image = wordcloud.to_image()
        image.save(buffer, format="PNG")
        plotly_figure = self._wordcloud_plotly_figure(
            buffer.getvalue(),
            title=f"Word Cloud - {label_name}",
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(12, 6))
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(f"Word Cloud - {label_name}")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name=f"wordcloud_{self._slugify(label_name)}",
            title=f"Word Cloud - {label_name}",
            section="text_analysis",
            description=f"Most prominent words for label {label_name}.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def _plot_horizontal_bar(
        self,
        dataframe: pd.DataFrame,
        title: str,
        name: str,
        section: str,
        y_column: str,
        color: str,
        description: str,
    ) -> FigureArtifact:
        if dataframe.empty:
            return self._register_skip(name=name, section=section, reason=f"No data available for {title}.")
        plotly_figure = px.bar(
            dataframe,
            x="count",
            y=y_column,
            orientation="h",
            title=title,
            template=self.plotly_template,
            color_discrete_sequence=[color],
        )
        plotly_figure.update_layout(xaxis_title="Count", yaxis_title="")

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=dataframe,
                x="count",
                y=y_column,
                orient="h",
                color=color,
                ax=axis,
            )
            axis.set_title(title)
            axis.set_xlabel("Count")
            axis.set_ylabel("")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name=name,
            title=title,
            section=section,
            description=description,
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_top_words_for_label(self, label: int, top_n: int = 20) -> FigureArtifact:
        label_name = self.label_display_map[label]
        subset = self.dataset[self.dataset["label"] == label]
        vectorizer = CountVectorizer(stop_words="english")
        try:
            matrix = vectorizer.fit_transform(subset["text"])
        except ValueError:
            return self._register_skip(
                name=f"top_words_{self._slugify(label_name)}",
                section="text_analysis",
                reason=f"No vocabulary available for label={label_name!r}.",
            )
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        words = np.asarray(vectorizer.get_feature_names_out())
        order = np.argsort(counts)[::-1][:top_n]
        dataframe = pd.DataFrame(
            {"token": words[order], "count": counts[order].astype(int)}
        ).sort_values("count", ascending=True)
        return self._plot_horizontal_bar(
            dataframe=dataframe,
            title=f"Top {top_n} Words - {label_name}",
            name=f"top_words_{self._slugify(label_name)}",
            section="text_analysis",
            y_column="token",
            color=self.color_map[label_name],
            description=f"Most frequent unigrams for label {label_name}.",
        )

    def plot_top_ngrams_for_label(
        self,
        label: int,
        ngram_value: int,
        top_n: int = 20,
    ) -> FigureArtifact:
        label_name = self.label_display_map[label]
        subset = self.dataset[self.dataset["label"] == label]
        dataframe = self._count_top_ngrams(
            texts=subset["text"],
            ngram_value=ngram_value,
            top_n=top_n,
        ).sort_values("count", ascending=True)
        ngram_name = {2: "bigrams", 3: "trigrams"}.get(ngram_value, f"{ngram_value}-grams")
        return self._plot_horizontal_bar(
            dataframe=dataframe,
            title=f"Top {top_n} {ngram_name.title()} - {label_name}",
            name=f"top_{self._slugify(ngram_name)}_{self._slugify(label_name)}",
            section="text_analysis",
            y_column="ngram",
            color=self.color_map[label_name],
            description=f"Most frequent {ngram_name} for label {label_name}.",
        )

    def plot_common_hashtags(self, limit: int = 20) -> FigureArtifact:
        dataframe = self._top_counter_from_series(self.dataset["hashtags"], limit=limit)
        dataframe = dataframe.sort_values("count", ascending=True)
        return self._plot_horizontal_bar(
            dataframe=dataframe,
            title="Most Common Hashtags",
            name="common_hashtags",
            section="text_analysis",
            y_column="token",
            color="#1976D2",
            description="Top hashtags across the dataset.",
        )

    def plot_common_mentions(self, limit: int = 20) -> FigureArtifact:
        dataframe = self._top_counter_from_series(self.dataset["mentions"], limit=limit)
        dataframe = dataframe.sort_values("count", ascending=True)
        return self._plot_horizontal_bar(
            dataframe=dataframe,
            title="Most Common Mentions",
            name="common_mentions",
            section="text_analysis",
            y_column="token",
            color="#6A1B9A",
            description="Top mentions across the dataset.",
        )

    def plot_sentiment_trend_over_time(self) -> FigureArtifact:
        temporal = self.dataset.dropna(subset=["parsed_timestamp"]).copy()
        if temporal.empty:
            return self._register_skip(
                name="sentiment_trend_over_time",
                section="temporal_analysis",
                reason="No timestamp column was found for temporal analysis.",
            )
        daily = (
            temporal.groupby(
                [temporal["parsed_timestamp"].dt.floor("D"), "label_name"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "count"})
        )
        plotly_figure = px.line(
            daily,
            x="parsed_timestamp",
            y="count",
            color="label_name",
            color_discrete_map=self.color_map,
            title="Sentiment Trend Over Time",
            template=self.plotly_template,
        )
        plotly_figure.update_layout(xaxis_title="Date", yaxis_title="Tweet Count")

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(12, 6))
            sns.lineplot(
                data=daily,
                x="parsed_timestamp",
                y="count",
                hue="label_name",
                palette=self.color_map,
                ax=axis,
            )
            axis.set_title("Sentiment Trend Over Time")
            axis.set_xlabel("Date")
            axis.set_ylabel("Tweet Count")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="sentiment_trend_over_time",
            title="Sentiment Trend Over Time",
            section="temporal_analysis",
            description="Daily sentiment counts when tweet timestamps are available.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_tweet_volume_over_time(self) -> FigureArtifact:
        temporal = self.dataset.dropna(subset=["parsed_timestamp"]).copy()
        if temporal.empty:
            return self._register_skip(
                name="tweet_volume_over_time",
                section="temporal_analysis",
                reason="No timestamp column was found for temporal analysis.",
            )
        daily = (
            temporal.groupby(temporal["parsed_timestamp"].dt.floor("D"))
            .size()
            .reset_index(name="count")
        )
        plotly_figure = px.line(
            daily,
            x="parsed_timestamp",
            y="count",
            title="Tweet Volume Over Time",
            template=self.plotly_template,
        )
        plotly_figure.update_traces(line_color="#1565C0")
        plotly_figure.update_layout(xaxis_title="Date", yaxis_title="Tweet Count")

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(12, 6))
            sns.lineplot(
                data=daily,
                x="parsed_timestamp",
                y="count",
                color="#1565C0",
                ax=axis,
            )
            axis.set_title("Tweet Volume Over Time")
            axis.set_xlabel("Date")
            axis.set_ylabel("Tweet Count")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="tweet_volume_over_time",
            title="Tweet Volume Over Time",
            section="temporal_analysis",
            description="Daily tweet volume when tweet timestamps are available.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_sentiment_by_hour_heatmap(self) -> FigureArtifact:
        temporal = self.dataset.dropna(subset=["hour_of_day"]).copy()
        if temporal.empty:
            return self._register_skip(
                name="sentiment_by_hour_heatmap",
                section="temporal_analysis",
                reason="No timestamp column was found for hourly analysis.",
            )
        heatmap = (
            temporal.groupby(["hour_of_day", "label_name"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[self.label_display_map[label] for label in self.labels], fill_value=0)
        )
        plotly_figure = px.imshow(
            heatmap,
            labels={"x": "Sentiment", "y": "Hour of Day", "color": "Tweet Count"},
            title="Sentiment Distribution by Hour of Day",
            template=self.plotly_template,
            aspect="auto",
            color_continuous_scale="YlOrRd",
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 7))
            sns.heatmap(heatmap, cmap="YlOrRd", ax=axis)
            axis.set_title("Sentiment Distribution by Hour of Day")
            axis.set_xlabel("Sentiment")
            axis.set_ylabel("Hour of Day")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="sentiment_by_hour_heatmap",
            title="Sentiment Distribution by Hour of Day",
            section="temporal_analysis",
            description="Heatmap of hourly tweet counts by sentiment label.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_embedding_projection(self) -> FigureArtifact:
        subset = self.dataset.sample(
            n=min(len(self.dataset), self.max_projection_samples),
            random_state=self.config.random_seed,
        ).copy()
        if len(subset) < 3:
            return self._register_skip(
                name="tweet_embedding_projection",
                section="advanced_analysis",
                reason="At least three rows are required for a 2D embedding projection.",
            )

        matrix = TfidfVectorizer(max_features=2500, stop_words="english").fit_transform(
            subset["text"]
        )
        n_components = max(2, min(50, matrix.shape[1] - 1, matrix.shape[0] - 1))
        if n_components >= 2 and matrix.shape[1] > n_components:
            reduced = TruncatedSVD(
                n_components=n_components,
                random_state=self.config.random_seed,
            ).fit_transform(matrix)
        else:
            reduced = matrix.toarray()

        method_name = "UMAP" if umap is not None else "t-SNE"
        if umap is not None:
            reducer = umap.UMAP(n_components=2, random_state=self.config.random_seed)
            coordinates = reducer.fit_transform(reduced)
        else:
            perplexity = min(30, max(5, len(subset) // 20))
            reducer = TSNE(
                n_components=2,
                random_state=self.config.random_seed,
                init="random",
                learning_rate="auto",
                perplexity=perplexity,
            )
            coordinates = reducer.fit_transform(reduced)

        projection = subset[["label_name"]].copy()
        projection["x"] = coordinates[:, 0]
        projection["y"] = coordinates[:, 1]

        plotly_figure = px.scatter(
            projection,
            x="x",
            y="y",
            color="label_name",
            color_discrete_map=self.color_map,
            title=f"{method_name} Projection of Tweet Embeddings",
            template=self.plotly_template,
            opacity=0.8,
        )
        plotly_figure.update_layout(xaxis_title="Component 1", yaxis_title="Component 2")

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                data=projection,
                x="x",
                y="y",
                hue="label_name",
                palette=self.color_map,
                s=40,
                alpha=0.75,
                ax=axis,
            )
            axis.set_title(f"{method_name} Projection of Tweet Embeddings")
            axis.set_xlabel("Component 1")
            axis.set_ylabel("Component 2")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="tweet_embedding_projection",
            title=f"{method_name} Projection of Tweet Embeddings",
            section="advanced_analysis",
            description="Two-dimensional embedding projection of tweets colored by sentiment.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
            notes="UMAP used." if umap is not None else "UMAP not installed; used t-SNE instead.",
        )

    def plot_correlation_heatmap(self) -> FigureArtifact:
        numeric_columns = [
            "char_length",
            "word_count",
            "hashtag_count",
            "mention_count",
            "url_count",
            "uppercase_ratio",
            "exclamation_count",
            "question_count",
        ]
        if {"vader_neg", "vader_neu", "vader_pos", "vader_compound"}.issubset(
            self.dataset.columns
        ):
            numeric_columns.extend(
                ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
            )
        correlation = self.dataset[numeric_columns].corr()
        plotly_figure = px.imshow(
            correlation.round(3),
            text_auto=True,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap of Numerical Features",
            template=self.plotly_template,
            aspect="auto",
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                correlation,
                annot=True,
                cmap="RdBu_r",
                center=0,
                fmt=".2f",
                ax=axis,
            )
            axis.set_title("Correlation Heatmap of Numerical Features")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="correlation_heatmap",
            title="Correlation Heatmap of Numerical Features",
            section="advanced_analysis",
            description="Correlations between engineered numeric tweet features.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_vader_boxplot(self) -> FigureArtifact:
        if "vader_compound" not in self.dataset.columns:
            return self._register_skip(
                name="vader_boxplot",
                section="advanced_analysis",
                reason="VADER scores are unavailable because vaderSentiment is not installed.",
            )
        plotly_figure = px.box(
            self.dataset,
            x="label_name",
            y="vader_compound",
            color="label_name",
            color_discrete_map=self.color_map,
            title="VADER Compound Score by Actual Sentiment",
            template=self.plotly_template,
        )
        plotly_figure.update_layout(
            xaxis_title="Actual Sentiment",
            yaxis_title="VADER Compound Score",
            showlegend=False,
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                data=self.dataset,
                x="label_name",
                y="vader_compound",
                hue="label_name",
                dodge=False,
                palette=self.color_map,
                ax=axis,
            )
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
            axis.set_title("VADER Compound Score by Actual Sentiment")
            axis.set_xlabel("Actual Sentiment")
            axis.set_ylabel("VADER Compound Score")
            axis.tick_params(axis="x", rotation=45)
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="vader_boxplot",
            title="VADER Compound Score by Actual Sentiment",
            section="advanced_analysis",
            description="Distribution of VADER compound scores across labels.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_prediction_confidence_distribution(
        self,
        models: Optional[list[str]] = None,
    ) -> FigureArtifact:
        bundle = self._model_prediction_bundle(models=models)
        frames = [
            pd.DataFrame({"model": model_name, "confidence": payload["confidences"]})
            for model_name, payload in bundle["per_model"].items()
        ]
        dataframe = pd.concat(frames, ignore_index=True)
        plotly_figure = px.histogram(
            dataframe,
            x="confidence",
            color="model",
            barmode="overlay",
            opacity=0.65,
            nbins=30,
            title="Distribution of Prediction Confidence Scores",
            template=self.plotly_template,
        )
        plotly_figure.update_layout(xaxis_title="Confidence", yaxis_title="Count")

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 6))
            sns.histplot(
                data=dataframe,
                x="confidence",
                hue="model",
                bins=30,
                multiple="layer",
                alpha=0.4,
                ax=axis,
            )
            axis.set_title("Distribution of Prediction Confidence Scores")
            axis.set_xlabel("Confidence")
            axis.set_ylabel("Count")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="prediction_confidence_distribution",
            title="Distribution of Prediction Confidence Scores",
            section="advanced_analysis",
            description="Confidence score distribution across available trained models.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
            notes=f"Computed on {bundle['sample_size']} evaluation rows from {bundle['evaluation_source']}.",
        )

    def plot_model_confusion_matrix(self, model_name: str) -> FigureArtifact:
        bundle = self._model_prediction_bundle(models=[model_name])
        payload = bundle["per_model"][model_name]
        heatmap_frame = pd.DataFrame(
            payload["confusion_matrix"],
            index=bundle["label_names"],
            columns=bundle["label_names"],
        )
        plotly_figure = px.imshow(
            heatmap_frame,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"{model_name.upper()} Confusion Matrix",
            template=self.plotly_template,
            aspect="auto",
        )
        plotly_figure.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_frame, annot=True, fmt="d", cmap="Blues", ax=axis)
            axis.set_title(f"{model_name.upper()} Confusion Matrix")
            axis.set_xlabel("Predicted Label")
            axis.set_ylabel("True Label")
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name=f"{model_name}_confusion_matrix_heatmap",
            title=f"{model_name.upper()} Confusion Matrix",
            section="model_performance",
            description=f"Confusion matrix heatmap for the {model_name} model.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
            notes=f"Computed on {bundle['sample_size']} evaluation rows.",
        )

    def plot_roc_curves_all_models(
        self,
        models: Optional[list[str]] = None,
    ) -> FigureArtifact:
        bundle = self._model_prediction_bundle(models=models)
        label_indices = bundle["label_indices"]

        # Pre-compute ROC data once so both renderers stay independent.
        roc_data: list[dict[str, Any]] = []
        for model_name, payload in bundle["per_model"].items():
            y_true_bin = label_binarize(payload["y_true"], classes=label_indices)
            fpr, tpr, _ = roc_curve(
                y_true_bin.ravel(),
                payload["probabilities"].ravel(),
            )
            curve_auc = roc_auc_score(
                y_true_bin,
                payload["probabilities"],
                average="macro",
                multi_class="ovr",
            )
            roc_data.append(
                {"model": model_name, "fpr": fpr, "tpr": tpr, "auc": curve_auc}
            )

        plotly_figure = go.Figure()
        for entry in roc_data:
            plotly_figure.add_trace(
                go.Scatter(
                    x=entry["fpr"],
                    y=entry["tpr"],
                    mode="lines",
                    name=f"{entry['model']} (AUC={entry['auc']:.3f})",
                )
            )
        plotly_figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash", "color": "gray"},
                name="Random",
            )
        )
        plotly_figure.update_layout(
            title="ROC Curves Across Models (Micro-Averaged)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template=self.plotly_template,
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 7))
            for entry in roc_data:
                axis.plot(
                    entry["fpr"],
                    entry["tpr"],
                    label=f"{entry['model']} (AUC={entry['auc']:.3f})",
                )
            axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            axis.set_title("ROC Curves Across Models (Micro-Averaged)")
            axis.set_xlabel("False Positive Rate")
            axis.set_ylabel("True Positive Rate")
            axis.legend()
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="all_models_roc_curves",
            title="ROC Curves Across Models",
            section="model_performance",
            description="Micro-averaged one-vs-rest ROC curves for all available models.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
            notes=f"Computed on {bundle['sample_size']} evaluation rows.",
        )

    def _load_history(self, model_name: str) -> Optional[dict[str, Any]]:
        return self._load_json_if_exists(self.reports_dir / f"{model_name}_history.json")

    def plot_training_curves(self) -> FigureArtifact:
        history_models = ["lstm", "bilstm", "cnn", "distilbert"]
        histories: dict[str, dict[str, Any]] = {}
        for model_name in history_models:
            if (h := self._load_history(model_name)) is not None:
                histories[model_name] = h
        if not histories:
            return self._register_skip(
                name="training_curves",
                section="model_performance",
                reason="No saved training history JSON files were found.",
            )

        plotly_figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Training / Validation Loss", "Training / Validation Accuracy"),
        )

        def builder(png_path: Path) -> None:
            figure, axes = plt.subplots(1, 2, figsize=(14, 5))
            for model_name, history in histories.items():
                loss_values = history.get("loss", history.get("train_loss", []))
                val_loss_values = history.get("val_loss", [])
                acc_values = history.get("accuracy", history.get("train_accuracy", []))
                val_acc_values = history.get("val_accuracy", [])
                max_len = max(
                    len(loss_values),
                    len(val_loss_values),
                    len(acc_values),
                    len(val_acc_values),
                )
                epochs = list(range(1, max_len + 1))

                if loss_values:
                    axes[0].plot(epochs[: len(loss_values)], loss_values, label=f"{model_name} train")
                    plotly_figure.add_trace(
                        go.Scatter(x=epochs[: len(loss_values)], y=loss_values, mode="lines", name=f"{model_name} train loss"),
                        row=1,
                        col=1,
                    )
                if val_loss_values:
                    axes[0].plot(epochs[: len(val_loss_values)], val_loss_values, linestyle="--", label=f"{model_name} val")
                    plotly_figure.add_trace(
                        go.Scatter(
                            x=epochs[: len(val_loss_values)],
                            y=val_loss_values,
                            mode="lines",
                            name=f"{model_name} val loss",
                            line={"dash": "dash"},
                        ),
                        row=1,
                        col=1,
                    )
                if acc_values:
                    axes[1].plot(epochs[: len(acc_values)], acc_values, label=f"{model_name} train")
                    plotly_figure.add_trace(
                        go.Scatter(x=epochs[: len(acc_values)], y=acc_values, mode="lines", name=f"{model_name} train acc"),
                        row=1,
                        col=2,
                    )
                if val_acc_values:
                    axes[1].plot(epochs[: len(val_acc_values)], val_acc_values, linestyle="--", label=f"{model_name} val")
                    plotly_figure.add_trace(
                        go.Scatter(
                            x=epochs[: len(val_acc_values)],
                            y=val_acc_values,
                            mode="lines",
                            name=f"{model_name} val acc",
                            line={"dash": "dash"},
                        ),
                        row=1,
                        col=2,
                    )

            axes[0].set_title("Training / Validation Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[1].set_title("Training / Validation Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].legend()
            self._save_matplotlib(figure, png_path)

        plotly_figure.update_layout(
            title="Training Curves for Deep Learning Models",
            template=self.plotly_template,
        )
        plotly_figure.update_xaxes(title_text="Epoch", row=1, col=1)
        plotly_figure.update_xaxes(title_text="Epoch", row=1, col=2)
        plotly_figure.update_yaxes(title_text="Loss", row=1, col=1)
        plotly_figure.update_yaxes(title_text="Accuracy", row=1, col=2)

        return self._save_dual_figure(
            name="deep_learning_training_curves",
            title="Training Curves for Deep Learning Models",
            section="model_performance",
            description="Loss and accuracy trends across saved deep-learning model histories.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_model_comparison_radar(
        self,
        models: Optional[list[str]] = None,
    ) -> FigureArtifact:
        bundle = self._model_prediction_bundle(models=models)
        comparison = bundle["comparison_df"].copy()
        if comparison.empty:
            return self._register_skip(
                name="model_comparison_radar",
                section="model_performance",
                reason="No model comparison data was available.",
            )

        comparison["speed_score"] = 1.0 / comparison["avg_prediction_time_ms"].clip(lower=1e-9)
        metric_columns = [
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "speed_score",
        ]
        normalized = comparison[metric_columns].copy()
        for column in metric_columns:
            column_min = normalized[column].min()
            column_max = normalized[column].max()
            if math.isclose(float(column_min), float(column_max)):
                normalized[column] = 1.0
            else:
                normalized[column] = (normalized[column] - column_min) / (column_max - column_min)
        normalized["model"] = comparison["model"].values

        # Pre-compute radar data so both renderers stay independent.
        radar_labels = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "speed"]
        radar_series: list[dict[str, Any]] = []
        for _, row in normalized.iterrows():
            values = [
                row["accuracy"],
                row["f1_macro"],
                row["precision_macro"],
                row["recall_macro"],
                row["speed_score"],
            ]
            values += values[:1]
            radar_series.append({"model": row["model"], "values": values})

        plotly_figure = go.Figure()
        for entry in radar_series:
            plotly_figure.add_trace(
                go.Scatterpolar(
                    r=entry["values"],
                    theta=radar_labels + radar_labels[:1],
                    fill="toself",
                    name=entry["model"],
                )
            )
        plotly_figure.update_layout(
            title="Model Comparison Radar Chart",
            template=self.plotly_template,
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        )

        def builder(png_path: Path) -> None:
            angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
            angles += angles[:1]
            figure, axis = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
            for entry in radar_series:
                axis.plot(angles, entry["values"], label=entry["model"])
                axis.fill(angles, entry["values"], alpha=0.08)
            axis.set_xticks(angles[:-1])
            axis.set_xticklabels(radar_labels)
            axis.set_title("Model Comparison Radar Chart")
            axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="model_comparison_radar",
            title="Model Comparison Radar Chart",
            section="model_performance",
            description="Normalized comparison across accuracy, F1, precision, recall, and speed.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def plot_model_f1_comparison(
        self,
        models: Optional[list[str]] = None,
    ) -> FigureArtifact:
        comparison = self._model_prediction_bundle(models=models)["comparison_df"].copy()
        plotly_figure = px.bar(
            comparison,
            x="model",
            y="f1_macro",
            color="model",
            title="Macro F1 Score Comparison Across Models",
            template=self.plotly_template,
        )
        plotly_figure.update_layout(
            xaxis_title="Model",
            yaxis_title="Macro F1",
            showlegend=False,
        )

        def builder(png_path: Path) -> None:
            figure, axis = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=comparison,
                x="model",
                y="f1_macro",
                hue="model",
                dodge=False,
                ax=axis,
            )
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
            axis.set_title("Macro F1 Score Comparison Across Models")
            axis.set_xlabel("Model")
            axis.set_ylabel("Macro F1")
            axis.set_ylim(0, 1)
            self._save_matplotlib(figure, png_path)

        return self._save_dual_figure(
            name="model_f1_comparison",
            title="Macro F1 Score Comparison Across Models",
            section="model_performance",
            description="Bar chart comparing macro F1 scores across trained models.",
            plotly_figure=plotly_figure,
            matplotlib_builder=builder,
        )

    def generate_dataset_overview(self) -> list[FigureArtifact]:
        section = "dataset_overview"
        return [
            self._safe_call(self.plot_sentiment_distribution_bar, "sentiment_bar", section),
            self._safe_call(self.plot_sentiment_distribution_donut, "sentiment_donut", section),
            self._safe_call(self.plot_tweet_length_distribution, "tweet_length", section),
            self._safe_call(self.plot_word_count_distribution, "word_count", section),
            self._safe_call(self.plot_dataset_statistics_table, "statistics_table", section),
        ]

    def generate_text_analysis(self) -> list[FigureArtifact]:
        section = "text_analysis"
        artifacts: list[FigureArtifact] = []
        for label in self.labels:
            artifacts.append(self._safe_call(self.plot_wordcloud_for_label, f"wordcloud_{label}", section, label=label))
            artifacts.append(self._safe_call(self.plot_top_words_for_label, f"top_words_{label}", section, label=label))
            artifacts.append(self._safe_call(self.plot_top_ngrams_for_label, f"bigrams_{label}", section, label=label, ngram_value=2))
            artifacts.append(self._safe_call(self.plot_top_ngrams_for_label, f"trigrams_{label}", section, label=label, ngram_value=3))
        artifacts.append(self._safe_call(self.plot_common_hashtags, "common_hashtags", section))
        artifacts.append(self._safe_call(self.plot_common_mentions, "common_mentions", section))
        return artifacts

    def generate_temporal_analysis(self) -> list[FigureArtifact]:
        section = "temporal_analysis"
        return [
            self._safe_call(self.plot_sentiment_trend_over_time, "sentiment_trend", section),
            self._safe_call(self.plot_tweet_volume_over_time, "tweet_volume", section),
            self._safe_call(self.plot_sentiment_by_hour_heatmap, "hour_heatmap", section),
        ]

    def generate_advanced_analysis(
        self,
        models: Optional[list[str]] = None,
    ) -> list[FigureArtifact]:
        section = "advanced_analysis"
        return [
            self._safe_call(self.plot_embedding_projection, "embedding_projection", section),
            self._safe_call(self.plot_correlation_heatmap, "correlation_heatmap", section),
            self._safe_call(self.plot_vader_boxplot, "vader_boxplot", section),
            self._safe_call(self.plot_prediction_confidence_distribution, "confidence_dist", section, models=models),
        ]

    def generate_model_performance_visuals(
        self,
        models: Optional[list[str]] = None,
    ) -> list[FigureArtifact]:
        section = "model_performance"
        artifacts: list[FigureArtifact] = []
        for model_name in self._resolve_models(models):
            artifacts.append(self._safe_call(self.plot_model_confusion_matrix, f"{model_name}_cm", section, model_name=model_name))
        artifacts.append(self._safe_call(self.plot_roc_curves_all_models, "roc_curves", section, models=models))
        artifacts.append(self._safe_call(self.plot_training_curves, "training_curves", section))
        artifacts.append(self._safe_call(self.plot_model_comparison_radar, "comparison_radar", section, models=models))
        artifacts.append(self._safe_call(self.plot_model_f1_comparison, "f1_comparison", section, models=models))
        return artifacts

    def generate_all_visuals(
        self,
        models: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        logger.info("Generating all visualizations.")
        self.generate_dataset_overview()
        self.generate_text_analysis()
        self.generate_temporal_analysis()
        self.generate_advanced_analysis(models=models)
        self.generate_model_performance_visuals(models=models)
        return self.build_summary()

    def build_summary(self) -> dict[str, Any]:
        performance_rows: list[dict[str, Any]] = []
        if self._performance_cache is not None:
            bundle = self._performance_cache
            performance_rows = bundle["comparison_df"].to_dict(orient="records")
            evaluation_source = bundle["evaluation_source"]
            evaluation_sample_size = bundle["sample_size"]
        elif self.model_report is not None and self.model_report.get("comparison"):
            performance_rows = list(self.model_report["comparison"])
            evaluation_source = self.model_report.get("evaluation_split", self.evaluation_source)
            evaluation_sample_size = 0
        else:
            evaluation_source = self.evaluation_source
            evaluation_sample_size = 0

        return {
            "generated_at_unix": time.time(),
            "dataset": {
                "rows": len(self.dataset),
                "labels": self.labels,
                "label_names": [self.label_display_map[label] for label in self.labels],
                "splits": sorted(self.dataset["split"].unique().tolist()),
                "has_timestamps": bool(self.dataset["parsed_timestamp"].notna().any()),
                "summary_table": self._dataset_statistics_table().to_dict(orient="records"),
            },
            "evaluation": {
                "source": evaluation_source,
                "sample_size": evaluation_sample_size,
            },
            "available_models": self.available_models(),
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "skipped_items": to_serializable(self.skipped_items),
            "performance_summary": to_serializable(performance_rows),
            "training_summary": to_serializable(self.training_summary),
            "existing_model_report": to_serializable(self.model_report),
        }

    def save_summary_json(self, output_path: Path) -> Path:
        summary = self.build_summary()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=to_serializable)
        logger.info("Visualization summary saved to %s", output_path)
        return output_path

    def build_full_html_report(
        self,
        output_path: Path,
        title: str = "Twitter Sentiment Analysis Visual Report",
    ) -> Path:
        summary = self.build_summary()
        artifacts_by_section: dict[str, list[dict[str, Any]]] = {}
        for artifact in summary["artifacts"]:
            artifacts_by_section.setdefault(artifact["section"], []).append(artifact)

        performance_df = pd.DataFrame(summary["performance_summary"])
        dataset_df = pd.DataFrame(summary["dataset"]["summary_table"])
        sections: list[str] = [
            "<html><head><meta charset='utf-8'>",
            f"<title>{html.escape(title)}</title>",
            """
            <style>
              body { font-family: Arial, sans-serif; margin: 24px; line-height: 1.55; }
              h1, h2, h3 { color: #263238; }
              .meta { background: #fafafa; border: 1px solid #e0e0e0; padding: 16px; margin-bottom: 24px; }
              .artifact { border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 20px 0; background: white; }
              .artifact img { max-width: 100%; border: 1px solid #eee; margin-top: 12px; }
              .artifact iframe { width: 100%; height: 520px; border: 1px solid #eee; margin-top: 12px; }
              table { border-collapse: collapse; width: 100%; margin: 12px 0; }
              th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
              th { background: #f5f5f5; }
              .skipped { color: #B71C1C; }
            </style>
            </head><body>
            """,
            f"<h1>{html.escape(title)}</h1>",
            "<div class='meta'>",
            f"<p><strong>Dataset rows:</strong> {summary['dataset']['rows']}</p>",
            f"<p><strong>Labels:</strong> {', '.join(map(str, summary['dataset']['label_names']))}</p>",
            f"<p><strong>Loaded splits:</strong> {', '.join(summary['dataset']['splits'])}</p>",
            f"<p><strong>Evaluation source:</strong> {html.escape(str(summary['evaluation']['source']))}</p>",
            f"<p><strong>Evaluation sample size:</strong> {summary['evaluation']['sample_size']}</p>",
            "</div>",
            "<h2>Dataset Summary</h2>",
            dataset_df.to_html(index=False, escape=True),
        ]

        if not performance_df.empty:
            sections.extend(
                [
                    "<h2>Model Performance Summary</h2>",
                    performance_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
                ]
            )

        if summary["skipped_items"]:
            skipped_df = pd.DataFrame(summary["skipped_items"])
            sections.extend(
                [
                    "<h2>Skipped Items</h2>",
                    f"<div class='skipped'>{skipped_df.to_html(index=False, escape=True)}</div>",
                ]
            )

        for section_name, artifacts in artifacts_by_section.items():
            sections.append(f"<h2>{html.escape(section_name.replace('_', ' ').title())}</h2>")
            for artifact in artifacts:
                sections.append("<div class='artifact'>")
                sections.append(f"<h3>{html.escape(artifact['title'])}</h3>")
                sections.append(f"<p>{html.escape(artifact['description'])}</p>")
                if artifact.get("notes"):
                    sections.append(f"<p><em>{html.escape(str(artifact['notes']))}</em></p>")
                if artifact["status"] == "skipped":
                    sections.append("<p class='skipped'>Skipped.</p>")
                else:
                    if artifact.get("html_path"):
                        sections.append(
                            f"<iframe src='{html.escape(artifact['html_path'])}' loading='lazy'></iframe>"
                        )
                    if artifact.get("png_path"):
                        sections.append(
                            f"<img src='{html.escape(artifact['png_path'])}' alt='{html.escape(artifact['title'])}'>"
                        )
                sections.append("</div>")

        sections.append("</body></html>")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(sections), encoding="utf-8")
        logger.info("Full HTML report saved to %s", output_path)
        return output_path


def parse_args() -> argparse.Namespace:
    _default_config = TrainingConfig()
    parser = argparse.ArgumentParser(description="Generate EDA and model visualizations.")
    parser.add_argument(
        "--section",
        choices=["dataset", "text", "temporal", "advanced", "performance", "all"],
        default="all",
        help="Which visualization family to run.",
    )
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--evaluation-path", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--projection-samples", type=int, default=1500)
    parser.add_argument("--performance-sample-size", type=int, default=300)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=_default_config.reports_dir / "visualization_summary.json",
    )
    parser.add_argument(
        "--report-html",
        type=Path,
        default=_default_config.reports_dir / "visualization_report.html",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualizer = Visualizer(
        dataset_path=args.dataset_path,
        evaluation_dataset_path=args.evaluation_path,
        max_projection_samples=args.projection_samples,
        performance_sample_size=args.performance_sample_size,
    )

    if args.section == "dataset":
        visualizer.generate_dataset_overview()
    elif args.section == "text":
        visualizer.generate_text_analysis()
    elif args.section == "temporal":
        visualizer.generate_temporal_analysis()
    elif args.section == "advanced":
        visualizer.generate_advanced_analysis(models=args.models)
    elif args.section == "performance":
        visualizer.generate_model_performance_visuals(models=args.models)
    else:
        visualizer.generate_all_visuals(models=args.models)

    visualizer.save_summary_json(args.summary_json)
    visualizer.build_full_html_report(args.report_html)


if __name__ == "__main__":
    main()
