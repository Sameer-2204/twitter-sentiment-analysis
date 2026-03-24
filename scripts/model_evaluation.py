"""
Comprehensive model evaluation and comparison for Twitter sentiment models.

This script evaluates all trained models on the best available evaluation
split, generates metrics and plots, performs error analysis, compares model
speed/size, and exports both JSON and HTML reports.

Usage
-----
    python scripts/model_evaluation.py
    python scripts/model_evaluation.py --model logistic
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.predict import MODEL_REGISTRY, SentimentPredictor  # noqa: E402
from scripts.training_config import TrainingConfig  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

NEGATION_RE = re.compile(r"\b(?:no|not|never|nothing|nowhere|neither|nor|n't)\b", re.I)
SARCASM_RE = re.compile(
    r"(?:/s\b|yeah right|as if|sure jan|love that for me|totally|great job|nice one)",
    re.I,
)
EMPHASIS_RE = re.compile(r"(?:[!?]{2,}|[A-Z]{3,})")


def to_serializable(value: Any) -> Any:
    """Convert numpy/pandas objects into JSON-serializable Python objects."""
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


class ModelEvaluator:
    """Evaluate, compare, and report on trained sentiment models."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model_name: str = "all",
    ) -> None:
        """Initialise report directories, labels, and the shared predictor."""
        self.config = config or TrainingConfig()
        self.config.ensure_dirs()
        self.model_name = model_name
        self.figures_dir = self.config.reports_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = SentimentPredictor(config=self.config)
        self.label_indices = sorted(self.config.sentiment_labels_inv)
        self.label_names = [str(index) for index in self.label_indices]
        self.label_display_map = {
            label_index: str(label_index) for label_index in self.label_indices
        }

    def resolve_evaluation_dataset(self) -> tuple[pd.DataFrame, str]:
        """Load the best available evaluation split, preferring test data."""
        candidates = [
            ("test_csv", self.config.test_csv),
            ("processed_test", self.config.model_dir.parent / "data" / "processed" / "processed_test.csv"),
            ("valid_csv", self.config.valid_csv),
            ("train_csv", self.config.train_csv),
        ]

        for split_name, path in candidates:
            if path.exists():
                dataframe = pd.read_csv(path)
                if {"text", "label"}.issubset(dataframe.columns):
                    logger.info("Using evaluation split '%s' from %s", split_name, path)
                    dataframe = dataframe.dropna(subset=["text", "label"]).reset_index(drop=True)
                    dataframe["label"] = dataframe["label"].astype(int)
                    self.label_indices = sorted(dataframe["label"].unique().tolist())
                    configured_labels = sorted(self.config.sentiment_labels_inv)
                    if self.label_indices == configured_labels:
                        self.label_display_map = {
                            label: self.config.sentiment_labels_inv[label]
                            for label in self.label_indices
                        }
                    else:
                        self.label_display_map = {
                            label: str(label) for label in self.label_indices
                        }
                    self.label_names = [
                        self.label_display_map[label] for label in self.label_indices
                    ]
                    return dataframe, split_name

        raise FileNotFoundError(
            "No evaluation dataset found. Expected one of: "
            f"{[str(path) for _, path in candidates]}"
        )

    def build_prediction_frame(
        self,
        texts: list[str],
        y_true: np.ndarray,
        probabilities: np.ndarray,
        class_labels: list[int],
        model_name: str,
    ) -> pd.DataFrame:
        """Build a per-example evaluation frame for one model."""
        pred_positions = np.argmax(probabilities, axis=1)
        predictions = np.asarray([class_labels[index] for index in pred_positions], dtype=int)
        confidences = np.max(probabilities, axis=1)
        frame = pd.DataFrame(
            {
                "text": texts,
                "true_label_index": y_true,
                "pred_label_index": predictions,
                "true_label": [self.label_display_map[int(label)] for label in y_true],
                "pred_label": [self.label_display_map[int(label)] for label in predictions],
                "confidence": confidences,
            }
        )
        frame["is_correct"] = frame["true_label_index"] == frame["pred_label_index"]
        frame["token_length"] = frame["text"].astype(str).str.split().str.len()
        frame["has_negation"] = frame["text"].astype(str).str.contains(NEGATION_RE)
        frame["has_sarcasm_cue"] = frame["text"].astype(str).str.contains(SARCASM_RE)
        frame["is_short"] = frame["token_length"] <= 4
        frame["has_emphasis"] = frame["text"].astype(str).str.contains(EMPHASIS_RE)

        for class_index, label_value in enumerate(class_labels):
            label_name = self.label_display_map[int(label_value)]
            frame[f"prob_{label_name}"] = probabilities[:, class_index]

        logger.info(
            "Built prediction frame for model=%s with %d examples.",
            model_name,
            len(frame),
        )
        return frame

    def compute_metrics(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        class_labels: list[int],
    ) -> dict[str, Any]:
        """Compute the core evaluation metrics for one model."""
        pred_positions = np.argmax(probabilities, axis=1)
        predictions = np.asarray([class_labels[index] for index in pred_positions], dtype=int)
        metrics = {
            "accuracy": accuracy_score(y_true, predictions),
            "precision_macro": precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "precision_weighted": precision_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall_weighted": recall_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "cohen_kappa": cohen_kappa_score(y_true, predictions),
            "log_loss": log_loss(
                y_true,
                probabilities,
                labels=self.label_indices,
            ),
        }

        y_true_binarized = label_binarize(y_true, classes=self.label_indices)
        try:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y_true_binarized,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
            metrics["roc_auc_ovr_weighted"] = roc_auc_score(
                y_true_binarized,
                probabilities,
                average="weighted",
                multi_class="ovr",
            )
        except ValueError as exc:
            logger.warning("ROC-AUC could not be computed: %s", exc)
            metrics["roc_auc_ovr_macro"] = None
            metrics["roc_auc_ovr_weighted"] = None

        report = classification_report(
            y_true,
            predictions,
            labels=self.label_indices,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0,
        )
        metrics["classification_report"] = report
        metrics["confusion_matrix"] = confusion_matrix(
            y_true,
            predictions,
            labels=self.label_indices,
        )
        return metrics

    def plot_confusion_matrix(
        self,
        confusion: np.ndarray,
        model_name: str,
    ) -> str:
        """Save a confusion-matrix heatmap and return its relative path."""
        figure_path = self.figures_dir / f"{model_name}_confusion_matrix.png"
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_names,
            yticklabels=self.label_names,
        )
        plt.title(f"{model_name.upper()} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()
        return f"figures/{figure_path.name}"

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        model_name: str,
    ) -> str:
        """Save the one-vs-rest ROC curve for one model."""
        figure_path = self.figures_dir / f"{model_name}_roc_curve.png"
        y_true_binarized = label_binarize(y_true, classes=self.label_indices)

        plt.figure(figsize=(8, 6))
        for class_index, label_name in enumerate(self.label_names):
            if y_true_binarized[:, class_index].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_binarized[:, class_index], probabilities[:, class_index])
            auc_value = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label_name} (AUC={auc_value:.3f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.title(f"{model_name.upper()} ROC Curve (OvR)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()
        return f"figures/{figure_path.name}"

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        model_name: str,
    ) -> str:
        """Save the one-vs-rest precision-recall curve for one model."""
        figure_path = self.figures_dir / f"{model_name}_precision_recall_curve.png"
        y_true_binarized = label_binarize(y_true, classes=self.label_indices)

        plt.figure(figsize=(8, 6))
        for class_index, label_name in enumerate(self.label_names):
            if y_true_binarized[:, class_index].sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(
                y_true_binarized[:, class_index],
                probabilities[:, class_index],
            )
            curve_area = auc(recall, precision)
            plt.plot(recall, precision, label=f"{label_name} (AUC={curve_area:.3f})")

        plt.title(f"{model_name.upper()} Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()
        return f"figures/{figure_path.name}"

    def plot_length_distribution(self, frame: pd.DataFrame, model_name: str) -> str:
        """Save the token-length distribution for correct vs wrong predictions."""
        figure_path = self.figures_dir / f"{model_name}_length_distribution.png"
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=frame,
            x="token_length",
            hue="is_correct",
            bins=30,
            multiple="layer",
            stat="density",
            common_norm=False,
        )
        plt.title(f"{model_name.upper()} Tweet Length Distribution")
        plt.xlabel("Token Count")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()
        return f"figures/{figure_path.name}"

    def analyze_errors(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Produce error-analysis views and heuristic misclassification patterns."""
        correct = frame[frame["is_correct"]].sort_values("confidence", ascending=False)
        wrong = frame[~frame["is_correct"]].sort_values("confidence", ascending=False)

        def select_examples(dataframe: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
            columns = [
                "text",
                "true_label",
                "pred_label",
                "confidence",
                "token_length",
            ]
            subset = dataframe[columns].head(limit).copy()
            return to_serializable(subset.to_dict(orient="records"))

        per_class_examples: dict[str, list[dict[str, Any]]] = {}
        for label_name in self.label_names:
            subset = wrong[wrong["true_label"] == label_name].head(5).copy()
            per_class_examples[label_name] = to_serializable(
                subset[
                    ["text", "pred_label", "confidence", "token_length"]
                ].to_dict(orient="records")
            )

        patterns: dict[str, Any] = {}
        for pattern_name in ["has_negation", "has_sarcasm_cue", "is_short", "has_emphasis"]:
            total = int(frame[pattern_name].sum())
            total_errors = int((frame[pattern_name] & ~frame["is_correct"]).sum())
            error_rate = (total_errors / total) if total else None
            share_of_errors = (
                total_errors / max(int((~frame["is_correct"]).sum()), 1)
                if total
                else None
            )
            patterns[pattern_name] = {
                "examples_with_pattern": total,
                "misclassified_with_pattern": total_errors,
                "error_rate": error_rate,
                "share_of_all_errors": share_of_errors,
            }

        patterns["length_summary"] = {
            "avg_length_correct": float(correct["token_length"].mean()) if not correct.empty else None,
            "avg_length_wrong": float(wrong["token_length"].mean()) if not wrong.empty else None,
            "median_length_correct": float(correct["token_length"].median()) if not correct.empty else None,
            "median_length_wrong": float(wrong["token_length"].median()) if not wrong.empty else None,
        }

        return {
            "most_confident_correct": select_examples(correct),
            "most_confident_wrong": select_examples(wrong),
            "misclassified_examples_per_class": per_class_examples,
            "misclassification_patterns": patterns,
        }

    def evaluate_one_model(
        self,
        model_name: str,
        texts: list[str],
        y_true: np.ndarray,
        timing_sample_size: int = 200,
    ) -> dict[str, Any]:
        """Evaluate one model end to end."""
        logger.info("Evaluating model '%s'.", model_name)

        timing_texts = texts[: min(timing_sample_size, len(texts))]
        class_labels = self.predictor.get_class_labels(model_name)
        start = time.perf_counter()
        raw_probabilities = self.predictor.predict_proba_batch(texts, model_name=model_name)
        total_prediction_time = time.perf_counter() - start

        timing_start = time.perf_counter()
        self.predictor.predict_proba_batch(
            timing_texts,
            model_name=model_name,
        )
        timing_elapsed = time.perf_counter() - timing_start
        avg_prediction_time_ms = (timing_elapsed / max(len(timing_texts), 1)) * 1000

        probabilities = self.align_probabilities_to_dataset(
            probabilities=raw_probabilities,
            class_labels=class_labels,
        )

        frame = self.build_prediction_frame(
            texts=texts,
            y_true=y_true,
            probabilities=probabilities,
            class_labels=self.label_indices,
            model_name=model_name,
        )
        metrics = self.compute_metrics(
            y_true=y_true,
            probabilities=probabilities,
            class_labels=self.label_indices,
        )
        error_analysis = self.analyze_errors(frame)

        figures = {
            "confusion_matrix": self.plot_confusion_matrix(
                confusion=metrics["confusion_matrix"],
                model_name=model_name,
            ),
            "roc_curve": self.plot_roc_curve(
                y_true=y_true,
                probabilities=probabilities,
                model_name=model_name,
            ),
            "precision_recall_curve": self.plot_precision_recall_curve(
                y_true=y_true,
                probabilities=probabilities,
                model_name=model_name,
            ),
            "length_distribution": self.plot_length_distribution(
                frame=frame,
                model_name=model_name,
            ),
        }

        summary_row = {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_macro": metrics["recall_macro"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "roc_auc_ovr_macro": metrics["roc_auc_ovr_macro"],
            "roc_auc_ovr_weighted": metrics["roc_auc_ovr_weighted"],
            "cohen_kappa": metrics["cohen_kappa"],
            "log_loss": metrics["log_loss"],
            "avg_prediction_time_ms": avg_prediction_time_ms,
            "timing_sample_size": len(timing_texts),
            "total_prediction_time_seconds": total_prediction_time,
            "model_size_mb": self.predictor.get_model_size_mb(model_name),
        }

        return {
            "summary": summary_row,
            "metrics": metrics,
            "figures": figures,
            "error_analysis": error_analysis,
            "predictions": frame,
        }

    def align_probabilities_to_dataset(
        self,
        probabilities: np.ndarray,
        class_labels: list[int],
    ) -> np.ndarray:
        """Align a model's probability columns to the evaluation dataset labels."""
        aligned = np.zeros((probabilities.shape[0], len(self.label_indices)), dtype=np.float64)
        dataset_position = {
            int(label): position for position, label in enumerate(self.label_indices)
        }
        for column_index, class_label in enumerate(class_labels):
            if int(class_label) not in dataset_position:
                logger.warning(
                    "Skipping model class label %s because it is absent from the evaluation split.",
                    class_label,
                )
                continue
            aligned[:, dataset_position[int(class_label)]] = probabilities[:, column_index]
        return aligned

    def plot_comparison_bars(self, comparison_df: pd.DataFrame) -> str:
        """Save a bar chart comparing accuracy and macro F1 across models."""
        figure_path = self.figures_dir / "model_accuracy_f1_comparison.png"
        melted = comparison_df.melt(
            id_vars="model",
            value_vars=["accuracy", "f1_macro"],
            var_name="metric",
            value_name="score",
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(data=melted, x="model", y="score", hue="metric")
        plt.title("Model Accuracy and Macro F1 Comparison")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()
        return f"figures/{figure_path.name}"

    def run_mcnemar_test(
        self,
        comparison_df: pd.DataFrame,
        predictions: dict[str, pd.DataFrame],
    ) -> Optional[dict[str, Any]]:
        """Run McNemar's test between the two best models by macro F1."""
        if len(comparison_df) < 2:
            return None

        top_two = comparison_df.sort_values(
            by=["f1_macro", "accuracy"],
            ascending=False,
        ).head(2)
        model_a = top_two.iloc[0]["model"]
        model_b = top_two.iloc[1]["model"]

        frame_a = predictions[model_a]
        frame_b = predictions[model_b]

        correct_a = frame_a["is_correct"].to_numpy(dtype=bool)
        correct_b = frame_b["is_correct"].to_numpy(dtype=bool)

        both_correct = int(np.sum(correct_a & correct_b))
        only_a_correct = int(np.sum(correct_a & ~correct_b))
        only_b_correct = int(np.sum(~correct_a & correct_b))
        both_wrong = int(np.sum(~correct_a & ~correct_b))

        discordant = only_a_correct + only_b_correct
        if discordant == 0:
            statistic = 0.0
            p_value = 1.0
            test_type = "degenerate"
        elif discordant < 25:
            statistic = None
            p_value = float(
                binomtest(
                    min(only_a_correct, only_b_correct),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            test_type = "exact"
        else:
            statistic = ((abs(only_a_correct - only_b_correct) - 1) ** 2) / discordant
            p_value = float(chi2.sf(statistic, df=1))
            test_type = "chi_square"

        return {
            "model_a": model_a,
            "model_b": model_b,
            "contingency_table": {
                "both_correct": both_correct,
                "only_model_a_correct": only_a_correct,
                "only_model_b_correct": only_b_correct,
                "both_wrong": both_wrong,
            },
            "test_type": test_type,
            "statistic": statistic,
            "p_value": p_value,
            "significant_at_0_05": p_value < 0.05,
        }

    def build_html_report(
        self,
        dataset_source: str,
        comparison_df: pd.DataFrame,
        per_model_results: dict[str, dict[str, Any]],
        significance: Optional[dict[str, Any]],
        comparison_plot: str,
        output_path: Path,
    ) -> Path:
        """Render a self-contained HTML report referencing saved figures."""
        sections: list[str] = []
        sections.append(
            f"""
            <html>
            <head>
              <meta charset="utf-8">
              <title>Model Comparison Report</title>
              <style>
                body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; }}
                table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
                th {{ background: #f4f4f4; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 8px 0; }}
                pre {{ white-space: pre-wrap; background: #fafafa; padding: 12px; border: 1px solid #eee; }}
              </style>
            </head>
            <body>
              <h1>Twitter Sentiment Model Comparison Report</h1>
              <p><strong>Evaluation split:</strong> {html.escape(dataset_source)}</p>
              <h2>Model Comparison</h2>
              {comparison_df.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) and not math.isnan(x) else str(x))}
              <img src="{html.escape(comparison_plot)}" alt="Accuracy and F1 comparison">
            """
        )

        if significance is not None:
            significance_html = pd.DataFrame([significance]).to_html(index=False)
            sections.append(f"<h2>Statistical Significance</h2>{significance_html}")

        for model_name, result in per_model_results.items():
            metrics = result["metrics"]
            report_df = pd.DataFrame(metrics["classification_report"]).T
            error_analysis = result["error_analysis"]

            sections.append(f"<h2>{html.escape(model_name.upper())}</h2>")
            sections.append("<h3>Core Metrics</h3>")
            metrics_table = pd.DataFrame([result["summary"]]).to_html(
                index=False,
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) and not math.isnan(x) else str(x),
            )
            sections.append(metrics_table)
            sections.append("<h3>Classification Report</h3>")
            sections.append(
                report_df.to_html(
                    float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
                )
            )
            sections.append("<h3>Figures</h3>")
            for figure_name, relative_path in result["figures"].items():
                sections.append(
                    f"<p><strong>{html.escape(figure_name.replace('_', ' ').title())}</strong></p>"
                    f'<img src="{html.escape(relative_path)}" alt="{html.escape(figure_name)}">'
                )

            sections.append("<h3>Error Analysis</h3>")
            sections.append(
                "<p><strong>Most confident correct predictions</strong></p>"
                + pd.DataFrame(error_analysis["most_confident_correct"]).to_html(index=False)
            )
            sections.append(
                "<p><strong>Most confident wrong predictions</strong></p>"
                + pd.DataFrame(error_analysis["most_confident_wrong"]).to_html(index=False)
            )

            sections.append("<h4>Misclassified Examples Per Class</h4>")
            for label_name, examples in error_analysis["misclassified_examples_per_class"].items():
                sections.append(f"<p><strong>{html.escape(label_name)}</strong></p>")
                sections.append(pd.DataFrame(examples).to_html(index=False))

            sections.append("<h4>Misclassification Patterns</h4>")
            pattern_frame = pd.DataFrame(
                error_analysis["misclassification_patterns"]
            ).T.reset_index(names="pattern")
            sections.append(pattern_frame.to_html(index=False))

        sections.append("</body></html>")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(sections), encoding="utf-8")
        logger.info("HTML report saved to %s", output_path)
        return output_path

    def run(self) -> dict[str, Any]:
        """Execute the full evaluation pipeline and save all outputs."""
        evaluation_df, dataset_source = self.resolve_evaluation_dataset()
        texts = evaluation_df["text"].astype(str).tolist()
        y_true = evaluation_df["label"].astype(int).to_numpy()

        models_to_evaluate = (
            list(MODEL_REGISTRY.keys()) if self.model_name == "all" else [self.model_name]
        )

        per_model_results: dict[str, dict[str, Any]] = {}
        comparison_rows: list[dict[str, Any]] = []
        predictions: dict[str, pd.DataFrame] = {}
        failures: list[dict[str, Any]] = []

        for model_name in tqdm(models_to_evaluate, desc="Evaluating models"):
            try:
                result = self.evaluate_one_model(
                    model_name=model_name,
                    texts=texts,
                    y_true=y_true,
                )
                per_model_results[model_name] = result
                comparison_rows.append(result["summary"])
                predictions[model_name] = result["predictions"]
            except Exception as exc:
                logger.exception("Evaluation failed for model=%s", model_name)
                failures.append({"model": model_name, "error": str(exc)})

        if not comparison_rows:
            raise RuntimeError(
                "No model evaluations succeeded. Check artifact paths and dependencies."
            )

        comparison_df = pd.DataFrame(comparison_rows).sort_values(
            by=["f1_macro", "accuracy"],
            ascending=False,
        ).reset_index(drop=True)
        comparison_plot = self.plot_comparison_bars(comparison_df)
        significance = self.run_mcnemar_test(comparison_df, predictions)

        report_payload = {
            "evaluation_split": dataset_source,
            "comparison": comparison_df,
            "significance_test": significance,
            "failures": failures,
            "per_model": {
                model_name: {
                    "summary": result["summary"],
                    "metrics": result["metrics"],
                    "figures": result["figures"],
                    "error_analysis": result["error_analysis"],
                }
                for model_name, result in per_model_results.items()
            },
        }

        json_path = self.config.reports_dir / "model_comparison_report.json"
        json_path.write_text(
            json.dumps(to_serializable(report_payload), indent=2),
            encoding="utf-8",
        )
        logger.info("JSON report saved to %s", json_path)

        html_path = self.build_html_report(
            dataset_source=dataset_source,
            comparison_df=comparison_df,
            per_model_results=per_model_results,
            significance=significance,
            comparison_plot=comparison_plot,
            output_path=self.config.reports_dir / "model_comparison_report.html",
        )

        return {
            "json_report": json_path,
            "html_report": html_path,
            "comparison": comparison_df,
            "failures": failures,
        }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare all trained sentiment models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Model to evaluate, or 'all' for full comparison.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for the model evaluation pipeline."""
    args = parse_args()
    evaluator = ModelEvaluator(model_name=args.model)
    result = evaluator.run()
    comparison_df = result["comparison"]

    print("\nModel Comparison")
    print(comparison_df.to_string(index=False))
    print(f"\nJSON report: {result['json_report']}")
    print(f"HTML report: {result['html_report']}")
    if result["failures"]:
        print("\nFailures:")
        print(pd.DataFrame(result["failures"]).to_string(index=False))


if __name__ == "__main__":
    main()
