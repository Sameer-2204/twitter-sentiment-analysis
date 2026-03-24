"""
Run the full visualization pipeline and compile one combined HTML report.

Usage
-----
    python scripts/generate_full_report.py
    python scripts/generate_full_report.py --models logistic cnn distilbert
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.training_config import TrainingConfig  # noqa: E402
from scripts.visualization import Visualizer  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the full report runner."""
    config = TrainingConfig()
    parser = argparse.ArgumentParser(
        description="Generate all visualizations and compile a single HTML report.",
    )
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--evaluation-path", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--projection-samples", type=int, default=1500)
    parser.add_argument("--performance-sample-size", type=int, default=300)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=config.reports_dir / "full_report_summary.json",
        help="Where to save the summary statistics JSON.",
    )
    parser.add_argument(
        "--report-html",
        type=Path,
        default=config.reports_dir / "full_visual_report.html",
        help="Where to save the combined HTML report.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Twitter Sentiment Analysis Full Report",
        help="HTML title for the generated report.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full visualization pipeline end to end."""
    args = parse_args()
    visualizer = Visualizer(
        dataset_path=args.dataset_path,
        evaluation_dataset_path=args.evaluation_path,
        max_projection_samples=args.projection_samples,
        performance_sample_size=args.performance_sample_size,
    )
    visualizer.generate_all_visuals(models=args.models)
    visualizer.save_summary_json(args.summary_json)
    visualizer.build_full_html_report(
        output_path=args.report_html,
        title=args.title,
    )
    logger.info("Full report generation complete.")


if __name__ == "__main__":
    main()
