#!/usr/bin/env python3
"""Aggregate per-model metrics.json files into one flat CSV.

Scans a results root directory (default: benchmark/results/metrics/gt_conditioned) for
``<model>/metrics.json`` files written by ``evaluate.py`` and combines their
per-task summary rows into ``<root>/all_metrics/all_models_metrics.csv``
(one row per model x task), including precision / recall / F1 at micro,
macro, and weighted averaging.

Usage:
    python -m experiments.evaluation.aggregate_all_metrics [--results-root PATH] [--output PATH]
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from paths import GT_CONDITIONED_METRICS_DIR

logger = logging.getLogger(__name__)

# Column order in the output CSV; keys are metrics.json "Task" row fields.
COLUMNS = [
    ("Exact Match Ratio", "exact_match_ratio"),
    ("Mean Jaccard", "mean_jaccard"),
    ("Micro Precision", "micro_precision"),
    ("Micro Recall", "micro_recall"),
    ("Micro F1", "micro_f1"),
    ("Macro Precision", "macro_precision"),
    ("Macro Recall", "macro_recall"),
    ("Macro F1", "macro_f1"),
    ("Weighted Precision", "weighted_precision"),
    ("Weighted Recall", "weighted_recall"),
    ("Weighted F1", "weighted_f1"),
    # Number of inventory labels averaged over in the macro scores (labels
    # with GT support or >=1 prediction; see evaluate.score_label_sets).
    ("Macro Classes", "macro_classes"),
    # Per-identity narrative role alignment metrics (only populated on the
    # "Narrative Roles (per-identity alignment)" task rows).
    ("GT Identities", "gt_identities"),
    ("Role Coverage", "role_coverage"),
    ("Role Hit", "role_hit"),
    ("Exact Role", "exact_role"),
    ("Per-Identity Jaccard", "per_identity_jaccard"),
    # Per-identity viewpoint alignment metrics (only populated on the
    # "Viewpoint (per-identity alignment)" task rows).
    ("Viewpoint Coverage", "viewpoint_coverage"),
    ("Viewpoint Hit", "viewpoint_hit"),
    ("Exact Viewpoint", "exact_viewpoint"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-model metrics.json files into all_models_metrics.csv.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(GT_CONDITIONED_METRICS_DIR),
        help="Directory containing <model>/metrics.json subdirectories "
             "(default: benchmark/results/metrics/gt_conditioned).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <results-root>/all_metrics/all_models_metrics.csv).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    )

    root = Path(args.results_root)
    if not root.is_dir():
        logger.error("Results root not found: %s", root)
        sys.exit(1)

    out_path = (
        Path(args.output)
        if args.output
        else root / "all_metrics" / "all_models_metrics.csv"
    )

    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        model_dir = metrics_path.parent
        # Skip the aggregated-output dir itself if it ever contains a
        # stray metrics.json.
        if model_dir.name == "all_metrics":
            continue
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping unreadable %s: %s", metrics_path, e)
            continue

        # Always key by directory name -- it's what the results live under
        # and stays correct even if a metrics.json was written with a stale
        # MODEL_SHORT_NAME from the environment.
        for task_row in data.get("metrics_summary", []):
            row = {"model": model_dir.name, "task": task_row.get("Task", "")}
            for json_key, csv_col in COLUMNS:
                row[csv_col] = task_row.get(json_key, "")
            rows.append(row)
        logger.info("Loaded %d tasks from %s",
                    len(data.get("metrics_summary", [])), metrics_path)

    if not rows:
        logger.error("No metrics found under %s", root)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "task"] + [col for _, col in COLUMNS]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            fields = []
            for col in header:
                val = str(row.get(col, ""))
                if "," in val or '"' in val:
                    val = '"' + val.replace('"', '""') + '"'
                fields.append(val)
            f.write(",".join(fields) + "\n")

    logger.info("Wrote %d rows (%d models) to %s",
                len(rows), len({r["model"] for r in rows}), out_path)


if __name__ == "__main__":
    main()
