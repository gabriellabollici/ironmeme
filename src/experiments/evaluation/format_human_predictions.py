#!/usr/bin/env python3
"""Format human-evaluator annotations into the same shape as
``format_predictions.py``'s VLM output, so they can be scored by the
unmodified ``evaluate.py``.

Reads the per-annotator JSON files under
``src/experiments/evaluation/human_evaluators/`` (each a flat
list of one-identity-group-per-entry annotations, keyed by the *old*
reddit-style ``image_filename``), maps each meme onto its *clean* benchmark
filename via build_benchmark's filename mapping, and writes one
``formatted_predictions.csv`` per annotator plus a pooled one (all
annotators combined, one row per meme) -- mirroring the columns
``build_dataframe()`` writes for VLM predictions.

Usage:
    python -m experiments.evaluation.format_human_predictions [--output-root PATH]
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


import pandas as pd

from experiments.inference.config import BENCHMARK_JSON
from experiments.evaluation.format_predictions import (
    normalize_and_map_identity,
    extract_viewpoints_only,
    extract_roles_only,
    extract_identity_roles,
    extract_identity_viewpoints,
    normalise_gt_identities,
    normalise_gt_viewpoints,
    normalise_gt_roles,
    normalise_gt_identity_roles,
    normalise_gt_identity_viewpoints,
    _normalise_viewpoint_label,
)
from dataset_construction.build_benchmark import original_to_clean_filenames
from paths import NO_GT_METRICS_DIR

logger = logging.getLogger(__name__)

HUMAN_EVAL_DIR = Path(__file__).resolve().parent / "human_evaluators"


def load_old_to_clean_mapping() -> Dict[str, str]:
    """Annotated (old-style) image filename -> its filename in the clean benchmark.

    Computed by build_benchmark from the annotation ids, so it is exact even for
    near-duplicate memes (same OCR/origin) that a content-based join can't tell apart.
    """
    return original_to_clean_filenames()


def load_benchmark_lookup() -> Dict[str, Any]:
    with open(Path(BENCHMARK_JSON), "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    return {e["image_filename"]: e for e in benchmark_data if "image_filename" in e}


def group_by_meme(annotator_entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group an annotator's flat entry list by old-style image_filename.

    Each entry corresponds to one identified social identity group (possibly
    a comma-separated list of near-synonymous identities sharing one
    viewpoint/role label), so a meme with N identified groups has N entries.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in annotator_entries:
        groups[entry["image_filename"]].append(entry)
    return groups


def build_prediction_row(meme_entries: List[Dict[str, Any]], clean_filename: str, gt_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Build one formatted-predictions row for a single meme, treating the
    human annotations as "predictions" using the exact same normalisation
    functions applied to VLM output in format_predictions.build_dataframe().
    """
    viewpoints_raw: List[Dict[str, str]] = []
    roles_raw: List[Dict[str, Any]] = []
    all_identities: List[str] = []

    for entry in meme_entries:
        labels = entry.get("labels", {})
        identity_field = str(labels.get("social_identity", "") or "").strip()
        if not identity_field:
            continue
        identities = [i.strip() for i in identity_field.split(",") if i.strip()]
        viewpoint = str(labels.get("view_point", "") or "").strip()
        roles = labels.get("narrative_roles") or []

        for identity in identities:
            all_identities.append(identity)
            roles_raw.append({"identity": identity, "roles": roles})
            if viewpoint:
                # extract_viewpoints_only() (unlike extract_identity_viewpoints())
                # does not normalise its input -- it assumes VLM output already
                # says "ingroup"/"outgroup" per the prompt spec. Human labels use
                # "in_group"/"out_group", so normalise here or the flat Viewpoint
                # metric would treat every correct human label as a mismatch.
                viewpoints_raw.append({
                    "identity": identity,
                    "viewpoint": _normalise_viewpoint_label(viewpoint),
                })

    norm_social_ids: List[str] = []
    for item in all_identities:
        mapped = normalize_and_map_identity(item)
        if mapped and mapped not in norm_social_ids:
            norm_social_ids.append(mapped)

    gt_social = normalise_gt_identities(gt_entry.get("social_identity_gt", "[]"))
    gt_viewpoint = normalise_gt_viewpoints(
        gt_entry.get("viewpoint_gt", "[]"), gt_entry.get("image_filename"),
    )
    gt_roles = normalise_gt_roles(gt_entry.get("narrative_roles_gt", "[]"))
    gt_roles_by_identity = normalise_gt_identity_roles(gt_entry)
    gt_viewpoints_by_identity = normalise_gt_identity_viewpoints(gt_entry)

    return {
        "image_filename": Path(clean_filename).stem,
        "social_identity_gt": gt_social,
        "viewpoint_gt": gt_viewpoint,
        "narrative_roles_gt": gt_roles,
        "narrative_roles_gt_by_identity": gt_roles_by_identity,
        "viewpoint_gt_by_identity": gt_viewpoints_by_identity,
        "predicted_social_identities": norm_social_ids,
        "predicted_viewpoints": extract_viewpoints_only(viewpoints_raw),
        "predicted_narrative_roles": extract_roles_only(roles_raw),
        "predicted_narrative_roles_by_identity": extract_identity_roles(roles_raw),
        "predicted_viewpoints_by_identity": extract_identity_viewpoints(viewpoints_raw),
    }


def build_annotator_dataframe(
    annotator_entries: List[Dict[str, Any]],
    old_to_clean: Dict[str, str],
    benchmark_lookup: Dict[str, Any],
    annotator_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_no_clean_mapping = 0
    n_no_gt = 0

    for old_filename, meme_entries in group_by_meme(annotator_entries).items():
        clean_filename = old_to_clean.get(old_filename)
        if clean_filename is None:
            n_no_clean_mapping += 1
            logger.warning(
                "[%s] No clean-benchmark mapping for %s; skipping (not in the benchmark)",
                annotator_name, old_filename,
            )
            continue
        gt_entry = benchmark_lookup.get(clean_filename)
        if gt_entry is None:
            n_no_gt += 1
            logger.warning(
                "[%s] No GT entry for clean filename %s (mapped from %s); skipping",
                annotator_name, clean_filename, old_filename,
            )
            continue
        rows.append(build_prediction_row(meme_entries, clean_filename, gt_entry))

    logger.info(
        "[%s] Built %d rows (%d skipped: no clean mapping, %d skipped: no GT)",
        annotator_name, len(rows), n_no_clean_mapping, n_no_gt,
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format human-evaluator annotations into formatted_predictions.csv files."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(NO_GT_METRICS_DIR),
        help="Directory under which human_<X>_evaluator/ and human/ subdirs are written.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    )

    old_to_clean = load_old_to_clean_mapping()
    benchmark_lookup = load_benchmark_lookup()
    logger.info(
        "Loaded %d old->clean filename mappings, %d GT benchmark entries",
        len(old_to_clean), len(benchmark_lookup),
    )

    output_root = Path(args.output_root)
    pooled_rows: List[pd.DataFrame] = []
    seen_clean_filenames: set = set()

    annotator_files = sorted(HUMAN_EVAL_DIR.glob("*.json"))
    if not annotator_files:
        logger.error("No annotator JSON files found in %s", HUMAN_EVAL_DIR)
        sys.exit(1)

    for annotator_path in annotator_files:
        annotator_name = annotator_path.stem
        with open(annotator_path, "r", encoding="utf-8") as f:
            annotator_entries = json.load(f)

        df = build_annotator_dataframe(annotator_entries, old_to_clean, benchmark_lookup, annotator_name)
        if df.empty:
            logger.warning("No scorable rows for annotator %s; skipping output", annotator_name)
            continue

        out_dir = output_root / f"human_{annotator_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "formatted_predictions.csv", index=False)
        logger.info("Wrote %d rows to %s", len(df), out_dir / "formatted_predictions.csv")

        # Pool into the combined "human" dataset, deduplicating on clean
        # filename (a meme is very rarely labeled by two annotators; keep
        # the first annotator's row so each meme counts once).
        for _, row in df.iterrows():
            if row["image_filename"] in seen_clean_filenames:
                continue
            seen_clean_filenames.add(row["image_filename"])
            pooled_rows.append(row)

    if pooled_rows:
        pooled_df = pd.DataFrame(pooled_rows)
        pooled_dir = output_root / "human"
        pooled_dir.mkdir(parents=True, exist_ok=True)
        pooled_df.to_csv(pooled_dir / "formatted_predictions.csv", index=False)
        logger.info(
            "Wrote pooled human dataset: %d rows to %s",
            len(pooled_df), pooled_dir / "formatted_predictions.csv",
        )
        print(f"Wrote pooled human dataset: {len(pooled_df)} rows to {pooled_dir / 'formatted_predictions.csv'}")
    else:
        logger.error("No pooled rows produced; nothing to write")
        sys.exit(1)


if __name__ == "__main__":
    main()
