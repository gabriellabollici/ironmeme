"""Turn the raw per-annotator JSON exports into the aggregated annotations dataset.

Each file in ``RAW_ANNOTATIONS_DIR`` is one export batch (a "set") containing one
row per (image, annotator, labeled instance): the same ``id`` (one per image)
and ``annotator_name`` can appear multiple times when an annotator tagged more
than one social identity in the same image (instance 1, instance 2, ...).

This script:
    1. Combines every file into a single set of per-annotator instances per
       image, tagging each with a human-readable ``original_dataset`` name.
    2. Pivots that into one row per image, with list-valued
       ``social_identity``/``view_point``/``narrative_roles`` columns per
       annotator (formatted as ``"I1: ...", "I2: ..."`` etc., matching each
       annotator's original instance numbering).
    3. Derives ``ground_truth``/``agreement_type`` by finding, for every pair of
       annotators who labeled an image, which (identity, view_point, role)
       triples they agree on (case-insensitively on identity); the union of
       all agreeing pairs' triples is the image's ground truth.
    4. Splits rows with a non-empty ground truth into train/test (rows with no
       cross-annotator agreement are marked ``no_consensus``).

Used in memory by ``dataset_construction.build_benchmark``; nothing is written
to disk. Run directly to print a summary and regenerate the annotation
agreement report and plots (``annotation_analysis``):
    python -m dataset_construction.annotation_formatting.annotation_postprocessing
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from paths import DATASETS_DIR

RAW_ANNOTATIONS_DIR = DATASETS_DIR / "annotation_results/raw_json_annotations"

# Fixed annotator order, used both for column ordering and for the order in
# which agreeing pairs are listed in ``agreement_type`` (e.g. "J-L"
# before "J-M" before "L-M").
ANNOTATORS = ("J", "L", "M")

# Train/test split parameters applied to rows that have a non-empty
# ground_truth (i.e. at least two annotators agreed on at least one triple).
TEST_SIZE = 0.38
RANDOM_STATE = 42
# Annotation sets added after the original split was made. Each is split on
# its own (same parameters) so earlier train/test assignments stay fixed; this
# reproduces the released benchmark's split exactly.
LATE_SETS = ("SET 12",)

# Maps a raw annotation filename's stem (lowercased) to the human-readable dataset name used throughout the rest of
# the pipeline (matches build_benchmark.py's ORIGINAL_DATASET_TO_FOLDER keys).
RAW_STEM_TO_DATASET_NAME = {
    "dry_run_1": "DRY RUN 1",
    "dry_run_2": "DRY RUN 2",
    "set1": "SET 1",
    "set_2": "SET 2",
    "set_3": "SET 3",
    "set_4": "SET 4",
    "set_5_1": "SET 5_1",
    "set_5_2": "SET 5_2",
    "set_6": "SET 6",
    "set_7": "SET 7",
    "set_8": "SET 8",
    "set_9": "SET 9",
    "set_10": "SET 10",
    "set_11": "SET 11",
    "set_12": "SET 12",
}


def annotator_short_name(annotator_name: str) -> str:
    """Turn e.g. ``"J_annotator"`` into ``"J"``."""
    return annotator_name.removesuffix("_annotator")


def load_raw_annotations(directory: Path) -> list[dict]:
    """Load every raw annotation record from ``directory``, tagged with its dataset name."""
    records = []
    for path in sorted(directory.glob("*.json")):
        dataset_name = RAW_STEM_TO_DATASET_NAME.get(path.stem.lower())
        if dataset_name is None:
            raise ValueError(
                f"No known dataset-name mapping for raw annotation file {path.name!r}; "
                f"add it to RAW_STEM_TO_DATASET_NAME."
            )
        with open(path) as f:
            file_records = json.load(f)
        for record in file_records:
            record = dict(record)
            record["original_dataset"] = dataset_name
            records.append(record)
    return records


def compute_consensus(triples_by_annotator: dict[str, set[str]]) -> tuple[list[str], str | None]:
    """Derive ``(ground_truth, agreement_type)`` from each annotator's triple set.

    For every pair of annotators who both labeled the image, checks whether
    they agree on at least one (identity, view_point, role) triple. The
    ground truth is the union of all agreeing pairs' shared triples;
    agreement_type lists which pairs agreed (``None`` if no pair agreed on
    anything).
    """
    present = [a for a in ANNOTATORS if a in triples_by_annotator]
    agreeing_pairs = []
    ground_truth_triples: set[str] = set()
    for annotator_a, annotator_b in combinations(present, 2):
        shared = triples_by_annotator[annotator_a] & triples_by_annotator[annotator_b]
        if shared:
            agreeing_pairs.append(f"{annotator_a}-{annotator_b}")
            ground_truth_triples |= shared

    ground_truth = sorted(ground_truth_triples)
    agreement_type = f"Consensus ({', '.join(agreeing_pairs)})" if agreeing_pairs else None
    return ground_truth, agreement_type


def agreement_level(triples_by_annotator: dict[str, set[str]], ground_truth: list[str]) -> str | None:
    """``"perfect agreement"`` when every ground-truth triple was given by every
    annotator who labeled the image; otherwise ``"pairwise agreement"`` (at least
    two annotators share at least one triple). ``None`` without a ground truth.

    Every annotator who labeled the image counts, also one whose labels give no
    complete triple (an identity without a role, or a role without an identity):
    they did not agree on the ground truth.
    """
    if not ground_truth:
        return None
    if set(ground_truth) <= set.intersection(*triples_by_annotator.values()):
        return "perfect agreement"
    return "pairwise agreement"


def aggregate_to_wide(records: list[dict]) -> pd.DataFrame:
    """Pivot per-annotator-instance raw records into one row per image."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["id"]].append(record)

    rows = []
    for group_id, group_records in groups.items():
        image_filename = group_records[0]["image_filename"]
        original_dataset = group_records[0]["original_dataset"]

        by_annotator: dict[str, list[dict]] = defaultdict(list)
        for record in group_records:
            by_annotator[annotator_short_name(record["annotator_name"])].append(record)

        row: dict[str, object] = {"image_filename": image_filename, "id": group_id}
        triples_by_annotator: dict[str, set[str]] = {}

        for annotator in ANNOTATORS:
            annotator_records = sorted(by_annotator.get(annotator, []), key=lambda r: r["date_time"])
            social_identity_col = f"social_identity_{annotator}_annotator"
            view_point_col = f"view_point_{annotator}_annotator"
            narrative_roles_col = f"narrative_roles_{annotator}_annotator"

            if not annotator_records:
                row[social_identity_col] = None
                row[view_point_col] = None
                row[narrative_roles_col] = None
                continue

            social_identities, view_points, narrative_roles = [], [], []
            triples: set[str] = set()
            for instance_num, record in enumerate(annotator_records, start=1):
                labels = record["labels"]
                identity = labels["social_identity"]
                view_point = labels["view_point"]
                roles = labels["narrative_roles"]

                social_identities.append(f"I{instance_num}: {identity}")
                view_points.append(f"V{instance_num}: {view_point}")
                narrative_roles.append(f"R{instance_num}: {roles!r}")

                for role in roles:
                    triples.add(f"{identity.strip().lower()}-{view_point}-{role}")

            row[social_identity_col] = social_identities
            row[view_point_col] = view_points
            row[narrative_roles_col] = narrative_roles
            triples_by_annotator[annotator] = triples

        ground_truth, agreement_type = compute_consensus(triples_by_annotator)
        row["ground_truth"] = ground_truth
        row["agreement_type"] = agreement_type
        row["agreement_level"] = agreement_level(triples_by_annotator, ground_truth)
        row["original_dataset"] = original_dataset
        rows.append(row)

    column_order = [
        "image_filename",
        "id",
        *(f"social_identity_{a}_annotator" for a in ANNOTATORS),
        *(f"view_point_{a}_annotator" for a in ANNOTATORS),
        *(f"narrative_roles_{a}_annotator" for a in ANNOTATORS),
        "ground_truth",
        "agreement_type",
        "agreement_level",
        "original_dataset",
    ]
    return pd.DataFrame(rows, columns=column_order)


def assign_train_test_split(
    df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Split rows with a non-empty ``ground_truth`` into train/test.

    Rows with an empty ground truth (no two annotators agreed on anything)
    are assigned ``"no_consensus"``. The split is a reproducible random split
    (fixed seed), done once for the original annotation sets and once for each
    set in ``LATE_SETS``, which is how the released benchmark was split.
    """
    df = df.copy()
    df["set"] = "no_consensus"

    has_consensus = df["ground_truth"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    is_late = df["original_dataset"].isin(LATE_SETS)
    groups = [has_consensus & ~is_late] + [has_consensus & (df["original_dataset"] == s) for s in LATE_SETS]
    for group in groups:
        group_index = df.index[group]
        if len(group_index) > 0:
            train_index, test_index = train_test_split(
                group_index, test_size=test_size, random_state=random_state
            )
            df.loc[train_index, "set"] = "train"
            df.loc[test_index, "set"] = "test"

    return df


def build_annotations_dataframe() -> pd.DataFrame:
    """Build the full aggregated annotations DataFrame straight from the raw JSON exports."""
    records = load_raw_annotations(RAW_ANNOTATIONS_DIR)
    df = aggregate_to_wide(records)
    df = assign_train_test_split(df)

    # Put `set` right after `agreement_type`, matching the historical column order.
    column_order = [c for c in df.columns if c not in ("set", "original_dataset")]
    column_order += ["set", "original_dataset"]
    return df[column_order]


def main() -> None:
    df = build_annotations_dataframe()

    print(f"Aggregated {len(df)} annotated memes")
    print(f"columns: {list(df.columns)}")
    print(df["set"].value_counts(dropna=False))
    print(df["original_dataset"].value_counts())

    # Recompute the annotation-analysis.ipynb exploration (agreement metrics,
    # Jaccard, benchmark breakdown and the agreement-comparison plots) straight
    # from the same raw annotation JSON exports.
    from dataset_construction.annotation_formatting.annotation_analysis import (
        run_annotation_analysis,
    )

    run_annotation_analysis()


if __name__ == "__main__":
    main()
