"""Replicate the annotation-analysis.ipynb exploration inside the post-processing pipeline.

The notebook reads the raw per-annotator JSON exports, pivots them per annotation
set, and computes three families of agreement statistics:

    1. Perfect agreement rates per category (``social_identity``, ``view_point``,
       ``narrative_roles``) for every set.
    2. Pairwise Jaccard agreement per set (averaged over rows) and a grand total
       across all sets.
    3. A train/test/no_consensus breakdown derived from relational
       (identity-viewpoint-role) triple consensus across annotator pairs.

It also renders a single plot comparing, per annotation stage, the average
pairwise Jaccard agreement for social identities against narrative roles. The
dry-run sets are collapsed into one "DRY RUN" stage and the two halves of SET 5
into one "SET 5" stage.

This module re-implements that analysis directly on the raw annotation files in
``benchmark/source_datasets/annotation_results/raw_json_annotations`` and, when run as part
of ``annotation_postprocessing.py``, writes:

    annotation_formatting/annotation_analysis_results.txt
    annotation_formatting/plots/jaccard_agreement_comparison.png

Run directly to regenerate them:
    python src/dataset_construction/annotation_formatting/annotation_analysis.py
"""

import ast
import io
import re
from collections import defaultdict
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataset_construction.annotation_formatting.annotation_postprocessing import (
    RAW_ANNOTATIONS_DIR,
    load_raw_annotations,
)

SCRIPT_DIR = Path(__file__).parent
PLOTS_DIR = SCRIPT_DIR / "plots"
REPORT_PATH = SCRIPT_DIR / "annotation_analysis_results.txt"

# Canonical annotation-stage order (matches the notebook's ``stages_names`` but
# extended to every set present in the raw annotations directory).
DATASET_ORDER = [
    "DRY RUN 1",
    "DRY RUN 2",
    "SET 1",
    "SET 2",
    "SET 3",
    "SET 4",
    "SET 5_1",
    "SET 5_2",
    "SET 6",
    "SET 7",
    "SET 8",
    "SET 9",
    "SET 10",
    "SET 11",
    "SET 12",
]

CORE_CATEGORIES = ["social_identity", "view_point", "narrative_roles"]

# Annotation sets merged into a single plot stage (the two dry runs and the two
# halves of SET 5).
STAGE_COLLAPSE = {
    "DRY RUN 1": "DRY RUN",
    "DRY RUN 2": "DRY RUN",
    "SET 5_1": "SET 5",
    "SET 5_2": "SET 5",
}

AGREEMENT_PALETTE = {
    "Social Identity Agreement": "#E74C3C",
    "Viewpoint Agreement": "#2ECC71",
    "Narrative Role Agreement": "#3498DB",
}


def collapse_stage(dataset_name: str) -> str:
    """Map a raw annotation set onto its collapsed plot stage."""
    return STAGE_COLLAPSE.get(dataset_name, dataset_name)


def _is_missing(value) -> bool:
    """True when a pivoted cell holds no annotation data (NaN / None / 'nan')."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and (value.strip() == "" or value.lower() == "nan"):
        return True
    if isinstance(value, list) and not value:
        return True
    return False


def parse_prefixed_string(value: object) -> dict[str, list[str]]:
    """Parse a pivoted cell into ``{instance_index: [lowercased labels]}``.

    Cells are lists of ``"I1: sex-worker"`` / ``"R1: ['fool', 'loser']"`` style
    strings (one per labeled instance). The index key keeps identity, viewpoint
    and role instances aligned so relational triples can be rebuilt.
    """
    labels_by_index: dict[str, list[str]] = defaultdict(list)
    if _is_missing(value):
        return dict(labels_by_index)
    items = list(value) if isinstance(value, list) else [value]
    for item in items:
        text = str(item).strip()
        match = re.match(r"^([A-Za-z])(\d+):\s*(.*)$", text)
        if not match:
            continue
        _, index_id, content = match.groups()
        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, list):
                labels_by_index[index_id].extend(str(label).strip().lower() for label in parsed)
                continue
        except (ValueError, SyntaxError):
            pass
        if content.strip():
            labels_by_index[index_id].append(content.strip().lower())
    return dict(labels_by_index)


def parse_cell_to_labels(value: object) -> set[str]:
    """Flatten a pivoted cell into a clean set of labels (prefixes stripped)."""
    labels = set()
    for labels_list in parse_prefixed_string(value).values():
        labels.update(labels_list)
    return labels


def extract_relational_annotations(row: pd.Series, annotators: list[str]) -> dict[str, set[str]]:
    """Rebuild each annotator's ``identity-viewpoint-role`` triple set from a row."""
    annotator_maps: dict[str, set[str]] = {annotator: set() for annotator in annotators}
    for annotator in annotators:
        social = parse_prefixed_string(row[("social_identity", annotator)])
        views = parse_prefixed_string(row[("view_point", annotator)])
        roles = parse_prefixed_string(row[("narrative_roles", annotator)])
        for index_id, identities in social.items():
            if not identities:
                continue
            identity_key = identities[0]
            viewpoint = (views.get(index_id) or ["unknown_viewpoint"])[0]
            for role in roles.get(index_id) or ["unknown_role"]:
                annotator_maps[annotator].add(f"{identity_key}-{viewpoint}-{role}")
    return annotator_maps


def find_relational_consensus(row: pd.Series, annotators: list[str]) -> tuple[list[str], str]:
    """Accumulate all triples shared by at least one pair of annotators."""
    annotator_maps = extract_relational_annotations(row, annotators)
    all_agreed_strings: set[str] = set()
    participating_pairs: set[str] = set()
    for annotator_a, annotator_b in combinations(annotators, 2):
        intersection = annotator_maps[annotator_a].intersection(annotator_maps[annotator_b])
        if intersection:
            all_agreed_strings.update(intersection)
            participating_pairs.add(f"{annotator_a.split('_')[0]}-{annotator_b.split('_')[0]}")
    if all_agreed_strings:
        agreement_type = f"Consensus ({', '.join(sorted(participating_pairs))})"
        return sorted(all_agreed_strings), agreement_type
    return [], "None"


def calculate_split_type(row: pd.Series, ground_truth: list[str], annotators: list[str]) -> str:
    """Structural test/train assignment, mirroring the notebook's ``calculate_split_type``."""
    if not ground_truth:
        return "no_consensus"
    annotator_maps = extract_relational_annotations(row, annotators)
    active_maps = [annotator_maps[a] for a in annotators if annotator_maps[a]]
    if len(active_maps) >= 2:
        perfect_intersection = set.intersection(*active_maps)
        if perfect_intersection and set(ground_truth).issubset(perfect_intersection):
            return "test"
    return "train"


def build_pivoted_dataframes(records: list[dict]) -> dict[str, pd.DataFrame]:
    """Pivot the raw records into one row per image per annotation set.

    Columns use the same ``(category, annotator_name)`` MultiIndex the notebook
    pivots on, so every metric below can be computed per set in one place.
    """
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_dataset[record["original_dataset"]].append(record)

    pivoted: dict[str, pd.DataFrame] = {}
    for dataset_name, dataset_records in by_dataset.items():
        df = pd.DataFrame(dataset_records)
        labels_expanded = pd.json_normalize(df["labels"])
        df = pd.concat(
            [df.drop(columns=["labels"]).reset_index(drop=True), labels_expanded.reset_index(drop=True)],
            axis=1,
        )

        def label_instances(series: pd.Series, prefix: str) -> list[str]:
            return [f"{prefix}{i + 1}: {value}" for i, value in enumerate(series)]

        df = (
            df.groupby(["image_filename", "annotator_name"])
            .agg(
                id=("id", "first"),
                date_time=("date_time", "max"),
                social_identity=("social_identity", lambda x: label_instances(x, "I")),
                view_point=("view_point", lambda x: label_instances(x, "V")),
                narrative_roles=("narrative_roles", lambda x: label_instances(x, "R")),
            )
            .reset_index()
        )
        pivoted[dataset_name] = df.pivot(
            index=["image_filename", "id"],
            columns="annotator_name",
            values=["social_identity", "view_point", "narrative_roles"],
        )
    return pivoted


def _row_annotators(df: pd.DataFrame) -> list[str]:
    return [a for a in df.columns.get_level_values(1).unique() if a]


def check_row_agreement(row: pd.Series) -> bool:
    """True when all annotators with data agree on a category (vacuous if <=1)."""
    clean_vals = [tuple(sorted(parse_cell_to_labels(x))) for x in row if not _is_missing(x)]
    if len(clean_vals) <= 1:
        return True
    return all(v == clean_vals[0] for v in clean_vals)


def calculate_agreement_metrics(df: pd.DataFrame, categories: list[str]) -> dict[str, float]:
    """Perfect-agreement percentage per category (and overall)."""
    total_perfect = pd.Series(True, index=df.index)
    results: dict[str, float] = {}
    for category in categories:
        category_agreement = df[category].apply(check_row_agreement, axis=1)
        results[category] = category_agreement.mean() * 100
        total_perfect = total_perfect & category_agreement
    results["total"] = total_perfect.mean() * 100
    return results


def compute_category_jaccard(df: pd.DataFrame, category: str) -> list[float]:
    """Average pairwise Jaccard per row for one category (empty-vs-empty = 1.0)."""
    annotators = _row_annotators(df)
    pairs = list(combinations(annotators, 2))
    row_scores = []
    for _, row in df.iterrows():
        sets = {a: parse_cell_to_labels(row[(category, a)]) for a in annotators}
        pairwise_scores = []
        for annotator_a, annotator_b in pairs:
            set_a, set_b = sets[annotator_a], sets[annotator_b]
            if not set_a and not set_b:
                pairwise_scores.append(1.0)
            elif not set_a or not set_b:
                pairwise_scores.append(0.0)
            else:
                pairwise_scores.append(len(set_a.intersection(set_b)) / len(set_a.union(set_b)))
        if pairwise_scores:
            row_scores.append(sum(pairwise_scores) / len(pairwise_scores))
    return row_scores


def compute_stage_jaccard(
    pivoted: dict[str, pd.DataFrame], stage_keys: list[str], category: str
) -> dict[str, float]:
    """Average pairwise Jaccard per collapsed plot stage for one category."""
    stage_order = []
    for key in stage_keys:
        collapsed = collapse_stage(key)
        if collapsed not in stage_order:
            stage_order.append(collapsed)

    stage_averages: dict[str, float] = {}
    for stage in stage_order:
        members = [key for key in stage_keys if collapse_stage(key) == stage]
        row_scores: list[float] = []
        for key in members:
            row_scores.extend(compute_category_jaccard(pivoted[key], category))
        stage_averages[stage] = sum(row_scores) / len(row_scores) if row_scores else 0.0
    return stage_averages


def generate_jaccard_comparison_plot(
    pivoted: dict[str, pd.DataFrame], stage_keys: list[str]
) -> None:
    """Single line plot: average pairwise Jaccard per stage for both categories.

    Dry runs are collapsed into one "DRY RUN" stage and the two halves of
    SET 5 into "SET 5"; only the Jaccard agreement metric is shown, with
    social identities and narrative roles drawn as two colored lines.
    """
    plot_records = []
    for category, agreement_label in (
        ("social_identity", "Social Identity Agreement"),
        ("view_point", "Viewpoint Agreement"),
        ("narrative_roles", "Narrative Role Agreement"),
    ):
        stage_averages = compute_stage_jaccard(pivoted, stage_keys, category)
        for stage, average in stage_averages.items():
            plot_records.append({"Stage": stage, "Agreement": agreement_label, "Jaccard": average})

    plot_df = pd.DataFrame(plot_records)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=plot_df,
        x="Stage",
        y="Jaccard",
        hue="Agreement",
        style="Agreement",
        markers=["o", "^", "s"],
        dashes=False,
        markersize=10,
        linewidth=2.5,
        palette=AGREEMENT_PALETTE,
        ax=ax,
    )

    ax.set_title("Pairwise Jaccard Agreement Across Annotation Stages", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Annotation Stage", fontsize=12, labelpad=12)
    ax.set_ylabel("Average Jaccard Agreement", fontsize=12, labelpad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(
        title="Agreement Type",
        bbox_to_anchor=(0.5, -0.26),
        loc="upper center",
        ncol=3,
        borderaxespad=0,
        fontsize=11,
        title_fontsize=12,
    )
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "jaccard_agreement_comparison.png", dpi=300)
    plt.close(fig)


def print_highest_disagreement_instances(pivoted: dict[str, pd.DataFrame], top_n: int = 10) -> None:
    """Print the instances with the lowest per-row Jaccard for the target categories."""
    for target_category in ["narrative_roles", "social_identity", "view_point"]:
        problematic_instances = []
        for key, df in pivoted.items():
            annotators = _row_annotators(df)
            pairs = list(combinations(annotators, 2))
            if not pairs:
                continue
            for idx, row in df.iterrows():
                sets = {a: parse_cell_to_labels(row[(target_category, a)]) for a in annotators}
                raw_values = {a: row[(target_category, a)] for a in annotators}
                pairwise_scores = []
                for annotator_a, annotator_b in pairs:
                    set_a, set_b = sets[annotator_a], sets[annotator_b]
                    if not set_a and not set_b:
                        pairwise_scores.append(1.0)
                    elif not set_a or not set_b:
                        pairwise_scores.append(0.0)
                    else:
                        pairwise_scores.append(len(set_a.intersection(set_b)) / len(set_a.union(set_b)))
                if pairwise_scores:
                    problematic_instances.append(
                        {
                            "dataset": key,
                            "image_filename": row.name[0],
                            "avg_jaccard": sum(pairwise_scores) / len(pairwise_scores),
                            "cleaned_annotations": sets,
                            "raw_annotations": raw_values,
                        }
                    )

        problematic_instances.sort(key=lambda x: x["avg_jaccard"])

        print("\n" + "=" * 60)
        print(f"=== TOP {top_n} DISAGREEMENT INSTANCES FOR: {target_category.upper()} ===")
        print("=" * 60)

        if not problematic_instances:
            print("No valid instances evaluated.")
            continue

        for i, instance in enumerate(problematic_instances[:top_n]):
            print(
                f"\n[Disagreement #{i + 1}] Dataset: '{instance['dataset']}' | "
                f"Image: {instance['image_filename']}"
            )
            print(f"Average Jaccard Score: {instance['avg_jaccard']:.3f}")
            print("Cleaned Annotator Outputs:")
            for annotator, cleaned_set in instance["cleaned_annotations"].items():
                print(f"  - {annotator}: {sorted(cleaned_set)}  (Raw: {instance['raw_annotations'][annotator]})")
            print("-" * 50)


def run_annotation_analysis(records: list[dict] | None = None) -> None:
    """Recompute the notebook analysis from raw annotations and write report + plots."""
    if records is None:
        records = load_raw_annotations(RAW_ANNOTATIONS_DIR)

    pivoted = build_pivoted_dataframes(records)
    stage_keys = [s for s in DATASET_ORDER if s in pivoted]

    report = io.StringIO()
    with redirect_stdout(report):
        print("=" * 80)
        print("PERFECT AGREEMENT METRICS PER ANNOTATION SET (all annotators agree):")
        print("=" * 80)
        for label in stage_keys:
            print(f"\n--- {label} ({len(pivoted[label])} images) ---")
            results = calculate_agreement_metrics(pivoted[label], CORE_CATEGORIES)
            for category in CORE_CATEGORIES:
                print(f"  Perfect Agreement for '{category}': {results[category]:.2f}%")
            print(f"  TOTAL PERFECT AGREEMENT: {results['total']:.2f}%")

        print("\n" + "=" * 80)
        print("PROVISIONAL BENCHMARK METRIC BREAKDOWN (structural test/train/no_consensus):")
        print("=" * 80)
        split_records = []
        for label in stage_keys:
            df = pivoted[label]
            annotators = _row_annotators(df)
            for _, row in df.iterrows():
                ground_truth, agreement_type = find_relational_consensus(row, annotators)
                split_type = calculate_split_type(row, ground_truth, annotators)
                split_records.append(
                    {
                        "image_filename": row.name[0],
                        "id": row.name[1],
                        "ground_truth": ground_truth,
                        "agreement_type": agreement_type,
                        "set": split_type,
                        "original_dataset": label,
                    }
                )
        final_combined_df = pd.DataFrame(split_records)
        total_samples = len(final_combined_df)
        set_counts = final_combined_df["set"].value_counts()
        perfect_cnt = set_counts.get("test", 0)
        majority_cnt = set_counts.get("train", 0)
        no_consensus_cnt = set_counts.get("no_consensus", 0)
        total_with_gt = perfect_cnt + majority_cnt

        print(f"Total Row Samples Analyzed: {total_samples}\n")
        print(f"Perfect Agreement (3/3 Split - 'test'): {perfect_cnt} samples ({perfect_cnt / total_samples * 100:.2f}%)")
        print(f"Majority Consensus (2/3 Split - 'train'): {majority_cnt} samples ({majority_cnt / total_samples * 100:.2f}%)")
        print(f"No Consensus (Excluded): {no_consensus_cnt} samples ({no_consensus_cnt / total_samples * 100:.2f}%)")
        print("-" * 50)
        print(f"Total Ground Truth Intersections Found: {total_with_gt} samples ({total_with_gt / total_samples * 100:.2f}%)")

        print("\n" + "=" * 80)
        print("JACCARD AGREEMENT PER ANNOTATION SET (average pairwise, per category):")
        print("=" * 80)
        for target_category in ["narrative_roles", "social_identity", "view_point"]:
            print(f"\n=== JACCARD AGREEMENT FOR: {target_category} ===")
            all_dataset_scores = []
            for label in stage_keys:
                row_scores = compute_category_jaccard(pivoted[label], target_category)
                if row_scores:
                    dataset_avg = sum(row_scores) / len(row_scores)
                    all_dataset_scores.extend(row_scores)
                    print(f"  '{label}' Average Jaccard: {dataset_avg:.3f}")
                else:
                    print(f"Dataset '{label}': No valid rows evaluated.")
            grand_avg = sum(all_dataset_scores) / len(all_dataset_scores) if all_dataset_scores else 0.0
            print(f"  GRAND TOTAL PAIRWISE JACCARD AGREEMENT: {grand_avg:.3f}")

        print("\n" + "=" * 80)
        print("AGREEMENT EVOLUTION OVER THE ANNOTATION PROCESS (collapsed stages):")
        print("=" * 80)
        for category, label in (
            ("social_identity", "Social Identity Agreement"),
            ("view_point", "Viewpoint Agreement"),
            ("narrative_roles", "Narrative Role Agreement"),
        ):
            stage_averages = compute_stage_jaccard(pivoted, stage_keys, category)
            ordered_stages = list(stage_averages)
            start_stage, end_stage = ordered_stages[0], ordered_stages[-1]
            start_value, end_value = stage_averages[start_stage], stage_averages[end_stage]
            absolute_increase = end_value - start_value
            relative_increase = (absolute_increase / start_value * 100) if start_value else 0.0
            overall_average = sum(stage_averages.values()) / len(stage_averages)

            print(f"\n{label}:")
            print(f"  Start ({start_stage}): {start_value:.3f}")
            print(f"  End ({end_stage}): {end_value:.3f}")
            print(f"  Average agreement at the end: {end_value:.3f}")
            print(f"  Increase over the process: {absolute_increase:+.3f} "
                  f"(i.e. {absolute_increase * 100:+.2f} percentage points, {relative_increase:+.1f}% relative)")
            print(f"  Average agreement across all stages: {overall_average:.3f}")

        print_highest_disagreement_instances(pivoted, top_n=10)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.getvalue(), encoding="utf-8")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_jaccard_comparison_plot(pivoted, stage_keys)

    print(f"Analysis report written to {REPORT_PATH}")
    print(f"Plots written to {PLOTS_DIR}")


if __name__ == "__main__":
    run_annotation_analysis()
