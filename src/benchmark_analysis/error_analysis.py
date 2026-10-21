"""Which labels do models and humans confuse?

- Viewpoint and narrative-role error matrices for Gemini 3.6 Flash (no-GT run)
  and the pooled human evaluators, on the same 214 memes
  (benchmark/results/metrics/subset_no_gt/). Neither side sees the ground
  truth, so this is the like-for-like comparison. Only identities the
  model/human got right are counted, which isolates the label-classification
  error from the identity-detection error.
- The most common social identity swaps summed over the 9 GT-conditioned VLMs
  (benchmark/results/metrics/gt_conditioned/): memes where exactly one GT
  identity is missing from the prediction and exactly one extra identity was
  predicted.

Labels are normalised with the same clean_and_normalize() as evaluate.py, so
counts match the published metrics.

Outputs:
  benchmark/results/metrics/subset_no_gt/all_metrics/error_analysis/{viewpoint,narrative_role}_error_matrix_{gemini,human}.csv
  benchmark/results/metrics/gt_conditioned/all_metrics/error_analysis/social_identity_top_swaps_9vlm.csv
  src/benchmark_analysis/plots/error_analysis/*.png

Run from the repo root: uv run python -m benchmark_analysis.error_analysis
"""
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from paths import GT_CONDITIONED_METRICS_DIR, SUBSET_NO_GT_METRICS_DIR
from experiments.evaluation.evaluate import clean_and_normalize, _norm_identity_key, _parse_identity_roles

plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "error_analysis")

GROUPS = [
    ("gemini", ["gemini-3.6-flash"], "Gemini 3.6 Flash, no GT"),
    ("human", ["human"], "human annotators"),  # all 5 evaluators pooled, one row per meme
]
VLM_MODELS = [
    "deepseek-vl2",
    "gemini-3.6-flash",
    "glm-4.1v-9b-thinking",
    "kimi-k3",
    "molmo-7b-d-0924",
    "pixtral-12b-2409",
    "qwen2.5-vl-7b-instruct",
    "qwen3-vl-8b-instruct",
    "qwen3-vl-8b-thinking",
]
NARRATIVE_ROLES = ["hero", "sage", "winner", "charmer", "fool", "loser", "victim", "villain", "monster"]
VIEWPOINTS = ["ingroup", "outgroup"]


def load_predictions(metrics_dir, model: str, columns: list) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(metrics_dir, model, "formatted_predictions.csv"))
    for col in columns:
        parse = _parse_identity_roles if col.endswith("_by_identity") else clean_and_normalize
        df[col] = df[col].apply(parse)
    return df


# ---------------------------------------------------------------------------
# Identity-aligned error matrices: Viewpoint and Narrative Roles
# ---------------------------------------------------------------------------
def _values_by_identity(entries: list, value_key: str) -> dict:
    """{normalised identity: set of roles / viewpoint} from a *_by_identity cell."""
    out = {}
    for e in entries:
        if not (isinstance(e, dict) and "identity" in e):
            continue
        key = _norm_identity_key(e["identity"])
        if value_key == "roles":
            out.setdefault(key, set()).update(
                str(r).strip().lower() for r in (e.get("roles") or []) if str(r).strip())
        elif e.get("viewpoint"):
            out.setdefault(key, set()).add(str(e["viewpoint"]).strip().lower())
    return out


def build_error_matrix(models: list, gt_col: str, pred_col: str, labels: list, value_key: str,
                       include_none_col: bool = True) -> pd.DataFrame:
    """value_key: 'roles' (multi-label set) or 'viewpoint' (single string).

    include_none_col adds a "(none predicted)" column for identity-matched GT
    labels that got no value at all -- drop it (False) when it is always zero,
    e.g. for Viewpoint, where a matched identity essentially always carries a
    predicted viewpoint, so the column adds no information."""
    all_labels = labels + ["(none predicted)"] if include_none_col else list(labels)
    matrix = pd.DataFrame(0, index=labels, columns=all_labels)
    for model in models:
        df = load_predictions(SUBSET_NO_GT_METRICS_DIR, model, [gt_col, pred_col])
        for gt_list, pred_list in zip(df[gt_col], df[pred_col]):
            gt_map, pred_map = _values_by_identity(gt_list, value_key), _values_by_identity(pred_list, value_key)
            for key, gt_vals in gt_map.items():
                if key not in pred_map:
                    continue  # identity itself not recognized -- a different error type
                pred_vals = pred_map[key]
                for g in gt_vals:
                    if g not in labels:
                        continue
                    if not pred_vals:
                        if include_none_col:
                            matrix.loc[g, "(none predicted)"] += 1
                    else:
                        for p in pred_vals:
                            if p in labels:
                                matrix.loc[g, p] += 1
    return matrix


def plot_error_matrix(matrix: pd.DataFrame, title: str, filename: str, figsize=(7, 6)):
    row_sums = matrix.sum(axis=1).replace(0, np.nan)
    pct = matrix.div(row_sums, axis=0) * 100
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pct, annot=matrix, fmt="d", cmap="Reds", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "% of that GT label's\nidentity-matched instances"}, ax=ax,
        annot_kws={"size": 9},
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontsize(9)
    cbar.ax.yaxis.labelpad = 12
    ax.set_xlabel("Predicted labels", fontsize=10, labelpad=8)
    ax.set_ylabel("Ground-truth label", fontsize=10, labelpad=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    # suptitle is centered on the whole figure canvas (not just the axes), so
    # a long title can't overflow past the left/right edge the way
    # ax.set_title() did when the axes were narrower than the title text.
    fig.suptitle(title, fontsize=13, weight="bold", y=1.02)
    fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(plots_dir, filename)}")


# ---------------------------------------------------------------------------
# Social identity swaps across the GT-conditioned VLMs
# ---------------------------------------------------------------------------
def social_identity_swaps(models: list) -> Counter:
    swaps = Counter()
    for model in models:
        df = load_predictions(GT_CONDITIONED_METRICS_DIR, model, ["social_identity_gt", "predicted_social_identities"])
        for gt, pred in zip(df["social_identity_gt"], df["predicted_social_identities"]):
            gt_set, pred_set = set(gt), set(pred)
            missed = gt_set - pred_set
            extra = pred_set - gt_set
            if len(missed) == 1 and len(extra) == 1:
                swaps[(next(iter(missed)), next(iter(extra)))] += 1
    return swaps


def plot_identity_swaps(swaps: Counter, title: str, filename: str, top_n: int = 15):
    top = swaps.most_common(top_n)
    if not top:
        return
    labels = [f"{gt} → {pred}" for (gt, pred), _ in top]
    counts = [c for _, c in top]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(12, max(3, 0.4 * len(labels))))
    ax.barh(y, counts, color="#8172b2", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    for yi, c in zip(y, counts):
        ax.text(c, yi, f" {c}", va="center", fontsize=8)
    ax.set_xlabel("Rows where this was the only missed GT identity and only extra prediction", fontsize=9)
    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close(fig)
    print(f"Saved {os.path.join(plots_dir, filename)}")


def main() -> None:
    os.makedirs(plots_dir, exist_ok=True)

    subset_csv_dir = os.path.join(SUBSET_NO_GT_METRICS_DIR, "all_metrics", "error_analysis")
    os.makedirs(subset_csv_dir, exist_ok=True)
    for group_name, models, label in GROUPS:
        viewpoint_matrix = build_error_matrix(models, "viewpoint_gt_by_identity", "predicted_viewpoints_by_identity",
                                              VIEWPOINTS, "viewpoint", include_none_col=False)
        role_matrix = build_error_matrix(models, "narrative_roles_gt_by_identity", "predicted_narrative_roles_by_identity",
                                         NARRATIVE_ROLES, "roles")
        viewpoint_matrix.to_csv(os.path.join(subset_csv_dir, f"viewpoint_error_matrix_{group_name}.csv"))
        role_matrix.to_csv(os.path.join(subset_csv_dir, f"narrative_role_error_matrix_{group_name}.csv"))
        plot_error_matrix(viewpoint_matrix, f"Viewpoint error matrix ({label}, identity-matched)",
                          f"viewpoint_error_matrix_{group_name}.png", figsize=(6.5, 5.5))
        plot_error_matrix(role_matrix, f"Narrative role error matrix ({label}, identity-matched)",
                          f"narrative_role_error_matrix_{group_name}.png")

    gt_csv_dir = os.path.join(GT_CONDITIONED_METRICS_DIR, "all_metrics", "error_analysis")
    os.makedirs(gt_csv_dir, exist_ok=True)
    swaps = social_identity_swaps(VLM_MODELS)
    csv_path = os.path.join(gt_csv_dir, "social_identity_top_swaps_9vlm.csv")
    pd.DataFrame(
        [(a, b, c) for (a, b), c in swaps.most_common(25)],
        columns=["gt_identity", "predicted_instead", "count"],
    ).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    plot_identity_swaps(swaps, f"Top social identity mismatches ({len(VLM_MODELS)} VLMs)",
                        "social_identity_top_swaps_9vlm.png")


if __name__ == "__main__":
    main()
