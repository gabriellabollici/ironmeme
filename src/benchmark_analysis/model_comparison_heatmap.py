"""Macro-F1 heatmaps (model x task) from the all_models_metrics.csv tables.

Outputs (src/benchmark_analysis/plots/):
  model_comparison_macro_f1_heatmap.png         GT-conditioned runs, all memes
  model_comparison_macro_f1_heatmap_subset.png  no-GT runs vs. the pooled human
                                                evaluators, on their shared subset

Run from the repo root: uv run python -m benchmark_analysis.model_comparison_heatmap
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from paths import GT_CONDITIONED_METRICS_DIR, SUBSET_NO_GT_METRICS_DIR

plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def plot_macro_f1_heatmap(metrics_dir, filename: str, title: str, title_size: float, figsize: tuple,
                          model_names: dict | None = None, ytick_rotation: float | None = None) -> None:
    df = pd.read_csv(os.path.join(metrics_dir, "all_metrics", "all_models_metrics.csv"))
    df = df[~df["task"].str.contains("per-identity alignment")]
    if model_names:
        df["model"] = df["model"].replace(model_names)

    pivot = df.pivot(index="model", columns="task", values="macro_f1")
    pivot = (pivot * 100).round(1)
    task_order = [t for t in ["Social Identity", "Viewpoint", "Narrative Roles"] if t in pivot.columns]
    pivot = pivot[task_order].sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 12},
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Macro F1 (%)"},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=15)
    if ytick_rotation is None:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    else:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation, fontsize=15)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Macro F1 (%)", fontsize=13)
    ax.set_title(title, fontsize=title_size, pad=15, weight="bold")
    ax.set_xlabel("Task", fontsize=15, labelpad=10)
    ax.set_ylabel("Model", fontsize=15, labelpad=10)

    fig.tight_layout()
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved to {out_path}")


def main() -> None:
    plot_macro_f1_heatmap(
        GT_CONDITIONED_METRICS_DIR, "model_comparison_macro_f1_heatmap.png",
        "Model Comparison - Macro F1 (%)", title_size=18, figsize=(10, 7),
    )

    # Humans never see ground truth, so they are compared against the no-GT runs.
    n_memes = pd.read_csv(os.path.join(SUBSET_NO_GT_METRICS_DIR, "human", "formatted_predictions.csv")).shape[0]
    plot_macro_f1_heatmap(
        SUBSET_NO_GT_METRICS_DIR, "model_comparison_macro_f1_heatmap_subset.png",
        f"No-GT Models vs. Human Comparison - Macro F1 (%)\n(subset-matched, n={n_memes} memes)",
        title_size=17, figsize=(10, 5.5), ytick_rotation=0,
        model_names={
            "glm-4.1v-9b-thinking": "GLM-4.1V-9B-Thinking\n(no GT)",
            "gemini-3.6-flash": "Gemini 3.6 Flash\n(no GT)",
            "human": "Human (pooled)",
        },
    )


if __name__ == "__main__":
    main()
