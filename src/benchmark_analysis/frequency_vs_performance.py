"""How is model performance related to the frequency of meme templates,
social identities, and narrative roles in the benchmark?

For each of the three dimensions, pairs a *ground-truth* frequency (a
property of the benchmark itself, independent of any model) with a
*performance* measure, and reports the Spearman rank correlation between
the two -- POOLED across models: one data point per (model, label) for
identities/roles, one data point per (model, image) for templates. This
means the same label/template frequency appears once per model, each time
paired with that model's own score, rather than being collapsed to a
single per-entity mean before correlating. This is the same pooling used
for the "Pooled Spearman" statistics reported in the paper text.

Produces a single 3-panel dot-plot figure plus a text summary of the
correlations and the underlying per-entity/per-model table.

Reuses evaluation/format_predictions.py (the same GT/prediction
normalisation used by the official evaluate.py pipeline) so frequencies and
scores here are computed the same way as the published metrics.
"""
import ast
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from paths import (
    BENCHMARK_JSON,
    GT_CONDITIONED_METRICS_DIR,
    RAW_GT_CONDITIONED_DIR,
)
from experiments.evaluation.format_predictions import build_dataframe, normalise_gt_identities, normalise_gt_roles
from matplotlib.ticker import NullFormatter, LogLocator


MODELS = [
    "deepseek-vl2", "glm-4.1v-9b-thinking", "kimi-k3", "molmo-7b-d-0924",
    "pixtral-12b-2409", "qwen2.5-vl-7b-instruct", "qwen3-vl-8b-instruct", "qwen3-vl-8b-thinking",
    "gemini-3.6-flash",
]
# Hosted-API runs that kept only formatted_predictions.csv (no per-image step
# outputs to rebuild from). That CSV was written by the same
# format_predictions.build_dataframe(), so parsing it back gives identical rows.
CSV_ONLY_MODELS = {"gemini-3.6-flash"}
LIST_COLUMNS = [
    "social_identity_gt", "viewpoint_gt", "narrative_roles_gt",
    "predicted_social_identities", "predicted_viewpoints", "predicted_narrative_roles",
]
RNG = np.random.default_rng(0)


# --- Panels 2 & 3 data: per-(model, label) F1, pooled across models --------
def per_label_f1_pooled(per_model_df, gt_col, pred_col, freq_counter):
    pooled_rows = []  # one row per (model, label): label's GT frequency + that model's F1
    mean_rows = {}    # label -> list of per-model F1 (for the human-readable table only)
    for model, df in per_model_df.items():
        mlb = MultiLabelBinarizer()
        mlb.fit(list(df[gt_col]) + list(df[pred_col]))
        yt = mlb.transform(df[gt_col])
        yp = mlb.transform(df[pred_col])
        _, _, f1, support = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)
        for label, s, f in zip(mlb.classes_, support, f1):
            if s == 0:
                continue
            freq = freq_counter.get(label)
            if freq is None:
                continue
            pooled_rows.append({"model": model, "label": label, "frequency": freq, "f1": f})
            mean_rows.setdefault(label, []).append(f)
    pooled_df = pd.DataFrame(pooled_rows)
    mean_df = pd.DataFrame(
        {"label": label, "frequency": freq_counter.get(label), "mean_f1": np.mean(f1s), "n_models": len(f1s)}
        for label, f1s in mean_rows.items()
    )
    return pooled_df, mean_df

# ---------------------------------------------------------------------------
# Spearman correlations, computed on the POOLED (model x entity) data --
# the same data that gets plotted below.
# ---------------------------------------------------------------------------
def corr(df, x, y):
    r, p = spearmanr(df[x], df[y])
    return r, p


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    results_path = os.path.join(script_dir, "frequency_vs_performance_results.txt")

    # ---------------------------------------------------------------------------
    # Ground-truth frequencies (model-independent: computed once from the
    # benchmark itself, not from any single model's output).
    # ---------------------------------------------------------------------------
    benchmark = json.load(open(BENCHMARK_JSON, encoding="utf-8"))
    benchmark_lookup = {e["image_filename"]: e for e in benchmark if "image_filename" in e}

    template_freq = Counter(e.get("template_name") for e in benchmark if e.get("template_name"))

    # ---------------------------------------------------------------------------
    # Per-model predictions (via the official format_predictions.py pipeline).
    # ---------------------------------------------------------------------------
    per_model_df = {}
    for model in MODELS:
        if model in CSV_ONLY_MODELS:
            # Re-normalised copy (recompute_all_metrics.py); the one under
            # The raw outputs predate the shared invalid-label rule.
            df = pd.read_csv(GT_CONDITIONED_METRICS_DIR / model / "formatted_predictions.csv")
            df["image_filename"] = df["image_filename"].astype(str)
            for col in LIST_COLUMNS:
                df[col] = df[col].apply(ast.literal_eval)
        else:
            df = build_dataframe(RAW_GT_CONDITIONED_DIR / model, benchmark_lookup)
        per_model_df[model] = df

    # Identity/role frequency must be counted on the SAME normalised (taxonomy
    # mapped, lowercased) labels that scoring uses (build_dataframe() applies
    # these same two functions to each row's GT). Computed once over the full
    # benchmark so it's independent of any single model's image subset.
    identity_freq = Counter()
    role_freq = Counter()
    for e in benchmark:
        identity_freq.update(normalise_gt_identities(e.get("social_identity_gt", "[]")))
        role_freq.update(normalise_gt_roles(e.get("narrative_roles_gt", "[]")))

    # --- Panel 1 data: per-(model, image) Jaccard, pooled across models --------
    template_rows = []
    for model, df in per_model_df.items():
        tmpl_by_img = {e["image_filename"]: e.get("template_name") for e in benchmark}
        for _, row in df.iterrows():
            tmpl = tmpl_by_img.get(row["image_filename"])
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
                if tmpl:
                    break
                tmpl = tmpl_by_img.get(row["image_filename"] + ext)
            if not tmpl:
                continue
            gt = set(row["social_identity_gt"]) | set(row["viewpoint_gt"]) | set(row["narrative_roles_gt"])
            pr = set(row["predicted_social_identities"]) | set(row["predicted_viewpoints"]) | set(row["predicted_narrative_roles"])
            union = gt | pr
            jac = len(gt & pr) / len(union) if union else 1.0
            template_rows.append({"model": model, "template": tmpl, "jaccard": jac})

    tmpl_pooled = pd.DataFrame(template_rows)
    tmpl_pooled["frequency"] = tmpl_pooled["template"].map(template_freq)
    tmpl_pooled = tmpl_pooled.dropna(subset=["frequency"])

    # Per-template mean (for the human-readable "top 10 templates" table only;
    # NOT used for the correlation or the figure).
    tmpl_mean = tmpl_pooled.groupby("template")["jaccard"].mean().reset_index()
    tmpl_mean["frequency"] = tmpl_mean["template"].map(template_freq)

    identity_pooled, identity_mean = per_label_f1_pooled(per_model_df, "social_identity_gt", "predicted_social_identities", identity_freq)
    role_pooled, role_mean = per_label_f1_pooled(per_model_df, "narrative_roles_gt", "predicted_narrative_roles", role_freq)

    r_tmpl, p_tmpl = corr(tmpl_pooled, "frequency", "jaccard")
    r_id, p_id = corr(identity_pooled, "frequency", "f1")
    r_role, p_role = corr(role_pooled, "frequency", "f1")

    # ---------------------------------------------------------------------------
    # Figure: 3 dot plots, one figure. Each dot is one (model, entity) pair, so
    # a given entity's GT frequency appears once per model, each time paired
    # with that model's own score -- a small multiplicative x-jitter (visual
    # only, not used for the correlation) keeps same-frequency dots from
    # stacking into a single vertical line.
    # ---------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    panels = [
        (axes[0], tmpl_pooled, "frequency", "jaccard", "Meme template",
         "Template frequency in benchmark (# images, log scale)", "Per-image Jaccard (pooled across models)",
         r_tmpl, p_tmpl, "#2a78d6", 10, 0.18),
        (axes[1], identity_pooled, "frequency", "f1", "Social identity",
         "Identity frequency in GT (# images, log scale)", "F1 (pooled across models)",
         r_id, p_id, "#eb6834", 24, 0.45),
        (axes[2], role_pooled, "frequency", "f1", "Narrative role",
         "Role frequency in GT (# images, log scale)", "F1 (pooled across models)",
         r_role, p_role, "#1baf7a", 24, 0.45),
    ]

    for ax, data, xcol, ycol, title, xlabel, ylabel, r, p, color, msize, alpha in panels:
        x_jittered = data[xcol] * RNG.uniform(0.85, 1.18, size=len(data))
        ax.scatter(x_jittered, data[ycol], s=msize, alpha=alpha, color=color, edgecolor="none")
        ax.set_xscale("log")
        # Only label the decade (major) ticks -- a narrow x-range (e.g. the 9
        # narrative-role labels) otherwise fills in with dense, overlapping
        # minor-tick labels (2x10^1, 3x10^1, 4x10^1, ...).
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f"{title}  (n={len(data)})", fontsize=17, fontweight="bold")
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, which="major", linestyle="--", alpha=0.25)
        p_label = "< 0.001" if p < 0.001 else f"= {p:.3f}"
        ax.text(
            0.03, 0.96, f"Pooled Spearman ρ = {r:.3f}\np {p_label}",
            transform=ax.transAxes, ha="left", va="top", fontsize=13.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#c3c2b7", alpha=0.9),
        )
        ax.set_ylim(-0.03, 1.03)

    fig.suptitle(
        "Label frequency vs model performance (pooled across models)",
        fontsize=21, fontweight="bold", y=1.04,
    )
    fig.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "frequency_vs_performance.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {plot_path}")

    # ---------------------------------------------------------------------------
    # Text results (correlations + underlying per-entity tables).
    # ---------------------------------------------------------------------------
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Frequency vs. performance -- Pooled Spearman correlations\n")
        f.write("=" * 60 + "\n\n")
        f.write("Each point is one (model, entity) pair -- a template/identity/role's\n")
        f.write("ground-truth frequency in the benchmark, repeated once per model and\n")
        f.write("paired with that model's own score (Jaccard for templates, F1 for\n")
        f.write("identity/role labels). This matches the 'Pooled Spearman' statistics\n")
        f.write("reported in the paper text (as opposed to correlating frequency\n")
        f.write("against each entity's mean score across models).\n\n")

        f.write(f"1. Meme template frequency vs. per-image Jaccard (n={len(tmpl_pooled)}, model x image pairs)\n")
        f.write(f"   Spearman rho = {r_tmpl:.4f}, p = {p_tmpl:.4g}\n\n")

        f.write(f"2. Social identity frequency vs. F1 (n={len(identity_pooled)}, model x label pairs)\n")
        f.write(f"   Spearman rho = {r_id:.4f}, p = {p_id:.4g}\n\n")

        f.write(f"3. Narrative role frequency vs. F1 (n={len(role_pooled)}, model x label pairs)\n")
        f.write(f"   Spearman rho = {r_role:.4f}, p = {p_role:.4g}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Social identity labels, sorted by frequency (desc) -- mean F1 across models:\n")
        f.write("-" * 60 + "\n")
        for _, row in identity_mean.sort_values("frequency", ascending=False).iterrows():
            f.write(f"  {row['label']:28s} freq={int(row['frequency']):5d}  mean_f1={row['mean_f1']:.3f}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("Narrative role labels, sorted by frequency (desc) -- mean F1 across models:\n")
        f.write("-" * 60 + "\n")
        for _, row in role_mean.sort_values("frequency", ascending=False).iterrows():
            f.write(f"  {row['label']:28s} freq={int(row['frequency']):5d}  mean_f1={row['mean_f1']:.3f}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write(f"Meme templates: {len(tmpl_mean)} total, top 10 most frequent -- mean Jaccard across models:\n")
        f.write("-" * 60 + "\n")
        for _, row in tmpl_mean.sort_values("frequency", ascending=False).head(10).iterrows():
            f.write(f"  {row['template']:45s} freq={int(row['frequency']):3d}  mean_jaccard={row['jaccard']:.3f}\n")

    print(f"Saved results to {results_path}")

    # Also save the raw pooled (model x entity) tables as CSV for reuse.
    tmpl_pooled.sort_values(["frequency", "model"], ascending=[False, True]).to_csv(
        os.path.join(script_dir, "frequency_vs_performance_templates.csv"), index=False)
    identity_pooled.sort_values(["frequency", "model"], ascending=[False, True]).to_csv(
        os.path.join(script_dir, "frequency_vs_performance_identities.csv"), index=False)
    role_pooled.sort_values(["frequency", "model"], ascending=[False, True]).to_csv(
        os.path.join(script_dir, "frequency_vs_performance_roles.csv"), index=False)
    print("Saved pooled per-(model, entity) CSVs (templates/identities/roles) alongside the results file.")


if __name__ == "__main__":
    main()
