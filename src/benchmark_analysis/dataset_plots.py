"""Figures describing the benchmark dataset itself (ground truth only, no model
predictions), all read from benchmark/benchmark_dataset/ironmeme_benchmark.csv.

Outputs (src/benchmark_analysis/plots/):
  narrative_role_heatmap.png           identity x role balloon plot (10 most frequent identities)
  narrative_role_heatmap_vertical.png  the same, portrait: roles down the side
  ingroup_outgroup_roles.png           role shares, in-group vs. out-group viewpoint
  identity_by_source.png               social identity distribution per source dataset
  domain_distribution.png              memes per discussion domain
  reused_vs_scraped.png                reused benchmarks vs. scraped subreddits
  viewpoint_by_identity.png            viewpoint shares of the 10 most frequent identities
  template_role_barplot.png            narrative role composition of the 10 most frequent templates

Run from the repo root: uv run python -m benchmark_analysis.dataset_plots
"""
import ast
import os

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from paths import BENCHMARK_CSV

plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

ROLES = ("villain", "fool", "loser", "victim", "sage", "winner", "hero", "monster", "charmer")
# Order origins so related sources sit next to each other: the climate-themed
# pair first, the conservative-themed pair second, and the remaining two
# benchmarks last.
ORIGIN_ORDER = ["Clime", "r/ClimateMemes", "Constraint22", "r/ConservativeMemes", "r/TrollXChromosomes", "MAMI", "FigMemes"]


def _pastel(rgb, lightness=0.45):
    return tuple(lightness * ch + (1 - lightness) for ch in rgb[:3]) + (1.0,)


def explode_ground_truth(memes: pd.DataFrame) -> pd.DataFrame:
    """One row per ground-truth "identity-viewpoint-role" entry, with the three
    parts in social_identity_gt / viewpoint_gt / narrative_roles_gt. Memes
    without ground truth are dropped."""
    df = memes.copy()
    df["ground_truth"] = df["ground_truth"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)
    df = df[df["ground_truth"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    df = df.explode("ground_truth").reset_index(drop=True)
    # rsplit: identities can contain hyphens ("sex-worker"); viewpoint/role can't.
    df[["social_identity_gt", "viewpoint_gt", "narrative_roles_gt"]] = df["ground_truth"].str.rsplit("-", n=2, expand=True)
    return df


def _role_pivot(labels: pd.DataFrame):
    """Identity x role counts, and the 10 most frequent identities (most frequent first)."""
    pivot_df = pd.crosstab(labels["social_identity_gt"], labels["narrative_roles_gt"])
    identity_freq = labels["social_identity_gt"].value_counts()
    top_identities = [i for i in identity_freq.head(10).index if i in pivot_df.index]
    roles = [c for c in ROLES if c in pivot_df.columns]
    return pivot_df.loc[top_identities, roles], top_identities, roles


@mpl.rc_context()
def plot_role_balloons(labels: pd.DataFrame) -> None:
    # Balloon (dot) plot: dot area encodes the count, and identity-role pairs that
    # never co-occur simply have no dot, so sparsity reads as whitespace instead of
    # a wall of zeros. Dots are colored by narrative role so the same role is
    # easily tracked across identities.
    balloon_data, top_identities, roles = _role_pivot(labels)
    role_colors = [plt.cm.tab10(j % 10) for j in range(len(roles))]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_theme(style="whitegrid")

    for i, identity in enumerate(top_identities):
        for j, role in enumerate(roles):
            count = balloon_data.loc[identity, role]
            if count == 0:
                continue
            ax.scatter(j, i, s=40 + 130 * np.sqrt(count), color=role_colors[j], alpha=0.85,
                       edgecolors="white", linewidths=1.0)
            ax.text(j, i, str(count), ha="center", va="center", fontsize=7, color="white")

    ax.set_yticks(range(len(top_identities)))
    ax.set_yticklabels(top_identities, fontsize=10)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, rotation=45, ha="right", fontsize=10)
    ax.set_title("Identity-Role Correlation for the 10 Most Frequent Social Identities",
                 fontsize=14, pad=15, weight="bold")
    ax.set_xlabel("Narrative Roles", fontsize=11, labelpad=10)
    ax.set_ylabel("Social Identities", fontsize=11, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        handles=[Patch(color=role_colors[j], label=role) for j, role in enumerate(roles)],
        title="narrative roles", bbox_to_anchor=(1.02, 1), loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "narrative_role_heatmap.png"), dpi=300)
    plt.close(fig)


@mpl.rc_context()
def plot_role_balloons_vertical(labels: pd.DataFrame) -> None:
    """Portrait version of plot_role_balloons: roles run down the side and
    identities across the bottom; the role names label the rows, so no legend."""
    # Same font as the landscape figure (which gets Arial via seaborn's theme).
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    balloon_data, top_identities, roles = _role_pivot(labels)
    role_colors = [plt.cm.tab10(j % 10) for j in range(len(roles))]

    fig, ax = plt.subplots(figsize=(8, 9))
    for j, role in enumerate(roles):
        y = len(roles) - 1 - j  # first role on top
        for i, identity in enumerate(top_identities):
            count = balloon_data.loc[identity, role]
            if count == 0:
                continue
            ax.scatter(i, y, s=60 + 190 * np.sqrt(count), color=role_colors[j], alpha=0.85,
                       edgecolors="white", linewidths=1.0)
            ax.text(i, y, str(count), ha="center", va="center", fontsize=9.5, color="white")

    ax.set_xticks(range(len(top_identities)))
    ax.set_xticklabels(top_identities, rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(len(roles)))
    ax.set_yticklabels(roles[::-1], fontsize=13)
    ax.set_xlim(-0.6, len(top_identities) - 0.4)
    ax.set_ylim(-0.6, len(roles) - 0.4)
    ax.set_title("Identity-Role Correlation for the\n10 Most Frequent Social Identities",
                 fontsize=17, pad=15, weight="bold")
    ax.set_xlabel("Social Identities", fontsize=14, labelpad=10)
    ax.set_ylabel("Narrative Roles", fontsize=14, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "narrative_role_heatmap_vertical.png"), dpi=300)
    plt.close(fig)


@mpl.rc_context()
def plot_ingroup_outgroup_roles(labels: pd.DataFrame) -> None:
    # Diverging bar chart: In-group vs Out-group role percentages
    sns.set_theme(style="whitegrid")
    crosstab = pd.crosstab(labels["viewpoint_gt"], labels["narrative_roles_gt"], normalize="index") * 100

    viewpoints = crosstab.index.tolist()
    roles = [c for c in crosstab.columns if c in ["villain", "fool", "loser", "victim", "sage", "winner", "hero", "monster"]]

    in_group_pcts = [crosstab.loc["in_group", r] if "in_group" in viewpoints else 0 for r in roles]
    out_group_pcts = [crosstab.loc["out_group", r] if "out_group" in viewpoints else 0 for r in roles]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(roles))

    ax.barh(y_pos, [-x for x in out_group_pcts], color="#e74c3c", align="center", label="Out-Group Perspective")
    ax.barh(y_pos, in_group_pcts, color="#2ecc71", align="center", label="In-Group Perspective")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r.title() for r in roles], fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)

    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(int(x))}%" for x in ticks])

    ax.set_xlabel("Percentage of Role Assignments (%)", fontsize=11)
    ax.set_title("Role Framing Shift: In-Group vs. Out-Group Perspective", fontsize=14, pad=15, weight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ingroup_outgroup_roles.png"), dpi=300)
    plt.close()


@mpl.rc_context()
def plot_identity_by_source(labels: pd.DataFrame) -> None:
    # Stacked bar chart: social identity distribution per source dataset. Each
    # row's ``origin`` records whether the meme came from a reused benchmark
    # (MAMI, Clime, Constraint22, FigMemes) or a subreddit (r/ClimateMemes, ...).
    sns.set_theme(style="whitegrid")
    df_cross = pd.crosstab(labels["origin"].fillna("Unknown"), labels["social_identity_gt"])
    top_n = 10
    top_cols = df_cross.sum().sort_values(ascending=False).head(top_n).index
    other = df_cross.drop(columns=top_cols).sum(axis=1)
    df_cross = df_cross[top_cols].copy()
    df_cross["Other"] = other
    df_perc = df_cross.div(df_cross.sum(axis=1), axis=0) * 100
    df_perc = df_perc.reindex([o for o in ORIGIN_ORDER if o in df_perc.index])

    # Evenly-spaced hues give each identity a distinct color (Set2 recycles similar
    # hues across 10+ categories, e.g. women vs climate activist). Hues span the
    # full circle without wrapping back to red, so the last category ('Other') is
    # not a duplicate of the first ('women'), and are lightened toward white for
    # softer pastel tones.
    n_categories = len(df_perc.columns)
    palette = [_pastel(plt.cm.hsv(i / n_categories)) for i in range(n_categories)]

    ax = df_perc.plot(kind="bar", stacked=True, figsize=(12, 9), color=palette, width=0.4)

    plt.title("Distribution of Social Identities Across Source Sub-Datasets", fontsize=20, pad=15, weight="bold")
    plt.xlabel("Source Dataset Origin", fontsize=18, labelpad=10)
    plt.ylabel("Proportion of Dataset (%)", fontsize=18)
    # Legend anchored to the bottom of the figure; tight_layout below reserves
    # the space for it so it never overlaps the x-axis title.
    ax.get_figure().legend(
        *ax.get_legend_handles_labels(),
        title="Social Identity", bbox_to_anchor=(0.5, 0.01), loc="lower center",
        ncol=3, fontsize=15, title_fontsize=16,
    )
    plt.xticks(rotation=25, ha="right", rotation_mode="anchor", fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    ax.get_legend().remove()
    plt.tight_layout(rect=[0, 0.19, 1, 1])
    plt.savefig(os.path.join(plots_dir, "identity_by_source.png"), dpi=300)
    plt.close()


@mpl.rc_context()
def plot_domain_distribution(memes: pd.DataFrame) -> None:
    # Donut chart: distribution of memes per discussion domain
    sns.set_theme(style="whitegrid")
    domain_map = {
        "r/TrollXChromosomes": "gender discussions",
        "MAMI": "gender discussions",
        "r/ClimateMemes": "climate discussions",
        "Clime": "climate discussions",
        "r/ConservativeMemes": "political discussions",
        "Constraint22": "political discussions",
        "FigMemes": "other/undetermined",
    }
    domain_df = memes[["image_filename", "origin"]].drop_duplicates("image_filename").copy()
    domain_df["domain"] = domain_df["origin"].map(domain_map)
    domain_order = ["gender discussions", "climate discussions", "political discussions", "other/undetermined"]
    domain_counts = domain_df["domain"].value_counts().reindex(
        [d for d in domain_order if d in domain_df["domain"].values])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    wedges, texts, autotexts = ax.pie(
        domain_counts.values,
        labels=domain_counts.index,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=11),
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color("white")
    ax.set_title("Distribution of Memes Across Discussion Domains", fontsize=14, pad=15, weight="bold")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "domain_distribution.png"), dpi=300)
    plt.close(fig)


@mpl.rc_context()
def plot_reused_vs_scraped(memes: pd.DataFrame) -> None:
    # Nested donut chart: reused benchmarks vs scraped subreddits
    sns.set_theme(style="whitegrid")
    reused_origins = {"MAMI", "Constraint22", "Clime", "FigMemes"}
    src_df = memes[["image_filename", "origin"]].drop_duplicates("image_filename").copy()
    total_src = len(src_df)

    origin_counts = src_df["origin"].value_counts()
    reused_n = sum(1 for o in origin_counts.index if o in reused_origins)
    scraped_n = len(origin_counts) - reused_n

    reused_vals, reused_labels, reused_colors = [], [], []
    scraped_vals, scraped_labels, scraped_colors = [], [], []
    for origin, count in origin_counts.items():
        if origin in reused_origins:
            k = len(reused_vals)
            reused_vals.append(count)
            reused_labels.append(f"{origin} ({count})")
            reused_colors.append(plt.cm.Blues(0.4 + 0.5 * k / max(1, reused_n - 1)))
        else:
            k = len(scraped_vals)
            scraped_vals.append(count)
            scraped_labels.append(f"{origin} ({count})")
            scraped_colors.append(plt.cm.Reds(0.45 + 0.45 * k / max(1, scraped_n - 1)))

    reused_sum = sum(reused_vals)
    scraped_sum = sum(scraped_vals)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("equal")

    # Outer ring: individual source origins
    ax.pie(
        reused_vals + scraped_vals,
        colors=reused_colors + scraped_colors,
        radius=1.0,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.30, edgecolor="white", linewidth=2),
    )

    # Inner ring: reused vs scraped
    _, inner_texts = ax.pie(
        [reused_sum, scraped_sum],
        colors=["#4c72b0", "#c44e52"],
        radius=0.68,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2),
        labels=[
            f"Reused\n{reused_sum / total_src * 100:.1f}%",
            f"Scraped\n{scraped_sum / total_src * 100:.1f}%",
        ],
        labeldistance=0.8,
        textprops=dict(color="black", fontsize=21.6, weight="bold"),
    )
    for t in inner_texts:
        t.set_bbox(dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"))

    ax.text(0, 0, f"{total_src}\nmemes", ha="center", va="center", fontsize=22.8, weight="bold")
    ax.set_title("Dataset Origin", fontsize=24, pad=12, weight="bold")
    fig.legend(
        handles=[Patch(facecolor=c, edgecolor="white", label=l) for c, l in zip(reused_colors + scraped_colors, reused_labels + scraped_labels)],
        loc="center left",
        bbox_to_anchor=(0.555, 0.5),
        ncol=1,
        fontsize=19.2,
    )

    # Donut on the left, single-column legend on the right.
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.01, right=0.55)
    fig.savefig(os.path.join(plots_dir, "reused_vs_scraped.png"), dpi=300)
    plt.close(fig)


@mpl.rc_context()
def plot_viewpoint_by_identity(labels: pd.DataFrame, top_n: int = 10) -> None:
    """Share of in-group vs. out-group annotations for the top_n most frequent
    identities (100% stacked bars, identities ordered by frequency).

    Counting unit: one (meme, identity, viewpoint) annotation. An identity with
    several roles in the same meme has one ground-truth entry per role; those
    are counted once, so its viewpoint isn't double-counted. Entries with no
    viewpoint are shown as a separate "unannotated" share."""
    # Categorical slots 1-2 of the validated default palette, plus a neutral for
    # the unannotated share; text stays in ink colors, never the series color.
    colors = {"ingroup": "#2a78d6", "outgroup": "#eb6834", "unannotated": "#b8b7b0"}
    INK, INK_MUTED = "#0b0b0b", "#52514e"

    df = labels.rename(columns={"social_identity_gt": "identity"})
    df["viewpoint"] = df["viewpoint_gt"].map({"in_group": "ingroup", "out_group": "outgroup"}).fillna("unannotated")
    pairs = df.drop_duplicates(["image_filename", "identity", "viewpoint"])

    top = pairs["identity"].value_counts().head(top_n).index
    counts = (pairs[pairs["identity"].isin(top)]
              .groupby(["identity", "viewpoint"]).size().unstack(fill_value=0)
              .reindex(columns=["ingroup", "outgroup", "unannotated"], fill_value=0)
              .loc[top])
    shares = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 0.48 * top_n + 1.4))
    y = range(len(top))[::-1]
    left = pd.Series(0.0, index=top)
    for vp in ["ingroup", "outgroup", "unannotated"]:
        if counts[vp].sum() == 0:
            continue
        ax.barh(list(y), shares[vp], left=left, height=0.68, color=colors[vp],
                edgecolor="white", linewidth=2, label=vp.capitalize())
        for yi, ident in zip(y, top):
            s = shares.loc[ident, vp]
            if s >= 8:
                ax.text(left[ident] + s / 2, yi, f"{s:.0f}%", ha="center", va="center",
                        fontsize=9, color="white" if vp != "unannotated" else INK)
        left += shares[vp]

    ax.set_yticks(list(y))
    ax.set_yticklabels([i.replace("-", " ").capitalize() for i in top], fontsize=10, color=INK)
    for yi, ident in zip(y, top):
        ax.text(101.5, yi, f"n={counts.loc[ident].sum()}", va="center", fontsize=9, color=INK_MUTED)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]], fontsize=9, color=INK_MUTED)
    ax.set_xlabel("Share of ground-truth annotations", fontsize=10, color=INK_MUTED)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cfcec8")
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Ground-truth viewpoint of the {top_n} most common social identities",
                 fontsize=12, weight="bold", color=INK, pad=28, loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9,
              handlelength=1.2, borderaxespad=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "viewpoint_by_identity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


@mpl.rc_context()
def plot_template_roles(labels: pd.DataFrame) -> None:
    # Top 10 templates by total narrative-role-tagged occurrences, broken down
    # by ALL narrative roles (stacked), not just each template's single most
    # common role.
    top_templates = labels.groupby("template_name").size().nlargest(10).index.tolist()

    roles_ordered = sorted(labels["narrative_roles_gt"].dropna().unique())
    counts = (
        labels[labels["template_name"].isin(top_templates)]
        .groupby(["template_name", "narrative_roles_gt"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=top_templates, columns=roles_ordered, fill_value=0)
    )

    pastel_palette = [
        "#E88D8D", "#6DA8E8", "#E8D967", "#67E89D", "#67BFE8",
        "#9D67E8", "#E867B3", "#8DE867", "#67E8D9", "#E8A567", "#A5E867",
    ]
    role_colors = {r: pastel_palette[i % len(pastel_palette)] for i, r in enumerate(roles_ordered)}

    fig, ax = plt.subplots(figsize=(24, 14))

    x_pos = range(len(top_templates))
    bottom = [0] * len(top_templates)
    for role in roles_ordered:
        values = counts[role].tolist()
        ax.bar(x_pos, values, bottom=bottom, color=role_colors[role],
               edgecolor="white", linewidth=0.6, width=0.65, label=role)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_templates, fontsize=24, rotation=30, ha="right")

    ax.set_ylabel("Count", fontsize=30)
    ax.tick_params(axis="y", labelsize=22)
    ax.set_title("Narrative Role Composition of Top 10 Templates", fontsize=38, pad=30, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 60)
    ax.set_yticks(range(0, 50, 10))  # extra headroom above the bars, without "50" or "60" ticks

    legend_handles = [mpatches.Patch(facecolor=role_colors[r], edgecolor="white", label=r) for r in roles_ordered]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        fontsize=24,
        title="Narrative Role",
        title_fontsize=28,
        frameon=True,
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
    )

    fig.subplots_adjust(left=0.15, bottom=0.30, top=0.85, right=0.78)
    fig.savefig(os.path.join(plots_dir, "template_role_barplot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    memes = pd.read_csv(BENCHMARK_CSV)
    labels = explode_ground_truth(memes)

    plot_role_balloons(labels)
    plot_role_balloons_vertical(labels)
    plot_ingroup_outgroup_roles(labels)
    plot_identity_by_source(labels)
    plot_domain_distribution(memes)
    plot_reused_vs_scraped(memes)
    plot_viewpoint_by_identity(labels)
    plot_template_roles(labels)
    print(f"Saved figures to {plots_dir}")


if __name__ == "__main__":
    main()
