import pandas as pd
import ast
import os
import sys
from paths import BENCHMARK_CSV


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.stdout = open(os.path.join(script_dir, "benchmark_analysis_results.txt"), "w", encoding="utf-8")

    # The benchmark CSV is the single source of truth: one row per meme with its
    # ``origin``, ``template_name``, ``description`` and FAISS ``similarity_score``.
    benchmark_df = pd.read_csv(BENCHMARK_CSV)

    # turn GT into real lists
    benchmark_df = benchmark_df[["image_filename", "ground_truth", "origin", "template_name"]]

    benchmark_df['ground_truth'] = benchmark_df['ground_truth'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    # Keep only non-empty lists
    benchmark_df = benchmark_df[benchmark_df['ground_truth'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    total_unique_templates = benchmark_df['template_name'].nunique()

    # Unpack lists into individual string rows
    benchmark_df = benchmark_df.explode('ground_truth').reset_index(drop=True)

    # Split the individual strings into 3 columns
    # Use rsplit (from the right) since social identities can themselves
    # contain hyphens (e.g. "sex-worker", "left-wing", "anti-feminist"),
    # while viewpoint/role are always single tokens.
    split_cols = benchmark_df['ground_truth'].str.rsplit('-', n=2, expand=True)

    # Assign to new columns
    benchmark_df[["social_identity_gt", "viewpoint_gt", "narrative_roles_gt"]] = split_cols
    # ------------------------
    print("=" * 80)
    print("FULL BENCHMARK DATAFRAME (all columns and rows after processing):")
    print("=" * 80)
    print(benchmark_df)
    print("\n" + "=" * 80)
    print("FIRST 5 ROWS (preview):")
    print("=" * 80)
    print(benchmark_df.head())
    print("\n" + "=" * 80)
    print("COLUMN NAMES IN THE DATAFRAME:")
    print("=" * 80)
    print(benchmark_df.columns.tolist())
    print("\n" + "=" * 80)
    print("TEMPLATES PER SOURCE ORIGIN (how many samples come from each source dataset):")
    print("=" * 80)
    print(benchmark_df.value_counts('origin'))
    print("\n" + "=" * 80)
    print("MEMES PER DISCUSSION DOMAIN (each meme counted once, percentage relative to the whole benchmark):")
    print("=" * 80)
    domain_map = {
        "r/TrollXChromosomes": "gender discussions",
        "MAMI": "gender discussions",
        "r/ClimateMemes": "climate discussions",
        "Clime": "climate discussions",
        "r/ConservativeMemes": "political discussions",
        "Constraint22": "political discussions",
        "FigMemes": "other/undetermined",
    }
    meme_domain = benchmark_df[['image_filename', 'origin']].drop_duplicates('image_filename').copy()
    meme_domain['domain'] = meme_domain['origin'].map(domain_map)
    total_memes = len(meme_domain)
    print(f"Total memes in benchmark: {total_memes}")
    domain_order = ["gender discussions", "climate discussions", "political discussions", "other/undetermined"]
    domain_counts = meme_domain['domain'].value_counts().reindex([d for d in domain_order if d in meme_domain['domain'].values])
    for domain, count in domain_counts.items():
        pct = count / total_memes * 100
        print(f"{domain} --> {count} memes ({pct:.2f}%)")
    print("\nBreakdown by source origin within each domain:")
    for domain in [d for d in domain_order if d in meme_domain['domain'].values]:
        print(f"\n--- {domain} ---")
        for origin, count in meme_domain[meme_domain['domain'] == domain]['origin'].value_counts().items():
            print(f"  {origin} --> {count} memes")
    print("\n" + "=" * 80)
    print("REUSED BENCHMARKS vs SCRAPED SOURCES (each meme counted once, percentage relative to the whole benchmark):")
    print("=" * 80)
    reused_origins = [o for o in ["MAMI", "Constraint22", "Clime", "FigMemes"] if o in meme_domain['origin'].values]
    scraped_origins = [o for o in sorted(meme_domain['origin'].unique()) if o.startswith("r/")]
    reused_count = len(meme_domain[meme_domain['origin'].isin(reused_origins)])
    scraped_count = len(meme_domain[meme_domain['origin'].isin(scraped_origins)])
    print(f"Reused benchmarks --> {reused_count} memes ({reused_count / total_memes * 100:.2f}%)")
    print(f"Scraped (subreddits) --> {scraped_count} memes ({scraped_count / total_memes * 100:.2f}%)")
    print("\nBreakdown by source origin:")
    for origin, count in meme_domain['origin'].value_counts().items():
        bucket = "reused benchmark" if origin in reused_origins else "scraped"
        print(f"  {origin} ({bucket}) --> {count} memes")
    print("\n" + "=" * 80)
    print("TOP 10 MOST COMMON TEMPLATES (how many memes use each template):")
    print("=" * 80)
    meme_templates = benchmark_df[['image_filename', 'template_name']].drop_duplicates('image_filename')
    top_templates = meme_templates['template_name'].value_counts().head(10)
    for template, count in top_templates.items():
        print(f"{template} --> {count} memes")
    print("\n" + "=" * 80)
    print("TOTAL UNIQUE TEMPLATES (how many distinct template names exist across the entire benchmark):")
    print("=" * 80)
    print(f"Total unique templates: {total_unique_templates}")
    print("\n" + "=" * 80)
    print("SAMPLES PER TEMPLATE (how many images use each template):")
    print("=" * 80)
    print(benchmark_df.value_counts('template_name'))
    print("\n" + "=" * 80)
    print("SOCIAL IDENTITY COUNTS PER ORIGIN DATASET (how many times each social identity appears, broken down by source):")
    print("=" * 80)
    for origin, group in benchmark_df.groupby('origin'):
        print(f"\n--- {origin} ---")
        counts = group['social_identity_gt'].value_counts()
        for identity, count in counts.items():
            print(f"  {identity} --> {count} times")
    print("\n" + "=" * 80)
    print("TOP 10 MOST COMMON SOCIAL IDENTITIES ACROSS ALL DATASETS (the most frequently occurring social identity labels):")
    print("=" * 80)
    top_identities = benchmark_df['social_identity_gt'].value_counts().head(10)
    for identity, count in top_identities.items():
        print(f"{identity} --> {count} times")
    all_roles = benchmark_df['narrative_roles_gt'].value_counts()
    print("\n" + "=" * 80)
    print(f"ALL {len(all_roles)} NARRATIVE ROLES (how many times each narrative role appears):")
    print("=" * 80)
    for role, count in all_roles.items():
        print(f"{role} --> {count} times")
    print("\n" + "=" * 80)
    print("TOP 10 MOST COMMON IDENTITY-ROLE ASSOCIATIONS (most frequent combinations of social identity + narrative role):")
    print("=" * 80)
    identity_role_pairs = benchmark_df.groupby(['social_identity_gt', 'narrative_roles_gt']).size().reset_index(name='count').sort_values('count', ascending=False).head(10)
    for _, row in identity_role_pairs.iterrows():
        print(f"{row['social_identity_gt']} as {row['narrative_roles_gt']} --> {row['count']} times")
    print("\n" + "=" * 80)
    print("TOP 10 IDENTITY-VIEWPOINT ASSOCIATIONS PER DATASET (most frequent combinations of social identity + viewpoint, broken down by source):")
    print("=" * 80)
    for origin, group in benchmark_df.groupby('origin'):
        print(f"\n--- {origin} ---")
        top_pairs = group.groupby(['social_identity_gt', 'viewpoint_gt']).size().reset_index(name='count').sort_values('count', ascending=False).head(5)
        for _, row in top_pairs.iterrows():
            print(f"  {row['social_identity_gt']} ({row['viewpoint_gt']}) --> {row['count']} times")
    print("\n" + "=" * 80)
    print("LEAST COMMON IDENTITIES (the 10 least frequently occurring social identities):")
    print("=" * 80)
    bottom_identities = benchmark_df['social_identity_gt'].value_counts().tail(10)
    for identity, count in bottom_identities.items():
        print(f"{identity} --> {count} times")
    print("\n" + "=" * 80)
    print("LEAST COMMON ROLES (the 10 least frequently occurring narrative roles):")
    print("=" * 80)
    bottom_roles = benchmark_df['narrative_roles_gt'].value_counts().tail(10)
    for role, count in bottom_roles.items():
        print(f"{role} --> {count} times")
    print("\n" + "=" * 80)
    print("TOP 15 MOST COMMON IDENTITY-VIEWPOINT-ROLE ASSOCIATIONS ACROSS THE ENTIRE DATASET:")
    print("=" * 80)
    identity_viewpoint_role = benchmark_df.groupby(['social_identity_gt', 'viewpoint_gt', 'narrative_roles_gt']).size().reset_index(name='count').sort_values('count', ascending=False).head(15)
    for _, row in identity_viewpoint_role.iterrows():
        print(f"{row['social_identity_gt']} ({row['viewpoint_gt']}) as {row['narrative_roles_gt']} --> {row['count']} times")
    print("\n" + "=" * 80)
    print("TOP 15 MOST COMMON IDENTITY-VIEWPOINT-TEMPLATE ASSOCIATIONS ACROSS THE ENTIRE DATASET:")
    print("=" * 80)
    identity_viewpoint_template = benchmark_df.groupby(['social_identity_gt', 'viewpoint_gt', 'template_name']).size().reset_index(name='count').sort_values('count', ascending=False).head(15)
    for _, row in identity_viewpoint_template.iterrows():
        print(f"{row['social_identity_gt']} ({row['viewpoint_gt']}) in '{row['template_name']}' --> {row['count']} times")
    print("\n" + "=" * 80)
    print("TOP 5 MOST COMMON IDENTITY-VIEWPOINT-ROLE-TEMPLATE ASSOCIATIONS ACROSS THE ENTIRE DATASET:")
    print("=" * 80)
    identity_viewpoint_role_template = benchmark_df.groupby(['social_identity_gt', 'viewpoint_gt', 'narrative_roles_gt', 'template_name']).size().reset_index(name='count').sort_values('count', ascending=False).head(5)
    for _, row in identity_viewpoint_role_template.iterrows():
        print(f"{row['social_identity_gt']} ({row['viewpoint_gt']}) as {row['narrative_roles_gt']} in '{row['template_name']}' --> {row['count']} times")
    print("\n" + "=" * 80)
    print("TOP 15 MOST COMMON TEMPLATE-ROLE ASSOCIATIONS ACROSS THE ENTIRE DATASET:")
    print("=" * 80)
    template_role = benchmark_df.groupby(['template_name', 'narrative_roles_gt']).size().reset_index(name='count').sort_values('count', ascending=False).head(15)
    for _, row in template_role.iterrows():
        print(f"'{row['template_name']}' → {row['narrative_roles_gt']} --> {row['count']} times")
    print("\n" + "=" * 80)
    print("POSITIVE vs NEGATIVE ROLE ANALYSIS (comparing images with more negative roles vs positive roles):")
    print("=" * 80)
    negative_roles = {'fool', 'villain', 'loser', 'monster'}
    benchmark_df['is_negative'] = benchmark_df['narrative_roles_gt'].isin(negative_roles)
    role_counts_per_image = benchmark_df.groupby('image_filename')['is_negative'].agg(
        num_negative=lambda x: x.sum(),
        num_positive=lambda x: (~x).sum()
    )
    more_negative_images = role_counts_per_image['num_negative'] > role_counts_per_image['num_positive']
    percentage = (more_negative_images.sum() / len(role_counts_per_image)) * 100
    print(f"Total unique images: {len(role_counts_per_image)}")
    print(f"Images with more negative than positive roles: {more_negative_images.sum()}")
    print(f"Percentage: {percentage:.2f}%")

    sys.stdout.close()


if __name__ == "__main__":
    main()
