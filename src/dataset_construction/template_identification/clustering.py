import numpy as np
import pandas as pd
import imagehash
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from paths import DATASETS_DIR, REPO_ROOT

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
FOR_CLUSTERING_DIR = DATASETS_DIR / "memes/clustering/for_clustering"
OUTPUT_DIR = DATASETS_DIR / "memes/clustering/clustering_results"
# Images will only be placed in the same cluster if their dHashes differ by
# this many bits or fewer (out of 64).
DHASH_DISTANCE_THRESHOLD = 6
# Manual mapping from cluster id -> template name. Leave empty and run this
# script once to get the cluster assignments, inspect clusters of interest
# with visualize_cluster(df, cluster_id=...), then fill this in and rerun to
# also produce a labeled subset.
TEMPLATE_MAPPING: dict[int, str] = {}


def build_image_dataframe(root: Path) -> pd.DataFrame:
    """Build a dataframe with an ``image_path`` column for every image under ``root``."""
    image_paths = [
        str(path.relative_to(REPO_ROOT)) for path in sorted(root.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return pd.DataFrame({"image_path": image_paths})


def compute_optimized_dhash(image_path: str, target_size=(256, 256), blur_radius: int = 3) -> str:
    """Compute a perceptual dHash that is robust to meme text overlays.

    Grayscales the image, letterboxes it to ``target_size`` (preserving aspect
    ratio), blurs it to neutralize text variations, then computes a 64-bit
    dHash (8x8 grid).
    """
    try:
        with Image.open(REPO_ROOT / image_path) as img:
            # Step A: Convert to grayscale.
            img = img.convert("L")

            # Step B: Aspect-ratio-preserving padding (letterbox to target_size).
            # This prevents distortion from raw resizing which ruins hashes.
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            delta_w = target_size[0] - img.size[0]
            delta_h = target_size[1] - img.size[1]
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            img = ImageOps.expand(img, padding, fill=0)  # Padding with black.

            # Step C: Apply Gaussian blur (neutralizes text variations/meme text).
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Step D: Compute dHash (hash_size=8 creates an 8x8 grid -> 64-bit hash).
            hash_value = imagehash.dhash(img, hash_size=8)

            return str(hash_value)
    except Exception as e:
        # Handle corrupted images gracefully.
        return f"Error: {str(e)}"


def hamming_distance(hash1_str: str, hash2_str: str) -> int:
    """Calculate the Hamming distance between two hex dHash strings."""
    h1 = imagehash.hex_to_hash(hash1_str)
    h2 = imagehash.hex_to_hash(hash2_str)
    return h1 - h2  # The '-' operator in imagehash calculates Hamming distance.


def hash_to_binary_vector(hash_str: str) -> np.ndarray:
    """Convert a hex dHash string into a 64-bit binary feature vector."""
    if "Error" in hash_str:
        return np.zeros(64)  # Fallback for broken images.
    # Convert hex back to an ImageHash object, then flatten its boolean matrix.
    return imagehash.hex_to_hash(hash_str).hash.flatten().astype(int)


def visualize_cluster(dataframe: pd.DataFrame, cluster_id: int, max_images: int = 12) -> None:
    """Plot a grid of images belonging to a specific cluster (for manual inspection)."""
    cluster_df = dataframe[dataframe["cluster"] == cluster_id]

    if cluster_df.empty:
        print(f"Cluster {cluster_id} is empty or doesn't exist.")
        return

    print(f"\nVisualizing Cluster {cluster_id} (Total images: {len(cluster_df)})")

    # Limit display if cluster is too large.
    sample_df = cluster_df.head(max_images)
    num_images = len(sample_df)

    # Determine grid dimensions (up to 4 columns).
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))

    # Handle case where subplots returns a single axis object instead of an array.
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (_, row) in enumerate(sample_df.iterrows()):
        try:
            img = Image.open(REPO_ROOT / row["image_path"])
            axes[i].imshow(img)
            axes[i].axis("off")
            # Show file name without full path to keep it clean.
            axes[i].set_title(Path(row["image_path"]).name, fontsize=8)
        except Exception:
            axes[i].text(0.5, 0.5, "Error Loading", ha="center", va="center")
            axes[i].axis("off")

    # Hide any unused subplot squares.
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    # ----------------------------------------------------
    # Step 1: Build the image dataframe from for_clustering.
    # ----------------------------------------------------
    df = build_image_dataframe(FOR_CLUSTERING_DIR)
    print(f"Found a total of {len(df)} images in {FOR_CLUSTERING_DIR}.")

    # ----------------------------------------------------
    # Step 2: Compute perceptual dHashes for every image.
    # ----------------------------------------------------
    print("Computing optimized dHashes...")
    tqdm.pandas(desc="Hashing images")
    df["dhash"] = df["image_path"].progress_apply(compute_optimized_dhash)

    print("\nFirst few results:")
    print(df.head())

    # ----------------------------------------------------
    # Step 3: Convert hex dHashes into 64-bit binary vectors.
    # ----------------------------------------------------
    print("\nConverting hashes to feature vectors...")

    # Images that failed to hash (corrupted/unreadable files) all map to the same
    # hash string and must not be clustered together on that basis; exclude them
    # and leave them as their own singleton cluster (-1) instead.
    corrupted_mask = df["dhash"].str.startswith("Error")
    if corrupted_mask.any():
        print(
            f"Warning: {corrupted_mask.sum()} image(s) failed to hash and are excluded from "
            f"clustering: {df.loc[corrupted_mask, 'image_path'].tolist()}"
        )

    clusterable_df = df.loc[~corrupted_mask].reset_index(drop=True)
    X = np.stack(clusterable_df["dhash"].apply(hash_to_binary_vector).values)

    # ----------------------------------------------------
    # Step 4: Strict-threshold agglomerative clustering.
    # ----------------------------------------------------
    # Setting distance_threshold=6 means images will ONLY cluster if they differ
    # by 6 bits or fewer (out of 64). We use linkage='complete' to guarantee that
    # EVERY pair in a cluster satisfies this.
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Dynamically determine the number of clusters.
        distance_threshold=DHASH_DISTANCE_THRESHOLD,
        metric="manhattan",  # Behaves exactly like Hamming distance for binary arrays.
        linkage="complete",  # Guarantees strict cluster boundaries.
    )
    clusterable_df["cluster"] = clustering.fit_predict(X)

    df["cluster"] = -1
    df.loc[~corrupted_mask, "cluster"] = clusterable_df["cluster"].values

    # ----------------------------------------------------
    # Step 5: Summarize the resulting clusters.
    # ----------------------------------------------------
    cluster_counts = df["cluster"].value_counts()
    print(f"\nTotal precise clusters found: {len(cluster_counts)}")
    print("\nTop 20 largest clusters:")
    print(cluster_counts.head(20))

    singletons = (cluster_counts == 1).sum()
    print(f"\nNumber of unique/unmatched images (clusters of size 1): {singletons}")

    multi_image_clusters = cluster_counts[cluster_counts > 1]
    print(f"\n{len(multi_image_clusters)} clusters have more than one image.")
    print("Use visualize_cluster(df, cluster_id=<id>) to inspect a cluster, then add")
    print("it to TEMPLATE_MAPPING at the top of this script and rerun to label it.")
    print("Multi-image cluster ids (largest first):")
    print(list(multi_image_clusters.index))

    # ----------------------------------------------------
    # Step 6: Save the cluster assignments.
    # ----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_metadata_path = OUTPUT_DIR / "cluster_metadata.csv"
    df.to_csv(cluster_metadata_path, index=False)
    print(f"\nSaved cluster metadata for {len(df)} images to {cluster_metadata_path}")

    # If TEMPLATE_MAPPING has been filled in, also save the labeled subset.
    if TEMPLATE_MAPPING:
        target_clusters = list(TEMPLATE_MAPPING.keys())
        filtered_df = df[df["cluster"].isin(target_clusters)].copy()
        filtered_df["template_name"] = filtered_df["cluster"].map(TEMPLATE_MAPPING)

        labeled_clusters_path = OUTPUT_DIR / "labeled_clusters.csv"
        filtered_df.to_csv(labeled_clusters_path, index=False)
        print(f"Saved {len(filtered_df)} labeled images to {labeled_clusters_path}")
    else:
        print("\nTEMPLATE_MAPPING is empty, skipping labeled_clusters.csv.")


if __name__ == "__main__":
    main()
