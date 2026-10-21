"""Suggest a template for benchmark memes that don't have one yet.

``benchmark/sources/templates.csv`` holds the hand-curated template name and
description of every benchmark meme (``curated=yes``) and is never overwritten.
For memes missing from it (e.g. after new annotations), this embeds the image
with the same DINO ViT model used for the template images, finds the closest
template with FAISS (top-1 cosine similarity) and appends a row with
``curated=no``: ``template_name`` is the cleaned template image filename and
``template_description`` comes from KYM (``kym.json`` or the scraped
``template_descriptions.json``). Review those rows and set ``curated=yes``.
"""

import json
import os
import re
from pathlib import Path

# torch/faiss bundle their own OpenMP runtime; on macOS a duplicate copy can
# segfault. Tolerate the duplicate and force a single thread before importing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

from dataset_construction.build_benchmark import benchmark_rows
from paths import DATASETS_DIR, TEMPLATES_TABLE

faiss.omp_set_num_threads(1)
torch.set_num_threads(1)

TEMPLATES_PATH = DATASETS_DIR / "embedding_datasets/templates.json"
KYM_SOURCE_PATH = DATASETS_DIR / "memes/template_sources/kym.json"
TEMPLATE_DESCRIPTIONS_PATH = DATASETS_DIR / "memes/template_sources/template_descriptions.json"
# Every benchmark meme went through the human annotation process, so its
# canonical image copy lives under ANNOTATED_IMAGES_DIR; that root is checked
# first so a basename that coincidentally collides with an unrelated image in
# one of the raw scrape/reuse roots doesn't silently shadow it. The other two
# roots are only a fallback for the handful of rows without an annotated copy
# on disk.
IMAGE_ROOTS = [
    DATASETS_DIR / "annotation_results/annotated_images",
    DATASETS_DIR / "memes/reused_benchmarks",
    DATASETS_DIR / "memes/scraped/reddit",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def build_name_to_path() -> dict[str, str]:
    """Map image basename -> absolute path, preferring earlier IMAGE_ROOTS on collision."""
    name_to_path: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for root in IMAGE_ROOTS:
        for path in Path(root).rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                if path.name in name_to_path:
                    collisions.setdefault(path.name, [name_to_path[path.name]]).append(str(path))
                    continue
                name_to_path[path.name] = str(path)
    if collisions:
        print(
            f"Warning: {len(collisions)} filename(s) exist in more than one IMAGE_ROOTS "
            f"directory; kept the copy from the highest-priority root for each, e.g. "
            f"{next(iter(collisions.items()))}"
        )
    return name_to_path


def clean_template_name(filename: str) -> str:
    """Turn a template image filename into a readable name.

    Drops the file extension and replaces hyphens/underscores with spaces.
    Example: ``drake-hotline-bling.png`` -> ``drake hotline bling``.
    """
    stem = Path(filename).stem
    return stem.replace("-", " ").replace("_", " ").strip()


def load_kym_descriptions() -> dict[str, str]:
    """Build a lookup from KYM template filename stem -> description.

    KYM template image filenames are derived from the entry's URL slug (the last
    segment of ``url``), so we key the lookup on that slug (lowercased for
    robustness). The description lives in each record's ``meta.description``.
    """
    with open(KYM_SOURCE_PATH) as f:
        records = json.load(f)

    descriptions: dict[str, str] = {}
    for record in records:
        slug = record.get("url", "").rstrip("/").rsplit("/", 1)[-1].lower()
        description = (record.get("meta") or {}).get("description")
        if slug and description:
            descriptions[slug] = description
    return descriptions


def load_template_descriptions() -> dict[str, str]:
    """Load the template filename -> description map scraped from KYM.

    Scraped once from KYM for templates that have no description in
    ``kym.json`` (imgflip templates and KYM entries whose filename differs
    from their URL slug); the scraper is no longer part of the repo, and
    ``scrapers/kym_full_descriptions.py`` extends the file.
    """
    if not TEMPLATE_DESCRIPTIONS_PATH.exists():
        return {}
    return json.loads(TEMPLATE_DESCRIPTIONS_PATH.read_text(encoding="utf-8"))


def lookup_description(template_path: str, kym_descriptions: dict[str, str]) -> str:
    """Return the description for a matched template, or an empty string.

    KYM-sourced templates are looked up in ``kym.json`` by URL slug; any
    template (kym or imgflip) present in the scraped ``template_descriptions``
    map is looked up there by filename. Templates with no description return
    ``""``.
    """
    path = Path(template_path)
    if path.parts and path.parts[0] == "kym":
        return kym_descriptions.get(path.stem.lower(), "")
    return ""


def reverse_map_name(filename: str) -> str | None:
    """Recover the original filename of a renamed cluster_* benchmark image."""
    patterns = [
        re.compile(r"for_clustering_2_MAMI_(\d+)\.(\w+)"),
        re.compile(r"for_clustering_2_CliME_image_(\d+)\.(\w+)"),
        re.compile(r"for_clustering_2_Constraint_memes_(\d+)\.(\w+)"),
        re.compile(r"for-clustering_([a-z0-9]+)\.(\w+)"),
        re.compile(r"(?:images_|test_images_)(\d+)\.(\w+)"),
        re.compile(r"meme_(\d+)_score_[\d.]+\.(\w+)"),
    ]
    for pattern in patterns:
        match = pattern.search(filename)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return f"{groups[0]}.{groups[1]}"
    return None


class ImageDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            pixel_values = self.processor(
                images=Image.open(self.paths[idx]).convert("RGB"),
                return_tensors="pt",
            )["pixel_values"].squeeze(0)
            return pixel_values, True
        except Exception as e:
            print(f"Warning: failed to load {self.paths[idx]} ({e}); leaving unmatched")
            return torch.zeros((3, 224, 224)), False


TEMPLATE_COLUMNS = ["id", "image_filename", "template_name", "template_description", "curated"]


def main() -> None:
    curated = (
        pd.read_csv(TEMPLATES_TABLE, dtype=str, keep_default_na=False)
        if TEMPLATES_TABLE.exists()
        else pd.DataFrame(columns=TEMPLATE_COLUMNS)
    )
    known = set(curated["id"])
    rows = [row for row in benchmark_rows() if row["id"] not in known]
    if not rows:
        print(f"Every benchmark meme already has an entry in {TEMPLATES_TABLE}; nothing to match.")
        return
    benchmark = pd.DataFrame(rows)

    name_to_path = build_name_to_path()
    image_paths = []
    for filename in benchmark["source_filename"]:
        path = name_to_path.get(filename) or name_to_path.get(
            reverse_map_name(filename)
        )
        if path is None:
            raise ValueError(f"No file found on disk for {filename!r}")
        image_paths.append(path)

    templates = pd.read_json(TEMPLATES_PATH)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
    model = ViTModel.from_pretrained("facebook/dino-vitb16")
    model.to(device).eval()

    loader = DataLoader(
        ImageDataset(image_paths, processor), batch_size=32, shuffle=False, num_workers=0
    )
    all_embeddings = []
    all_ok = []
    with torch.no_grad():
        for batch, ok in tqdm(loader, desc="Embedding benchmark images"):
            embeddings = model(pixel_values=batch.to(device)).last_hidden_state[:, 0, :]
            all_embeddings.extend(embeddings.cpu().numpy())
            all_ok.extend(ok.tolist())

    meme_features = np.asarray(all_embeddings, dtype="float32")
    failed_mask = np.array([not ok for ok in all_ok])
    if failed_mask.any():
        print(
            f"Warning: {failed_mask.sum()} benchmark image(s) failed to load and will be "
            f"left without a template match: {benchmark['source_filename'][failed_mask].tolist()}"
        )

    template_features = np.stack(templates["VIT_embeddings"].values).astype("float32")
    index = faiss.index_factory(template_features.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(template_features)
    index.add(template_features)
    faiss.normalize_L2(meme_features)
    scores, indices = index.search(meme_features, 1)

    template_titles = templates["filenames"].to_numpy()
    template_paths = templates["full_image_path"].to_numpy()
    kym_descriptions = load_kym_descriptions()
    template_descriptions = load_template_descriptions()
    matched_indices = indices[:, 0]

    suggested = pd.DataFrame({
        "id": benchmark["id"],
        "image_filename": benchmark["source_filename"],
        "template_name": [clean_template_name(title) for title in template_titles[matched_indices]],
        "template_description": [
            template_descriptions.get(Path(path).name, lookup_description(path, kym_descriptions))
            for path in template_paths[matched_indices]
        ],
        "curated": "no",
    })
    suggested.loc[failed_mask, ["template_name", "template_description"]] = ""

    table = pd.concat([curated, suggested], ignore_index=True).sort_values("id")
    table[TEMPLATE_COLUMNS].to_csv(TEMPLATES_TABLE, index=False)
    print(f"Added {len(suggested)} suggested template(s) (curated=no) to {TEMPLATES_TABLE}")
    print(pd.Series(scores[:, 0], name="similarity_score").describe().to_string())


if __name__ == "__main__":
    main()
