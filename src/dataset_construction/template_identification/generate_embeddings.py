import argparse
import os

# torch bundles its own OpenMP runtime; on macOS a duplicate copy can segfault
# during inference. Tolerate the duplicate and force a single OpenMP thread
# before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from paths import DATASETS_DIR

# Keep torch single-threaded to avoid the OpenMP runtime clash above.
torch.set_num_threads(1)

TEMPLATES_DIR = DATASETS_DIR / "memes/scraped/templates"
EMBEDDING_DATASETS_DIR = DATASETS_DIR / "embedding_datasets"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

# Datasets to embed: name -> (root directory, id column, origin prefix).
# Only the template images: match_benchmark_templates embeds the benchmark
# memes itself and searches them against these.
DATASETS = {
    "templates": (TEMPLATES_DIR, "template_id", ""),
}

_processor = None
_model = None
_device = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device


def get_model_and_processor():
    """Lazily load the ViT model/processor so `--help` and dry runs stay fast."""
    global _processor, _model
    if _model is None:
        _processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
        _model = ViTModel.from_pretrained("facebook/dino-vitb16")
        _model.to(get_device()).eval()
    return _model, _processor


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
            print(f"Warning: failed to load {self.paths[idx]} ({e}); excluding from embeddings")
            return torch.zeros((3, 224, 224)), False


def add_vit_embeddings(
    df: pd.DataFrame,
    root: Path,
    batch_size: int = 16,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Add a ``VIT_embeddings`` column with DINO ViT [CLS] embeddings for each image."""
    model, processor = get_model_and_processor()
    device = get_device()

    image_paths = [root / path for path in df["full_image_path"]]
    dataset = ImageDataset(image_paths, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_embeddings = []
    all_ok = []
    with torch.no_grad():
        for batch, ok in tqdm(loader, desc="Extracting ViT Features", leave=False):
            batch = batch.to(device)
            outputs = model(pixel_values=batch)

            # Pull the [CLS] token for the entire batch
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)
            all_ok.extend(ok.tolist())

    df = df.reset_index(drop=True)
    df["VIT_embeddings"] = list(all_embeddings)

    failed = [not ok for ok in all_ok]
    if any(failed):
        bad_paths = df.loc[failed, "full_image_path"].tolist()
        print(
            f"Warning: {sum(failed)} image(s) failed to load and were dropped "
            f"from this dataset: {bad_paths}"
        )
        df = df.loc[[not f for f in failed]].reset_index(drop=True)
    return df


def build_image_dataframe(root: Path, origin_prefix: str = "") -> pd.DataFrame:
    """Build a dataframe of all images in the given directory.

    Columns:
        filenames: the file name including the extension, e.g. ``memes_13.png``.
        full_image_path: the path relative to ``root``, e.g. ``Constraint22/memes_13.png``.
        origin: the top-level subfolder name, optionally prefixed, e.g. ``Constraint22``
            or ``r/ClimateMemes``.
    """
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative_path = path.relative_to(root)
            records.append(
                {
                    "filenames": path.name,
                    "full_image_path": relative_path.as_posix(),
                    "origin": f"{origin_prefix}{relative_path.parts[0]}",
                }
            )
    return pd.DataFrame(records, columns=["filenames", "full_image_path", "origin"])


def load_existing(output_path: Path) -> pd.DataFrame | None:
    if not output_path.exists():
        return None
    existing = pd.read_json(output_path)
    if "VIT_embeddings" not in existing.columns:
        return None
    return existing


def process_dataset(
    name: str,
    root: Path,
    id_col: str,
    origin_prefix: str,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    chunk_size: int,
    limit: int | None = None,
) -> None:
    """Embed every image under ``root`` that isn't already in ``output_path``.

    Processes the remaining images in ``chunk_size``-sized chunks, saving the
    combined JSON dataset to disk after every chunk so progress is never lost
    and the run can safely be interrupted and resumed.
    """
    full_df = build_image_dataframe(root, origin_prefix=origin_prefix)
    if limit is not None:
        full_df = full_df.head(limit).reset_index(drop=True)

    existing = load_existing(output_path)
    if existing is not None and len(existing) > 0:
        done_paths = set(existing["full_image_path"])
        todo_df = full_df[~full_df["full_image_path"].isin(done_paths)].reset_index(drop=True)
    else:
        todo_df = full_df

    print(f"[{name}] {len(full_df)} total images, {len(todo_df)} remaining to embed")

    if len(todo_df) == 0:
        print(f"[{name}] nothing to do, already up to date")
        return

    combined = existing
    num_chunks = (len(todo_df) + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        chunk = todo_df.iloc[start : start + chunk_size].reset_index(drop=True)
        print(f"[{name}] chunk {chunk_idx + 1}/{num_chunks} ({len(chunk)} images)")
        chunk = add_vit_embeddings(chunk, root, batch_size=batch_size, num_workers=num_workers)

        if combined is None:
            combined = chunk
        else:
            combined = pd.concat([combined, chunk], ignore_index=True)

        combined[id_col] = range(len(combined))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_json(output_path, orient="records", indent=2)
        print(f"[{name}] saved {len(combined)}/{len(full_df)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DINO ViT embeddings for the template images."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Which datasets to process (default: all of them).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker processes for image decoding."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of images to embed between JSON checkpoints (saves progress incrementally).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of images per dataset (useful for smoke tests).",
    )
    args = parser.parse_args()

    EMBEDDING_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {get_device()}")

    for name in args.datasets:
        root, id_col, origin_prefix = DATASETS[name]
        output_path = EMBEDDING_DATASETS_DIR / f"{name}.json"
        process_dataset(
            name=name,
            root=root,
            id_col=id_col,
            origin_prefix=origin_prefix,
            output_path=output_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()