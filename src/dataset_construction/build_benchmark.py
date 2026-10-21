"""Build the released IRONMeme benchmark from the raw annotations.

Writes only the benchmark itself, ``benchmark/benchmark_dataset/``:

    ironmeme_benchmark.json   one record per meme (see COLUMNS)
    ironmeme_benchmark.csv    the same records as CSV
    images/                   the meme images, renamed to ``<row number>.<ext>``

Everything in between happens in memory:

1. Aggregate the raw per-annotator exports into one row per meme, derive the
   consensus ground truth and agreement level
   (``annotation_formatting.annotation_postprocessing``).
2. Keep the memes with a consensus ground truth, minus ``EXCLUDED_IDS``; the
   original annotation sets come first, then ``LATE_SETS``, each in annotation
   order.
3. Apply ``IDENTITY_RENAMES`` to the ground truth and split it into the
   ``social_identity_gt`` / ``viewpoint_gt`` / ``narrative_roles_gt`` columns.
4. Join the hand-curated / external per-meme data from ``benchmark/sources/``
   (template name and description, OCR, origin; keyed by annotation id).
5. Copy each meme's original image file under its new name.

Usage (from the repo root):
    python -m dataset_construction.build_benchmark [--out DIR]
"""
import argparse
import json
import re
import shutil
from functools import lru_cache
from pathlib import Path

import pandas as pd

from dataset_construction.annotation_formatting.annotation_postprocessing import (
    LATE_SETS,
    aggregate_to_wide,
    load_raw_annotations,
)
from paths import (
    ANNOTATED_IMAGES_DIR,
    BENCHMARK_DATASET_DIR,
    DATASETS_DIR,
    MEME_METADATA_TABLE,
    RAW_ANNOTATIONS_DIR,
    TEMPLATES_TABLE,
)

# Memes with a consensus ground truth that were removed by hand during
# template curation (marked "remove this one" / "not a template").
EXCLUDED_IDS = {
    "ac3d124c-19aa-495a-8651-2179d15c8b6b",  # 2478.jpg (DRY RUN 2)
    "09676c6d-0602-4368-8f21-0f54c3314460",  # memes_4532.png (SET 8)
}

# Identity labels renamed after annotation ("unify corporativist label").
IDENTITY_RENAMES = {"corporativist": "corporativist/capitalist"}

# ``original_dataset`` -> its folder under ANNOTATED_IMAGES_DIR.
ORIGINAL_DATASET_TO_FOLDER = {
    "DRY RUN 1": "Pilot1",
    "DRY RUN 2": "Pilot2",
    "SET 1": "Set1",
    "SET 2": "Set2",
    "SET 3": "Set3",
    "SET 4": "set4",
    "SET 5_1": "set5.1.",
    "SET 5_2": "set5.2",
    "SET 6": "set6",
    "SET 7": "set7",
    "SET 8": "set8",
    "SET 9": "set9",
    "SET 10": "set10",
    "SET 11": "set 11",
    "SET 12": "set 12",
}

# Where to look for an image that has no annotated copy (a few SET 12 memes):
# the scraped / reused meme pools it was sampled from.
FALLBACK_IMAGE_ROOTS = [
    DATASETS_DIR / "memes" / "reused_benchmarks",
    DATASETS_DIR / "memes" / "scraped" / "reddit",
]

COLUMNS = [
    "image_filename", "ground_truth", "social_identity_gt", "viewpoint_gt", "narrative_roles_gt",
    "agreement_type", "origin", "template_name", "template_description", "ocr",
]

_TRIPLE_RE = re.compile(r"^(.*?)-(in_group|out_group)-(.*)$")


def split_triple(triple: str) -> tuple[str, str, str]:
    """``"sex-worker-out_group-victim"`` -> ``("sex-worker", "out_group", "victim")``.

    Anchored on the viewpoint token because identities may contain hyphens;
    a blank viewpoint (``"feminist--villain"``) gives ``("feminist", "", "villain")``.
    """
    match = _TRIPLE_RE.match(triple)
    if match:
        return match.groups()
    identity, viewpoint, role = triple.rsplit("-", 2)
    return identity, viewpoint, role


def rename_identities(triples: list[str]) -> list[str]:
    renamed = []
    for triple in triples:
        identity, viewpoint, role = split_triple(triple)
        renamed.append(f"{IDENTITY_RENAMES.get(identity, identity)}-{viewpoint}-{role}")
    return renamed


def read_source_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("id")


@lru_cache(maxsize=1)
def _fallback_image_index() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in FALLBACK_IMAGE_ROOTS:
        for path in root.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, []).append(path)
    return index


def find_source_image(original_dataset: str, filename: str) -> Path:
    annotated = ANNOTATED_IMAGES_DIR / ORIGINAL_DATASET_TO_FOLDER[original_dataset] / filename
    if annotated.is_file():
        return annotated
    candidates = _fallback_image_index().get(filename, [])
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{filename!r} ({original_dataset}) has no annotated copy and "
            f"{len(candidates)} candidates in {[str(r) for r in FALLBACK_IMAGE_ROOTS]}"
        )
    return candidates[0]


def load_annotations() -> pd.DataFrame:
    """One row per annotated meme with consensus ``ground_truth`` and ``agreement_level``."""
    return aggregate_to_wide(load_raw_annotations(RAW_ANNOTATIONS_DIR))


def benchmark_rows() -> list[dict]:
    """The benchmark memes in output order, before the per-meme source data is joined.

    Each row has ``id``, ``source_filename`` (the annotated image's original
    name), ``original_dataset``, ``agreement_level``, ``ground_truth`` (renamed triples)
    and ``image_filename`` (its name in the released benchmark).
    """
    annotations = load_annotations()
    selected = annotations[annotations["agreement_level"].notna() & ~annotations["id"].isin(EXCLUDED_IDS)]
    is_late = selected["original_dataset"].isin(LATE_SETS)
    ordered = pd.concat([selected[~is_late], selected[is_late]])

    rows = []
    for number, record in enumerate(ordered.to_dict("records"), start=1):
        extension = record["image_filename"].rsplit(".", 1)[1]
        rows.append({
            "id": record["id"],
            "source_filename": record["image_filename"],
            "original_dataset": record["original_dataset"],
            "agreement_level": record["agreement_level"],
            "ground_truth": rename_identities(record["ground_truth"]),
            "image_filename": f"{number}.{extension}",
        })
    return rows


def original_to_clean_filenames() -> dict[str, str]:
    """Annotated image filename -> its filename in the released benchmark."""
    return {row["source_filename"]: row["image_filename"] for row in benchmark_rows()}


def build_records(rows: list[dict]) -> list[dict]:
    templates = read_source_table(TEMPLATES_TABLE)
    metadata = read_source_table(MEME_METADATA_TABLE)
    missing = [row["source_filename"] for row in rows if row["id"] not in templates.index or row["id"] not in metadata.index]
    if missing:
        raise KeyError(f"{len(missing)} benchmark meme(s) missing from {TEMPLATES_TABLE.name} / "
                       f"{MEME_METADATA_TABLE.name}, e.g. {missing[:5]}")

    records = []
    for row in rows:
        parts = [split_triple(t) for t in row["ground_truth"]]
        template, meta = templates.loc[row["id"]], metadata.loc[row["id"]]
        records.append({
            "image_filename": row["image_filename"],
            "ground_truth": str(row["ground_truth"]),
            "social_identity_gt": str([p[0] for p in parts]),
            "viewpoint_gt": str([p[1] for p in parts]),
            "narrative_roles_gt": str([p[2] for p in parts]),
            "agreement_type": row["agreement_level"],
            "origin": meta["origin"],
            "template_name": template["template_name"],
            "template_description": template["template_description"],
            "ocr": meta["ocr"],
        })
    return records


def build(out_dir: Path = BENCHMARK_DATASET_DIR) -> None:
    rows = benchmark_rows()
    records = build_records(rows)

    # Build next to the target, then swap, so a failed run never leaves a
    # half-written benchmark behind.
    staging = out_dir.with_name(out_dir.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "images").mkdir(parents=True)
    for row in rows:
        source = find_source_image(row["original_dataset"], row["source_filename"])
        shutil.copyfile(source, staging / "images" / row["image_filename"])
    (staging / "ironmeme_benchmark.json").write_text(json.dumps(records, indent=2) + "\n")
    pd.DataFrame(records, columns=COLUMNS).to_csv(staging / "ironmeme_benchmark.csv", index=False)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)
    print(f"Wrote {len(records)} memes to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=BENCHMARK_DATASET_DIR,
                        help=f"Output directory (default: {BENCHMARK_DATASET_DIR}).")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
