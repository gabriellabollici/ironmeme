"""Extract the meme text (OCR) with EasyOCR.

The ``ocr`` column of ``benchmark/sources/meme_metadata.csv`` was produced
outside this repository and is the reference; this script only approximates
it (on a 21-meme sample spanning every source: mean similarity ~0.85, see the
README). It therefore never overwrites stored OCR:

    # compare EasyOCR with the stored OCR on a sample of memes
    python -m dataset_construction.extract_ocr --compare 21

    # OCR every benchmark meme into a separate CSV
    python -m dataset_construction.extract_ocr --out ocr_easyocr.csv

    # add OCR rows for memes missing from meme_metadata.csv (e.g. new annotations);
    # their ``origin`` is left empty and must be filled in by hand
    python -m dataset_construction.extract_ocr --fill-missing

Needs the ``ocr`` dependency group (``uv sync --group ocr``). EasyOCR downloads
its models on first use; if that fails with CERTIFICATE_VERIFY_FAILED (python.org
builds on macOS), point SSL_CERT_FILE at ``python -m certifi``.

Each meme is read from its original annotated image (located by
build_benchmark), so this works before the benchmark is built. Text regions
are grouped into paragraphs and joined with single spaces, matching the
one-line format of the stored OCR.
"""
import argparse
import difflib
import random
from functools import lru_cache
from pathlib import Path

import pandas as pd

from dataset_construction.build_benchmark import benchmark_rows, find_source_image
from paths import MEME_METADATA_TABLE

METADATA_COLUMNS = ["id", "image_filename", "origin", "ocr"]


@lru_cache(maxsize=1)
def _reader():
    import easyocr  # optional dependency (ocr group); imported only when OCR is run

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def extract_text(image_path: Path) -> str:
    """OCR one image into a single line of text."""
    paragraphs = _reader().readtext(str(image_path), detail=0, paragraph=True)
    return " ".join(p.strip() for p in paragraphs if p.strip())


def read_metadata() -> pd.DataFrame:
    if MEME_METADATA_TABLE.exists():
        return pd.read_csv(MEME_METADATA_TABLE, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=METADATA_COLUMNS)


def ocr_rows(rows: list[dict]) -> pd.DataFrame:
    records = []
    for i, row in enumerate(rows, start=1):
        image = find_source_image(row["original_dataset"], row["source_filename"])
        records.append({"id": row["id"], "image_filename": row["source_filename"], "ocr": extract_text(image)})
        if i % 50 == 0 or i == len(rows):
            print(f"  OCR {i}/{len(rows)}")
    return pd.DataFrame(records, columns=["id", "image_filename", "ocr"])


def compare(n: int, seed: int = 0) -> None:
    """OCR ``n`` memes (stratified by origin) and report similarity to the stored OCR."""
    stored = read_metadata().set_index("id")
    rows = [r for r in benchmark_rows() if r["id"] in stored.index]
    by_origin: dict[str, list[dict]] = {}
    for row in rows:
        by_origin.setdefault(stored.loc[row["id"], "origin"], []).append(row)
    rng = random.Random(seed)
    per_origin = max(1, n // len(by_origin))
    sample = [r for origin in sorted(by_origin)
              for r in rng.sample(by_origin[origin], min(per_origin, len(by_origin[origin])))]

    similarities, exact = [], 0
    for row in sample:
        reference = stored.loc[row["id"], "ocr"]
        text = extract_text(find_source_image(row["original_dataset"], row["source_filename"]))
        similarity = difflib.SequenceMatcher(None, text, reference).ratio()
        similarities.append(similarity)
        exact += text == reference
        print(f"{similarity:.2f}  {row['source_filename']}\n      stored:  {reference[:90]!r}\n      easyocr: {text[:90]!r}")
    print(f"\n{len(sample)} memes: exact matches {exact}, mean similarity {sum(similarities) / len(similarities):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compare", type=int, metavar="N", help="Compare with the stored OCR on N sampled memes.")
    mode.add_argument("--out", type=Path, help="Write OCR for every benchmark meme to this CSV.")
    mode.add_argument("--fill-missing", action="store_true",
                      help=f"Append OCR rows for memes missing from {MEME_METADATA_TABLE.name} (never overwrites).")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
    elif args.out:
        ocr_rows(benchmark_rows()).to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    else:
        metadata = read_metadata()
        missing = [r for r in benchmark_rows() if r["id"] not in set(metadata["id"])]
        if not missing:
            print(f"Every benchmark meme already has OCR in {MEME_METADATA_TABLE}; nothing to do.")
            return
        new = ocr_rows(missing).assign(origin="")[METADATA_COLUMNS]
        pd.concat([metadata, new], ignore_index=True).sort_values("id").to_csv(MEME_METADATA_TABLE, index=False)
        print(f"Added OCR for {len(new)} meme(s) to {MEME_METADATA_TABLE}; fill in their origin by hand.")


if __name__ == "__main__":
    main()
