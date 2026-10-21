import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from dataset_construction.scrapers.common import (
    DEFAULT_CONFIG_PATH,
    HEADERS,
    PROJECT_ROOT,
    SCRAPED_DIR,
    load_config,
)

IMGFLIP_DIR = SCRAPED_DIR / "templates" / "imgflip"
LOG_FILE = IMGFLIP_DIR / "download_log.txt"


def _log(message):
    """Print a message and append it to the download log file."""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} {message}\n")


def get_template_image(template_url):
    """Scrape the direct image URL for a template page, or ``None``."""
    response = requests.get(template_url, headers=HEADERS, timeout=20)
    if response.status_code != 200:
        return None

    body = BeautifulSoup(response.text, "html.parser")

    img = body.find("img", {"class": "base-img"})
    if img and img.get("src"):
        src = str(img["src"])
        if src.startswith("//"):
            src = "https:" + src
        return src

    return None


def download_templates(template_source, min_delay=30, max_delay=45, limit=None):
    """Download imgflip template images listed in a parquet file.

    ``template_source`` may be absolute or relative to the project root and must
    point to a parquet file containing a ``template_link`` column. Images are
    saved to ``scraped/templates/imgflip/``. When ``limit`` is set, only the
    first ``limit`` templates are processed.
    """
    template_path = Path(template_source)
    if not template_path.is_absolute():
        template_path = PROJECT_ROOT / template_path

    if not template_path.exists():
        raise FileNotFoundError(f"Template source not found: {template_path}")

    IMGFLIP_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(template_path)
    if "template_link" not in df.columns:
        raise ValueError(
            f"Parquet file {template_path} is missing a 'template_link' column."
        )

    if limit is not None:
        df = df.head(limit)

    already_exists = 0
    downloaded = 0
    failed = 0

    _log(f"Starting imgflip download for {len(df)} templates.")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        slug = row["template_link"].split("/")[-1]
        template_url = f"https://imgflip.com/memetemplate/{slug}"

        out_file = IMGFLIP_DIR / f"{slug}.jpg"

        # Skip if the file already exists.
        if out_file.exists():
            already_exists += 1
            continue

        # Prefer the scraped image URL, fall back to the conventional path.
        img_url = get_template_image(template_url)
        if not img_url:
            img_url = f"https://imgflip.com/s/meme/{slug}.jpg"

        try:
            response = requests.get(img_url, headers=HEADERS, timeout=20)
        except Exception as e:
            failed += 1
            _log(f"Failed to download {slug}: {e}")
            continue

        if response.status_code == 200:
            out_file.write_bytes(response.content)
            downloaded += 1
        else:
            failed += 1
            _log(f"Failed to download {slug}: status code {response.status_code}")

        time.sleep(random.uniform(min_delay, max_delay))

    _log("--- PROCESS COMPLETED ---")
    _log(f"Already existing: {already_exists}")
    _log(f"New downloaded: {downloaded}")
    _log(f"Failed: {failed}")


def run_from_config(config_path=DEFAULT_CONFIG_PATH):
    """Download imgflip templates using settings from the config file."""
    config = load_config(config_path)
    imgflip_config = config.get("imgflip", {})

    template_source = imgflip_config.get("template_source")
    if not template_source:
        raise ValueError(
            f"No 'template_source' defined under 'imgflip' in config: {config_path}"
        )

    min_delay = imgflip_config.get("min_delay", 30)
    max_delay = imgflip_config.get("max_delay", 45)
    limit = imgflip_config.get("count")

    download_templates(
        template_source, min_delay=min_delay, max_delay=max_delay, limit=limit
    )


if __name__ == "__main__":
    run_from_config()