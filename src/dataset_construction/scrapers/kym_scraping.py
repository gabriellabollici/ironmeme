import json
import random
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm

from dataset_construction.scrapers.common import (
    DEFAULT_CONFIG_PATH,
    HEADERS,
    PROJECT_ROOT,
    SCRAPED_DIR,
    load_config,
)

KYM_DIR = SCRAPED_DIR / "templates" / "kym"
LOG_FILE = KYM_DIR / "download_log.txt"


def _log(message):
    """Print a message and append it to the download log file."""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} {message}\n")


def load_json_to_df(json_path, orient_index=False):
    """Load a JSON file and convert it into a pandas DataFrame.

    Parameters:
    - json_path: path to JSON file
    - orient_index: if True, treat dict keys as index (for metadata)

    Returns:
    - pandas DataFrame
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if orient_index:
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.name = "source_image"
    else:
        df = pd.DataFrame(data)
        # Convert nested dicts/lists to JSON strings
        df = df.map(
            lambda x: json.dumps(x, ensure_ascii=False)
            if isinstance(x, (dict, list))
            else x
        )

    return df


def is_confirmed(details_str):
    """Return True if the entry's details mark it as a confirmed meme."""
    try:
        details = json.loads(details_str)
        return details.get("status") == "confirmed"
    except (json.JSONDecodeError, TypeError):
        return False


def download_image(image_url, save_path):
    """Download a single image to ``save_path``. Returns True on success."""
    try:
        response = requests.get(image_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        save_path.write_bytes(response.content)
        return True
    except Exception as e:
        _log(f"Error downloading {image_url}: {e}")
        return False


def download_templates(template_source, min_delay=35, max_delay=45, limit=None):
    """Download Know Your Meme template images listed in a JSON file.

    ``template_source`` may be absolute or relative to the project root. Only
    confirmed entries in the "Meme" category that have a ``template_image_url``
    are downloaded. Images are saved to ``scraped/templates/kym/``. When
    ``limit`` is set, only the first ``limit`` matching entries are downloaded.
    """
    source_path = Path(template_source)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    if not source_path.exists():
        raise FileNotFoundError(f"Template source not found: {source_path}")

    KYM_DIR.mkdir(parents=True, exist_ok=True)

    df = load_json_to_df(source_path)

    # Filter: category = Meme, status = confirmed, has a template image URL.
    df = df[df["category"] == "Meme"]
    df = df[df["details"].apply(is_confirmed)]
    df = df[df["template_image_url"].notna()]

    if limit is not None:
        df = df.head(limit)

    already_exists = 0
    downloaded = 0
    failed = 0

    _log(f"Starting KYM download for {len(df)} templates.")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_url = row["template_image_url"]
        title = str(row["title"])

        # Build a filesystem-safe name from the title.
        safe_name = "".join(c for c in title if c.isalnum() or c in (" ", "_")).rstrip()
        safe_name = safe_name.replace(" ", "_")

        parsed = urlparse(image_url)
        ext = Path(parsed.path).suffix or ".jpg"

        file_path = KYM_DIR / f"{safe_name}{ext}"

        # Skip if the file already exists.
        if file_path.exists():
            already_exists += 1
            continue

        if download_image(image_url, file_path):
            downloaded += 1
        else:
            failed += 1

        time.sleep(random.uniform(min_delay, max_delay))

    _log("--- PROCESS COMPLETED ---")
    _log(f"Already existing: {already_exists}")
    _log(f"New downloaded: {downloaded}")
    _log(f"Failed: {failed}")


def run_from_config(config_path=DEFAULT_CONFIG_PATH):
    """Download KYM templates using settings from the config file."""
    config = load_config(config_path)
    kym_config = config.get("kym", {})

    template_source = kym_config.get("template_source")
    if not template_source:
        raise ValueError(
            f"No 'template_source' defined under 'kym' in config: {config_path}"
        )

    min_delay = kym_config.get("min_delay", 35)
    max_delay = kym_config.get("max_delay", 45)
    limit = kym_config.get("count")

    download_templates(
        template_source, min_delay=min_delay, max_delay=max_delay, limit=limit
    )


if __name__ == "__main__":
    run_from_config()