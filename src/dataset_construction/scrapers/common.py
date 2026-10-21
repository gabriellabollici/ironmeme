"""Shared configuration and helpers for the meme scrapers.

Centralizes the project paths, HTTP headers and config loading so that the
individual scrapers (Reddit, imgflip, ...) stay consistent with each other.
"""

from pathlib import Path

import yaml
from paths import DATASETS_DIR, REPO_ROOT

# === PATHS ===

PROJECT_ROOT = REPO_ROOT

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

MEMES_DIR = DATASETS_DIR / "memes"
SCRAPED_DIR = MEMES_DIR / "scraped"

# === HEADERS ===

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load the scraping configuration from a YAML file.

    Expected format:

        reddit:
          enabled: true
          subreddits:
            - name: TrollXChromosomes
              count: 5000

        imgflip:
          enabled: true
          template_source: benchmark/source_datasets/memes/scraped/templates/meme_template_links.parquet
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    return config
