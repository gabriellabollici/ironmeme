import logging
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache

from PIL import Image

from experiments.inference.config import IMAGES_DIR, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def discover_images(images_dir: str = IMAGES_DIR) -> list[str]:
    p = Path(images_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    files = sorted(
        str(f)
        for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info("Discovered %d images in %s", len(files), images_dir)
    return files


@lru_cache(maxsize=256)
def load_image_metadata(image_path: str) -> Dict[str, Optional[str]]:
    img = Image.open(image_path)
    img.load()
    meta = {
        "path": image_path,
        "filename": Path(image_path).name,
        "stem": Path(image_path).stem,
        "format": img.format,
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
    }
    logger.debug("Loaded image metadata: %s", meta)
    return meta


def clear_image_cache() -> None:
    load_image_metadata.cache_clear()
