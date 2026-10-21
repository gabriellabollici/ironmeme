import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.inference import config

logger = logging.getLogger(__name__)


def get_image_output_dir(image_stem: str, model_name: str = "") -> Path:
    d = Path(config.OUTPUT_DIR) / (model_name or config.MODEL_SHORT_NAME) / image_stem
    d.mkdir(parents=True, exist_ok=True)
    return d


STEP_NAMES = {
    1: "01_social_identities",
    2: "02_viewpoints",
    3: "03_narrative_roles",
}


def write_step_output(
    step: int,
    image_stem: str,
    parsed_json: Optional[Dict[str, Any]],
    raw_text: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    model_name: str = "",
) -> None:
    tag = STEP_NAMES.get(step, f"{step:02d}_unknown")
    out_dir = get_image_output_dir(image_stem, model_name=model_name)

    json_payload: Dict[str, Any] = {
        "step": step,
        "step_name": tag,
        "image_stem": image_stem,
        "model": model_name or "default",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parsed_json": parsed_json,
        "raw_text": raw_text,
    }
    if extra_metadata:
        json_payload["metadata"] = extra_metadata

    json_path = out_dir / f"{tag}.json"
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", json_path)

    txt_path = out_dir / f"{tag}.txt"
    txt_path.write_text(raw_text, encoding="utf-8")
    logger.info("Wrote %s", txt_path)


def write_final_output(
    image_stem: str,
    all_steps: List[Dict[str, Any]],
    model_name: str = "",
) -> None:
    out_dir = get_image_output_dir(image_stem, model_name=model_name)

    merged: Dict[str, Any] = {
        "image_stem": image_stem,
        "model": model_name or "default",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": all_steps,
    }

    for s in all_steps:
        if s.get("parsed_json"):
            merged[s["step_name"]] = s["parsed_json"]

    path = out_dir / "final_output.json"
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    logger.info("Wrote final output: %s", path)


def setup_logging(log_file: Optional[str] = None) -> None:
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )
