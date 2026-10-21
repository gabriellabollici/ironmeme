import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.inference.config import PROMPTS_DIR, BENCHMARK_JSON
from dataset_construction.labels import gt_viewpoint_at

logger = logging.getLogger(__name__)


def load_prompt(step: int) -> Tuple[str, str]:
    filenames = {
        1: "01_social_identities.txt",
        2: "02_viewpoints.txt",
        3: "03_narrative_roles.txt",
    }
    path = Path(PROMPTS_DIR) / filenames[step]
    text = path.read_text(encoding="utf-8")

    system = ""
    user = ""

    lines = text.split("\n")
    section = None
    buf: List[str] = []

    for line in lines:
        if line.upper().startswith("SYSTEM:"):
            section = "system"
            buf = [line[len("SYSTEM:"):].strip()]
        elif line.upper().startswith("USER:"):
            section = "user"
            if buf:
                system = " ".join(buf).strip()
            buf = [line[len("USER:"):].strip()]
        elif section is not None:
            buf.append(line)

    if section == "user":
        user = " ".join(buf).strip()
    elif section == "system":
        system = " ".join(buf).strip()

    return system, user


def format_user_prompt(
    template: str,
    social_identities: Optional[List[str]] = None,
    viewpoints_context: Optional[List[Dict[str, str]]] = None,
) -> str:
    if social_identities is not None:
        ids_str = json.dumps(social_identities, ensure_ascii=False)
    else:
        ids_str = "[]"

    if viewpoints_context is not None:
        vp_str = json.dumps(viewpoints_context, ensure_ascii=False)
    else:
        vp_str = "[]"

    result = template.replace("{social_identities}", ids_str)
    result = result.replace("{viewpoints_context}", vp_str)
    return result


def _parse_gt_string(val: Any) -> Optional[list]:
    """Safely parse a GT string like "['a', 'b']" into a Python list."""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if not isinstance(val, str):
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            logger.warning("Could not parse GT value: %s", val)
            return None


_benchmark_cache: Optional[Dict[str, Dict[str, Any]]] = None


def load_benchmark_lookup() -> Dict[str, Dict[str, Any]]:
    """Load the benchmark JSON and return a dict keyed by image_filename (cached)."""
    global _benchmark_cache
    if _benchmark_cache is not None:
        return _benchmark_cache

    path = Path(BENCHMARK_JSON)
    if not path.is_file():
        logger.warning("Benchmark JSON not found: %s", BENCHMARK_JSON)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup: Dict[str, Dict[str, Any]] = {}
    for entry in data:
        fname = entry.get("image_filename", "")
        if fname:
            lookup[fname] = entry
    logger.info("Loaded %d benchmark entries from %s", len(lookup), BENCHMARK_JSON)
    _benchmark_cache = lookup
    return lookup


def get_gt_context(
    image_path: str,
    benchmark_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[List[str]], Optional[List[Dict[str, str]]]]:
    """Return (gt_social_identities, gt_viewpoints) for an image.

    gt_social_identities is a list of identity strings, e.g. ["Left wing", "Conservative"].
    gt_viewpoints is a list of dicts like [{"identity": "...", "viewpoint": "ingroup"}, ...].
    A blank GT viewpoint is filled from another annotator when possible
    (labels.gt_viewpoint_at); if it stays invalid the identity is passed
    without a viewpoint (the step-3 prompt says "exactly 1 viewpoint (if
    provided)").
    """
    fname = Path(image_path).name

    entry = benchmark_lookup.get(fname)
    if entry is None:
        logger.debug("No benchmark entry for %s", fname)
        return None, None

    raw_ids = _parse_gt_string(entry.get("social_identity_gt"))
    raw_vps = _parse_gt_string(entry.get("viewpoint_gt"))

    if not raw_ids:
        return None, None

    gt_social_identities = [str(x) for x in raw_ids]

    gt_viewpoints: Optional[List[Dict[str, str]]] = None
    if raw_vps and len(raw_vps) == len(raw_ids):
        gt_viewpoints = []
        for pos, (identity, vp_raw) in enumerate(zip(raw_ids, raw_vps)):
            vp_label = gt_viewpoint_at(fname, pos, vp_raw)
            item = {"identity": str(identity)}
            if vp_label is not None:
                item["viewpoint"] = vp_label
            gt_viewpoints.append(item)
    else:
        logger.warning(
            "Viewpoint GT length mismatch for %s (ids=%d, vps=%d)",
            fname, len(raw_ids), len(raw_vps) if raw_vps else 0,
        )

    return gt_social_identities, gt_viewpoints
