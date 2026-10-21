#!/usr/bin/env python3
"""Format pipeline output into a flat CSV, matching the notebook's preprocessing.

Reads per-image ``final_output.json`` files produced by the pipeline and the
benchmark JSON to produce a single CSV with normalised GT and prediction columns.

Usage:
    python -m experiments.evaluation.format_predictions [--output PATH]
"""
import argparse
import ast
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


import pandas as pd

from experiments.inference.config import OUTPUT_DIR, BENCHMARK_JSON, MODEL_SHORT_NAME
from dataset_construction.labels import (  # noqa: F401  (APPROVED_IDENTITIES re-exported for older callers)
    APPROVED_IDENTITIES, canon_identity, canon_role, canon_viewpoint, coerce_to_list,
    gt_viewpoint_at, load_aliases, set_aliases,
)
from experiments.parsing import try_parse_json, validate_list_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identity normalisation -- see labels.py for the shared invalid-label rule
# ---------------------------------------------------------------------------
def normalize_and_map_identity(raw_identity: str) -> Optional[str]:
    """Map a raw identity string to the approved taxonomy (None if invalid)."""
    return canon_identity(raw_identity)


# ---------------------------------------------------------------------------
# Strip hyphens (matches notebook cell 3)
# ---------------------------------------------------------------------------
def strip_hyphens_from_data(val: Any) -> Any:
    if pd.isna(val):
        return val
    val_str = str(val)
    val_str = val_str.replace("-", " ")
    val_str = val_str.replace("_", "")
    return val_str


# ---------------------------------------------------------------------------
# Parse GT column (matches notebook)
# ---------------------------------------------------------------------------
def parse_gt_column(
    val: Any, is_viewpoint: bool = False
) -> Optional[Union[List, Dict]]:
    """Convert a GT cell value to a Python object, standardising viewpoint tags."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None

    parsed = val
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                logger.warning("Could not parse GT value: %s", val)
                return None

    if isinstance(parsed, dict):
        if "social_identities" in parsed:
            parsed = parsed["social_identities"]
        elif is_viewpoint or any(
            k in parsed for k in ["in_group", "out_group", "ingroup", "outgroup"]
        ):
            formatted_viewpoints: List[Dict[str, str]] = []
            group_mappings = [
                (["in_group", "ingroup", "IN-GROUP", "IN_GROUP"], "ingroup"),
                (["out_group", "outgroup", "OUT-GROUP", "OUT_GROUP"], "outgroup"),
            ]
            for keys, vp_label in group_mappings:
                for group_key in keys:
                    if group_key in parsed and parsed[group_key]:
                        raw_ids = parsed[group_key]
                        if isinstance(raw_ids, str):
                            raw_ids = [raw_ids]
                        for raw_id in raw_ids:
                            mapped_id = normalize_and_map_identity(raw_id)
                            if mapped_id:
                                formatted_viewpoints.append(
                                    {"identity": mapped_id, "viewpoint": vp_label}
                                )
            return formatted_viewpoints

    if isinstance(parsed, list):
        sanitized: List = []
        for item in parsed:
            if isinstance(item, str):
                mapped = normalize_and_map_identity(item)
                if mapped and mapped not in sanitized:
                    sanitized.append(mapped)
            elif isinstance(item, dict) and "identity" in item:
                mapped_id = normalize_and_map_identity(item["identity"])
                if mapped_id:
                    item_copy = item.copy()
                    item_copy["identity"] = mapped_id
                    if "viewpoint" in item_copy:
                        vp = canon_viewpoint(item_copy["viewpoint"])
                        if vp is None:
                            del item_copy["viewpoint"]
                        else:
                            item_copy["viewpoint"] = vp
                    sanitized.append(item_copy)
        return sanitized

    return parsed


# ---------------------------------------------------------------------------
# Extraction helpers (match notebook cells 9-10)
# ---------------------------------------------------------------------------
def extract_viewpoints_only(viewpoints: Any) -> List[str]:
    """Return only the viewpoint string values, e.g. ['outgroup', 'ingroup'].

    Handles the same dict shapes as ``extract_identity_viewpoints``: a
    Molmo-style {"identities": [...]} wrapper and a single-identity
    {"identity", "viewpoint"} object (glm). Without the latter, glm's
    one-identity answers scored as no prediction in the flat Viewpoint task.
    """
    if isinstance(viewpoints, dict):
        if isinstance(viewpoints.get("identities"), list):
            viewpoints = viewpoints["identities"]
        elif "viewpoint" in viewpoints:
            viewpoints = [viewpoints]
        else:
            viewpoints = []
    if not isinstance(viewpoints, list):
        return []
    out: List[str] = []
    for vp in viewpoints:
        if isinstance(vp, dict) and "viewpoint" in vp:
            label = canon_viewpoint(vp["viewpoint"])
            if label is not None:
                out.append(label)
    return out


def extract_roles_only(roles_list: Any) -> List[str]:
    """Return a flat, deduplicated, lowercased list of role strings across
    all identities, e.g. ["villain", "winner", "hero"].

    Flattened (rather than nested per-identity, e.g. [["villain"], [...]])
    to match normalise_gt_roles()'s format below, so scores can be
    computed with set operations (set() can't hash a list).
    """
    # Some models (e.g. Molmo) return {"identities": [...]} instead of a
    # flat list.  Others (e.g. glm) return a single {"identity": ..., "roles":
    # [...]} object for a one-identity result.  Unwrap/handle both dict shapes
    # before processing.
    if isinstance(roles_list, dict):
        if isinstance(roles_list.get("identities"), list):
            roles_list = roles_list["identities"]
        elif isinstance(roles_list.get("roles"), (str, list)):
            roles_list = [roles_list]
        else:
            roles_list = []
    if not isinstance(roles_list, list):
        return []
    out: List[str] = []
    for entry in roles_list:
        if not (isinstance(entry, dict) and "roles" in entry):
            continue
        roles = entry["roles"]
        if isinstance(roles, str):
            roles = [roles]
        if not isinstance(roles, list):
            continue
        for r in roles:
            r_clean = canon_role(r)
            if r_clean and r_clean not in out:
                out.append(r_clean)
    return out


def extract_identity_roles(roles_list: Any) -> List[Dict[str, Any]]:
    """Return the per-identity role assignment from the raw step-3 output,
    e.g. [{"identity": "Republican", "roles": ["winner"]}, ...].

    Unlike ``extract_roles_only`` (which flattens roles into a bag and
    discards the identity each role belongs to), this preserves the
    identity<->role correspondence so it can be scored with per-identity
    alignment.  It handles the same shapes as ``extract_roles_only``:
    a list of {"identity", "roles"} dicts, a Molmo-style dict wrapper
    {"identities": [...]}, and a single-identity {"identity", "roles"}
    dict (glm).  Identities are taxonomy-mapped; off-inventory roles are
    discarded (an identity whose roles are all invalid keeps an empty list).
    """
    if isinstance(roles_list, dict):
        if isinstance(roles_list.get("identities"), list):
            roles_list = roles_list["identities"]
        elif isinstance(roles_list.get("roles"), (str, list)) and "identity" in roles_list:
            roles_list = [roles_list]
        else:
            roles_list = []
    if not isinstance(roles_list, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in roles_list:
        if not (isinstance(entry, dict) and "roles" in entry):
            continue
        mapped = normalize_and_map_identity(str(entry.get("identity", "")))
        if not mapped:
            continue
        roles = entry["roles"]
        if isinstance(roles, str):
            roles = [roles]
        if not isinstance(roles, list):
            continue
        roles_clean: List[str] = []
        for r in roles:
            r_clean = canon_role(r)
            if r_clean and r_clean not in roles_clean:
                roles_clean.append(r_clean)
        out.append({"identity": mapped, "roles": roles_clean})
    return out


def extract_identity_viewpoints(viewpoints_list: Any) -> List[Dict[str, Any]]:
    """Return the per-identity viewpoint assignment from the raw step-2
    output, e.g. [{"identity": "Republican", "viewpoint": "ingroup"}, ...].

    Unlike ``extract_viewpoints_only`` (which flattens viewpoints into a bag
    and discards the identity each viewpoint belongs to), this preserves the
    identity<->viewpoint correspondence so it can be scored with per-identity
    alignment. Mirrors ``extract_identity_roles``, handling the same shapes:
    a list of {"identity", "viewpoint"} dicts, a Molmo-style dict wrapper
    {"identities": [...]}, and a single-identity dict (glm). Identities are
    taxonomy-mapped; entries whose viewpoint is not ingroup/outgroup (e.g.
    "neutral") are discarded.
    """
    if isinstance(viewpoints_list, dict):
        if isinstance(viewpoints_list.get("identities"), list):
            viewpoints_list = viewpoints_list["identities"]
        elif "viewpoint" in viewpoints_list and "identity" in viewpoints_list:
            viewpoints_list = [viewpoints_list]
        else:
            viewpoints_list = []
    if not isinstance(viewpoints_list, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in viewpoints_list:
        if not (isinstance(entry, dict) and "viewpoint" in entry):
            continue
        mapped = normalize_and_map_identity(str(entry.get("identity", "")))
        vp = canon_viewpoint(entry["viewpoint"])
        if not mapped or vp is None:
            continue
        out.append({"identity": mapped, "viewpoint": vp})
    return out


# ---------------------------------------------------------------------------
# GT normalisation helpers
#
# These parse the ground-truth columns from the benchmark_dataset
# benchmark JSON:
#   social_identity_gt   e.g. "['anti-feminist', 'left-wing']" (a stringified
#                        Python list of raw, sometimes-hyphenated labels)
#   viewpoint_gt         e.g. "['out_group', 'in_group']" (a flat list of
#                        plain strings, positionally aligned with
#                        social_identity_gt -- NOT the {"viewpoint": ...}
#                        dict shape used by the *prediction* side)
#   narrative_roles_gt   e.g. "['loser', 'winner']" (a flat list of plain
#                        role strings)
#
# Used by format_predictions.py's build_dataframe() and the GT-fixing
# scripts so GT parsing only needs to be correct in one place.
# ---------------------------------------------------------------------------
def normalise_gt_identities(val: Any) -> List[str]:
    """Normalise the social_identity_gt column into a list of identities
    mapped onto the approved taxonomy (e.g. "Left wing", "Anti feminist").
    """
    val = coerce_to_list(val)
    if isinstance(val, dict) and "social_identities" in val:
        val = val["social_identities"]
    if not isinstance(val, list):
        return []
    out: List[str] = []
    for item in val:
        if isinstance(item, str):
            # Strip hyphens before taxonomy mapping. The real GT data uses
            # hyphenated labels (e.g. "anti-feminist", "sex-worker",
            # "anti-lgtbq+", "anti-racist") that don't exist verbatim in
            # APPROVED_IDENTITIES (which uses spaces: "Anti feminist",
            # "Sex worker", ...). Without this, normalize_and_map_identity()
            # silently drops them from GT (only "left-wing"/"right-wing"
            # survived, via an unrelated keyword-substring fallback rule),
            # which deflated the Social Identity score.
            cleaned = strip_hyphens_from_data(item)
            mapped = normalize_and_map_identity(cleaned)
            if mapped and mapped not in out:
                out.append(mapped)
    return out


def _normalise_viewpoint_label(raw: Any) -> Optional[str]:
    """Normalise a viewpoint spelling to 'ingroup'/'outgroup' (None if it is
    neither, e.g. "neutral", "mixed" or blank)."""
    return canon_viewpoint(raw)


def normalise_gt_viewpoints(val: Any, image_filename: Optional[str] = None) -> List[str]:
    """Normalise the viewpoint_gt column into a flat list of
    'ingroup'/'outgroup' strings.

    The real GT data stores this as a flat list of plain strings
    positionally aligned with social_identity_gt, e.g.
    ['out_group', 'in_group'] -- NOT a list of {"viewpoint": ...} dicts
    (that's the *prediction* shape produced by the pipeline). Handling
    only the dict-item shape would make every real GT viewpoint list
    evaluate to [] -- collapsing the Viewpoint (and Combined) scores
    towards zero regardless of prediction quality.

    A blank GT viewpoint is filled from another annotator when possible
    (labels/gt_viewpoint_fills.csv, needs *image_filename*); otherwise it
    is discarded like any other invalid label.
    """
    val = coerce_to_list(val)
    if isinstance(val, dict):
        vps: List[str] = []
        for key, label in [("in_group", "ingroup"), ("out_group", "outgroup"),
                           ("ingroup", "ingroup"), ("outgroup", "outgroup")]:
            if key in val and val[key]:
                ids = val[key]
                if isinstance(ids, str):
                    ids = [ids]
                for _ in ids:
                    vps.append(label)
        return vps
    if isinstance(val, list):
        out = []
        for pos, item in enumerate(val):
            if isinstance(item, dict) and "viewpoint" in item:
                # Prediction-style shape: [{"identity": ..., "viewpoint": ...}]
                vp = _normalise_viewpoint_label(item["viewpoint"])
            elif isinstance(item, str):
                # Real GT shape: flat strings like "in_group"/"out_group"
                vp = gt_viewpoint_at(image_filename or "", pos, item)
            else:
                continue
            if vp is not None:
                out.append(vp)
        return out
    return []


def normalise_gt_roles(val: Any) -> List[str]:
    """Normalise the narrative_roles_gt column into a flat, lowercased list
    of role strings, e.g. ['loser', 'winner']. Mirrors extract_roles_only()'s
    output shape for the prediction side."""
    val = coerce_to_list(val)
    if isinstance(val, dict) and "narrative_roles" in val:
        val = val["narrative_roles"]
    if not isinstance(val, list):
        return []
    out: List[str] = []
    for item in val:
        if isinstance(item, dict) and "roles" in item:
            roles = item["roles"]
            if isinstance(roles, str):
                roles = [roles]
            for r in roles:
                r_clean = canon_role(r)
                if r_clean and r_clean not in out:
                    out.append(r_clean)
        elif isinstance(item, str):
            r_clean = canon_role(item)
            if r_clean and r_clean not in out:
                out.append(r_clean)
    return out


def normalise_gt_identity_roles(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the ground-truth per-identity role assignment as a list of
    {"identity": <mapped>, "roles": [<lower>]}.

    ``social_identity_gt`` and ``narrative_roles_gt`` are parallel lists
    aligned by index (e.g. image 1: identity 'democrat' -> 'loser',
    'republican' -> 'winner').  This keeps that positional alignment --
    including repeated identities (e.g. image 2: 'women','women' ->
    'fool','loser') which flat normalisers would silently collapse -- so a
    separate per-identity role metric can be computed.  Only an explicit
    {"identity", "roles"} dict shape (used by some prediction outputs) is
    also accepted.
    """
    vals = entry if isinstance(entry, dict) else {}
    ids = coerce_to_list(vals.get("social_identity_gt", []))
    roles = coerce_to_list(vals.get("narrative_roles_gt", []))
    if isinstance(ids, dict) and "social_identities" in ids:
        ids = ids["social_identities"]
    if isinstance(roles, dict) and "narrative_roles" in roles:
        roles = roles["narrative_roles"]
    if not isinstance(ids, list) or not isinstance(roles, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, ident in enumerate(ids):
        if isinstance(ident, dict) and "roles" in ident:
            ri = ident
        else:
            mapped = normalize_and_map_identity(strip_hyphens_from_data(str(ident)))
            if not mapped:
                continue
            r_raw = roles[i] if i < len(roles) else None
            if isinstance(r_raw, dict) and "roles" in r_raw:
                r_raw = r_raw["roles"]
            if isinstance(r_raw, str):
                r_raw = [r_raw]
            ri = {"identity": mapped, "roles": list(r_raw or [])}
        if "identity" in ri and ri["identity"]:
            out.append({
                "identity": str(ri["identity"]),
                "roles": [r for r in (canon_role(x) for x in (ri.get("roles") or [])) if r],
            })
    return out


def normalise_gt_identity_viewpoints(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the ground-truth per-identity viewpoint assignment as a list of
    {"identity": <mapped>, "viewpoint": <ingroup|outgroup>}.

    ``social_identity_gt`` and ``viewpoint_gt`` are parallel lists aligned by
    index (e.g. image 1: identity 'democrat' -> 'in_group', 'republican' ->
    'out_group'), same as ``social_identity_gt``/``narrative_roles_gt``.
    Mirrors ``normalise_gt_identity_roles`` so a per-identity viewpoint
    metric can be computed the same way as the existing per-identity role
    metric.
    """
    vals = entry if isinstance(entry, dict) else {}
    ids = coerce_to_list(vals.get("social_identity_gt", []))
    vps = coerce_to_list(vals.get("viewpoint_gt", []))
    if isinstance(ids, dict) and "social_identities" in ids:
        ids = ids["social_identities"]
    if not isinstance(ids, list) or not isinstance(vps, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, ident in enumerate(ids):
        mapped = normalize_and_map_identity(strip_hyphens_from_data(str(ident)))
        if not mapped:
            continue
        vp = gt_viewpoint_at(str(vals.get("image_filename", "")), i, vps[i]) if i < len(vps) else None
        if vp is None:
            continue
        out.append({"identity": mapped, "viewpoint": vp})
    return out


# ---------------------------------------------------------------------------
# Saved pipeline output -> per-step parsed JSON
# ---------------------------------------------------------------------------
STEP_KEYS = {1: "01_social_identities", 2: "02_viewpoints", 3: "03_narrative_roles"}


def load_step_outputs(final: Dict[str, Any]) -> Dict[int, Any]:
    """Return {step: parsed_json} for a final_output.json, re-parsing each
    step's saved raw_text with the current vlm_client.try_parse_json so every
    model's output goes through the same parser (the parsed_json stored at
    inference time came from whichever parser version was current then).
    Falls back to the stored parse only when a step has no raw_text."""
    raw_by_step = {
        s.get("step"): s.get("raw_text")
        for s in final.get("steps", []) if isinstance(s, dict)
    }
    out: Dict[int, Any] = {}
    for step, key in STEP_KEYS.items():
        raw = raw_by_step.get(step)
        out[step] = try_parse_json(raw) if raw else final.get(key)
    return out


def extract_social_identities(parsed_json: Any) -> List[str]:
    """Taxonomy-mapped step-1 identities. Accepts the {"social_identities":
    [...]} object the prompt asks for, a bare list, or a single-key object
    (same shapes as parsing.validate_list_result)."""
    _, items = validate_list_result(parsed_json, "social_identities")
    out: List[str] = []
    for item in items:
        mapped = normalize_and_map_identity(str(item))
        if mapped and mapped not in out:
            out.append(mapped)
    return out


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------
def build_dataframe(model_output_dir: Path, benchmark_lookup: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if not model_output_dir.is_dir():
        logger.error("Model output directory does not exist: %s", model_output_dir)
        return pd.DataFrame()

    image_dirs = sorted(
        d for d in model_output_dir.iterdir() if d.is_dir()
    )
    logger.info("Found %d image output dirs in %s", len(image_dirs), model_output_dir)

    for img_dir in image_dirs:
        image_stem = img_dir.name
        final_path = img_dir / "final_output.json"
        if not final_path.is_file():
            logger.warning("No final_output.json for %s, skipping", image_stem)
            continue

        with open(final_path, "r", encoding="utf-8") as f:
            final = json.load(f)

        # --- Extract predictions from pipeline output ---
        parsed = load_step_outputs(final)
        norm_social_ids = extract_social_identities(parsed[1])
        pred_viewpoints_raw = parsed[2]
        pred_roles_raw = parsed[3]

        pred_viewpoints = extract_viewpoints_only(pred_viewpoints_raw)
        pred_roles = extract_roles_only(pred_roles_raw)
        pred_roles_by_identity = extract_identity_roles(pred_roles_raw)
        pred_viewpoints_by_identity = extract_identity_viewpoints(pred_viewpoints_raw)

        # --- Load ground truth from benchmark JSON ---
        # Try common extensions
        gt_entry = None
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
            gt_entry = benchmark_lookup.get(image_stem + ext)
            if gt_entry is not None:
                break

        if gt_entry is None:
            # No matching benchmark entry -- almost always a stale output
            # dir left over from a previous run against a *different*
            # image set (out/<model>/ is never cleared between runs).
            # Skip it entirely rather than scoring against an empty GT,
            # which would silently deflate every metric (a non-empty
            # prediction vs. empty GT scores 0 on Jaccard/exact match).
            logger.warning(
                "No benchmark entry for %s; skipping (stale output dir?)",
                image_stem,
            )
            continue

        # Parse and normalise GT the same way as the predictions
        # (taxonomy-mapped identities, ingroup/outgroup viewpoints,
        # lowercased roles) so both sides are directly comparable.
        # Previously these were written as raw, unparsed benchmark
        # strings (e.g. "['anti-feminist', 'left-wing']"), which
        # evaluate.py's clean_and_normalize() would lowercase but never
        # taxonomy-map or hyphen-strip -- silently mismatching against
        # taxonomy-mapped predictions.
        gt_social = normalise_gt_identities(gt_entry.get("social_identity_gt", "[]"))
        gt_viewpoint = normalise_gt_viewpoints(
            gt_entry.get("viewpoint_gt", "[]"), gt_entry.get("image_filename"),
        )
        gt_roles = normalise_gt_roles(gt_entry.get("narrative_roles_gt", "[]"))
        gt_roles_by_identity = normalise_gt_identity_roles(gt_entry)
        gt_viewpoints_by_identity = normalise_gt_identity_viewpoints(gt_entry)

        row = {
            "image_filename": image_stem,
            "social_identity_gt": gt_social,
            "viewpoint_gt": gt_viewpoint,
            "narrative_roles_gt": gt_roles,
            "narrative_roles_gt_by_identity": gt_roles_by_identity,
            "viewpoint_gt_by_identity": gt_viewpoints_by_identity,
            "predicted_social_identities": norm_social_ids,
            "predicted_viewpoints": pred_viewpoints,
            "predicted_narrative_roles": pred_roles,
            "predicted_narrative_roles_by_identity": pred_roles_by_identity,
            "predicted_viewpoints_by_identity": pred_viewpoints_by_identity,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Built DataFrame with %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format pipeline output into a flat CSV for evaluation."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output CSV (default: out/<model>/formatted_predictions.csv).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to the model output directory (default: out/<model>/).",
    )
    parser.add_argument(
        "--aliases",
        type=str,
        default=None,
        help="Alias file (e.g. labels/aliases.yaml) for a separate *lenient* "
             "score. Off by default; never use it for the main results.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    )

    if args.aliases:
        set_aliases(load_aliases(Path(args.aliases)))
        logger.warning("LENIENT scoring: aliases from %s are active", args.aliases)

    # Load benchmark lookup
    benchmark_path = Path(BENCHMARK_JSON)
    if not benchmark_path.is_file():
        logger.error("Benchmark JSON not found: %s", BENCHMARK_JSON)
        sys.exit(1)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    benchmark_lookup = {e["image_filename"]: e for e in benchmark_data if "image_filename" in e}
    logger.info("Loaded %d benchmark entries", len(benchmark_lookup))

    # Determine model output directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = Path(OUTPUT_DIR) / MODEL_SHORT_NAME

    # Build DataFrame
    df = build_dataframe(model_dir, benchmark_lookup)

    if df.empty:
        logger.error("No data to format. Exiting.")
        sys.exit(1)

    # GT columns are already normalised (taxonomy-mapped identities,
    # ingroup/outgroup viewpoints, lowercased roles) by build_dataframe()
    # via normalise_gt_identities()/normalise_gt_viewpoints()/
    # normalise_gt_roles(), which each apply strip_hyphens_from_data()
    # internally where relevant -- no further processing needed here.

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = model_dir / "formatted_predictions.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved formatted predictions to %s", out_path)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
