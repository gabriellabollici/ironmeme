#!/usr/bin/env python3
"""Per-identity narrative role alignment analysis.

Problem being addressed:
  - format_predictions.py::extract_roles_only() flattens the per-identity
    [{identity, roles}] step-3 output into a flat bag of role labels, and
    evaluate.py scores that flat bag against the flat GT role list. So the
    identity<->role correspondence is never checked: a model that gets the
    right roles but assigns them to the wrong identities scores identically
    to one that assigns them correctly.

  - The GT itself IS a positional per-identity role assignment: in the
    benchmark JSON, social_identity_gt, viewpoint_gt and narrative_roles_gt
    are parallel lists aligned by index (e.g. image 1: identity 'democrat' ->
    'loser', 'republican' -> 'winner').

This script reconstructs the per-identity GT (raw identities + roles) and the
per-identity model predictions (raw step-3 [{identity, roles}] output), aligns
them by a normalised identity label, and reports role metrics PER GT identity:

  - role_jaccard     : per-GT-identity Jaccard between GT roles and the roles
                       the model assigned to that identity (0 if the identity
                       is missing from the prediction).
  - covered_rate     : fraction of GT identities that appear in the model
                       output with >=1 role.
  - correct_role     : fraction of GT identities whose assigned role exactly
                       matches (set equality).
  - gt_identity_loss : fraction of GT identities absent from the prediction.

A flat-bag baseline column is also shown for comparison.

Usage:
    python -m experiments.evaluation.per_identity_roles [--results-dir benchmark/results/raw/gt_conditioned]
                                  [--benchmark benchmark/benchmark_dataset/ironmeme_benchmark.json]
                                  [--models a,b,c]
"""
import argparse
import ast
import json
import re
from pathlib import Path

from dataset_construction.labels import canon_identity, canon_role
from experiments.parsing import try_parse_json
from paths import BENCHMARK_JSON, RAW_GT_CONDITIONED_DIR

DEFAULT_RESULTS = RAW_GT_CONDITIONED_DIR
DEFAULT_BENCHMARK = BENCHMARK_JSON


# ---------------------------------------------------------------------------
# Normalisation keys
# ---------------------------------------------------------------------------
def norm_key(s: object) -> str:
    """Alignment key of the taxonomy label *s* maps to ("" if off-inventory)."""
    mapped = canon_identity(s)
    return re.sub(r"[^a-z0-9]+", "", mapped.lower()) if mapped else ""


def valid_roles(roles) -> list[str]:
    return [r for r in (canon_role(x) for x in (roles or [])) if r]


# ---------------------------------------------------------------------------
# GT per-identity roles (positional ids + roles)
# ---------------------------------------------------------------------------
def gt_identity_roles(entry: dict) -> list[tuple[str, list[str]]]:
    """Return [(normalised_identity_key, [gt roles]), ...] positionally aligned."""
    ids = entry.get("social_identity_gt") or []
    roles = entry.get("narrative_roles_gt") or []

    def parse(raw):
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw or raw == "[]":
                return []
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    return []
        return []

    ids = parse(ids)
    roles = parse(roles)
    ids = ids if isinstance(ids, list) else []
    roles = roles if isinstance(roles, list) else []
    out = []
    for i, ident in enumerate(ids):
        if not isinstance(ident, str):
            continue
        key = norm_key(ident)
        if not key:
            continue
        r = roles[i] if i < len(roles) else None
        if isinstance(r, str):
            r = [r]
        out.append((key, valid_roles(r)))
    return out


# ---------------------------------------------------------------------------
# Predicted per-identity roles from raw step-3 output
# ---------------------------------------------------------------------------
def pred_identity_roles(parsed_json) -> list[tuple[str, list[str]]]:
    """Extract [(normalised_identity_key, [pred roles]), ...] handling all shapes:

    - list of {"identity": ..., "roles": [...]}
    - dict {"identities": [{...}, ...]}
    - dict {"identity": ..., "roles": [...]} (single identity, e.g. glm)
    - roles can be a string or a list.
    """
    if isinstance(parsed_json, dict):
        if isinstance(parsed_json.get("identities"), list):
            items = parsed_json["identities"]
        elif "identity" in parsed_json:
            items = [parsed_json]
        else:
            items = []
    elif isinstance(parsed_json, list):
        items = parsed_json
    else:
        items = []

    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ident = item.get("identity")
        key = norm_key(ident) if ident is not None else ""
        roles = item.get("roles")
        if isinstance(roles, str):
            roles = [roles]
        roles = valid_roles(roles)
        if key:
            out.append((key, roles))
    return out


# ---------------------------------------------------------------------------
# Per-identity alignment scoring
# ---------------------------------------------------------------------------
def identity_metrics(gt_pairs, pred_pairs) -> dict:
    """Align GT identity->roles with predicted identity->roles.

    Returns aggregates across all GT identities:
      n_gt_identities, n_covered, n_correct, role_jaccard_sum, flat_roles, flat_gt
    """
    pred_map: dict[str, list[str]] = {}
    for key, roles in pred_pairs:
        pred_map.setdefault(key, []).extend(roles)

    n_gt = 0
    n_covered = 0
    n_correct = 0
    n_hit = 0
    jac_sum = 0.0

    flat_pred: list[str] = []
    flat_gt: list[str] = []
    for r in pred_pairs:
        for role in r[1]:
            if role not in flat_pred:
                flat_pred.append(role)
    for _, roles in gt_pairs:
        for role in roles:
            if role not in flat_gt:
                flat_gt.append(role)
            n_gt += 1
            pr = set(pred_map.get(_, []))
            gr = set(roles)
            if pr:
                n_covered += 1
            if gr and gr <= pr:
                n_hit += 1
            if pr == gr and gr:
                n_correct += 1
            if pr or gr:
                jac_sum += len(pr & gr) / len(pr | gr)
            else:
                jac_sum += 1.0

    return {
        "n_gt": n_gt,
        "n_covered": n_covered,
        "n_correct": n_correct,
        "n_hit": n_hit,
        "jac_sum": jac_sum,
        "flat_pred": flat_pred,
        "flat_gt": flat_gt,
    }


def flat_jaccard(pred, gt) -> float:
    p, g = set(pred), set(gt)
    return len(p & g) / len(p | g) if (p or g) else 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model dir names; default: all under results dir.")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Optional path to write the summary table as CSV.")
    args = parser.parse_args()

    with open(args.benchmark) as f:
        benchmark = json.load(f)
    gt_lookup = {}
    for e in benchmark:
        if "image_filename" in e:
            gt_lookup[e["image_filename"]] = e

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        skip = {"all_metrics", "vllm_logs"}
        models = sorted(
            d.name for d in args.results_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name not in skip
        )

    header = (f"{'model':28s} {'n_img':>5s} {'n_id':>5s} {'cov%':>6s} "
              f"{'hit%':>6s} {'corr%':>6s} {'jac%':>6s} {'flatJac%':>9s}")
    print(header)
    print("-" * len(header))

    out_rows = []
    for m in models:
        mdir = args.results_dir / m
        if not mdir.is_dir():
            continue
        tot = {"n_gt": 0, "n_covered": 0, "n_correct": 0, "n_hit": 0, "jac_sum": 0.0}
        n_img = 0
        flat_jac_sum = 0.0
        n_flat_img = 0
        for img in sorted(mdir.iterdir()):
            if not img.is_dir() or not img.name.isdigit():
                continue
            s3 = img / "03_narrative_roles.json"
            if not s3.is_file():
                continue
            try:
                j = json.load(open(s3, encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            # Re-parse raw_text with the shared pipeline parser.
            pj = try_parse_json(j["raw_text"]) if j.get("raw_text") else j.get("parsed_json")
            gt_entry = None
            for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
                gt_entry = gt_lookup.get(img.name + ext)
                if gt_entry is not None:
                    break
            if gt_entry is None:
                continue
            gt_pairs = gt_identity_roles(gt_entry)
            pred_pairs = pred_identity_roles(pj)
            met = identity_metrics(gt_pairs, pred_pairs)
            tot["n_gt"] += met["n_gt"]
            tot["n_covered"] += met["n_covered"]
            tot["n_correct"] += met["n_correct"]
            tot["n_hit"] += met["n_hit"]
            tot["jac_sum"] += met["jac_sum"]
            n_img += 1
            flat_jac_sum += flat_jaccard(met["flat_pred"], met["flat_gt"])
            n_flat_img += 1

        n_gt = tot["n_gt"]
        cov = 100 * tot["n_covered"] / n_gt if n_gt else 0.0
        hit = 100 * tot["n_hit"] / n_gt if n_gt else 0.0
        corr = 100 * tot["n_correct"] / n_gt if n_gt else 0.0
        jac = 100 * tot["jac_sum"] / n_gt if n_gt else 0.0
        fjac = 100 * flat_jac_sum / n_flat_img if n_flat_img else 0.0
        print(f"{m:28s} {n_img:5d} {n_gt:5d} {cov:6.1f} {hit:6.1f} {corr:6.1f} "
              f"{jac:6.1f} {fjac:9.1f}")
        out_rows.append({
            "model": m, "n_images": n_img, "n_gt_identities": n_gt,
            "role_assigned_rate_%": round(cov, 2),
            "gt_role_hit_%": round(hit, 2),
            "exact_role_%": round(corr, 2),
            "per_identity_jaccard_%": round(jac, 2),
            "flat_jaccard_%": round(fjac, 2),
        })

    if args.output_csv:
        import csv as _csv
        with open(args.output_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=out_rows[0].keys())
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nSaved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
