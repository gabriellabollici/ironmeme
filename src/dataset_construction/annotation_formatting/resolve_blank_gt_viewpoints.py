#!/usr/bin/env python3
"""Resolve blank ground-truth viewpoints from the per-annotator annotations.

21 benchmark GT triples have an empty viewpoint (e.g. "feminist--villain"):
the two annotators whose triples formed the consensus both left the
viewpoint empty. When another annotator *did* give a viewpoint for that
same identity in that meme, we use it:

  same_identity_and_role   an annotator labeled the identity with the same
                           role and a non-empty viewpoint
  same_identity_other_role only a different role was given, but the identity
                           has a non-empty viewpoint (viewpoint is a property
                           of the identity, not of the role)
  ambiguous                annotators gave conflicting viewpoints -> not filled
  unresolved               no annotator gave one -> not filled (the blank is
                           then discarded as an invalid label)

Writes labels/gt_viewpoint_fills.csv, read by labels.gt_viewpoint_at().
The benchmark JSON itself is not modified.

Usage (from the repo root):
    python -m dataset_construction.annotation_formatting.resolve_blank_gt_viewpoints
"""
import ast
import csv
import io
import json
import re
from collections import defaultdict


from dataset_construction.annotation_formatting.annotation_postprocessing import aggregate_to_wide, load_raw_annotations
from dataset_construction.build_benchmark import original_to_clean_filenames
from dataset_construction.labels import (
    GT_VIEWPOINT_FILLS_CSV, canon_identity, canon_role, canon_viewpoint, coerce_to_list, surface_key,
)
from paths import BENCHMARK_JSON, RAW_ANNOTATIONS_DIR

ANNOTATORS = ("J", "L", "M")
_PREFIX_RE = re.compile(r"^[A-Za-z]\d+:\s*")


def _strip_prefix(s: str) -> str:
    return _PREFIX_RE.sub("", str(s)).strip()


def _identity_key(raw: str, renames: dict) -> str:
    raw = renames.get(raw.strip().lower(), raw)
    return canon_identity(raw) or surface_key(raw)


def _renames(old_gt: list, new_gt: list) -> dict:
    """Old-label -> clean-label identity renames (e.g. 'corporativist' ->
    'corporativist/capitalist'), from the two aligned GT triple lists."""
    out = {}
    if len(old_gt) == len(new_gt):
        for o, n in zip(sorted(old_gt), sorted(new_gt)):
            oi, *o_rest = o.rsplit("-", 2)
            ni, *n_rest = n.rsplit("-", 2)
            if o_rest == n_rest and oi != ni:
                out[oi.strip().lower()] = ni
    return out


def main() -> None:
    bench = {e["image_filename"]: e for e in json.load(open(BENCHMARK_JSON, encoding="utf-8"))}
    clean_to_old = defaultdict(list)
    for old, clean in original_to_clean_filenames().items():
        clean_to_old[clean].append(old)
    rows_by_old = defaultdict(list)
    for row in aggregate_to_wide(load_raw_annotations(RAW_ANNOTATIONS_DIR)).to_dict("records"):
        rows_by_old[row["image_filename"]].append(row)

    out_rows = []
    for fname, entry in bench.items():
        ids = coerce_to_list(entry.get("social_identity_gt", "[]"))
        vps = coerce_to_list(entry.get("viewpoint_gt", "[]"))
        roles = coerce_to_list(entry.get("narrative_roles_gt", "[]"))
        new_gt = coerce_to_list(entry.get("ground_truth", "[]"))
        for pos, raw_vp in enumerate(vps):
            if canon_viewpoint(raw_vp) is not None:
                continue
            gt_id = str(ids[pos])
            gt_role = canon_role(roles[pos]) if pos < len(roles) else None
            same_role, other_role = defaultdict(set), defaultdict(set)
            for old in clean_to_old.get(fname, []):
                for arow in rows_by_old.get(old, []):
                    renames = _renames(arow.get("ground_truth") or [], new_gt)
                    target = _identity_key(gt_id, renames)
                    for ann in ANNOTATORS:
                        si = arow.get(f"social_identity_{ann}_annotator") or []
                        vp = arow.get(f"view_point_{ann}_annotator") or []
                        rl = arow.get(f"narrative_roles_{ann}_annotator") or []
                        for i, ident in enumerate(si):
                            if _identity_key(_strip_prefix(ident), renames) != target:
                                continue
                            v = canon_viewpoint(_strip_prefix(vp[i])) if i < len(vp) else None
                            if v is None:
                                continue
                            try:
                                inst_roles = {canon_role(r) for r in ast.literal_eval(_strip_prefix(rl[i]))}
                            except Exception:
                                inst_roles = set()
                            (same_role if gt_role in inst_roles else other_role)[v].add(ann)
            source = same_role or other_role
            if len(source) == 1:
                filled = next(iter(source))
                match = "same_identity_and_role" if same_role else "same_identity_other_role"
                annotators = ",".join(sorted(source[filled]))
            else:
                filled = ""
                match = "ambiguous" if source else "unresolved"
                annotators = "; ".join(f"{v}:{','.join(sorted(a))}" for v, a in sorted(source.items()))
            out_rows.append({
                "image_filename": fname, "position": pos, "identity": gt_id,
                "role": roles[pos] if pos < len(roles) else "",
                "filled_viewpoint": filled, "match_type": match, "source_annotators": annotators,
            })

    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(sorted(out_rows, key=lambda r: (int(r["image_filename"].split(".")[0]), r["position"])))
    counts = defaultdict(int)
    for r in out_rows:
        counts[r["match_type"]] += 1
    # Rewrite only on change, so recomputing metrics never touches the tracked file needlessly.
    content = buf.getvalue().encode("utf-8")
    if GT_VIEWPOINT_FILLS_CSV.exists() and GT_VIEWPOINT_FILLS_CSV.read_bytes() == content:
        print(f"{GT_VIEWPOINT_FILLS_CSV} is up to date ({len(out_rows)} blank GT viewpoints: {dict(counts)})")
        return
    GT_VIEWPOINT_FILLS_CSV.parent.mkdir(parents=True, exist_ok=True)
    GT_VIEWPOINT_FILLS_CSV.write_bytes(content)
    print(f"Wrote {len(out_rows)} blank GT viewpoints to {GT_VIEWPOINT_FILLS_CSV}: {dict(counts)}")


if __name__ == "__main__":
    main()
