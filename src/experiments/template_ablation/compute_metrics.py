#!/usr/bin/env python3
"""Compute full narrative-role metrics (F1/P/R/Jaccard/exact) for a template
ablation run dir, from its results.jsonl.

Uses evaluate.py's score_label_sets (fixed role inventory, same invalid-label
rule as the main results via labels.canon_role) so numbers are comparable
with the rest of the pipeline. A failed prediction is scored as an empty
prediction, as in the main pipeline. Writes <run_dir>/<variant>/metrics.json.
"""
import argparse
import json
import sys
from pathlib import Path
from paths import TEMPLATE_ABLATION_DIR


try:
    from experiments.evaluation.evaluate import score_label_sets
    from dataset_construction.labels import canon_role, inventory
except ImportError as e:
    sys.exit(f"Missing dependency: {e} (need numpy + scikit-learn)")

BASE_DIR = TEMPLATE_ABLATION_DIR / "gemini-3.6-flash"

VARIANTS = ["with_template", "no_template"]


def load_records(results_file):
    records = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_list(v):
    if not v:
        return []
    if isinstance(v, str):
        import ast
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = [v]
    return list(v) if isinstance(v, list) else []


def to_set(v):
    """Inventory roles in *v*; off-inventory strings are discarded."""
    return {r for r in (canon_role(x) for x in to_list(v)) if r}


def compute(records):
    y_true, y_pred = [], []
    failures = 0
    n_pred_labels = n_discarded = 0
    for r in records:
        if not r.get("success", False):
            failures += 1
        raw_pred = to_list(r.get("predicted_narrative_roles"))
        pr = to_set(raw_pred)
        n_pred_labels += len(raw_pred)
        n_discarded += sum(1 for x in raw_pred if canon_role(x) is None)
        y_true.append(to_set(r.get("narrative_roles_gt")))
        y_pred.append(pr)

    if not y_true:
        return {"total_images": len(records), "successful": 0, "failed": failures,
                "note": "no predictions"}

    sc = score_label_sets(y_true, y_pred, inventory("role"))

    def r4(x):
        return round(float(x), 4)

    return {
        "total_images": len(records),
        "successful_predictions": len(records) - failures,
        "failed_predictions": failures,
        "exact_match": sc["exact_match"],
        "mean_jaccard": sc["mean_jaccard"],
        "narrative_roles_classes": sc["macro_classes"],
        "predicted_labels": n_pred_labels,
        "discarded_off_inventory": n_discarded,
        "micro_precision": r4(sc["micro_precision"]), "micro_recall": r4(sc["micro_recall"]),
        "micro_f1": r4(sc["micro_f1"]),
        "macro_precision": r4(sc["macro_precision"]), "macro_recall": r4(sc["macro_recall"]),
        "macro_f1": r4(sc["macro_f1"]),
        "weighted_precision": r4(sc["weighted_precision"]), "weighted_recall": r4(sc["weighted_recall"]),
        "weighted_f1": r4(sc["weighted_f1"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=str(BASE_DIR))
    ap.add_argument("--variant", type=str, default=None, choices=VARIANTS)
    ap.add_argument("--model", type=str, default="gemini-3.6-flash")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    variants = [args.variant] if args.variant else VARIANTS
    all_out = {}
    for v in variants:
        results_file = run_dir / v / "results.jsonl"
        if not results_file.is_file():
            print(f"[warn] no results.jsonl for {v}, skipping")
            continue
        m = compute(load_records(results_file))
        m.update({"variant": v, "config": run_dir.name, "model": args.model})
        all_out[v] = m
        out_path = run_dir / v / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        print(f"{v}: wrote {out_path}")
        for k, val in m.items():
            print(f"  {k}: {val}")

    if args.variant is None and len(all_out) == 2:
        wt, nt = all_out["with_template"], all_out["no_template"]
        print("\n=== DELTA (with_template - no_template) ===")
        for key in ["exact_match", "mean_jaccard", "micro_f1", "macro_f1", "weighted_f1"]:
            if wt.get(key) is not None and nt.get(key) is not None:
                print(f"  {key}: {wt[key] - nt[key]:+.4f}")
            else:
                print(f"  {key}: n/a")


if __name__ == "__main__":
    main()
