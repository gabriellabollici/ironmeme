#!/usr/bin/env python3
"""Stratified random baseline for the three flat tasks.

For every meme and every inventory label, predict the label independently
with probability equal to its prevalence in the ground truth (the share of
evaluated memes whose GT contains it) -- the multi-label equivalent of
sklearn's DummyClassifier(strategy="stratified"). Scored with
evaluate.score_label_sets (fixed inventory, same macro rule as the main
table) and averaged over --runs random draws.

GT and meme set come from a formatted_predictions.csv of the main table
(default: kimi-k3, 1109 memes; the GT columns are identical across models).

Usage (from the repo root):
    python -m experiments.evaluation.random_baseline [--runs 1000] [--output PATH]
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


from experiments.evaluation.evaluate import TASK_CLASSES, score_label_sets
from paths import GT_CONDITIONED_METRICS_DIR

TASKS = [
    ("Social Identity", "social_identity_gt"),
    ("Viewpoint", "viewpoint_gt"),
    ("Narrative Roles", "narrative_roles_gt"),
]
METRICS = ["micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt-csv", type=Path,
                    default=GT_CONDITIONED_METRICS_DIR / "kimi-k3" / "formatted_predictions.csv")
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path,
                    default=GT_CONDITIONED_METRICS_DIR / "all_metrics" / "random_baseline.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.gt_csv)
    rng = np.random.default_rng(args.seed)
    rows = []
    for task, col in TASKS:
        classes = TASK_CLASSES[task]
        y_true = [[str(x).lower() for x in ast.literal_eval(v)] for v in df[col]]
        prevalence = np.array([np.mean([c in set(t) for t in y_true]) for c in classes])
        runs = []
        for _ in range(args.runs):
            draw = rng.random((len(y_true), len(classes))) < prevalence
            y_pred = [[classes[j] for j in np.flatnonzero(r)] for r in draw]
            sc = score_label_sets(y_true, y_pred, classes)
            runs.append([sc[m] for m in METRICS])
        runs = np.array(runs)
        row = {"task": task, "n_memes": len(y_true), "runs": args.runs}
        for i, m in enumerate(METRICS):
            row[m] = runs[:, i].mean()
            row[f"{m}_std"] = runs[:, i].std()
        rows.append(row)

    out = pd.DataFrame(rows)
    avg = {"task": "Average", "n_memes": len(df), "runs": args.runs}
    avg.update({m: out[m].mean() for m in METRICS})
    out = pd.concat([out, pd.DataFrame([avg])], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    pd.set_option("display.width", 200)
    print((out[["task"] + METRICS].set_index("task") * 100).round(2).to_string())
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
