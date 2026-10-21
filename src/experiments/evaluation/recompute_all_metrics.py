#!/usr/bin/env python3
"""Recompute every reported metric from the SAVED model outputs (no inference).

Reads (never writes) the raw model outputs in benchmark/results/raw/ and the
ablation results.jsonl files, and writes the derived files into --out-root
(default: benchmark/, i.e. in place; pass a scratch directory for a dry run),
at the same relative paths:

  results/metrics/gt_conditioned/<model>/{formatted_predictions.csv,metrics.json}
  results/metrics/gt_conditioned/all_metrics/{<model>_metrics.json,all_models_metrics.csv}
  results/metrics/subset_gt_conditioned/...  (GT-conditioned VLMs + human, same 214 memes)
  results/metrics/no_gt/<model|human*>/{formatted_predictions.csv,metrics.json}
  results/metrics/subset_no_gt/...
  results/metrics/gt_conditioned/all_metrics/per_identity_roles_summary.csv
  results/template_ablation/<run>/<variant>/{predictions.csv,stats.json,metrics.json}
  results/diagnostics/{diagnostics.csv,discarded_labels_top20.csv}

Gemini's GT-conditioned run has no per-image outputs, only its original
formatted_predictions.csv (in raw/): its predictions are re-normalised from
that CSV with the same rule and its GT columns are rebuilt from the benchmark.

With --aliases, writes a separate *lenient* score to
<out-root>/results/metrics/gt_conditioned_lenient/ instead (main results untouched).

Usage (from the repo root):
    python -m experiments.evaluation.recompute_all_metrics [--out-root DIR] [--aliases src/dataset_construction/labels/aliases.yaml]
"""
import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd  # noqa: E402

from experiments.inference.config import BENCHMARK_JSON
from experiments.evaluation.format_predictions import (
    normalise_gt_identities, normalise_gt_identity_roles, normalise_gt_identity_viewpoints,
    normalise_gt_roles, normalise_gt_viewpoints,
)
from dataset_construction.labels import canon_identity, canon_role, canon_viewpoint
from paths import (
    BENCHMARK_DIR,
    DIAGNOSTICS_DIR,
    GT_CONDITIONED_METRICS_DIR,
    LENIENT_GT_CONDITIONED_METRICS_DIR,
    NO_GT_METRICS_DIR,
    RAW_ANNOTATIONS_DIR,
    RAW_GT_CONDITIONED_DIR,
    RAW_NO_GT_DIR,
    SUBSET_GT_CONDITIONED_METRICS_DIR,
    SUBSET_NO_GT_METRICS_DIR,
    TEMPLATE_ABLATION_DIR,
)

DATA = BENCHMARK_DIR
RAW_GT = RAW_GT_CONDITIONED_DIR
RAW_NO_GT = RAW_NO_GT_DIR
ABLATION = TEMPLATE_ABLATION_DIR
GT_VLMS = [
    "deepseek-vl2", "glm-4.1v-9b-thinking", "kimi-k3", "molmo-7b-d-0924", "pixtral-12b-2409",
    "qwen2.5-vl-7b-instruct", "qwen3-vl-8b-instruct", "qwen3-vl-8b-thinking",
]
CSV_ONLY_GT_VLMS = ["gemini-3.6-flash"]
NO_GT_VLMS = ["gemini-3.6-flash", "glm-4.1v-9b-thinking"]
HUMANS = ["human", "human_E_evaluator", "human_H_evaluator", "human_M_evaluator", "human_R_evaluator", "human_S_evaluator"]


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, stdout=subprocess.DEVNULL)


def fmt_and_eval(model_dir: Path, out_dir: Path, model: str, aliases: str = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["-m", "experiments.evaluation.format_predictions", "--model-dir", str(model_dir),
           "--output", str(out_dir / "formatted_predictions.csv")]
    if aliases:
        cmd += ["--aliases", aliases]
    run(*cmd)
    evaluate(out_dir, model)


def evaluate(out_dir: Path, model: str) -> None:
    run("-m", "experiments.evaluation.evaluate", "--input", str(out_dir / "formatted_predictions.csv"),
        "--output", str(out_dir / "metrics.json"), "--model", model)


def _lit(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v) if v.strip() else []
        except (ValueError, SyntaxError):
            return []
    return v if isinstance(v, list) else []


def renormalise_csv(src_csv: Path, out_dir: Path, model: str) -> None:
    """Rescore a run that only exists as formatted_predictions.csv."""
    bench = {Path(e["image_filename"]).stem: e for e in json.load(open(BENCHMARK_JSON, encoding="utf-8"))}
    df = pd.read_csv(src_csv, dtype={"image_filename": str})

    def keep(fn, xs):
        out = []
        for x in _lit(xs):
            c = fn(x)
            if c and c not in out:
                out.append(c)
        return out

    def by_id(items, field, fn):
        out = []
        for e in _lit(items):
            ident = canon_identity(e.get("identity", "")) if isinstance(e, dict) else None
            if not ident:
                continue
            if field == "roles":
                out.append({"identity": ident, "roles": keep(canon_role, e.get("roles") or [])})
            else:
                vp = canon_viewpoint(e.get("viewpoint"))
                if vp:
                    out.append({"identity": ident, "viewpoint": vp})
        return out

    df["predicted_social_identities"] = df["predicted_social_identities"].map(lambda v: keep(canon_identity, v))
    df["predicted_viewpoints"] = df["predicted_viewpoints"].map(lambda v: [x for x in map(canon_viewpoint, _lit(v)) if x])
    df["predicted_narrative_roles"] = df["predicted_narrative_roles"].map(lambda v: keep(canon_role, v))
    df["predicted_narrative_roles_by_identity"] = df["predicted_narrative_roles_by_identity"].map(
        lambda v: by_id(v, "roles", canon_role))
    df["predicted_viewpoints_by_identity"] = df["predicted_viewpoints_by_identity"].map(
        lambda v: by_id(v, "viewpoint", canon_viewpoint))
    gt = df["image_filename"].map(bench)
    df["social_identity_gt"] = gt.map(lambda e: normalise_gt_identities(e.get("social_identity_gt", "[]")))
    df["viewpoint_gt"] = gt.map(lambda e: normalise_gt_viewpoints(e.get("viewpoint_gt", "[]"), e["image_filename"]))
    df["narrative_roles_gt"] = gt.map(lambda e: normalise_gt_roles(e.get("narrative_roles_gt", "[]")))
    df["narrative_roles_gt_by_identity"] = gt.map(normalise_gt_identity_roles)
    df["viewpoint_gt_by_identity"] = gt.map(normalise_gt_identity_viewpoints)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "formatted_predictions.csv", index=False)
    evaluate(out_dir, model)


def restrict(src_dir: Path, out_dir: Path, memes: set, model: str) -> None:
    df = pd.read_csv(src_dir / "formatted_predictions.csv", dtype={"image_filename": str})
    out_dir.mkdir(parents=True, exist_ok=True)
    df[df["image_filename"].isin(memes)].to_csv(out_dir / "formatted_predictions.csv", index=False)
    evaluate(out_dir, model)


def under(out: Path, path: Path) -> Path:
    """``path`` (somewhere under benchmark/) relocated under ``out``."""
    return out / path.relative_to(BENCHMARK_DIR)


def aggregate(root: Path) -> None:
    run("-m", "experiments.evaluation.aggregate_all_metrics", "--results-root", str(root))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", type=Path, default=DATA)
    ap.add_argument("--aliases", type=str, default=None,
                    help="Lenient score only; written to <out-root>/results/metrics/gt_conditioned_lenient/.")
    args = ap.parse_args()
    out = args.out_root.resolve()

    # Refresh the blank-GT-viewpoint fills from the raw annotations when they are
    # available (they are not in git); otherwise score with the committed fills.
    if RAW_ANNOTATIONS_DIR.is_dir():
        run("-m", "dataset_construction.annotation_formatting.resolve_blank_gt_viewpoints")
    else:
        print(f"{RAW_ANNOTATIONS_DIR} not found; using the committed GT viewpoint fills")

    if args.aliases:
        root = under(out, LENIENT_GT_CONDITIONED_METRICS_DIR)
        for m in GT_VLMS:
            fmt_and_eval(RAW_GT / m, root / m, m, args.aliases)
        aggregate(root)
        print(f"Lenient scores written to {root}")
        return

    # 1. GT-conditioned VLMs (main results table)
    corrected = under(out, GT_CONDITIONED_METRICS_DIR)
    for m in GT_VLMS:
        fmt_and_eval(RAW_GT / m, corrected / m, m)
    for m in CSV_ONLY_GT_VLMS:
        renormalise_csv(RAW_GT / m / "formatted_predictions.csv", corrected / m, m)
    (corrected / "all_metrics").mkdir(parents=True, exist_ok=True)
    for m in GT_VLMS + CSV_ONLY_GT_VLMS:
        shutil.copyfile(corrected / m / "metrics.json", corrected / "all_metrics" / f"{m}_metrics.json")
    aggregate(corrected)

    # 2. No-GT runs + human baselines
    no_gt = under(out, NO_GT_METRICS_DIR)
    for m in NO_GT_VLMS:
        fmt_and_eval(RAW_NO_GT / m, no_gt / m, m)
    run("-m", "experiments.evaluation.format_human_predictions", "--output-root", str(no_gt))
    for h in HUMANS:
        evaluate(no_gt / h, h)

    # 3. Subset-matched comparisons (same meme sets as before)
    def memes_of(csv_path: Path) -> set:
        return set(pd.read_csv(csv_path, dtype={"image_filename": str})["image_filename"])

    sub = under(out, SUBSET_NO_GT_METRICS_DIR)
    memes = memes_of(SUBSET_NO_GT_METRICS_DIR / "human" / "formatted_predictions.csv")
    for m in NO_GT_VLMS + ["human"]:
        restrict(no_gt / m, sub / m, memes, m)
    aggregate(sub)

    sub_gt = under(out, SUBSET_GT_CONDITIONED_METRICS_DIR)
    memes = memes_of(SUBSET_GT_CONDITIONED_METRICS_DIR / "human" / "formatted_predictions.csv")
    for m in GT_VLMS:
        restrict(corrected / m, sub_gt / m, memes, m)
    restrict(no_gt / "human", sub_gt / "human", memes, "human")
    aggregate(sub_gt)

    # 4. Template ablation (results.jsonl only) and per-identity roles
    from experiments.template_ablation import ablation, compute_metrics

    ablation_out = under(out, TEMPLATE_ABLATION_DIR)
    for cond in sorted(p for p in ABLATION.glob("*/*") if p.is_dir()):
        dst = ablation_out / cond.relative_to(ABLATION)
        dst.mkdir(parents=True, exist_ok=True)
        if dst.resolve() != cond.resolve():
            shutil.copyfile(cond / "results.jsonl", dst / "results.jsonl")
        if (cond / "stats.json").is_file():
            ablation.generate_summary_csv(dst, variant=cond.name)
        if (cond / "metrics.json").is_file():
            m = compute_metrics.compute(compute_metrics.load_records(dst / "results.jsonl"))
            old = json.load(open(cond / "metrics.json"))
            m.update({k: old[k] for k in ("variant", "config", "model") if k in old})
            json.dump(m, open(dst / "metrics.json", "w"), indent=2, ensure_ascii=False)
    per_identity = under(out, GT_CONDITIONED_METRICS_DIR) / "all_metrics" / "per_identity_roles_summary.csv"
    run("-m", "experiments.evaluation.per_identity_roles", "--results-dir", str(RAW_GT),
        "--benchmark", str(BENCHMARK_JSON), "--output-csv", str(per_identity))

    # 5. Diagnostics
    run("-m", "experiments.evaluation.eval_diagnostics", "--out-dir", str(under(out, DIAGNOSTICS_DIR)),
        "--derived-root", str(out))
    print(f"All metrics recomputed under {out}")


if __name__ == "__main__":
    main()
