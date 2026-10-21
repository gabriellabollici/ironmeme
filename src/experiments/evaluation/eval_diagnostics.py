#!/usr/bin/env python3
"""Output-quality diagnostics behind the scored results.

For every model run (GT-conditioned, no-GT, human baselines) and every
template-ablation condition, reports:

  * unparseable responses per step (raw_text re-parsed with the pipeline's
    vlm_client.try_parse_json), count and % of memes;
  * predicted labels per task and how many the invalid-label rule
    (labels.canon_*) discards, count and % of predicted labels;
  * invalid viewpoint strings (anything that is not an ingroup/outgroup
    spelling);
  * the step-1 shape bug: memes whose step-1 answer was valid but not a
    {"social_identities": [...]} object, so the old extraction scored them
    as an empty prediction.

Runs without raw outputs (Gemini's GT-conditioned run, the human baselines,
the ablation results.jsonl files) only have already-extracted labels, so
parse failures / identity discards that happened before extraction cannot be
counted there; those cells are left empty and `source` says why.

Writes <out-dir>/diagnostics.csv and <out-dir>/discarded_labels_top20.csv.

Usage (from the repo root):
    python -m experiments.evaluation.eval_diagnostics [--out-dir benchmark/results/diagnostics]
"""
import argparse
import ast
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


from experiments.evaluation.format_predictions import load_step_outputs
from dataset_construction.labels import canon_identity, canon_role, canon_viewpoint
from experiments.parsing import validate_list_result
from paths import (
    BENCHMARK_DIR,
    BENCHMARK_JSON,
    DIAGNOSTICS_DIR,
    GT_CONDITIONED_METRICS_DIR,
    NO_GT_METRICS_DIR,
    RAW_GT_CONDITIONED_DIR,
    RAW_NO_GT_DIR,
    REPO_ROOT,
    TEMPLATE_ABLATION_DIR,
)

DATA = BENCHMARK_DIR
TASKS = ("identity", "viewpoint", "role")
CANON = {"identity": canon_identity, "viewpoint": canon_viewpoint, "role": canon_role}


# ---------------------------------------------------------------------------
# Raw label iterators (same shapes the format_predictions extractors accept)
# ---------------------------------------------------------------------------
def _entries(parsed: Any, field: str) -> List[dict]:
    if isinstance(parsed, dict):
        if isinstance(parsed.get("identities"), list):
            parsed = parsed["identities"]
        elif field in parsed:
            parsed = [parsed]
        else:
            parsed = []
    if not isinstance(parsed, list):
        return []
    return [e for e in parsed if isinstance(e, dict) and field in e]


def raw_identities(parsed1: Any) -> List[str]:
    _, items = validate_list_result(parsed1, "social_identities")
    return [str(x) for x in items]


def raw_viewpoints(parsed2: Any) -> List[str]:
    return [str(e["viewpoint"]) for e in _entries(parsed2, "viewpoint")]


def raw_roles(parsed3: Any) -> List[str]:
    out: List[str] = []
    for e in _entries(parsed3, "roles"):
        roles = e["roles"]
        if isinstance(roles, str):
            roles = [roles]
        if isinstance(roles, list):
            out.extend(str(r) for r in roles)
    return out


def _old_step1_empty(final: Dict[str, Any]) -> bool:
    """True when the pre-fix extraction (dict with key 'social_identities'
    only, from the stored parse) yields no identities."""
    sid = final.get("01_social_identities", {})
    return not (isinstance(sid, dict) and sid.get("social_identities"))


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------
class Diag:
    def __init__(self, experiment: str, model: str, source: str):
        self.row: Dict[str, Any] = {"experiment": experiment, "model": model, "source": source}
        self.n = 0
        self.unparseable = Counter()
        self.labels = Counter()
        self.discarded = {t: Counter() for t in TASKS}
        self.step1_bug: Optional[int] = None

    def add_labels(self, task: str, raw: Iterable[str]) -> None:
        for x in raw:
            self.labels[task] += 1
            if CANON[task](x) is None:
                self.discarded[task][str(x).strip()] += 1

    def finish(self, steps: Iterable[int], tasks: Iterable[str]) -> Dict[str, Any]:
        r = dict(self.row, n_memes=self.n)
        for s in (1, 2, 3):
            if s in steps:
                r[f"step{s}_unparseable"] = self.unparseable[s]
                r[f"step{s}_unparseable_pct"] = round(100 * self.unparseable[s] / self.n, 2) if self.n else 0.0
        for t in TASKS:
            if t in tasks:
                d = sum(self.discarded[t].values())
                r[f"{t}_labels"] = self.labels[t]
                r[f"{t}_discarded"] = d
                r[f"{t}_discarded_pct"] = round(100 * d / self.labels[t], 2) if self.labels[t] else 0.0
        r["invalid_viewpoint_strings"] = "; ".join(
            f"{k} ({v})" for k, v in self.discarded["viewpoint"].most_common()
        ) if "viewpoint" in tasks else ""
        r["step1_bug_memes_before_fix"] = self.step1_bug
        return r

    def top_discarded(self, k: int = 20) -> List[Dict[str, Any]]:
        rows = []
        for t in TASKS:
            for rank, (s, c) in enumerate(self.discarded[t].most_common(k), 1):
                rows.append({**self.row, "task": t, "rank": rank, "discarded_string": s, "count": c})
        return rows


def _benchmark_stems() -> set:
    bench = json.load(open(BENCHMARK_JSON))
    return {Path(e["image_filename"]).stem for e in bench}


def diag_raw_run(experiment: str, model_dir: Path, stems: set) -> Optional[Diag]:
    img_dirs = [d for d in sorted(model_dir.iterdir())
                if d.is_dir() and (d / "final_output.json").is_file() and d.name in stems]
    if not img_dirs:
        return None
    dg = Diag(experiment, model_dir.name, f"raw_text ({model_dir.relative_to(REPO_ROOT)})")
    dg.step1_bug = 0
    for d in img_dirs:
        final = json.load(open(d / "final_output.json", encoding="utf-8"))
        parsed = load_step_outputs(final)
        dg.n += 1
        for s in (1, 2, 3):
            if parsed[s] is None:
                dg.unparseable[s] += 1
        ids = raw_identities(parsed[1])
        dg.add_labels("identity", ids)
        dg.add_labels("viewpoint", raw_viewpoints(parsed[2]))
        dg.add_labels("role", raw_roles(parsed[3]))
        if ids and _old_step1_empty(final):
            dg.step1_bug += 1
    return dg


def _literal(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    try:
        out = ast.literal_eval(v) if isinstance(v, str) and v.strip() else []
    except (ValueError, SyntaxError):
        return []
    return [str(x) for x in out] if isinstance(out, list) else []


def diag_csv_run(experiment: str, model: str, csv_path: Path, shown_path: Path) -> Diag:
    import pandas as pd

    df = pd.read_csv(csv_path)
    dg = Diag(experiment, model,
              f"formatted_predictions.csv only ({shown_path.relative_to(REPO_ROOT)}); "
              "no raw outputs: parse failures and pre-extraction identity discards not countable")
    dg.n = len(df)
    for v in df["predicted_viewpoints"]:
        dg.add_labels("viewpoint", _literal(v))
    for v in df["predicted_narrative_roles"]:
        dg.add_labels("role", _literal(v))
    return dg


def diag_ablation(cond_dir: Path) -> Optional[Diag]:
    f = cond_dir / "results.jsonl"
    recs = [json.loads(l) for l in open(f) if l.strip()] if f.is_file() else []
    if not recs:
        return None
    dg = Diag("template_ablation", f"{cond_dir.parent.name}/{cond_dir.name}",
              "results.jsonl (extracted roles only; failed = success:false)")
    dg.n = len(recs)
    for r in recs:
        if not r.get("success", False):
            dg.unparseable[3] += 1
        dg.add_labels("role", r.get("predicted_narrative_roles") or [])
    return dg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DIAGNOSTICS_DIR)
    ap.add_argument("--derived-root", type=Path, default=DATA,
                    help="Where recompute_all_metrics wrote the formatted_predictions.csv files it "
                         "regenerates (e.g. the human evaluators'); read from here when present, "
                         f"else from {DATA} (default: {DATA}).")
    args = ap.parse_args()
    logging.disable(logging.WARNING)  # try_parse_json logs every failure

    stems = _benchmark_stems()
    rows, tops = [], []

    def emit(dg: Optional[Diag], steps, tasks):
        if dg is not None:
            rows.append(dg.finish(steps, tasks))
            tops.extend(dg.top_discarded())

    # Raw runs are diagnosed from their per-image outputs; runs that only exist
    # as formatted_predictions.csv (Gemini's GT-conditioned record in raw/, the
    # human evaluators in metrics/) from that CSV.
    for experiment, raw_root, metrics_root in [
        ("gt_conditioned", RAW_GT_CONDITIONED_DIR, GT_CONDITIONED_METRICS_DIR),
        ("no_gt", RAW_NO_GT_DIR, NO_GT_METRICS_DIR),
    ]:
        names = sorted({p.name for p in raw_root.iterdir() if p.is_dir()}
                       | {p.name for p in metrics_root.iterdir() if p.is_dir() and p.name != "all_metrics"})
        for name in names:
            mdir = raw_root / name
            dg = diag_raw_run(experiment, mdir, stems) if mdir.is_dir() else None
            if dg is not None:
                emit(dg, (1, 2, 3), TASKS)
                continue
            shown = mdir / "formatted_predictions.csv"
            if not shown.is_file():
                shown = metrics_root / name / "formatted_predictions.csv"
            if not shown.is_file():
                continue
            derived = args.derived_root / shown.relative_to(BENCHMARK_DIR)
            emit(diag_csv_run(experiment, name, derived if derived.is_file() else shown, shown),
                 (), ("viewpoint", "role"))

    for cond in sorted(TEMPLATE_ABLATION_DIR.glob("*/*")):
        if cond.is_dir():
            emit(diag_ablation(cond), (3,), ("role",))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    for r in rows:
        cols.extend(k for k in r if k not in cols)
    with open(args.out_dir / "diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(args.out_dir / "discarded_labels_top20.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "model", "source", "task", "rank", "discarded_string", "count"])
        w.writeheader()
        w.writerows(tops)
    print(f"Wrote {len(rows)} rows to {args.out_dir / 'diagnostics.csv'} "
          f"and {len(tops)} rows to {args.out_dir / 'discarded_labels_top20.csv'}")


if __name__ == "__main__":
    main()
