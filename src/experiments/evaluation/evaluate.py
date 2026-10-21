#!/usr/bin/env python3
"""Evaluate formatted predictions against ground truth.

Reads the CSV produced by ``format_predictions.py`` and computes the same
metrics as the notebook: Exact Match Ratio, Mean Jaccard, per-class
classification reports, and Micro / Macro / Weighted F1.

Usage:
    python -m experiments.evaluation.evaluate [--input PATH] [--output PATH]
"""
import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from dataset_construction.labels import canon_role, canon_viewpoint, inventory

logger = logging.getLogger(__name__)

# Fixed metric classes per task (lowercase, as produced by clean_and_normalize).
TASK_CLASSES: Dict[str, List[str]] = {
    "Social Identity": inventory("identity"),
    "Viewpoint": inventory("viewpoint"),
    "Narrative Roles": inventory("role"),
    "All Combined Predictions": inventory("identity") + inventory("viewpoint") + inventory("role"),
}


# ---------------------------------------------------------------------------
# clean_and_normalize (matches notebook cell 12 exactly)
# ---------------------------------------------------------------------------
def clean_and_normalize(val: Any) -> List[str]:
    """Normalise a cell value into a flat, deduplicated list of lowercase strings."""
    # 1. Handle NaNs / None
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []

    # 2. Convert Arrays/Series to standard Python objects
    if isinstance(val, (np.ndarray, pd.Series)):
        val = val.tolist()

    # 3. If it's a string, try parsing stringified lists
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str.lower() in ["nan", "none", "[]", "['']"]:
            return []
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                val = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                val = re.findall(r"['\"]([^'\"]+)['\"]", val_str)

    # 4. Single string token
    if isinstance(val, str):
        cleaned = val.strip().lower()
        return [cleaned] if cleaned else []

    # 5. List (or nested list)
    if isinstance(val, list):
        extracted: List[str] = []
        for item in val:
            if isinstance(item, list):
                for sub_item in item:
                    s = str(sub_item).strip().lower()
                    if s and s not in ["nan", "none"]:
                        extracted.append(s)
            else:
                s = str(item).strip().lower()
                s = s.strip("'\"").strip("[").strip("]").strip()
                if s and s not in ["nan", "none", ""]:
                    extracted.append(s)
        return list(dict.fromkeys(extracted))

    return []


# ---------------------------------------------------------------------------
# Merge helpers (match notebook)
# ---------------------------------------------------------------------------
def merge_row_predictions(row: pd.Series) -> List[str]:
    all_preds = (
        row["predicted_social_identities"]
        + row["predicted_viewpoints"]
        + row["predicted_narrative_roles"]
    )
    return list(dict.fromkeys(all_preds))


def merge_row_gt(row: pd.Series) -> List[str]:
    all_gt = (
        row["social_identity_gt"]
        + row["viewpoint_gt"]
        + row["narrative_roles_gt"]
    )
    return list(dict.fromkeys(all_gt))


# ---------------------------------------------------------------------------
# Per-identity narrative role alignment
# ---------------------------------------------------------------------------
def _norm_identity_key(s: Any) -> str:
    """Lowercase, strip non-alphanumerics -> alignment key.  Identity labels
    from both the GT and prediction sides may differ in casing/whitespace
    (e.g. 'republic an' vs 'Republican'), so alignment is done on a
    punctuation-insensitive key."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _parse_identity_roles(val: Any) -> List[Dict[str, Any]]:
    """Parse a cell holding a list of {identity, roles} dicts (as written by
    format_predictions.py) back into a Python list.  The CSV stores it as a
    stringified Python repr, e.g. "[{'identity': 'Democrat', 'roles': ['fool']}]",
    which json.loads can't read (single quotes) but ast.literal_eval can."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val or val in ("[]", "nan", "None"):
            return []
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return parsed
        return []
    return []


def evaluate_identity_roles(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    target_name: str,
    summary_metrics: List[Dict[str, Any]],
) -> None:
    """Score narrative roles aligned to their social identity.

    Reads the per-identity role columns (list of {identity, roles} dicts)
    written by format_predictions.py.  For every ground-truth identity it
    checks whether the model assigned a role to THAT identity and whether that
    role matches, so a model that guesses the right roles but pairs them with
    the wrong identities is correctly penalised.

    Metrics (per GT identity): coverage rate (identity present with >=1 role),
    role-hit rate (GT role present among the model's roles for that identity),
    exact-match rate (role sets equal), and per-identity Jaccard.
    """
    print("\n" + "=" * 60)
    print(f" EVALUATING (per-identity alignment): {target_name.upper()}")
    print(f" GT Column: '{gt_col}'  |  Pred Column: '{pred_col}'")
    print("=" * 60)

    n_gt = 0
    n_covered = 0
    n_hit = 0
    n_correct = 0
    jac_sum = 0.0

    for _, row in df.iterrows():
        gt_list = row[gt_col] if isinstance(row[gt_col], list) else []
        pred_list = row[pred_col] if isinstance(row[pred_col], list) else []

        # Aggregate per-identity into identity -> set(roles), merging repeated
        # identities (e.g. GT 'women','women' -> {fool, loser}).
        gt_map: Dict[str, set] = {}
        for e in gt_list:
            if not (isinstance(e, dict) and "identity" in e):
                continue
            gt_map.setdefault(_norm_identity_key(e["identity"]), set()).update(
                r for r in (canon_role(x) for x in (e.get("roles") or [])) if r
            )
        pred_map: Dict[str, set] = {}
        for e in pred_list:
            if not (isinstance(e, dict) and "identity" in e):
                continue
            pred_map.setdefault(_norm_identity_key(e["identity"]), set()).update(
                r for r in (canon_role(x) for x in (e.get("roles") or [])) if r
            )

        for key, gr in gt_map.items():
            pr = pred_map.get(key, set())
            n_gt += 1
            if pr:
                n_covered += 1
            if gr and gr <= pr:
                n_hit += 1
            if gr and gr == pr:
                n_correct += 1
            if gr or pr:
                jac_sum += len(gr & pr) / len(gr | pr)
            else:
                jac_sum += 1.0

    def pct(x: float) -> str:
        return f"{100 * x:.2f}%"

    coverage = n_covered / n_gt if n_gt else 0.0
    hit = n_hit / n_gt if n_gt else 0.0
    correct = n_correct / n_gt if n_gt else 0.0
    ja = jac_sum / n_gt if n_gt else 0.0

    print(f"GT identities evaluated        : {n_gt}")
    print(f"Role assigned to identity      : {pct(coverage)} (coverage)")
    print(f"GT role present for identity   : {pct(hit)} (role-hit)")
    print(f"Exact role set per identity    : {pct(correct)}")
    print(f"Per-identity Mean Jaccard      : {pct(ja)}")

    summary_metrics.append({
        "Task": target_name,
        "GT Identities": n_gt,
        "Role Coverage": pct(coverage),
        "Role Hit": pct(hit),
        "Exact Role": pct(correct),
        "Per-Identity Jaccard": pct(ja),
    })


def evaluate_identity_viewpoints(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    target_name: str,
    summary_metrics: List[Dict[str, Any]],
) -> None:
    """Score viewpoints aligned to their social identity.

    Mirrors ``evaluate_identity_roles`` for the viewpoint task: reads the
    per-identity viewpoint columns (list of {identity, viewpoint} dicts)
    written by format_predictions.py. For every ground-truth identity it
    checks whether the model assigned a viewpoint to THAT identity and
    whether it matches, so a model that gets the right ingroup/outgroup
    labels overall but attaches them to the wrong identities (e.g. calls
    "democrats" ingroup when it's actually "republicans") is correctly
    penalised instead of scoring identically to a model that pairs them
    correctly.

    Metrics (per GT identity): coverage rate (identity assigned a
    viewpoint), viewpoint-hit rate, exact-match rate, and per-identity
    Jaccard. Since viewpoint is single-valued (ingroup xor outgroup) rather
    than a multi-label set like roles, hit rate and exact-match rate
    coincide here, but are both reported for symmetry with the roles metric.
    """
    print("\n" + "=" * 60)
    print(f" EVALUATING (per-identity alignment): {target_name.upper()}")
    print(f" GT Column: '{gt_col}'  |  Pred Column: '{pred_col}'")
    print("=" * 60)

    n_gt = 0
    n_covered = 0
    n_hit = 0
    n_correct = 0
    jac_sum = 0.0

    for _, row in df.iterrows():
        gt_list = row[gt_col] if isinstance(row[gt_col], list) else []
        pred_list = row[pred_col] if isinstance(row[pred_col], list) else []

        # Aggregate per-identity into identity -> set(viewpoints), merging
        # repeated identities the same way evaluate_identity_roles() does.
        gt_map: Dict[str, set] = {}
        for e in gt_list:
            if not (isinstance(e, dict) and "identity" in e and e.get("viewpoint")):
                continue
            vp = canon_viewpoint(e["viewpoint"])
            if vp:
                gt_map.setdefault(_norm_identity_key(e["identity"]), set()).add(vp)
        pred_map: Dict[str, set] = {}
        for e in pred_list:
            if not (isinstance(e, dict) and "identity" in e and e.get("viewpoint")):
                continue
            vp = canon_viewpoint(e["viewpoint"])
            if vp:
                pred_map.setdefault(_norm_identity_key(e["identity"]), set()).add(vp)

        for key, gr in gt_map.items():
            pr = pred_map.get(key, set())
            n_gt += 1
            if pr:
                n_covered += 1
            if gr and gr <= pr:
                n_hit += 1
            if gr and gr == pr:
                n_correct += 1
            if gr or pr:
                jac_sum += len(gr & pr) / len(gr | pr)
            else:
                jac_sum += 1.0

    def pct(x: float) -> str:
        return f"{100 * x:.2f}%"

    coverage = n_covered / n_gt if n_gt else 0.0
    hit = n_hit / n_gt if n_gt else 0.0
    correct = n_correct / n_gt if n_gt else 0.0
    ja = jac_sum / n_gt if n_gt else 0.0

    print(f"GT identities evaluated        : {n_gt}")
    print(f"Viewpoint assigned to identity : {pct(coverage)} (coverage)")
    print(f"GT viewpoint present for id.   : {pct(hit)} (viewpoint-hit)")
    print(f"Exact viewpoint per identity   : {pct(correct)}")
    print(f"Per-identity Mean Jaccard      : {pct(ja)}")

    summary_metrics.append({
        "Task": target_name,
        "GT Identities": n_gt,
        "Viewpoint Coverage": pct(coverage),
        "Viewpoint Hit": pct(hit),
        "Exact Viewpoint": pct(correct),
        "Per-Identity Jaccard": pct(ja),
    })


# ---------------------------------------------------------------------------
# evaluate_target (matches notebook cell 12 exactly)
# ---------------------------------------------------------------------------
def score_label_sets(
    y_true: List[List[str]], y_pred: List[List[str]], classes: List[str],
) -> Dict[str, Any]:
    """Set-level and P/R/F1 scores for multi-label sets over a fixed inventory.

    Labels outside *classes* are dropped from both sides first (they should
    already have been discarded upstream; the count is returned so leaks are
    visible). The binarizer is fit on *classes*, not on the observed labels,
    so an invented label can never become an extra zero-F1 class. Macro
    averages are taken over the inventory labels that have GT support or at
    least one prediction in the evaluated set: a label with neither has an
    undefined F1 and would otherwise count as 0. Micro and weighted averages
    are unaffected by that choice. zero_division=0 throughout.
    """
    allowed = set(classes)
    dropped = 0
    t_sets, p_sets = [], []
    for gt, pred in zip(y_true, y_pred):
        g = {x for x in gt if x in allowed}
        p = {x for x in pred if x in allowed}
        dropped += len(set(gt) - g) + len(set(pred) - p)
        t_sets.append(g)
        p_sets.append(p)

    jaccards = [len(g & p) / len(g | p) if (g | p) else 1.0 for g, p in zip(t_sets, p_sets)]
    exacts = [g == p for g, p in zip(t_sets, p_sets)]

    mlb = MultiLabelBinarizer(classes=classes)
    yt = mlb.fit_transform(t_sets)
    yp = mlb.transform(p_sets)
    active = [i for i in range(len(classes)) if yt[:, i].any() or yp[:, i].any()]

    out: Dict[str, Any] = {
        "n": len(t_sets),
        "exact_match": float(np.mean(exacts)) if exacts else 0.0,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "macro_classes": len(active),
        "dropped_off_inventory": dropped,
        "yt": yt, "yp": yp, "active": active, "classes": classes,
    }
    for avg in ("micro", "macro", "weighted"):
        if active:
            p, r, f, _ = precision_recall_fscore_support(
                yt, yp, labels=active, average=avg, zero_division=0,
            )
        else:
            p = r = f = 0.0
        out[f"{avg}_precision"], out[f"{avg}_recall"], out[f"{avg}_f1"] = float(p), float(r), float(f)
    return out


def evaluate_target(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    target_name: str,
    summary_metrics: List[Dict[str, Any]],
    classes: List[str],
) -> None:
    print("\n" + "=" * 60)
    print(f" EVALUATING: {target_name.upper()}")
    print(f" GT Column: '{gt_col}'  |  Pred Column: '{pred_col}'")
    print("=" * 60)

    sc = score_label_sets(list(df[gt_col]), list(df[pred_col]), classes)
    if sc["dropped_off_inventory"]:
        logger.warning("%s: dropped %d off-inventory labels at scoring time",
                       target_name, sc["dropped_off_inventory"])

    print(f"Exact Match Ratio (Subset Accuracy): {sc['exact_match']:.2%}")
    print(f"Mean Jaccard Overlap Score         : {sc['mean_jaccard']:.2%}")
    print(f"Macro-averaged classes             : {sc['macro_classes']}/{len(classes)}")

    if sc["active"]:
        print("\nClassification Report:")
        print(
            classification_report(
                sc["yt"],
                sc["yp"],
                labels=sc["active"],
                target_names=[classes[i] for i in sc["active"]],
                zero_division=0,
            )
        )
    else:
        print("\nNo valid classes extracted to evaluate.")

    summary_metrics.append({
        "Task": target_name,
        "Exact Match Ratio": f"{sc['exact_match']:.2%}",
        "Mean Jaccard": f"{sc['mean_jaccard']:.2%}",
        "Micro Precision": f"{sc['micro_precision']:.4f}",
        "Micro Recall": f"{sc['micro_recall']:.4f}",
        "Micro F1": f"{sc['micro_f1']:.4f}",
        "Macro Precision": f"{sc['macro_precision']:.4f}",
        "Macro Recall": f"{sc['macro_recall']:.4f}",
        "Macro F1": f"{sc['macro_f1']:.4f}",
        "Weighted Precision": f"{sc['weighted_precision']:.4f}",
        "Weighted Recall": f"{sc['weighted_recall']:.4f}",
        "Weighted F1": f"{sc['weighted_f1']:.4f}",
        "Macro Classes": sc["macro_classes"],
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate formatted predictions against ground truth."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to formatted_predictions.csv (default: out/<model>/formatted_predictions.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the metrics JSON output (default: out/<model>/metrics.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name recorded in the metrics JSON (default: MODEL_SHORT_NAME "
             "from config; falls back to the input CSV's parent directory name).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    )

    from experiments.inference.config import OUTPUT_DIR, MODEL_SHORT_NAME

    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path(OUTPUT_DIR) / MODEL_SHORT_NAME / "formatted_predictions.csv"

    if not input_path.is_file():
        logger.error("Input CSV not found: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)

    # ---------------------------------------------------------------
    # 1. Apply clean_and_normalize to prediction columns
    # ---------------------------------------------------------------
    pred_cols = [
        "predicted_social_identities",
        "predicted_viewpoints",
        "predicted_narrative_roles",
    ]
    for col in pred_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_and_normalize)

    # ---------------------------------------------------------------
    # 2. Merge predictions into combined column
    # ---------------------------------------------------------------
    if all(c in df.columns for c in pred_cols):
        df["combined_predictions"] = df.apply(merge_row_predictions, axis=1)

    # ---------------------------------------------------------------
    # 3. Apply clean_and_normalize to GT columns
    # ---------------------------------------------------------------
    gt_cols = [
        "social_identity_gt",
        "viewpoint_gt",
        "narrative_roles_gt",
    ]
    for col in gt_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_and_normalize)

    # ---------------------------------------------------------------
    # 4. Merge GT into combined column
    # ---------------------------------------------------------------
    if all(c in df.columns for c in gt_cols):
        df["full_gt_combined"] = df.apply(merge_row_gt, axis=1)

    # ---------------------------------------------------------------
    # 5. Run evaluations
    # ---------------------------------------------------------------
    summary_metrics: List[Dict[str, Any]] = []

    if "social_identity_gt" in df.columns and "predicted_social_identities" in df.columns:
        evaluate_target(
            df, "social_identity_gt", "predicted_social_identities",
            "Social Identity", summary_metrics, TASK_CLASSES["Social Identity"],
        )

    if "viewpoint_gt" in df.columns and "predicted_viewpoints" in df.columns:
        evaluate_target(
            df, "viewpoint_gt", "predicted_viewpoints",
            "Viewpoint", summary_metrics, TASK_CLASSES["Viewpoint"],
        )

    if "narrative_roles_gt" in df.columns and "predicted_narrative_roles" in df.columns:
        evaluate_target(
            df, "narrative_roles_gt", "predicted_narrative_roles",
            "Narrative Roles", summary_metrics, TASK_CLASSES["Narrative Roles"],
        )

    if "full_gt_combined" in df.columns and "combined_predictions" in df.columns:
        evaluate_target(
            df, "full_gt_combined", "combined_predictions",
            "All Combined Predictions", summary_metrics, TASK_CLASSES["All Combined Predictions"],
        )

    # Per-identity narrative role alignment (needs the by-identity columns
    # written by format_predictions.py: lists of {identity, roles} dicts).
    id_gt_col = "narrative_roles_gt_by_identity"
    id_pred_col = "predicted_narrative_roles_by_identity"
    if id_gt_col in df.columns and id_pred_col in df.columns:
        df[id_gt_col] = df[id_gt_col].apply(
            lambda v: _parse_identity_roles(v)
        )
        df[id_pred_col] = df[id_pred_col].apply(
            lambda v: _parse_identity_roles(v)
        )
        evaluate_identity_roles(
            df, id_gt_col, id_pred_col,
            "Narrative Roles (per-identity alignment)", summary_metrics,
        )

    # Per-identity viewpoint alignment (needs the by-identity columns
    # written by format_predictions.py: lists of {identity, viewpoint} dicts).
    vp_id_gt_col = "viewpoint_gt_by_identity"
    vp_id_pred_col = "predicted_viewpoints_by_identity"
    if vp_id_gt_col in df.columns and vp_id_pred_col in df.columns:
        df[vp_id_gt_col] = df[vp_id_gt_col].apply(
            lambda v: _parse_identity_roles(v)
        )
        df[vp_id_pred_col] = df[vp_id_pred_col].apply(
            lambda v: _parse_identity_roles(v)
        )
        evaluate_identity_viewpoints(
            df, vp_id_gt_col, vp_id_pred_col,
            "Viewpoint (per-identity alignment)", summary_metrics,
        )

    # ---------------------------------------------------------------
    # 6. Summary table
    # ---------------------------------------------------------------
    summary_df = pd.DataFrame(summary_metrics)
    print("\n" + "=" * 60)
    print(" METRICS OVERVIEW SUMMARY TABLE")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # ---------------------------------------------------------------
    # 7. Save metrics
    # ---------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(OUTPUT_DIR) / MODEL_SHORT_NAME / "metrics.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = args.model or MODEL_SHORT_NAME
    if not args.model:
        # Running against a standalone results dir (no env/pipeline context):
        # prefer the directory name so the JSON isn't mislabeled with whatever
        # MODEL_SHORT_NAME happens to default to.
        dir_name = input_path.parent.name
        if dir_name and dir_name != "out":
            model_name = dir_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "total_samples": len(df),
                "metrics_summary": summary_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved metrics to %s", out_path)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
