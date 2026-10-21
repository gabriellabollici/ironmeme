Subset-matched VLM-vs-human comparison for the ironmeme benchmark
Generated: 2026-09-17 | 8 VLMs + 1 pooled human row x 6 tasks, n=214 memes (identical set for every row)

Why this exists
----------------
The human evaluators (experiments/evaluation/human_evaluators/*.json) only labeled
a subset of the benchmark, so their rows in
benchmark/results/metrics/gt_conditioned/all_metrics/all_models_metrics.csv (n=215
for the pooled "human" row) aren't directly comparable to the VLM rows
there (n=1109/1108/1100, the full benchmark). This directory holds every
model (VLMs + human) re-scored on the exact same 214-meme subset, built by
src/experiments/evaluation/recompute_all_metrics.py (originally build_subset_comparison.py):

  1. Start from the pooled human subset (215 memes).
  2. Intersect with every VLM's covered memes (each VLM is missing image
     "415" from its predictions for unrelated reasons -- an image no VLM
     produced output for -- and pixtral/qwen2.5 are separately missing a
     handful of others), leaving 214 memes common to all 9 rows.
  3. Restrict every model's (and human's) formatted_predictions.csv to
     that shared set and re-run evaluate.py.

"Viewpoint (per-identity alignment)" now included for all 9 rows
------------------------------------------------------------------
The VLM formatted_predictions.csv files under
benchmark/results/metrics/gt_conditioned/<model>/ used to predate the
viewpoint_gt_by_identity / predicted_viewpoints_by_identity columns (see
that directory's README, "Why 'corrected'" section, item 3), so this
task was missing for every VLM here. Fixed 2026-09-17 by re-running
format_predictions.py against the raw per-image final_output.json files
under benchmark/results/raw/gt_conditioned/<model>/ for all 8 models (current code
already writes those columns) and re-scoring with evaluate.py -- both
here and in benchmark/results/metrics/gt_conditioned/ itself. The other 5 tasks'
numbers are unchanged by this (verified byte-identical row counts and
scores before/after), since this only added the two missing columns.

Contents
--------
<model>/formatted_predictions.csv   Each model's predictions restricted to
                                     the 214-meme shared subset.
<model>/metrics.json                Same schema as ironmeme_results_corrected.
all_models_metrics.csv              One row per model x task, same columns
                                     as ironmeme_results_corrected's.

See benchmark/results/metrics/gt_conditioned/all_metrics/README.txt for the task
definitions (Social Identity / Viewpoint / Narrative Roles / All Combined
Predictions / Narrative Roles (per-identity alignment)).
