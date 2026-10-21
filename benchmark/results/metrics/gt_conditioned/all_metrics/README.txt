Corrected per-model metrics for the ironmeme VLM benchmark
Generated: 2026-08-28, updated 2026-08-31, 2026-09-17, 2026-09-30 | 8 VLMs x 6 tasks

Contents
--------
<model>_metrics.json              Summary metrics per task.  Each file now has
                                  6 task rows: Social Identity, Viewpoint,
                                  Narrative Roles, All Combined Predictions,
                                  Narrative Roles (per-identity alignment),
                                  and Viewpoint (per-identity alignment).
all_models_metrics.csv            All models' summary metrics in one flat
                                  table, one row per model x task.

Tasks
-----
* Social Identity / Viewpoint / Narrative Roles / All Combined Predictions:
  exact-match ratio, mean Jaccard, micro/macro/weighted P/R/F1 (this is the
  original flat-set scoring).

* Narrative Roles (per-identity alignment): Scores narrative roles
  aligned to the social identity they belong to (the GT is a positional
  identity->role assignment, e.g. image 1: democrat->loser, republican->
  winner).  Columns:
    - GT Identities : number of ground-truth identity->role pairs scored
    - Role Coverage : % of GT identities that received >=1 predicted role
    - Role Hit      : % of GT identities whose GT role is present among the
                      predicted roles for that identity
    - Exact Role    : % of GT identities whose predicted role set exactly
                      equals the GT role set
    - Per-Identity Jaccard : mean per-identity Jaccard over GT identities

* Viewpoint (per-identity alignment): NEW (2026-08-31).  Mirrors the
  narrative-role alignment above for viewpoints: scores whether the
  ingroup/outgroup label the model assigns is attached to the *correct*
  social identity, instead of just checking the flat bag of viewpoint
  labels (which lets a model that swaps which identity is ingroup vs.
  outgroup score identically to one that gets it right).  Columns:
    - GT Identities      : number of ground-truth identity->viewpoint pairs
                            scored
    - Viewpoint Coverage : % of GT identities that received a predicted
                            viewpoint
    - Viewpoint Hit      : % of GT identities whose GT viewpoint is present
                            among the predicted viewpoint(s) for that identity
    - Exact Viewpoint     : % of GT identities whose predicted viewpoint
                            exactly equals the GT viewpoint
    - Per-Identity Jaccard : mean per-identity Jaccard over GT identities
  (Viewpoint is single-valued per identity, so Viewpoint Hit and Exact
  Viewpoint coincide unless a model attaches more than one viewpoint label
  to the same identity.)

Why "corrected"
---------------
(1) format_predictions.py no longer drops a single-identity role result
    ({"identity", "roles"}) -- glm previously had its flat Narrative Roles
    Jaccard deflated from ~40% to 9.86% because that shape was discarded.
(2) Per-identity role alignment is now scored explicitly (identity<->role
    correspondence was previously never checked; roles were flattened into
    a bag, so wrong-identity assignments scored identically to correct ones).
(3) Per-identity viewpoint alignment (2026-08-31): the same identity<->label
    correspondence gap existed for viewpoints -- extract_viewpoints_only()
    discarded the identity each predicted viewpoint was attached to before
    scoring, even though the model's raw step-2 output and the GT both
    carry that pairing. format_predictions.py now also preserves it via
    extract_identity_viewpoints()/normalise_gt_identity_viewpoints(), scored
    by evaluate.py's new evaluate_identity_viewpoints().

    NOTE: when (3) landed, only all_models_metrics.csv was regenerated
    from fresher per-model runs -- the individual <model>/metrics.json
    and formatted_predictions.csv files here were never updated to
    match, so every VLM was silently missing the "Viewpoint
    (per-identity alignment)" row (only the human rows added 2026-09-17
    had it). Fixed 2026-09-17: re-ran format_predictions.py against the
    raw per-image outputs in benchmark/results/raw/gt_conditioned/<model>/ for all 8
    VLMs and re-scored with evaluate.py; row counts and every other
    task's scores are unchanged (verified), only the two per-identity-
    viewpoint columns were newly populated.
(4) glm flat Viewpoint (2026-09-30): the same single-identity-dict gap as
    (1), but in the flat viewpoint extraction. vlm_client._try_parse_json()
    parsed glm's one-item answers (<answer>[{...}]) as the bare inner
    {"identity", "viewpoint"} object, and extract_viewpoints_only() then
    scored that object as no prediction. 815/1109 glm images were affected,
    deflating its flat Viewpoint Micro F1 from 0.8216 to 0.4764 (and All
    Combined Micro F1 from 0.6012 to 0.4990). Fixed in both places; glm was
    re-scored from its raw outputs in benchmark/results/raw/gt_conditioned/. Only glm's
    Viewpoint and All Combined rows changed (verified): its other four
    task rows are identical, and no other model ever returned that shape.

Scoring protocol identical across models (see format_predictions.py /
evaluate.py; blank GT viewpoints coerced to outgroup as in prior runs).
Generated separately from benchmark/results/raw/gt_conditioned/ (the original experiment
records remain untouched).

Human evaluator baseline
------------------------
Moved to benchmark/results/metrics/no_gt/ (2026-09-30). Every VLM here is
given the ground-truth social identities (Step 2) and identity/viewpoint
pairs (Step 3), which the human annotators never saw, so the two are not
comparable. The humans are compared instead against the no-ground-truth
pipeline run (scripts/run_pipeline_no_gt.py), where the model -- like the
annotators -- derives viewpoints and roles from its own identities.
