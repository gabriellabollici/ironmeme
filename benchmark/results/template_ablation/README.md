# Template ablation: does the meme template description help?

Narrative-role classification (pipeline step 3) given the **ground-truth**
social identities and viewpoints, run twice on all 1109 memes with Gemini 3.6
Flash:

- **no template**: prompt `src/experiments/template_ablation/prompts/04b_narrative_roles_notemplate.txt`
  (the same prompt as the general pipeline's step 3)
- **with template**: prompt `src/experiments/template_ablation/prompts/04_template_roles.txt`,
  which adds the meme template's description (`template_description` in the
  benchmark JSON)

Script: `src/experiments/template_ablation/ablation.py`; metrics:
`src/experiments/template_ablation/compute_metrics.py` (same scoring logic as
`evaluate.py`). Settings: `gemini-3.6-flash` via Google's OpenAI-compatible API, max_tokens 16384,
temperature 0, 2 retries, `--json-mode freeform`; SLURM: `src/slurm/run_gemini_template_ablation.sbatch`.

## Results (all values %)

All 1109 memes were scored in both variants (0 failed predictions).

**Set-level**

| Variant | Exact Match | Mean Jaccard |
|---|---:|---:|
| no template | 42.4 | 61.5 |
| **with template** | 37.5 | 58.4 |
| *with − without* | *-4.9* | *-3.2* |

**Micro**

| Variant | P | R | F1 |
|---|---:|---:|---:|
| no template | 60.3 | 77.7 | 67.9 |
| **with template** | 57.1 | 76.0 | 65.2 |
| *with − without* | *-3.2* | *-1.7* | *-2.7* |

**Macro**

| Variant | P | R | F1 |
|---|---:|---:|---:|
| no template | 60.5 | 77.1 | 65.2 |
| **with template** | 55.6 | 75.5 | 61.9 |
| *with − without* | *-4.9* | *-1.6* | *-3.3* |

**Weighted**

| Variant | P | R | F1 |
|---|---:|---:|---:|
| no template | 66.3 | 77.7 | 69.0 |
| **with template** | 62.6 | 76.0 | 66.4 |
| *with − without* | *-3.8* | *-1.7* | *-2.6* |

The template description does not help: Gemini scores lower on every metric
with it. Both precision and recall drop, so it predicts more wrong roles, not
just more roles; the drop is mainly in precision (macro −4.9 vs. recall −1.6).

### Consistency check: general pipeline run

The "no template" variant is the same task as step 3 of the general
GT-conditioned pipeline run, which was a separate run with its own settings.
Its narrative-role scores match the "no template" row above, so the comparison
is not an artifact of the ablation script or settings:

| Source | Exact Match | Mean Jaccard | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|
| `benchmark/results/raw/gt_conditioned/gemini-3.6-flash/` | 42.2 | 61.0 | 67.3 | 64.1 |

## Files

`gemini-3.6-flash/{with_template,no_template}/`: `results.jsonl` (raw per-meme
outputs of the run), `predictions.csv`, `stats.json` and `metrics.json`
(rebuilt by `recompute_all_metrics`).
