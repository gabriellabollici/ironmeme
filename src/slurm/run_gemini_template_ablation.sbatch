#!/bin/bash
#SBATCH --job-name=gemini-template-ablation
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# Template ablation (experiments/template_ablation/ablation.py) with gemini-3.6-flash via
# Google's hosted OpenAI-compatible API: Step 3 (narrative roles) given the GT
# social identities + GT viewpoints, once WITH the meme's template description
# in the prompt and once WITHOUT it. max_tokens 16384, temperature 0, 2 retries,
# --json-mode freeform (prompt-only JSON + the script's bracket-matching parser).
#
# The model runs on Google's side, so the GPU is idle -- a CPU partition
# works just as well and costs fewer credits, e.g.
#   sbatch --partition=rome --gres=none src/slurm/run_gemini_template_ablation.sbatch
# ~4-9 s per call -> ~3-5 h for both variants (2 x 1109 images).
#
# Needs the API key in <repo root>/.env (GEMINI_API_KEY=...), gitignored.
# Writes to benchmark/results/template_ablation/${RUN_NAME}/.
#
# stdout and stderr share ONE file; logs/ must exist in the submit dir.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#
# Usage:
#   cd <repo root>
#   mkdir -p logs
#   LIMIT=5 sbatch src/slurm/run_gemini_template_ablation.sbatch   # quick test
#   sbatch src/slurm/run_gemini_template_ablation.sbatch           # full run
#   VARIANTS=no_template sbatch src/slurm/run_gemini_template_ablation.sbatch   # one variant only

set -euo pipefail

MODEL_ID="${MODEL_ID:-gemini-3.6-flash}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RETRIES="${RETRIES:-2}"
RUN_NAME="${RUN_NAME:-gemini-3.6-flash}"
BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}"
LIMIT="${LIMIT:-}"
# Space-separated subset of: with_template no_template
VARIANTS="${VARIANTS:-with_template no_template}"

CONDA_ENV="${CONDA_ENV:-narrative-memes2}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "${CONDA_ENV}" || {
    echo "[FAIL] Conda env '${CONDA_ENV}' not found"
    exit 1
}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
[[ -f .env ]] || { echo "[FAIL] ${REPO_ROOT}/.env not found (needs GEMINI_API_KEY=...)"; exit 1; }
set -a; . ./.env; set +a
: "${GEMINI_API_KEY:?GEMINI_API_KEY missing from .env}"
export VLLM_API_KEY="${GEMINI_API_KEY}"
# Ride out 429/5xx with SDK backoff rather than spending a --retries attempt.
export VLM_MAX_RETRIES="${VLM_MAX_RETRIES:-8}"
export PYTHONUNBUFFERED=1

if [[ -n "${LIMIT}" ]]; then
    RUN_NAME="${RUN_NAME}_test${LIMIT}"
fi

echo "Run name:  ${RUN_NAME}"
echo "Model:     ${MODEL_ID}  (max_tokens=${MAX_TOKENS}, temperature=${TEMPERATURE}, retries=${RETRIES})"
echo "Limit:     ${LIMIT:-<all>}"
echo "Variants:  ${VARIANTS}"
echo ""

bash src/slurm/install_inference_deps.sh python

EXTRA=()
[[ -n "${LIMIT}" ]] && EXTRA+=(--limit "${LIMIT}")

for VARIANT in ${VARIANTS}; do
    python -m experiments.template_ablation.ablation \
        --variant "${VARIANT}" \
        --run-name "${RUN_NAME}" \
        --model "${MODEL_ID}" \
        --max-tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --retries "${RETRIES}" \
        --json-mode freeform \
        --base-url "${BASE_URL}" \
        ${EXTRA[@]+"${EXTRA[@]}"}
done

python -m experiments.template_ablation.compute_metrics \
    --run-dir "benchmark/results/template_ablation/${RUN_NAME}" \
    --model "${MODEL_ID}"

echo ""
echo "Done. Results in benchmark/results/template_ablation/${RUN_NAME}/{with_template,no_template}/"
