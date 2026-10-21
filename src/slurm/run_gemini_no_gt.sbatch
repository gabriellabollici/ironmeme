#!/bin/bash
#SBATCH --job-name=gemini-no-gt
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# No-ground-truth run with gemini-3.6-flash via Google's hosted API (see
# run_gemini_no_gt.sh). The model runs on Google's side, so the GPU here is
# idle -- a CPU partition works just as well and costs fewer credits, e.g.
#   sbatch --partition=rome --gres=none src/slurm/run_gemini_no_gt.sbatch
# ~15-20 s per image for all 3 steps -> ~5-6 h for 1109 images.
#
# Needs the API key in <repo root>/.env (GEMINI_API_KEY=...). It is
# gitignored, so copy it to Snellius separately -- never commit it.
#
# stdout and stderr share ONE file; logs/ must exist in the submit dir.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#
# Usage:
#   cd <repo root>
#   mkdir -p logs
#   LIMIT_IMAGES=5 sbatch src/slurm/run_gemini_no_gt.sbatch   # quick test
#   sbatch src/slurm/run_gemini_no_gt.sbatch                  # full run

set -euo pipefail

MODEL_ID="${MODEL_ID:-gemini-3.6-flash}"
LIMIT_IMAGES="${LIMIT_IMAGES:-}"

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

export PYTHONUNBUFFERED=1
export IMAGES_DIR="${IMAGES_DIR:-${REPO_ROOT}/benchmark/benchmark_dataset/images}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark/out_no_gt}"
export MODEL_SHORT_NAME="${MODEL_ID}"
export MODEL_ID

bash src/slurm/install_inference_deps.sh python

ARGS=(--clean-output)
[[ -n "${LIMIT_IMAGES}" ]] && ARGS+=(--limit-images "${LIMIT_IMAGES}")

# run_pipeline exits 1 if any image failed validation; keep going to scoring.
set +e
bash src/slurm/run_gemini_no_gt.sh "${ARGS[@]}"
PIPELINE_EXIT=$?
set -e
echo "Pipeline exit code: ${PIPELINE_EXIT}"

python -m experiments.evaluation.format_predictions
python -m experiments.evaluation.evaluate --model "${MODEL_SHORT_NAME}-no-gt"

echo ""
echo "Done. Results in ${OUTPUT_DIR}/${MODEL_SHORT_NAME}/"
