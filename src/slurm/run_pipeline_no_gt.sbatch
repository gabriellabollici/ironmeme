#!/bin/bash
#SBATCH --job-name=vlm-pipeline-no-gt
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# No-ground-truth run of the 3-step classification pipeline (run_pipeline
# --no-gt): Step 2 gets Step 1's predicted identities,
# Step 3 gets Step 2's predicted viewpoints -- no GT is ever put in a prompt.
# All 3 steps are thinking-model calls at max_tokens=8192 per image (the
# template ablation ran only 1 step and took ~5h), and run_pipeline.py has
# no resume, so the time limit is generous: a timeout means starting over.
#
# stdout and stderr share ONE file (python logging writes to stderr).
# The logs/ dir must already exist in the directory you submit from.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#
# Usage:
#   cd <repo root>          (the IRONMeme folder)
#   mkdir -p logs
#   sbatch src/slurm/run_pipeline_no_gt.sbatch
# Quick test first:
#   LIMIT_IMAGES=5 sbatch src/slurm/run_pipeline_no_gt.sbatch

set -euo pipefail
set -x

MODEL_ID="${MODEL_ID:-zai-org/GLM-4.1V-9B-Thinking}"
MODEL_SHORT_NAME="${MODEL_SHORT_NAME:-$(echo "${MODEL_ID##*/}" | tr '[:upper:]_' '[:lower:]-')}"
LIMIT_IMAGES="${LIMIT_IMAGES:-}"

echo ""
echo " VLM Pipeline -- no ground truth (chained predictions)"
echo "========================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo ""

# ── 1. Modules ────────────────────────────────────────────────
module purge 2>/dev/null || true
module load cuda/12.4 2>/dev/null || echo "[WARN] CUDA module not available"

# ── 2. Conda ──────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-narrative-memes2}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "${CONDA_ENV}" || {
    echo "[FAIL] Conda env '${CONDA_ENV}' not found"
    exit 1
}

# ── 3. Paths ──────────────────────────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${REPO_ROOT}"

CONDA_PYTHON="$(which python)"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
bash src/slurm/migrate_hf_cache.sh "${REPO_ROOT}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export IMAGES_DIR="${IMAGES_DIR:-${REPO_ROOT}/benchmark/benchmark_dataset/images}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark/out_no_gt}"
export MODEL_SHORT_NAME
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-30000}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
export VLLM_SERVER_LOG_DIR="${VLLM_SERVER_LOG_DIR:-${OUTPUT_DIR}/vllm_logs}"

echo "Config:"
echo "  MODEL_ID         = ${MODEL_ID}"
echo "  MODEL_SHORT_NAME = ${MODEL_SHORT_NAME}"
echo "  IMAGES_DIR       = ${IMAGES_DIR}"
echo "  OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "  HF_HOME          = ${HF_HOME}"
echo "  LIMIT_IMAGES     = ${LIMIT_IMAGES:-<all>}"
echo ""

# ── 4. Install dependencies ─────────────────────────────────
bash src/slurm/install_inference_deps.sh "${CONDA_PYTHON}"

if [[ -z "${CUDA_HOME:-}" ]]; then
    NVCC_PKG_DIR="$("${CONDA_PYTHON}" -c "
import importlib.util
spec = importlib.util.find_spec('nvidia.cu13')
if spec and spec.submodule_search_locations:
    print(spec.submodule_search_locations[0])
" 2>/dev/null || true)"
    if [[ -n "${NVCC_PKG_DIR}" && -x "${NVCC_PKG_DIR}/bin/nvcc" ]]; then
        export CUDA_HOME="${NVCC_PKG_DIR}"
        export PATH="${CUDA_HOME}/bin:${PATH}"
    fi
fi
"${CONDA_PYTHON}" -m experiments.inference.patch_flashinfer || true

# ── 5. GPU check ──────────────────────────────────────────────
"${CONDA_PYTHON}" -c "
import torch
print(f'torch: {torch.__version__}')
if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU available')
print(f'GPU:   {torch.cuda.get_device_name(0)}')
"

# ── 6. Run the no-GT pipeline (starts/stops its own vLLM server) ──
ARGS=(--model "${MODEL_ID}" --start-server --clean-output)
[[ -n "${LIMIT_IMAGES}" ]] && ARGS+=(--limit-images "${LIMIT_IMAGES}")

# run_pipeline exits 1 if any image failed validation; keep going to scoring.
set +e
"${CONDA_PYTHON}" -m experiments.inference.run_pipeline --no-gt "${ARGS[@]}"
PIPELINE_EXIT=$?
set -e
echo "Pipeline exit code: ${PIPELINE_EXIT}"

# ── 7. Score against GT ──────────────────────────────────────
"${CONDA_PYTHON}" -m experiments.evaluation.format_predictions
"${CONDA_PYTHON}" -m experiments.evaluation.evaluate --model "${MODEL_SHORT_NAME}-no-gt"

echo ""
echo "Done. Results in ${OUTPUT_DIR}/${MODEL_SHORT_NAME}/"
echo "  formatted_predictions.csv, metrics.json"
echo "Compare against the GT-conditioned run:"
echo "  benchmark/results/metrics/gt_conditioned/${MODEL_SHORT_NAME}/metrics.json"
