#!/bin/bash
#SBATCH --job-name=vlm-pipeline
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# NOTE: #SBATCH directives cannot expand shell variables ($HOME, etc.),
# so this relative "logs/" path resolves against whatever directory
# `sbatch` is invoked from (SLURM_SUBMIT_DIR), NOT this script's own
# location -- and SLURM opens this file *before* any script commands
# run, so a `mkdir -p` inside the script body can't help. The
# directory must already exist, in the directory you run `sbatch`
# from, or the job runs with NO .out/.err file at all and fails
# silently (no error shown to you at submission time). Always:
#   1. cd to the repo root (the IRONMeme folder) first
#   2. mkdir -p logs   (one-time, or after deleting/moving it)
#   3. sbatch src/slurm/run_pipeline.sbatch
# Submitting from another directory puts the logs in <that dir>/logs/
# instead -- also fine, as long as that dir exists first and you look for
# the log there (the repo root is still found from SLURM_SUBMIT_DIR, so
# submit from the repo root unless you know what you're doing).
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Requesting 1 GPU (not a full 4-GPU node): every model currently in
# resources/models.yaml (Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Thinking,
# deepseek-ai/deepseek-vl2, GLM-4.1V-9B-Thinking, Molmo-7B-D) fits on a
# single
# H100 94GB, and vllm_manager.py doesn't support tensor-parallel/multi-GPU
# serving. Larger models (Qwen3-VL-235B, InternVL3.5-241B, Llama-4-Maverick) are
# excluded from models.yaml for this reason -- see comments there.

set -euo pipefail
set -x

echo ""
echo " VLM Knowledge Extraction Pipeline"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo ""

# ── 1. Modules ────────────────────────────────────────────────
# This cluster has no `cuda` environment module at all (confirmed via
# `module avail cuda` -> "No module(s) or extension(s) found!"), so this
# has always been a silent no-op. CUDA_HOME is set below (step 5) from
# the pip-installed nvidia-cuda-nvcc package instead.
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
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark/out}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_BASE_PORT="${VLLM_BASE_PORT:-30000}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
export VLLM_SERVER_LOG_DIR="${VLLM_SERVER_LOG_DIR:-${OUTPUT_DIR}/vllm_logs}"

# ── 4. Config ─────────────────────────────────────────────────
LIMIT_IMAGES="${LIMIT_IMAGES:-}"
ALL_MODELS="${ALL_MODELS:-false}"
MODEL_ID="${MODEL_ID:-}"
# Opt-in: deletes OUTPUT_DIR/<model_short_name>/ right before each model
# starts. Off by default. Useful after switching IMAGES_DIR/BENCHMARK_JSON
# to a different image set -- without this, stale per-image output dirs
# from the old image set stick around on disk (format_predictions.py
# already filters them out of scoring regardless,
# but they still clutter OUTPUT_DIR unless removed).
CLEAN_OUTPUT="${CLEAN_OUTPUT:-false}"

echo "Config:"
echo "  IMAGES_DIR       = ${IMAGES_DIR}"
echo "  OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "  HF_HOME          = ${HF_HOME}"
echo "  VLLM_HOST        = ${VLLM_HOST}"
echo "  VLLM_BASE_PORT   = ${VLLM_BASE_PORT}"
echo "  VLLM_MAX_MODEL_LEN         = ${VLLM_MAX_MODEL_LEN}"
echo "  VLLM_GPU_MEMORY_UTILIZATION = ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "  VLLM_SERVER_LOG_DIR = ${VLLM_SERVER_LOG_DIR}"
echo "  MODEL_ID         = ${MODEL_ID:-<from models.yaml>}"
echo "  LIMIT_IMAGES     = ${LIMIT_IMAGES:-<all>}"
echo "  ALL_MODELS       = ${ALL_MODELS}"
echo "  CLEAN_OUTPUT     = ${CLEAN_OUTPUT}"
echo ""

# ── 5. Install dependencies ─────────────────────────────────
bash src/slurm/install_inference_deps.sh "${CONDA_PYTHON}"

# flashinfer JIT-compiles some CUDA sampling kernels at runtime and needs
# nvcc on PATH / CUDA_HOME set. This cluster has no `cuda` environment
# module at all (confirmed via `module avail cuda` ->
# "No module(s) or extension(s) found!"), so point at the nvcc shipped by
# the pip-installed `nvidia-cuda-nvcc` package instead of relying on a
# system CUDA toolkit. Run after pip install so the package is guaranteed
# to already be present (it's a transitive dependency of vllm/flashinfer).
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
        echo "[INFO] CUDA_HOME set from pip-installed nvidia-cuda-nvcc: ${CUDA_HOME}"
    else
        echo "[WARN] Could not locate pip-installed nvcc; CUDA_HOME left unset."
    fi
fi
echo "  CUDA_HOME        = ${CUDA_HOME:-<unset>}"

# flashinfer-python>=0.6.16 crashes on import on Python <=3.11 due to an
# `array.array[int]` type annotation (array only supports subscripting
# since Python 3.12). This patches the installed file in-place rather
# than pinning an older flashinfer-python, which would otherwise force
# pip to backtrack through incompatible old vllm/torch combinations.
# See src/experiments/inference/patch_flashinfer.py for details.
"${CONDA_PYTHON}" -m experiments.inference.patch_flashinfer || true

# ── 6. GPU check ──────────────────────────────────────────────
"${CONDA_PYTHON}" -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:   {torch.cuda.get_device_name(0)}')
else:
    raise RuntimeError('No CUDA GPU available')
"

# ── 7. Run pipeline ───────────────────────────────────────────

ARGS=()
[[ -n "${LIMIT_IMAGES}" ]] && ARGS+=(--limit-images "${LIMIT_IMAGES}")
ARGS+=(--start-server)
[[ "${CLEAN_OUTPUT}" == "true" ]] && ARGS+=(--clean-output)

if [[ "${ALL_MODELS}" == "true" ]]; then
    # Sequential (not --parallel): a single A100 can't host multiple
    # concurrent vLLM servers, especially for the 235B/241B-param models
    # in resources/models.yaml. process_single_model() starts/stops the
    # server between each model.
    ARGS+=(--all-models)
else
    [[ -n "${MODEL_ID}" ]] && ARGS+=(--model "${MODEL_ID}")
fi

echo "Command: ${CONDA_PYTHON} -m experiments.inference.run_pipeline ${ARGS[*]}"
echo ""

"${CONDA_PYTHON}" -m experiments.inference.run_pipeline "${ARGS[@]}"

PIPELINE_EXIT=$?
echo "Pipeline exit code: ${PIPELINE_EXIT}"

# ── 8. Finished ───────────────────────────────────────────────

echo ""
if [[ "${PIPELINE_EXIT}" -eq 0 ]]; then
    echo " Pipeline completed successfully."
else
    echo " Pipeline finished with errors (exit=${PIPELINE_EXIT})."
fi
echo "Results saved to: ${OUTPUT_DIR}"
