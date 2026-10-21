#!/usr/bin/env bash
# No-ground-truth pipeline run (run_pipeline --no-gt) with Gemini
# via Google's hosted OpenAI-compatible Gemini API. Step 2 gets Step 1's
# predicted identities, Step 3 gets Step 2's predicted viewpoints -- no GT is
# ever put in a prompt. Hosted API, so no GPU/SLURM needed: runs from any
# terminal (laptop or a Snellius login node).
#
# The API key is read from GEMINI_API_KEY, or else from <repo root>/.env
# (gitignored -- never commit it), e.g.:
#   GEMINI_API_KEY=...
#
# Usage (same extra args as run_pipeline.py):
#   ./src/slurm/run_gemini_no_gt.sh --limit-images 5   # smoke test
#   ./src/slurm/run_gemini_no_gt.sh                    # full benchmark
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ -z "${GEMINI_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
    set -a; . "${REPO_ROOT}/.env"; set +a
fi
: "${GEMINI_API_KEY:?GEMINI_API_KEY is not set (export it or put it in ${REPO_ROOT}/.env)}"

export MODEL_ID="${MODEL_ID:-gemini-3.6-flash}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}"
export VLLM_API_KEY="${GEMINI_API_KEY}"
# Greedy decoding, same as the local-vLLM runs.
export TEMPERATURE="${TEMPERATURE:-0.0}"
# Gemini's hidden thinking tokens count against max_tokens. At the global
# 1024 default, answers get truncated before the JSON; a previous
# gemini-3.6-flash run never used more than ~3.6k output+thinking tokens,
# so 8192 is headroom, not a target. Thinking level is left at the API
# default (REASONING_EFFORT unset -> the field is never sent).
export MAX_TOKENS="${MAX_TOKENS:-8192}"
# The pipeline has no resume, so ride out 429/5xx with more SDK retries
# (exponential backoff) rather than losing the image.
export VLM_MAX_RETRIES="${VLM_MAX_RETRIES:-8}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark/out_no_gt}"

python -m experiments.inference.run_pipeline --no-gt \
    --model "$MODEL_ID" \
    "$@"
