#!/usr/bin/env bash
# Run the ironmeme pipeline with Kimi K3 (moonshotai/Kimi-K3) via Moonshot's
# hosted OpenAI-compatible API.
#
# Why the API and not a local vLLM server: K3 is a 2.8T-param MoE (~104B
# active) whose MXFP4 weights alone are ~1.4TB -- it cannot fit on a single
# H100/A100, and vllm_manager.py only launches single-GPU servers (see the
# excluded-models notes in resources/models.yaml). The hosted endpoint is
# natively multimodal, so the pipeline's image+text chat calls work as-is.
#
# Required:
#   export MOONSHOT_API_KEY=sk-...   # from https://platform.kimi.ai
#
# Usage (same extra args as run_pipeline.py):
#   ./src/slurm/run_kimi_k3.sh                        # full benchmark
#   ./src/slurm/run_kimi_k3.sh --limit-images 5       # smoke test
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

: "${MOONSHOT_API_KEY:?MOONSHOT_API_KEY is not set -- get a key at https://platform.kimi.ai}"

export MODEL_ID="${MODEL_ID:-kimi-k3}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-https://api.moonshot.ai/v1}"
# K3 always reasons (cannot be disabled) and its default effort is "max",
# with reasoning tokens billed as output ($15/M). "low" keeps the
# chain-of-thought short for this simple classification task -- the same
# role the per-model max_tokens override plays for local Qwen3-VL-Thinking.
# Override with REASONING_EFFORT=high|max if outputs look degraded.
export REASONING_EFFORT="${REASONING_EFFORT:-low}"
# K3's API locks sampling -- it hard-rejects any other value
# ("invalid temperature: only 1 is allowed for this model"), so the
# pipeline's deterministic TEMPERATURE=0.0 setting isn't available here.
# Note for cross-model comparisons: K3 runs non-greedy by necessity.
export TEMPERATURE="${TEMPERATURE:-1}"
# Hard per-call ceiling (reasoning + final JSON combined). With low effort
# actual usage lands far below this; it exists purely as a worst-case cost
# guard so a single hung-up call can't burn unbounded output tokens.
export MAX_TOKENS="${MAX_TOKENS:-4096}"

python -m experiments.inference.run_pipeline \
    --model "$MODEL_ID" \
    "$@"
