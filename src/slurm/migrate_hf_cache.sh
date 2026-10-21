#!/usr/bin/env bash
# One-time move of the Hugging Face cache from its pre-restructure location
# (<repo>/narrative-memes/.hf_cache) to <repo>/.hf_cache, the new default in
# experiments/inference/config.py, so the model weights aren't downloaded again.
# A no-op once moved, or when the new cache already exists.
#
# Usage: bash src/slurm/migrate_hf_cache.sh <repo root>
set -euo pipefail

REPO_ROOT="${1:?usage: migrate_hf_cache.sh <repo root>}"
OLD="${REPO_ROOT}/narrative-memes/.hf_cache"
NEW="${REPO_ROOT}/.hf_cache"

if [[ -d "${OLD}" && ! -e "${NEW}" ]]; then
    echo "[INFO] Moving HF cache ${OLD} -> ${NEW}"
    # Same filesystem, so this is a rename. `|| true`: two jobs starting at
    # once may race; whichever loses finds the cache already moved.
    mv "${OLD}" "${NEW}" || true
    rmdir "${REPO_ROOT}/narrative-memes" 2>/dev/null || true
fi
