#!/usr/bin/env bash
# Install the `inference` dependency group from pyproject.toml into the given
# Python (default: python on PATH). Run from the repo root.
#
# Only the group is installed, not the project: the slurm jobs put src/ on
# PYTHONPATH instead, so the project's requires-python and its `analysis`
# group (torch, numpy>=2.5, ...) never touch the cluster's conda env.
# `pip install --group` needs pip >= 25.1.
set -euo pipefail

PYTHON="${1:-python}"

"${PYTHON}" -m pip install -q --upgrade "pip>=25.1"
echo "[INFO] Installing dependency group 'inference' from pyproject.toml"
"${PYTHON}" -m pip install -q --group inference
