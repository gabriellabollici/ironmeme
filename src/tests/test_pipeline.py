"""Regression tests: the benchmark and the reported results are reproducible.

Run from the repo root:
    uv run pytest                 # everything
    uv run pytest -m "not slow"   # skip the full metrics recompute (~45 s)
"""
import filecmp
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import paths

needs_data = pytest.mark.skipif(
    not paths.RAW_ANNOTATIONS_DIR.is_dir() or not paths.RAW_RESULTS_DIR.is_dir(),
    reason="benchmark data not available",
)


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != ".DS_Store"}


def _assert_same_tree(expected: Path, actual: Path) -> None:
    expected_files, actual_files = _files(expected), _files(actual)
    assert actual_files == expected_files, (
        f"missing: {sorted(expected_files - actual_files)[:5]}, extra: {sorted(actual_files - expected_files)[:5]}")
    different = [f for f in sorted(expected_files) if not filecmp.cmp(expected / f, actual / f, shallow=False)]
    assert not different, f"{len(different)} file(s) differ, e.g. {different[:5]}"


@needs_data
def test_build_reproduces_the_benchmark_dataset(tmp_path):
    from dataset_construction.build_benchmark import build

    out = tmp_path / "benchmark_dataset"
    build(out)
    _assert_same_tree(paths.BENCHMARK_DATASET_DIR, out)


@needs_data
@pytest.mark.slow
def test_recompute_reproduces_the_stored_results(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "experiments.evaluation.recompute_all_metrics",
         "--out-root", str(tmp_path)],
        check=True, capture_output=True,
    )
    for stored in (paths.METRICS_DIR, paths.TEMPLATE_ABLATION_DIR, paths.DIAGNOSTICS_DIR):
        produced = tmp_path / stored.relative_to(paths.BENCHMARK_DIR)
        # recompute writes only the files it derives; compare those
        for rel in sorted(_files(produced)):
            assert filecmp.cmp(stored / rel, produced / rel, shallow=False), f"{stored.name}/{rel} differs"
    assert _files(tmp_path / paths.METRICS_DIR.relative_to(paths.BENCHMARK_DIR)) >= {
        rel for rel in _files(paths.METRICS_DIR)
        if rel.endswith(("metrics.json", "formatted_predictions.csv", "all_models_metrics.csv"))
    }


def test_slurm_jobs_only_call_existing_modules():
    modules = set()
    for job in (paths.REPO_ROOT / "src" / "slurm").iterdir():
        modules |= set(re.findall(r"-m ((?:dataset_construction|experiments|benchmark_analysis)(?:\.\w+)+)", job.read_text()))
    assert modules, "no python -m calls found in src/slurm"
    missing = sorted(m for m in modules if importlib.util.find_spec(m) is None)
    assert not missing, f"slurm jobs call missing modules: {missing}"


def test_importing_the_package_writes_no_files(tmp_path):
    watched = [paths.SRC_DIR, paths.METRICS_DIR, paths.BENCHMARK_DATASET_DIR, paths.SOURCES_DIR]

    def snapshot():
        return {p: p.stat().st_mtime_ns for root in watched if root.is_dir()
                for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}

    before = snapshot()
    code = (
        "import importlib, pkgutil, paths\n"
        "for name in ('dataset_construction', 'experiments', 'benchmark_analysis'):\n"
        "    package = importlib.import_module(name)\n"
        "    for m in pkgutil.walk_packages(package.__path__, name + '.'):\n"
        "        importlib.import_module(m.name)\n"
    )
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path, env=env, capture_output=True)
    after = snapshot()
    changed = sorted(str(p) for p in after if before.get(p) != after[p])
    assert not changed, f"importing modules wrote files: {changed[:5]}"
    assert not any(tmp_path.iterdir()), "importing modules wrote files into the working directory"


def test_paths_point_at_existing_locations():
    for name in ("BENCHMARK_DATASET_DIR", "BENCHMARK_JSON", "BENCHMARK_CSV", "IMAGES_DIR", "SOURCES_DIR",
                 "TEMPLATES_TABLE", "MEME_METADATA_TABLE", "RAW_GT_CONDITIONED_DIR", "RAW_NO_GT_DIR",
                 "GT_CONDITIONED_METRICS_DIR", "NO_GT_METRICS_DIR", "SUBSET_GT_CONDITIONED_METRICS_DIR",
                 "SUBSET_NO_GT_METRICS_DIR", "TEMPLATE_ABLATION_DIR",
                 "DIAGNOSTICS_DIR"):
        assert getattr(paths, name).exists(), f"paths.{name} -> {getattr(paths, name)} does not exist"
