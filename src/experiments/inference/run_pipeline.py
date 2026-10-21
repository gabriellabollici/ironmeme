#!/usr/bin/env python3
"""VLM Knowledge Extraction Pipeline.

Supports:
  - Single model: ``--model MODEL_ID`` (default: first model in config)
  - All models sequentially: ``--all-models``
  - All models in parallel: ``--all-models --parallel`` (launches one vLLM
    server per model on separate ports, processes images concurrently)
  - No-ground-truth variant: ``--no-gt`` (steps 2-3 see the model's own
    predictions instead of the GT; see pipeline.process_image_no_gt).
    Defaults to GLM-4.1V-9B-Thinking and writes to benchmark/out_no_gt/ so it
    never overwrites the GT-conditioned outputs in benchmark/out/.

Usage (from the repo root):
    python -m experiments.inference.run_pipeline --start-server --model Qwen/Qwen3-VL-8B-Instruct
    python -m experiments.inference.run_pipeline --start-server --no-gt --limit-images 5
"""
import argparse
import concurrent.futures
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml


import paths
from experiments.inference import config
from experiments.inference.config import MODEL_ID, IMAGES_DIR, MODELS_CONFIG, VLLM_BASE_PORT, VLLM_HOST, VLLM_BASE_URL
from experiments.inference.image_utils import discover_images
from experiments.inference.vllm_manager import start_servers_for_models, stop_all_servers
from experiments.inference.vlm_client import check_server
from experiments.inference.pipeline import process_image, process_image_no_gt
from experiments.inference.output_utils import setup_logging

logger = logging.getLogger("pipeline_main")

# --no-gt default: run via --model (not MODEL_ID) so models.yaml overrides
# apply, e.g. max_tokens: 8192 for GLM's thinking budget.
DEFAULT_NO_GT_MODEL_ID = "zai-org/GLM-4.1V-9B-Thinking"


# ── YAML helpers ────────────────────────────────────────────────────────────
def load_models_from_yaml(yaml_path: str) -> List[Dict[str, str]]:
    """Return list of {"id": ..., "short_name": ...} dicts from models.yaml."""
    path = Path(yaml_path)
    if not path.is_file():
        logger.error("Models config not found: %s", yaml_path)
        sys.exit(1)

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not data or "models" not in data:
        logger.error("Invalid models config: missing 'models' key")
        sys.exit(1)

    models_raw = data["models"]
    if not isinstance(models_raw, list) or len(models_raw) == 0:
        logger.error("Models config must contain a non-empty list")
        sys.exit(1)

    models: List[Dict[str, str]] = []
    for entry in models_raw:
        if isinstance(entry, str):
            mid = entry
            short = entry.split("/")[-1].lower().replace("_", "-")
            max_model_len = None
            max_tokens = None
            no_system_role = False
        elif isinstance(entry, dict):
            mid = entry["id"]
            short = entry.get("short_name", mid.split("/")[-1].lower().replace("_", "-"))
            max_model_len = entry.get("max_model_len")
            max_tokens = entry.get("max_tokens")
            no_system_role = bool(entry.get("no_system_role", False))
        else:
            logger.warning("Skipping invalid models.yaml entry: %s", entry)
            continue
        models.append({
            "id": mid, "short_name": short,
            "max_model_len": max_model_len, "max_tokens": max_tokens,
            "no_system_role": no_system_role,
        })
    return models


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLM Knowledge Extraction Pipeline",
    )
    p.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Specific image filenames (in images/) to process. "
             "If omitted, all images are processed.",
    )
    p.add_argument(
        "--start-server",
        action="store_true",
        default=False,
        help="Launch vLLM server(s) before processing.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override MODEL_ID for a single model run.",
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        default=False,
        help="Process all models from inference/models.yaml.",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run all models in parallel (each gets its own vLLM server). "
             "Only valid together with --all-models.",
    )
    p.add_argument(
        "--models-file",
        type=str,
        default=MODELS_CONFIG,
        help="Path to models YAML config (default: inference/models.yaml).",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path for an additional log file.",
    )
    p.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Limit the number of images to process.",
    )
    p.add_argument(
        "--clean-output",
        action="store_true",
        default=False,
        help="Delete OUTPUT_DIR/<model_short_name>/ before processing each "
             "model. Off by default -- without this, per-image output dirs "
             "from a previous run against a *different* image set (e.g. "
             "after switching IMAGES_DIR/BENCHMARK_JSON) stay on disk "
             "alongside the new ones. format_predictions.py filters stale "
             "dirs out of scoring regardless, but they still clutter the "
             "output dir unless removed here.",
    )
    p.add_argument(
        "--no-gt",
        action="store_true",
        default=False,
        help="Never show ground truth to the model: steps 2-3 use the "
             "model's own step 1-2 predictions. Writes to "
             "benchmark/out_no_gt/ (unless OUTPUT_DIR is set) and defaults "
             f"to {DEFAULT_NO_GT_MODEL_ID} when no --model/--all-models is given.",
    )
    return p.parse_args()


# ── Per-model worker (runs in a separate process) ──────────────────────────
def _process_model_worker(
    model_id: str,
    model_short_name: str,
    base_url: str,
    image_paths: List[str],
    no_gt: bool = False,
    max_tokens: int = None,
    no_system_role: bool = False,
) -> Dict[str, object]:
    """Process all images for one model. Designed to be called by ProcessPoolExecutor."""
    process = process_image_no_gt if no_gt else process_image
    logger.info("Worker starting  model=%s  base_url=%s  images=%d",
                model_id, base_url, len(image_paths))

    t0 = time.time()
    results: List[Dict] = []

    for i, img_path in enumerate(image_paths, 1):
        stem = Path(img_path).stem
        logger.info("  [%s] Image %d/%d: %s", model_short_name, i, len(image_paths), stem)
        try:
            result = process(
                img_path,
                base_url=base_url,
                model_name=model_short_name,
                max_tokens=max_tokens,
                no_system_role=no_system_role,
            )
            results.append(result)
        except Exception:
            logger.exception("  [%s] Error processing %s", model_short_name, stem)
            results.append({"image_path": img_path, "success": False})

    elapsed = time.time() - t0
    succeeded = sum(1 for r in results if r.get("success"))
    failed = len(results) - succeeded

    logger.info("  [%s] Done — %d/%d succeeded in %.1fs",
                model_short_name, succeeded, len(results), elapsed)

    return {
        "model": model_id,
        "short_name": model_short_name,
        "success": failed == 0,
        "succeeded": succeeded,
        "failed": failed,
        "elapsed": elapsed,
    }


# ── Sequential single-model run ─────────────────────────────────────────────
def process_single_model(
    model_id: str,
    model_short_name: str,
    image_paths: List[str],
    start_server_flag: bool,
    max_model_len: int = None,
    clean_output: bool = False,
    max_tokens: int = None,
    no_system_role: bool = False,
    no_gt: bool = False,
) -> Dict[str, object]:
    """Run one model sequentially (no multi-server)."""
    process = process_image_no_gt if no_gt else process_image
    config.MODEL_ID = model_id
    config.MODEL_SHORT_NAME = model_short_name

    logger.info("=" * 60)
    logger.info("Processing model: %s  (short: %s)", model_id, model_short_name)
    if max_model_len:
        logger.info("  max_model_len override: %d", max_model_len)
    if max_tokens:
        logger.info("  max_tokens override: %d", max_tokens)
    if no_system_role:
        logger.info("  no_system_role: system prompt merged into user message")
    logger.info("=" * 60)

    if clean_output:
        model_out_dir = Path(config.OUTPUT_DIR) / model_short_name
        if model_out_dir.is_dir():
            logger.info("--clean-output: removing %s", model_out_dir)
            shutil.rmtree(model_out_dir)

    from experiments.inference.vllm_manager import start_server, stop_server
    server_started_by_us = False
    if start_server_flag:
        try:
            start_server(max_model_len=max_model_len)
            server_started_by_us = True
        except Exception as e:
            logger.error("Failed to start server for %s: %s", model_id, e)
            return {"model": model_id, "success": False, "error": str(e)}

    try:
        t0 = time.time()
        results: List[Dict] = []
        for i, img_path in enumerate(image_paths, 1):
            logger.info("  >>> Image %d/%d: %s", i, len(image_paths), Path(img_path).stem)
            try:
                result = process(
                    img_path,
                    model_name=model_short_name,
                    max_tokens=max_tokens,
                    no_system_role=no_system_role,
                )
                results.append(result)
            except Exception:
                logger.exception("  Error processing %s", img_path)
                results.append({"image_path": img_path, "success": False})

        elapsed = time.time() - t0
        succeeded = sum(1 for r in results if r.get("success"))
        failed = len(results) - succeeded

        return {
            "model": model_id,
            "short_name": model_short_name,
            "success": failed == 0,
            "succeeded": succeeded,
            "failed": failed,
            "elapsed": elapsed,
        }
    finally:
        if server_started_by_us:
            stop_server()


# ── Parallel run ────────────────────────────────────────────────────────────
def process_models_parallel(
    models: List[Dict[str, str]],
    image_paths: List[str],
    no_gt: bool = False,
) -> List[Dict[str, object]]:
    """Launch one vLLM server per model, then process images in parallel."""
    model_ids = [m["id"] for m in models]
    served_names = {m["id"]: m["short_name"] for m in models}
    n_models = len(model_ids)
    n_images = len(image_paths)
    workers = min(n_models, 8)

    logger.info("Parallel run: %d models × %d images  (workers=%d)", n_models, n_images, workers)

    # Launch all servers
    try:
        base_urls = start_servers_for_models(
            model_ids, served_names=served_names,
            max_model_lens={m["id"]: m.get("max_model_len") for m in models},
        )
    except Exception as e:
        logger.error("Failed to launch servers: %s", e)
        return [{"model": m, "success": False, "error": str(e)} for m in model_ids]

    try:
        summaries: List[Dict[str, object]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_model: Dict[concurrent.futures.Future, Dict[str, str]] = {}
            for model in models:
                future = executor.submit(
                    _process_model_worker,
                    model_id=model["id"],
                    model_short_name=model["short_name"],
                    base_url=base_urls[model["id"]],
                    image_paths=image_paths,
                    no_gt=no_gt,
                    max_tokens=model.get("max_tokens"),
                    no_system_role=bool(model.get("no_system_role", False)),
                )
                future_to_model[future] = model

            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    summary = future.result()
                    summaries.append(summary)
                except Exception as exc:
                    logger.exception("Model %s raised: %s", model["id"], exc)
                    summaries.append({"model": model["id"], "success": False, "error": str(exc)})

        return summaries
    finally:
        stop_all_servers()


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.no_gt:
        if "OUTPUT_DIR" not in os.environ:
            config.OUTPUT_DIR = str(paths.PIPELINE_OUT_NO_GT_DIR)
            # --parallel workers re-import config in a fresh process.
            os.environ["OUTPUT_DIR"] = config.OUTPUT_DIR
        if not args.model and not args.all_models:
            args.model = DEFAULT_NO_GT_MODEL_ID

    log_file = args.log_file or str(Path(config.OUTPUT_DIR) / "pipeline.log")
    setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("VLM Knowledge Extraction Pipeline")
    logger.info("Images dir: %s", IMAGES_DIR)
    logger.info("Output dir: %s", config.OUTPUT_DIR)
    if args.no_gt:
        logger.info("Mode: no ground truth (steps 2-3 use predicted context)")
    logger.info("=" * 60)

    # Discover images
    if args.images:
        image_paths = [str(Path(IMAGES_DIR) / name) for name in args.images]
        for p in image_paths:
            if not Path(p).is_file():
                logger.error("Image not found: %s", p)
                sys.exit(1)
    else:
        image_paths = discover_images()

    if not image_paths:
        logger.warning("No images found. Exiting.")
        sys.exit(0)

    if args.limit_images is not None:
        image_paths = image_paths[: args.limit_images]

    logger.info("Will process %d image(s).", len(image_paths))

    # Determine which models to process
    if args.all_models:
        models = load_models_from_yaml(args.models_file)
        logger.info("Loaded %d models from %s", len(models), args.models_file)
    elif args.model:
        short = args.model.split("/")[-1].lower().replace("_", "-")
        # Look up per-model overrides (max_model_len, max_tokens) from
        # models.yaml so that --model respects model-specific config
        # instead of always falling back to global defaults.
        all_models = load_models_from_yaml(args.models_file)
        override = next((m for m in all_models if m["id"] == args.model), {})
        models = [{"id": args.model, "short_name": short,
                    "max_model_len": override.get("max_model_len"),
                    "max_tokens": override.get("max_tokens"),
                    "no_system_role": override.get("no_system_role", False)}]
    else:
        short = MODEL_ID.split("/")[-1].lower().replace("_", "-")
        models = [{"id": MODEL_ID, "short_name": short}]

    # ── Pre-flight server check ──────────────────────────────────────────
    if not args.start_server:
        if args.parallel and len(models) > 1:
            for idx, model in enumerate(models):
                url = f"http://{VLLM_HOST}:{VLLM_BASE_PORT + idx}/v1"
                if not check_server(base_url=url):
                    logger.error(
                        "No vLLM server running for %s at %s. "
                        "Start it first or pass --start-server.",
                        model["id"], url,
                    )
                    sys.exit(1)
        else:
            if not check_server():
                logger.error(
                    "No vLLM server running at %s. "
                    "Start it first or pass --start-server.",
                    VLLM_BASE_URL,
                )
                sys.exit(1)

    # Run
    if args.parallel and len(models) > 1:
        summaries = process_models_parallel(models, image_paths, no_gt=args.no_gt)
    else:
        summaries = []
        for model in models:
            summary = process_single_model(
                model_id=model["id"],
                model_short_name=model["short_name"],
                image_paths=image_paths,
                start_server_flag=args.start_server,
                max_model_len=model.get("max_model_len"),
                clean_output=args.clean_output,
                max_tokens=model.get("max_tokens"),
                no_system_role=bool(model.get("no_system_role", False)),
                no_gt=args.no_gt,
            )
            summaries.append(summary)

    # Final summary
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for s in summaries:
        status = "OK" if s.get("success") else "FAILED"
        logger.info(
            "  %s: %s  succeeded=%s  failed=%s  elapsed=%.1fs",
            s.get("model", "?"), status,
            s.get("succeeded", "?"), s.get("failed", "?"),
            s.get("elapsed", 0),
        )

    all_ok = all(s.get("success") for s in summaries)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
