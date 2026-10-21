#!/usr/bin/env python3
"""Template-description ablation for narrative-role classification.

Pipeline step 3 (narrative roles) given the ground-truth social identities and
viewpoints, run once WITH the meme's template description in the prompt
(prompts/04_template_roles.txt) and once WITHOUT it
(prompts/04b_narrative_roles_notemplate.txt). The reported run uses Gemini 3.6
Flash through Google's OpenAI-compatible API (src/slurm/run_gemini_template_ablation.sbatch);
any OpenAI-compatible endpoint works (--base-url, --model, VLLM_API_KEY).

    python -m experiments.template_ablation.ablation --variant with_template
    python -m experiments.template_ablation.ablation --variant no_template --resume
    python -m experiments.template_ablation.ablation --variant with_template --limit 10

Outputs to benchmark/results/template_ablation/<run-name>/<variant>/:
  - results.jsonl    (per-image detail, append-friendly)
  - predictions.csv  (one row per meme)
  - stats.json       (aggregate Jaccard / exact match)
Full metrics: compute_metrics.py.
"""
import argparse
import ast
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dataset_construction.labels import canon_role, gt_viewpoint_at
from experiments.parsing import strip_thinking, try_parse_json  # noqa: F401  (same parser as the pipeline)
from paths import (
    BENCHMARK_JSON,
    IMAGES_DIR,
    TEMPLATE_ABLATION_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
)
logger = logging.getLogger("template_ablation")

# ── Paths ────────────────────────────────────────────────────────────────────
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "04_template_roles.txt"
PROMPT_FILE_NOTEMPLATE = Path(__file__).resolve().parent / "prompts" / "04b_narrative_roles_notemplate.txt"

# Defaults of the reported run: Gemini 3.6 Flash via Google's OpenAI-compatible API.
VLLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
MODEL_NAME = "gemini-3.6-flash"
DEFAULT_RUN_NAME = "gemini-3.6-flash"
MAX_TOKENS = 16384  # Gemini's hidden thinking tokens count against max_tokens
DEFAULT_JSON_MODE = "freeform"  # guided_json / json_object need a vLLM server
# Bearer token for the OpenAI-compatible endpoint (Gemini: GEMINI_API_KEY;
# a local vLLM server ignores it). Same env var as
# config.py, so the pipeline and this script are configured alike.
API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

# Schema for the narrative-roles output: an array of {identity, roles} objects.
# Used with vLLM's guided_json to force correct multi-identity JSON. The model
# rambles in its thinking, but guided decoding constrains the final answer to
# this shape, so the answer survives the long reasoning.
ROLES_ARRAY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "identity": {"type": "string"},
            "roles": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["identity", "roles"],
    },
}


# ── Prompt loading ───────────────────────────────────────────────────────────
def load_prompt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    system, user = "", ""
    section = None
    buf: list[str] = []
    for line in text.split("\n"):
        if line.upper().startswith("SYSTEM:"):
            section = "system"
            buf = [line[len("SYSTEM:"):].strip()]
        elif line.upper().startswith("USER:"):
            section = "user"
            if buf:
                system = " ".join(buf).strip()
            buf = [line[len("USER:"):].strip()]
        elif section is not None:
            buf.append(line)
    if section == "user":
        user = " ".join(buf).strip()
    elif section == "system":
        system = " ".join(buf).strip()
    return system, user


def format_user_prompt(
    template: str,
    social_identities: list[str],
    viewpoints_context: list[dict[str, str]],
    template_description: str,
) -> str:
    ids_str = json.dumps(social_identities, ensure_ascii=False)
    vp_str = json.dumps(viewpoints_context, ensure_ascii=False)
    result = template.replace("{social_identities}", ids_str)
    result = result.replace("{viewpoints_context}", vp_str)
    result = result.replace("{template_description}", template_description)
    return result


# ── Benchmark loading ────────────────────────────────────────────────────────
def load_benchmark() -> dict[str, dict]:
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {e["image_filename"]: e for e in data if "image_filename" in e}


def parse_gt_string(val: Any) -> Optional[list]:
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if not isinstance(val, str):
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None


def get_gt_context(entry: dict) -> tuple[list[str], list[dict[str, str]]]:
    raw_ids = parse_gt_string(entry.get("social_identity_gt"))
    raw_vps = parse_gt_string(entry.get("viewpoint_gt"))
    if not raw_ids:
        return [], []
    gt_ids = [str(x) for x in raw_ids]
    gt_vps = []
    if raw_vps and len(raw_vps) == len(raw_ids):
        for pos, (identity, vp_raw) in enumerate(zip(raw_ids, raw_vps)):
            item = {"identity": str(identity)}
            vp_label = gt_viewpoint_at(str(entry.get("image_filename", "")), pos, vp_raw)
            if vp_label is not None:
                item["viewpoint"] = vp_label
            gt_vps.append(item)
    return gt_ids, gt_vps


# ── VLM inference via vLLM ──────────────────────────────────────────────────
def call_vllm(
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str = VLLM_BASE_URL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
    json_mode: str = DEFAULT_JSON_MODE,
    model: str = MODEL_NAME,
) -> dict:
    """Call vLLM-served model via OpenAI-compatible API.

    json_mode controls structured-output decoding:
      - "guided_json": force the array-of-{identity,roles} schema via vLLM's
        guided_json. Slow (the model still reasons) but returns complete
        multi-identity role sets that survive the rambling.
      - "json_object": OpenAI json_object mode. Fast (0.2s, skips reasoning)
        but returns a single {identity, roles} object (one identity).
      - "freeform": no constraint; rely on the prompt + post-hoc parsing.
    """
    import base64
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url, api_key=API_KEY,
        max_retries=int(os.getenv("VLM_MAX_RETRIES", "2")),
    )

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    media_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}
    media_type = media_map.get(ext, "image/png")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            ],
        },
    ]

    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode == "guided_json":
        kwargs["extra_body"] = {"guided_json": ROLES_ARRAY_SCHEMA}
    elif json_mode == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)

    raw_text = response.choices[0].message.content.strip()
    usage = response.usage.model_dump() if response.usage else {}

    # Strip <think>...</think> tags
    think_close = list(re.finditer(r"</think\s*>", raw_text, re.IGNORECASE))
    if think_close:
        raw_text = raw_text[think_close[-1].end():].strip()

    parsed = try_parse_json(raw_text)

    return {
        "raw_text": raw_text,
        "parsed_json": parsed,
        "usage": usage,
    }


def extract_roles(parsed_json: Any) -> list[str]:
    roles = []
    items = parsed_json
    if isinstance(parsed_json, dict):
        if "identity" in parsed_json and "roles" in parsed_json:
            # single-identity object (e.g. glm)
            items = [parsed_json]
        else:
            for key in ["narrative_roles", "roles", "identities"]:
                if key in parsed_json:
                    items = parsed_json[key]
                    break
            else:
                items = []

    if not isinstance(items, list):
        return []

    for entry in items:
        if isinstance(entry, dict) and "roles" in entry:
            r_list = entry["roles"]
            if isinstance(r_list, str):
                r_list = [r_list]
            if isinstance(r_list, list):
                for r in r_list:
                    r_clean = canon_role(r)
                    if r_clean and r_clean not in roles:
                        roles.append(r_clean)
    return roles


# ── Metrics ──────────────────────────────────────────────────────────────────
def _valid_roles(roles: Any) -> set:
    """Inventory roles only (same invalid-label rule as the main results)."""
    return {r for r in (canon_role(x) for x in (roles or [])) if r}


def jaccard(a: list, b: list) -> float:
    sa, sb = _valid_roles(a), _valid_roles(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def exact_match(a: list, b: list) -> bool:
    return _valid_roles(a) == _valid_roles(b)


# ── Summary generation ──────────────────────────────────────────────────────
def generate_summary_csv(out_dir: Path, variant: str = "with_template"):
    import pandas as pd

    results_file = out_dir / "results.jsonl"
    if not results_file.is_file() or results_file.stat().st_size == 0:
        logger.error("No results file found at %s", out_dir)
        return

    records = []
    with open(results_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Deduplicate by image_filename keeping the last entry (retries overwrite)
    seen = {}
    for r in records:
        seen[r["image_filename"]] = r
    records = list(seen.values())

    # Older records also carry a Qwen3-VL baseline prediction; not part of the
    # reported (Gemini-only) ablation.
    df = pd.DataFrame(records).drop(columns=["predicted_narrative_roles_original"], errors="ignore")
    csv_path = out_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved predictions CSV to %s (%d rows)", csv_path, len(df))

    # Compute stats vs GT
    df["jaccard"] = df.apply(
        lambda r: jaccard(r["narrative_roles_gt"], r["predicted_narrative_roles"]), axis=1
    )
    df["exact"] = df.apply(
        lambda r: exact_match(r["narrative_roles_gt"], r["predicted_narrative_roles"]), axis=1
    )
    df["success"] = df["success"].astype(bool)

    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY — variant: {variant}")
    print("=" * 70)
    print(f"Total images processed: {len(df)}")
    print(f"Successful predictions: {int(df['success'].sum())}")
    print(f"  Mean Jaccard:    {df['jaccard'].mean():.4f}")
    print(f"  Exact Match:     {df['exact'].mean():.2%}")

    print("=" * 70)

    stats = {
        "variant": variant,
        "total_images": len(df),
        "successful_predictions": int(df["success"].sum()),
        "failed_predictions": int((~df["success"]).sum()),
        "mean_jaccard": float(df["jaccard"].mean()),
        "exact_match": float(df["exact"].mean()),
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved stats to %s", stats_path)
    return stats


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Template description ablation for narrative role classification"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to process")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results")
    parser.add_argument("--base-url", type=str, default=VLLM_BASE_URL,
                        help=f"vLLM server URL (default: {VLLM_BASE_URL})")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only regenerate summary CSV from existing results")
    parser.add_argument("--retries", type=int, default=0,
                        help="Retry failed JSON parses up to N times")
    parser.add_argument("--retry-failures", action="store_true",
                        help="Only reprocess images that previously failed to parse")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens for generation (default: {MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--json-mode", type=str, default=DEFAULT_JSON_MODE,
                        choices=["guided_json", "json_object", "freeform"],
                        help="Structured-output decoding mode (default: guided_json)")
    parser.add_argument("--variant", type=str, choices=["with_template", "no_template"],
                        default="with_template",
                        help="Prompt variant to run (default: with_template)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Override prompt file (default derived from --variant)")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME,
                        help="Output subdir under benchmark/results/template_ablation/ "
                             f"(default: {DEFAULT_RUN_NAME})")
    parser.add_argument("--subset-file", type=str, default=None,
                        help="File listing image filenames (one per line) to restrict "
                             "processing to (e.g. for quick tests)")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"Model name to send to the endpoint (default: {MODEL_NAME})")
    args = parser.parse_args()

    out_dir = TEMPLATE_ABLATION_DIR / args.run_name / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine prompt file from variant (or explicit override)
    prompt_file = Path(args.prompt_file) if args.prompt_file else (
        PROMPT_FILE if args.variant == "with_template" else PROMPT_FILE_NOTEMPLATE
    )

    if args.summary_only:
        generate_summary_csv(out_dir=out_dir, variant=args.variant)
        return

    # Check vLLM server
    import requests
    try:
        r = requests.get(args.base_url + "/models", timeout=5,
                         headers={"Authorization": f"Bearer {API_KEY}"})
        if r.status_code != 200:
            logger.error("vLLM server at %s returned status %d", args.base_url, r.status_code)
            sys.exit(1)
    except requests.RequestException as e:
        logger.error("Cannot reach vLLM server at %s: %s", args.base_url, e)
        sys.exit(1)

    logger.info("vLLM server reachable at %s", args.base_url)

    # Load data
    benchmark = load_benchmark()
    logger.info("Loaded %d benchmark entries", len(benchmark))


    system_prompt, user_template = load_prompt(prompt_file)
    logger.info("Loaded prompt from %s (variant=%s)", prompt_file, args.variant)

    # Build image list from benchmark entries
    valid_stems = []
    for fname in sorted(benchmark.keys()):
        stem = Path(fname).stem
        valid_stems.append((stem, fname))
    logger.info("Found %d images with benchmark entries", len(valid_stems))

    if args.subset_file:
        subset = set()
        with open(args.subset_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    subset.add(line)
        valid_stems = [(s, fn) for s, fn in valid_stems if fn in subset]
        logger.info("Restricted to %d images from subset file", len(valid_stems))

    if args.limit:
        valid_stems = valid_stems[:args.limit]
        logger.info("Limited to %d images", len(valid_stems))

    # Resume support
    partial_results = {}
    results_file = out_dir / "results.jsonl"
    if args.resume and results_file.is_file():
        with open(results_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    partial_results[entry["image_filename"]] = entry
        logger.info("Resuming: found %d existing results", len(partial_results))

    # Retry-failures mode: filter to only failed images
    if args.retry_failures:
        failed_images = [k for k, v in partial_results.items() if not v.get("success", False)]
        logger.info("Retry mode: found %d previously failed images to retry", len(failed_images))
        # We'll still iterate all valid_stems but skip successes
        retry_set = set(failed_images)
    else:
        retry_set = None

    # Process
    results_file_mode = "a" if args.resume and partial_results else "w"
    total = len(valid_stems)
    succeeded = 0
    failed = 0
    retried = 0

    with open(results_file, results_file_mode) as f:
        for i, (stem, bench_filename) in enumerate(valid_stems, 1):
            # In retry-failures mode, skip non-failed images
            if retry_set is not None and bench_filename not in retry_set:
                continue
            # In normal resume mode, skip already-succeeded images
            if bench_filename in partial_results and partial_results[bench_filename].get("success", False):
                succeeded += 1
                continue

            entry = benchmark[bench_filename]
            image_path = str(IMAGES_DIR / bench_filename)

            if not Path(image_path).is_file():
                logger.warning("Image not found: %s, skipping", image_path)
                failed += 1
                continue

            gt_ids, gt_vps = get_gt_context(entry)
            if not gt_ids:
                logger.warning("No GT social IDs for %s, skipping", stem)
                failed += 1
                continue

            gt_roles_raw = parse_gt_string(entry.get("narrative_roles_gt", "[]"))
            gt_roles = [str(r).strip().lower() for r in (gt_roles_raw or [])]
            template_desc = entry.get("template_description", "")

            user_prompt = format_user_prompt(user_template, gt_ids, gt_vps, template_desc)

            # Retry loop
            max_attempts = 1 + args.retries
            for attempt in range(max_attempts):
                logger.info("[%d/%d] %s (template: %s) attempt %d/%d",
                            i, total, stem, entry.get("template_name", "?"),
                            attempt + 1, max_attempts)

                try:
                    t0 = time.time()
                    result = call_vllm(image_path, system_prompt, user_prompt,
                                       base_url=args.base_url,
                                       max_tokens=args.max_tokens,
                                       temperature=args.temperature,
                                       json_mode=args.json_mode,
                                       model=args.model)
                    elapsed = time.time() - t0
                    new_roles = extract_roles(result["parsed_json"])

                    if result["parsed_json"] is not None or attempt == max_attempts - 1:
                        record = {
                            "image_filename": bench_filename,
                            "template_name": entry.get("template_name", ""),
                            "template_description": template_desc,
                            "narrative_roles_gt": gt_roles,
                            "predicted_narrative_roles": new_roles,
                            "variant": args.variant,
                            "max_tokens": args.max_tokens,
                            "temperature": args.temperature,
                            "success": result["parsed_json"] is not None,
                            "attempts": attempt + 1,
                            "latency_s": round(elapsed, 2),
                            "usage": result.get("usage", {}),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        succeeded += 1
                        if attempt > 0:
                            retried += 1

                        if result["parsed_json"] is None:
                            logger.warning("  Failed to parse JSON after %d attempts", max_attempts)
                        else:
                            logger.info("  GT: %s | Predicted: %s (%.1fs)", gt_roles, new_roles, elapsed)
                        break
                    else:
                        logger.warning("  Parse failed, retrying... (%d/%d)", attempt + 1, max_attempts)
                        retried += 1

                except Exception as e:
                    logger.exception("Error processing %s: %s", stem, e)
                    if attempt == max_attempts - 1:
                        failed += 1

    logger.info("Done: %d succeeded (%d retried), %d failed out of %d total",
                succeeded, retried, failed, total)
    generate_summary_csv(out_dir=out_dir, variant=args.variant)


if __name__ == "__main__":
    main()
