import os
from pathlib import Path

import paths

INFERENCE_DIR = Path(__file__).resolve().parent

MODEL_ID: str = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

MODEL_SHORT_NAME: str = os.getenv(
    "MODEL_SHORT_NAME",
    MODEL_ID.split("/")[-1].lower().replace("_", "-"),
)

VLLM_HOST: str = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT: int = int(os.getenv("VLLM_PORT", "30000"))
# Full-URL override takes precedence over the host/port derivation so the
# pipeline can also target *external* OpenAI-compatible APIs (e.g. Moonshot's
# hosted Kimi models at https://api.moonshot.ai/v1) instead of a local vLLM
# server -- those need https and don't fit the host:port scheme.
VLLM_BASE_URL: str = os.getenv(
    "VLLM_BASE_URL", f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
)
# Bearer token sent by vlm_client. Local vLLM ignores it ("EMPTY" default);
# external APIs (Moonshot etc.) require a real key, e.g. MOONSHOT_API_KEY.
VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", os.getenv("MOONSHOT_API_KEY", "EMPTY"))

VLLM_STARTUP_TIMEOUT: int = int(os.getenv("VLLM_STARTUP_TIMEOUT", "600"))
VLLM_POLL_INTERVAL: int = int(os.getenv("VLLM_POLL_INTERVAL", "10"))

# Qwen3-VL's default max_position_embeddings is 262144, which needs ~36 GiB
# of KV cache -- more than a single A100 (80GB) has available after model
# weights + activations. Our task (short prompts + one image + short JSON
# output) needs nowhere near that, so cap it well below the model's max.
# Override via VLLM_MAX_MODEL_LEN if a specific model/task needs more.
VLLM_MAX_MODEL_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "16384"))
VLLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))


PROMPTS_DIR: str = str(INFERENCE_DIR / "prompts")
# benchmark_dataset/ (written by dataset_construction.build_benchmark)
# is self-contained: its images/ filenames ("1.png", "2.jpg", ...) are exactly
# the image_filename values in its ironmeme_benchmark.json.
IMAGES_DIR: str = os.getenv("IMAGES_DIR", str(paths.IMAGES_DIR))
# Read at call time (config.OUTPUT_DIR), not imported by value: run_pipeline
# --no-gt switches it to paths.PIPELINE_OUT_NO_GT_DIR after argument parsing.
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", str(paths.PIPELINE_OUT_DIR))
MODELS_CONFIG: str = str(INFERENCE_DIR / "models.yaml")
BENCHMARK_JSON: str = os.getenv("BENCHMARK_JSON", str(paths.BENCHMARK_JSON))

# Respect $HF_HOME when $HF_CACHE_DIR is not explicitly set, so that
# caches converge on one directory instead of splitting across
# <repo>/.hf_cache (default) and whatever $HF_HOME points to on
# the cluster.  Explicit HF_CACHE_DIR always wins.
HF_CACHE_DIR: str = os.getenv(
    "HF_CACHE_DIR",
    os.getenv("HF_HOME", str(paths.REPO_ROOT / ".hf_cache")),
)

MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
# Reasoning-effort control for hosted reasoning models (e.g. Kimi K3's
# top-level `reasoning_effort`: low|high|max, default max -- and reasoning
# tokens bill as output). Empty string = never send the field, which keeps
# local vLLM models (Qwen etc.) untouched.
REASONING_EFFORT: str = os.getenv("REASONING_EFFORT", "")

IMAGE_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

# ---------------------------------------------------------------------------
# Parallel model execution
# ---------------------------------------------------------------------------
VLLM_BASE_PORT: int = int(os.getenv("VLLM_BASE_PORT", "30000"))
MAX_PARALLEL_MODELS: int = int(os.getenv("MAX_PARALLEL_MODELS", "6"))
