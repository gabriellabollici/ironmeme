"""Manage vLLM OpenAI-compatible VLM servers.

Supports launching a single server (legacy) or multiple servers in parallel,
one per model, each bound to a distinct port.

We use vLLM instead of sglang because sglang's precompiled `sgl-kernel`
wheels dropped SM80 (NVIDIA A100, compute capability 8.0) support in favor
of SM90/SM100-only binaries starting around sgl-kernel 0.3.x. Every sglang
release that supports Qwen3-VL (>=0.5.3) pins one of those SM80-incompatible
sgl-kernel versions, so sglang cannot serve Qwen3-VL on A100 hardware
without building sgl-kernel from source. vLLM has broader GPU architecture
support and is the officially recommended serving backend for Qwen3-VL on
A100 (see https://github.com/QwenLM/Qwen3-VL, "vllm>=0.11.0").
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import logging
from typing import Dict, List, Optional

import requests

from experiments.inference.config import (
    VLLM_HOST, VLLM_PORT, VLLM_BASE_URL,
    VLLM_STARTUP_TIMEOUT, VLLM_POLL_INTERVAL,
    VLLM_BASE_PORT, HF_CACHE_DIR,
    VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTILIZATION,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy single-server helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------
_server_proc: Optional[subprocess.Popen] = None
_server_log_path: Optional[str] = None


def start_server(max_model_len: Optional[int] = None) -> None:
    global _server_proc, _server_log_path
    from experiments.inference.config import MODEL_ID, MODEL_SHORT_NAME
    _server_proc, _server_log_path = _launch_server(
        MODEL_ID, VLLM_PORT, served_name=MODEL_SHORT_NAME, max_model_len=max_model_len,
    )
    _wait_until_ready(VLLM_HOST, VLLM_PORT, _server_proc, _server_log_path)
    logger.info("vLLM server ready on %s", VLLM_BASE_URL)


def stop_server() -> None:
    global _server_proc
    _stop_proc(_server_proc)
    _server_proc = None


def get_base_url() -> str:
    return VLLM_BASE_URL


# ---------------------------------------------------------------------------
# Multi-server helpers
# ---------------------------------------------------------------------------
_server_procs: Dict[str, subprocess.Popen] = {}
_server_log_paths: Dict[str, str] = {}


def start_servers_for_models(
    model_ids: List[str],
    served_names: Optional[Dict[str, str]] = None,
    max_model_lens: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, str]:
    """Launch one vLLM server per model, each on a unique port.

    Parameters
    ----------
    model_ids : list of str
        Hugging Face model IDs to serve.
    served_names : dict, optional
        Maps model_id -> the name vLLM should register it under
        (``--served-model-name``). Callers must pass the same name as
        ``model_name`` in their ``call_vlm``/``process_image`` calls,
        since vLLM (unlike sglang) validates the ``model`` field in API
        requests against its served model name.
    max_model_lens : dict, optional
        Maps model_id -> per-model ``max_model_len`` override (models.yaml);
        models without one use VLLM_MAX_MODEL_LEN.

    Returns a dict mapping model_id → base_url (e.g. "http://127.0.0.1:30001/v1").
    """
    if len(model_ids) > 64:
        raise ValueError("Too many models; max 64 supported.")

    served_names = served_names or {}
    max_model_lens = max_model_lens or {}

    port_map: Dict[str, int] = {}
    for idx, model_id in enumerate(model_ids):
        port_map[model_id] = VLLM_BASE_PORT + idx

    # Launch all servers
    procs: Dict[str, subprocess.Popen] = {}
    log_paths: Dict[str, str] = {}
    for model_id, port in port_map.items():
        logger.info("Launching server for %s on port %d", model_id, port)
        proc, log_path = _launch_server(
            model_id, port, served_name=served_names.get(model_id),
            max_model_len=max_model_lens.get(model_id),
        )
        procs[model_id] = proc
        log_paths[model_id] = log_path

    # Wait for all to be ready
    base_urls: Dict[str, str] = {}
    for model_id, port in port_map.items():
        host_port_url = f"http://{VLLM_HOST}:{port}/v1"
        _wait_until_ready(VLLM_HOST, port, procs[model_id], log_paths[model_id])
        logger.info("Server ready for %s on %s", model_id, host_port_url)
        base_urls[model_id] = host_port_url

    _server_procs.update(procs)
    _server_log_paths.update(log_paths)
    return base_urls


def stop_all_servers() -> None:
    """Stop all servers launched by ``start_servers_for_models``."""
    for model_id, proc in _server_procs.items():
        logger.info("Stopping server for %s (pid=%s)", model_id, proc.pid)
        _stop_proc(proc)
    _server_procs.clear()


def get_server_for_model(model_id: str) -> Optional[str]:
    """Return the base_url for a running model server, or None."""
    proc = _server_procs.get(model_id)
    if proc is None or proc.poll() is not None:
        return None
    idx = list(_server_procs.keys()).index(model_id)
    port = VLLM_BASE_PORT + idx
    return f"http://{VLLM_HOST}:{port}/v1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _patch_hf_config_for_deepseek_vl2(model_id: str) -> None:
    """Patch cached config.json for deepseek-vl2 to fill in fields the
    checkpoint's ``language_config`` leaves unset.

    The HF config.json for deepseek-vl2 nests the V2 config inside
    ``language_config`` but omits several fields entirely (rather than
    setting them to the correct value). vLLM's ``--hf-overrides`` only
    patches the top-level config, so ``DeepseekV2ForCausalLM``/
    ``DeepseekVLV2ForCausalLM`` still sees these fields as unset and either
    crashes or silently falls back to ``transformers``' ``DeepseekV2Config``
    class defaults. We fix this by writing the values directly into the
    cached config.json.

    The most important of these is ``num_hidden_layers``: it is *not
    present* in the real checkpoint's ``language_config``, so vLLM falls
    back to ``transformers.DeepseekV2Config``'s class default of 32.
    The actual deepseek-vl2 checkpoint only has 30 language-model layers
    (indices 0-29). That 32-vs-30 mismatch is what produces:
        ValueError: Following weights were not initialized from checkpoint:
        {'language_model.model.layers.31...', 'language_model.model.layers.30...'}
    This used to be silently tolerated by vLLM's V0 engine, but V0 has
    since been removed entirely (the ``VLLM_USE_V1=0`` workaround that
    used to live here is a no-op on current vLLM -- see
    "Unknown vLLM environment variable detected: VLLM_USE_V1" in the
    server log), so V1's strict weight-loading check now hard-fails
    instead. Explicitly setting num_hidden_layers=30 is the actual fix.
    """
    # NOTE: deliberately use HF_CACHE_DIR (config.py's computed default),
    # NOT os.environ.get("HF_HOME"). _launch_server() always overrides the
    # *subprocess*'s HF_HOME to HF_CACHE_DIR before spawning vLLM, but this
    # function runs in the *parent* process, whose HF_HOME may still be
    # whatever the calling shell/sbatch script exported (which can point
    # to a different directory than HF_CACHE_DIR). Using os.environ here
    # previously caused this function to silently patch the wrong
    # directory (or no-op entirely) whenever the shell's HF_HOME didn't
    # match HF_CACHE_DIR.
    cache_dir = HF_CACHE_DIR
    # Resolve the snapshot dir
    repo_id = model_id
    config_path = os.path.join(
        cache_dir, "hub", f"models--{repo_id.replace('/', '--')}",
        "snapshots",
    )
    if not os.path.isdir(config_path):
        return
    for snap in os.listdir(config_path):
        cfg = os.path.join(config_path, snap, "config.json")
        if not os.path.isfile(cfg):
            continue
        with open(cfg, "r") as f:
            data = json.load(f)
        lang_cfg = data.get("language_config") or data.get("text_config")
        if lang_cfg and (
            lang_cfg.get("kv_lora_rank") is None
            or lang_cfg.get("num_hidden_layers") is None
        ):
            lang_cfg["kv_lora_rank"] = 512
            # NOTE: q_lora_rank is deliberately NOT patched here. The real
            # checkpoint's config.json explicitly sets q_lora_rank=null
            # (confirmed via the actual safetensors tensor shapes: layer 0's
            # self_attn.q_proj.weight is [6144, 2560], a single dense
            # projection -- not the low-rank q_a_proj/q_b_proj split used
            # when q_lora_rank is set). transformers.DeepseekV2Config's
            # class default for q_lora_rank is 1536, and a previous version
            # of this function used `lang_cfg.get("q_lora_rank") is None`
            # to decide whether to fill it in -- but that can't distinguish
            # "key absent" from "key explicitly null", so it always forced
            # q_lora_rank=1536, making vLLM build the wrong (low-rank)
            # attention module and then fail with:
            #   KeyError: 'layers.0.self_attn.q_proj.weight'
            # (vLLM looks for q_a_proj/q_b_proj weights instead, which
            # don't exist in the checkpoint). Leave q_lora_rank as-is.
            #
            # qk_rope_head_dim=64 and v_head_dim=128 already match
            # transformers.DeepseekV2Config's class defaults for this
            # checkpoint (also confirmed via tensor shapes), so they don't
            # strictly need patching, but setting them explicitly is
            # harmless and documents the expected architecture.
            if lang_cfg.get("qk_rope_head_dim") is None:
                lang_cfg["qk_rope_head_dim"] = 64
            if lang_cfg.get("v_head_dim") is None:
                lang_cfg["v_head_dim"] = 128
            # Real checkpoint only has layers 0-29 (30 layers); without
            # this, vLLM defaults to transformers' DeepseekV2Config class
            # default of num_hidden_layers=32 and instantiates two extra
            # layers with no matching weights in the checkpoint.
            if lang_cfg.get("num_hidden_layers") is None:
                lang_cfg["num_hidden_layers"] = 30
            with open(cfg, "w") as wf:
                json.dump(data, wf, indent=2)
            logger.info(
                "Patched %s: added kv_lora_rank=512, num_hidden_layers=30 "
                "(and other missing fields) to language_config", cfg,
            )
            break


def _launch_server(
    model_id: str, port: int, served_name: Optional[str] = None,
    max_model_len: Optional[int] = None,
) -> "tuple[subprocess.Popen, str]":
    """Start a vLLM OpenAI-compatible server subprocess for a given model.

    stdout/stderr are redirected to a log file on disk rather than
    ``subprocess.PIPE``. A pipe's OS buffer is small (typically 64 KiB);
    since vLLM logs far more than that during startup and nothing reads
    from the pipe until the process has already exited, the subprocess
    would eventually block on write() once the pipe filled up -- hanging
    forever with no output and no crash, which looks identical to a slow
    startup until the ``_wait_until_ready`` timeout fires. Writing to a
    file avoids this deadlock entirely and gives us a live log to tail.

    Parameters
    ----------
    max_model_len : int, optional
        Overrides the global VLLM_MAX_MODEL_LEN for this model. Needed
        for models whose native context length is shorter than the
        pipeline's default (e.g. deepseek-vl2-small's 4096) -- vLLM
        refuses to start if --max-model-len exceeds what the model
        actually supports.

    Returns
    -------
    (proc, log_path) : the subprocess handle and the path to its
        combined stdout/stderr log file.
    """
    effective_max_model_len = max_model_len or VLLM_MAX_MODEL_LEN
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", VLLM_HOST,
        "--port", str(port),
        "--trust-remote-code",
        "--max-model-len", str(effective_max_model_len),
        "--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION),
    ]
    # DeepSeek-VL2 uses a custom architecture (DeepseekVLV2ForCausalLM)
    # that vLLM doesn't auto-detect from the model's config.json. Without
    # this override, vLLM falls back to DeepseekV2Attention which expects
    # kv_lora_rank to be set in the V2 config -- but VL2's config uses a
    # different structure, leaving kv_lora_rank=None and crashing with
    # TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'.
    # DeepSeek-VL2 needs patches applied to the cached config.json on disk
    # because --hf-overrides only patches the top-level config, but
    # DeepseekVLV2ForCausalLM passes language_config (a nested sub-config)
    # to DeepseekV2ForCausalLM, which reads kv_lora_rank from there.
    if "deepseek-vl2" in model_id.lower():
        cmd += ["--hf-overrides", json.dumps({
            "architectures": ["DeepseekVLV2ForCausalLM"],
            "kv_lora_rank": 512,
        })]
        _patch_hf_config_for_deepseek_vl2(model_id)
    if served_name:
        cmd += ["--served-model-name", served_name]
    logger.info("vLLM command: %s", " ".join(cmd))

    # Disable vLLM's flashinfer-based top-k/top-p sampler. It JIT-compiles
    # a CUDA kernel at warmup time via nvcc+ninja, which fails with
    # "CUDA compiler and CUDA toolkit headers are incompatible" whenever
    # the pip-installed nvcc/cccl/cutlass headers used for CUDA_HOME don't
    # exactly match each other's expected toolkit version (a mismatch we
    # can't reliably control on a cluster with no system CUDA toolkit and
    # multiple independent pip-installed CUDA component packages). Setting
    # this falls back to vLLM's built-in (non-JIT) sampler implementation,
    # which is functionally equivalent for our use case and avoids the
    # whole JIT compilation step entirely. See
    # https://github.com/vllm-project/vllm/issues/44305 for the identical
    # error on a different pip-based CUDA 13 setup.
    env = {
        **os.environ,
        "HF_HOME": HF_CACHE_DIR,
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    # NOTE: DeepSeek-VL2's config.json has a layer-count mismatch with its
    # checkpoint (config falls back to num_hidden_layers=32 while the
    # checkpoint only has layers 0-29), which used to make vLLM V1's strict
    # track_weights_loading raise ValueError. This used to be worked around
    # by forcing the (more lenient) V0 engine via VLLM_USE_V1=0, but V0 has
    # since been removed from vLLM entirely -- that env var is now a no-op
    # ("Unknown vLLM environment variable detected: VLLM_USE_V1" in recent
    # server logs) and the real fix is patching num_hidden_layers=30 into
    # the cached config.json, done in _patch_hf_config_for_deepseek_vl2.

    log_dir = os.environ.get("VLLM_SERVER_LOG_DIR") or tempfile.gettempdir()
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vllm_server_port{port}.log")
    logger.info("vLLM server log: %s", log_path)

    log_file = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    log_file.close()  # the child keeps its own fd; safe to close ours
    return proc, log_path


def _stop_proc(proc: Optional[subprocess.Popen]) -> None:
    """Terminate a subprocess, killing it after 30 s if needed."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("Server (pid=%s) did not stop; killing.", proc.pid)
        proc.kill()


def _read_log_tail(log_path: str, n_bytes: int = 4000) -> str:
    """Read the log file written by ``_launch_server`` and return the tail."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            return f.read().decode(errors="replace")
    except OSError as e:
        return f"<could not read log file {log_path}: {e}>"


def _wait_until_ready(
    host: str, port: int, proc: subprocess.Popen, log_path: str,
) -> None:
    """Poll the health endpoint until the server is ready."""
    url = f"http://{host}:{port}/health"
    deadline = time.time() + VLLM_STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            output = _read_log_tail(log_path, n_bytes=200_000)
            # Keep the tail of the output (where the actual exception/traceback
            # is) rather than the head, and log the full output separately so
            # nothing is lost even if it's very long.
            logger.error(
                "Full stdout/stderr for failed server on port %d (log: %s):\n%s",
                port, log_path, output,
            )
            tail = output[-4000:]
            raise RuntimeError(
                f"vLLM server on port {port} exited early "
                f"(code={proc.returncode}).\n"
                f"{tail}"
            )
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        logger.debug("Port %d not ready; retrying in %ds", port, VLLM_POLL_INTERVAL)
        time.sleep(VLLM_POLL_INTERVAL)

    tail = _read_log_tail(log_path)
    logger.error(
        "vLLM server on port %d did not become ready within %ds. "
        "Last log output (log: %s):\n%s",
        port, VLLM_STARTUP_TIMEOUT, log_path, tail,
    )
    raise TimeoutError(
        f"vLLM server on port {port} did not become ready within "
        f"{VLLM_STARTUP_TIMEOUT}s. Log tail:\n{tail}"
    )
