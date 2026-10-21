"""VLM client for calling vLLM-served models via the OpenAI-compatible API.

Supports both a default client (legacy single-server) and per-request custom
``base_url`` for multi-server parallel execution.
"""
import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

from experiments.inference.config import VLLM_BASE_URL, VLLM_API_KEY, MAX_TOKENS, TEMPERATURE, REASONING_EFFORT
from experiments.parsing import try_parse_json

logger = logging.getLogger(__name__)

_clients: Dict[str, OpenAI] = {}


def check_server(base_url: Optional[str] = None, timeout: float = 5.0) -> bool:
    """Return True if the OpenAI-compatible server at *base_url* is reachable.

    Local vLLM servers expose ``/health`` (no auth). External hosted APIs
    (e.g. Moonshot's platform.moonshot.ai) don't -- for those we fall back
    to an authenticated GET on ``/models``, which every OpenAI-compatible
    endpoint implements.
    """
    url = base_url or VLLM_BASE_URL
    root = url.rsplit("/v1", 1)[0]
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    try:
        r = requests.get(root + "/health", timeout=timeout)
        if r.status_code == 200:
            return True
    except requests.RequestException:
        pass
    try:
        r = requests.get(url + "/models", headers=headers, timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _get_client(base_url: str) -> OpenAI:
    """Return (or create) an OpenAI client for a given base_url."""
    if base_url not in _clients:
        # "EMPTY" is what local vLLM expects; external APIs get the real
        # key via VLLM_API_KEY / MOONSHOT_API_KEY (see config.py).
        # Hosted APIs rate-limit (429) and occasionally 5xx; the SDK retries
        # those with backoff. Default 2 matches the SDK; raise it for long
        # hosted runs, since a call that exhausts retries loses that image.
        _clients[base_url] = OpenAI(
            base_url=base_url, api_key=VLLM_API_KEY,
            max_retries=int(os.getenv("VLM_MAX_RETRIES", "2")),
        )
    return _clients[base_url]


def encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }
    return mapping.get(ext, "image/png")


def call_vlm(
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    base_url: Optional[str] = None,
    model_name: str = "default",
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    no_system_role: bool = False,
) -> Dict[str, Any]:
    """Call a VLM server and return the parsed response.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    system_prompt : str
        System message content.
    user_prompt : str
        User message content (text part).
    base_url : str, optional
        Override the default vLLM base URL.  Used when multiple servers
        run in parallel, each on a different port.
    model_name : str
        Model name sent in the API request (default ``"default"``).
    max_tokens : int, optional
        Overrides the global MAX_TOKENS (default 1024). Needed for
        "thinking"/reasoning models, which emit a long chain-of-thought
        *before* the final JSON answer -- at the default budget they get
        cut off mid-reasoning, before ever emitting the JSON, causing
        both slow generation (always hitting the token cap) and
        systematic JSON-parse failures on the truncated output.
    reasoning_effort : str, optional
        Reasoning-depth control for hosted reasoning models (e.g. Kimi
        K3's ``low|high|max``; its default ``max`` bills a lot of
        chain-of-thought as output tokens). Falls back to the global
        REASONING_EFFORT env setting; when both are unset the field is
        not sent at all, so local vLLM models are unaffected. Passed via
        ``extra_body`` for compatibility with older OpenAI SDK versions.
    no_system_role : bool, optional
        Some models' chat templates have no ``system`` role and hard-fail
        on non-alternating roles (e.g. Molmo: "Conversation roles must
        alternate user/assistant/..."). When True, ``system_prompt`` is
        merged into the head of the user message instead of being sent
        as a separate system turn.
    """
    url = base_url or VLLM_BASE_URL
    client = _get_client(url)

    b64 = encode_image_b64(image_path)
    media = _guess_media_type(image_path)

    user_text = (
        f"{system_prompt.rstrip()}\n\n{user_prompt}" if no_system_role else user_prompt
    )
    messages = []
    if not no_system_role:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media};base64,{b64}",
                    },
                },
            ],
        },
    )

    logger.debug(
        "Calling VLM  url=%s  system=%.80s...  user=%.80s...",
        url, system_prompt, user_prompt,
    )

    effective_max_tokens = max_tokens or MAX_TOKENS
    effective_effort = reasoning_effort if reasoning_effort is not None else REASONING_EFFORT
    logger.info(
        "VLM call  model=%s  max_tokens=%d  reasoning_effort=%s  system=%.120s",
        model_name, effective_max_tokens, effective_effort or "<unset>", system_prompt,
    )

    extra: Dict[str, Any] = {}
    if effective_effort:
        extra["extra_body"] = {"reasoning_effort": effective_effort}

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=TEMPERATURE,
        **extra,
    )

    raw_text = response.choices[0].message.content.strip()
    usage = response.usage.model_dump() if response.usage else {}

    parsed = try_parse_json(raw_text)
    logger.info(
        "VLM response  model=%s  finish=%s  prompt_tokens=%s  completion_tokens=%s",
        response.model,
        getattr(getattr(response.choices[0], "finish_reason", None), "value", response.choices[0].finish_reason),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )

    return {
        "raw_text": raw_text,
        "parsed_json": parsed,
        "model": response.model or "",
        "usage": usage,
    }
