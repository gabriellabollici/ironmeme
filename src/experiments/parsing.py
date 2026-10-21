"""Parsing of raw VLM output, shared by inference, evaluation and experiments.

Kept free of client dependencies (openai, requests) so scoring can run
without the inference extras installed.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Return only the content after the last ``</think>`` tag, if present.

    "Thinking"/reasoning models (e.g. the Qwen3 "-thinking" variants) prepend
    a long chain-of-thought wrapped in ``<think>...</think>`` before the
    actual JSON answer. Some of these models even draft a full (sometimes
    invalid or later-revised) JSON blob *inside* their reasoning before
    settling on a final answer -- so we can't just strip the *first*
    ``<think>...</think>`` pair; we take everything after the *last*
    ``</think>`` closing tag, which is where the model's final, considered
    answer actually lives. If no closing tag is found, the text is
    returned unchanged (covers non-reasoning models, which never emit
    ``<think>`` at all).
    """
    matches = list(_THINK_CLOSE_RE.finditer(text))
    if matches:
        return text[matches[-1].end():].strip()
    return text


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_thinking(text.strip())
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the outermost bracketed span and try to parse it.
    # Try both object ({...}) and array ([...]) framing -- the expected
    # top-level shape differs by pipeline step (step 1 expects a JSON
    # object, steps 2/3 expect a JSON array of objects), and a naive
    # object-only search (find first "{", find last "}") mis-extracts
    # array-shaped answers: for "[{"a":1}, {"b":2}]" it would grab
    # '{"a":1}, {"b":2}' (missing the enclosing brackets -- invalid JSON,
    # "Extra data"). Trying both and keeping whichever actually parses
    # handles both shapes correctly. Whichever bracket opens first is tried
    # first: for a single-item array like '<answer>[{"a":1}]' (GLM wraps its
    # answer in <answer> tags, so the whole-text parse above fails), trying
    # "{" first would parse just the inner object and drop the list.
    frames = sorted(
        (("{", "}"), ("[", "]")),
        key=lambda f: cleaned.find(f[0]) if f[0] in cleaned else len(cleaned),
    )
    for open_ch, close_ch in frames:
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue

    logger.warning("Failed to parse JSON from VLM output:\n%s", text[:500])
    return None


def validate_list_result(
    parsed_json: Optional[Any],
    expected_key: str,
) -> Tuple[bool, List[Any]]:
    """Validate and extract a list of items from a VLM's parsed JSON output.

    Handles two response shapes seen from different steps/prompts:
      - dict: {"<expected_key>": [...]}  (or a single-key dict with any name)
      - list: [...] directly (steps 2/3's prompts explicitly ask for a bare
        JSON array, e.g. [{"identity": ..., "viewpoint": ...}, ...])
    """
    if parsed_json is None:
        return False, []

    if isinstance(parsed_json, list):
        items = parsed_json
    elif isinstance(parsed_json, dict):
        items = parsed_json.get(expected_key)
        if items is None:
            if len(parsed_json) == 1:
                items = next(iter(parsed_json.values()))
            else:
                return False, []
    else:
        return False, []

    if not isinstance(items, list):
        return False, []

    # Step 1 items are plain identity-name strings; steps 2/3 items are
    # dicts (e.g. {"identity": ..., "viewpoint": ...}) that get re-embedded
    # as JSON in the next prompt via format_user_prompt(), so they must be
    # preserved as-is rather than stringified into a Python repr.
    normalized = [x if isinstance(x, (dict, list)) else str(x) for x in items]
    return True, normalized
