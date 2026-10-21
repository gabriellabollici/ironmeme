import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.inference.vlm_client import call_vlm
from experiments.inference.prompt_utils import load_prompt, format_user_prompt, load_benchmark_lookup, get_gt_context
from experiments.parsing import validate_list_result
from experiments.inference.output_utils import write_step_output, write_final_output
from experiments.inference.image_utils import load_image_metadata

logger = logging.getLogger(__name__)

EXPECTED_KEYS = {
    1: "social_identities",
    2: "viewpoints",
    3: "narrative_roles",
}


def process_image(
    image_path: str,
    base_url: Optional[str] = None,
    model_name: str = "default",
    max_tokens: Optional[int] = None,
    no_system_role: bool = False,
) -> Dict[str, Any]:
    """Run the 3-step pipeline on a single image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    base_url : str, optional
        vLLM server URL to use.  When ``None`` the default from config is used.
    model_name : str
        Model name sent in the API request.
    max_tokens : int, optional
        Overrides the global MAX_TOKENS for this model's VLM calls. See
        vlm_client.call_vlm()'s docstring -- needed for "thinking" models
        that emit a long chain-of-thought before the final JSON answer.
    no_system_role : bool, optional
        Forwarded to vlm_client.call_vlm() -- merges the system prompt
        into the user message for models whose chat template has no
        system role (e.g. Molmo).
    """
    meta = load_image_metadata(image_path)
    image_stem = meta["stem"]
    logger.info("===== Processing image: %s =====", image_stem)

    benchmark_lookup = load_benchmark_lookup()
    gt_social_ids, gt_viewpoints = get_gt_context(image_path, benchmark_lookup)

    all_steps: List[Dict[str, Any]] = []
    social_ids: Optional[List[str]] = None
    viewpoints_context: Optional[List[Dict[str, str]]] = None
    success = True

    for step in (1, 2, 3):
        if step == 2 and gt_social_ids is not None:
            context_ids = gt_social_ids
            logger.info("  Step 2 using GT social identities: %s", context_ids)
        else:
            context_ids = social_ids

        if step == 3 and gt_viewpoints is not None:
            context_vps = gt_viewpoints
            logger.info("  Step 3 using GT viewpoints: %s", context_vps)
        else:
            context_vps = viewpoints_context

        step_result = _run_step(
            step, image_path, context_ids, context_vps,
            base_url=base_url, model_name=model_name, max_tokens=max_tokens,
            no_system_role=no_system_role,
        )
        all_steps.append(step_result)

        if step == 1:
            valid, extracted = validate_list_result(
                step_result.get("parsed_json"), EXPECTED_KEYS[1]
            )
            if valid:
                social_ids = extracted
                logger.info("  Step 1 -> %d social identities: %s", len(social_ids), social_ids)
            else:
                logger.warning("  Step 1 failed validation; Steps 2-3 will proceed without social identities.")
                success = False
                social_ids = []
        elif step == 2 and gt_viewpoints is None:
            valid, extracted = validate_list_result(
                step_result.get("parsed_json"), EXPECTED_KEYS[2]
            )
            if valid:
                viewpoints_context = extracted
                logger.info("  Step 2 -> %d viewpoints: %s", len(viewpoints_context), viewpoints_context)
            else:
                logger.warning("  Step 2 failed validation; Step 3 will proceed without viewpoints context.")
                success = False
                viewpoints_context = []
        elif step == 3:
            valid, extracted = validate_list_result(
                step_result.get("parsed_json"), EXPECTED_KEYS[3]
            )
            if valid:
                logger.info("  Step 3 -> %d narrative roles: %s", len(extracted), extracted)
            else:
                logger.warning("  Step 3 failed validation.")
                success = False

    write_final_output(image_stem, all_steps, model_name=model_name)
    logger.info("===== Finished image: %s (success=%s) =====", image_stem, success)

    return {
        "image_path": image_path,
        "social_identities": social_ids or [],
        "viewpoints": _extract_list_from_step(all_steps, 2),
        "narrative_roles": _extract_list_from_step(all_steps, 3),
        "steps": all_steps,
        "success": success,
    }


def process_image_no_gt(
    image_path: str,
    base_url: Optional[str] = None,
    model_name: str = "default",
    max_tokens: Optional[int] = None,
    no_system_role: bool = False,
) -> Dict[str, Any]:
    """Run the 3-step pipeline on a single image without any GT context.

    The regular process_image injects benchmark GT into the later steps
    (step 2 gets the GT identities, step 3 the GT identity/viewpoint pairs).
    Here the steps are fully chained on the model's own predictions instead:

      Step 1  social identities   <- image only (unchanged)
      Step 2  viewpoints          <- Step 1's *predicted* identities
      Step 3  narrative roles     <- Step 2's *predicted* identity/viewpoint pairs

    The prompts are the same; only the values substituted into
    {social_identities} and {viewpoints_context} change. The GT is still
    used downstream for scoring, just never shown to the model. Same
    signature and return shape as process_image.
    """
    meta = load_image_metadata(image_path)
    image_stem = meta["stem"]
    logger.info("===== Processing image (no GT): %s =====", image_stem)

    all_steps: List[Dict[str, Any]] = []
    social_ids: Optional[List[str]] = None
    viewpoints_context: Optional[List[Dict[str, str]]] = None
    success = True

    for step in (1, 2, 3):
        if step == 2:
            logger.info("  Step 2 using predicted social identities: %s", social_ids)
        elif step == 3:
            logger.info("  Step 3 using predicted viewpoints: %s", viewpoints_context)

        step_result = _run_step(
            step, image_path, social_ids, viewpoints_context,
            base_url=base_url, model_name=model_name, max_tokens=max_tokens,
            no_system_role=no_system_role,
        )
        all_steps.append(step_result)

        valid, extracted = validate_list_result(
            step_result.get("parsed_json"), EXPECTED_KEYS[step]
        )
        if step == 1:
            if valid:
                social_ids = extracted
                logger.info("  Step 1 -> %d social identities: %s", len(social_ids), social_ids)
            else:
                logger.warning("  Step 1 failed validation; Steps 2-3 will proceed without social identities.")
                success = False
                social_ids = []
        elif step == 2:
            if valid:
                viewpoints_context = extracted
                logger.info("  Step 2 -> %d viewpoints: %s", len(viewpoints_context), viewpoints_context)
            else:
                logger.warning("  Step 2 failed validation; Step 3 will proceed without viewpoints context.")
                success = False
                viewpoints_context = []
        else:
            if valid:
                logger.info("  Step 3 -> %d narrative roles: %s", len(extracted), extracted)
            else:
                logger.warning("  Step 3 failed validation.")
                success = False

    write_final_output(image_stem, all_steps, model_name=model_name)
    logger.info("===== Finished image: %s (success=%s) =====", image_stem, success)

    return {
        "image_path": image_path,
        "social_identities": social_ids or [],
        "viewpoints": _extract_list_from_step(all_steps, 2),
        "narrative_roles": _extract_list_from_step(all_steps, 3),
        "steps": all_steps,
        "success": success,
    }


def _run_step(
    step: int,
    image_path: str,
    social_ids: Optional[List[str]],
    viewpoints_context: Optional[List[Dict[str, str]]] = None,
    base_url: Optional[str] = None,
    model_name: str = "default",
    max_tokens: Optional[int] = None,
    no_system_role: bool = False,
) -> Dict[str, Any]:
    logger.info("  Step %d starting...", step)

    system_prompt, user_template = load_prompt(step)

    if step == 1:
        user_prompt = user_template
    else:
        user_prompt = format_user_prompt(
            user_template, social_ids, viewpoints_context
        )

    result = call_vlm(
        image_path, system_prompt, user_prompt,
        base_url=base_url, model_name=model_name, max_tokens=max_tokens,
        no_system_role=no_system_role,
    )

    valid, extracted = validate_list_result(result["parsed_json"], EXPECTED_KEYS[step])
    logger.info(
        "  Step %d -> valid=%s, items=%d, raw_len=%d",
        step, valid, len(extracted), len(result["raw_text"]),
    )

    write_step_output(
        step=step,
        image_stem=Path(image_path).stem,
        parsed_json=result["parsed_json"],
        raw_text=result["raw_text"],
        extra_metadata={
            "model": result["model"],
            "usage": result["usage"],
            "validation": {"valid": valid, "item_count": len(extracted)},
        },
        model_name=model_name,
    )

    return {
        "step": step,
        "step_name": f"{step:02d}_{EXPECTED_KEYS[step]}",
        "parsed_json": result["parsed_json"],
        "raw_text": result["raw_text"],
        "metadata": {
            "model": result["model"],
            "usage": result["usage"],
            "validation": {"valid": valid, "item_count": len(extracted)},
        },
    }


def _extract_list_from_step(steps: List[Dict[str, Any]], step: int) -> List[str]:
    for s in steps:
        if s["step"] == step:
            _, items = validate_list_result(s.get("parsed_json"), EXPECTED_KEYS[step])
            return items
    return []
