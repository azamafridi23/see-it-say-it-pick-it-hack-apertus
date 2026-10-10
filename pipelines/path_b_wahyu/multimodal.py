"""Orchestrate measured perception and Apertus instruction planning for one scene."""

import json
import os
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

# Keep earlier helper imports working for notebooks and existing callers.
from .apertus_client import encode_image as encode_image
from .apertus_client import image_to_data_url as image_to_data_url
from .apertus_client import make_json_request
from .apertus_client import parse_model_json as parse_model_json
from .instruction_planning import plan_instructions
from .measure import detect_objects
from .perception import (
    load_inventory,
    refine_components,
    refine_stacks,
    save_candidate_artifacts,
)
from .planner import compile_transfer_groups as compile_transfer_groups
from .validation import validate_inventory as validate_inventory
from .validation import validate_pixel as validate_pixel
from .validation import validate_plan as validate_plan


def llm_processing(
    instruction, image_path, *, client=None, model=None, debug_dir=None, cache_dir=None
):
    """Return validated actions while saving diagnostics even if a stage fails.

    A caller may inject a compatible SDK client for offline replay or local serving.
    Each instruction is interpreted independently against the original scene.
    """
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    model = model or os.getenv("APERTUS_MODEL", "apertus-v1.5-70b")
    client = client or OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("APERTUS_BASE_URL", "https://hackapertus.livemap.sh/v1"),
        timeout=180,
        max_retries=2,
    )
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    candidates, mask = detect_objects(image)
    artifacts = {"candidates": candidates}
    save_candidate_artifacts(image, candidates, mask, debug_dir)
    request = make_json_request(client, model, artifacts)
    try:
        inventory = load_inventory(
            image, image_path, candidates, client, model, cache_dir, request, artifacts
        )
        inventory = refine_components(image, inventory, candidates, request, artifacts)
        inventory = refine_stacks(image, inventory, candidates, request, artifacts)
        return plan_instructions(
            instruction, inventory, request, artifacts, width, height
        )
    finally:
        # Preserve failed replies as well as accepted stages for diagnosis.
        if debug_dir:
            (Path(debug_dir) / "trace.json").write_text(json.dumps(artifacts, indent=2))
