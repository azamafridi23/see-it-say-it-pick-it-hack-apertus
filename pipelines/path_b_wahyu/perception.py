"""Build scene inventories from coarse visual boxes and targeted close-up passes."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .apertus_client import encode_image, parse_model_json
from .grounding import (
    add_dark_tray_reference,
    bbox_iou,
    ground_inventory,
    reconcile_tray_membership,
    reconcile_visible_occlusion,
)
from .template_prompt import COMPONENT_PROMPT, PERCEPTION_PROMPT, STACK_PROMPT
from .validation import validate_inventory


def save_candidate_artifacts(image, candidates, mask, debug_dir):
    """Save boxes, segmentation, and close-up tiles only when tracing is enabled."""
    if not debug_dir:
        return
    height, width = image.shape[:2]
    annotated = image.copy()
    for obj in candidates:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(annotated, tuple(obj["pixel"]), 6, (255, 0, 255), -1)
        cv2.putText(
            annotated,
            obj["id"],
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    # Close-up tiles preserve details that disappear in a full workcell photo.
    tiles = []
    for obj in candidates:
        x1, y1, x2, y2 = obj["bbox"]
        crop = image[
            max(0, y1 - 20) : min(height, y2 + 20),
            max(0, x1 - 20) : min(width, x2 + 20),
        ]
        factor = min(240 / crop.shape[1], 210 / crop.shape[0])
        crop = cv2.resize(crop, None, fx=factor, fy=factor)
        tile = np.full((256, 256, 3), 35, dtype=np.uint8)
        tile[40 : 40 + crop.shape[0], : crop.shape[1]] = crop
        cv2.putText(
            tile, obj["id"], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        tiles.append(tile)
    sheet = None
    if tiles:
        columns = 4
        while len(tiles) % columns:
            tiles.append(np.zeros((256, 256, 3), dtype=np.uint8))
        sheet = np.vstack(
            [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
        )
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "candidates.jpg"), annotated)
    cv2.imwrite(str(debug_dir / "mask.png"), mask)
    if sheet is not None:
        cv2.imwrite(str(debug_dir / "crops.jpg"), sheet)


def load_inventory(
    image, image_path, candidates, client, model, cache_dir, request, artifacts
):
    """Load or request raw perception, then ground it using current image evidence."""
    height, width = image.shape[:2]
    photo = {"type": "image_url", "image_url": {"url": encode_image(image)}}
    cache_path = None
    if cache_dir:
        key = hashlib.sha256(
            Path(image_path).read_bytes()
            + PERCEPTION_PROMPT.encode()
            + model.encode()
            + str(client.base_url).encode()
        ).hexdigest()
        cache_path = Path(cache_dir) / (key + ".json")

    def grounded(raw):
        return validate_inventory(
            reconcile_tray_membership(
                add_dark_tray_reference(ground_inventory(raw, image, candidates), image)
            ),
            [],
            width,
            height,
        )

    if cache_path and cache_path.exists():
        raw = json.loads(cache_path.read_text())
        inventory = grounded(raw)
        artifacts["inventory_cached"] = True
        artifacts["inventory"] = inventory
    else:
        inventory = request(
            PERCEPTION_PROMPT,
            [photo, {"type": "text", "text": "Identify the objects and their boxes."}],
            grounded,
            "inventory",
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            last_reply = next(
                artifacts[k]
                for k in reversed(artifacts)
                if k.startswith("inventory_reply_")
            )
            cache_path.write_text(json.dumps(parse_model_json(last_reply), indent=2))
    return inventory


def refine_components(image, inventory, candidates, request, artifacts):
    """Classify uncovered neutral proposals without duplicating known objects."""
    height, width = image.shape[:2]
    # Inspect unexplained neutral proposals at full local detail.
    unexplained = []
    for candidate in candidates:
        if candidate["color"] != "neutral" or candidate["aspect_ratio"] > 2.8:
            continue
        cx, cy = candidate["pixel"]
        if not any(
            o["name"] in ("grey_tray", "black_tray")
            and o["bbox"][0] < cx < o["bbox"][2]
            and o["bbox"][1] < cy < o["bbox"][3]
            for o in inventory["objects"]
        ):
            continue
        if any(
            o["name"] not in ("grey_tray", "black_tray")
            and (
                o["bbox"][0] <= cx < o["bbox"][2]
                and o["bbox"][1] <= cy < o["bbox"][3]
                or bbox_iou(o["bbox"], candidate["bbox"]) > 0.1
                or (
                    (o["name"] == "mug" or o["held_by_human"])
                    and np.linalg.norm(np.array(o["pixel"]) - candidate["pixel"])
                    < max(o["bbox"][2] - o["bbox"][0], o["bbox"][3] - o["bbox"][1])
                    * 0.8
                )
            )
            for o in inventory["objects"]
        ):
            continue
        unexplained.append(candidate)
    if unexplained:
        part_content = []
        for candidate in unexplained:
            x1, y1, x2, y2 = candidate["bbox"]
            crop = image[
                max(0, y1 - 80) : min(height, y2 + 80),
                max(0, x1 - 80) : min(width, x2 + 80),
            ]
            part_content.extend(
                [
                    {"type": "text", "text": candidate["id"]},
                    {"type": "image_url", "image_url": {"url": encode_image(crop)}},
                ]
            )
        lookup = {o["id"]: o for o in unexplained}

        def validate_parts(raw):
            if not isinstance(raw, dict) or not isinstance(raw.get("parts"), list):
                raise ValueError(
                    "Expected parts array, or empty parts array if no components"
                )
            seen = set()
            for part in raw["parts"]:
                if (
                    part.get("candidate_id") not in lookup
                    or part["candidate_id"] in seen
                    or part.get("name")
                    not in ("grey_part", "white_part", "small_white_part")
                ):
                    raise ValueError("Invalid component category/candidate")
                seen.add(part["candidate_id"])
            return raw["parts"]

        try:
            parts = request(
                COMPONENT_PROMPT,
                part_content,
                validate_parts,
                "components",
            )
        except (ValueError, KeyError, TypeError) as exc:
            artifacts.setdefault("refinement_errors", {})["components"] = str(exc)
            parts = []
        for n, part in enumerate(parts, 1):
            proposal = lookup[part["candidate_id"]]
            support = next(
                o["name"]
                for o in inventory["objects"]
                if o["name"] in ("grey_tray", "black_tray")
                and o["bbox"][0] < proposal["pixel"][0] < o["bbox"][2]
                and o["bbox"][1] < proposal["pixel"][1] < o["bbox"][3]
            )
            inventory["objects"].append(
                {
                    "id": f'{part["name"]}_cv_{n}',
                    "name": part["name"],
                    "bbox": proposal["bbox"],
                    "pixel": proposal["pixel"],
                    "tray": support,
                    "held_by_human": False,
                    "candidate_id": proposal["id"],
                    "grounding": "component_closeup",
                }
            )

    return inventory


def refine_stacks(image, inventory, candidates, request, artifacts):
    """Resolve overlapping object support and validate the final grounded graph."""
    height, width = image.shape[:2]
    # Zoom in on overlapping manipulation objects to resolve physical stacking.
    manipulable = [
        o
        for o in inventory["objects"]
        if o["name"] not in ("grey_tray", "black_tray", "plate", "saucer", "hand")
    ]
    pairs = []
    for i, obj in enumerate(manipulable):
        for other in manipulable[i + 1 :]:
            a, b = obj["bbox"], other["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if intersection / max(1, smaller) > 0.15:
                pairs.append((obj, other))
    if pairs:
        content = []
        for obj, other in pairs:
            a, b = obj["bbox"], other["bbox"]
            x1, y1 = max(0, min(a[0], b[0]) - 35), max(0, min(a[1], b[1]) - 35)
            x2, y2 = min(width, max(a[2], b[2]) + 35), min(height, max(a[3], b[3]) + 35)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "objects": [
                                    {
                                        "id": o["id"],
                                        "name": o["name"],
                                        "box_in_crop": [
                                            o["bbox"][0] - x1,
                                            o["bbox"][1] - y1,
                                            o["bbox"][2] - x1,
                                            o["bbox"][3] - y1,
                                        ],
                                    }
                                    for o in (obj, other)
                                ]
                            }
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image(image[y1:y2, x1:x2])},
                    },
                ]
            )

        def validate_stacks(raw):
            if not isinstance(raw, dict) or not isinstance(raw.get("relations"), list):
                raise ValueError("Expected relations array")
            known = {o["id"] for o in manipulable}
            for rel in raw["relations"]:
                if (
                    rel.get("relation") != "on_top_of"
                    or rel.get("subject") not in known
                    or rel.get("object") not in known
                    or rel["subject"] == rel["object"]
                ):
                    raise ValueError("Use distinct known IDs and on_top_of only")
            return raw["relations"]

        try:
            stacks = request(
                STACK_PROMPT,
                content,
                validate_stacks,
                "stacks",
            )
        except (ValueError, KeyError, TypeError) as exc:
            artifacts.setdefault("refinement_errors", {})["stacks"] = str(exc)
            stacks = None
        paired_ids = {frozenset((a["id"], b["id"])) for a, b in pairs}
        if stacks is not None:
            inventory["relations"] = [
                r
                for r in inventory["relations"]
                if frozenset((r["subject"], r["object"])) not in paired_ids
            ] + stacks
    inventory = validate_inventory(
        reconcile_visible_occlusion(inventory, image), candidates, width, height
    )
    artifacts["inventory"] = inventory
    return inventory
