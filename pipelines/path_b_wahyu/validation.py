"""Validate grounded inventory and enforce the single-gripper action contract."""

import re

from .planner import compile_transfer_groups


def validate_pixel(pixel, width, height):
    """Require an integer [u, v] inside the original image, excluding booleans."""
    if (
        not isinstance(pixel, list)
        or len(pixel) != 2
        or any(type(v) is not int for v in pixel)
        or not (0 <= pixel[0] < width and 0 <= pixel[1] < height)
    ):
        raise ValueError(f"Invalid original-image pixel: {pixel}")
    return pixel


def validate_inventory(raw, candidates, width, height):
    """Verify unique IDs, measured pixels, support labels, and physical relation references."""
    if not isinstance(raw, dict) or not isinstance(raw.get("objects"), list):
        raise ValueError("Perception must return an objects list")
    proposals = {o["id"]: o for o in candidates}
    objects = {}
    used = set()
    for item in raw["objects"]:
        oid = item.get("id")
        if (
            not isinstance(oid, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", oid)
            or oid in objects
        ):
            raise ValueError("Missing or duplicate object id")
        if not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError("Missing object category")
        candidate = item.get("candidate_id")
        if candidate is not None:
            if candidate not in proposals or candidate in used:
                raise ValueError("Unknown or reused candidate id")
            used.add(candidate)
            item["pixel"] = proposals[candidate]["pixel"]
        validate_pixel(item.get("pixel"), width, height)
        bbox = item.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(v) is not int for v in bbox)
            or not (
                0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height
            )
        ):
            raise ValueError("Invalid bounding box")
        if not (
            bbox[0] <= item["pixel"][0] < bbox[2]
            and bbox[1] <= item["pixel"][1] < bbox[3]
        ):
            raise ValueError("Object point is outside bounding box")
        if item.get("tray") not in ("grey_tray", "black_tray", "outside", "unknown"):
            raise ValueError("Invalid tray membership")
        if type(item.get("held_by_human")) is not bool:
            raise ValueError("Missing human-contact flag")
        objects[oid] = item
    relations = raw.get("relations", [])
    if not isinstance(relations, list):
        raise ValueError("Invalid relations")
    for rel in relations:
        if (
            rel.get("subject") not in objects
            or rel.get("object") not in objects
            or rel.get("relation") != "on_top_of"
            or rel["subject"] == rel["object"]
        ):
            raise ValueError("Invalid stacking relation")
    return {"objects": list(objects.values()), "relations": relations}


def validate_plan(raw, inventory, count, width, height):
    """Simulate gripper occupancy and attach trusted inventory pixels to accepted actions."""
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError("Expected exactly one result per instruction")
    objects = {o["id"]: o for o in inventory["objects"]}
    results = []
    for index, entry in enumerate(raw, 1):
        if entry.get("instruction_id") != index:
            raise ValueError(f"instruction {index}: Instruction order/id mismatch")
        actions = compile_transfer_groups(entry.get("actions_data"))
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"instruction {index}: Missing actions")
        # State belongs to this instruction only; later instructions use the original scene.
        holding = None
        remaining = {(r["subject"], r["object"]) for r in inventory["relations"]}
        clean = []
        for action in actions:
            kind = action.get("action")
            if kind in ("ask_user", "refuse"):
                field = "question" if kind == "ask_user" else "reason"
                if (
                    len(actions) != 1
                    or not isinstance(action.get(field), str)
                    or not action[field].strip()
                ):
                    raise ValueError(f"instruction {index}: Invalid ask/refuse action")
                clean.append({"action": kind, field: action[field]})
                continue
            if kind not in ("pick", "place") or action.get("object") not in objects:
                raise ValueError(f"instruction {index}: Unknown action or object id")
            obj = objects[action["object"]]
            if kind == "pick":
                if holding is not None:
                    raise ValueError(
                        f"instruction {index}: Single gripper cannot pick twice without place"
                    )
                if obj["tray"] not in ("grey_tray", "black_tray") or obj["name"] in (
                    "grey_tray",
                    "black_tray",
                    "plate",
                    "saucer",
                    "hand",
                ):
                    raise ValueError(f"instruction {index}: Ineligible pick target")
                if obj["held_by_human"] or any(
                    bottom == obj["id"] for _, bottom in remaining
                ):
                    raise ValueError(
                        f"instruction {index}: Pick blocked by human or stacked object"
                    )
                holding = obj["id"]
                remaining = {
                    (top, bottom) for top, bottom in remaining if top != holding
                }
            else:
                if holding is None:
                    raise ValueError(f"instruction {index}: Place without pick")
                if (
                    obj["held_by_human"]
                    or obj["name"] == "hand"
                    or obj["id"] == holding
                    or obj["tray"] not in ("grey_tray", "black_tray")
                ):
                    raise ValueError(f"instruction {index}: Invalid destination")
                if obj["name"] not in ("grey_tray", "black_tray", "plate", "saucer"):
                    remaining.add((holding, obj["id"]))
                holding = None
            # Ignore planner coordinates and use the trusted measured inventory point.
            clean.append(
                {
                    "action": kind,
                    "object": obj["id"],
                    "pixel": validate_pixel(obj["pixel"], width, height),
                }
            )
        results.append({"instruction_id": index, "actions_data": clean})
    return results
