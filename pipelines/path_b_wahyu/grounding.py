"""Ground Apertus's coarse boxes on measured foreground in original pixels."""

import json
import math
import re

import cv2
import numpy as np

CONTAINERS = {"grey_tray", "black_tray", "plate", "saucer"}
ALIASES = {
    "small_part": "small_white_part",
    "cup": "mug",
    "green_mug": "mug",
    "orange": "mandarin",
    "red_block": "red_cube",
    "white_plate": "plate",
}


def normalized_bbox_to_pixels(raw, width, height):
    """Convert a validated 0–1000 box to original-image pixels exactly once."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raise ValueError("Bounding box string is not a JSON array")
    if (
        not isinstance(raw, list)
        or len(raw) != 4
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in raw
        )
    ):
        raise ValueError("Missing normalized bounding box")
    if not (0 <= raw[0] < raw[2] <= 1000 and 0 <= raw[1] < raw[3] <= 1000):
        raise ValueError(f"Invalid normalized box: {raw}")
    box = [
        round(raw[0] * width / 1000),
        round(raw[1] * height / 1000),
        round(raw[2] * width / 1000),
        round(raw[3] * height / 1000),
    ]
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Empty box after pixel conversion")
    return box


def bbox_iou(a, b):
    """Return intersection-over-union for two [x1, y1, x2, y2] pixel boxes."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0


def foreground_mask(image, name):
    """Select category-compatible HSV foreground for local pixel grounding."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if name in ("red_cube", "red_gear"):
        mask = ((h < 12) | (h >= 170)) & (s > 65) & (v > 55)
    elif name == "yellow_gear":
        mask = (h >= 15) & (h < 45) & (s > 65) & (v > 55)
    elif name == "mandarin":
        mask = ((h < 25) | (h >= 170)) & (s > 90) & (v > 75)
    elif name == "banana":
        mask = (h >= 12) & (h < 45) & (s > 65) & (v > 75)
    elif name == "mug":
        mask = (h >= 20) & (h < 90) & (s > 50) & (v > 55)
    elif name in ("grey_part", "white_part", "small_white_part", "plate", "saucer"):
        mask = (s < 100) & (v >= 125)
    else:
        mask = ((s > 65) & (v > 55)) | ((s < 100) & (v > 140))
    return mask.astype(np.uint8) * 255


def ground_object(image, item, candidates):
    """Use a compatible proposal, otherwise segment inside the object's coarse box."""
    height, width = image.shape[:2]
    name = item["name"]
    box = normalized_bbox_to_pixels(item["bbox"], width, height)
    x1, y1, x2, y2 = box
    if name in ("grey_tray", "black_tray", "hand"):
        return box, [(x1 + x2) // 2, (y1 + y2) // 2], "box_reference"
    compatible = []
    for proposal in candidates:
        px, py = proposal["pixel"]
        margin_x, margin_y = (x2 - x1) * 0.3, (y2 - y1) * 0.3
        if not (
            x1 - margin_x <= px < x2 + margin_x and y1 - margin_y <= py < y2 + margin_y
        ):
            continue
        overlap = bbox_iou(box, proposal["bbox"])
        pb = proposal["bbox"]
        size_ratio = ((pb[2] - pb[0]) * (pb[3] - pb[1])) / max(1, (x2 - x1) * (y2 - y1))
        if overlap < 0.06 or size_ratio > 1.8:
            continue
        # Check the actual proposed point's color against this object's category.
        color_mask = foreground_mask(
            image[max(0, py - 2) : py + 3, max(0, px - 2) : px + 3], name
        )
        if np.mean(color_mask > 0) < 0.5:
            continue
        compatible.append((overlap, proposal))
    if compatible:
        proposal = max(compatible, key=lambda pair: pair[0])[1]
        return proposal["bbox"], proposal["pixel"], "candidate:" + proposal["id"]
    crop = image[y1:y2, x1:x2]
    mask = foreground_mask(crop, name)
    # Remove tiny texture speckles without closing real holes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    padded = np.pad(mask, 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    # Prefer this object's middle when a coarse box also includes its neighbour.
    yy, xx = np.indices(mask.shape)
    centrality = np.exp(
        -2
        * (
            ((xx - mask.shape[1] / 2) / max(1, mask.shape[1])) ** 2
            + ((yy - mask.shape[0] / 2) / max(1, mask.shape[0])) ** 2
        )
    )
    score = distance * centrality
    _, best, _, pixel = cv2.minMaxLoc(score.astype(np.float32))
    if best < 2:
        raise ValueError(f'No solid foreground for {item["id"]}')
    return box, [x1 + pixel[0], y1 + pixel[1]], "local_foreground"


def ground_inventory(raw, image, candidates):
    """Normalize common model schemas and ground every visible object instance."""
    if isinstance(raw, list):
        raw = {"objects": raw, "relations": []}
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("objects"), list)
        or not raw["objects"]
    ):
        raise ValueError("Perception must return visible objects")
    objects = []
    ids = set()
    for source in raw["objects"]:
        item = dict(source)
        label = item.get("name", item.get("label"))
        if label in (
            "heart",
            "background_grid",
            "background",
            "sticker",
            "camera_hardware",
        ):
            continue
        oid = item.get("id", label)
        if not isinstance(oid, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", oid):
            raise ValueError("Missing or invalid object id")
        if oid in ids:
            suffix = 2
            while f"{oid}_{suffix}" in ids:
                suffix += 1
            oid = f"{oid}_{suffix}"
        item["id"] = oid
        ids.add(oid)
        item["bbox"] = item.get("bbox", item.get("bbox_2d"))
        name = ALIASES.get(label, label)
        if not isinstance(name, str) or not name:
            raise ValueError("Missing category")
        # Elongated metal bars are distinct from the broad white mechanical part.
        # This cue is category-level geometry, independent of scene/instruction IDs.
        if name in ("white_part", "grey_part"):
            geom = normalized_bbox_to_pixels(
                item["bbox"], image.shape[1], image.shape[0]
            )
            bw, bh = geom[2] - geom[0], geom[3] - geom[1]
            name = "grey_part" if max(bw / bh, bh / bw) >= 1.65 else "white_part"
        item["name"] = name
        box, pixel, method = ground_object(image, item, candidates)
        item.update(bbox=box, pixel=pixel, grounding=method, candidate_id=None)
        if name in ("grey_tray", "black_tray"):
            item["tray"] = name
        item.setdefault("tray", "unknown")
        if item.get("tray") not in ("grey_tray", "black_tray", "outside", "unknown"):
            raise ValueError("Missing tray membership")
        if "held_by_human" not in item:
            item["held_by_human"] = False
        if type(item["held_by_human"]) is not bool:
            raise ValueError("Invalid human contact flag")
        objects.append(item)
    by_id = {o["id"]: o for o in objects}
    relations = []
    for rel in raw.get("relations", []):
        top, bottom = rel.get("subject"), rel.get("object")
        if rel.get("relation") in (
            "left_of",
            "right_of",
            "above",
            "below",
            "next_to",
            "beside",
        ):
            continue  # These 2D relations are reconstructed from measured centers.
        if (
            top not in by_id
            or bottom not in by_id
            or top == bottom
            or rel.get("relation") != "on_top_of"
        ):
            raise ValueError("Invalid stacking relation")
        # Being on a tray/dish is support membership, not a blocking stack.
        if by_id[bottom]["name"] not in CONTAINERS:
            relations.append(
                {"subject": top, "relation": "on_top_of", "object": bottom}
            )
    # If a hand box overlaps the mug substantially, require human-contact handling.
    hands = [o for o in objects if o["name"] == "hand"]
    for obj in objects:
        if obj["name"] == "mug" and any(
            bbox_iou(obj["bbox"], h["bbox"]) > 0.05 for h in hands
        ):
            obj["held_by_human"] = True
    return {"objects": objects, "relations": relations}


def add_dark_tray_reference(inventory, image):
    """Recover the dark workcell support beside a detected grey tray, if visible."""
    objects = inventory["objects"]
    if any(o["name"] == "black_tray" for o in objects):
        return inventory
    grey = next((o for o in objects if o["name"] == "grey_tray"), None)
    if grey is None:
        return inventory
    h, w = image.shape[:2]
    _, top, left, bottom = grey["bbox"]
    if w - left < w * 0.2 or bottom - top < h * 0.4:
        return inventory
    region = cv2.cvtColor(image[top:bottom, left:w], cv2.COLOR_BGR2GRAY)
    # Require a broad genuinely dark visible supporting region, not empty metadata.
    if np.mean(region < 100) < 0.55:
        return inventory
    oid = "black_tray"
    if any(o["id"] == oid for o in objects):
        raise ValueError("black_tray id already used by another category")
    objects.append(
        {
            "id": oid,
            "name": "black_tray",
            "bbox": [left, top, w, bottom],
            "pixel": [(left + w) // 2, (top + bottom) // 2],
            "tray": "black_tray",
            "held_by_human": False,
            "candidate_id": None,
            "grounding": "dark_support_beside_grey_tray",
        }
    )
    return inventory


def planning_inventory(inventory):
    """Small measured scene graph, including explicit distance and ordering facts."""
    objects = []
    for obj in inventory["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        objects.append({k: obj[k] for k in ("id", "name", "tray", "held_by_human")})
        objects[-1]["center"] = [(x1 + x2) // 2, (y1 + y2) // 2]
    eligible = [
        o
        for o in objects
        if o["name"] not in CONTAINERS | {"hand"}
        and o["tray"] in ("grey_tray", "black_tray")
        and not o["held_by_human"]
    ]
    nearest = {}
    for ref in objects:
        if ref["name"] in ("grey_tray", "black_tray", "hand"):
            continue
        ranked = sorted(
            (math.dist(ref["center"], other["center"]), other["id"])
            for other in eligible
            if other["id"] != ref["id"]
        )
        nearest[ref["id"]] = [
            {"id": oid, "distance_px": round(distance)} for distance, oid in ranked
        ]
    return {
        "objects": objects,
        "eligible_pick_ids": [o["id"] for o in eligible if not o["held_by_human"]],
        "destination_ids": [o["id"] for o in objects if o["name"] in CONTAINERS],
        "relations": inventory["relations"],
        "blocked_by": {
            o["id"]: [
                r["subject"] for r in inventory["relations"] if r["object"] == o["id"]
            ]
            for o in eligible
        },
        "nearest_to_farthest": nearest,
        "left_to_right": [
            o["id"] for o in sorted(eligible, key=lambda o: o["center"][0])
        ],
        "nearest_left_of": {
            ref["id"]: [
                oid
                for _, oid in sorted(
                    (math.dist(ref["center"], other["center"]), other["id"])
                    for other in eligible
                    if other["center"][0] < ref["center"][0]
                )
            ]
            for ref in objects
        },
        "nearest_right_of": {
            ref["id"]: [
                oid
                for _, oid in sorted(
                    (math.dist(ref["center"], other["center"]), other["id"])
                    for other in eligible
                    if other["center"][0] > ref["center"][0]
                )
            ]
            for ref in objects
        },
    }


def reconcile_tray_membership(inventory):
    """Correct box-inconsistent support labels using detected tray rectangles."""
    trays = [
        o for o in inventory["objects"] if o["name"] in ("grey_tray", "black_tray")
    ]
    grey = next((o for o in trays if o["name"] == "grey_tray"), None)
    black = next((o for o in trays if o["name"] == "black_tray"), None)
    # A box around the entire workcell is not a tight black tray box.
    if grey and black and bbox_iou(grey["bbox"], black["bbox"]) > 0.35:
        if black["bbox"][0] < grey["bbox"][2]:
            black["bbox"][0] = grey["bbox"][2]
            x1, y1, x2, y2 = black["bbox"]
            if x1 >= x2:
                raise ValueError("Tray boxes overlap without a visible dark region")
            black["pixel"] = [(x1 + x2) // 2, (y1 + y2) // 2]
    for obj in inventory["objects"]:
        if obj["name"] in ("grey_tray", "black_tray", "hand"):
            continue
        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        supports = [
            t
            for t in trays
            if t["bbox"][0] <= cx < t["bbox"][2] and t["bbox"][1] <= cy < t["bbox"][3]
        ]
        if len(supports) == 1 and obj["tray"] != supports[0]["name"]:
            obj["perceived_tray"] = obj["tray"]
            obj["tray"] = supports[0]["name"]
    return inventory


def reconcile_visible_occlusion(inventory, image):
    """Resolve conflicting flat-gear support using color evidence in overlap regions.

    Only distinct-color gear/mug pairs with dominant visible foreground are used;
    arbitrary box overlap alone is never treated as physical stacking.
    """
    objects = inventory["objects"]
    for i, obj in enumerate(objects):
        for other in objects[i + 1 :]:
            names = {obj["name"], other["name"]}
            if names not in ({"mug", "red_gear"}, {"yellow_gear", "red_gear"}):
                continue
            if not all(
                o.get("grounding", "").startswith("candidate:") for o in (obj, other)
            ):
                continue
            a, b = obj["bbox"], other["bbox"]
            x1, y1, x2, y2 = (
                max(a[0], b[0]),
                max(a[1], b[1]),
                min(a[2], b[2]),
                min(a[3], b[3]),
            )
            if x1 >= x2 or y1 >= y2:
                continue
            fraction = (
                (x2 - x1)
                * (y2 - y1)
                / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            )
            if fraction < 0.2:
                continue
            crop = image[y1:y2, x1:x2]
            visible_a = int(np.sum(foreground_mask(crop, obj["name"]) > 0))
            visible_b = int(np.sum(foreground_mask(crop, other["name"]) > 0))
            if max(visible_a, visible_b) < crop.shape[0] * crop.shape[1] * 0.35:
                continue
            if max(visible_a, visible_b) / max(1, min(visible_a, visible_b)) < 2:
                continue
            upper, lower = (obj, other) if visible_a > visible_b else (other, obj)
            # A dominant foreground color is an occlusion cue, kept explicit in trace.
            inventory["relations"] = [
                r
                for r in inventory["relations"]
                if {r["subject"], r["object"]} != {obj["id"], other["id"]}
            ]
            inventory["relations"].append(
                {"subject": upper["id"], "relation": "on_top_of", "object": lower["id"]}
            )
            upper.setdefault("occlusion_evidence", []).append(lower["id"])
    return inventory
