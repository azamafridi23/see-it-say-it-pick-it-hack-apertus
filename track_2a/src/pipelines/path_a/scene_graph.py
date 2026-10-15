#!/usr/bin/env python3
"""Turn SAM 3 detections into a scene graph, the description Apertus uses.

    python pipelines/path_a/scene_graph.py dataset/scene_06   # one scene
    python pipelines/path_a/scene_graph.py dataset            # every scene

Writes pipelines/path_a/out/<scene>_graph.json and out/<scene>_graph.jpg
(the graph drawn on the photo).

Steps: detect (run_sam3.detect, cached) -> drop what is off the table ->
merge duplicates and fragments -> add what nobody named -> name the
objects -> measure them -> work out what lies on what.

Objects come from three sources, recorded as found_by:
  prompt    the fixed prompt list in run_sam3.py
  apertus   a name Apertus gave when asked to list what it sees in the
            photo (apertus_list.py)
  leftover  a hole in the tray's or the mat's outline that no known object
            explains
Uses only the photo and calibration.json: never the ground truth.

Two kinds of stacking relation come out:
  on        A covers a clear share of where B should be, and depth does
            not contradict it. Reliable.
  maybe_on  A touches B and stands higher (monocular depth), but B's
            hidden part cannot be measured; or the overlap says A is on B
            while depth says B is higher. A may only be standing next to
            B: a later step has to decide.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from apertus_list import names_for
from run_sam3 import CACHE, DEVICE, detect, draw, interior_pixel, photo_key
from workcell import CELL

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "starter"))
from scene_io import load_scene, pixel_to_table  # noqa: E402

# Not objects on the table: left out of stacking and distances.
SURFACES_AND_HANDS = ("grey tray", "hand")

_depth = {}


# ---------------------------------------------------------- mask helpers

def tier(det: dict) -> int:
    """Rank a detection's label.

    Args:
        det: A detection.

    Returns:
        1 for a named object, 2 for a vague prompt, 3 for unnamed: the
        "object" catch-all or a name Apertus made up.
    """
    if det.get("extra"):
        return 3

    return CELL.label_tiers.get(det["label"], 1)


def slug(label: str) -> str:
    """Turn a label into an id: lower case, words joined by ``_``."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow a mask by ``px`` pixels in every direction."""
    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def hull(mask: np.ndarray) -> np.ndarray:
    """Fill in a mask's outline.

    Roughly where the object is, even where something covers it.

    Args:
        mask: A bool mask.

    Returns:
        The mask's convex hull, as a bool mask.
    """
    points = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(points), 1)
    return out.astype(bool)


def circle(mask: np.ndarray) -> np.ndarray:
    """Fill the smallest circle around a mask.

    The full outline of a gear that is partly hidden.

    Args:
        mask: A bool mask.

    Returns:
        The circle, as a bool mask.
    """
    points = cv2.findNonZero(mask.astype(np.uint8))
    (x, y), radius = cv2.minEnclosingCircle(points)
    centre_px = (int(round(x)), int(round(y)))
    out = np.zeros(mask.shape, np.uint8)
    cv2.circle(out, centre_px, int(round(radius)), 1, -1)
    return out.astype(bool)


def centre(det: dict) -> np.ndarray:
    """Return the centre of a detection's box."""
    x0, y0, x1, y1 = det["box_xyxy"]
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


# ------------------------------------------------- photo measurements

def black_mat(rgb: np.ndarray) -> np.ndarray | None:
    """Find the black mat (the "black tray") as a box.

    From the largest dark region to the right edge of the photo, and down
    to the bright rail below the table. The mat is lit unevenly, grey
    towards the bottom, and objects can cut it in pieces, so darkness
    alone does not outline it.

    Args:
        rgb: The photo, RGB.

    Returns:
        The mat as a bool mask, or None if no large dark region is found.
    """
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (grey < CELL.mat_max_brightness).astype(np.uint8)
    kernel = np.ones((CELL.mat_open_px, CELL.mat_open_px), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    if count < 2:
        return None

    largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    x, y, _, height, area = stats[largest]
    if area < CELL.mat_min_share * dark.size:
        return None

    below = grey[y + height:, x:]
    rail_rows = np.flatnonzero(np.median(below, axis=1) > CELL.rail_min_grey)
    if len(rail_rows):
        bottom = y + height + int(rail_rows[0])
    else:
        bottom = y + height + (grey.shape[0] - y - height)

    out = np.zeros(grey.shape, bool)
    out[y:bottom, x:] = True
    return out


def relative_depth(scene_dir: Path, rgb: np.ndarray) -> np.ndarray | None:
    """Estimate monocular depth, larger = closer to the camera.

    Only good for comparing neighbours. Cached.

    Args:
        scene_dir: The scene folder, for the cache name.
        rgb: The photo, RGB.

    Returns:
        The depth map at the photo's size, or None if the depth model is
        unavailable.
    """
    key = photo_key(scene_dir / "rgb.png")
    path = CACHE / f"{scene_dir.name}_{key}_depth.npy"
    if path.exists():
        return np.load(path)

    try:
        from PIL import Image
        from transformers import pipeline
        if "pipe" not in _depth:
            _depth["pipe"] = pipeline("depth-estimation",
                                      model=CELL.depth_model, device=DEVICE)

        predicted = _depth["pipe"](Image.fromarray(rgb))["predicted_depth"]
        depth = predicted.squeeze().cpu().numpy().astype(np.float32)
    # No depth model: the graph is still built, without maybe_on.
    except Exception as error:
        print(f"  no depth for {scene_dir.name}: {error}")
        return None

    size = (rgb.shape[1], rgb.shape[0])
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, depth)
    return depth


def colour_name(rgb: np.ndarray, mask: np.ndarray) -> str:
    """Name the median colour under a mask (white, grey, red, ...)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.median(hsv[mask], axis=0)
    if saturation < CELL.pale_max_saturation:
        if value > CELL.white_min_value:
            return "white"

        return "black" if value < CELL.black_max_value else "grey"

    for name, top in CELL.hues:
        if hue <= top:
            return name

    return "red"


def size_mm(scene, box) -> list[int]:
    """Measure a box on the table plane.

    Too large for tall objects away from the image centre.

    Args:
        scene: The scene, for the calibration.
        box: The box, [x0, y0, x1, y1] in pixels.

    Returns:
        [long side, short side] in mm.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, right = pixel_to_table(scene, x0, cy), pixel_to_table(scene, x1, cy)
    top, bottom = pixel_to_table(scene, cx, y0), pixel_to_table(scene, cx, y1)
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(bottom - top)
    return sorted((int(round(width)), int(round(height))), reverse=True)


def mm_per_pixel(scene) -> float:
    """Measure the table's scale at the photo centre, over 100 pixels."""
    height, width = scene.rgb.shape[:2]
    cx, cy = width / 2, height / 2
    right = pixel_to_table(scene, cx + 50, cy)
    left = pixel_to_table(scene, cx - 50, cy)
    return float(np.linalg.norm(right - left)) / 100


# ------------------------------------------------ cleaning detections

def merge(dets: list[dict]) -> list[dict]:
    """Keep one detection per object.

    Named objects first, then vague prompts, then Apertus's names, then
    "object"; best score first within each.

    Args:
        dets: The detections, each with its ``area``.

    Returns:
        The kept detections.
    """
    def priority(det):
        return tier(det), det["label"] == "object", -det["score"]

    kept: list[dict] = []
    for det in sorted(dets, key=priority):
        mask, area = det["mask"], det["area"]
        overlaps = [int((mask & other["mask"]).sum()) for other in kept]
        pairs = list(zip(overlaps, kept))

        # The same object under another prompt.
        if any(overlap / (area + other["area"] - overlap) >= CELL.same_iou
               for overlap, other in pairs):
            continue

        # Kept masks lying inside this one, and kept masks this one lies
        # inside.
        inside = [other for overlap, other in pairs
                  if overlap / other["area"] >= CELL.inside]
        around = [other for overlap, other in pairs
                  if overlap / area >= CELL.inside]

        if tier(det) == 3:
            kept = _place_unnamed(det, kept, inside, around)
            continue

        if _is_fragment(det, around) or _is_plate_middle(det, around):
            continue

        # Replace the kept pieces of this object with the whole.
        kept = [other for other in kept
                if not (other["label"] == det["label"]
                        and any(other is piece for piece in inside))]
        kept.append(det)

    return kept


def _place_unnamed(det, kept, inside, around):
    """Decide what an unnamed detection adds to the kept ones."""
    labels_inside = {piece["label"] for piece in inside}
    # A whole object whose pieces a vague prompt found: take their label
    # and replace them.
    if (inside and all(tier(piece) == 2 for piece in inside)
            and len(labels_inside) == 1):
        whole = {**det, "label": inside[0]["label"], "extra": False}
        kept = [other for other in kept
                if not any(other is piece for piece in inside)]
        kept.append(whole)
        return kept

    # Something no other prompt found. Otherwise it is several objects
    # merged, or a part of one: dropped.
    lies_in_a_thing = any(other["label"] not in CELL.containers
                          for other in around)
    if not inside and not lies_in_a_thing:
        kept.append(det)

    return kept


def _is_fragment(det, around):
    """Tell whether a detection is a fragment of an object already kept."""
    for other in around:
        if other["label"] == det["label"]:
            return True

        if other["label"] not in CELL.containers and tier(other) < tier(det):
            return True

    return False


def _is_plate_middle(det, around):
    """Tell whether a vague detection is the white middle of a plate."""
    if tier(det) != 2:
        return False

    for other in around:
        if other["label"] != "plate":
            continue

        plate_width = other["box_xyxy"][2] - other["box_xyxy"][0]
        distance = np.linalg.norm(centre(det) - centre(other))
        if distance < CELL.plate_middle_share * plate_width:
            return True

    return False


def leftovers(surfaces: list[np.ndarray], objs: list[dict]) -> list[dict]:
    """Find objects nobody named.

    SAM 3 outlines the tray and the mat with holes where things lie on
    them: a hole that no known object explains is something on the table
    that no prompt found.

    Args:
        surfaces: The masks of the tray and the mat, as SAM 3 saw them.
        objs: The objects found so far.

    Returns:
        One "object" detection per unexplained hole.
    """
    area = np.zeros(surfaces[0].shape, bool)
    seen = np.zeros(surfaces[0].shape, bool)
    for surface in surfaces:
        area |= hull(surface)
        seen |= surface

    gap = area & ~seen
    for obj in objs:
        gap &= ~grow(obj["mask"], CELL.leftover_margin_px)

    kernel = np.ones((CELL.leftover_open_px, CELL.leftover_open_px), np.uint8)
    gap = cv2.morphologyEx(gap.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(gap)

    out = []
    for index in range(1, count):
        x, y, width, height, area_px = stats[index]
        if area_px < CELL.leftover_min_area_px:
            continue

        box = [int(x), int(y), int(x + width), int(y + height)]
        out.append({"label": "object", "score": 0.0, "box_xyxy": box,
                    "mask": labels == index, "area": int(area_px),
                    "found_by": "leftover"})

    return out


# ------------------------------------------------- building the graph

def _label_detections(dets, extra):
    """Tag every detection with its source, its label and its area."""
    labelled = [{**det, "label": CELL.labels.get(det["label"], det["label"]),
                 "found_by": "prompt"} for det in dets]
    labelled += [{**det, "extra": True, "found_by": "apertus"}
                 for det in extra]
    for det in labelled:
        det["area"] = int(det["mask"].sum())

    return labelled


def _best_mask(dets):
    """Return the best-scoring detection's mask, or None."""
    if not dets:
        return None

    return max(dets, key=lambda det: det["score"])["mask"]


def _find_table(dets, rgb):
    """Find the two surfaces: the grey tray and the black mat.

    Returns:
        The tray as SAM 3 saw it and filled in, and the same for the mat
        (the filled-in mat may come from the photo alone).
    """
    trays = [det for det in dets
             if det["label"] == "grey tray" and not det.get("extra")]
    tray_seen = _best_mask(trays)
    tray = hull(tray_seen) if trays else None

    # SAM 3 also returns the whole table as a "black mat": take the best
    # one that is not the tray's side.
    def is_mat(det):
        if det["label"] != "black mat" or det.get("extra"):
            return False

        if tray is None:
            return True

        overlap = (det["mask"] & tray).sum() / det["area"]
        return overlap < CELL.mat_max_tray_overlap

    mats = [det for det in dets if is_mat(det)]
    mat_seen = _best_mask(mats)
    mat = hull(mat_seen) if mats else black_mat(rgb)
    return tray_seen, tray, mat_seen, mat


def _on_table(dets, tray, mat):
    """Drop what is off the table (the rails, the robot) and specks."""
    if tray is None and mat is None:
        table = None
    elif mat is None:
        table = tray
    elif tray is None:
        table = mat
    else:
        table = tray | mat

    def on_table(det):
        share = (det["mask"] & table).sum() / det["area"]
        return det["label"] == "hand" or share >= CELL.min_on_table

    if table is not None:
        dets = [det for det in dets if on_table(det)]

    return [det for det in dets if det["area"] >= CELL.min_area_px]


def _add_leftovers(objs, tray_seen, mat_seen):
    """Add what nobody named; a patch touching a hand is the arm."""
    surfaces = [mask for mask in (tray_seen, mat_seen) if mask is not None]
    if not surfaces:
        return

    for gap in leftovers(surfaces, objs):
        arm = None
        for obj in objs:
            near_hand = grow(obj["mask"], CELL.hand_reach_px)
            if obj["label"] == "hand" and (near_hand & gap["mask"]).any():
                arm = obj
                break

        if arm is None:
            objs.append(gap)
        else:
            arm["mask"] = arm["mask"] | gap["mask"]
            arm["area"] = int(arm["mask"].sum())


def _measure(objs, scene):
    """Give each object its pixel, size and outline; spot the saucer."""
    for obj in objs:
        obj["pixel"] = interior_pixel(obj["mask"])
        obj["size_mm"] = size_mm(scene, obj["box_xyxy"])
        if obj["label"] == "plate" and obj["size_mm"][0] < CELL.saucer_max_mm:
            obj["label"] = "saucer"

        if obj["label"] in CELL.round_kinds:
            obj["footprint"] = circle(obj["mask"])
        else:
            obj["footprint"] = hull(obj["mask"])

        obj["on"], obj["maybe_on"], obj["covered_by"] = [], [], []


def _assign_ids(objs):
    """Number objects of one kind from left to right: red_gear_1, ...

    An object that is the only one of its kind gets no number.
    """
    for label in {obj["label"] for obj in objs}:
        same = [obj for obj in objs if obj["label"] == label]
        same.sort(key=lambda obj: obj["pixel"][0])
        base = CELL.ids.get(label, slug(label))
        for number, obj in enumerate(same, 1):
            obj["id"] = base if len(same) == 1 else f"{base}_{number}"


def _place(objs, tray, mat, rgb_shape):
    """Record which surface each object is in and whether a hand touches it.

    Returns:
        The hands found.
    """
    hands = [obj for obj in objs if obj["label"] == "hand"]
    reach = np.zeros(rgb_shape[:2], bool)
    for hand in hands:
        reach |= grow(hand["mask"], CELL.hand_reach_px)

    for obj in objs:
        u, v = obj["pixel"]
        if obj["label"] == "grey tray":
            obj["in"] = None
        elif tray is not None and tray[v, u]:
            obj["in"] = "grey_tray"
        elif mat is not None and mat[v, u]:
            obj["in"] = "black_tray"
        else:
            obj["in"] = None

        touched = int((obj["mask"] & reach).sum()) > CELL.hand_min_overlap_px
        obj["touching_hand"] = (bool(hands)
                                and obj["label"] not in SURFACES_AND_HANDS
                                and touched)

    return hands


def _height_gap_finder(scene_dir, rgb):
    """Make a function that measures how two touching objects meet.

    The depth map is computed on first use only.
    """
    depth = []

    def height_gap(upper, lower):
        """Measure the contact between two objects.

        Returns:
            The contact length in px and how much higher ``upper`` stands
            than ``lower`` beside the contact, or None without contact or
            depth.
        """
        upper_edge = grow(upper["mask"], CELL.touch_px)
        lower_edge = grow(lower["mask"], CELL.touch_px)
        zone = upper_edge & lower_edge
        if not zone.any():
            return None

        if not depth:
            depth.append(relative_depth(scene_dir, rgb))

        if depth[0] is None:
            return None

        near = grow(zone, CELL.depth_window_px)
        upper_height = np.median(depth[0][upper["mask"] & near])
        lower_height = np.median(depth[0][lower["mask"] & near])
        contact_px = zone.sum() / (2 * CELL.touch_px + 1)
        return contact_px, float(upper_height - lower_height)

    return height_gap


def _touch(one, two):
    """Tell whether two objects touch or overlap at all."""
    one_reaches = (grow(one["mask"], CELL.touch_px) & two["footprint"]).any()
    two_reaches = (grow(two["mask"], CELL.touch_px) & one["footprint"]).any()
    return one_reaches or two_reaches


def _cover(seen, beneath):
    """Share of the smaller of two objects that one fills inside the other.

    A small object on a large one counts as fully as a large one across a
    small one.
    """
    filled = (seen["mask"] & beneath["footprint"]).sum()
    return filled / min(seen["area"], beneath["footprint"].sum())


def _stacking(things, height_gap):
    """Work out what lies on what, between every two objects.

    A lies on B when A covers part of where B should be. Seen from above,
    the upper object is the one seen whole: the lower one cannot be seen
    where the upper one sits.
    """
    for index, one in enumerate(things):
        for two in things[index + 1:]:
            if not _touch(one, two):
                continue

            cover = {one["id"]: _cover(one, two), two["id"]: _cover(two, one)}
            if cover[one["id"]] >= cover[two["id"]]:
                top, low = one, two
            else:
                top, low = two, one

            if top["label"] in CELL.containers:
                continue

            if low["label"] in CELL.containers:
                if cover[top["id"]] >= CELL.inside:
                    top["on"].append(low["id"])

                continue

            if cover[top["id"]] >= CELL.on_min_cover:
                gap = height_gap(top, low)
                # Depth says the "lower" one is higher: not sure.
                if gap is not None and gap[1] <= -CELL.maybe_min_higher:
                    top["maybe_on"].append(low["id"])
                else:
                    top["on"].append(low["id"])
                    low["covered_by"].append(top["id"])

                continue

            # Touching, with no measurable overlap: the higher one may be
            # resting on the other.
            gap = height_gap(one, two)
            if gap is None:
                continue

            contact_px, one_higher_by = gap
            if (contact_px >= CELL.maybe_min_contact_px
                    and abs(one_higher_by) >= CELL.maybe_min_higher):
                if one_higher_by > 0:
                    one["maybe_on"].append(two["id"])
                else:
                    two["maybe_on"].append(one["id"])


def _nearest_gaps(objs, things, scene):
    """Measure the gap to every other object, in mm, nearest first.

    So that "closest to" and "next to" are read off, not calculated by the
    language model.
    """
    mm_per_px = mm_per_pixel(scene)
    away = {}
    for thing in things:
        outside = (~thing["mask"]).astype(np.uint8)
        away[thing["id"]] = cv2.distanceTransform(outside, cv2.DIST_L2, 3)

    for obj in objs:
        if obj["id"] not in away:
            obj["nearest"] = []
            continue

        gaps = []
        for other in things:
            if other is obj:
                continue

            gap_px = float(away[other["id"]][obj["mask"]].min())
            gaps.append([other["id"], int(round(gap_px * mm_per_px))])

        obj["nearest"] = sorted(gaps, key=lambda gap: gap[1])


def _graph_entry(obj, rgb):
    """Describe one object for the scene graph."""
    return {
        "id": obj["id"],
        "label": obj["label"],
        "colour": colour_name(rgb, obj["mask"]),
        "pixel": obj["pixel"],
        "box_xyxy": obj["box_xyxy"],
        "size_mm": obj["size_mm"],
        "pickable": obj["label"] not in CELL.not_pickable,
        "why_not_pickable": CELL.not_pickable.get(obj["label"]),
        "in": obj["in"],
        "on": obj["on"],
        "maybe_on": obj["maybe_on"],
        "covered_by": obj["covered_by"],
        "touching_hand": obj["touching_hand"],
        "nearest_gap_mm": obj["nearest"],
        "found_by": obj["found_by"],
        "score": round(obj["score"], 2),
    }


BLACK_TRAY_ENTRY = {
    "id": "black_tray",
    "label": "black tray",
    "pickable": False,
    "why_not_pickable": CELL.not_pickable["black tray"],
    "note": "the black mat on the right; a place to put things",
}


def build(scene_dir: Path, dets: list[dict], extra: list[dict] = (),
          leftover: bool = True) -> tuple[dict, list[dict]]:
    """Build the scene graph.

    Args:
        scene_dir: The scene folder.
        dets: SAM 3's detections for the fixed prompts.
        extra: The detections for the names Apertus gave.
        leftover: Whether to also look for objects nobody named.

    Returns:
        The scene graph as a JSON-ready dict, and the objects with their
        masks (for drawing and checking).
    """
    scene = load_scene(scene_dir)
    dets = _label_detections(dets, extra)

    # The table is the grey tray and the black mat; detections elsewhere
    # are the rails and the robot.
    tray_seen, tray, mat_seen, mat = _find_table(dets, scene.rgb)
    dets = [det for det in dets if det["label"] != "black mat"]
    dets = _on_table(dets, tray, mat)

    objs = merge(dets)
    if leftover:
        _add_leftovers(objs, tray_seen, mat_seen)

    _measure(objs, scene)
    _assign_ids(objs)
    hands = _place(objs, tray, mat, scene.rgb.shape)

    things = [obj for obj in objs
              if obj["label"] not in SURFACES_AND_HANDS]
    _stacking(things, _height_gap_finder(scene_dir, scene.rgb))
    _nearest_gaps(objs, things, scene)

    objs.sort(key=lambda obj: (obj["label"] in CELL.not_pickable, obj["id"]))
    left_to_right = sorted(things, key=lambda obj: obj["pixel"][0])
    top_to_bottom = sorted(things, key=lambda obj: obj["pixel"][1])
    height, width = scene.rgb.shape[:2]
    graph = {
        "scene": scene.name,
        "image_size": [width, height],
        "gripper_max_opening_mm": scene.gripper_max_opening_mm,
        "hand_present": bool(hands),
        "left_to_right": [obj["id"] for obj in left_to_right],
        "top_to_bottom": [obj["id"] for obj in top_to_bottom],
        "objects": [_graph_entry(obj, scene.rgb) for obj in objs],
    }
    if mat is not None:
        graph["objects"].append(dict(BLACK_TRAY_ENTRY))

    return graph, objs


# ------------------------------------------------------- output

def render(scene_dir: Path, objs: list[dict], path: Path) -> None:
    """Draw the objects and their relations on the photo and save it.

    Args:
        scene_dir: The scene folder.
        objs: The objects, as returned by ``build()``.
        path: Where to write the image.
    """
    canvas = cv2.imread(str(scene_dir / "rgb.png"), cv2.IMREAD_COLOR)
    rng = np.random.default_rng(0)
    for obj in objs:
        text = obj["id"]
        if obj["on"]:
            text += f" on {','.join(obj['on'])}"

        if obj["maybe_on"]:
            text += f" on? {','.join(obj['maybe_on'])}"

        if obj["touching_hand"]:
            text += " HAND"

        colour = rng.integers(60, 256, 3).tolist()
        draw(canvas, obj["mask"], obj["pixel"], text, colour)

    cv2.imwrite(str(path), canvas)


def scene_dirs(path: Path) -> list[Path]:
    """List the scene folders: ``path`` itself, or the scenes inside it."""
    if (path / "rgb.png").exists():
        return [path]

    return sorted(sub for sub in path.iterdir()
                  if (sub / "rgb.png").exists())


def _summary(graph):
    """Describe a graph in one line: the ids and what lies on what."""
    parts = []
    for entry in graph["objects"]:
        text = entry["id"]
        if entry.get("on"):
            text += f" (on {'+'.join(entry['on'])})"

        if entry.get("maybe_on"):
            text += f" (on? {'+'.join(entry['maybe_on'])})"

        parts.append(text)

    return f"{graph['scene']}: " + ", ".join(parts)


def main():
    """Build and save the scene graph of one scene or every scene."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path,
                    help="a scene folder, or a folder of scenes")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--no-apertus", action="store_true",
                    help="do not ask Apertus to name the objects it sees")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    for scene in scene_dirs(args.data):
        extra = [] if args.no_apertus else detect(scene, names_for(scene))
        graph, objs = build(scene, detect(scene), extra=extra)
        graph_json = json.dumps(graph, indent=2)
        (args.out / f"{scene.name}_graph.json").write_text(graph_json)
        render(scene, objs, args.out / f"{scene.name}_graph.jpg")
        print(_summary(graph))


if __name__ == "__main__":
    main()
