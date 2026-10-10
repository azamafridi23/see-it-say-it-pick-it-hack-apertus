#!/usr/bin/env python3
"""Turn SAM 3 detections into a scene graph: the JSON description of the photo that Apertus reasons over.

    python pipelines/path_a/scene_graph.py dataset/scene_06     one scene
    python pipelines/path_a/scene_graph.py dataset              every scene in the folder

Writes pipelines/path_a/out/<scene>_graph.json and out/<scene>_graph.jpg (the graph drawn on the photo).

Steps: detect (run_sam3.detect, cached) -> drop what is off the table -> merge duplicates and
fragments -> add what nobody named -> name the objects -> measure them -> work out what lies on what.

Objects come from three sources, recorded as found_by:
  prompt    the fixed prompt list in run_sam3.py
  apertus   a name Apertus gave when asked to list what it sees in the photo (apertus_list.py)
  leftover  a hole in the tray's or the mat's outline that no known object explains
Uses only the photo and calibration.json: never the ground truth.

Two kinds of stacking relation come out:
  on        A covers a clear share of where B should be, and depth does not contradict it. Reliable.
  maybe_on  A touches B and stands higher (monocular depth), but B's hidden part cannot be measured;
            or the overlap says A is on B while depth says B is higher.
            A may only be standing next to B: a later step has to decide.
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import cv2
import numpy as np

from apertus_list import names_for
from run_sam3 import CACHE, DEVICE, detect, draw, interior_pixel, photo_key

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "starter"))
from scene_io import load_scene, pixel_to_table  # noqa: E402

CANON = {"orange fruit": "mandarin", "small yellow gear": "yellow gear", "saucer": "plate"}  # prompt -> label
# Which label wins when two prompts find the same thing: a named object beats a vague prompt beats "object".
TIER = {"white part": 2, "object": 3}
CONTAINERS = {"grey tray", "plate", "saucer"}          # things lie in or on these; they are never fragments of them
# What cannot be picked, and why. The organisers confirmed that this is a property of the object, not a size rule:
# the gripper closes on one narrow part (a gear's hub, a mug's rim, a part's edge), and only the plate and the
# saucer have none. Every other object can be picked, including ones nobody named.
FLAT = "it lies flat and has no narrow part for the gripper to close on; things can still be put on it"
TRAY = "a tray is never picked up; things are put in it"
NOT_PICKABLE = {"plate": FLAT, "saucer": FLAT, "grey tray": TRAY, "black tray": TRAY, "hand": "it is a person's hand"}
ROUND = {"red gear", "yellow gear"}                    # a circle recovers their outline when part is hidden
NAMES = {"metal bracket": "grey_part", "object": "unknown_object"}

MIN_AREA_PX = 2000        # smaller masks are specks
MIN_ON_TABLE = 0.5        # share of a mask that must lie on the tray or the mat
SAME_IOU = 0.7            # two masks are the same object
INSIDE = 0.6              # share of a mask inside another to count as part of it
LEFT_MARGIN_PX = 12       # ignore gaps this close to a known object: its edge and shadow
LEFT_OPEN_PX = 25         # ignore gaps narrower than this
LEFT_MIN_AREA_PX = 4000   # an unexplained gap this large is an object nobody named
SAUCER_MAX_MM = 155       # a plate narrower than this is the saucer
ON_MIN_COVER = 0.12       # share of the lower object's footprint the upper one must cover
RAIL_MIN_GREY = 170       # the aluminium rails around the table are brighter than this
TOUCH_PX = 10             # masks closer than twice this are touching
MAYBE_MIN_CONTACT_PX = 100
MAYBE_MIN_HIGHER = 0.3    # in the depth model's relative units
HAND_REACH_PX = 25
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

_depth = {}


def tier(d: dict) -> int:
    """1 a named object, 2 a vague prompt, 3 unnamed: the "object" catch-all or a name Apertus made up."""
    return 3 if d.get("extra") else TIER.get(d["label"], 1)


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1,) * 2, np.uint8)).astype(bool)


def hull(mask: np.ndarray) -> np.ndarray:
    """The mask with its outline filled in: roughly where the object is, even where something covers it."""
    out = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(cv2.findNonZero(mask.astype(np.uint8))), 1)
    return out.astype(bool)


def circle(mask: np.ndarray) -> np.ndarray:
    """The smallest circle around the mask: the full outline of a gear that is partly hidden."""
    (x, y), r = cv2.minEnclosingCircle(cv2.findNonZero(mask.astype(np.uint8)))
    out = np.zeros(mask.shape, np.uint8)
    cv2.circle(out, (int(round(x)), int(round(y))), int(round(r)), 1, -1)
    return out.astype(bool)


def black_mat(rgb: np.ndarray) -> np.ndarray | None:
    """The black mat (the "black tray") as a box: from the largest dark region to the right edge of the photo,
    and down to the bright rail below the table. The mat is lit unevenly, grey towards the bottom, and objects
    can cut it in pieces, so darkness alone does not outline it."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = cv2.morphologyEx((grey < 80).astype(np.uint8), cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n < 2:
        return None
    x, y, _, h, area = stats[1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())]
    if area < 0.05 * dark.size:
        return None
    rail = np.flatnonzero(np.median(grey[y + h:, x:], axis=1) > RAIL_MIN_GREY)
    bottom = y + h + (int(rail[0]) if len(rail) else grey.shape[0] - y - h)
    out = np.zeros(grey.shape, bool)
    out[y:bottom, x:] = True
    return out


def relative_depth(scene_dir: Path, rgb: np.ndarray) -> np.ndarray | None:
    """Monocular depth, larger = closer to the camera. Only good for comparing neighbours. Cached; None if unavailable."""
    path = CACHE / f"{scene_dir.name}_{photo_key(scene_dir / 'rgb.png')}_depth.npy"
    if path.exists():
        return np.load(path)
    try:
        from PIL import Image
        from transformers import pipeline
        if "pipe" not in _depth:
            _depth["pipe"] = pipeline("depth-estimation", model=DEPTH_MODEL, device=DEVICE)
        d = _depth["pipe"](Image.fromarray(rgb))["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
    except Exception as e:  # no depth model: the graph is still built, without maybe_on
        print(f"  no depth for {scene_dir.name}: {e}")
        return None
    d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, d)
    return d


def colour_name(rgb: np.ndarray, mask: np.ndarray) -> str:
    h, s, v = np.median(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask], axis=0)
    if s < 60:
        return "white" if v > 170 else "black" if v < 70 else "grey"
    for name, top in (("red", 8), ("orange", 20), ("yellow", 34), ("green", 85), ("blue", 130), ("purple", 160)):
        if h <= top:
            return name
    return "red"


def size_mm(scene, box) -> list[int]:
    """[long side, short side] of the box on the table plane, mm. Too large for tall objects away from the image centre."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = np.linalg.norm(pixel_to_table(scene, x1, cy) - pixel_to_table(scene, x0, cy))
    h = np.linalg.norm(pixel_to_table(scene, cx, y1) - pixel_to_table(scene, cx, y0))
    return sorted((int(round(w)), int(round(h))), reverse=True)


def centre(d: dict) -> np.ndarray:
    x0, y0, x1, y1 = d["box_xyxy"]
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def merge(dets: list[dict]) -> list[dict]:
    """One detection per object. Named objects first, then vague prompts, then Apertus's names, then "object";
    best score first within each."""
    kept: list[dict] = []
    for d in sorted(dets, key=lambda d: (tier(d), d["label"] == "object", -d["score"])):
        m, area = d["mask"], d["area"]
        inter = [int((m & k["mask"]).sum()) for k in kept]
        if any(i / (area + k["area"] - i) >= SAME_IOU for i, k in zip(inter, kept)):
            continue                                                    # the same object under another prompt
        inside = [k for i, k in zip(inter, kept) if i / k["area"] >= INSIDE]   # kept masks lying inside this one
        around = [k for i, k in zip(inter, kept) if i / area >= INSIDE]        # kept masks this one lies inside
        if tier(d) == 3:
            # whole object whose pieces a vague prompt found: take their label and replace them
            if inside and all(tier(k) == 2 for k in inside) and len({k["label"] for k in inside}) == 1:
                d = {**d, "label": inside[0]["label"], "extra": False}
                kept = [k for k in kept if not any(k is x for x in inside)]
                kept.append(d)
            elif not inside and not any(k["label"] not in CONTAINERS for k in around):
                kept.append(d)                                          # something no other prompt found
            continue                                                    # else: several objects merged, or a part of one
        if any(k["label"] == d["label"] or (k["label"] not in CONTAINERS and tier(k) < tier(d))
               for k in around):
            continue                                                    # a fragment of an object already kept
        if tier(d) == 2 and any(k["label"] == "plate" and np.linalg.norm(centre(d) - centre(k))
                                         < 0.1 * (k["box_xyxy"][2] - k["box_xyxy"][0]) for k in around):
            continue                                                    # the white middle of a plate or saucer
        kept = [k for k in kept if not (k["label"] == d["label"] and any(k is x for x in inside))]
        kept.append(d)
    return kept


def leftovers(surfaces: list[np.ndarray], objs: list[dict]) -> list[dict]:
    """Objects nobody named. SAM 3 outlines the tray and the mat with holes where things lie on them:
    a hole that no known object explains is something on the table that no prompt found."""
    area = np.zeros(surfaces[0].shape, bool)
    seen = np.zeros(surfaces[0].shape, bool)
    for m in surfaces:
        area |= hull(m)
        seen |= m
    gap = area & ~seen
    for o in objs:
        gap &= ~grow(o["mask"], LEFT_MARGIN_PX)
    gap = cv2.morphologyEx(gap.astype(np.uint8), cv2.MORPH_OPEN, np.ones((LEFT_OPEN_PX,) * 2, np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gap)
    out = []
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if a >= LEFT_MIN_AREA_PX:
            out.append({"label": "object", "score": 0.0, "box_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                        "mask": labels == k, "area": int(a), "found_by": "leftover"})
    return out


def build(scene_dir: Path, dets: list[dict], extra: list[dict] = (), leftover: bool = True) -> tuple[dict, list[dict]]:
    """The scene graph as a JSON-ready dict, and the objects with their masks (for drawing and checking).
    extra = detections for the names Apertus gave; leftover = also look for objects nobody named."""
    scene = load_scene(scene_dir)
    dets = [{**d, "label": CANON.get(d["label"], d["label"]), "found_by": "prompt"} for d in dets] \
        + [{**d, "extra": True, "found_by": "apertus"} for d in extra]
    for d in dets:
        d["area"] = int(d["mask"].sum())

    # the table: the grey tray and the black mat. Detections elsewhere are the rails and the robot.
    trays = [d for d in dets if d["label"] == "grey tray" and not d.get("extra")]
    tray_seen = max(trays, key=lambda d: d["score"])["mask"] if trays else None
    tray = hull(tray_seen) if trays else None
    # SAM 3 also returns the whole table as a "black mat": take the best one that is not the tray's side
    mats = [d for d in dets if d["label"] == "black mat" and not d.get("extra")
            and (tray is None or (d["mask"] & tray).sum() / d["area"] < 0.2)]
    mat_seen = max(mats, key=lambda d: d["score"])["mask"] if mats else None
    mat = hull(mat_seen) if mats else black_mat(scene.rgb)
    dets = [d for d in dets if d["label"] != "black mat"]
    table = None if tray is None and mat is None else (tray if mat is None else mat if tray is None else tray | mat)
    if table is not None:
        dets = [d for d in dets if d["label"] == "hand" or (d["mask"] & table).sum() / d["area"] >= MIN_ON_TABLE]
    dets = [d for d in dets if d["area"] >= MIN_AREA_PX]

    objs = merge(dets)
    if leftover:
        surfaces = [m for m in (tray_seen, mat_seen) if m is not None]
        for gap in leftovers(surfaces, objs) if surfaces else []:
            # an unexplained patch touching a hand is the rest of the arm: it belongs to the hand
            arm = next((o for o in objs if o["label"] == "hand" and (grow(o["mask"], HAND_REACH_PX) & gap["mask"]).any()), None)
            if arm is None:
                objs.append(gap)
            else:
                arm["mask"] = arm["mask"] | gap["mask"]
                arm["area"] = int(arm["mask"].sum())
    for o in objs:
        o["pixel"] = interior_pixel(o["mask"])
        o["size_mm"] = size_mm(scene, o["box_xyxy"])
        if o["label"] == "plate" and o["size_mm"][0] < SAUCER_MAX_MM:
            o["label"] = "saucer"
        o["footprint"] = circle(o["mask"]) if o["label"] in ROUND else hull(o["mask"])
        o["on"], o["maybe_on"], o["covered_by"] = [], [], []

    # ids: red_gear_1, red_gear_2 ... from left to right; no number when there is only one
    for label in {o["label"] for o in objs}:
        same = sorted((o for o in objs if o["label"] == label), key=lambda o: o["pixel"][0])
        base = NAMES.get(label, slug(label))
        for n, o in enumerate(same, 1):
            o["id"] = base if len(same) == 1 else f"{base}_{n}"

    hands = [o for o in objs if o["label"] == "hand"]
    reach = np.zeros(scene.rgb.shape[:2], bool)
    for hnd in hands:
        reach |= grow(hnd["mask"], HAND_REACH_PX)
    for o in objs:
        u, v = o["pixel"]
        o["in"] = None if o["label"] == "grey tray" else "grey_tray" if tray is not None and tray[v, u] \
            else "black_tray" if mat is not None and mat[v, u] else None
        o["touching_hand"] = bool(hands) and o["label"] not in ("hand", "grey tray") and int((o["mask"] & reach).sum()) > 200

    # A lies on B when A covers part of where B should be. Seen from above, the upper object is the one seen whole:
    # the lower one cannot be seen where the upper one sits.
    things = [o for o in objs if o["label"] not in ("grey tray", "hand")]
    depth = []

    def height_gap(a, b):
        """(contact length in px, how much higher a stands than b beside the contact), or None without depth."""
        zone = grow(a["mask"], TOUCH_PX) & grow(b["mask"], TOUCH_PX)
        if not zone.any():
            return None
        if not depth:
            depth.append(relative_depth(scene_dir, scene.rgb))
        if depth[0] is None:
            return None
        near = grow(zone, 30)
        return zone.sum() / (2 * TOUCH_PX + 1), float(np.median(depth[0][a["mask"] & near]) - np.median(depth[0][b["mask"] & near]))

    for i, a in enumerate(things):
        for b in things[i + 1:]:
            if not (grow(a["mask"], TOUCH_PX) & b["footprint"]).any() and not (grow(b["mask"], TOUCH_PX) & a["footprint"]).any():
                continue
            # the share of the smaller of the two that x's visible pixels fill inside y's footprint:
            # a small object on a large one counts as fully as a large one across a small one
            cover = {x["id"]: (x["mask"] & y["footprint"]).sum() / min(x["area"], y["footprint"].sum())
                     for x, y in ((a, b), (b, a))}
            top, low = (a, b) if cover[a["id"]] >= cover[b["id"]] else (b, a)
            if top["label"] in CONTAINERS:
                continue
            if low["label"] in CONTAINERS:
                if cover[top["id"]] >= INSIDE:
                    top["on"].append(low["id"])
                continue
            if cover[top["id"]] >= ON_MIN_COVER:
                gap = height_gap(top, low)
                if gap is not None and gap[1] <= -MAYBE_MIN_HIGHER:
                    top["maybe_on"].append(low["id"])                   # depth says the "lower" one is higher: not sure
                else:
                    top["on"].append(low["id"])
                    low["covered_by"].append(top["id"])
                continue
            # touching, with no measurable overlap: the higher one may be resting on the other
            gap = height_gap(a, b)
            if gap is not None and gap[0] >= MAYBE_MIN_CONTACT_PX and abs(gap[1]) >= MAYBE_MIN_HIGHER:
                (a if gap[1] > 0 else b)["maybe_on"].append((b if gap[1] > 0 else a)["id"])

    # distances, so that "closest to" and "next to" are read off, not calculated by the language model
    mm_per_px = float(np.linalg.norm(pixel_to_table(scene, scene.rgb.shape[1] / 2 + 50, scene.rgb.shape[0] / 2)
                                     - pixel_to_table(scene, scene.rgb.shape[1] / 2 - 50, scene.rgb.shape[0] / 2))) / 100
    away = {o["id"]: cv2.distanceTransform((~o["mask"]).astype(np.uint8), cv2.DIST_L2, 3) for o in things}
    for o in objs:
        o["nearest"] = sorted(([b["id"], int(round(float(away[b["id"]][o["mask"]].min()) * mm_per_px))]
                               for b in things if b is not o), key=lambda x: x[1]) if o["id"] in away else []

    objs.sort(key=lambda o: (o["label"] in NOT_PICKABLE, o["id"]))
    h, w = scene.rgb.shape[:2]
    graph = {"scene": scene.name, "image_size": [w, h], "gripper_max_opening_mm": scene.gripper_max_opening_mm,
             "hand_present": bool(hands),
             "left_to_right": [o["id"] for o in sorted(things, key=lambda o: o["pixel"][0])],
             "top_to_bottom": [o["id"] for o in sorted(things, key=lambda o: o["pixel"][1])],
             "objects": [{"id": o["id"], "label": o["label"], "colour": colour_name(scene.rgb, o["mask"]),
                          "pixel": o["pixel"], "box_xyxy": o["box_xyxy"], "size_mm": o["size_mm"],
                          "pickable": o["label"] not in NOT_PICKABLE,
                          "why_not_pickable": NOT_PICKABLE.get(o["label"]), "in": o["in"], "on": o["on"],
                          "maybe_on": o["maybe_on"], "covered_by": o["covered_by"],
                          "touching_hand": o["touching_hand"], "nearest_gap_mm": o["nearest"], "found_by": o["found_by"],
                          "score": round(o["score"], 2)} for o in objs]}
    if mat is not None:
        graph["objects"].append({"id": "black_tray", "label": "black tray", "pickable": False,
                                 "why_not_pickable": TRAY,
                                 "note": "the black mat on the right; a place to put things"})
    return graph, objs


def render(scene_dir: Path, objs: list[dict], path: Path) -> None:
    canvas = cv2.imread(str(scene_dir / "rgb.png"), cv2.IMREAD_COLOR)
    rng = np.random.default_rng(0)
    for o in objs:
        text = o["id"] + (f" on {','.join(o['on'])}" if o["on"] else "") \
            + (f" on? {','.join(o['maybe_on'])}" if o["maybe_on"] else "") + (" HAND" if o["touching_hand"] else "")
        draw(canvas, o["mask"], o["pixel"], text, rng.integers(60, 256, 3).tolist())
    cv2.imwrite(str(path), canvas)


def scene_dirs(path: Path) -> list[Path]:
    return [path] if (path / "rgb.png").exists() else sorted(p for p in path.iterdir() if (p / "rgb.png").exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="a scene folder, or a folder of scenes")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--no-apertus", action="store_true", help="do not ask Apertus to name the objects it sees")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    for d in scene_dirs(args.data):
        graph, objs = build(d, detect(d), extra=[] if args.no_apertus else detect(d, names_for(d)))
        (args.out / f"{d.name}_graph.json").write_text(json.dumps(graph, indent=2))
        render(d, objs, args.out / f"{d.name}_graph.jpg")
        print(f"{d.name}: " + ", ".join(
            o["id"] + (f" (on {'+'.join(o['on'])})" if o.get("on") else "")
            + (f" (on? {'+'.join(o['maybe_on'])})" if o.get("maybe_on") else "") for o in graph["objects"]))


if __name__ == "__main__":
    main()
