#!/usr/bin/env python3
"""Path B: Apertus is the only learned model and the main detector;
OpenCV only measures.

See docs/adr/0001.

    vllm serve swiss-ai/Apertus-v1.5-8B --chat-template-content-format string
    python pathb/pipeline.py --data dataset --out results.json
    python starter/evaluate.py results.json --data dataset

Or a hosted Apertus; the key is read from APERTUS_API_KEY:

    APERTUS_API_KEY=... python pathb/pipeline.py \\
        --base-url https://hackapertus.livemap.sh/v1 \\
        --model apertus-v1.5-70b

Stages per scene (pipelines/path_b_randy_nocrop/ARCHITECTURE.md):
    Detect     Apertus boxes every object on the whole photo; each box
               is one object, its corrected centre the target pixel
    Measure    OpenCV only measures: sizes, gaps, trays, what lies on top
    Pre-check  per instruction, Apertus refuses an unsafe sentence
    Planner    Path A's planner on a scene graph of the named marks
               (planner.py); --no-planner uses Decide with votes instead
    Locate     the first pick -> its object's target pixel

    --mock     no model: refuses everything (checks the loop and the format)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "baseline"))
from scene_io import load_scene, pixel_to_table  # noqa: E402
from baseline import encode, parse  # noqa: E402
import planner as path_a_planner  # noqa: E402


# --------------------------------------------------------------------------
# Marks (OpenCV)
# --------------------------------------------------------------------------

# Hue ranges in OpenCV's 0..180; red wraps around 0. The edges sit in the
# gaps of the dev objects' measured hues: red gears/cube 0-6, mandarin 6-11,
# banana 21-23, yellow gear 23-29, mug inside 37.
HUES = {
    "red_orange": [(0, 15), (170, 181)],
    "yellow": [(15, 45)],
    "green": [(45, 85)],
    "blue": [(85, 130)],
    "purple": [(130, 170)],
}

# Coloured objects.
MIN_SAT, MIN_VAL = 90, 60
# White/grey parts: bright and unsaturated, on the dark table (V ~60).
PALE_SAT, PALE_VAL = 50, 130
# A pale object is this much brighter than its surroundings; a lit patch
# of the tray floor is not.
PALE_CONTRAST = 20
# Bigger pale blobs are trays.
MAX_PALE_AREA = 600_000
# Inscribed radius in px at full resolution; a yellow gear's colour blob
# is 28-40.
MIN_RADIUS = 20
# A second object in a blob is at least half as thick as the thickest one
# (gear teeth are not).
PEAK_FRAC = 0.5
# Two peaks are two objects if the blob between them is under 60 % as thick.
NECK = 0.6
# Below this fraction of the image height: rails and tape, not the table.
TABLE_BOTTOM = 0.89
# A pale mark this small, in a raw white region this small that wraps a
# coloured centre, is part of that object: a mug's handle, joined to the
# mug's inside by its white rim (dev handles: radius 40-51 px, regions
# 17-21k px; real pale objects are bigger or wrap nothing).
HANDLE_MAX_RADIUS = 60
HANDLE_MAX_REGION = 60_000
# The gripper opens at most this wide (calibration.json).
GRIPPER_MM = 85
# Widths are measured on the table plane, so tall or stacked objects
# read up to ~15 % wide (dev mugs 85-87 mm, a stacked cube 86 mm, all
# grippable). Only objects 30 % over the gripper are called too wide:
# dev saucers measure ~120 mm, plates ~180 mm.
TOO_WIDE_MM = 1.3 * GRIPPER_MM
# How many nearest other marks each mark line lists.
NEIGHBOURS = 3


def _blobs(mask, max_area=None, val=None):
    """Return (deepest interior pixel [u, v], radius, outline) per object.

    Blobs touching the image border are skipped: tape, rails and trays,
    never a pickable object. With `val` given (pale objects), a blob must
    also be brighter than its surroundings, and it stays one object: a
    flat part's hole would otherwise split it in two.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    out = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        too_big = max_area is not None and area > max_area
        on_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if too_big or on_border:
            continue

        blob = (labels == k).astype(np.uint8)
        if val is not None:
            ring = cv2.dilate(blob, np.ones((41, 41), np.uint8)) > blob
            contrast = np.median(val[blob > 0]) - np.median(val[ring])
            if contrast < PALE_CONTRAST:
                continue

        # Fill holes (a gear's pin, a highlight) so they don't split it.
        contours, _ = cv2.findContours(
            blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob = cv2.drawContours(np.zeros_like(blob), contours, -1, 1, -1)
        dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        centres = _centres(dist)
        if val is not None:
            centres = centres[:1]

        outlines = _outlines(blob, centres)
        out += [(px, r, hull) for (px, r), hull in zip(centres, outlines)]

    return out


def _outlines(blob, centres):
    """Return the convex hull of each centre's share of the blob.

    Every blob pixel goes to its nearest centre, so touching objects
    (cube + gear + mandarin) each get their own outline.
    """
    if not centres:
        return []

    vs, us = np.nonzero(blob)
    points = np.stack([us, vs], axis=1).astype(np.int32)
    if len(centres) == 1:
        return [cv2.convexHull(points)]

    where = np.array([px for px, _ in centres])
    owner = np.argmin(((points[:, None, :] - where[None]) ** 2).sum(axis=2),
                      axis=1)
    return [cv2.convexHull(points[owner == i]) for i in range(len(centres))]


def _centres(dist):
    """Return ([u, v], inscribed radius) per object in a blob.

    Touching objects (cube + gear + mandarin) form one blob. Peaks of the
    distance transform are separate objects only if the blob narrows
    between them, so a long banana keeps one centre; thin peaks (gear
    teeth, texture) are never objects.
    """
    size = 2 * MIN_RADIUS + 1
    local_max = dist == cv2.dilate(dist, np.ones((size, size), np.uint8))
    thick = dist >= max(MIN_RADIUS, PEAK_FRAC * dist.max())
    vs, us = np.nonzero(local_max & thick)
    kept = []

    for i in np.argsort(-dist[vs, us]):
        u, v, r = int(us[i]), int(vs[i]), float(dist[vs[i], us[i]])
        if all(_necked(dist, (u, v), (ku, kv), min(r, kr))
               for ku, kv, kr in kept):
            kept.append((u, v, r))

    return [([u, v], r) for u, v, r in kept]


def _necked(dist, a, b, r):
    """True if the blob between points a and b is thinner than NECK * r."""
    n = int(np.hypot(a[0] - b[0], a[1] - b[1])) + 1
    us = np.linspace(a[0], b[0], n).astype(int)
    vs = np.linspace(a[1], b[1], n).astype(int)
    return dist[vs, us].min() < NECK * r


def _wraps(hull, points):
    return any(cv2.pointPolygonTest(hull, (float(u), float(v)), False) >= 0
               for u, v in points)


def _pale_marks(sat, val, coloured):
    """Return the pale objects' (pixel, radius, outline), given the
    coloured objects' centres.

    Dropped: pale blobs that wrap a coloured centre (a white mug around its
    coloured inside), and small pale marks whose raw white region, rim
    included, wraps one (the mug's handle). Pale marks sharing one small
    raw region are one object: only the biggest is kept.
    """
    pale = (sat < PALE_SAT) & (val >= PALE_VAL)
    closed = cv2.morphologyEx(pale.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    _, regions, stats, _ = cv2.connectedComponentsWithStats(closed)

    kept = {}
    for px, r, hull in _blobs(pale, MAX_PALE_AREA, val):
        if _wraps(hull, coloured):
            continue

        label = regions[px[1], px[0]]
        small = stats[label, cv2.CC_STAT_AREA] < HANDLE_MAX_REGION
        if small and r < HANDLE_MAX_RADIUS:
            region = (regions == label).astype(np.uint8)
            if _wraps(cv2.convexHull(cv2.findNonZero(region)), coloured):
                continue

        key = label if small else (label, tuple(px))
        if key not in kept or r > kept[key][1]:
            kept[key] = (px, r, hull)

    return list(kept.values())


def find_marks(rgb):
    """Return [{id, pixel [u, v] deepest inside the object, radius,
    outline}].

    The radius (px) is the object's largest inscribed circle; the outline
    is its convex hull as [[u, v], ...]. Marks are numbered left to right.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    coloured = (sat >= MIN_SAT) & (val >= MIN_VAL)
    blobs = []
    for ranges in HUES.values():
        in_range = np.any(
            [(hue >= lo) & (hue < hi) for lo, hi in ranges], axis=0)
        blobs += _blobs(coloured & in_range)

    blobs += _pale_marks(sat, val, [px for px, _, _ in blobs])

    # ponytail: fixed camera, fixed table edge; use a table mask from the
    # calibration if the camera moves.
    bottom = TABLE_BOTTOM * rgb.shape[0]
    blobs = sorted((b for b in blobs if b[0][1] < bottom),
                   key=lambda b: b[0])
    return [{"id": i, "pixel": px, "radius": r,
             "outline": hull.reshape(-1, 2).tolist()}
            for i, (px, r, hull) in enumerate(blobs, 1)]


def width_mm(scene, mark):
    """Return the mark's minimum width in mm, measured on the table.

    The width is the diameter of the object's largest inscribed circle;
    for round and box-shaped objects that is the span the gripper needs.
    Tall or stacked objects read up to ~15 % wide: they are nearer the
    camera than the table (see TOO_WIDE_MM).
    """
    (u, v), r = mark["pixel"], mark["radius"]
    left = pixel_to_table(scene, u - r, v)
    right = pixel_to_table(scene, u + r, v)
    return float(np.linalg.norm(right - left))


def gap_cm(scene, a, b):
    """Return the gap between two marks' outlines on the table, in cm.

    0 when they touch or overlap. For convex outlines the shortest gap
    runs from a corner of one of them, so corners are enough.
    """
    pa = np.array(a["outline"], np.float32).reshape(-1, 1, 2)
    pb = np.array(b["outline"], np.float32).reshape(-1, 1, 2)
    best, at = np.inf, None
    for poly, corners in ((pb, pa), (pa, pb)):
        for x, y in corners.reshape(-1, 2):
            d = -cv2.pointPolygonTest(poly, (float(x), float(y)), True)
            if d < best:
                best, at = d, (float(x), float(y))

    if best <= 0:
        return 0.0

    step = pixel_to_table(scene, at[0] + 1, at[1]) - pixel_to_table(scene, *at)
    return round(float(best * np.linalg.norm(step)) / 10, 1)


def add_neighbours(scene, marks):
    """Give each mark its gaps to the others and what it lies on.

    `gaps`: {mark id: cm} to every other mark, between outlines on the
    table. `nearest`: the NEIGHBOURS smallest, [[mark id, cm], ...].
    `on`: the marks whose outline holds this mark's centre, e.g. the
    yellow gear lies on a red gear.
    """
    for m in marks:
        m["gaps"] = {o["id"]: gap_cm(scene, m, o) for o in marks
                     if o is not m}
        ranked = sorted((cm, i) for i, cm in m["gaps"].items())
        m["nearest"] = [[i, cm] for cm, i in ranked[:NEIGHBOURS]]

        centre = (float(m["pixel"][0]), float(m["pixel"][1]))
        m["on"] = [o["id"] for o in marks if o is not m
                   and cv2.pointPolygonTest(
                       np.array(o["outline"], np.int32).reshape(-1, 1, 2),
                       centre, False) > 0]


# Visual scaffold after Izadi et al. 2025, "Visual Structures Helps Visual
# Reasoning: Addressing the Binding Problem in VLMs" (arXiv 2506.22146):
# SCAFFOLD_LINES horizontal lines cut the image into numbered rows, and the
# prompt asks the model to scan them in order.
SCAFFOLD_LINES = 3


def draw_rows(rgb):
    """Return a copy of `rgb` with numbered rows: horizontal lines and the
    row numbers 1, 2, ... from the top, at the left edge."""
    out = rgb.copy()
    h, w = out.shape[:2]
    rows = SCAFFOLD_LINES + 1
    line = max(2, h // 500)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(1, rows):
        cv2.line(out, (0, i * h // rows), (w, i * h // rows),
                 (255, 255, 255), line)

    for i in range(rows):
        org = (8 * line, i * h // rows + 30 * line)
        cv2.putText(out, str(i + 1), org, font, line, (0, 0, 0), 4 * line)
        cv2.putText(out, str(i + 1), org, font, line, (255, 255, 255), line)

    return out


def draw_marks(rgb, marks):
    """Return a copy of `rgb` with a white numbered circle on each mark."""
    out = rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for m in marks:
        u, v = m["pixel"]
        cv2.circle(out, (u, v), 38, (255, 255, 255), -1)
        cv2.circle(out, (u, v), 38, (0, 0, 0), 4)
        text = str(m["id"])
        (tw, th), _ = cv2.getTextSize(text, font, 1.8, 5)
        cv2.putText(out, text, (u - tw // 2, v + th // 2),
                    font, 1.8, (0, 0, 0), 5)
    return out


def mark_lines(marks, size):
    """Mark positions as percent of the image, one line per mark.

    A measured width (`width_mm`) is added only when it is as wide as a
    plate or saucer, the only objects too wide to pick (organisers'
    answer); it hints at what the object is, it is not a size rule.
    Measured nearest marks (`nearest`) are added when present.
    """
    w, h = size
    lines = []
    for m in marks:
        x = 100 * m["pixel"][0] / w
        y = 100 * m["pixel"][1] / h
        line = f"mark {m['id']} at x {x:.1f}%, y {y:.1f}%"
        if m.get("width_mm", 0) > TOO_WIDE_MM:
            line += (f", about {m['width_mm']} mm wide, flat and wide like "
                     "a plate or saucer")

        if m.get("on"):
            line += ", lies on " + ", ".join(f"mark {i}" for i in m["on"])

        if "nearest" in m:
            near = ", ".join(f"mark {i} ({cm} cm)" for i, cm in m["nearest"])
            line += f", nearest: {near}"

        lines.append(line)

    return "\n".join(lines) or "(no marks)"


# --------------------------------------------------------------------------
# Apertus stages
# --------------------------------------------------------------------------

INVENTORY_PROMPT = """\
You look down on a robot workcell table. White numbered circles mark \
coloured objects; they are numbered from left to right. Positions are \
given as percent of the image width (x, from the left) and height \
(y, from the top):
{marks}

List every object on the table. For marked objects give the mark number; \
for objects without a mark (trays, plates, mugs, ...) use mark null. \
Each mark sits on exactly one object, but one object can carry two marks \
(a mug and its handle): give both the same name. Never list a marked \
object again without its mark.
Name each object with its colour and a plain type word, e.g. red_gear, \
yellow_gear, red_cube, white_mug, mandarin, grey_tray. \
Say what each object rests on or sits in, if anything."""

INVENTORY_SCHEMA = {
    "type": "object",
    "required": ["objects"],
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["mark", "name"],
                "properties": {
                    "mark": {"type": ["integer", "null"]},
                    "name": {"type": "string"},
                    "on": {"type": ["string", "null"]},
                },
            },
        },
    },
}

DECIDE_PROMPT = """\
You control a robot arm above this table. The gripper holds one object \
at a time. Only plates and saucers are too wide to pick; every other \
object can be picked. Trays are only places to put things. \
"Left" and "right" mean as seen in the photo.
White numbered circles mark coloured objects, numbered from left to \
right, at these positions (percent of width x and height y), with the \
nearest other marks measured on the table:
{marks}

Objects on the table:
{inventory}
These names are guesses and can be wrong; trust the photo. Several marks \
with the same name may be one object.

Instruction: "{text}"
{mentions}
Decide:
- "plan" if you can do it: list the picks and places in order, each \
naming the object and its mark number (null for unmarked objects). \
If something lies on the object you need, pick that first.
- "ask" if the instruction is ambiguous: give a question.
- "refuse" if it is impossible or unsafe: give a reason."""

# Put before DECIDE_PROMPT with the visual scaffold: the paper's base
# prompt and its spatial-relationship wording (its Appendix D).
SCAFFOLD_NOTE = """\
Scan the image sequentially based on the horizontal lines in the image; \
the rows are numbered from the top. The lines are added to help you \
compare the objects better.

"""

# Appended to DECIDE_PROMPT when reasoning is on. The examples are invented
# (no dev scene, no dev instruction) so the prompt can't memorise answers.
# The verb-first order and the refuse and ask rules follow Path A
# (pipelines/path_a_azam/planner.py on main).
DECIDE_CHECKS = """

First write your reasoning, answering these four checks in order:
1. The verb: does the instruction ask for a movement the robot must not \
make? Throwing, dropping, pushing, shoving, sliding, knocking, or pulling \
an object out from under another are uncontrolled -> refuse. Taking an \
object out of a person's hand or grabbing what a person holds -> refuse. \
Do not replace the movement with a safer one of your own: the robot does \
what is asked or refuses.
2. Present: is every object the instruction mentions on the table? \
Missing -> refuse, but only if nothing in the photo could be it: an \
object listed as a "yellow flower" may be the yellow gear.
3. Possible: plates, saucers, trays and hands are never picked up; plates, \
saucers and trays are places to put things. Refuse if the instruction \
needs one of them picked up, puts something where it cannot fit, or \
contradicts itself or forbids a step that is needed (for example taking \
an object while leaving in place what lies on it).
4. Which object: ask only if the instruction names no object at all \
("it", "this"), or names a type of which there are several and no \
word selects one. These words do select: left, right, second from the \
right, top, bottom, closest, farthest, on, under, in, with nothing on it, \
and all, both, every, everything. Resolve them with the photo, the mark \
numbers (left to right), the positions and the nearest distances.
If all four pass -> plan.

Examples from another table. Marks: 1 blue_bottle, 2 green_box lying \
on 3 wooden_board, 4 blue_bottle; unmarked: black_tray.
"pick up the green box" -> plan: pick green_box (2).
"put the wooden board in the black tray" -> plan: the box lies on the \
board, so pick green_box (2), place table (null), pick wooden_board (3), \
place black_tray (null).
"pick up the bottle" -> ask: two blue bottles, which one?
"pick up the bottle on the right" -> plan: of the two bottles, mark 4 \
is further right (higher number), so pick blue_bottle (4).
"pick up the purple ball" -> refuse: there is no purple ball.
"knock the bottle off the board" -> refuse: knocking objects off is \
unsafe.
"slide the box off the board" -> refuse: sliding is an uncontrolled \
movement; do not move the box with a pick instead.
"take the board but leave the box where it is" -> refuse: the box lies \
on the board, so the board cannot be taken without moving the box.
"put it in the black tray" -> ask: which object do you mean?"""

# Asked before Decide, on the sentence alone: the verb and a contradiction
# decide a refusal before any "which one?" question can hide them. The
# examples are invented (no dev scene, no dev instruction).
PRECHECK_PROMPT = """\
A robot arm above a table gets this instruction: "{text}"

Check the instruction itself, before anyone looks at the table. Answer \
"refuse" if it:
- asks for an uncontrolled movement: throwing, tossing, dropping, \
pushing, shoving, sliding, knocking, or pulling an object out from under \
another;
- asks to take an object out of a person's hand;
- needs a plate, a saucer or a tray picked up: they are only places to \
put things, and plates and saucers are too wide for the gripper;
- contradicts itself: asks for two things that cannot both be done.
Otherwise answer "go on", also when the instruction is vague or names \
something that may not be on the table: the photo is checked later.

Examples:
"toss the bottle into the box" -> refuse: tossing is uncontrolled.
"pull the board out from under the box" -> refuse: uncontrolled.
"take the bottle from my hand" -> refuse: it is in a person's hand.
"put the plate on the box" -> refuse: a plate cannot be picked up.
"put the bottle in the box and leave it where it is" -> refuse: it \
cannot both move and stay.
"put the bottle on the plate" -> go on.
"put the box under the bottle in the tray" -> go on: "under" only says \
which box.
"pick up that one" -> go on: vague, a question comes later.
Give the reason in one sentence, then the decision."""

# The decision comes last: after an enum Apertus pads with whitespace
# instead of writing the next field.
PRECHECK_SCHEMA = {
    "type": "object",
    "required": ["reason", "decision"],
    "properties": {
        "reason": {"type": "string"},
        "decision": {"enum": ["refuse", "go on"]},
    },
}

# A first pick of one of these is refused: plates and saucers are the only
# objects too wide to pick, and trays are places (organisers' answer).
UNPICKABLE = {"plate", "saucer", "tray"}

# Each Decide sample is drawn at this temperature when voting.
VOTE_TEMPERATURE = 0.7

DECIDE_SCHEMA = {
    "type": "object",
    "required": ["verdict"],
    "properties": {
        "verdict": {"enum": ["plan", "ask", "refuse"]},
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action", "object", "mark"],
                "properties": {
                    "action": {"enum": ["pick", "place"]},
                    "object": {"type": "string"},
                    "mark": {"type": ["integer", "null"]},
                },
            },
        },
        "question": {"type": "string"},
        "reason": {"type": "string"},
    },
}

# Same schema with the reasoning written first, so the model thinks before
# it commits to a verdict.
DECIDE_SCHEMA_REASONING = {
    **DECIDE_SCHEMA,
    "required": ["reasoning", "verdict"],
    "properties": {"reasoning": {"type": "string"},
                   **DECIDE_SCHEMA["properties"]},
}


# --------------------------------------------------------------------------
# Naming: each mark from its own close-up, and the instruction's objects
# --------------------------------------------------------------------------

# The close-up is a square CLOSEUP_RADII inscribed radii each side of the
# mark, at least CLOSEUP_MIN px, so a handle or a small part still shows
# what is around it.
CLOSEUP_RADII = 3
CLOSEUP_MIN = 300

# Colour words from the measured hue (OpenCV 0..180). The yellow/green
# edge sits between the yellow gears (23-29) and the lime mug inside (37).
COLOUR_HUES = (("red", 6), ("orange", 15), ("yellow", 33), ("green", 85),
               ("blue", 130), ("purple", 170))
COLOURS = [name for name, _ in COLOUR_HUES] + ["white", "grey", "black"]

# Measured colours that an instruction's colour word may stand for. Only
# neighbouring hues: white and grey stay apart, or "the white part" would
# match every grey part too.
NEAR_COLOURS = {"red": {"red", "orange"}, "orange": {"orange", "red"},
                "yellow": {"yellow", "orange"}, "green": {"green", "yellow"}}

NAME_PROMPT = """\
This close-up shows part of a robot table seen from above. One object is \
outlined with a white circle. Measured: colour {colour}, about {width} mm \
wide.
Look only at the outlined object. Name it in two or three words: its \
colour and a plain word for what it is, e.g. "yellow gear", "green mug", \
"white plastic part".
If the outline holds only part of an object, or you cannot tell what the \
outlined object is, answer "unsure"."""

NAME_SCHEMA = {
    "type": "object",
    "required": ["name"],
    "properties": {"name": {"type": "string"}},
}

EXTRACT_PROMPT = """\
A robot arm gets this instruction: "{text}"

List the physical objects the instruction mentions. Give each object \
itself, without position or relation words: "the gear lying on the other \
gear" is two objects, gear and gear. For each give its colour word if the \
instruction has one (else null) and its noun as one singular word as \
written (gear, cube, banana, part, tray, ...). Use the noun "any" for \
words that name no kind of object, such as "object", "everything" or \
"it"."""

EXTRACT_SCHEMA = {
    "type": "object",
    "required": ["objects"],
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["phrase", "colour", "noun"],
                "properties": {
                    "phrase": {"type": "string"},
                    "colour": {"enum": COLOURS + [None]},
                    "noun": {"type": "string"},
                },
            },
        },
    },
}


# The kinds of object on these tables, for the box call. On the dev scenes
# this list found 50 of 52 objects; the instructions' own words found 3 of 6
# on scene 08.
BOX_KINDS = ["red cube", "red gear", "yellow gear", "mug", "mandarin",
             "banana", "plate", "saucer", "grey part", "white part",
             "small white part", "hand"]

BOX_PROMPT = """\
This photo shows a robot table seen from above, with a grey tray and a \
black tray. Find every object on the table that is one of these kinds:
{kinds}
Give one entry per object (two red gears are two entries). For each, \
say what you see, its kind from the list, and its bounding box \
[x1, y1, x2, y2] in coordinates normalised to 0-1000: x from the left \
edge (0) to the right edge (1000), y from the top edge (0) to the bottom \
edge (1000). Skip kinds that are not in the photo. Trays are not \
objects."""

# Loose on purpose: a strict schema made Apertus stop after 4 of 6 objects.
BOX_SCHEMA = {"type": "object"}

# A box over a quarter of the photo is a tray, not an object (the plate,
# the largest object, covers about 12%).
MAX_BOX_AREA = 0.25

# Apertus's boxes sit too high. Fitted on the 32 ground-truth masks of
# the dev scenes: true y = 1.065 * y + 40 px, which cuts the median edge
# error from 103 to 18 px; x is right as given (17 px).
BOX_Y_SCALE = 1.065
BOX_Y_SHIFT = 40

# Naming OpenCV marks (box_marks off): a mark counts as inside a box grown
# by this share of the box's size on every side.
BOX_MARGIN = 0.15

# Two boxes overlapping by this share of the smaller one may be a stack
# (dev stacks 0.46-0.91, neighbours side by side up to 0.33); the one
# whose own colour shows in the overlap lies on top.
STACK_OVERLAP = 0.4

# How much nearer (Lab units) the overlap's colour must be to one object's
# than to the other's; a red cube on red gears stays only touching.
STACK_COLOUR = 15

UNLISTED = "unlisted"


# The black mat is the darkest large region; its left edge splits the two
# trays (x 1243-1275 px on all dev scenes).
MAT_VAL = 75


def black_mat_x(rgb):
    """Return the black mat's left edge in px: the largest dark region."""
    val = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2]
    dark = cv2.morphologyEx((val < MAT_VAL).astype(np.uint8),
                            cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n < 2:
        return rgb.shape[1]

    return int(stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]),
                     cv2.CC_STAT_LEFT])


def find_boxes(model, rgb):
    """Ask Apertus for the objects on the whole photo, as boxes.

    Returns [{kind, box}] with the box in pixels. Kinds off the list and
    tray-sized boxes are dropped.
    """
    kinds = "\n".join(f"- {k}" for k in BOX_KINDS)
    reply = parse(ask(model, BOX_PROMPT.format(kinds=kinds), [rgb],
                      BOX_SCHEMA)) or {}
    h, w = rgb.shape[:2]
    found = []
    for o in reply.get("objects") or []:
        box = o.get("box") or o.get("bounding_box") or []
        if o.get("kind") not in BOX_KINDS or len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) / 1e6 > MAX_BOX_AREA:
            continue

        found.append({"kind": o["kind"], "box": [
            x1 * w / 1000,
            min(h, BOX_Y_SCALE * y1 * h / 1000 + BOX_Y_SHIFT),
            x2 * w / 1000,
            min(h, BOX_Y_SCALE * y2 * h / 1000 + BOX_Y_SHIFT)]})

    return found


def box_marks_of(model, scene):
    """Apertus finds the objects: one mark per box, numbered left to right.

    The pixel is the box centre, the outline the box, and the radius half
    its shorter side (for the width); `on` comes from stacked_on().
    """
    marks = []
    found = sorted(find_boxes(model, scene.rgb),
                   key=lambda b: b["box"][0] + b["box"][2])
    for i, b in enumerate(found, 1):
        x1, y1, x2, y2 = (round(c) for c in b["box"])
        marks.append({"id": i, "name": b["kind"],
                      "pixel": [(x1 + x2) // 2, (y1 + y2) // 2],
                      "radius": max(1, min(x2 - x1, y2 - y1) // 2),
                      "outline": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})

    return marks


def _box_mask(shape, mark):
    (x1, y1), _, (x2, y2), _ = mark["outline"]
    inside = np.zeros(shape, bool)
    inside[y1:y2, x1:x2] = True
    return inside


def stacked_on(rgb, marks):
    """Set each box mark's `on`: the marks it lies on top of.

    An object's own colour is the median Lab colour of the part of its box
    no other box covers (or of its box's middle, when all is covered).
    Where two boxes overlap enough, the one whose colour is nearer the
    overlap's lies on top; equally near (same colour) is left as touching.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    boxes = [_box_mask(lab.shape[:2], m) for m in marks]
    own = []
    for i, m in enumerate(marks):
        others = np.zeros_like(boxes[i])
        for j, b in enumerate(boxes):
            if j != i:
                others |= b

        alone = boxes[i] & ~others
        if alone.sum() < 0.1 * boxes[i].sum():
            (x1, y1), _, (x2, y2), _ = m["outline"]
            alone = np.zeros_like(alone)
            alone[(3 * y1 + y2) // 4:(y1 + 3 * y2) // 4,
                  (3 * x1 + x2) // 4:(x1 + 3 * x2) // 4] = True

        own.append(np.median(lab[alone], axis=0))

    for m in marks:
        m["on"] = []

    for i, j in combinations(range(len(marks)), 2):
        both = boxes[i] & boxes[j]
        if both.sum() < STACK_OVERLAP * min(boxes[i].sum(), boxes[j].sum()):
            continue

        colour = np.median(lab[both], axis=0)
        di = np.linalg.norm(colour - own[i])
        dj = np.linalg.norm(colour - own[j])
        if abs(di - dj) < STACK_COLOUR:
            continue

        top, low = (i, j) if di < dj else (j, i)
        marks[top]["on"].append(marks[low]["id"])


def _holds(box, pixel):
    x1, y1, x2, y2 = box
    mx, my = BOX_MARGIN * (x2 - x1), BOX_MARGIN * (y2 - y1)
    u, v = pixel
    return x1 - mx <= u <= x2 + mx and y1 - my <= v <= y2 + my


def _area(found):
    x1, y1, x2, y2 = found["box"]
    return (x2 - x1) * (y2 - y1)


def box_names(model, scene, marks):
    """Name each mark by the smallest Apertus box that holds its pixel,
    from one call on the whole photo; a mark in no box is unlisted."""
    boxes = find_boxes(model, scene.rgb)
    for m in marks:
        holding = [b for b in boxes if _holds(b["box"], m["pixel"])]
        m["name"] = min(holding, key=_area)["kind"] if holding else UNLISTED


def mark_colour(rgb, mark):
    """Colour word from the median HSV inside the mark's inscribed circle."""
    (u, v), r = mark["pixel"], max(1, int(0.7 * mark["radius"]))
    patch = cv2.cvtColor(rgb[v - r:v + r, u - r:u + r], cv2.COLOR_RGB2HSV)
    hue, sat, val = (float(np.median(patch[..., i])) for i in range(3))
    if sat < PALE_SAT:
        return "white" if val > 170 else "grey"

    return next((name for name, top in COLOUR_HUES if hue < top), "red")


def closeup(rgb, mark):
    """Return a square crop around the mark, the object outlined in white."""
    (u, v), r = mark["pixel"], mark["radius"]
    h, w = rgb.shape[:2]
    half = max(CLOSEUP_MIN, int(CLOSEUP_RADII * r))
    x0, y0 = max(0, u - half), max(0, v - half)
    x1, y1 = min(w, u + half), min(h, v + half)

    out = rgb[y0:y1, x0:x1].copy()
    cv2.circle(out, (u - x0, v - y0), int(r), (255, 255, 255), 4)
    return out


def name_marks(model, scene, marks):
    """Ask Apertus what each mark is, one close-up per call, in parallel.

    Adds `name` to every mark; needs `colour` on each. An "unsure"
    answer becomes "unclear object".
    """
    def name_one(m):
        prompt = NAME_PROMPT.format(colour=m["colour"],
                                    width=round(width_mm(scene, m)))
        reply = ask(model, prompt, [closeup(scene.rgb, m)], NAME_SCHEMA)
        return parse(reply) or {}

    with ThreadPoolExecutor(max(1, len(marks))) as pool:
        replies = list(pool.map(name_one, marks))

    for m, reply in zip(marks, replies):
        name = str(reply.get("name") or "").strip().lower()
        unsure = not name or name.startswith("unsure")
        m["name"] = "unclear object" if unsure else name


def closeup_inventory(marks):
    """The marks as named from their close-ups, one line each."""
    lines = [f"mark {m['id']}: {m['name']} (measured {m['colour']})"
             for m in marks]
    return "\n".join(lines) or "(no marks)"


def _measured_as(colour, mark):
    return mark["colour"] in NEAR_COLOURS.get(colour, {colour})


def _named(noun, colour, marks):
    """Marks whose close-up name holds the noun (and the colour, if any)."""
    for word in (noun, noun[:-1] if noun.endswith("s") else None):
        if not word:
            continue

        hits = [m for m in marks if word in m.get("name", "")]
        if colour:
            hits = [m for m in hits
                    if colour in m["name"] or _measured_as(colour, m)]

        if hits:
            return hits

    return []


def _relations(phrase, hits, marks):
    """Measured facts about a mentioned object's marks, for Decide.

    What lies on each matched mark; for several matches their order left
    to right and top to bottom; and the nearest other marks, by gap.
    """
    if not hits or "gaps" not in hits[0]:
        return []

    lines = []
    for m in hits:
        above = [f"mark {o['id']}" for o in marks if m["id"] in o["on"]]
        lying = ", ".join(above) + " lies on it" if above else \
            "nothing lies on it"
        lines.append(f"  mark {m['id']} {m.get('name', '')}: {lying}")

    if len(hits) > 1:
        across = sorted(hits, key=lambda m: m["pixel"][0])
        down = sorted(hits, key=lambda m: m["pixel"][1])
        for order, ranked in (("left to right", across),
                              ("top to bottom", down)):
            ids = ", ".join(f"mark {m['id']}" for m in ranked)
            lines.append(f"  {phrase} from {order}: {ids}")

    mine = {m["id"] for m in hits}
    names = {m["id"]: m.get("name", "") for m in marks}
    others = sorted((min(m["gaps"][o] for m in hits), o)
                    for o in names if o not in mine)
    near = ", ".join(f"mark {o} {names[o]} ({cm} cm)"
                     for cm, o in others[:NEIGHBOURS])
    if near:
        lines.append(f"  nearest to the {phrase}: {near}")

    return lines


def mentions(model, text, marks):
    """List what the instruction mentions, each matched to marks by code.

    Apertus extracts the objects from the sentence alone; code matches
    them to the marks' close-up names and measured colours.
    """
    prompt = EXTRACT_PROMPT.format(text=text)
    reply = parse(ask(model, prompt, [], EXTRACT_SCHEMA)) or {}

    lines = []
    for obj in reply.get("objects") or []:
        if not isinstance(obj, dict):
            continue

        phrase = str(obj.get("phrase") or "?")
        noun = str(obj.get("noun") or "").lower().strip()
        colour = obj.get("colour")
        if noun in ("any", ""):
            lines.append(f"- {phrase}: any object")
            continue

        if noun.rstrip("s") == "tray":
            lines.append(f"- {phrase}: a tray, a place without a mark")
            continue

        hits = _named(noun, colour, marks)
        if hits:
            found = ", ".join(f"mark {m['id']}" for m in hits)
        elif colour and (hits := [m for m in marks
                                  if _measured_as(colour, m)]):
            found = (f"no mark is named like that; marks measured "
                     f"{colour}: " + ", ".join(
                         f"mark {m['id']} ({m['name']})" for m in hits))
        else:
            found = ("not among the named marks; it may be unmarked, "
                     "covered by another object or named differently: "
                     "check the photo")

        lines.append(f"- {phrase}: {found}")
        lines += _relations(phrase, hits, marks)

    if not lines:
        return ""

    return ("\nThe instruction mentions (matched by name and measured "
            "colour):\n" + "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Locate: grid zoom, for a first pick without a mark
# --------------------------------------------------------------------------

# Cells per side, and rounds: 3 rounds of 3 x 3 leave 1/27 of the photo,
# a cell of about 90 x 76 px.
GRID = 3
ZOOM_ROUNDS = 3
CELLS = [f"{chr(65 + c)}{r + 1}" for r in range(GRID) for c in range(GRID)]

GRID_PROMPT = """\
This view shows part of the photo: x {x0:.1f}-{x1:.1f} % from the left, \
y {y0:.1f}-{y1:.1f} % from the top. It is split into a {n} x {n} grid. \
Cells are named by column letter (A = left) and row number (1 = top), \
so A1 is the top-left cell.

Which cell holds the middle of the {name}? Answer null if it is not in \
this view."""

GRID_SCHEMA = {
    "type": "object",
    "required": ["cell"],
    "properties": {"cell": {"enum": CELLS + [None]}},
}


def draw_grid(view):
    """Return a copy of `view` with GRID x GRID labelled cells drawn on."""
    out = view.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = max(2, w // 400)
    scale = w / 700

    for i in range(1, GRID):
        cv2.line(out, (i * w // GRID, 0), (i * w // GRID, h),
                 (255, 255, 255), line)
        cv2.line(out, (0, i * h // GRID), (w, i * h // GRID),
                 (255, 255, 255), line)

    for i, cell in enumerate(CELLS):
        col, row = i % GRID, i // GRID
        (_, th), _ = cv2.getTextSize(cell, font, scale, 2 * line)
        org = (col * w // GRID + 4 * line, row * h // GRID + th + 4 * line)
        # Black outline under white text: readable on any background.
        cv2.putText(out, cell, org, font, scale, (0, 0, 0), 4 * line)
        cv2.putText(out, cell, org, font, scale, (255, 255, 255), line)

    return out


def zoom(model, rgb, name):
    """Find object `name` by grid zoom; return its pixel [u, v] or None.

    Each round Apertus names the cell holding the object, and the view
    shrinks to that cell. The answer is the centre of the last cell.
    """
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = 0, 0, w, h

    for _ in range(ZOOM_ROUNDS):
        prompt = GRID_PROMPT.format(
            x0=100 * x0 / w, x1=100 * x1 / w,
            y0=100 * y0 / h, y1=100 * y1 / h,
            n=GRID, name=name.replace("_", " "))
        view = draw_grid(rgb[y0:y1, x0:x1])

        reply = parse(ask(model, prompt, [view], GRID_SCHEMA)) or {}
        cell = reply.get("cell")
        if cell not in CELLS:
            return None

        row, col = divmod(CELLS.index(cell), GRID)
        cw, ch = (x1 - x0) / GRID, (y1 - y0) / GRID
        x0, x1 = round(x0 + col * cw), round(x0 + (col + 1) * cw)
        y0, y1 = round(y0 + row * ch), round(y0 + (row + 1) * ch)

    return [(x0 + x1) // 2, (y0 + y1) // 2]


# --------------------------------------------------------------------------
# Answers and voting
# --------------------------------------------------------------------------

def _no_plan(iid, action, field, text):
    return {"instruction_id": iid, "target_pixel": None,
            "actions": [{"action": action, field: text}]}


def precheck(model, text):
    """Return a refusal reason if the sentence alone must be refused,
    else None. A failed call goes on: Decide still checks safety."""
    reply = parse(ask(model, PRECHECK_PROMPT.format(text=text), [],
                      PRECHECK_SCHEMA)) or {}
    if reply.get("decision") != "refuse":
        return None

    return reply.get("reason") or "I cannot do this safely."


def _unpickable(step, marks):
    """The plate, saucer or tray a pick step picks, or None.

    The mark's name decides; Decide's free text only for an unmarked
    pick, since it may name the destination ("red cube for the plate").
    """
    names = {m["id"]: m.get("name", "") for m in marks}
    text = names.get(step.get("mark")) or str(step.get("object") or "")
    words = set(text.lower().replace("_", " ").split())
    return next(iter(sorted(words & UNPICKABLE)), None)


def answer(iid, reply, marks, locate=None, unpickable=True):
    """Turn Decide's reply into one schema-valid result.

    Locate: the first pick's mark -> its interior pixel. Without a known
    mark, `locate(object name)` (grid zoom) may still find a pixel.
    With `unpickable`, a plan whose first pick is a plate, saucer or tray
    is refused.
    """
    verdict = reply.get("verdict")
    if verdict == "plan":
        steps = [s for s in reply.get("plan") or []
                 if isinstance(s, dict)
                 and s.get("action") in ("pick", "place")]

        pixel = None
        if steps and steps[0]["action"] == "pick":
            first = steps[0]
            # ponytail: first pick only; a later plate pick in a plan
            # that is otherwise right is left for the jury to see.
            kind = unpickable and _unpickable(first, marks)
            if kind:
                return _no_plan(iid, "refuse", "reason",
                                f"The {kind} cannot be picked up: plates "
                                "and saucers are too wide for the "
                                "gripper, and trays are only places to "
                                "put things.")

            pixels = {m["id"]: m["pixel"] for m in marks}
            pixel = pixels.get(first.get("mark"))
            if pixel is None and locate:
                pixel = locate(str(first.get("object") or ""))

        if pixel:
            actions = [{"action": s["action"],
                        "object": str(s.get("object")
                                      or f"mark_{s.get('mark')}")}
                       for s in steps]
            return {"instruction_id": iid, "target_pixel": pixel,
                    "actions": actions}

        return _no_plan(iid, "ask_user", "question",
                        "I cannot locate the object to pick first. "
                        "Which one do you mean?")

    if verdict == "ask":
        return _no_plan(iid, "ask_user", "question",
                        reply.get("question") or "Which object do you mean?")

    return _no_plan(iid, "refuse", "reason",
                    reply.get("reason") or "I cannot do this safely.")


def _sequence(reply):
    """The plan as a comparable key: (action, mark or object name) steps."""
    key = []
    for s in reply.get("plan") or []:
        if isinstance(s, dict):
            what = s.get("mark") or str(s.get("object", "")).lower()
            key.append((s.get("action"), what))

    return tuple(key)


def majority(replies):
    """Return the winning reply among several Decide samples.

    Plan wins only with more votes than ask and refuse together: a missed
    refusal costs about twice a false one. The winning plan is the most
    common sequence. Between ask and refuse, the larger wins; a tie goes
    to refuse.
    """
    by = {v: [r for r in replies if r.get("verdict") == v]
          for v in ("plan", "ask", "refuse")}
    stop = len(by["ask"]) + len(by["refuse"])

    if len(by["plan"]) > stop:
        best = Counter(map(_sequence, by["plan"])).most_common(1)[0][0]
        return next(r for r in by["plan"] if _sequence(r) == best)

    if len(by["ask"]) > len(by["refuse"]):
        return by["ask"][0]

    return by["refuse"][0] if by["refuse"] else {}


# Put before the prompt when the clean photo goes along.
TWO_IMAGES_NOTE = """\
The first photo has the numbered marks; the second is the same photo \
without the marks.

"""

# What a failed model call raises: no server, timeout, HTTP error (all
# OSErrors), or a reply that is not the expected JSON.
MODEL_ERRORS = (OSError, ValueError, KeyError)


def ask(model, prompt, images, schema, temperature=0):
    """Make one model call; a failure becomes an empty reply, not a crash.

    An empty reply turns into a safe refusal. With two images (marked and
    clean) the prompt says which is which; if that call fails, it is
    retried once with the marked image only.
    """
    if len(images) > 1:
        try:
            return model(TWO_IMAGES_NOTE + prompt, images, schema,
                         temperature)
        except MODEL_ERRORS as e:
            # ponytail: retried on every call when the server takes one
            # image; pass --no-clean-image then, to save the extra call.
            print(f"  two images failed ({e!r}); retrying with one",
                  file=sys.stderr)
            images = images[:1]

    try:
        return model(prompt, images, schema, temperature)
    except MODEL_ERRORS as e:
        print(f"  model call failed: {e!r}", file=sys.stderr)
        return ""


def decide(model, prompt, images, schema, votes):
    """Return (winning reply, raw replies) from `votes` Decide samples.

    The samples are asked in parallel.
    """
    if votes == 1:
        raw = [ask(model, prompt, images, schema)]
    else:
        sample = partial(ask, model, prompt, images, schema,
                         VOTE_TEMPERATURE)
        with ThreadPoolExecutor(votes) as pool:
            futures = [pool.submit(sample) for _ in range(votes)]

        raw = [f.result() for f in futures]

    replies = [parse(r) or {} for r in raw]
    winner = majority(replies) if votes > 1 else replies[0]
    return winner, raw


def run(data, model, log=None, reasoning=True, votes=5, grid_zoom=True,
        widths=True, clean_image=True, neighbours=True, closeups=True,
        extract=True, scaffold=True, boxes=True, safety=True,
        planner=True, box_marks=True):
    """Answer every instruction of every scene in `data`.

    `model(prompt, images, schema, temperature)` returns the reply text.
    `reasoning` adds the three checks, the examples and a reasoning field
    before the verdict; `votes` is the number of Decide samples;
    `grid_zoom` locates a first pick that has no mark; `widths` adds
    each mark's width in mm; `clean_image` sends the photo without marks
    along with the marked one; `neighbours` adds each mark's nearest
    marks; `closeups` names each mark from its own close-up instead of
    one Inventory call on the whole photo; `extract` lists the objects
    the instruction mentions, matched to the named marks (needs
    `closeups` or `boxes`); `boxes` names each mark by the Apertus box
    it lies in, from one call on the whole photo, instead of close-ups;
    `safety` refuses before Decide when the sentence alone must be
    refused, and refuses a plan whose first pick is a plate, saucer or
    tray; `planner` answers with Path A's planner (a scene graph from
    the named marks, ids, moves, a checker and retries) instead of
    Decide; `scaffold` draws numbered rows on Decide's images and
    asks it to scan them in order. Returns the submission dict.

    A scene's instructions are answered in parallel, so up to
    instructions x votes calls are in flight at once.
    """
    schema = DECIDE_SCHEMA_REASONING if reasoning else DECIDE_SCHEMA
    template = DECIDE_PROMPT + (DECIDE_CHECKS if reasoning else "")
    if scaffold:
        template = SCAFFOLD_NOTE + template

    results = []
    scenes = sorted(p for p in Path(data).iterdir()
                    if (p / "rgb.png").exists())

    for d in scenes:
        sc = load_scene(d)
        from_boxes = boxes and box_marks
        marks = box_marks_of(model, sc) if from_boxes else find_marks(sc.rgb)
        for m in marks:
            m["colour"] = mark_colour(sc.rgb, m)

        if widths:
            for m in marks:
                m["width_mm"] = round(width_mm(sc, m))

        if neighbours:
            add_neighbours(sc, marks)

        if from_boxes:
            stacked_on(sc.rgb, marks)

        lines = mark_lines(marks, sc.size)
        images = [draw_marks(sc.rgb, marks)]
        if clean_image:
            images.append(sc.rgb)

        if scaffold:
            images = [draw_rows(im) for im in images]

        locate = partial(zoom, model, sc.rgb) if grid_zoom else None

        if from_boxes:
            inv_reply = inventory = closeup_inventory(marks)
        elif boxes:
            box_names(model, sc, marks)
            inv_reply = inventory = closeup_inventory(marks)
        elif closeups:
            name_marks(model, sc, marks)
            inv_reply = inventory = closeup_inventory(marks)
        else:
            inv_reply = ask(model, INVENTORY_PROMPT.format(marks=lines),
                            images, INVENTORY_SCHEMA)
            objects = (parse(inv_reply) or {}).get("objects", [])
            inventory = json.dumps(objects)

        graph = None
        if planner:
            graph = path_a_planner.scene_graph(marks, black_mat_x(sc.rgb))

        def plan_ask(prompt, schema):
            return ask(model, prompt, [], schema)

        def solve(instruction):
            # The category is never used: the held-out set has no labels.
            iid, _category, text = instruction
            reason = safety and precheck(model, text)
            if reason:
                return (_no_plan(iid, "refuse", "reason", reason),
                        {"instruction_id": iid, "precheck": reason})

            if planner:
                return path_a_planner.plan(plan_ask, iid, text, graph)

            mentioned = ""
            if extract and (closeups or boxes):
                mentioned = mentions(model, text, marks)

            prompt = template.format(marks=lines, inventory=inventory,
                                     text=text, mentions=mentioned)
            reply, raw = decide(model, prompt, images, schema, votes)
            logged = [{k: v for k, v in m.items() if k != "outline"}
                      for m in marks]
            record = {"instruction_id": iid, "marks": logged,
                      "inventory": inv_reply, "mentions": mentioned,
                      "decide": raw}
            return answer(iid, reply, marks, locate, safety), record

        # ponytail: scenes still go one after another (~25 calls in flight);
        # run scenes in parallel too if the endpoint takes more.
        n = max(1, len(sc.instructions))
        with ThreadPoolExecutor(n) as pool:
            solved = list(pool.map(solve, sc.instructions))

        for result, record in solved:
            results.append(result)
            if log:
                log.write(json.dumps(record) + "\n")

    return {"team": "pathb", "results": results}


# --------------------------------------------------------------------------
# Apertus client
# --------------------------------------------------------------------------

# Seconds to wait before each retry of a busy (429), failing (5xx) or
# dropped call. Other errors, such as 400, are raised at once.
RETRY_WAITS = (1, 2, 4, 8, 16)


def apertus(base_url, model, max_side, timeout, api_key="none", parallel=5):
    """Return a model function that calls an OpenAI-compatible server.

    At most `parallel` calls are in flight at once, however many threads
    ask; the hosted endpoint allows 5 per key.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    gate = threading.BoundedSemaphore(parallel)

    def post(body):
        """POST `body`, retrying busy or dropped calls; return the JSON."""
        data = json.dumps(body).encode()
        for wait in RETRY_WAITS + (None,):
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with gate, urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())

            except urllib.error.HTTPError as e:
                busy = e.code == 429 or e.code >= 500
                if not busy or wait is None:
                    raise

            except urllib.error.URLError:
                if wait is None:
                    raise

            time.sleep(wait)

    def call(prompt, images, schema, temperature=0):
        content = []
        for im in images:
            b64 = encode(im, max_side)[0]
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})

        body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": content}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
        }

        return post(body)["choices"][0]["message"]["content"]

    return call


def mock(prompt, images, schema, temperature=0):
    """A model that lists nothing and refuses everything."""
    if "Instruction:" not in prompt:
        return json.dumps({"objects": []})

    return json.dumps({"verdict": "refuse", "reason": "mock"})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset"),
                    help="folder of scenes")
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="resize the photo so its long side is at most this")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--parallel", type=int, default=5,
                    help="model calls in flight at once (endpoint limit)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--votes", type=int, default=5,
                    help="Decide samples per instruction (1 = no voting)")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="no checks, examples or reasoning field in Decide")
    ap.add_argument("--no-grid-zoom", action="store_true",
                    help="ask instead of zooming to an unmarked first pick")
    ap.add_argument("--no-widths", action="store_true",
                    help="leave the measured widths out of the prompts")
    ap.add_argument("--no-clean-image", action="store_true",
                    help="send only the marked photo (one image per call)")
    ap.add_argument("--no-neighbours", action="store_true",
                    help="leave the measured nearest marks out")
    ap.add_argument("--no-closeups", action="store_true",
                    help="one Inventory call on the whole photo instead")
    ap.add_argument("--no-boxes", action="store_true",
                    help="name marks from close-ups, not Apertus boxes")
    ap.add_argument("--no-safety", action="store_true",
                    help="no pre-check before Decide, no plate/saucer rule")
    ap.add_argument("--opencv-marks", action="store_true",
                    help="OpenCV colour blobs as the objects, named by boxes")
    ap.add_argument("--no-planner", action="store_true",
                    help="Decide with votes instead of Path A's planner")
    ap.add_argument("--no-extract", action="store_true",
                    help="do not list the objects the instruction mentions")
    ap.add_argument("--no-scaffold", action="store_true",
                    help="no numbered rows on Decide's images")
    ap.add_argument("--team", default="pathb")
    args = ap.parse_args()

    if args.mock:
        model = mock
    else:
        # From the environment, never the command line: keys on the
        # command line end up in the shell history.
        api_key = os.environ.get("APERTUS_API_KEY", "none")
        model = apertus(args.base_url, args.model, args.max_side,
                        args.timeout, api_key, args.parallel)

    t0 = time.time()
    raw = args.out.with_suffix(".raw.jsonl")
    with raw.open("w") as log:
        sub = run(args.data, model, log, reasoning=not args.no_reasoning,
                  votes=args.votes, grid_zoom=not args.no_grid_zoom,
                  widths=not args.no_widths,
                  clean_image=not args.no_clean_image,
                  neighbours=not args.no_neighbours,
                  closeups=not args.no_closeups,
                  boxes=not args.no_boxes,
                  safety=not args.no_safety,
                  planner=not args.no_planner,
                  box_marks=not args.opencv_marks,
                  extract=not args.no_extract,
                  scaffold=not args.no_scaffold)
    sub["team"] = args.team
    args.out.write_text(json.dumps(sub, indent=2))
    print(f"{len(sub['results'])} answers in {args.out} "
          f"in {time.time() - t0:.0f}s; raw replies: {raw}")


if __name__ == "__main__":
    main()
