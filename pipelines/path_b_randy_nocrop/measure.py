"""OpenCV as a helper: it measures, it never decides what is an object.

Sizes and gaps on the table (through the camera calibration), a colour
word per object, which object lies on top where two Apertus boxes
overlap, and where the black mat starts.
"""
from itertools import combinations

import cv2
import numpy as np

import paths  # noqa: F401
from scene_io import pixel_to_table


# Below this saturation an object is white or grey, not a hue.
PALE_SAT = 50

# Colour words from the measured hue (OpenCV 0..180). The yellow/green
# edge sits between the yellow gears (23-29) and the lime mug inside (37).
COLOUR_HUES = (("red", 6), ("orange", 15), ("yellow", 33), ("green", 85),
               ("blue", 130), ("purple", 170))

# Two boxes overlapping by this share of the smaller one may be a stack
# (dev stacks 0.46-0.91, neighbours side by side up to 0.33); the one
# whose own colour shows in the overlap lies on top.
STACK_OVERLAP = 0.4

# How much nearer (Lab units) the overlap's colour must be to one object's
# than to the other's; a red cube on red gears stays only touching.
STACK_COLOUR = 15

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


def width_mm(scene, mark):
    """Return the mark's width in mm, measured on the table: its box's
    shorter side (twice the mark's radius). Tall objects read up to ~15 %
    wide: they are nearer the camera than the table."""
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


def add_gaps(scene, marks):
    """Give each mark `gaps`: {mark id: cm} to every other mark, edge to
    edge on the table."""
    for m in marks:
        m["gaps"] = {o["id"]: gap_cm(scene, m, o) for o in marks
                     if o is not m}


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


def mark_colour(rgb, mark):
    """Colour word from the median HSV inside the mark's inscribed circle."""
    (u, v), r = mark["pixel"], max(1, int(0.7 * mark["radius"]))
    patch = cv2.cvtColor(rgb[v - r:v + r, u - r:u + r], cv2.COLOR_RGB2HSV)
    hue, sat, val = (float(np.median(patch[..., i])) for i in range(3))
    if sat < PALE_SAT:
        return "white" if val > 170 else "grey"

    return next((name for name, top in COLOUR_HUES if hue < top), "red")
