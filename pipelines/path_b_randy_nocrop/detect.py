"""Apertus finds the objects: one call on the whole photo boxes every
object of the known kinds. Each box becomes one mark; its corrected
centre is the target pixel.

`box_names` is the older use of the same boxes: naming OpenCV marks
(`--opencv-marks`).
"""
import paths  # noqa: F401
from apertus import ask
from baseline import parse


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

UNLISTED = "unlisted"


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


def objects(model, scene):
    """Apertus finds the objects: one mark per box, numbered left to right.

    The pixel is the box centre, the outline the box, and the radius half
    its shorter side (for the width); `on` comes from measure.stacked_on().
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
