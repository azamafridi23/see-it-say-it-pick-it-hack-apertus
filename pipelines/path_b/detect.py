"""Object detection by Apertus.

One call on the whole photo asks Apertus to box every object of the
workcell's kinds. Each box becomes one object, and the box's corrected
centre is the object's target pixel.
"""
import numpy as np

import paths  # noqa: F401
from apertus import Model, ask
from baseline import parse
from scene_io import Scene
from workcell import DetectedObject, Workcell

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

# Any JSON object: a strict schema made Apertus stop after 4 of 6 objects.
BOX_SCHEMA = {"type": "object"}


def find_boxes(model: Model, rgb: np.ndarray, cell: Workcell) -> list[dict]:
    """Ask Apertus for the objects on the whole photo, as boxes.

    Kinds that are not the workcell's, tray-sized boxes and boxes at the
    photo's edge (except a hand) are dropped, and each box's y is
    corrected (see ``workcell.ini``).

    Args:
        model: A model, such as an ``apertus.ApertusClient``.
        rgb: The photo as an RGB array.
        cell: The workcell's settings.

    Returns:
        A list of ``{"kind": str, "box": [x1, y1, x2, y2]}``, the box in
        pixels.
    """
    kinds = "\n".join(f"- {kind}" for kind in cell.kinds)
    reply = parse(ask(model, BOX_PROMPT.format(kinds=kinds), [rgb],
                      BOX_SCHEMA)) or {}
    h, w = rgb.shape[:2]
    found = []
    for entry in reply.get("objects") or []:
        box = entry.get("box") or entry.get("bounding_box") or []
        if entry.get("kind") not in cell.kinds or len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) / 1e6 > cell.max_box_area:
            continue

        at_edge = (min(x1, y1) <= cell.edge_margin
                   or max(x2, y2) >= 1000 - cell.edge_margin)
        if at_edge and entry["kind"] != "hand":
            continue

        found.append({"kind": entry["kind"], "box": [
            x1 * w / 1000,
            min(h, cell.y_scale * y1 * h / 1000 + cell.y_shift_px),
            x2 * w / 1000,
            min(h, cell.y_scale * y2 * h / 1000 + cell.y_shift_px)]})

    return found


def find_objects(model: Model, scene: Scene,
                 cell: Workcell) -> list[DetectedObject]:
    """Find the scene's objects, one per Apertus box.

    Args:
        model: A model, such as an ``apertus.ApertusClient``.
        scene: The scene, as loaded by ``scene_io.load_scene()``.
        cell: The workcell's settings.

    Returns:
        The objects, numbered from 1 left to right, each with its kind,
        box and target pixel; ``measure`` adds the measured facts.
    """
    boxes = sorted(find_boxes(model, scene.rgb, cell),
                   key=lambda found: found["box"][0] + found["box"][2])
    objects = []
    for number, found in enumerate(boxes, 1):
        x1, y1, x2, y2 = (round(c) for c in found["box"])
        objects.append(DetectedObject(
            id=number, kind=found["kind"], box=[x1, y1, x2, y2],
            pixel=[(x1 + x2) // 2, (y1 + y2) // 2]))

    return objects
