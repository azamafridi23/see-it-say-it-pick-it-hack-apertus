"""Measurements by OpenCV, for the objects Apertus found.

OpenCV only measures; it never decides what is an object. It gives each
object a width and gaps on the table (through the camera calibration), a
colour name, and what it lies on where two boxes overlap; it finds where
the black mat starts; and it tells the part kinds apart by size. Every
threshold comes from the workcell's settings (``workcell.ini``).
"""
from itertools import combinations

import cv2
import numpy as np

import paths  # noqa: F401
from scene_io import Scene, pixel_to_table
from workcell import DetectedObject, Workcell


def black_mat_left(rgb: np.ndarray, cell: Workcell) -> int:
    """Find the left edge of the black mat, the photo's largest dark region.

    Args:
        rgb: The photo as an RGB array.
        cell: The workcell's settings.

    Returns:
        The edge's x in pixels, or the photo's width when there is no dark
        region.
    """
    brightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2]
    dark = (brightness < cell.mat_max_brightness).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    if count < 2:
        return rgb.shape[1]

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return int(stats[largest, cv2.CC_STAT_LEFT])


def box_size_mm(scene: Scene, obj: DetectedObject) -> tuple[float, float]:
    """Measure an object's box on the table, across its centre.

    Args:
        scene: The scene, with its camera calibration.
        obj: A detected object.

    Returns:
        ``(long, short)``: the box's longer and shorter side in
        millimetres.
    """
    x1, y1, x2, y2 = obj.box
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    across = np.linalg.norm(pixel_to_table(scene, x2, v)
                            - pixel_to_table(scene, x1, v))
    down = np.linalg.norm(pixel_to_table(scene, u, y2)
                          - pixel_to_table(scene, u, y1))
    return float(max(across, down)), float(min(across, down))


def name_parts_by_size(scene: Scene, objects: list[DetectedObject],
                       cell: Workcell) -> None:
    """Decide which kind of part each part is, by its size on the table.

    Apertus decides that an object is a part; under this light it cannot
    tell grey metal from white plastic, so the measured size decides which
    part it is. The size comes from the upright box, so a part lying
    diagonally measures squarer than it is.

    Args:
        scene: The scene, with its camera calibration.
        objects: The detected objects; changed in place.
        cell: The workcell's settings.
    """
    for obj in objects:
        if obj.kind not in cell.part_kinds:
            continue

        long, short = box_size_mm(scene, obj)
        if long < cell.small_part_max_mm:
            obj.kind = "small white part"
        elif long >= cell.grey_part_min_aspect * short:
            obj.kind = "grey part"
        else:
            obj.kind = "white part"


def _half_size(obj: DetectedObject) -> int:
    """Return half the object's box's shorter side, in pixels (at least 1)."""
    x1, y1, x2, y2 = obj.box
    return max(1, min(x2 - x1, y2 - y1) // 2)


def width_mm(scene: Scene, obj: DetectedObject) -> float:
    """Measure an object's width on the table: its box's shorter side.

    The width is taken across the box's centre. Tall objects read up to
    about 15% too wide, because their top is nearer the camera than the
    table.

    Args:
        scene: The scene, with its camera calibration.
        obj: A detected object.

    Returns:
        The width in millimetres.
    """
    (u, v), half = obj.pixel, _half_size(obj)
    left = pixel_to_table(scene, u - half, v)
    right = pixel_to_table(scene, u + half, v)
    return float(np.linalg.norm(right - left))


def _corners(obj: DetectedObject) -> np.ndarray:
    """Return the object's box as a polygon of four corners."""
    x1, y1, x2, y2 = obj.box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    np.float32).reshape(-1, 1, 2)


def gap_cm(scene: Scene, a: DetectedObject, b: DetectedObject) -> float:
    """Measure the gap between two objects' boxes on the table.

    For convex shapes the shortest gap runs from a corner of one of them,
    so the corners are enough.

    Args:
        scene: The scene, with its camera calibration.
        a: A detected object.
        b: Another detected object.

    Returns:
        The gap in centimetres, rounded to 0.1; 0.0 when the boxes touch
        or overlap.
    """
    box_a, box_b = _corners(a), _corners(b)
    best, at = np.inf, None
    for polygon, corners in ((box_b, box_a), (box_a, box_b)):
        for x, y in corners.reshape(-1, 2):
            distance = -cv2.pointPolygonTest(polygon, (float(x), float(y)),
                                             True)
            if distance < best:
                best, at = distance, (float(x), float(y))

    if best <= 0:
        return 0.0

    step = pixel_to_table(scene, at[0] + 1, at[1]) - pixel_to_table(scene, *at)
    return round(float(best * np.linalg.norm(step)) / 10, 1)


def add_gaps(scene: Scene, objects: list[DetectedObject]) -> None:
    """Give each object ``gaps``: ``{object id: cm}`` to every other object.

    Args:
        scene: The scene, with its camera calibration.
        objects: The detected objects; changed in place.
    """
    for obj in objects:
        obj.gaps = {other.id: gap_cm(scene, obj, other)
                    for other in objects if other is not obj}


def _box_mask(shape: tuple[int, ...], obj: DetectedObject) -> np.ndarray:
    """Return a boolean mask of the object's box in an image of ``shape``."""
    x1, y1, x2, y2 = obj.box
    inside = np.zeros(shape, bool)
    inside[y1:y2, x1:x2] = True
    return inside


def add_stacking(rgb: np.ndarray, objects: list[DetectedObject],
                 cell: Workcell) -> None:
    """Give each object ``on``: the ids of the objects it lies on top of.

    An object's own colour is the median Lab colour of the part of its box
    that no other box covers, or of its box's middle when all of it is
    covered. Where two boxes overlap enough, the object whose own colour is
    nearer the overlap's colour lies on top. When both are about equally
    near, neither is on the other.

    Args:
        rgb: The photo as an RGB array.
        objects: The detected objects; changed in place.
        cell: The workcell's settings.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    masks = [_box_mask(lab.shape[:2], obj) for obj in objects]
    own_colours = []
    for i, obj in enumerate(objects):
        covered = np.zeros_like(masks[i])
        for j, mask in enumerate(masks):
            if j != i:
                covered |= mask

        alone = masks[i] & ~covered
        if alone.sum() < 0.1 * masks[i].sum():
            x1, y1, x2, y2 = obj.box
            alone = np.zeros_like(alone)
            alone[(3 * y1 + y2) // 4:(y1 + 3 * y2) // 4,
                  (3 * x1 + x2) // 4:(x1 + 3 * x2) // 4] = True

        own_colours.append(np.median(lab[alone], axis=0))

    for obj in objects:
        obj.on = []

    for i, j in combinations(range(len(objects)), 2):
        overlap = masks[i] & masks[j]
        smaller = min(masks[i].sum(), masks[j].sum())
        if overlap.sum() < cell.stack_min_overlap * smaller:
            continue

        colour = np.median(lab[overlap], axis=0)
        distance_i = np.linalg.norm(colour - own_colours[i])
        distance_j = np.linalg.norm(colour - own_colours[j])
        if abs(distance_i - distance_j) < cell.stack_colour_margin:
            continue

        top, below = (i, j) if distance_i < distance_j else (j, i)
        objects[top].on.append(objects[below].id)


def colour_name(rgb: np.ndarray, obj: DetectedObject,
                cell: Workcell) -> str:
    """Name an object's colour from the median HSV around its centre.

    Args:
        rgb: The photo as an RGB array.
        obj: A detected object.
        cell: The workcell's settings.

    Returns:
        A colour name from the workcell's hues, "white" or "grey".
    """
    (u, v), r = obj.pixel, max(1, int(0.7 * _half_size(obj)))
    patch = cv2.cvtColor(rgb[v - r:v + r, u - r:u + r], cv2.COLOR_RGB2HSV)
    hue, saturation, brightness = (float(np.median(patch[..., i]))
                                   for i in range(3))
    if saturation < cell.pale_max_saturation:
        return "white" if brightness > 170 else "grey"

    return next((name for name, top in cell.hues if hue < top), "red")
