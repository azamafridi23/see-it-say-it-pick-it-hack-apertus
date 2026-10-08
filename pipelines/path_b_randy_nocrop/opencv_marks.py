"""The older detector (`--opencv-marks`): OpenCV colour blobs as the
objects. Kept for comparison; Apertus's boxes replaced it as the
detector.
"""
import cv2
import numpy as np

from measure import PALE_SAT, PALE_VAL


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
