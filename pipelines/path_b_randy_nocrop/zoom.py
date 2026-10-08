"""Decide mode: grid zoom, for a first pick without a mark. Apertus
names the cell of a 3 x 3 grid that holds the object, three times."""
import cv2

import paths  # noqa: F401
from apertus import ask
from baseline import parse


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
