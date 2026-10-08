"""The older naming (`--no-boxes`): Apertus names each OpenCV mark from
its own close-up crop, one call per mark."""
from concurrent.futures import ThreadPoolExecutor

import cv2

import paths  # noqa: F401
from apertus import ask
from baseline import parse
from measure import width_mm


# The close-up is a square CLOSEUP_RADII inscribed radii each side of the
# mark, at least CLOSEUP_MIN px, so a handle or a small part still shows
# what is around it.
CLOSEUP_RADII = 3
CLOSEUP_MIN = 300

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
