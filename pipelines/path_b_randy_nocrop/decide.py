"""The older planning stage (`--no-planner`): Apertus sees the photo
with numbered marks and decides plan, ask or refuse, voting over
several samples. Path A's planner (planner.py) replaced it."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2

import paths  # noqa: F401
from apertus import ask
from baseline import parse
from measure import TOO_WIDE_MM


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


def named_inventory(marks):
    """The marks with their names and measured colours, one line each."""
    lines = [f"mark {m['id']}: {m['name']} (measured {m['colour']})"
             for m in marks]
    return "\n".join(lines) or "(no marks)"


def _no_plan(iid, action, field, text):
    return {"instruction_id": iid, "target_pixel": None,
            "actions": [{"action": action, field: text}]}


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
