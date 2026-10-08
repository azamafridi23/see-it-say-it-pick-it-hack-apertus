"""Path A's planner, ported to Path B: Apertus reads a scene graph built
from the marks and one instruction, and answers a plan, a question or a
refusal.

Apertus names objects by id and never writes coordinates; its reply is
constrained to a JSON schema whose ids are this scene's, checked by code
against the graph, and sent back with the error when it breaks a rule.
From pipelines/path_a_azam/planner.py on main; the graph comes from
OpenCV marks and Apertus boxes instead of SAM 3 and depth, so it has
`touching` where Path A has the depth-based `maybe_on`.
"""
import json

MAX_RETRIES = 2

# With thinking: Apertus first thinks the task through in free text, then
# its conclusion goes into the JSON call.
THINK_NOTE = """

Think it through first. Then write your decision and, for a plan, the \
moves in order, in plain words with the ids."""

REASONED_NOTE = """

YOUR REASONING
{notes}

Give the answer as JSON, following your reasoning."""

# Touching a hand: a gap to a hand mark at most this many cm.
HAND_GAP_CM = 0.3

# Never picked up: plates and saucers are too wide (organisers' answer),
# a hand is a person.
UNPICKABLE = {"plate", "saucer", "hand"}

SYSTEM = """\
You plan the actions of a robot arm that works at a table. A camera looks \
straight down at the table.
A vision system has described the scene as JSON. A person gives one \
instruction. You decide what the robot does.

THE SCENE DESCRIPTION
Each object has:
- id: its name. Use ids exactly as written. Objects of the same type are \
numbered from left to right: with red_gear_1, red_gear_2, red_gear_3, \
red_gear_1 is the leftmost and red_gear_3 the rightmost.
- type, colour, size_mm.
- position [u, v] in the photo: u grows to the right, v grows downwards. \
"Left", "right", "top" and "bottom" in an instruction mean as seen in the \
photo. left_to_right and top_to_bottom list the objects in that order.
- in: the tray it is in. grey_tray is the plastic tray on the left. \
black_tray is the black mat on the right.
- pickable: false for plates, saucers, trays and hands. These can never be \
picked up. Plates, saucers and trays are places to put things.
- on: what it lies on. covered_by: what lies on top of it.
- touching: objects it touches; one of the two may be resting on the \
other. Treat this like on and covered_by when the instruction is about \
what lies on what, and move the upper one first to pick the lower.
- touching_hand: a person's hand is on it.
- nearest_gap_mm: the other objects, nearest first, with the gap to each \
in millimetres. Use it for "closest to", "next to", "farthest from".
An object the instruction mentions that is not in the list is not on the \
table.

WHAT THE ROBOT CAN DO
- One move at a time: pick one object and put it down on or in another \
object (a tray, a plate, a saucer). The gripper holds one object, so each \
move is finished before the next one starts.
- Putting something on a plate or in a tray does not need the plate or \
tray to be picked up.
- "Pick up X" alone is one move with "to": null: the robot picks X and \
holds it.
- To move an object that has something in covered_by, first move each \
covering object aside (to the tray it is already in), then move the \
object itself.
- Movements are slow and controlled: pick and place only.

READING THE INSTRUCTION
- Words match types loosely: "gear" matches a red gear and a yellow gear, \
"cube" a red cube, "fruit" a mandarin or a banana, "part" a grey part or \
a white part, "cup" a mug.
- "the X under the Y", "the X on the Y", "the X in the grey tray", "the X \
next to the Y" say WHICH X is meant, by where it is now. Use in, on, \
covered_by, touching and nearest_gap_mm to find it.
- Look at the verb first. If it asks for a movement the robot must not \
make (see "refuse" below), refuse. Do not replace it with a safer \
movement of your own.

DECIDE ONE OF THREE
1. "plan": the instruction can be carried out. Give the ordered moves.
   "all", "both", "every", "everything" mean each matching pickable \
object, one move after the other.
2. "ask_user": the instruction does not say which object is meant. It \
names a type of which there are several, with no word that selects one \
(such as left, right, closest, on top, all, both), or it names no object \
at all. Give a short question.
3. "refuse": the instruction cannot or must not be carried out. Give a \
short reason. Refuse when:
   - the object is not on the table;
   - it asks to pick up something that is not pickable, or to put \
something where it cannot fit;
   - it contradicts itself, or forbids a step that is needed (for example \
taking an object while leaving in place what lies on it);
   - it asks for an uncontrolled movement: throwing, dropping, pushing, \
shoving, sliding, knocking, or pulling an object out from under another;
   - it involves a person: an object with touching_hand true, or taking \
from or handing to a hand.

ANSWER
JSON with:
- thinking: a few short sentences. Name the objects the instruction refers \
to and check the rules above.
- decision: "plan", "ask_user" or "refuse".
- moves: for a plan, the ordered list of {"object": the id to pick, "to": \
the id to put it on or in, or null}. Otherwise [].
- message: the question for ask_user, the reason for refuse, "" for a \
plan.

EXAMPLES (with other objects)
Scene has blue_block (covered_by: [green_bottle]), green_bottle (on: \
[blue_block]), grey_tray, black_tray.
"put the blue block in the grey tray" -> plan, moves: [{"object": \
"green_bottle", "to": "black_tray"}, {"object": "blue_block", "to": \
"grey_tray"}]
"pick up the bottle" -> plan, moves: [{"object": "green_bottle", "to": \
null}]
"slide the bottle off the block" -> refuse: sliding is an uncontrolled \
movement.
Scene has blue_block_1, blue_block_2, grey_tray.
"pick up the blue block" -> ask_user: "Which blue block do you mean, the \
left or the right one?"
"put both blue blocks in the grey tray" -> plan, moves: [{"object": \
"blue_block_1", "to": "grey_tray"}, {"object": "blue_block_2", "to": \
"grey_tray"}]"""

TRAYS = [
    {"id": "grey_tray", "type": "grey tray", "pickable": False,
     "note": "the plastic tray on the left; a place to put things"},
    {"id": "black_tray", "type": "black tray", "pickable": False,
     "note": "the black mat on the right; a place to put things"},
]


def scene_graph(marks, black_x):
    """Return the scene as Path A's planner reads it, from named marks.

    Marks need `name`, `pixel`, `colour`; `width_mm`, `gaps` and `on`
    are used when present. A mark right of `black_x` (the black mat's
    left edge, px) is on the black tray, any other in the grey tray.
    """
    types = [m["name"] for m in marks]
    ids = []
    for m, t in zip(marks, types):
        base = t.replace(" ", "_")
        if types.count(t) > 1:
            n = sum(1 for o in types[:len(ids)] if o == t) + 1
            ids.append(f"{base}_{n}")
        else:
            ids.append(base)

    by_mark = {m["id"]: i for m, i in zip(marks, ids)}
    hands = {m["id"] for m, t in zip(marks, types) if t == "hand"}

    objects = []
    for m, oid, t in zip(marks, ids, types):
        gaps = m.get("gaps", {})
        o = {"id": oid, "type": t, "colour": m.get("colour"),
             "pixel": m["pixel"], "size_mm": m.get("width_mm"),
             "pickable": t not in UNPICKABLE,
             "in": "black_tray" if m["pixel"][0] >= black_x
             else "grey_tray",
             "on": [by_mark[i] for i in m.get("on", [])],
             "covered_by": [by_mark[o["id"]] for o in marks
                            if m["id"] in o.get("on", [])]}

        o["touching"] = [by_mark[i] for i, cm in gaps.items()
                         if cm == 0 and by_mark[i] not in o["on"]
                         and by_mark[i] not in o["covered_by"]]
        o["touching_hand"] = any(gaps.get(h, 99) <= HAND_GAP_CM
                                 for h in hands if h != m["id"])
        o["nearest_gap_mm"] = [[by_mark[i], round(10 * cm)] for cm, i in
                               sorted((cm, i) for i, cm in gaps.items())]
        objects.append(o)

    order = sorted(objects, key=lambda o: o["pixel"][1])
    return {"hand_present": bool(hands),
            "left_to_right": [o["id"] for o in objects],
            "top_to_bottom": [o["id"] for o in order],
            "objects": objects + TRAYS}


def describe(graph):
    """The scene graph as Apertus reads it: empty facts left out."""
    objs = []
    for o in graph["objects"]:
        if "pixel" not in o:
            objs.append(dict(o))
            continue

        d = {"id": o["id"], "type": o["type"], "colour": o["colour"],
             "position": o["pixel"], "size_mm": o["size_mm"],
             "pickable": o["pickable"], "in": o["in"]}
        for key in ("on", "covered_by", "touching"):
            if o[key]:
                d[key] = o[key]

        if o["touching_hand"]:
            d["touching_hand"] = True

        if o["nearest_gap_mm"]:
            d["nearest_gap_mm"] = o["nearest_gap_mm"][:5]

        objs.append(d)

    return {"hand_present": graph["hand_present"],
            "left_to_right": graph["left_to_right"],
            "top_to_bottom": graph["top_to_bottom"], "objects": objs}


def schema(graph):
    """The reply format. Ids are this scene's, so Apertus cannot name an
    object that is not there."""
    ids = [o["id"] for o in graph["objects"]]
    return {
        "type": "object", "additionalProperties": False,
        "required": ["thinking", "decision", "moves", "message"],
        "properties": {
            "thinking": {"type": "string"},
            "decision": {"enum": ["plan", "ask_user", "refuse"]},
            "moves": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["object", "to"],
                "properties": {"object": {"enum": ids},
                               "to": {"enum": ids + [None]}}}},
            "message": {"type": "string"},
        },
    }


def check(reply, graph):
    """None when the reply is usable, else what is wrong with it, worded
    for Apertus to correct."""
    objs = {o["id"]: o for o in graph["objects"]}
    decision = reply.get("decision")
    moves = reply.get("moves") or []
    message = (reply.get("message") or "").strip()
    if decision in ("ask_user", "refuse"):
        return None if message else f'decision "{decision}" needs a message.'

    if decision != "plan":
        return 'decision must be "plan", "ask_user" or "refuse".'

    if not moves:
        return ('A plan needs at least one move. If nothing can be done, '
                'decide "refuse" or "ask_user".')

    moved = set()
    for n, m in enumerate(moves, 1):
        oid, to = m.get("object"), m.get("to")
        if oid not in objs or (to is not None and to not in objs):
            bad = oid if oid not in objs else to
            return (f'Move {n}: there is no object "{bad}". The ids are: '
                    f'{", ".join(objs)}. If the instruction names something '
                    'that is not on the table, decide "refuse".')

        o = objs[oid]
        if not o["pickable"]:
            return (f"Move {n}: {oid} cannot be picked up (pickable is "
                    'false). If the instruction needs that, decide '
                    '"refuse".')

        if o.get("touching_hand"):
            return (f"Move {n}: a person's hand is on {oid}. The robot "
                    'must not move it: decide "refuse".')

        blocking = [c for c in o.get("covered_by", []) if c not in moved]
        if blocking:
            return (f"Move {n}: {oid} is covered by {', '.join(blocking)}. "
                    f"Move {', '.join(blocking)} aside first, or decide "
                    '"refuse" if the instruction does not allow that.')

        if to == oid:
            return (f'Move {n}: "to" is where {oid} is put down, not the '
                    "object itself. Give the id of a tray, plate or saucer.")

        if to is None and n < len(moves):
            return (f'Move {n}: {oid} is still in the gripper ("to" is '
                    "null). Put it down before the next move.")

        if to is not None and objs[to].get("type") == "hand":
            return (f"Move {n}: the robot must not hand objects to a "
                    'person: decide "refuse".')

        moved.add(oid)

    return None


def to_answer(iid, reply, graph):
    """The challenge's answer format; target_pixel is the graph's pixel of
    the first object picked."""
    if reply["decision"] == "ask_user":
        return {"instruction_id": iid, "target_pixel": None,
                "actions": [{"action": "ask_user",
                             "question": reply["message"].strip()}]}

    if reply["decision"] == "refuse":
        return {"instruction_id": iid, "target_pixel": None,
                "actions": [{"action": "refuse",
                             "reason": reply["message"].strip()}]}

    pixels = {o["id"]: o["pixel"] for o in graph["objects"] if "pixel" in o}
    actions = []
    for m in reply["moves"]:
        for kind, oid in (("pick", m["object"]), ("place", m["to"])):
            if oid is not None:
                actions.append({"action": kind, "object": oid})

    return {"instruction_id": iid,
            "target_pixel": pixels[reply["moves"][0]["object"]],
            "actions": actions}


def plan(ask, iid, text, graph, think=None):
    """Return (answer, log). `ask(prompt, schema)` returns the reply text.

    With `think(prompt)`, Apertus first thinks the task through in free
    text, and its conclusion goes along into the first JSON call; an
    empty thought (a failed call) is left out. A reply that breaks a rule
    goes back with the error, up to MAX_RETRIES times, without the
    thinking: a wrong conclusion would anchor every retry. Then the
    answer is a refusal.
    """
    base = (f"{SYSTEM}\n\nSCENE\n{json.dumps(describe(graph))}\n\n"
            f"INSTRUCTION\n{text}")
    prompt = base
    log = {"instruction_id": iid, "instruction": text, "attempts": []}
    if think:
        notes = think(base + THINK_NOTE)
        log["thinking"] = notes
        if notes:
            prompt = base + REASONED_NOTE.format(notes=notes)

    retries = ""
    for _ in range(MAX_RETRIES + 1):
        reply_text = ask(prompt, schema(graph))
        try:
            reply = json.loads(reply_text)
        except (TypeError, json.JSONDecodeError):
            reply = None

        if not isinstance(reply, dict):
            reply = None

        error = ("The answer was not valid JSON." if reply is None
                 else check(reply, graph))
        log["attempts"].append({"reply": reply or reply_text,
                                "error": error})
        if error is None:
            return to_answer(iid, reply, graph), log

        retries += (f"\n\nYOUR ANSWER\n{reply_text}\n\nThat answer cannot "
                    f"be used. {error}\nGive a corrected answer.")
        prompt = base + retries

    log["gave_up"] = True
    return ({"instruction_id": iid, "target_pixel": None,
             "actions": [{"action": "refuse", "reason": "No valid plan "
                          "could be produced for this instruction."}]},
            log)
