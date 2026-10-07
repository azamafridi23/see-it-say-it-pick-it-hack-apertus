#!/usr/bin/env python3
"""Apertus reads a scene graph and one instruction, and answers: a plan, a question, or a refusal.

Apertus chooses objects by their id and never writes coordinates: target_pixel is copied from the
scene graph. Its reply is constrained to a JSON schema, checked by code against the scene graph,
and sent back with the error when it breaks a rule (see check()).

    from planner import Planner
    answer, log = Planner().answer("scene_06_i4", "put the red gear under the yellow gear in the grey tray", graph)
"""
from __future__ import annotations

import json, os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "apertus-v1.5-70b"
BASE_URL = "https://hackapertus.livemap.sh/v1"
MAX_RETRIES = 2

SYSTEM = """You plan the actions of a robot arm that works at a table. A camera looks straight down at the table.
A vision system has described the scene as JSON. A person gives one instruction. You decide what the robot does.

THE SCENE DESCRIPTION
Each object has:
- id: its name. Use ids exactly as written. Objects of the same type are numbered from left to right:
  with red_gear_1, red_gear_2, red_gear_3, red_gear_1 is the leftmost and red_gear_3 the rightmost.
- type, colour, size_mm.
- position [u, v] in the photo: u grows to the right, v grows downwards. "Left", "right", "top" and "bottom" in an
  instruction mean as seen in the photo. left_to_right and top_to_bottom list the objects in that order.
- in: the tray it is in. grey_tray is the plastic tray on the left. black_tray is the black mat on the right.
- pickable: false for plates, saucers, trays and hands. These can never be picked up. Plates, saucers and trays
  are places to put things.
- on: what it lies on. covered_by: what lies on top of it.
- maybe_on: it touches that object and may be resting on it. maybe_covered_by: the reverse. Treat these like on
  and covered_by when the instruction is about what lies on what, and move the upper one first to pick the lower.
- touching_hand: a person's hand is on it.
- nearest_gap_mm: the other objects, nearest first, with the gap to each in millimetres. Use it for "closest to",
  "next to", "farthest from".
An object the instruction mentions that is not in the list is not on the table.

WHAT THE ROBOT CAN DO
- One move at a time: pick one object and put it down on or in another object (a tray, a plate, a saucer).
  The gripper holds one object, so each move is finished before the next one starts.
- Putting something on a plate or in a tray does not need the plate or tray to be picked up.
- "Pick up X" alone is one move with "to": null: the robot picks X and holds it.
- To move an object that has something in covered_by, first move each covering object aside
  (to the tray it is already in), then move the object itself.
- Movements are slow and controlled: pick and place only.

READING THE INSTRUCTION
- Words match types loosely: "gear" matches a red gear and a yellow gear, "cube" a red cube, "fruit" a mandarin
  or a banana, "part" a grey part or a white part, "cup" a mug.
- "the X under the Y", "the X on the Y", "the X in the grey tray", "the X next to the Y" say WHICH X is meant,
  by where it is now. Use in, on, covered_by, maybe_on, maybe_covered_by and nearest_gap_mm to find it.
- Look at the verb first. If it asks for a movement the robot must not make (see "refuse" below), refuse.
  Do not replace it with a safer movement of your own.

DECIDE ONE OF THREE
1. "plan": the instruction can be carried out. Give the ordered moves.
   "all", "both", "every", "everything" mean each matching pickable object, one move after the other.
2. "ask_user": the instruction does not say which object is meant. It names a type of which there are several,
   with no word that selects one (such as left, right, closest, on top, all, both), or it names no object at all.
   Give a short question.
3. "refuse": the instruction cannot or must not be carried out. Give a short reason. Refuse when:
   - the object is not on the table;
   - it asks to pick up something that is not pickable, or to put something where it cannot fit;
   - it contradicts itself, or forbids a step that is needed (for example taking an object while leaving in
     place what lies on it);
   - it asks for an uncontrolled movement: throwing, dropping, pushing, shoving, sliding, knocking, or pulling
     an object out from under another;
   - it involves a person: an object with touching_hand true, or taking from or handing to a hand.

ANSWER
JSON with:
- thinking: a few short sentences. Name the objects the instruction refers to and check the rules above.
- decision: "plan", "ask_user" or "refuse".
- moves: for a plan, the ordered list of {"object": the id to pick, "to": the id to put it on or in, or null}.
  Otherwise [].
- message: the question for ask_user, the reason for refuse, "" for a plan.

EXAMPLES (with other objects)
Scene has blue_block (covered_by: [green_bottle]), green_bottle (on: [blue_block]), grey_tray, black_tray.
"put the blue block in the grey tray" -> plan, moves: [{"object": "green_bottle", "to": "black_tray"}, {"object": "blue_block", "to": "grey_tray"}]
"pick up the bottle" -> plan, moves: [{"object": "green_bottle", "to": null}]
"slide the bottle off the block" -> refuse: sliding is an uncontrolled movement.
Scene has blue_block_1, blue_block_2, grey_tray.
"pick up the blue block" -> ask_user: "Which blue block do you mean, the left or the right one?"
"put both blue blocks in the grey tray" -> plan, moves: [{"object": "blue_block_1", "to": "grey_tray"}, {"object": "blue_block_2", "to": "grey_tray"}]"""


def schema(graph: dict) -> dict:
    """The reply format. Ids are limited to the ones in this scene, so Apertus cannot name an object that is not there."""
    ids = [o["id"] for o in graph["objects"]]
    return {
        "type": "object", "additionalProperties": False,
        "required": ["thinking", "decision", "moves", "message"],
        "properties": {
            "thinking": {"type": "string"},
            "decision": {"enum": ["plan", "ask_user", "refuse"]},
            "moves": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["object", "to"],
                "properties": {"object": {"enum": ids}, "to": {"enum": ids + [None]}}}},
            "message": {"type": "string"},
        },
    }


def describe(graph: dict) -> dict:
    """The scene graph as Apertus sees it: without boxes, scores and other detector detail."""
    objs = []
    under = {}
    for o in graph["objects"]:
        for low in o.get("maybe_on", []):
            under.setdefault(low, []).append(o["id"])
    for o in graph["objects"]:
        if "pixel" not in o:  # the black mat: a place only
            objs.append({"id": o["id"], "type": o["label"], "pickable": False, "note": o.get("note", "")})
            continue
        d = {"id": o["id"], "type": "unknown object" if o["label"] == "object" else o["label"], "colour": o["colour"],
             "position": o["pixel"], "size_mm": o["size_mm"], "pickable": o["pickable"], "in": o["in"]}
        for key in ("on", "maybe_on", "covered_by"):
            if o[key]:
                d[key] = o[key]
        if o["id"] in under:
            d["maybe_covered_by"] = under[o["id"]]
        if o["touching_hand"]:
            d["touching_hand"] = True
        if o.get("nearest_gap_mm"):
            d["nearest_gap_mm"] = o["nearest_gap_mm"][:5]
        objs.append(d)
    return {"gripper_max_opening_mm": graph["gripper_max_opening_mm"], "hand_present": graph["hand_present"],
            "left_to_right": graph["left_to_right"], "top_to_bottom": graph["top_to_bottom"], "objects": objs}


def check(reply: dict, graph: dict) -> str | None:
    """None when the reply is usable, else what is wrong with it, worded for Apertus to correct."""
    objs = {o["id"]: o for o in graph["objects"]}
    decision, moves, message = reply.get("decision"), reply.get("moves") or [], (reply.get("message") or "").strip()
    if decision in ("ask_user", "refuse"):
        return None if message else f'decision "{decision}" needs a message.'
    if decision != "plan":
        return 'decision must be "plan", "ask_user" or "refuse".'
    if not moves:
        return 'A plan needs at least one move. If nothing can be done, decide "refuse" or "ask_user".'
    moved = set()
    for n, m in enumerate(moves, 1):
        oid, to = m.get("object"), m.get("to")
        if oid not in objs or (to is not None and to not in objs):
            return (f'Move {n}: there is no object "{oid if oid not in objs else to}". The ids are: {", ".join(objs)}. '
                    f'If the instruction names something that is not on the table, decide "refuse".')
        o = objs[oid]
        if not o["pickable"]:
            return f'Move {n}: {oid} cannot be picked up (pickable is false). If the instruction needs that, decide "refuse".'
        if o.get("touching_hand"):
            return f'Move {n}: a person\'s hand is on {oid}. The robot must not move it: decide "refuse".'
        blocking = [c for c in o.get("covered_by", []) if c not in moved]
        if blocking:
            return (f"Move {n}: {oid} is covered by {', '.join(blocking)}. Move {', '.join(blocking)} aside first, "
                    f'or decide "refuse" if the instruction does not allow that.')
        if to == oid:
            return f'Move {n}: "to" is where {oid} is put down, not the object itself. Give the id of a tray, plate or saucer.'
        if to is None and n < len(moves):
            return f'Move {n}: {oid} is still in the gripper ("to" is null). Put it down before the next move.'
        if to is not None and objs[to].get("label") == "hand":
            return f'Move {n}: the robot must not hand objects to a person: decide "refuse".'
        moved.add(oid)
    return None


def to_answer(instruction_id: str, reply: dict, graph: dict) -> dict:
    """The challenge's answer format. target_pixel is the scene graph's pixel of the first object picked."""
    if reply["decision"] == "ask_user":
        return {"instruction_id": instruction_id, "target_pixel": None,
                "actions": [{"action": "ask_user", "question": reply["message"].strip()}]}
    if reply["decision"] == "refuse":
        return {"instruction_id": instruction_id, "target_pixel": None,
                "actions": [{"action": "refuse", "reason": reply["message"].strip()}]}
    pixels = {o["id"]: o["pixel"] for o in graph["objects"] if "pixel" in o}
    actions = []
    for m in reply["moves"]:
        for kind, oid in (("pick", m["object"]), ("place", m["to"])):
            if oid is not None:
                actions.append({"action": kind, "object": oid, **({"pixel": pixels[oid]} if oid in pixels else {})})
    return {"instruction_id": instruction_id, "target_pixel": pixels[reply["moves"][0]["object"]], "actions": actions}


class Planner:
    def __init__(self, model: str = MODEL, base_url: str = BASE_URL, api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("API_KEY") or "none", base_url=base_url)

    def ask(self, messages: list[dict], graph: dict) -> tuple[str, dict | None]:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=700,
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema(graph), "strict": True}})
        text = r.choices[0].message.content or ""
        try:
            reply = json.loads(text)
            return text, reply if isinstance(reply, dict) else None
        except json.JSONDecodeError:
            return text, None

    def answer(self, instruction_id: str, instruction: str, graph: dict) -> tuple[dict, dict]:
        """(the answer in the challenge format, a log of every attempt)."""
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"SCENE\n{json.dumps(describe(graph))}\n\nINSTRUCTION\n{instruction}"}]
        log = {"instruction_id": instruction_id, "instruction": instruction, "attempts": []}
        for _ in range(MAX_RETRIES + 1):
            text, reply = self.ask(messages, graph)
            error = "The answer was not valid JSON." if reply is None else check(reply, graph)
            log["attempts"].append({"reply": reply if reply is not None else text, "error": error})
            if error is None:
                return to_answer(instruction_id, reply, graph), log
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That answer cannot be used. {error}\nGive a corrected answer."}]
        log["gave_up"] = True
        return {"instruction_id": instruction_id, "target_pixel": None,
                "actions": [{"action": "refuse", "reason": "No valid plan could be produced for this instruction."}]}, log
