"""Planning by Apertus on a scene graph, with a checker in code.

Apertus reads the scene graph and one instruction and answers with a
plan, a question or a refusal. It names objects by id and never writes
coordinates: its reply is constrained to a JSON schema whose ids are this
scene's, then checked against the graph, and sent back with the error
when it breaks a rule.

Ported from Path A (``pipelines/path_a_azam/planner.py`` on main). Path B
builds the graph from Apertus's boxes instead of SAM 3 and depth, so it
has ``touching`` where Path A has the depth-based ``maybe_on``.
"""
import json
from typing import Callable

from workcell import DetectedObject, Workcell


# How often an answer that breaks a rule goes back to Apertus.
MAX_RETRIES = 2

# With thinking, Apertus first thinks the task through in free text
# (THINK_NOTE); its conclusion then goes into the JSON call (REASONED_NOTE).
THINK_NOTE = """ Think it through first. Then write your decision and, \
for a plan, the moves in order, in plain words with the ids."""

REASONED_NOTE = """

YOUR REASONING
{notes}

Give the answer as JSON, following your reasoning."""

# The planner's instructions, as in Path A, with ``touching`` in place of
# its depth-based ``maybe_on``. The examples use invented objects.
PLANNER_PROMPT = """\
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

# The two trays, added to every graph as places to put things.
TRAYS = [
    {"id": "grey_tray", "type": "grey tray", "pickable": False,
     "note": "the plastic tray on the left; a place to put things"},
    {"id": "black_tray", "type": "black tray", "pickable": False,
     "note": "the black mat on the right; a place to put things"},
]


def scene_graph(objects: list[DetectedObject], black_mat_left: int,
                cell: Workcell) -> dict:
    """Build the scene graph the planner reads, from the measured objects.

    Each object gets an id from its kind, numbered left to right when
    there are several (``red_gear_1``, ``red_gear_2``). An object right of
    ``black_mat_left`` is on the black tray, any other in the grey tray.
    The two trays are added as places to put things.

    Args:
        objects: The detected objects, with their measured facts.
        black_mat_left: The black mat's left edge, in pixels.
        cell: The workcell's settings: which kinds are never picked up,
            and how near a hand counts as touching it.

    Returns:
        ``{"hand_present", "left_to_right", "top_to_bottom", "objects"}``,
        the objects including the two trays.
    """
    kinds = [obj.kind for obj in objects]
    graph_ids = []
    for kind in kinds:
        base = kind.replace(" ", "_")
        if kinds.count(kind) > 1:
            number = kinds[:len(graph_ids)].count(kind) + 1
            graph_ids.append(f"{base}_{number}")
        else:
            graph_ids.append(base)

    id_of = {obj.id: graph_id for obj, graph_id in zip(objects, graph_ids)}
    hands = {obj.id for obj in objects if obj.kind == "hand"}

    nodes = []
    for obj, graph_id in zip(objects, graph_ids):
        gaps = obj.gaps
        node = {"id": graph_id, "type": obj.kind,
                "colour": obj.colour, "pixel": obj.pixel,
                "size_mm": obj.width_mm,
                "pickable": obj.kind not in cell.unpickable,
                "in": ("black_tray" if obj.pixel[0] >= black_mat_left
                       else "grey_tray"),
                "on": [id_of[i] for i in obj.on],
                "covered_by": [id_of[other.id] for other in objects
                               if obj.id in other.on]}

        node["touching"] = [id_of[i] for i, cm in gaps.items()
                            if cm == 0 and id_of[i] not in node["on"]
                            and id_of[i] not in node["covered_by"]]
        node["touching_hand"] = any(gaps.get(hand, 99) <= cell.hand_gap_cm
                                    for hand in hands if hand != obj.id)
        node["nearest_gap_mm"] = [
            [id_of[i], round(10 * cm)]
            for cm, i in sorted((cm, i) for i, cm in gaps.items())]
        nodes.append(node)

    top_down = sorted(nodes, key=lambda node: node["pixel"][1])
    return {"hand_present": bool(hands),
            "left_to_right": [node["id"] for node in nodes],
            "top_to_bottom": [node["id"] for node in top_down],
            "objects": nodes + TRAYS}


def prompt_view(graph: dict) -> dict:
    """Shorten the scene graph for the prompt.

    Pixels become ``position``, empty facts are left out, and each object
    keeps its 5 nearest gaps.

    Args:
        graph: The scene graph, see ``scene_graph()``.

    Returns:
        The graph as Apertus reads it.
    """
    entries = []
    for node in graph["objects"]:
        if "pixel" not in node:
            entries.append(dict(node))
            continue

        entry = {"id": node["id"], "type": node["type"],
                 "colour": node["colour"], "position": node["pixel"],
                 "size_mm": node["size_mm"], "pickable": node["pickable"],
                 "in": node["in"]}
        for key in ("on", "covered_by", "touching"):
            if node[key]:
                entry[key] = node[key]

        if node["touching_hand"]:
            entry["touching_hand"] = True

        if node["nearest_gap_mm"]:
            entry["nearest_gap_mm"] = node["nearest_gap_mm"][:5]

        entries.append(entry)

    return {"hand_present": graph["hand_present"],
            "left_to_right": graph["left_to_right"],
            "top_to_bottom": graph["top_to_bottom"], "objects": entries}


def reply_schema(graph: dict) -> dict:
    """Build the JSON schema of the planner's reply.

    Ids are limited to this scene's, so Apertus cannot name an object that
    is not there.

    Args:
        graph: The scene graph, see ``scene_graph()``.

    Returns:
        The JSON schema.
    """
    ids = [node["id"] for node in graph["objects"]]
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


def rule_error(reply: dict, graph: dict) -> str | None:
    """Check a reply against the graph and the robot's rules.

    Args:
        reply: The planner's parsed reply.
        graph: The scene graph, see ``scene_graph()``.

    Returns:
        None when the reply is usable, else what is wrong with it, worded
        for Apertus to correct.
    """
    by_id = {node["id"]: node for node in graph["objects"]}
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
    for number, move in enumerate(moves, 1):
        object_id, to = move.get("object"), move.get("to")
        if object_id not in by_id or (to is not None and to not in by_id):
            unknown = object_id if object_id not in by_id else to
            return (f'Move {number}: there is no object "{unknown}". The '
                    f'ids are: {", ".join(by_id)}. If the instruction names '
                    'something that is not on the table, decide "refuse".')

        node = by_id[object_id]
        if not node["pickable"]:
            return (f"Move {number}: {object_id} cannot be picked up "
                    '(pickable is false). If the instruction needs that, '
                    'decide "refuse".')

        if node.get("touching_hand"):
            return (f"Move {number}: a person's hand is on {object_id}. The "
                    'robot must not move it: decide "refuse".')

        blocking = [c for c in node.get("covered_by", []) if c not in moved]
        if blocking:
            return (f"Move {number}: {object_id} is covered by "
                    f"{', '.join(blocking)}. Move {', '.join(blocking)} "
                    'aside first, or decide "refuse" if the instruction '
                    "does not allow that.")

        if to == object_id:
            return (f'Move {number}: "to" is where {object_id} is put down, '
                    "not the object itself. Give the id of a tray, plate or "
                    "saucer.")

        if to is None and number < len(moves):
            return (f'Move {number}: {object_id} is still in the gripper '
                    '("to" is null). Put it down before the next move.')

        if to is not None and by_id[to].get("type") == "hand":
            return (f"Move {number}: the robot must not hand objects to a "
                    'person: decide "refuse".')

        moved.add(object_id)

    return None


def to_result(instruction_id: str, reply: dict, graph: dict) -> dict:
    """Turn a checked reply into the challenge's answer format.

    Each move becomes a pick and, unless the object is held, a place. The
    target pixel is the pixel of the first object picked.

    Args:
        instruction_id: The instruction's id.
        reply: A reply that passed ``rule_error()``.
        graph: The scene graph, see ``scene_graph()``.

    Returns:
        The answer.
    """
    if reply["decision"] == "ask_user":
        return {"instruction_id": instruction_id, "target_pixel": None,
                "actions": [{"action": "ask_user",
                             "question": reply["message"].strip()}]}

    if reply["decision"] == "refuse":
        return {"instruction_id": instruction_id, "target_pixel": None,
                "actions": [{"action": "refuse",
                             "reason": reply["message"].strip()}]}

    pixels = {node["id"]: node["pixel"] for node in graph["objects"]
              if "pixel" in node}
    actions = []
    for move in reply["moves"]:
        for action, object_id in (("pick", move["object"]),
                                  ("place", move["to"])):
            if object_id is not None:
                actions.append({"action": action, "object": object_id})

    return {"instruction_id": instruction_id,
            "target_pixel": pixels[reply["moves"][0]["object"]],
            "actions": actions}


def plan(ask_json: Callable[[str, dict], str], instruction_id: str,
         text: str, graph: dict,
         ask_thinking: Callable[[str], str] | None = None
         ) -> tuple[dict, dict]:
    """Plan one instruction, with retries when the checker rejects a reply.

    A reply that breaks a rule goes back to Apertus with the error, up to
    MAX_RETRIES times; after that the answer is a refusal. With
    ``ask_thinking``, Apertus first thinks the task through in free text,
    and its conclusion goes into the first attempt only: a wrong
    conclusion would anchor every retry.

    Args:
        ask_json: ``ask_json(prompt, schema)`` returns Apertus's reply
            text, constrained to ``schema``.
        instruction_id: The instruction's id.
        text: The instruction.
        graph: The scene graph, see ``scene_graph()``.
        ask_thinking: ``ask_thinking(prompt)`` returns Apertus's
            conclusion after thinking, or ``""`` when the call failed;
            None to skip thinking.

    Returns:
        ``(result, log)``: the result in the challenge's format, and a
        record of every attempt.
    """
    base = (f"{PLANNER_PROMPT}\n\nSCENE\n{json.dumps(prompt_view(graph))}"
            f"\n\nINSTRUCTION\n{text}")
    prompt = base
    log = {"instruction_id": instruction_id, "instruction": text,
           "attempts": []}
    if ask_thinking:
        notes = ask_thinking(base + THINK_NOTE)
        log["thinking"] = notes
        if notes:
            prompt = base + REASONED_NOTE.format(notes=notes)

    retries = ""
    for _ in range(MAX_RETRIES + 1):
        reply_text = ask_json(prompt, reply_schema(graph))
        try:
            reply = json.loads(reply_text)
        except (TypeError, json.JSONDecodeError):
            reply = None

        if not isinstance(reply, dict):
            reply = None

        error = ("The answer was not valid JSON." if reply is None
                 else rule_error(reply, graph))
        log["attempts"].append({"reply": reply or reply_text,
                                "error": error})
        if error is None:
            return to_result(instruction_id, reply, graph), log

        retries += (f"\n\nYOUR ANSWER\n{reply_text}\n\nThat answer cannot "
                    f"be used. {error}\nGive a corrected answer.")
        prompt = base + retries

    log["gave_up"] = True
    refused = {"instruction_id": instruction_id, "target_pixel": None,
               "actions": [{"action": "refuse", "reason": "No valid plan "
                            "could be produced for this instruction."}]}
    return refused, log
