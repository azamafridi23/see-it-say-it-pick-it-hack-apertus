"""Safety pre-check by Apertus, on the instruction alone.

Before any planning, Apertus reads the sentence without the photo and
refuses unsafe or self-contradicting instructions, so that a "which one?"
question cannot hide an unsafe verb.
"""
import paths  # noqa: F401
from apertus import ask
from baseline import parse


# The examples are invented: no dev scene, no dev instruction.
PRECHECK_PROMPT = """\
A robot arm above a table gets this instruction: "{text}"

Check the instruction itself, before anyone looks at the table. Answer \
"refuse" if it:
- asks for an uncontrolled movement: throwing, tossing, dropping, \
pushing, shoving, sliding, knocking, or pulling an object out from under \
another;
- asks to take an object out of a person's hand;
- needs a plate, a saucer or a tray picked up: they are only places to \
put things, and plates and saucers are too wide for the gripper;
- contradicts itself: asks for two things that cannot both be done.
Otherwise answer "go on", also when the instruction is vague or names \
something that may not be on the table: the photo is checked later.

Examples:
"toss the bottle into the box" -> refuse: tossing is uncontrolled.
"pull the board out from under the box" -> refuse: uncontrolled.
"take the bottle from my hand" -> refuse: it is in a person's hand.
"put the plate on the box" -> refuse: a plate cannot be picked up.
"put the bottle in the box and leave it where it is" -> refuse: it \
cannot both move and stay.
"put the bottle on the plate" -> go on.
"put the box under the bottle in the tray" -> go on: "under" only says \
which box.
"pick up that one" -> go on: vague, a question comes later.
Give the reason in one sentence, then the decision."""

# The decision comes last: after an enum field, Apertus pads with
# whitespace instead of writing the next field.
PRECHECK_SCHEMA = {
    "type": "object",
    "required": ["reason", "decision"],
    "properties": {
        "reason": {"type": "string"},
        "decision": {"enum": ["refuse", "go on"]},
    },
}


def precheck(model, text):
    """Check whether an instruction must be refused on its words alone.

    A failed call lets the instruction go on: the planner still checks
    safety.

    Args:
        model: A model function, see ``apertus.server_model()``.
        text: The instruction.

    Returns:
        Apertus's reason for refusing, or None to go on.
    """
    reply = parse(ask(model, PRECHECK_PROMPT.format(text=text), [],
                      PRECHECK_SCHEMA)) or {}
    if reply.get("decision") != "refuse":
        return None

    return reply.get("reason") or "I cannot do this safely."


def refusal(instruction_id, reason):
    """Build the answer for a refused instruction.

    Args:
        instruction_id: The instruction's id.
        reason: Why the robot refuses.

    Returns:
        The answer in the challenge's format.
    """
    return {"instruction_id": instruction_id, "target_pixel": None,
            "actions": [{"action": "refuse", "reason": reason}]}
