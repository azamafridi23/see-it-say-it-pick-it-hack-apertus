"""Response schemas and local validation for the three Apertus agents."""
from jsonschema import Draft202012Validator


SAFETY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["thinking", "allowed", "reason"],
    "properties": {"thinking": {"type": "string"},
                   "allowed": {"type": "boolean"},
                   "reason": {"type": "string"}},
}


def reference_schema(graph: dict) -> dict:
    """Build the reference agent's reply schema for one scene.

    Args:
        graph: The scene graph; its object ids are the only allowed
            objects and destinations.

    Returns:
        A JSON schema for the reply.
    """
    ids = [obj["id"] for obj in graph["objects"]]
    return {
        "type": "object", "additionalProperties": False,
        "required": ["thinking", "status", "tasks", "message"],
        "properties": {
            "thinking": {"type": "string"},
            "status": {"enum": ["resolved", "ambiguous", "not_on_table"]},
            "tasks": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["objects", "to"],
                "properties": {
                    "objects": {"type": "array", "items": {"enum": ids}},
                    "to": {"enum": ids + [None]}}}},
            "message": {"type": "string"},
        },
    }


def constraint_schema(moved: list[str]) -> dict:
    """Build the constraint agent's reply schema.

    Args:
        moved: The ids of the objects the plan would move; a restriction
            may name only these.

    Returns:
        A JSON schema for the reply.
    """
    return {
        "type": "object", "additionalProperties": False,
        "required": ["thinking", "status", "restrictions", "message"],
        "properties": {
            "thinking": {"type": "string"},
            "status": {"enum": ["clear", "conflict", "uncertain"]},
            "restrictions": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["object", "quote"],
                "properties": {"object": {"enum": moved},
                               "quote": {"type": "string"}}}},
            "message": {"type": "string"},
        },
    }


def _has_empty_task(tasks: list[dict]) -> bool:
    """Tell whether some task names no object."""
    return any(not task["objects"] for task in tasks)


def _repeats_an_object(tasks: list[dict]) -> bool:
    """Tell whether some task names the same object twice."""
    for task in tasks:
        if len(set(task["objects"])) != len(task["objects"]):
            return True

    return False


def validate_reply(role: str, reply, schema: dict,
                   instruction: str) -> str | None:
    """Check a reply against its schema and the rules a schema cannot hold.

    Constrained decoding enforces the schema on strict servers only, and it
    cannot enforce rules that link fields, such as "a resolved reference
    has tasks and no message".

    Args:
        role: ``"safety"``, ``"reference"`` or ``"constraint"``.
        reply: The parsed reply.
        schema: The schema the reply must follow.
        instruction: The instruction, to check that quotes come from it.

    Returns:
        A message describing the first problem, or None if the reply is
        valid.
    """
    errors = list(Draft202012Validator(schema).iter_errors(reply))
    if errors:
        error = errors[0]
        path = ".".join(map(str, error.absolute_path)) or "reply"
        return f"{path}: {error.message}"

    if not reply["thinking"].strip():
        return "thinking needs a short statement of the decisive evidence."

    if role == "safety":
        if reply["allowed"] == bool(reply["reason"].strip()):
            return ("Allowed movements need an empty reason; "
                    "prohibited movements need a reason.")

    elif role == "reference":
        tasks, message = reply["tasks"], reply["message"].strip()
        if reply["status"] != "resolved":
            if tasks or not message:
                return "Unresolved references need a message and no tasks."

        elif not tasks or message or _has_empty_task(tasks):
            return ("Resolved references need nonempty tasks and objects, "
                    "and an empty message.")
        elif _repeats_an_object(tasks):
            return "Do not repeat an object within a task."

    elif role == "constraint":
        status = reply["status"]
        restrictions = reply["restrictions"]
        message = reply["message"].strip()
        if status == "clear" and (restrictions or message):
            return ("A clear constraint check needs no restrictions and an "
                    "empty message.")

        if status == "uncertain" and (restrictions or not message):
            return ("An uncertain constraint check needs a specific question "
                    "and no restrictions.")

        if status == "conflict" and (not restrictions or message):
            return ("A conflict needs object/quote restrictions and an empty "
                    "message.")

        for restriction in restrictions:
            quote = restriction["quote"].strip().strip('"\'').strip()
            if not quote or quote.lower() not in instruction.lower():
                return ("Every restriction must quote a nonempty exact "
                        "excerpt from the instruction.")

    return None
