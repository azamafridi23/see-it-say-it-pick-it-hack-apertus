"""Response schemas and local validation for the three Apertus agents."""
from jsonschema import Draft202012Validator


SAFETY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["thinking", "allowed", "reason"],
    "properties": {"thinking": {"type": "string"}, "allowed": {"type": "boolean"}, "reason": {"type": "string"}},
}


def reference_schema(graph: dict) -> dict:
    ids = [o["id"] for o in graph["objects"]]
    return {
        "type": "object", "additionalProperties": False, "required": ["thinking", "status", "tasks", "message"],
        "properties": {
            "thinking": {"type": "string"},
            "status": {"enum": ["resolved", "ambiguous", "not_on_table"]},
            "tasks": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["objects", "to"],
                "properties": {"objects": {"type": "array", "items": {"enum": ids}}, "to": {"enum": ids + [None]}}}},
            "message": {"type": "string"},
        },
    }


def constraint_schema(moved: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["thinking", "status", "restrictions", "message"],
        "properties": {
            "thinking": {"type": "string"},
            "status": {"enum": ["clear", "conflict", "uncertain"]},
            "restrictions": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["object", "quote"],
                "properties": {"object": {"enum": moved}, "quote": {"type": "string"}}}},
            "message": {"type": "string"},
        },
    }


def validate_reply(role: str, reply, schema: dict, instruction: str) -> str | None:
    """Schema constraints plus invariants that constrained decoding alone cannot enforce."""
    errors = list(Draft202012Validator(schema).iter_errors(reply))
    if errors:
        error = errors[0]
        path = ".".join(map(str, error.absolute_path)) or "reply"
        return f"{path}: {error.message}"
    if not reply["thinking"].strip():
        return "thinking needs a short statement of the decisive evidence."
    if role == "safety":
        if reply["allowed"] == bool(reply["reason"].strip()):
            return "Allowed movements need an empty reason; prohibited movements need a reason."
    elif role == "reference":
        tasks, message = reply["tasks"], reply["message"].strip()
        if reply["status"] != "resolved":
            if tasks or not message:
                return "Unresolved references need a message and no tasks."
        elif not tasks or message or any(not task["objects"] for task in tasks):
            return "Resolved references need nonempty tasks and objects, and an empty message."
        elif any(len(set(task["objects"])) != len(task["objects"]) for task in tasks):
            return "Do not repeat an object within a task."
    elif role == "constraint":
        status, restrictions, message = reply["status"], reply["restrictions"], reply["message"].strip()
        if status == "clear" and (restrictions or message):
            return "A clear constraint check needs no restrictions and an empty message."
        if status == "uncertain" and (restrictions or not message):
            return "An uncertain constraint check needs a specific question and no restrictions."
        if status == "conflict" and (not restrictions or message):
            return "A conflict needs object/quote restrictions and an empty message."
        for restriction in restrictions:
            quote = restriction["quote"].strip().strip('"\'').strip()
            if not quote or quote.lower() not in instruction.lower():
                return "Every restriction must quote a nonempty exact excerpt from the instruction."
    return None
