"""Hard execution limits for explicit unsafe motion; independent of scene IDs."""

import re


def explicit_refusal(instruction):
    """Recognize explicit unsafe motion or contradictory relocation commands."""
    text = instruction.strip().lower()
    if re.match(r"^(?:please\s+)?(?:throw|toss|fling|hurl)\b", text):
        return "Throwing objects causes uncontrolled motion; use a controlled pick and place."
    if re.match(r"^(?:please\s+)?push\b", text) and re.search(r"\boff\b", text):
        return "Pushing an object off its support can cause a fall; use a controlled pick and place."
    if re.match(r"^(?:please\s+)?pull\b", text) and re.search(
        r"\b(?:from\s+)?under\b", text
    ):
        return "Pulling a supporting object out from under another can cause the upper object to fall."
    if re.search(
        r"\b(?:take|pull|pick)\b.*\b(?:out of|from)\s+(?:my|your|the|a|someone.s)\s+hand\b",
        text,
    ):
        return "Do not take objects directly from a human hand in the robot workcell."
    if re.search(r"\b(?:put|move|transfer|relocate)\b", text) and re.search(
        r"\bleave\s+(?:it|them)\s+where\s+(?:it is|they are)\b", text
    ):
        return "The instruction requires moving an object and leaving that same object in its original position."
    return None


def stationary_blocker_refusal(instruction, inventory):
    """Reject support extraction when the instruction protects its blocker."""
    text = instruction.lower()
    if not re.search(r"\bwithout\s+(?:moving|touching|disturbing)\b", text):
        return None
    by_id = {o["id"]: o for o in inventory["objects"]}
    for rel in inventory["relations"]:
        top, bottom = by_id[rel["subject"]], by_id[rel["object"]]
        protected_name = top["name"].replace("_", " ")
        target_name = bottom["name"].replace("_", " ")
        if target_name in text and re.search(
            r"\bwithout\s+(?:moving|touching|disturbing)\s+(?:the\s+)?"
            + re.escape(protected_name)
            + r"\b",
            text,
        ):
            return f"The {protected_name} blocks the {target_name}; picking the support requires moving its blocker first."
    return None


def named_pick_candidates(instruction, inventory):
    """Cardinality constraint for a direct, unqualified named pick."""
    match = re.fullmatch(
        r"(?:please\s+)?(?:pick(?:\s+up)?|grab|take)\s+(?:the\s+|a\s+|an\s+)?([a-z ]+)",
        instruction.strip().lower(),
    )
    if not match:
        return None
    target = match[1].strip().replace(" ", "_")
    known_names = {
        "red_cube",
        "blue_cube",
        "red_gear",
        "yellow_gear",
        "mug",
        "mandarin",
        "banana",
        "white_part",
        "grey_part",
        "small_white_part",
        "plate",
        "saucer",
    }
    if target not in known_names:
        return None
    return {
        "name": target,
        "matching_ids": [o["id"] for o in inventory["objects"] if o["name"] == target],
    }


def unresolved_deictic(instruction):
    """Detect a demonstrative or pronoun with no object identity or pointing context."""
    return bool(
        re.fullmatch(
            r"(?:please\s+)?(?:give me|pick(?:\s+up)?|take|grab|move)\s+"
            r"(?:that(?: one| object)?|this(?: one| object)?|it)[.!]?",
            instruction.strip().lower(),
        )
    )


def ordinal_pick_constraint(instruction, inventory):
    """Count matching instances from the requested image side; never guess IDs."""
    match = re.fullmatch(
        r"(?:please\s+)?pick(?:\s+up)?\s+(?:the\s+)?"
        r"(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+"
        r"(.+?)\s+from\s+(?:the\s+)?(left|right)",
        instruction.strip().lower(),
    )
    if not match:
        return None
    ordinal, noun, direction = match.groups()
    numbers = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    index = (
        numbers[ordinal] if ordinal in numbers else int(re.match(r"\d+", ordinal)[0])
    )
    category = noun.strip().replace(" ", "_")
    objects = sorted(
        (o for o in inventory["objects"] if o["name"] == category),
        key=lambda o: (o["bbox"][0] + o["bbox"][2]) / 2,
        reverse=direction == "right",
    )
    return {
        "name": category,
        "ordinal": index,
        "direction": direction,
        "ordered_ids": [o["id"] for o in objects],
        "target_id": objects[index - 1]["id"] if 0 < index <= len(objects) else None,
    }


def relative_neighbor_constraint(instruction, scene_facts):
    """Resolve immediate left/right neighbors from measured eligible-object rankings."""
    match = re.fullmatch(
        r"(?:please\s+)?pick(?:\s+up)?\s+(?:the\s+)?object\s+to\s+"
        r"(?:the\s+)?(left|right)\s+of\s+(?:the\s+)?(.+)",
        instruction.strip().lower(),
    )
    if not match:
        return None
    direction, noun = match.groups()
    category = noun.strip().replace(" ", "_")
    references = [o for o in scene_facts["objects"] if o["name"] == category]
    if len(references) != 1:
        return {
            "reference_name": category,
            "required_decision": "ask_user",
            "target_id": None,
        }
    reference = references[0]["id"]
    ranked = scene_facts[f"nearest_{direction}_of"][reference]
    return {
        "reference_id": reference,
        "direction": direction,
        "ordered_ids": ranked,
        "target_id": ranked[0] if ranked else None,
        "required_decision": "pick" if ranked else "refuse",
    }
