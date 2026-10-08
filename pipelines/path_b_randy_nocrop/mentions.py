"""Decide mode: what the instruction mentions. Apertus extracts the
objects from the sentence alone; code matches them to the named marks
and adds measured facts about them."""
import paths  # noqa: F401
from apertus import ask
from baseline import parse
from measure import COLOURS, NEIGHBOURS


# Measured colours that an instruction's colour word may stand for. Only
# neighbouring hues: white and grey stay apart, or "the white part" would
# match every grey part too.
NEAR_COLOURS = {"red": {"red", "orange"}, "orange": {"orange", "red"},
                "yellow": {"yellow", "orange"}, "green": {"green", "yellow"}}

EXTRACT_PROMPT = """\
A robot arm gets this instruction: "{text}"

List the physical objects the instruction mentions. Give each object \
itself, without position or relation words: "the gear lying on the other \
gear" is two objects, gear and gear. For each give its colour word if the \
instruction has one (else null) and its noun as one singular word as \
written (gear, cube, banana, part, tray, ...). Use the noun "any" for \
words that name no kind of object, such as "object", "everything" or \
"it"."""

EXTRACT_SCHEMA = {
    "type": "object",
    "required": ["objects"],
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["phrase", "colour", "noun"],
                "properties": {
                    "phrase": {"type": "string"},
                    "colour": {"enum": COLOURS + [None]},
                    "noun": {"type": "string"},
                },
            },
        },
    },
}


def _measured_as(colour, mark):
    return mark["colour"] in NEAR_COLOURS.get(colour, {colour})


def _named(noun, colour, marks):
    """Marks whose close-up name holds the noun (and the colour, if any)."""
    for word in (noun, noun[:-1] if noun.endswith("s") else None):
        if not word:
            continue

        hits = [m for m in marks if word in m.get("name", "")]
        if colour:
            hits = [m for m in hits
                    if colour in m["name"] or _measured_as(colour, m)]

        if hits:
            return hits

    return []


def _relations(phrase, hits, marks):
    """Measured facts about a mentioned object's marks, for Decide.

    What lies on each matched mark; for several matches their order left
    to right and top to bottom; and the nearest other marks, by gap.
    """
    if not hits or "gaps" not in hits[0]:
        return []

    lines = []
    for m in hits:
        above = [f"mark {o['id']}" for o in marks if m["id"] in o["on"]]
        lying = ", ".join(above) + " lies on it" if above else \
            "nothing lies on it"
        lines.append(f"  mark {m['id']} {m.get('name', '')}: {lying}")

    if len(hits) > 1:
        across = sorted(hits, key=lambda m: m["pixel"][0])
        down = sorted(hits, key=lambda m: m["pixel"][1])
        for order, ranked in (("left to right", across),
                              ("top to bottom", down)):
            ids = ", ".join(f"mark {m['id']}" for m in ranked)
            lines.append(f"  {phrase} from {order}: {ids}")

    mine = {m["id"] for m in hits}
    names = {m["id"]: m.get("name", "") for m in marks}
    others = sorted((min(m["gaps"][o] for m in hits), o)
                    for o in names if o not in mine)
    near = ", ".join(f"mark {o} {names[o]} ({cm} cm)"
                     for cm, o in others[:NEIGHBOURS])
    if near:
        lines.append(f"  nearest to the {phrase}: {near}")

    return lines


def mentions(model, text, marks):
    """List what the instruction mentions, each matched to marks by code.

    Apertus extracts the objects from the sentence alone; code matches
    them to the marks' close-up names and measured colours.
    """
    prompt = EXTRACT_PROMPT.format(text=text)
    reply = parse(ask(model, prompt, [], EXTRACT_SCHEMA)) or {}

    lines = []
    for obj in reply.get("objects") or []:
        if not isinstance(obj, dict):
            continue

        phrase = str(obj.get("phrase") or "?")
        noun = str(obj.get("noun") or "").lower().strip()
        colour = obj.get("colour")
        if noun in ("any", ""):
            lines.append(f"- {phrase}: any object")
            continue

        if noun.rstrip("s") == "tray":
            lines.append(f"- {phrase}: a tray, a place without a mark")
            continue

        hits = _named(noun, colour, marks)
        if hits:
            found = ", ".join(f"mark {m['id']}" for m in hits)
        elif colour and (hits := [m for m in marks
                                  if _measured_as(colour, m)]):
            found = (f"no mark is named like that; marks measured "
                     f"{colour}: " + ", ".join(
                         f"mark {m['id']} ({m['name']})" for m in hits))
        else:
            found = ("not among the named marks; it may be unmarked, "
                     "covered by another object or named differently: "
                     "check the photo")

        lines.append(f"- {phrase}: {found}")
        lines += _relations(phrase, hits, marks)

    if not lines:
        return ""

    return ("\nThe instruction mentions (matched by name and measured "
            "colour):\n" + "\n".join(lines) + "\n")
