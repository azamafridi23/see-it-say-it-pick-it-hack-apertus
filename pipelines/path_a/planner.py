#!/usr/bin/env python3
"""Apertus reads a scene graph and one instruction, and answers: a plan, a question, or a refusal.

Apertus chooses objects by their id and never writes coordinates: target_pixel is copied from the
scene graph. Its reply is constrained to a JSON schema, checked by code against the scene graph,
and sent back with the error when it breaks a rule (see check()).

The rules Apertus is given are a text file in prompts/ (default: planner_v2). Add new versions to try
other wordings; the version used is written to the log with every answer.

    from planner import Planner
    answer, log = Planner().answer("scene_06_i4", "put the red gear under the yellow gear in the grey tray", graph)
"""
from __future__ import annotations

import json, os, time
from pathlib import Path

from planning import check

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

load_dotenv()

MODEL = "apertus-v1.5-70b"
BASE_URL = "https://hackapertus.livemap.sh/v1"
MAX_RETRIES = 2

PROMPTS = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT = "planner_v1"


def load_prompt(name: str = DEFAULT_PROMPT) -> tuple[str, str]:
    """(version name, text) of a planner prompt: a name in prompts/ such as "planner_v2", or a path to a text file."""
    path = Path(name)
    if not path.is_file():
        path = PROMPTS / (name if name.endswith(".txt") else f"{name}.txt")
    if not path.is_file():
        have = ", ".join(sorted(p.stem for p in PROMPTS.glob("*.txt")))
        raise SystemExit(f'no planner prompt "{name}". Available in {PROMPTS}: {have}')
    return path.stem, path.read_text().strip()


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
    """The scene graph as Apertus sees it: without boxes, scores and other detector detail.
    why_not_pickable is left out on purpose: with it in the description the score fell from about 59 to about 52
    in three runs. Apertus gets the reason from check() only when it tries to pick such an object."""
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


def with_patience(call, **kwargs):
    """Call the server, waiting and trying again when it is busy or unreachable (up to about five minutes)."""
    for wait in (5, 10, 20, 40, 60, 60, 60, None):
        try:
            return call(**kwargs)
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            if wait is None:
                raise
            print(f"  Apertus is busy ({type(e).__name__}): trying again in {wait}s")
            time.sleep(wait)


class Planner:
    def __init__(self, model: str = MODEL, base_url: str = BASE_URL, api_key: str | None = None,
                 prompt: str = DEFAULT_PROMPT):
        self.model = model
        self.prompt_name, self.system = load_prompt(prompt)
        self.client = OpenAI(api_key=api_key or os.environ.get("APERTUS_API_KEY") or "none", base_url=base_url, timeout=120)

    def ask(self, messages: list[dict], graph: dict) -> tuple[str, dict | None]:
        r = with_patience(self.client.chat.completions.create,
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
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"SCENE\n{json.dumps(describe(graph))}\n\nINSTRUCTION\n{instruction}"}]
        log = {"instruction_id": instruction_id, "instruction": instruction, "model": self.model,
               "prompt": self.prompt_name, "attempts": []}
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
