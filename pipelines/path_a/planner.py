#!/usr/bin/env python3
"""Answer with one Apertus call: a plan, a question or a refusal.

Apertus chooses objects by their id and never writes coordinates:
target_pixel is copied from the scene graph. Its reply is constrained to a
JSON schema, checked by code against the scene graph, and sent back with
the error when it breaks a rule (see ``check()``).

The rules Apertus is given are a text file in prompts/ (workcell.ini
names the default). Add new versions to try other wordings; the version used is
written to the log with every answer.

    from planner import Planner
    answer, log = Planner().answer(
        "scene_06_i4",
        "put the red gear under the yellow gear in the grey tray", graph)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from planning import check
from workcell import CELL

from dotenv import load_dotenv
from openai import (APIConnectionError, APITimeoutError,
                    InternalServerError, OpenAI, RateLimitError)

load_dotenv()

PROMPT_FOLDER = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str = CELL.single_call_prompt) -> tuple[str, str]:
    """Load a planner prompt.

    Args:
        name: A name in prompts/, such as ``"planner_v1"``, or the path to
            a text file.

    Returns:
        The version name and the prompt text.

    Raises:
        SystemExit: No such prompt; the message lists the available ones.
    """
    path = Path(name)
    if not path.is_file():
        file_name = name if name.endswith(".txt") else f"{name}.txt"
        path = PROMPT_FOLDER / file_name

    if not path.is_file():
        available = sorted(file.stem for file in PROMPT_FOLDER.glob("*.txt"))
        have = ", ".join(available)
        raise SystemExit(f'no planner prompt "{name}". '
                         f'Available in {PROMPT_FOLDER}: {have}')

    return path.stem, path.read_text().strip()


def schema(graph: dict) -> dict:
    """Build the reply format for one scene.

    Ids are limited to the ones in this scene, so Apertus cannot name an
    object that is not there.

    Args:
        graph: The scene graph.

    Returns:
        A JSON schema for the reply.
    """
    ids = [obj["id"] for obj in graph["objects"]]
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


def describe(graph: dict) -> dict:
    """Describe the scene graph as Apertus sees it.

    Boxes, scores and other detector detail are left out. So is
    why_not_pickable, on purpose: with it in the description the score fell
    from about 59 to about 52 in three runs. Apertus gets the reason from
    ``check()`` only when it tries to pick such an object.

    Args:
        graph: The scene graph.

    Returns:
        The description, ready to be sent as JSON.
    """
    # What may be resting on each object, read the other way round.
    maybe_under = {}
    for obj in graph["objects"]:
        for lower in obj.get("maybe_on", []):
            maybe_under.setdefault(lower, []).append(obj["id"])

    described = [_describe_object(obj, maybe_under)
                 for obj in graph["objects"]]
    return {
        "gripper_max_opening_mm": graph["gripper_max_opening_mm"],
        "hand_present": graph["hand_present"],
        "left_to_right": graph["left_to_right"],
        "top_to_bottom": graph["top_to_bottom"],
        "objects": described,
    }


def _describe_object(obj: dict, maybe_under: dict) -> dict:
    """Describe one object for Apertus; relations only when there are any."""
    # The black mat has no pixel: it is a place only.
    if "pixel" not in obj:
        return {"id": obj["id"], "type": obj["label"], "pickable": False,
                "note": obj.get("note", "")}

    kind = obj["label"]
    if kind == "object":
        kind = "unknown object"

    entry = {
        "id": obj["id"],
        "type": kind,
        "colour": obj["colour"],
        "position": obj["pixel"],
        "size_mm": obj["size_mm"],
        "pickable": obj["pickable"],
        "in": obj["in"],
    }
    for key in ("on", "maybe_on", "covered_by"):
        if obj[key]:
            entry[key] = obj[key]

    if obj["id"] in maybe_under:
        entry["maybe_covered_by"] = maybe_under[obj["id"]]

    if obj["touching_hand"]:
        entry["touching_hand"] = True

    if obj.get("nearest_gap_mm"):
        entry["nearest_gap_mm"] = obj["nearest_gap_mm"][:5]

    return entry


def to_answer(instruction_id: str, reply: dict, graph: dict) -> dict:
    """Turn a checked reply into the challenge's answer format.

    Args:
        instruction_id: The instruction's id.
        reply: The reply, already checked by ``check()``.
        graph: The scene graph, for the pixels.

    Returns:
        The answer; target_pixel is the scene graph's pixel of the first
        object picked.
    """
    message = reply["message"].strip()
    if reply["decision"] == "ask_user":
        action = {"action": "ask_user", "question": message}
        return {"instruction_id": instruction_id, "target_pixel": None,
                "actions": [action]}

    if reply["decision"] == "refuse":
        action = {"action": "refuse", "reason": message}
        return {"instruction_id": instruction_id, "target_pixel": None,
                "actions": [action]}

    pixels = {obj["id"]: obj["pixel"] for obj in graph["objects"]
              if "pixel" in obj}
    actions = []
    for move in reply["moves"]:
        for kind, oid in (("pick", move["object"]), ("place", move["to"])):
            if oid is None:
                continue

            action = {"action": kind, "object": oid}
            if oid in pixels:
                action["pixel"] = pixels[oid]

            actions.append(action)

    first_pick = reply["moves"][0]["object"]
    return {"instruction_id": instruction_id,
            "target_pixel": pixels[first_pick], "actions": actions}


def with_patience(call, **kwargs):
    """Call the server, trying again while it is busy or unreachable.

    It waits a little longer each time, up to about five minutes in all.

    Args:
        call: The function that calls the server.
        **kwargs: Its arguments.

    Returns:
        Whatever ``call`` returns.
    """
    for wait in (5, 10, 20, 40, 60, 60, 60, None):
        try:
            return call(**kwargs)
        except (RateLimitError, APIConnectionError, APITimeoutError,
                InternalServerError) as error:
            if wait is None:
                raise

            print(f"  Apertus is busy ({type(error).__name__}): "
                  f"trying again in {wait}s")
            time.sleep(wait)


class Planner:
    """The single-call planner: one Apertus call decides everything."""

    def __init__(self, model: str = CELL.model,
                 base_url: str = CELL.base_url, api_key: str | None = None,
                 prompt: str = CELL.single_call_prompt):
        """Set up the client and load the prompt.

        Args:
            model: The model name on the server.
            base_url: The server's OpenAI-compatible URL.
            api_key: The key; APERTUS_API_KEY from the environment if None.
            prompt: The planner prompt, as for ``load_prompt()``.
        """
        self.model = model
        self.prompt_name, self.system = load_prompt(prompt)
        self.client = OpenAI(
            api_key=api_key or os.environ.get("APERTUS_API_KEY") or "none",
            base_url=base_url, timeout=120)

    def ask(self, messages: list[dict],
            graph: dict) -> tuple[str, dict | None]:
        """Send the conversation so far and parse the reply.

        Args:
            messages: The chat messages.
            graph: The scene graph, for the reply schema.

        Returns:
            The raw reply text, and the parsed reply or None when it is not
            a JSON object.
        """
        response_format = {"type": "json_schema",
                           "json_schema": {"name": "answer",
                                           "schema": schema(graph),
                                           "strict": True}}
        response = with_patience(
            self.client.chat.completions.create, model=self.model,
            messages=messages, temperature=0,
            max_tokens=CELL.single_call_max_tokens,
            response_format=response_format)
        text = response.choices[0].message.content or ""
        try:
            reply = json.loads(text)
            return text, reply if isinstance(reply, dict) else None
        except json.JSONDecodeError:
            return text, None

    def answer(self, instruction_id: str, instruction: str,
               graph: dict) -> tuple[dict, dict]:
        """Answer one instruction, asking Apertus to correct broken answers.

        Args:
            instruction_id: The instruction's id.
            instruction: The instruction.
            graph: The scene graph.

        Returns:
            The answer in the challenge format, and a log of every attempt.
        """
        scene = json.dumps(describe(graph))
        user = f"SCENE\n{scene}\n\nINSTRUCTION\n{instruction}"
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": user}]
        log = {"instruction_id": instruction_id, "instruction": instruction,
               "model": self.model, "prompt": self.prompt_name,
               "attempts": []}
        for _ in range(CELL.single_call_retries + 1):
            text, reply = self.ask(messages, graph)
            if reply is None:
                error = "The answer was not valid JSON."
                logged = text
            else:
                error = check(reply, graph)
                logged = reply

            log["attempts"].append({"reply": logged, "error": error})
            if error is None:
                return to_answer(instruction_id, reply, graph), log

            correction = (f"That answer cannot be used. {error}\n"
                          f"Give a corrected answer.")
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": correction}]

        log["gave_up"] = True
        refusal = {"action": "refuse",
                   "reason": "No valid plan could be produced for this "
                             "instruction."}
        answer = {"instruction_id": instruction_id, "target_pixel": None,
                  "actions": [refusal]}
        return answer, log
