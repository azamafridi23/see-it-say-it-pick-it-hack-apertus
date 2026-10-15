#!/usr/bin/env python3
"""Answer with three Apertus agents, wired as a fixed LangGraph graph.

Each agent has one narrow job:

    instruction + scene graph
      -> safety agent      is the movement asked for allowed?   -> refuse
      -> reference agent   which objects does the sentence mean?
                             -> refuse (not there) / ask_user (unclear)
      -> order the moves   code: lift off what covers an object first
                             -> refuse (not pickable, a hand is on it)
      -> constraint agent  does the instruction forbid moving one of
                           these?                               -> refuse
      -> answer

The route is fixed in code: no agent decides which agent runs next. Each
agent's rules are a text file in prompts/, and its reply is constrained to
a JSON schema. Ordering the moves is code, not an agent: given the objects
and what covers what there is one sensible order, and an agent given that
job misread the relations. The single-call planner in planner.py does the
whole job in one call; run.py picks one with --planner.

    from agents import GraphPlanner
    answer, log = GraphPlanner().answer(
        "scene_06_i4",
        "put the red gear under the yellow gear in the grey tray", graph)
"""
from __future__ import annotations

import hashlib
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIError, OpenAI

from planner import check, describe, load_prompt, to_answer, with_patience
from planning import order_moves
from agent_contracts import (SAFETY_SCHEMA, constraint_schema,
                             reference_schema, validate_reply)
import workcell
from workcell import CELL


def scene_for_reference(graph: dict) -> dict:
    """Describe the scene for the reference agent.

    The description gets two things read off the geometry, so that Apertus
    does not have to work them out: the nearest object in each direction,
    and the gap to every other object (not only the five nearest).

    Args:
        graph: The scene graph.

    Returns:
        The description, ready to be sent as JSON.
    """
    description = describe(graph)
    by_id = {obj["id"]: obj for obj in graph["objects"]}
    things = [obj for obj in graph["objects"]
              if "pixel" in obj and obj["label"] not in ("grey tray", "hand")]
    for entry in description["objects"]:
        obj = by_id[entry["id"]]
        gaps = dict(map(tuple, obj.get("nearest_gap_mm") or []))
        entry.pop("nearest_gap_mm", None)
        if not gaps:
            continue

        entry["gaps_mm"] = obj["nearest_gap_mm"]
        entry["neighbours"] = _nearest_by_side(obj, things, gaps)

    return description


def _side(obj: dict, other: dict) -> str:
    """Say on which side of ``obj`` the other object is, in the photo."""
    dx = other["pixel"][0] - obj["pixel"][0]
    dy = other["pixel"][1] - obj["pixel"][1]
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"

    return "above" if dy < 0 else "below"


def _nearest_by_side(obj: dict, things: list[dict], gaps: dict) -> dict:
    """Find the nearest object on each side of ``obj``.

    Args:
        obj: The object.
        things: The objects on the table.
        gaps: The gap from ``obj`` to each other object, by id, in mm.

    Returns:
        The nearest object's id per side (left, right, above, below).
    """
    nearest = {}
    for other in things:
        if other["id"] == obj["id"] or other["id"] not in gaps:
            continue

        side = _side(obj, other)
        if side not in nearest or gaps[other["id"]] < gaps[nearest[side]]:
            nearest[side] = other["id"]

    return nearest


def _decision(kind: str, message: str) -> dict:
    """Build a final question or refusal in the single-call format."""
    return {"decision": kind, "moves": [], "message": message}


class State(TypedDict, total=False):
    """What the graph carries from one step to the next."""

    instruction: str
    graph: dict
    tasks: list
    # The moves worked out by order_moves, before the constraint agent has
    # seen them.
    plan: dict
    # The final decision, in the single-call planner's format.
    reply: dict
    log: list
    gave_up: bool


class GraphPlanner:
    """The multi-agent planner."""

    def __init__(self, model: str = CELL.model,
                 base_url: str = CELL.base_url,
                 api_key: str | None = None,
                 prompt_names: dict[str, str] | None = None):
        """Set up the client, load the prompts and build the graph.

        Args:
            model: The model name on the server.
            base_url: The server's OpenAI-compatible URL.
            api_key: The key; LLM_API_KEY or APERTUS_API_KEY if None.
            prompt_names: Prompts to use instead of workcell.ini's, by
                role.

        Raises:
            ValueError: An override names an unknown role.
        """
        self.model = model
        names = {**CELL.agent_prompts, **(prompt_names or {})}
        if names.keys() != CELL.agent_prompts.keys():
            raise ValueError("Prompt overrides must name safety, reference, "
                             "or constraint.")

        self.prompts = {role: load_prompt(name)
                        for role, name in names.items()}
        self.prompt_name = "+".join(name for name, _ in self.prompts.values())
        key = api_key or workcell.api_key() or "none"
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=120)
        self.app = self.build()

    def ask(self, messages: list[dict],
            schema: dict) -> tuple[str, dict | None]:
        """Send the conversation so far and parse the reply.

        Args:
            messages: The chat messages.
            schema: The reply schema.

        Returns:
            The raw reply text, and the parsed reply or None when it is
            missing, cut short or not a JSON object.
        """
        response_format = {"type": "json_schema",
                           "json_schema": {"name": "answer",
                                           "schema": schema,
                                           "strict": True}}
        response = with_patience(
            self.client.chat.completions.create, model=self.model,
            messages=messages, temperature=0,
            max_tokens=CELL.agent_max_tokens,
            response_format=response_format)
        if not response.choices:
            return "", None

        choice = response.choices[0]
        text = choice.message.content or ""
        if choice.finish_reason != "stop":
            return text, None

        try:
            reply = json.loads(text)
        except json.JSONDecodeError:
            return text, None

        return text, reply if isinstance(reply, dict) else None

    def checked_ask(self, role: str, user: str, schema: dict,
                    instruction: str) -> tuple[dict | None, list]:
        """Ask one agent, sending invalid replies back for correction.

        A missing or invalid reply never becomes permission to proceed.

        Args:
            role: The agent's role.
            user: The user message.
            schema: The reply schema.
            instruction: The instruction, for checking quotes.

        Returns:
            The valid reply or None, and a log of every attempt.
        """
        system = self.prompts[role][1]
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        attempts = []
        for attempt in range(CELL.agent_retries + 1):
            try:
                raw, reply = self.ask(messages, schema)
            except APIError as exc:
                # with_patience already exhausted transport retries. Do not
                # expose response bodies/credentials.
                failure = f"Model service failed: {type(exc).__name__}"
                attempts.append({"agent": role, "attempt": attempt + 1,
                                 "reply": None, "error": failure})
                return None, attempts

            error = validate_reply(role, reply, schema, instruction)
            logged = reply if reply is not None else raw
            attempts.append({"agent": role, "attempt": attempt + 1,
                             "reply": logged, "error": error})
            if error is None:
                return reply, attempts

            correction = (f"Invalid response: {error}\n"
                          f"Return corrected JSON for the same task.")
            messages.extend([{"role": "assistant", "content": raw},
                             {"role": "user", "content": correction}])

        return None, attempts

    @staticmethod
    def failed(role: str, log: list) -> dict:
        """Refuse because one check could not be completed.

        Args:
            role: The agent that failed.
            log: The log so far.

        Returns:
            The graph state update.
        """
        message = (f"The system could not complete the {role} check. "
                   f"No action plan was issued; please retry.")
        return {"log": log, "gave_up": True,
                "reply": _decision("refuse", message)}

    # -------------------------------------------------------- the agents
    def safety(self, state: State) -> dict:
        """Judge whether the movement asked for is allowed.

        Args:
            state: The graph state.

        Returns:
            The state update: a refusal, or nothing new.
        """
        instruction = state["instruction"]
        # The words only: told about a hand in the scene, it refused
        # everything.
        user = f"INSTRUCTION\n{instruction}"
        reply, attempts = self.checked_ask("safety", user, SAFETY_SCHEMA,
                                           instruction)
        log = state["log"] + attempts
        if reply is None:
            return self.failed("safety", log)

        if reply["allowed"] is False:
            refusal = _decision("refuse", reply["reason"].strip())
            return {"log": log, "reply": refusal}

        return {"log": log}

    def reference(self, state: State) -> dict:
        """Work out which objects the sentence means, and where they go.

        Args:
            state: The graph state.

        Returns:
            The state update: the tasks, a question or a refusal.
        """
        instruction, graph = state["instruction"], state["graph"]
        scene = json.dumps(scene_for_reference(graph))
        user = f"SCENE\n{scene}\n\nINSTRUCTION\n{instruction}"
        reply, attempts = self.checked_ask("reference", user,
                                           reference_schema(graph),
                                           instruction)
        log = state["log"] + attempts
        if reply is None:
            return self.failed("reference", log)

        message = reply["message"].strip()
        if reply["status"] == "ambiguous" and message:
            return {"log": log, "reply": _decision("ask_user", message)}

        if reply["status"] == "not_on_table" and message:
            return {"log": log, "reply": _decision("refuse", message)}

        return {"log": log, "tasks": reply["tasks"]}

    def order(self, state: State) -> dict:
        """Order the moves in code and check them.

        Args:
            state: The graph state.

        Returns:
            The state update: the plan, a question or a refusal.
        """
        plan = order_moves(state["tasks"], state["graph"])
        error = check(plan, state["graph"])
        step = {"agent": "order (code)", "reply": plan, "error": error}
        log = state["log"] + [step]
        # No objects resolved, or an order the gripper cannot do.
        if error is not None:
            message = "No valid plan could be produced for this instruction."
            return {"log": log, "gave_up": True,
                    "reply": _decision("refuse", message)}

        if plan["decision"] != "plan":
            return {"log": log, "reply": plan}

        return {"log": log, "plan": plan}

    def constraint(self, state: State) -> dict:
        """Check whether the instruction forbids moving one of the objects.

        Args:
            state: The graph state.

        Returns:
            The state update: the plan, a question or a refusal.
        """
        instruction = state["instruction"]
        moves = state["plan"]["moves"]
        moved = list(dict.fromkeys(move["object"] for move in moves))
        # Only the objects that will move, not the scene or the plan: given
        # those, it judged whether the plan reached the goal and rejected
        # correct plans ("under the mug" read as "the mug must stay").
        moving = ", ".join(moved)
        user = f"INSTRUCTION\n{instruction}\n\nTHE ROBOT WILL MOVE\n{moving}"
        reply, attempts = self.checked_ask("constraint", user,
                                           constraint_schema(moved),
                                           instruction)
        log = state["log"] + attempts
        if reply is None:
            return self.failed("constraint", log)

        if reply["status"] == "uncertain":
            question = _decision("ask_user", reply["message"].strip())
            return {"log": log, "reply": question}

        if reply["status"] == "conflict":
            reasons = []
            for restriction in reply["restrictions"]:
                quote = restriction["quote"].strip()
                reasons.append(f'the instruction says "{quote}", but '
                               f'{restriction["object"]} would have to be '
                               f'moved')

            message = ("This cannot be carried out: " + "; ".join(reasons)
                       + ".")
            return {"log": log, "reply": _decision("refuse", message)}

        return {"log": log, "reply": state["plan"]}

    # -------------------------------------------------------- the wiring
    def build(self):
        """Wire the steps into a fixed graph with early exits.

        Returns:
            The compiled LangGraph app.
        """
        graph = StateGraph(State)
        graph.add_node("safety", self.safety)
        graph.add_node("reference", self.reference)
        graph.add_node("order", self.order)
        graph.add_node("constraint", self.constraint)

        def done(state):
            return "done" if state.get("reply") else "next"

        graph.add_edge(START, "safety")
        graph.add_conditional_edges("safety", done,
                                    {"done": END, "next": "reference"})
        graph.add_conditional_edges("reference", done,
                                    {"done": END, "next": "order"})
        graph.add_conditional_edges("order", done,
                                    {"done": END, "next": "constraint"})
        graph.add_edge("constraint", END)
        return graph.compile()

    def answer(self, instruction_id: str, instruction: str,
               graph: dict) -> tuple[dict, dict]:
        """Answer one instruction.

        Args:
            instruction_id: The instruction's id.
            instruction: The instruction.
            graph: The scene graph.

        Returns:
            The answer in the challenge format, and a log of what every
            agent said.
        """
        end = self.app.invoke({"instruction": instruction, "graph": graph,
                               "log": []})
        prompt_hashes = {}
        for role, (_, text) in self.prompts.items():
            prompt_hashes[role] = hashlib.sha256(text.encode()).hexdigest()

        log = {"instruction_id": instruction_id, "instruction": instruction,
               "model": self.model, "prompt": self.prompt_name,
               "prompt_sha256": prompt_hashes, "attempts": end["log"]}
        if end.get("gave_up"):
            log["gave_up"] = True

        return to_answer(instruction_id, end["reply"], graph), log
