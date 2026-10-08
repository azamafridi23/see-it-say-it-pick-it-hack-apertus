#!/usr/bin/env python3
"""The multi-agent planner: three Apertus agents with one narrow job each, wired as a fixed LangGraph graph.

    instruction + scene graph
         -> safety agent       is the movement asked for allowed?           -> refuse
         -> reference agent    which objects does the sentence mean?        -> refuse (not there) / ask_user (unclear)
         -> order the moves    code: lift off what covers an object first   -> refuse (not pickable, a hand is on it)
         -> constraint agent   does the instruction forbid moving one of these? -> refuse
         -> answer

The route is fixed in code: no agent decides which agent runs next. Each agent's rules are a text file in
prompts/, and its reply is constrained to a JSON schema. Ordering the moves is code, not an agent: given the
objects and what covers what there is one sensible order, and an agent given that job misread the relations.
The single-call planner in planner.py does the whole job in one call; run.py picks one with --planner.

    from agents import GraphPlanner
    answer, log = GraphPlanner().answer("scene_06_i4", "put the red gear under the yellow gear in the grey tray", graph)
"""
from __future__ import annotations

import hashlib, json, os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIError, OpenAI

from planner import BASE_URL, MODEL, check, describe, load_prompt, to_answer, with_patience
from planning import order_moves
from agent_contracts import SAFETY_SCHEMA, constraint_schema, reference_schema, validate_reply

PROMPTS = {"safety": "agent_safety_v2", "reference": "agent_reference_v2", "constraint": "agent_constraint_v4"}
MAX_AGENT_RETRIES = 2
MAX_TOKENS = 2000   # the reference agent thinks step by step: in a scene with ten objects, 700 cut its reply short


def scene_for_reference(graph: dict) -> dict:
    """The scene description plus two things read off the geometry, so that Apertus does not have to work them
    out: the nearest object in each direction, and the gap to every other object (not only the five nearest)."""
    d = describe(graph)
    full = {o["id"]: o for o in graph["objects"]}
    things = [o for o in graph["objects"] if "pixel" in o and o["label"] not in ("grey tray", "hand")]
    for o in d["objects"]:
        me = full[o["id"]]
        gaps = dict(map(tuple, me.get("nearest_gap_mm") or []))
        o.pop("nearest_gap_mm", None)
        if not gaps:
            continue
        o["gaps_mm"] = me["nearest_gap_mm"]
        near = {}
        for b in things:
            if b["id"] == me["id"] or b["id"] not in gaps:
                continue
            dx, dy = b["pixel"][0] - me["pixel"][0], b["pixel"][1] - me["pixel"][1]
            side = ("left" if dx < 0 else "right") if abs(dx) >= abs(dy) else ("above" if dy < 0 else "below")
            if side not in near or gaps[b["id"]] < gaps[near[side]]:
                near[side] = b["id"]
        o["neighbours"] = near
    return d

class State(TypedDict, total=False):
    instruction: str
    graph: dict
    tasks: list
    plan: dict           # the moves worked out by order_moves, before the constraint agent has seen them
    reply: dict          # the final decision, in the single-call planner's format
    log: list
    gave_up: bool


class GraphPlanner:
    def __init__(self, model: str = MODEL, base_url: str = BASE_URL, api_key: str | None = None,
                 prompt_names: dict[str, str] | None = None):
        self.model = model
        names = {**PROMPTS, **(prompt_names or {})}
        if names.keys() != PROMPTS.keys():
            raise ValueError("Prompt overrides must name safety, reference, or constraint.")
        self.prompts = {role: load_prompt(name) for role, name in names.items()}
        self.prompt_name = "+".join(name for name, _ in self.prompts.values())
        self.client = OpenAI(api_key=api_key or os.environ.get("API_KEY") or "none", base_url=base_url, timeout=120)
        self.app = self.build()

    def ask(self, messages: list[dict], schema: dict) -> tuple[str, dict | None]:
        r = with_patience(
            self.client.chat.completions.create, model=self.model, messages=messages, temperature=0, max_tokens=MAX_TOKENS,
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}})
        if not r.choices:
            return "", None
        text = r.choices[0].message.content or ""
        if r.choices[0].finish_reason != "stop":
            return text, None
        try:
            reply = json.loads(text)
            return text, reply if isinstance(reply, dict) else None
        except json.JSONDecodeError:
            return text, None

    def checked_ask(self, role: str, user: str, schema: dict, instruction: str) -> tuple[dict | None, list]:
        """Retry invalid content; never turn missing/invalid checks into permission to proceed."""
        messages = [{"role": "system", "content": self.prompts[role][1]}, {"role": "user", "content": user}]
        attempts = []
        for attempt in range(MAX_AGENT_RETRIES + 1):
            try:
                raw, reply = self.ask(messages, schema)
            except APIError as exc:
                # with_patience already exhausted transport retries. Do not expose response bodies/credentials.
                attempts.append({"agent": role, "attempt": attempt + 1, "reply": None,
                                 "error": f"Model service failed: {type(exc).__name__}"})
                return None, attempts
            error = validate_reply(role, reply, schema, instruction)
            attempts.append({"agent": role, "attempt": attempt + 1,
                             "reply": reply if reply is not None else raw, "error": error})
            if error is None:
                return reply, attempts
            messages.extend([{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Invalid response: {error}\nReturn corrected JSON for the same task."}])
        return None, attempts

    @staticmethod
    def failed(role: str, log: list) -> dict:
        return {"log": log, "gave_up": True, "reply": {
            "decision": "refuse", "moves": [],
            "message": f"The system could not complete the {role} check. No action plan was issued; please retry."}}

    # ------------------------------------------------------------------ the agents
    def safety(self, s: State) -> dict:
        user = f'INSTRUCTION\n{s["instruction"]}'   # the words only: told about a hand in the scene, it refused everything
        reply, attempts = self.checked_ask("safety", user, SAFETY_SCHEMA, s["instruction"])
        log = s["log"] + attempts
        if reply is None:
            return self.failed("safety", log)
        if reply["allowed"] is False:
            return {"log": log, "reply": {"decision": "refuse", "moves": [], "message": reply["reason"].strip()}}
        return {"log": log}

    def reference(self, s: State) -> dict:
        user = f'SCENE\n{json.dumps(scene_for_reference(s["graph"]))}\n\nINSTRUCTION\n{s["instruction"]}'
        reply, attempts = self.checked_ask("reference", user, reference_schema(s["graph"]), s["instruction"])
        log = s["log"] + attempts
        if reply is None:
            return self.failed("reference", log)
        message = reply["message"].strip()
        if reply["status"] == "ambiguous" and message:
            return {"log": log, "reply": {"decision": "ask_user", "moves": [], "message": message}}
        if reply["status"] == "not_on_table" and message:
            return {"log": log, "reply": {"decision": "refuse", "moves": [], "message": message}}
        return {"log": log, "tasks": reply["tasks"]}

    def order(self, s: State) -> dict:
        plan = order_moves(s["tasks"], s["graph"])
        error = check(plan, s["graph"])
        log = s["log"] + [{"agent": "order (code)", "reply": plan, "error": error}]
        if error is not None:      # no objects resolved, or an order the gripper cannot do
            return {"log": log, "gave_up": True, "reply": {
                "decision": "refuse", "moves": [], "message": "No valid plan could be produced for this instruction."}}
        if plan["decision"] != "plan":
            return {"log": log, "reply": plan}
        return {"log": log, "plan": plan}

    def constraint(self, s: State) -> dict:
        moved = list(dict.fromkeys(m["object"] for m in s["plan"]["moves"]))
        # Only the objects that will move, not the scene or the plan: given those, it judged whether the plan
        # reached the goal and rejected correct plans ("under the mug" read as "the mug must stay").
        user = f'INSTRUCTION\n{s["instruction"]}\n\nTHE ROBOT WILL MOVE\n{", ".join(moved)}'
        reply, attempts = self.checked_ask("constraint", user, constraint_schema(moved), s["instruction"])
        log = s["log"] + attempts
        if reply is None:
            return self.failed("constraint", log)
        if reply["status"] == "uncertain":
            return {"log": log, "reply": {"decision": "ask_user", "moves": [], "message": reply["message"].strip()}}
        if reply["status"] == "conflict":
            reasons = [f'the instruction says "{r["quote"].strip()}", but {r["object"]} would have to be moved'
                       for r in reply["restrictions"]]
            return {"log": log, "reply": {"decision": "refuse", "moves": [], "message":
                    "This cannot be carried out: " + "; ".join(reasons) + "."}}
        return {"log": log, "reply": s["plan"]}

    # ------------------------------------------------------------------ the wiring
    def build(self):
        g = StateGraph(State)
        for name, node in (("safety", self.safety), ("reference", self.reference), ("order", self.order),
                           ("constraint", self.constraint)):
            g.add_node(name, node)
        g.add_edge(START, "safety")
        done = lambda s: "done" if s.get("reply") else "next"  # noqa: E731
        g.add_conditional_edges("safety", done, {"done": END, "next": "reference"})
        g.add_conditional_edges("reference", done, {"done": END, "next": "order"})
        g.add_conditional_edges("order", done, {"done": END, "next": "constraint"})
        g.add_edge("constraint", END)
        return g.compile()

    def answer(self, instruction_id: str, instruction: str, graph: dict) -> tuple[dict, dict]:
        """(the answer in the challenge format, a log of what every agent said)."""
        end = self.app.invoke({"instruction": instruction, "graph": graph, "log": []})
        log = {"instruction_id": instruction_id, "instruction": instruction, "model": self.model,
               "prompt": self.prompt_name,
               "prompt_sha256": {role: hashlib.sha256(text.encode()).hexdigest()
                                 for role, (_, text) in self.prompts.items()},
               "attempts": end["log"]}
        if end.get("gave_up"):
            log["gave_up"] = True
        return to_answer(instruction_id, end["reply"], graph), log
