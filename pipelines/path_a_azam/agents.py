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

import json, os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from planner import BASE_URL, MODEL, check, describe, load_prompt, to_answer, with_patience

PROMPTS = {"safety": "agent_safety_v2", "reference": "agent_reference_v2", "constraint": "agent_constraint_v2"}

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
        "type": "object", "additionalProperties": False, "required": ["thinking", "must_not_move", "quote"],
        "properties": {"thinking": {"type": "string"},
                       "must_not_move": {"type": "array", "items": {"enum": moved}},
                       "quote": {"type": "string"}},
    }


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


def order_moves(tasks: list, graph: dict) -> dict:
    """The moves for the tasks, in an order the robot can carry out: whatever lies on an object is lifted aside
    first, into the tray it is already in. Refuses when an object that has to move cannot be picked."""
    objs = {o["id"]: o for o in graph["objects"]}
    covers = {o["id"]: list(o.get("covered_by", [])) for o in graph["objects"]}
    for o in graph["objects"]:                      # "may be resting on" counts: lifting it off first is the safe side
        for low in o.get("maybe_on", []):
            covers.setdefault(low, []).append(o["id"])
    moves, moved = [], set()

    def blocked(oid):
        o = objs[oid]
        if o.get("touching_hand"):
            return f"A person's hand is on {oid}, so the robot must not move it."
        if not o["pickable"]:
            return f"{oid} cannot be picked up: {o.get('why_not_pickable') or 'it is not pickable'}."
        return None

    def uncover(oid, below):
        for c in covers.get(oid, []):
            if c in moved or c in below:
                continue
            problem = blocked(c) or uncover(c, below + [oid])
            if problem:
                return problem
            moves.append({"object": c, "to": objs[c].get("in") or "black_tray"})
            moved.add(c)
        return None

    for task in tasks:
        for oid in task["objects"]:
            problem = blocked(oid) or uncover(oid, [])
            if problem:
                return {"decision": "refuse", "moves": [], "message": problem}
            moves.append({"object": oid, "to": task["to"]})
            moved.add(oid)
    return {"decision": "plan", "moves": moves, "message": ""}


class State(TypedDict, total=False):
    instruction: str
    graph: dict
    tasks: list
    plan: dict           # the moves worked out by order_moves, before the constraint agent has seen them
    reply: dict          # the final decision, in the single-call planner's format
    log: list
    gave_up: bool


class GraphPlanner:
    def __init__(self, model: str = MODEL, base_url: str = BASE_URL, api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("API_KEY") or "none", base_url=base_url, timeout=120)
        self.prompts = {role: load_prompt(name) for role, name in PROMPTS.items()}
        self.prompt_name = "+".join(name for name, _ in self.prompts.values())
        self.app = self.build()

    def ask(self, messages: list[dict], schema: dict) -> tuple[str, dict | None]:
        r = with_patience(
            self.client.chat.completions.create, model=self.model, messages=messages, temperature=0, max_tokens=700,
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}})
        text = r.choices[0].message.content or ""
        try:
            reply = json.loads(text)
            return text, reply if isinstance(reply, dict) else None
        except json.JSONDecodeError:
            return text, None

    # ------------------------------------------------------------------ the agents
    def safety(self, s: State) -> dict:
        user = f'INSTRUCTION\n{s["instruction"]}'   # the words only: told about a hand in the scene, it refused everything
        _, reply = self.ask([{"role": "system", "content": self.prompts["safety"][1]}, {"role": "user", "content": user}], SAFETY_SCHEMA)
        log = s["log"] + [{"agent": "safety", "reply": reply}]
        if reply is not None and reply["allowed"] is False and reply["reason"].strip():
            return {"log": log, "reply": {"decision": "refuse", "moves": [], "message": reply["reason"].strip()}}
        return {"log": log}

    def reference(self, s: State) -> dict:
        user = f'SCENE\n{json.dumps(scene_for_reference(s["graph"]))}\n\nINSTRUCTION\n{s["instruction"]}'
        _, reply = self.ask([{"role": "system", "content": self.prompts["reference"][1]}, {"role": "user", "content": user}],
                            reference_schema(s["graph"]))
        log = s["log"] + [{"agent": "reference", "reply": reply}]
        if reply is None:
            return {"log": log, "tasks": []}
        message = reply["message"].strip()
        if reply["status"] == "ambiguous" and message:
            return {"log": log, "reply": {"decision": "ask_user", "moves": [], "message": message}}
        if reply["status"] == "not_on_table" and message:
            return {"log": log, "reply": {"decision": "refuse", "moves": [], "message": message}}
        return {"log": log, "tasks": [t for t in reply["tasks"] if t["objects"]]}

    def order(self, s: State) -> dict:
        plan = order_moves(s["tasks"], s["graph"])
        error = check(plan, s["graph"])
        log = s["log"] + [{"agent": "order (code)", "reply": plan, "error": error}]
        if error is not None:      # no objects resolved, or an order the gripper cannot do
            return {"log": log, "gave_up": True, "reply": {
                "decision": "refuse", "moves": [], "message": "No valid plan could be produced for this instruction."}}
        if plan["decision"] == "refuse":
            return {"log": log, "reply": plan}
        return {"log": log, "plan": plan}

    def constraint(self, s: State) -> dict:
        moved = list(dict.fromkeys(m["object"] for m in s["plan"]["moves"]))
        user = f'INSTRUCTION\n{s["instruction"]}\n\nTHE ROBOT WILL MOVE\n{", ".join(moved)}'
        _, reply = self.ask([{"role": "system", "content": self.prompts["constraint"][1]}, {"role": "user", "content": user}],
                            constraint_schema(moved))
        log = s["log"] + [{"agent": "constraint", "reply": reply}]
        if reply is not None and reply["must_not_move"]:
            quote = reply["quote"].strip().strip('"').strip()
            # only believed when the quoted words really are in the instruction
            if quote and quote.lower() in s["instruction"].lower():
                what = " and ".join(reply["must_not_move"])
                return {"log": log, "reply": {"decision": "refuse", "moves": [], "message":
                        f'The instruction says "{quote}", but {what} would have to be moved to carry it out.'}}
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
               "prompt": self.prompt_name, "attempts": end["log"]}
        if end.get("gave_up"):
            log["gave_up"] = True
        return to_answer(instruction_id, end["reply"], graph), log
