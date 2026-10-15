"""Offline regression tests: no model calls, downloads or credentials.

Run them from the repository root:

    python -m unittest discover -s pipelines/path_a -p 'test_*.py' -v

They also run in the jury's environment without an API key.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from jsonschema import Draft202012Validator
from openai import APIConnectionError
import httpx

from agents import GraphPlanner
from agent_contracts import (SAFETY_SCHEMA, constraint_schema,
                             reference_schema, validate_reply)
from planning import check, order_moves
from dev.replay_agents import cases


def scene(stacked=True, uncertain=False):
    """Build a small scene graph: a gear, a mug and the two trays.

    Args:
        stacked: Whether the mug lies on the gear.
        uncertain: Whether the mug may be lying on the gear.

    Returns:
        The scene graph.
    """
    def obj(oid, label, x, **extra):
        return {"id": oid, "label": label, "colour": "red",
                "pixel": [x, 100], "size_mm": [50, 30], "pickable": True,
                "in": "grey_tray", "on": [], "covered_by": [],
                "maybe_on": [], "touching_hand": False,
                "nearest_gap_mm": [], **extra}

    return {"scene": "test", "gripper_max_opening_mm": 85,
            "hand_present": False, "left_to_right": ["gear", "mug"],
            "top_to_bottom": ["gear", "mug"], "objects": [
                obj("gear", "red gear", 100,
                    covered_by=["mug"] if stacked else []),
                obj("mug", "mug", 200, on=["gear"] if stacked else [],
                    maybe_on=["gear"] if uncertain else []),
                obj("grey_tray", "grey tray", 400, pickable=False,
                    **{"in": None}),
                obj("black_tray", "black tray", 600, pickable=False,
                    **{"in": None}),
            ]}


def task(*objects, to=None):
    """Build a reference agent task."""
    return {"objects": list(objects), "to": to}


def resolved(*tasks):
    """Build a resolved reference agent reply."""
    return {"thinking": "The named objects match the scene.",
            "status": "resolved", "tasks": list(tasks), "message": ""}


ALLOW = {"thinking": "Controlled pick-and-place is requested.",
         "allowed": True, "reason": ""}
CLEAR = {"thinking": "No movement restriction is stated.",
         "status": "clear", "restrictions": [], "message": ""}


class PlanningTests(unittest.TestCase):
    """Ordering and checking moves in code."""

    def test_first_pick_is_confirmed_blocker(self):
        graph = scene()
        plan = order_moves([task("gear", to="black_tray")], graph)
        self.assertEqual(plan["moves"],
                         [{"object": "mug", "to": "grey_tray"},
                          {"object": "gear", "to": "black_tray"}])
        self.assertIsNone(check(plan, graph))

    def test_blocker_already_at_goal_is_not_moved_twice(self):
        plan = order_moves([task("gear", "mug", to="grey_tray")], scene())
        self.assertEqual([move["object"] for move in plan["moves"]],
                         ["mug", "gear"])

    def test_shared_blocker_is_moved_once(self):
        graph = scene()
        second = {**graph["objects"][0], "id": "gear2"}
        graph["objects"].append(second)
        plan = order_moves([task("gear", "gear2", to="black_tray")], graph)
        self.assertEqual([move["object"] for move in plan["moves"]],
                         ["mug", "gear", "gear2"])

    def test_uncertain_support_is_lifted_first_by_default(self):
        graph = scene(stacked=False, uncertain=True)
        plan = order_moves([task("gear")], graph)
        self.assertEqual(plan["decision"], "plan")
        self.assertEqual([move["object"] for move in plan["moves"]],
                         ["mug", "gear"])
        self.assertIsNone(check(plan, graph))

    def test_uncertain_support_asks_under_the_ask_policy(self):
        graph = scene(stacked=False, uncertain=True)
        plan = order_moves([task("gear")], graph, uncertain="ask")
        self.assertEqual(plan["decision"], "ask_user")
        self.assertEqual(plan["moves"], [])
        self.assertIn("beside", plan["message"])

    def test_uncertainty_not_relevant_to_upper_pick(self):
        graph = scene(stacked=False, uncertain=True)
        plan = order_moves([task("mug")], graph)
        self.assertEqual(plan["moves"], [{"object": "mug", "to": None}])
        self.assertIsNone(check(plan, graph))

    def test_checker_rejects_unresolved_lower_pick_under_the_ask_policy(
            self):
        graph = scene(stacked=False, uncertain=True)
        plan = {"decision": "plan",
                "moves": [{"object": "gear", "to": None}], "message": ""}
        self.assertIn("resting", check(plan, graph, uncertain="ask"))
        # Default: the single-call planner decides about maybe_on itself.
        self.assertIsNone(check(plan, graph))

    def test_explicit_upper_move_clears_uncertainty(self):
        graph = scene(stacked=False, uncertain=True)
        plan = order_moves([task("mug", to="black_tray"), task("gear")],
                           graph)
        self.assertEqual(plan["decision"], "plan")
        self.assertIsNone(check(plan, graph))

    def test_replacing_blocker_does_not_clear_support(self):
        plan = {"decision": "plan",
                "moves": [{"object": "mug", "to": "gear"},
                          {"object": "gear", "to": None}],
                "message": ""}
        self.assertIn("covered", check(plan, scene()))

    def test_new_placement_creates_new_blocker(self):
        plan = {"decision": "plan",
                "moves": [{"object": "mug", "to": "gear"},
                          {"object": "gear", "to": None}],
                "message": ""}
        self.assertIn("covered", check(plan, scene(stacked=False)))

    def test_cycle_is_not_silently_ignored(self):
        graph = scene()
        graph["objects"][0]["on"] = ["mug"]
        plan = order_moves([task("gear")], graph)
        self.assertEqual(plan["decision"], "ask_user")
        self.assertIn("cycle", plan["message"])

    def test_multiple_holds_ask_for_destination(self):
        plan = order_moves([task("gear", "mug")], scene())
        self.assertEqual(plan["decision"], "ask_user")
        self.assertIn("hold only one", plan["message"])

    def test_single_hold_is_ordered_last(self):
        plan = order_moves([task("gear"), task("mug", to="grey_tray")],
                           scene(stacked=False))
        self.assertEqual([move["object"] for move in plan["moves"]],
                         ["mug", "gear"])
        self.assertIsNone(check(plan, scene(stacked=False)))

    def test_incompatible_destinations_ask(self):
        plan = order_moves([task("mug", to="grey_tray"),
                            task("mug", to="black_tray")], scene())
        self.assertEqual(plan["decision"], "ask_user")

    def test_final_goal_cannot_be_undone_by_later_uncovering(self):
        graph = scene(stacked=False)
        plan = order_moves([task("mug", to="gear"),
                            task("gear", to="black_tray")], graph)
        self.assertEqual(plan["decision"], "ask_user")
        self.assertIn("final placement", plan["message"])

    def test_hand_and_unpickable_objects_refused(self):
        for key, value in [("touching_hand", True), ("pickable", False)]:
            with self.subTest(key=key):
                graph = scene()
                graph["objects"][1][key] = value
                self.assertEqual(
                    order_moves([task("gear")], graph)["decision"],
                    "refuse")

    def test_missing_parking_surface_is_not_invented(self):
        graph = scene()
        graph["objects"][1]["in"] = None
        self.assertEqual(order_moves([task("gear")], graph)["decision"],
                         "ask_user")

    def test_no_moves_on_question_or_refusal(self):
        self.assertIsNotNone(check(
            {"decision": "ask_user", "message": "Which?",
             "moves": [{"object": "gear", "to": None}]}, scene()))


class ContractTests(unittest.TestCase):
    """Reply schemas and the rules a schema cannot hold."""

    def test_safety_needs_real_boolean_and_consistent_reason(self):
        for reply in [{**ALLOW, "allowed": "false"},
                      {**ALLOW, "allowed": False},
                      {**ALLOW, "reason": "no"}, {}]:
            with self.subTest(reply=reply):
                self.assertIsNotNone(validate_reply(
                    "safety", reply, SAFETY_SCHEMA, "pick gear"))

    def test_reference_cannot_hide_invalid_status_with_tasks(self):
        schema = reference_schema(scene())
        for reply in [resolved(), resolved(task()), resolved(task("absent")),
                      {**resolved(task("gear")), "status": "ambiguous",
                       "message": "Which?"}]:
            self.assertIsNotNone(validate_reply("reference", reply, schema,
                                                "pick gear"))

    def test_quotes_and_restriction_ids_are_checked(self):
        schema = constraint_schema(["mug"])
        reply = {"thinking": "A restriction applies.", "status": "conflict",
                 "restrictions": [{"object": "mug",
                                   "quote": "without moving the mug"}],
                 "message": ""}
        self.assertIsNone(validate_reply(
            "constraint", reply, schema,
            "pick the gear without moving the mug"))
        self.assertIsNotNone(validate_reply("constraint", reply, schema,
                                            "pick the gear"))
        reply["restrictions"][0]["object"] = "gear"
        self.assertIsNotNone(validate_reply(
            "constraint", reply, schema,
            "pick the gear without moving the mug"))

    def test_uncertain_constraints_need_question_and_no_conflict(self):
        schema = constraint_schema(["mug"])
        reply = {**CLEAR, "status": "uncertain",
                 "message": "Which object must stay?"}
        self.assertIsNone(validate_reply("constraint", reply, schema,
                                         "leave that there"))
        reply["message"] = ""
        self.assertIsNotNone(validate_reply("constraint", reply, schema,
                                            "leave that there"))

    def test_all_few_shot_outputs_satisfy_contracts(self):
        prompts = Path(__file__).parent / "prompts"
        for role in ("safety", "reference", "constraint"):
            instruction = ""
            for line in (prompts / f"agent_{role}_v3.txt").read_text() \
                    .splitlines():
                if line.startswith("Instruction: "):
                    instruction = line.removeprefix("Instruction: ")

                if not line.startswith("Answer: "):
                    continue

                reply = json.loads(line.removeprefix("Answer: "))
                if role == "safety":
                    schema = SAFETY_SCHEMA
                elif role == "reference":
                    ids = {oid for task in reply["tasks"]
                           for oid in [*task["objects"], task["to"]]
                           if oid is not None}
                    schema = reference_schema(
                        {"objects": [{"id": oid}
                                     for oid in ids or {"unused"}]})
                else:
                    schema = constraint_schema(
                        [restriction["object"]
                         for restriction in reply["restrictions"]]
                        or ["unused"])

                with self.subTest(role=role, instruction=instruction):
                    self.assertIsNone(validate_reply(role, reply, schema,
                                                     instruction))


class AgentFlowTests(unittest.TestCase):
    """The graph of agents, with scripted replies."""

    def setUp(self):
        self.planner = GraphPlanner(api_key="offline-test",
                                    base_url="http://127.0.0.1:1/v1")

    def tearDown(self):
        self.planner.client.close()

    def run_replies(self, replies, instruction="pick up the gear",
                    graph=None):
        def response(*args):
            reply = next(it)
            return json.dumps(reply), reply

        it = iter(replies)
        with patch.object(self.planner, "ask", side_effect=response) as ask:
            answer, log = self.planner.answer("test_i1", instruction,
                                              graph or scene())

        result_schema = json.loads(
            (Path(__file__).resolve().parents[2]
             / "schema/result.schema.json").read_text())
        Draft202012Validator(result_schema).validate(answer)
        return answer, log, ask

    def test_malformed_safety_never_reaches_reference(self):
        answer, log, ask = self.run_replies([None] * 3)
        self.assertEqual(ask.call_count, 3)
        self.assertTrue(log["gave_up"])
        self.assertEqual(answer["actions"][0]["action"], "refuse")
        self.assertIn("system", answer["actions"][0]["reason"])

    def test_invalid_reply_can_be_corrected(self):
        answer, log, ask = self.run_replies(
            [{}, ALLOW, resolved(task("gear")), CLEAR])
        self.assertEqual(answer["target_pixel"], [200, 100])
        self.assertEqual(ask.call_count, 4)
        self.assertIsNotNone(log["attempts"][0]["error"])
        self.assertNotIn("gave_up", log)

    def test_failed_constraint_never_releases_plan(self):
        answer, log, _ = self.run_replies(
            [ALLOW, resolved(task("gear")), None, None, None])
        self.assertTrue(log["gave_up"])
        self.assertIsNone(answer["target_pixel"])
        self.assertEqual(answer["actions"][0]["action"], "refuse")

    def test_failed_reference_does_not_become_scene_absence(self):
        answer, log, _ = self.run_replies([ALLOW, {}, {}, {}])
        self.assertTrue(log["gave_up"])
        self.assertIn("reference check", answer["actions"][0]["reason"])

    def test_constraints_receive_only_the_objects_that_will_move(self):
        _, _, ask = self.run_replies(
            [ALLOW, resolved(task("gear")), CLEAR])
        user = ask.call_args_list[-1].args[0][1]["content"]
        self.assertIn("THE ROBOT WILL MOVE\nmug, gear", user)
        # No scene and no plan: with them it judged the plan's goal.
        self.assertNotIn('"position"', user)
        self.assertNotIn('"to"', user)

    def test_verified_constraint_refuses_with_evidence(self):
        reply = {"thinking": "The mug is forbidden to move.",
                 "status": "conflict",
                 "restrictions": [{"object": "mug",
                                   "quote": "without moving the mug"}],
                 "message": ""}
        answer, log, _ = self.run_replies(
            [ALLOW, resolved(task("gear")), reply],
            instruction="pick up the gear without moving the mug")
        self.assertEqual(answer["actions"][0]["action"], "refuse")
        self.assertIn("without moving the mug",
                      answer["actions"][0]["reason"])
        self.assertNotIn("gave_up", log)

    def test_uncertain_constraints_ask(self):
        reply = {**CLEAR, "status": "uncertain",
                 "message": "Which object must remain untouched?"}
        answer, _, _ = self.run_replies(
            [ALLOW, resolved(task("gear")), reply])
        self.assertEqual(answer["actions"][0]["action"], "ask_user")

    def test_unresolved_support_stops_before_constraint_under_the_ask_policy(
            self):
        with patch("planning.UNCERTAIN_SUPPORT", "ask"):
            answer, _, ask = self.run_replies(
                [ALLOW, resolved(task("gear"))],
                graph=scene(stacked=False, uncertain=True))

        self.assertEqual(ask.call_count, 2)
        self.assertEqual(answer["actions"][0]["action"], "ask_user")

    def test_unresolved_support_is_lifted_first_in_the_graph(self):
        answer, _, ask = self.run_replies(
            [ALLOW, resolved(task("gear")), CLEAR],
            graph=scene(stacked=False, uncertain=True))
        self.assertEqual(ask.call_count, 3)
        picked = [action["object"] for action in answer["actions"]
                  if action["action"] == "pick"]
        self.assertEqual(picked, ["mug", "gear"])

    def test_truncated_valid_json_is_not_accepted(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(ALLOW)),
            finish_reason="length")])
        with patch("agents.with_patience", return_value=response):
            _, reply = self.planner.ask([], SAFETY_SCHEMA)

        self.assertIsNone(reply)

    def test_missing_completion_is_not_accepted(self):
        with patch("agents.with_patience",
                   return_value=SimpleNamespace(choices=[])):
            self.assertEqual(self.planner.ask([], SAFETY_SCHEMA),
                             ("", None))

    def test_transport_failure_is_a_system_failure(self):
        error = APIConnectionError(
            request=httpx.Request("POST", "http://127.0.0.1:1/v1"))
        with patch.object(self.planner, "ask", side_effect=error):
            answer, log = self.planner.answer("test_i1", "pick up the gear",
                                              scene())

        self.assertTrue(log["gave_up"])
        self.assertIn("system", answer["actions"][0]["reason"])
        self.assertEqual(len(log["attempts"]), 1)

    def test_invalid_quote_is_retried_not_ignored(self):
        invalid = {"thinking": "The mug cannot move.", "status": "conflict",
                   "restrictions": [{"object": "mug",
                                     "quote": "invented restriction"}],
                   "message": ""}
        answer, log, _ = self.run_replies(
            [ALLOW, resolved(task("gear")), invalid, invalid, invalid])
        self.assertTrue(log["gave_up"])
        self.assertEqual(answer["actions"][0]["action"], "refuse")

    def test_prompt_overrides_are_loaded(self):
        planner = GraphPlanner(api_key="offline-test",
                               prompt_names={"safety": "agent_safety_v3"})
        self.addCleanup(planner.client.close)
        self.assertTrue(planner.prompt_name.startswith(
            "agent_safety_v3+agent_reference_v2"))


class ReplayTests(unittest.TestCase):
    """The replay tool's inputs."""

    def test_instruction_categories_are_not_sent_to_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graphs, data = root / "graphs", root / "dataset"
            graphs.mkdir()
            (data / "scene_01").mkdir(parents=True)
            (graphs / "scene_01_graph.json").write_text(
                json.dumps({**scene(), "scene": "scene_01"}))
            (data / "scene_01/instructions.txt").write_text(
                "# ignored\nsimple\tpick up the gear\n\npick up the mug\n")
            rows = list(cases(graphs, data))
            self.assertEqual([(iid, text) for iid, text, _, _ in rows],
                             [("scene_01_i1", "pick up the gear"),
                              ("scene_01_i2", "pick up the mug")])
            self.assertEqual(rows[0][3], rows[1][3])
            self.assertEqual(len(rows[0][3]), 64)

    def test_missing_graphs_fail_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "No .*graph.json"):
                list(cases(Path(tmp), Path(tmp)))


if __name__ == "__main__":
    unittest.main()
