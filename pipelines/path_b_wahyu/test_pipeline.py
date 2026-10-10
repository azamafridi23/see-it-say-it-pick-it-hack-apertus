"""Offline regression checks: python -m unittest pipelines.path_b_wahyu.test_pipeline."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import jsonschema
import numpy as np

from .apertus_client import make_json_request, parse_model_json
from .grounding import (
    add_dark_tray_reference,
    ground_inventory,
    ground_object,
    planning_inventory,
)
from .measure import detect_objects
from .multimodal import llm_processing
from .pipeline import finalizing_output_format, pipeline
from .planner import compile_transfer_groups, expand_blockers
from .safety import (
    explicit_refusal,
    named_pick_candidates,
    ordinal_pick_constraint,
    relative_neighbor_constraint,
    stationary_blocker_refusal,
    unresolved_deictic,
)
from .scene import extract_instructions, load_scenes
from .validation import validate_inventory, validate_plan

ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {
            "objects": [
                {
                    "id": "cube",
                    "name": "red_cube",
                    "pixel": [1500, 900],
                    "tray": "grey_tray",
                    "held_by_human": False,
                },
                {
                    "id": "gear",
                    "name": "red_gear",
                    "pixel": [1450, 950],
                    "tray": "grey_tray",
                    "held_by_human": False,
                },
                {
                    "id": "tray",
                    "name": "black_tray",
                    "pixel": [1800, 1000],
                    "tray": "black_tray",
                    "held_by_human": False,
                },
            ],
            "relations": [
                {"subject": "cube", "relation": "on_top_of", "object": "gear"}
            ],
        }

    def plan(self, actions):
        return validate_plan(
            [{"instruction_id": 1, "actions_data": actions}],
            self.inventory,
            1,
            2448,
            2048,
        )

    def test_unblock_and_preserve_original_pixels(self):
        actions = [
            {"action": k, "object": o}
            for k, o in [
                ("pick", "cube"),
                ("place", "tray"),
                ("pick", "gear"),
                ("place", "tray"),
            ]
        ]
        result = finalizing_output_format("scene_01", self.plan(actions))[0]
        self.assertEqual(result["target_pixel"], [1500, 900])
        jsonschema.validate(
            result, json.loads((ROOT / "schema/result.schema.json").read_text())
        )

    def test_invalid_plans_rejected(self):
        for actions in (
            [{"action": "pick", "object": "gear"}],
            [{"action": "place", "object": "tray"}],
            [{"action": "pick", "object": "missing"}],
            [{"action": "pick", "object": "tray"}],
            [
                {"action": "pick", "object": "cube"},
                {"action": "pick", "object": "gear"},
            ],
        ):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                self.plan(actions)
        self.inventory["objects"][0]["held_by_human"] = True
        with self.assertRaises(ValueError):
            self.plan([{"action": "pick", "object": "cube"}])

    def test_ask_refuse_schema(self):
        for kind, field in [("ask_user", "question"), ("refuse", "reason")]:
            result = finalizing_output_format(
                "scene_01", self.plan([{"action": kind, field: "Explain."}])
            )[0]
            self.assertIsNone(result["target_pixel"])
            jsonschema.validate(
                result, json.loads((ROOT / "schema/result.schema.json").read_text())
            )

    def test_measurement_overrides_model_coordinate(self):
        raw = {
            "objects": [
                dict(
                    self.inventory["objects"][0],
                    candidate_id="obj_1",
                    pixel=[1, 1],
                    bbox=[1400, 800, 1600, 1000],
                )
            ],
            "relations": [],
        }
        result = validate_inventory(
            raw, [{"id": "obj_1", "pixel": [1500, 900]}], 2448, 2048
        )
        self.assertEqual(result["objects"][0]["pixel"], [1500, 900])

    def test_grouped_transfer_compiles_to_single_gripper(self):
        self.inventory["relations"] = []
        result = self.plan(
            [
                {"action": "pick", "object": "cube"},
                {"action": "pick", "object": "gear"},
                {"action": "place", "object": "tray"},
                {"action": "place", "object": "tray"},
            ]
        )
        self.assertEqual(
            [a["action"] for a in result[0]["actions_data"]],
            ["pick", "place", "pick", "place"],
        )

    def test_replacing_blocker_keeps_support_blocked(self):
        with self.assertRaises(ValueError):
            self.plan(
                [
                    {"action": "pick", "object": "cube"},
                    {"action": "place", "object": "gear"},
                    {"action": "pick", "object": "gear"},
                ]
            )

    def test_local_grounding_avoids_gear_hole(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 55, (0, 0, 220), -1)
        cv2.circle(image, (100, 100), 20, (0, 0, 0), -1)
        box, point, method = ground_object(
            image, {"id": "gear", "name": "red_gear", "bbox": [200, 200, 800, 800]}, []
        )
        self.assertEqual(method, "local_foreground")
        self.assertGreater(int(image[point[1], point[0], 2]), 100)
        self.assertGreater(np.linalg.norm(np.array(point) - [100, 100]), 20)

    def test_common_detection_schema_is_normalized(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (100, 100), (0, 0, 220), -1)
        raw = [{"label": "red_cube", "bbox_2d": "[200,200,600,600]"}]
        inv = ground_inventory(raw, image, [])
        self.assertEqual(inv["objects"][0]["id"], "red_cube")
        self.assertEqual(inv["objects"][0]["tray"], "unknown")

    def test_dark_tray_reference_requires_dark_image_evidence(self):
        grey = {"id": "grey_tray", "name": "grey_tray", "bbox": [0, 0, 100, 200]}
        bright = np.full((200, 200, 3), 220, dtype=np.uint8)
        inv = add_dark_tray_reference(
            {"objects": [dict(grey)], "relations": []}, bright
        )
        self.assertEqual(len(inv["objects"]), 1)
        bright[:, 100:] = 20
        inv = add_dark_tray_reference(
            {"objects": [dict(grey)], "relations": []}, bright
        )
        self.assertEqual(inv["objects"][1]["name"], "black_tray")

    def test_distance_ranking_excludes_containers(self):
        inv = self.inventory
        inv["objects"][0]["bbox"] = [1400, 800, 1600, 1000]
        inv["objects"][1]["bbox"] = [1400, 900, 1500, 1000]
        inv["objects"][2]["bbox"] = [1700, 900, 1900, 1100]
        facts = planning_inventory(inv)
        self.assertEqual(facts["nearest_to_farthest"]["cube"][0]["id"], "gear")
        self.assertNotIn("tray", facts["eligible_pick_ids"])

    def test_explicit_motion_constraints(self):
        for text in (
            "throw the component into the tray",
            "push the block off the stack",
            "pull the bottom part from under the upper part",
            "take the cup out of my hand",
            "move the orange to the plate and leave it where it is",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(explicit_refusal(text))
        for text in (
            "put the banana on the plate",
            "pick up the gear on the left",
            "move the gears from one tray to the other",
        ):
            self.assertIsNone(explicit_refusal(text))

    def test_direction_relations_are_not_stack_blockers(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (30, 30), (70, 70), (0, 0, 220), -1)
        cv2.rectangle(image, (130, 30), (170, 70), (0, 0, 220), -1)
        raw = {
            "objects": [
                {"id": "cube_a", "name": "red_cube", "bbox": [100, 100, 400, 400]},
                {"id": "cube_b", "name": "red_cube", "bbox": [600, 100, 900, 400]},
            ],
            "relations": [
                {"subject": "cube_a", "relation": "left_of", "object": "cube_b"}
            ],
        }
        inv = ground_inventory(raw, image, [])
        self.assertEqual(inv["relations"], [])

    def test_json_parser_ignores_trailing_explanation(self):
        reply = '```json\n[{"instruction_id":1,"actions_data":[]}]\n```\nReasoning: [something else]'
        self.assertEqual(
            parse_model_json(reply), [{"instruction_id": 1, "actions_data": []}]
        )

    def test_transfer_clears_visible_blocker_first(self):
        raw = [
            {
                "instruction_id": 1,
                "actions_data": [
                    {"action": "pick", "object": "gear"},
                    {"action": "place", "object": "tray"},
                ],
            }
        ]
        compiled = expand_blockers(raw, self.inventory)
        result = validate_plan(compiled, self.inventory, 1, 2448, 2048)
        self.assertEqual(
            [a["object"] for a in result[0]["actions_data"]],
            ["cube", "tray", "gear", "tray"],
        )
        self.assertEqual(raw[0]["actions_data"][0]["object"], "gear")

    def test_protected_support_and_cardinality(self):
        self.assertIsNotNone(
            stationary_blocker_refusal(
                "pick up the red gear without moving the red cube", self.inventory
            )
        )
        self.assertIsNone(
            stationary_blocker_refusal("put the red gear in the tray", self.inventory)
        )
        constraint = named_pick_candidates("pick up the red cube", self.inventory)
        self.assertEqual(constraint["matching_ids"], ["cube"])
        self.assertIsNone(
            named_pick_candidates("pick up the red gear on the left", self.inventory)
        )

    def test_cyclic_stack_cannot_compile(self):
        self.inventory["relations"].append(
            {"subject": "gear", "relation": "on_top_of", "object": "cube"}
        )
        with self.assertRaises(ValueError):
            expand_blockers(
                [{"actions_data": [{"action": "pick", "object": "gear"}]}],
                self.inventory,
            )

    def test_broadcast_transfer_clears_blocker_before_any_support(self):
        inv = self.inventory
        inv["objects"].append(
            {
                "id": "gear2",
                "name": "red_gear",
                "pixel": [1200, 950],
                "bbox": [1150, 900, 1250, 1000],
                "tray": "grey_tray",
                "held_by_human": False,
            }
        )
        actions = compile_transfer_groups(
            [
                {"action": "pick", "object": "gear2"},
                {"action": "pick", "object": "gear"},
                {"action": "place", "object": "tray"},
            ]
        )
        compiled = expand_blockers(
            [{"instruction_id": 1, "actions_data": actions}], inv
        )
        result = validate_plan(compiled, inv, 1, 2448, 2048)
        self.assertEqual(
            [a["object"] for a in result[0]["actions_data"]],
            ["cube", "tray", "gear2", "tray", "gear", "tray"],
        )

    def test_deictic_reference_needs_clarification(self):
        self.assertTrue(unresolved_deictic("give me that one"))
        self.assertTrue(unresolved_deictic("pick this object"))
        self.assertFalse(unresolved_deictic("pick the red gear on the left"))

    def test_ordinal_is_counted_from_requested_side(self):
        inv = {
            "objects": [
                {
                    "id": f"gear_{i}",
                    "name": "red_gear",
                    "bbox": [i * 100, 10, i * 100 + 50, 60],
                }
                for i in (1, 2, 3)
            ]
        }
        constraint = ordinal_pick_constraint(
            "pick up the second red gear from the right", inv
        )
        self.assertEqual(constraint["ordered_ids"], ["gear_3", "gear_2", "gear_1"])
        self.assertEqual(constraint["target_id"], "gear_2")
        missing = ordinal_pick_constraint(
            "pick up the fourth red gear from the left", inv
        )
        self.assertIsNone(missing["target_id"])

    def test_relative_neighbor_uses_measured_ranking(self):
        facts = {
            "objects": [{"id": "yellow", "name": "yellow_gear"}],
            "nearest_left_of": {"yellow": ["cube", "mug", "gear"]},
            "nearest_right_of": {"yellow": []},
        }
        constraint = relative_neighbor_constraint(
            "pick up the object to the left of the yellow gear", facts
        )
        self.assertEqual(constraint["target_id"], "cube")
        missing = relative_neighbor_constraint(
            "pick up the object to the right of the yellow gear", facts
        )
        self.assertEqual(missing["required_decision"], "refuse")

    def test_unlabelled_instructions_keep_the_verb(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "instructions.txt"
            path.write_text(
                "# header\nsimple\tpick up the cube\npick up the gear\nrelative pick the left gear\n"
            )
            self.assertEqual(
                extract_instructions(path),
                "- pick up the cube\n- pick up the gear\n- pick the left gear\n",
            )

    def test_all_ten_scenes_foreground_points(self):
        scenes = load_scenes(ROOT / "dataset")
        self.assertEqual(len(scenes), 10)
        self.assertEqual(
            sum(len(s["instruction_data"].splitlines()) for s in scenes.values()), 50
        )
        for name, scene in scenes.items():
            image = cv2.imread(str(scene["image_scene_path"]))
            objects, mask = detect_objects(image)
            self.assertTrue(objects, name)
            for obj in objects:
                x, y = obj["pixel"]
                x1, y1, x2, y2 = obj["bbox"]
                self.assertTrue(x1 <= x < x2 and y1 <= y < y2)
                self.assertEqual(int(mask[y, x]), 255, (name, obj["id"]))


class FakeApertusClient:
    """Minimal SDK stand-in that records requests and never opens a connection."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.base_url = "https://example.invalid/v1/"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        reply = next(self.replies)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=reply), finish_reason="stop"
                )
            ]
        )


class IntegrationTests(unittest.TestCase):
    """Exercise orchestration, caching, and checkpoints without a live model."""

    def create_scene(self, folder, name="scene_01"):
        scene = Path(folder) / name
        scene.mkdir()
        image = np.full((500, 700, 3), 90, dtype=np.uint8)
        image[:, 350:] = 20
        cv2.rectangle(image, (70, 100), (209, 249), (0, 0, 220), -1)
        cv2.imwrite(str(scene / "rgb.png"), image)
        (scene / "instructions.txt").write_text("simple\tpick up the red cube\n")
        return scene

    def perception_reply(self):
        objects = [
            {
                "id": "grey_tray",
                "name": "grey_tray",
                "bbox": [0, 0, 500, 900],
                "tray": "grey_tray",
                "held_by_human": False,
            },
            {
                "id": "black_tray",
                "name": "black_tray",
                "bbox": [500, 0, 1000, 900],
                "tray": "black_tray",
                "held_by_human": False,
            },
            {
                "id": "red_cube",
                "name": "red_cube",
                "bbox": [100, 200, 300, 500],
                "tray": "grey_tray",
                "held_by_human": False,
            },
        ]
        return json.dumps({"objects": objects, "relations": []})

    def test_scene_orchestration_and_cache_reuse(self):
        plan = json.dumps(
            [
                {
                    "instruction_id": 1,
                    "actions_data": [{"action": "pick", "object": "red_cube"}],
                }
            ]
        )
        with tempfile.TemporaryDirectory() as folder:
            scene = self.create_scene(folder)
            cache = Path(folder) / "cache"
            client = FakeApertusClient([self.perception_reply(), plan])
            result = llm_processing(
                "- pick up the red cube\n",
                scene / "rgb.png",
                client=client,
                model="test-apertus",
                cache_dir=cache,
                debug_dir=Path(folder) / "trace",
            )
            warmed_client = FakeApertusClient([plan])
            cached = llm_processing(
                "- pick up the red cube\n",
                scene / "rgb.png",
                client=warmed_client,
                model="test-apertus",
                cache_dir=cache,
            )
            self.assertEqual(result, cached)
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(len(warmed_client.requests), 1)
            self.assertTrue((Path(folder) / "trace/trace.json").exists())
            formatted = finalizing_output_format("scene_01", result)[0]
            x, y = formatted["target_pixel"]
            self.assertTrue(70 <= x < 210 and 100 <= y < 250)
            jsonschema.validate(
                formatted, json.loads((ROOT / "schema/result.schema.json").read_text())
            )

    def test_repeated_invalid_reply_stops_repair(self):
        client = FakeApertusClient(["{}", "{}"])
        artifacts = {}
        request = make_json_request(client, "test-apertus", artifacts)

        def reject(raw):
            raise ValueError("Missing objects")

        with self.assertRaisesRegex(ValueError, "repeated an invalid response"):
            request(
                "Return visible objects",
                [{"type": "text", "text": "scene"}],
                reject,
                "inventory",
            )
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(artifacts["inventory_error_0"], "Missing objects")

    def test_partial_scene_failure_preserves_ordered_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            self.create_scene(folder, "scene_01")
            self.create_scene(folder, "scene_02")
            output = Path(folder) / "results.json"

            def predict(instruction, image_path, **kwargs):
                if image_path.parent.name == "scene_02":
                    raise RuntimeError("Simulated perception failure")
                return [
                    {
                        "instruction_id": 1,
                        "actions_data": [
                            {
                                "action": "pick",
                                "object": "red_cube",
                                "pixel": [100, 100],
                            }
                        ],
                    }
                ]

            with patch(
                "pipelines.path_b_wahyu.pipeline.llm_processing", side_effect=predict
            ):
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    pipeline(folder, output, workers=2)
            entries = json.loads(output.read_text())["results"]
            self.assertEqual(
                [e["instruction_id"] for e in entries], ["scene_01_i1", "scene_02_i1"]
            )
            self.assertEqual(entries[0]["target_pixel"], [100, 100])
            self.assertEqual(entries[1]["actions"][0]["action"], "ask_user")
            self.assertIn(
                "scene_02", json.loads(Path(str(output) + ".errors.json").read_text())
            )


if __name__ == "__main__":
    unittest.main()
