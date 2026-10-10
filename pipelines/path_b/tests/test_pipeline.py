"""Tests for Path B, with fakes in place of Apertus.

The pipeline is tested through ``pipeline.run()``, the Apertus client
against a fake HTTP server. No real server is needed:

    .venv/bin/python -m unittest discover pipelines/path_b_randy_nocrop/tests
"""
import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
ROOT = PIPELINE.parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(PIPELINE))
from evaluate import answer_type  # noqa: E402
import apertus  # noqa: E402
import detect  # noqa: E402
import pipeline  # noqa: E402
import safety  # noqa: E402

DATA = ROOT / "dataset"
GT_FILE = DATA / "ground_truth" / "ground_truth.json"
GT = json.loads(GT_FILE.read_text())["instructions"]


def nothing(prompt, images, schema, temperature=0):
    """A model that finds nothing: every instruction ends refused."""
    return "{}"


def flaky(prompt, images, schema, temperature=0):
    """A model that times out on one instruction and talks nonsense."""
    if "red cube" in prompt:
        raise TimeoutError("model too slow")

    return "sorry, I am not sure {"


def one_scene(scene, tmp):
    """Make ``tmp`` a scenes folder holding only ``scene``."""
    folder = Path(tmp)
    (folder / scene).symlink_to(DATA / scene)
    (folder / "calibration.json").symlink_to(DATA / "calibration.json")
    return folder


class PipelineSeam(unittest.TestCase):
    """Every instruction gets a valid answer, whatever the model does."""

    def test_every_instruction_gets_a_schema_valid_answer(self):
        sub = pipeline.run(DATA, nothing)
        got = {result["instruction_id"]: result
               for result in sub["results"]}
        self.assertEqual(set(got), {truth["instruction_id"] for truth in GT})
        for result in got.values():
            self.assertEqual(answer_type(result), ("refuse", None))

    def test_timeouts_and_garbage_still_give_schema_valid_answers(self):
        with tempfile.TemporaryDirectory() as t:
            sub = pipeline.run(one_scene("scene_06", t), flaky)
        self.assertEqual(len(sub["results"]), 5)
        for result in sub["results"]:
            self.assertIsNone(answer_type(result)[1], result)


class FakeServer(BaseHTTPRequestHandler):
    """A slow OpenAI-style endpoint that records what it is sent.

    It answers 429 to the first ``busy`` requests, replies ``reply`` to
    the others, and counts the most requests in flight at once.
    """
    busy = 0
    seen = 0
    in_flight = 0
    most = 0
    lock = threading.Lock()
    reply = "ok"
    bodies = []

    def do_POST(self):
        sent = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        with cls.lock:
            cls.bodies.append(sent)
            cls.seen += 1
            refuse = cls.seen <= cls.busy
            cls.in_flight += 1
            cls.most = max(cls.most, cls.in_flight)

        time.sleep(0.05)
        with cls.lock:
            cls.in_flight -= 1

        if refuse:
            self.send_response(429)
            self.end_headers()
            return

        body = json.dumps({"choices": [{"message": {"content": cls.reply}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        """Keep the test output quiet."""


class ApertusClient(unittest.TestCase):
    """The client against a fake server: retries, limits, thinking."""

    def serve(self, busy=0, reply="ok"):
        """Start a fake server; return its handler class and its URL."""
        handler = type("Server", (FakeServer,), {
            "busy": busy, "seen": 0, "in_flight": 0, "most": 0,
            "lock": threading.Lock(), "reply": reply, "bodies": []})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}/v1"
        return handler, url

    def setUp(self):
        """Make the retry waits short for the test."""
        waits = apertus.RETRY_WAITS
        apertus.RETRY_WAITS = (0.01,) * len(waits)
        self.addCleanup(setattr, apertus, "RETRY_WAITS", waits)

    def test_too_many_requests_is_retried(self):
        server, url = self.serve(busy=2)
        call = apertus.ApertusClient(url, "m", 64, 5)
        self.assertEqual(call("hi", [], {"type": "object"}), "ok")
        self.assertEqual(server.seen, 3)

    def test_no_more_calls_in_flight_than_allowed(self):
        server, url = self.serve()
        call = apertus.ApertusClient(url, "m", 64, 5, parallel=3)
        with ThreadPoolExecutor(12) as pool:
            for _ in range(12):
                pool.submit(call, "hi", [], {"type": "object"})

        self.assertEqual(server.seen, 12)
        self.assertLessEqual(server.most, 3)

    def test_thinking_turns_on_the_template_and_drops_the_schema(self):
        # A JSON schema would make Apertus skip its thinking.
        server, url = self.serve(reply="<|inner_prefix|>hmm, the cube"
                                       "<|inner_suffix|>Move the cube.")
        think = apertus.ApertusClient(url, "m", 64, 5, thinking=True)
        self.assertEqual(think("hi", [], None), "Move the cube.")
        sent = server.bodies[0]
        self.assertEqual(sent["chat_template_kwargs"],
                         {"enable_thinking": True})
        self.assertNotIn("response_format", sent)
        self.assertGreaterEqual(sent["max_tokens"], 2000)

    def test_cut_off_thinking_is_a_failed_call(self):
        _, url = self.serve(reply="<|inner_prefix|>hmm, the cube is")
        think = apertus.ApertusClient(url, "m", 64, 5, thinking=True)
        self.assertEqual(apertus.ask(think, "hi", [], None), "")


# Apertus's boxes for scene_01 (0-1000), as it answered on the hosted 70B:
# they sit a little high, as all its boxes do.
SCENE_01_BOXES = [
    {"kind": "red cube", "bounding_box": [70, 370, 225, 555]},
    {"kind": "yellow gear", "bounding_box": [295, 335, 385, 435]},
    {"kind": "mug", "bounding_box": [260, 140, 460, 305]},
]
GREY_TRAY_BOX = {"kind": "grey part", "bounding_box": [0, 30, 525, 830]}


def planning(*replies, boxes=None, decision="go on"):
    """Build a fake Apertus for one scene.

    Args:
        *replies: The planner's replies, in turn; the last one repeats.
        boxes: The box call's objects; scene_01's when None.
        decision: The pre-check's decision, "go on" or "refuse".

    Returns:
        The model function, with ``seen``: the planner's prompts and
        schemas, and how many pre-checks were asked.
    """
    boxes = SCENE_01_BOXES if boxes is None else boxes
    seen = {"prompts": [], "schemas": [], "prechecks": 0}
    lock = threading.Lock()

    def model(prompt, images, schema, temperature=0):
        if schema is detect.BOX_SCHEMA:
            return json.dumps({"objects": boxes})

        if schema is safety.PRECHECK_SCHEMA:
            with lock:
                seen["prechecks"] += 1

            return json.dumps({"reason": "throwing is uncontrolled",
                               "decision": decision})

        with lock:
            seen["prompts"].append(prompt)
            seen["schemas"].append(schema)
            retry = prompt.count("YOUR ANSWER")

        return json.dumps(replies[min(retry, len(replies) - 1)])

    model.seen = seen
    return model


def moves(*pairs):
    """Build a planner reply that plans ``(object, to)`` moves."""
    return {"thinking": "...", "decision": "plan", "message": "",
            "moves": [{"object": object_id, "to": to}
                      for object_id, to in pairs]}


def scene_ids(model):
    """Return the object ids the planner was allowed to name."""
    return set(model.seen["schemas"][0]["properties"]["moves"]["items"][
        "properties"]["object"]["enum"])


class Detection(unittest.TestCase):
    """Apertus's boxes become the scene's objects."""

    def ids_for(self, boxes):
        """Return the object ids the planner sees for these boxes."""
        model = planning(moves(("grey_tray", None)), boxes=boxes)
        with tempfile.TemporaryDirectory() as t:
            pipeline.run(one_scene("scene_01", t), model)

        return scene_ids(model)

    def test_each_box_is_one_object(self):
        self.assertEqual(self.ids_for(SCENE_01_BOXES),
                         {"red_cube", "yellow_gear", "mug", "grey_tray",
                          "black_tray"})

    def test_a_tray_sized_box_is_no_object(self):
        self.assertEqual(self.ids_for([GREY_TRAY_BOX]),
                         {"grey_tray", "black_tray"})

    def test_a_kind_off_the_list_is_no_object(self):
        black = {"kind": "black part", "bounding_box": [70, 370, 225, 555]}
        self.assertEqual(self.ids_for([black]), {"grey_tray", "black_tray"})

    def test_a_long_narrow_part_is_a_grey_part(self):
        # 139 x 45 mm on the table, whatever Apertus called it.
        bar = {"kind": "small white part",
               "bounding_box": [560, 560, 790, 650]}
        self.assertIn("grey_part", self.ids_for([bar]))

    def test_a_broad_part_is_a_white_part(self):
        # 150 x 100 mm.
        sheet = {"kind": "grey part", "bounding_box": [560, 560, 810, 760]}
        self.assertIn("white_part", self.ids_for([sheet]))

    def test_a_small_part_is_a_small_white_part(self):
        # 78 x 50 mm.
        block = {"kind": "white part", "bounding_box": [560, 560, 690, 660]}
        self.assertIn("small_white_part", self.ids_for([block]))

    def test_a_box_touching_the_photo_edge_is_no_object(self):
        # A rail or a tray rim, never an object on the table.
        rail = {"kind": "small white part", "bounding_box": [0, 40, 530, 70]}
        self.assertEqual(self.ids_for([rail]), {"grey_tray", "black_tray"})

    def test_a_hand_may_touch_the_photo_edge(self):
        hand = {"kind": "hand", "bounding_box": [600, 0, 800, 200]}
        self.assertIn("hand", self.ids_for([hand]))


class SafetyCheck(unittest.TestCase):
    """Unsafe instructions are refused before planning."""

    def results(self, model, **switches):
        """Run scene_01; return each answer's first action."""
        with tempfile.TemporaryDirectory() as t:
            sub = pipeline.run(one_scene("scene_01", t), model, **switches)

        return [result["actions"][0] for result in sub["results"]]

    def test_an_unsafe_instruction_is_refused_before_planning(self):
        model = planning(moves(("red_cube", None)), decision="refuse")
        first = self.results(model)
        self.assertTrue(all(action["action"] == "refuse" for action in first))
        self.assertIn("throwing is uncontrolled", first[0]["reason"])
        self.assertEqual(model.seen["prompts"], [])

    def test_a_safe_instruction_goes_on_to_the_planner(self):
        model = planning(moves(("red_cube", None)))
        first = self.results(model)
        self.assertTrue(all(action["action"] == "pick" for action in first))
        self.assertEqual(model.seen["prechecks"], 5)

    def test_safety_off_asks_no_precheck(self):
        model = planning(moves(("red_cube", None)), decision="refuse")
        self.results(model, safety=False)
        self.assertEqual(model.seen["prechecks"], 0)

    def test_picking_a_plate_is_corrected_to_a_refusal(self):
        plate = [dict(SCENE_01_BOXES[0], kind="plate")]
        model = planning(moves(("plate", "black_tray")),
                         {"thinking": "...", "decision": "refuse",
                          "message": "A plate cannot be picked up.",
                          "moves": []}, boxes=plate)
        first = self.results(model)
        self.assertEqual(first[0], {"action": "refuse", "reason":
                                    "A plate cannot be picked up."})


class Planner(unittest.TestCase):
    """The scene graph, the planner, its checker and thinking."""

    def run_scene_01(self, model, **switches):
        """Run scene_01; return the answers."""
        with tempfile.TemporaryDirectory() as t:
            sub = pipeline.run(one_scene("scene_01", t), model,
                               **switches)

        return sub["results"]

    def test_a_move_becomes_a_pick_and_a_place_on_the_tray(self):
        got = self.run_scene_01(planning(moves(("red_cube", "black_tray"))))
        self.assertEqual(got[0]["actions"], [
            {"action": "pick", "object": "red_cube"},
            {"action": "place", "object": "black_tray"}])

    def test_apertus_can_only_name_this_scenes_ids(self):
        model = planning(moves(("red_cube", None)))
        self.run_scene_01(model)
        ids = model.seen["schemas"][0]["properties"]["moves"]["items"][
            "properties"]["object"]["enum"]
        self.assertEqual(set(ids), {"red_cube", "yellow_gear", "mug",
                                    "grey_tray", "black_tray"})

    def test_the_graph_says_which_tray_each_object_is_in(self):
        model = planning(moves(("red_cube", None)))
        self.run_scene_01(model)
        self.assertIn('"id": "red_cube", "type": "red cube"',
                      model.seen["prompts"][0])
        self.assertIn('"in": "grey_tray"', model.seen["prompts"][0])

    def test_a_broken_rule_goes_back_to_apertus_to_correct(self):
        model = planning(moves(("mug", "mug")), moves(("mug", "black_tray")))
        got = self.run_scene_01(model)
        retried = [prompt for prompt in model.seen["prompts"]
                   if "YOUR ANSWER" in prompt]
        self.assertTrue(retried)
        self.assertIn('"to" is where mug is put down', retried[0])
        self.assertEqual(got[0]["actions"][1]["object"], "black_tray")

    def test_after_the_retries_it_refuses(self):
        got = self.run_scene_01(planning(moves(("mug", "mug"))))
        self.assertEqual(got[0]["actions"][0]["action"], "refuse")
        self.assertIn("No valid plan", got[0]["actions"][0]["reason"])

    def test_apertus_finds_the_objects_opencv_does_not(self):
        # A box on the empty black mat, where no colour blob would be.
        banana = {"kind": "banana", "bounding_box": [700, 600, 800, 700]}
        model = planning(moves(("banana", None)),
                         boxes=SCENE_01_BOXES + [banana])
        got = self.run_scene_01(model)
        self.assertEqual(got[0]["actions"][0]["object"], "banana")

    def test_target_pixel_is_the_corrected_box_centre(self):
        # The red cube's box [70, 370, 225, 555] on 2448 x 2048: x 171-551,
        # y 758-1137, moved down to y 847-1251 (1.065 y + 40).
        got = self.run_scene_01(planning(moves(("red_cube", None))))
        self.assertEqual(got[0]["target_pixel"], [361, 1049])

    def test_the_object_whose_colour_shows_in_the_overlap_is_on_top(self):
        # Apertus's boxes for scene_07: the mug stands on the red gear.
        boxes = [{"kind": "red gear", "bounding_box": [130, 170, 275, 310]},
                 {"kind": "mug", "bounding_box": [165, 225, 355, 405]}]
        model = planning(moves(("mug", None)), boxes=boxes)
        with tempfile.TemporaryDirectory() as t:
            pipeline.run(one_scene("scene_07", t), model)

        self.assertIn('"covered_by": ["mug"]', model.seen["prompts"][0])

    def test_the_planner_thinks_first_then_answers_with_it(self):
        thoughts = []

        def thinker(prompt, images, schema, temperature=0):
            thoughts.append((prompt, images, schema))
            return "The red cube goes to the black tray."

        model = planning(moves(("red_cube", "black_tray")))
        self.run_scene_01(model, thinker=thinker)
        self.assertEqual(len(thoughts), 5)
        prompt, images, schema = thoughts[0]
        self.assertIn("INSTRUCTION", prompt)
        self.assertEqual((images, schema), ([], None))
        self.assertIn("The red cube goes to the black tray.",
                      model.seen["prompts"][0])

    def test_a_retry_drops_the_thinking_that_led_to_the_error(self):
        # A wrong conclusion in the thinking would anchor every retry.
        def thinker(prompt, images, schema, temperature=0):
            return "Hold the mug, then move it: mug to null, mug to tray."

        model = planning(moves(("mug", "mug")), moves(("mug", "black_tray")))
        got = self.run_scene_01(model, thinker=thinker)
        retried = [prompt for prompt in model.seen["prompts"]
                   if "YOUR ANSWER" in prompt]
        self.assertTrue(retried)
        self.assertNotIn("Hold the mug", retried[0])
        self.assertEqual(got[0]["actions"][1]["object"], "black_tray")

    def test_a_failed_thinking_call_still_answers(self):
        def thinker(prompt, images, schema, temperature=0):
            raise OSError("timed out")

        got = self.run_scene_01(planning(moves(("red_cube", None))),
                                thinker=thinker)
        self.assertEqual(got[0]["actions"][0]["object"], "red_cube")
