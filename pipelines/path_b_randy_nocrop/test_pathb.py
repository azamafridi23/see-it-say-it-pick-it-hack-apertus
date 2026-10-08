"""Tests at the run() seam and the Apertus client, with fakes in place of
Apertus.

    .venv/bin/python -m unittest pipelines/path_b_randy_nocrop/test_pathb.py
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import answer_type  # noqa: E402
import apertus  # noqa: E402
import detect  # noqa: E402
import pipeline  # noqa: E402
import safety  # noqa: E402

DATA = ROOT / "dataset"
GT_FILE = DATA / "ground_truth" / "ground_truth.json"
GT = json.loads(GT_FILE.read_text())["instructions"]


def flaky(prompt, images, schema, temperature=0):
    if "red cube" in prompt:
        raise TimeoutError("model too slow")

    return "sorry, I am not sure {"


def one_scene(scene, tmp):
    """Make `tmp` a scenes folder holding only `scene`."""
    d = Path(tmp)
    (d / scene).symlink_to(DATA / scene)
    (d / "calibration.json").symlink_to(DATA / "calibration.json")
    return d


class PipelineSeam(unittest.TestCase):
    def test_every_instruction_gets_a_schema_valid_answer(self):
        sub = pipeline.run(DATA, apertus.mock)
        got = {r["instruction_id"]: r for r in sub["results"]}
        self.assertEqual(set(got), {g["instruction_id"] for g in GT})
        for r in got.values():
            self.assertEqual(answer_type(r), ("refuse", None))

    def test_timeouts_and_garbage_still_give_schema_valid_answers(self):
        with tempfile.TemporaryDirectory() as t:
            sub = pipeline.run(one_scene("scene_06", t), flaky)
        self.assertEqual(len(sub["results"]), 5)
        for r in sub["results"]:
            self.assertIsNone(answer_type(r)[1], r)


class FakeServer(BaseHTTPRequestHandler):
    """An OpenAI-style endpoint: busy for the first `busy` requests,
    slow, and counting how many requests are in flight at once."""
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
        pass


class ApertusClient(unittest.TestCase):
    def serve(self, busy=0, reply="ok"):
        handler = type("Server", (FakeServer,), {
            "busy": busy, "seen": 0, "in_flight": 0, "most": 0,
            "lock": threading.Lock(), "reply": reply, "bodies": []})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}/v1"
        return handler, url

    def setUp(self):
        waits = apertus.RETRY_WAITS
        apertus.RETRY_WAITS = (0.01,) * len(waits)
        self.addCleanup(setattr, apertus, "RETRY_WAITS", waits)

    def test_too_many_requests_is_retried(self):
        server, url = self.serve(busy=2)
        call = apertus.apertus(url, "m", 64, 5)
        self.assertEqual(call("hi", [], {"type": "object"}), "ok")
        self.assertEqual(server.seen, 3)

    def test_no_more_calls_in_flight_than_allowed(self):
        server, url = self.serve()
        call = apertus.apertus(url, "m", 64, 5, parallel=3)
        with ThreadPoolExecutor(12) as pool:
            for _ in range(12):
                pool.submit(call, "hi", [], {"type": "object"})

        self.assertEqual(server.seen, 12)
        self.assertLessEqual(server.most, 3)

    def test_thinking_turns_on_the_template_and_drops_the_schema(self):
        # A JSON schema would make Apertus skip its thinking.
        server, url = self.serve(reply="<|inner_prefix|>hmm, the cube"
                                       "<|inner_suffix|>Move the cube.")
        think = apertus.apertus(url, "m", 64, 5, thinking=True)
        self.assertEqual(think("hi", [], None), "Move the cube.")
        sent = server.bodies[0]
        self.assertEqual(sent["chat_template_kwargs"],
                         {"enable_thinking": True})
        self.assertNotIn("response_format", sent)
        self.assertGreaterEqual(sent["max_tokens"], 2000)

    def test_cut_off_thinking_is_a_failed_call(self):
        _, url = self.serve(reply="<|inner_prefix|>hmm, the cube is")
        think = apertus.apertus(url, "m", 64, 5, thinking=True)
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
    """Return a fake that answers the box call with `boxes` (scene_01's by
    default), the pre-check with `decision`, and the planner with
    `replies` in turn (the last one repeats)."""
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
            n = prompt.count("YOUR ANSWER")

        return json.dumps(replies[min(n, len(replies) - 1)])

    model.seen = seen
    return model


def moves(*pairs):
    return {"thinking": "...", "decision": "plan", "message": "",
            "moves": [{"object": o, "to": t} for o, t in pairs]}


def scene_ids(model):
    """The object ids the planner was allowed to name."""
    return set(model.seen["schemas"][0]["properties"]["moves"]["items"][
        "properties"]["object"]["enum"])


class Detection(unittest.TestCase):
    def ids_for(self, boxes):
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


class SafetyCheck(unittest.TestCase):
    def results(self, model, **switches):
        with tempfile.TemporaryDirectory() as t:
            sub = pipeline.run(one_scene("scene_01", t), model, **switches)

        return [r["actions"][0] for r in sub["results"]]

    def test_an_unsafe_instruction_is_refused_before_planning(self):
        model = planning(moves(("red_cube", None)), decision="refuse")
        first = self.results(model)
        self.assertTrue(all(a["action"] == "refuse" for a in first))
        self.assertIn("throwing is uncontrolled", first[0]["reason"])
        self.assertEqual(model.seen["prompts"], [])

    def test_a_safe_instruction_goes_on_to_the_planner(self):
        model = planning(moves(("red_cube", None)))
        first = self.results(model)
        self.assertTrue(all(a["action"] == "pick" for a in first))
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
    def run_scene_01(self, model, **switches):
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
        retried = [p for p in model.seen["prompts"] if "YOUR ANSWER" in p]
        self.assertTrue(retried)
        self.assertIn('"to" is where mug is put down', retried[0])
        self.assertEqual(got[0]["actions"][1]["object"], "black_tray")

    def test_after_the_retries_it_refuses(self):
        got = self.run_scene_01(planning(moves(("mug", "mug"))))
        self.assertEqual(got[0]["actions"][0]["action"], "refuse")
        self.assertIn("No valid plan", got[0]["actions"][0]["reason"])

    def test_apertus_finds_the_objects_opencv_does_not(self):
        # A box on the empty black mat: no OpenCV mark there.
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
        retried = [p for p in model.seen["prompts"] if "YOUR ANSWER" in p]
        self.assertTrue(retried)
        self.assertNotIn("Hold the mug", retried[0])
        self.assertEqual(got[0]["actions"][1]["object"], "black_tray")

    def test_a_failed_thinking_call_still_answers(self):
        def thinker(prompt, images, schema, temperature=0):
            raise OSError("timed out")

        got = self.run_scene_01(planning(moves(("red_cube", None))),
                                thinker=thinker)
        self.assertEqual(got[0]["actions"][0]["object"], "red_cube")
