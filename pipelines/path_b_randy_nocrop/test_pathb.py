"""Tests at the two agreed seams, with a scripted fake in place of Apertus.

    .venv/bin/python -m unittest pipelines/path_b_randy_nocrop/test_pathb.py
"""
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import SceneGT, answer_type  # noqa: E402
import apertus  # noqa: E402
import closeups  # noqa: E402
import detect  # noqa: E402
import measure  # noqa: E402
import mentions  # noqa: E402
import opencv_marks  # noqa: E402
import pipeline  # noqa: E402
import safety  # noqa: E402
import zoom  # noqa: E402

DATA = ROOT / "dataset"
GT_FILE = DATA / "ground_truth" / "ground_truth.json"
GT = json.loads(GT_FILE.read_text())["instructions"]
MARK_LINE = r"mark (\d+) at x (\d+(?:\.\d+)?)%, y (\d+(?:\.\d+)?)%"


# The Decide path (votes, mentions, grid zoom) on OpenCV marks, as before
# the planner and Apertus-found marks became the default.
decide_run = partial(pipeline.run, planner=False, box_marks=False)


def refuse_everything(prompt, images, schema, temperature=0):
    return json.dumps({"verdict": "refuse", "reason": "fake says no"})


def flaky(prompt, images, schema, temperature=0):
    if "red cube" in prompt:
        raise TimeoutError("model too slow")

    return "sorry, I am not sure {"


def oracle(scene):
    """Return a perfect Apertus for one scene.

    It 'sees' through the ground-truth masks and names the right mark.
    """
    sg = SceneGT(DATA, DATA / "ground_truth", scene)
    w, h = sg.scene.size
    gts = {g["instruction"]: g for g in GT if g["scene"] == scene}

    def model(prompt, images, schema, temperature=0):
        if "verdict" not in schema.get("properties", {}):
            # Naming, extraction or inventory: only Decide matters here.
            return json.dumps({"objects": []})

        marks = {}
        for m, x, y in re.findall(MARK_LINE, prompt):
            pixel = (round(float(x) / 100 * w), round(float(y) / 100 * h))
            marks[int(m)] = sg.object_at(pixel)

        g = next(g for t, g in gts.items() if f'"{t}"' in prompt)

        if g["expected"] != "plan":
            return json.dumps({"verdict": g["expected"],
                               "question": "which?", "reason": "no"})

        first = next(m for m, n in marks.items() if n in g["targets"])
        step = {"action": "pick", "mark": first, "object": marks[first]}
        return json.dumps({"verdict": "plan", "plan": [step]})

    return model


def scripted(*votes):
    """Return a fake whose Decide samples cycle through `votes`.

    Each vote is "refuse", "ask" or a plan given as (mark, object name).
    Calls are counted per prompt, so every instruction sees the same
    samples even when they are asked in parallel.
    """
    calls = Counter()
    lock = threading.Lock()
    seen = {}

    def model(prompt, images, schema, temperature=0):
        seen["schema"], seen["prompt"] = schema, prompt
        seen["images"] = images
        if "verdict" not in schema.get("properties", {}):
            return json.dumps({"objects": []})

        with lock:
            vote = votes[calls[prompt] % len(votes)]
            calls[prompt] += 1

        if vote == "refuse":
            return json.dumps({"verdict": "refuse", "reason": "no"})

        if vote == "ask":
            return json.dumps({"verdict": "ask", "question": "which?"})

        mark, name = vote
        step = {"action": "pick", "mark": mark, "object": name}
        return json.dumps({"verdict": "plan", "plan": [step]})

    model.seen = seen
    return model


def deepest_point(scene, name):
    """The ground-truth pixel deepest inside object `name`."""
    sg = SceneGT(DATA, DATA / "ground_truth", scene)
    oid = next(k for k, v in sg.names.items() if v == name)
    mask = (sg.labels == oid).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    v, u = np.unravel_index(int(dist.argmax()), dist.shape)
    return int(u), int(v), sg.scene.size


def unmarked_pick(scene, name, sees=True):
    """Return a fake that plans to pick unmarked `name`, then grid-zooms.

    In the grid rounds it 'sees' the object through the ground truth and
    names the cell holding it; with `sees=False` it answers null.
    """
    u, v, (w, h) = deepest_point(scene, name)
    view = r"x ([\d.]+)-([\d.]+) %.*?y ([\d.]+)-([\d.]+) %"

    def model(prompt, images, schema, temperature=0):
        props = schema.get("properties", {})
        if "objects" in props:
            return json.dumps({"objects": []})

        if "verdict" in props:
            step = {"action": "pick", "mark": None, "object": name}
            return json.dumps({"verdict": "plan", "plan": [step]})

        if "cell" not in props:
            return json.dumps({})

        x0, x1, y0, y1 = map(float, re.search(view, prompt, re.S).groups())
        col = int((100 * u / w - x0) / (x1 - x0) * zoom.GRID)
        row = int((100 * v / h - y0) / (y1 - y0) * zoom.GRID)
        inside = 0 <= col < zoom.GRID and 0 <= row < zoom.GRID
        cell = f"{chr(65 + col)}{row + 1}" if sees and inside else None
        return json.dumps({"cell": cell})

    return model


def score(sub, data):
    """Return per-instruction rows from the official scorer."""
    with tempfile.TemporaryDirectory() as t:
        res, rep = Path(t) / "r.json", Path(t) / "rep.json"
        res.write_text(json.dumps(sub))
        subprocess.run(
            [sys.executable, str(ROOT / "starter" / "evaluate.py"),
             str(res), "--data", str(data),
             "--gt", str(DATA / "ground_truth"), "--report", str(rep)],
            check=True, capture_output=True)
        rows = json.loads(rep.read_text())
    rows = rows["instructions"] if isinstance(rows, dict) else rows
    return {r["instruction_id"]: r for r in rows}


def one_scene(scene, tmp):
    """Make `tmp` a scenes folder holding only `scene`."""
    d = Path(tmp)
    (d / scene).symlink_to(DATA / scene)
    (d / "calibration.json").symlink_to(DATA / "calibration.json")
    return d


class PipelineSeam(unittest.TestCase):
    def test_every_instruction_gets_a_schema_valid_answer(self):
        sub = decide_run(DATA, refuse_everything)
        got = {r["instruction_id"]: r for r in sub["results"]}
        self.assertEqual(set(got), {g["instruction_id"] for g in GT})
        for r in got.values():
            self.assertEqual(answer_type(r), ("refuse", None))

    def test_target_pixel_is_on_the_mark_decide_names(self):
        with tempfile.TemporaryDirectory() as t:
            data = one_scene("scene_06", t)
            rows = score(decide_run(data, oracle("scene_06")), data)
        for i in (1, 2, 3, 4, 5):
            row = rows[f"scene_06_i{i}"]
            self.assertEqual(row["score"], 1.0, row)

    def test_timeouts_and_garbage_still_give_schema_valid_answers(self):
        with tempfile.TemporaryDirectory() as t:
            sub = decide_run(one_scene("scene_06", t), flaky)
        self.assertEqual(len(sub["results"]), 5)
        for r in sub["results"]:
            self.assertIsNone(answer_type(r)[1], r)


class DecideVoting(unittest.TestCase):
    def run_scene_06(self, model, **switches):
        with tempfile.TemporaryDirectory() as t:
            sub = decide_run(one_scene("scene_06", t), model, **switches)
        return sub["results"]

    def test_majority_plan_takes_the_most_common_sequence(self):
        model = scripted((1, "a"), (1, "a"), (2, "b"), "refuse", "ask")
        results = self.run_scene_06(model, votes=5)
        mark_1 = opencv_marks.find_marks(SceneGT(
            DATA, DATA / "ground_truth", "scene_06").scene.rgb)[0]
        for r in results:
            self.assertEqual(answer_type(r), ("plan", None))
            self.assertEqual(r["actions"][0]["object"], "a")
            self.assertEqual(r["target_pixel"], mark_1["pixel"])

    def test_tie_between_plan_and_not_plan_goes_to_not_plan(self):
        model = scripted((1, "a"), (1, "a"), "refuse", "refuse")
        for r in self.run_scene_06(model, votes=4):
            self.assertEqual(answer_type(r), ("refuse", None))

    def test_one_vote_takes_the_single_sample(self):
        model = scripted("refuse", (1, "a"), (1, "a"))
        for r in self.run_scene_06(model, votes=1):
            self.assertEqual(answer_type(r), ("refuse", None))

    def test_votes_and_instructions_are_asked_in_parallel(self):
        # Scene 06: 1 inventory + 5 instructions x 5 votes = 26 calls of
        # 0.2 s, 5.2 s one after another.
        slow_refusal = scripted("refuse")

        def slow(prompt, images, schema, temperature=0):
            time.sleep(0.2)
            return slow_refusal(prompt, images, schema, temperature)

        t0 = time.time()
        self.run_scene_06(slow, votes=5)
        self.assertLess(time.time() - t0, 2.0)

    def test_reasoning_comes_before_the_verdict(self):
        model = scripted("refuse")
        self.run_scene_06(model, reasoning=True)
        self.assertEqual(list(model.seen["schema"]["properties"])[0],
                         "reasoning")
        self.assertEqual(model.seen["schema"]["required"][0], "reasoning")

    def test_reasoning_off_asks_for_the_verdict_only(self):
        model = scripted("refuse")
        self.run_scene_06(model, reasoning=False)
        self.assertNotIn("reasoning", model.seen["schema"]["properties"])


class GridZoom(unittest.TestCase):
    def run_scene_04(self, model, **switches):
        with tempfile.TemporaryDirectory() as t:
            data = one_scene("scene_04", t)
            sub = decide_run(data, model, votes=1, **switches)
            return score(sub, data)["scene_04_i1"]

    def test_unmarked_first_pick_is_found_by_grid_zoom(self):
        row = self.run_scene_04(unmarked_pick("scene_04", "white_part"))
        self.assertEqual(row["score"], 1.0, row)

    def test_grid_zoom_off_asks_instead(self):
        model = unmarked_pick("scene_04", "white_part")
        row = self.run_scene_04(model, grid_zoom=False)
        self.assertEqual(row["answer"], "ask_user")

    def test_object_not_seen_in_the_grid_asks_instead(self):
        model = unmarked_pick("scene_04", "white_part", sees=False)
        row = self.run_scene_04(model)
        self.assertEqual(row["answer"], "ask_user")


class WidthFacts(unittest.TestCase):
    def decide_prompt(self, **switches):
        model = scripted("refuse")
        with tempfile.TemporaryDirectory() as t:
            decide_run(one_scene("scene_08", t), model, votes=1,
                       **switches)

        return model.seen["prompt"]

    def test_a_wide_flat_object_is_hinted_as_a_plate_in_decide(self):
        # Only plates and saucers are too wide (organisers' answer); the
        # width is a hint to what the object is, not a size rule.
        prompt = self.decide_prompt(widths=True)
        self.assertRegex(prompt, r"about \d+ mm wide, flat and wide like a "
                                 r"plate or saucer")

    def test_widths_off_leaves_them_out(self):
        self.assertNotIn("flat and wide", self.decide_prompt(widths=False))

    def test_decide_is_told_only_plates_and_saucers_are_too_wide(self):
        prompt = self.decide_prompt()
        self.assertIn("Only plates and saucers are too wide to pick",
                      prompt)
        self.assertNotIn("85 mm", prompt)


def one_image_only(prompt, images, schema, temperature=0):
    """A server that rejects two images per message, then plans."""
    if len(images) > 1:
        raise OSError("400: at most 1 image per prompt")

    if "verdict" not in schema.get("properties", {}):
        return json.dumps({"objects": []})

    step = {"action": "pick", "mark": 1, "object": "red_cube"}
    return json.dumps({"verdict": "plan", "plan": [step]})


class CleanImage(unittest.TestCase):
    def run_scene_06(self, model, **switches):
        with tempfile.TemporaryDirectory() as t:
            data = one_scene("scene_06", t)
            return decide_run(data, model, votes=1, **switches)

    def test_decide_sees_the_marked_and_the_clean_photo(self):
        model = scripted("refuse")
        self.run_scene_06(model, clean_image=True, scaffold=False)
        marked, clean = model.seen["images"]
        rgb = SceneGT(DATA, DATA / "ground_truth", "scene_06").scene.rgb
        self.assertTrue(np.array_equal(clean, rgb))
        self.assertFalse(np.array_equal(marked, rgb))
        self.assertIn("without the marks", model.seen["prompt"])

    def test_a_server_taking_one_image_still_gets_answers(self):
        sub = self.run_scene_06(one_image_only, clean_image=True)
        for r in sub["results"]:
            self.assertEqual(answer_type(r), ("plan", None))

    def test_clean_image_off_sends_the_marked_photo_only(self):
        model = scripted("refuse")
        self.run_scene_06(model, clean_image=False)
        self.assertEqual(len(model.seen["images"]), 1)
        self.assertNotIn("without the marks", model.seen["prompt"])


class FakeServer(BaseHTTPRequestHandler):
    """An OpenAI-style endpoint: busy for the first `busy` requests,
    slow, and counting how many requests are in flight at once."""
    busy = 0
    seen = 0
    in_flight = 0
    most = 0
    lock = threading.Lock()

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        cls = type(self)
        with cls.lock:
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

        body = json.dumps({"choices": [{"message": {"content": "ok"}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


class ApertusClient(unittest.TestCase):
    def serve(self, busy=0):
        handler = type("Server", (FakeServer,), {
            "busy": busy, "seen": 0, "in_flight": 0, "most": 0,
            "lock": threading.Lock()})
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


class NeighbourFacts(unittest.TestCase):
    def decide_prompt(self, **switches):
        model = scripted("refuse")
        with tempfile.TemporaryDirectory() as t:
            decide_run(one_scene("scene_10", t), model, votes=1,
                       **switches)

        return model.seen["prompt"]

    def test_neighbours_reach_decide(self):
        prompt = self.decide_prompt(neighbours=True)
        self.assertRegex(prompt, r"nearest: mark \d+ \(\d+(\.\d)? cm\)")

    def test_neighbours_off_leaves_them_out(self):
        self.assertNotIn("nearest:", self.decide_prompt(neighbours=False))


def catalogue(extracted, namer=lambda colour: f"{colour} cube"):
    """Return a fake that names every close-up namer(measured colour),
    extracts `extracted` from every instruction, and refuses in Decide."""
    seen = {"names": [], "decide": None, "inventory": False}
    lock = threading.Lock()

    def model(prompt, images, schema, temperature=0):
        if schema is closeups.NAME_SCHEMA:
            colour = re.search(r"colour (\w+)", prompt).group(1)
            with lock:
                seen["names"].append((prompt, images))

            return json.dumps({"name": namer(colour)})

        if schema is mentions.EXTRACT_SCHEMA:
            return json.dumps({"objects": extracted})

        if "verdict" in schema.get("properties", {}):
            seen["decide"] = prompt
            return json.dumps({"verdict": "refuse", "reason": "no"})

        seen["inventory"] = True
        return json.dumps({"objects": []})

    model.seen = seen
    return model


RED_CUBE = {"phrase": "red cube", "colour": "red", "noun": "cube"}
BLUE_CUBE = {"phrase": "blue cube", "colour": "blue", "noun": "cube"}
# scene_01 by measured colour: red cube, yellow gear, green mug, white handle
SCENE_01 = {"red": "red cube", "yellow": "yellow gear", "green": "green mug",
            "white": "white mug"}.get


class Closeups(unittest.TestCase):
    def run_scene(self, scene, model, **switches):
        switches.setdefault("boxes", False)
        with tempfile.TemporaryDirectory() as t:
            decide_run(one_scene(scene, t), model, votes=1, **switches)

        return model.seen

    def run_scene_01(self, model, **switches):
        return self.run_scene("scene_01", model, **switches)

    def test_each_mark_is_named_from_its_own_closeup(self):
        seen = self.run_scene_01(catalogue([]))
        rgb = SceneGT(DATA, DATA / "ground_truth", "scene_01").scene.rgb
        self.assertEqual(len(seen["names"]), len(opencv_marks.find_marks(rgb)))
        for prompt, images in seen["names"]:
            self.assertEqual(len(images), 1)
            self.assertLess(images[0].size, rgb.size)
            self.assertIn("Measured: colour", prompt)

    def test_decide_sees_each_marks_closeup_name(self):
        seen = self.run_scene_01(catalogue([]))
        self.assertIn("mark 1: red cube", seen["decide"])

    def test_a_mentioned_object_is_matched_to_its_mark(self):
        seen = self.run_scene_01(catalogue([RED_CUBE]))
        self.assertIn("red cube: mark 1", seen["decide"])

    def test_a_mentioned_object_without_a_mark_is_flagged_softly(self):
        # Not "missing": it may be covered, unmarked or named differently.
        seen = self.run_scene_01(catalogue([BLUE_CUBE]))
        self.assertIn("blue cube: not among the named marks; it may be",
                      seen["decide"])

    def test_a_colour_word_does_not_match_other_colours(self):
        # scene_04: mark 1 is white, marks 2 and 3 are grey.
        white = {"phrase": "white cube", "colour": "white", "noun": "cube"}
        seen = self.run_scene("scene_04", catalogue([white]))
        line = next(x for x in seen["decide"].splitlines()
                    if x.startswith("- white cube:"))
        self.assertIn("mark 1", line)
        self.assertNotIn("mark 2", line)
        self.assertNotIn("mark 3", line)

    def test_an_unsure_name_reaches_decide_as_unclear(self):
        model = catalogue([])

        def unsure(prompt, images, schema, temperature=0):
            if schema is closeups.NAME_SCHEMA:
                return json.dumps({"name": "unsure"})

            return model(prompt, images, schema, temperature)

        unsure.seen = model.seen
        seen = self.run_scene_01(unsure)
        self.assertIn("mark 1: unclear object (measured", seen["decide"])

    def test_a_mentioned_object_gets_its_nearest_other_marks(self):
        seen = self.run_scene_01(catalogue([RED_CUBE]))
        self.assertRegex(seen["decide"],
                         r"nearest to the red cube: mark \d+ [a-z ]+ "
                         r"\(\d+(\.\d)? cm\)")

    def test_several_matches_are_ordered_left_to_right(self):
        # scene_03: three red gears, all named "red cube" by the fake.
        seen = self.run_scene("scene_03", catalogue([RED_CUBE]))
        self.assertIn("red cube from left to right: mark 1, mark 3, "
                      "mark 4", seen["decide"])

    def test_a_noun_matches_only_marks_named_with_it(self):
        gear = {"phrase": "the gear", "colour": None, "noun": "gear"}
        seen = self.run_scene_01(catalogue([gear], SCENE_01))
        self.assertIn("- the gear: mark 2\n", seen["decide"])

    def test_a_misnamed_object_is_still_offered_by_colour(self):
        flower = {"phrase": "yellow flower", "colour": "yellow",
                  "noun": "flower"}
        seen = self.run_scene_01(catalogue([flower], SCENE_01))
        self.assertIn("- yellow flower: no mark is named like that; marks "
                      "measured yellow: mark 2 (yellow gear)", seen["decide"])

    def test_each_candidate_says_what_lies_on_it(self):
        # scene_06: the yellow gear (mark 2) lies on red gear mark 3;
        # red gear mark 1 lies free.
        red = {"phrase": "red gear", "colour": "red", "noun": "gear"}
        names = {"red": "red gear", "yellow": "yellow gear"}.get
        seen = self.run_scene("scene_06", catalogue([red], names))
        self.assertIn("mark 3 red gear: mark 2 lies on it", seen["decide"])
        self.assertIn("mark 1 red gear: nothing lies on it", seen["decide"])

    def test_closeups_off_uses_the_whole_photo_inventory(self):
        seen = self.run_scene_01(catalogue([]), closeups=False)
        self.assertTrue(seen["inventory"])
        self.assertEqual(seen["names"], [])

    def test_extraction_off_leaves_the_mentions_out(self):
        seen = self.run_scene_01(catalogue([RED_CUBE]), extract=False)
        self.assertNotIn("The instruction mentions", seen["decide"])


# Apertus's boxes for scene_01 (0-1000), as it answered on the hosted 70B:
# they sit a little high, as all its boxes do.
SCENE_01_BOXES = [
    {"kind": "red cube", "bounding_box": [70, 370, 225, 555]},
    {"kind": "yellow gear", "bounding_box": [295, 335, 385, 435]},
    {"kind": "mug", "bounding_box": [260, 140, 460, 305]},
]
GREY_TRAY_BOX = {"kind": "grey part", "bounding_box": [0, 30, 525, 830]}


def boxer(boxes):
    """Return a fake that answers the box call with `boxes`, extracts
    nothing and refuses in Decide."""
    seen = {"names": [], "decide": None}

    def model(prompt, images, schema, temperature=0):
        if schema is closeups.NAME_SCHEMA:
            seen["names"].append(prompt)
            return json.dumps({"name": "unsure"})

        if "verdict" in schema.get("properties", {}):
            seen["decide"] = prompt
            return json.dumps({"verdict": "refuse", "reason": "no"})

        if schema is detect.BOX_SCHEMA:
            return json.dumps({"objects": boxes})

        return json.dumps({"objects": []})

    model.seen = seen
    return model


class Boxes(unittest.TestCase):
    """Boxes naming OpenCV marks (box_marks off)."""

    def run_scene_01(self, model):
        with tempfile.TemporaryDirectory() as t:
            decide_run(one_scene("scene_01", t), model, votes=1)

        return model.seen

    def test_each_mark_is_named_by_the_box_it_lies_in(self):
        seen = self.run_scene_01(boxer(SCENE_01_BOXES))
        self.assertIn("mark 1: red cube", seen["decide"])
        self.assertIn("mark 2: yellow gear", seen["decide"])
        self.assertIn("mark 3: mug", seen["decide"])

    def test_no_closeup_is_cropped(self):
        seen = self.run_scene_01(boxer(SCENE_01_BOXES))
        self.assertEqual(seen["names"], [])

    def test_a_mark_in_no_box_is_unlisted(self):
        seen = self.run_scene_01(boxer(SCENE_01_BOXES[1:]))
        self.assertIn("mark 1: unlisted", seen["decide"])

    def test_a_tray_sized_box_names_nothing(self):
        seen = self.run_scene_01(boxer([GREY_TRAY_BOX]))
        self.assertNotIn("grey part", seen["decide"])

    def test_a_kind_off_the_list_names_nothing(self):
        seen = self.run_scene_01(boxer(
            [{"kind": "black part", "bounding_box": [70, 370, 225, 555]}]))
        self.assertNotIn("black part", seen["decide"])


def guarded(decision="go on", boxes=SCENE_01_BOXES, mark=1):
    """Return a fake whose safety pre-check answers `decision`, whose box
    call answers `boxes`, and whose Decide plans to pick `mark`."""
    seen = {"prechecks": 0, "decides": 0}
    lock = threading.Lock()

    def model(prompt, images, schema, temperature=0):
        if schema is safety.PRECHECK_SCHEMA:
            with lock:
                seen["prechecks"] += 1

            return json.dumps({"reason": "throwing is uncontrolled",
                               "decision": decision})

        if schema is detect.BOX_SCHEMA:
            return json.dumps({"objects": boxes})

        if "verdict" in schema.get("properties", {}):
            with lock:
                seen["decides"] += 1

            return json.dumps({"verdict": "plan", "plan": [
                {"action": "pick", "object": "it", "mark": mark}]})

        return json.dumps({"objects": []})

    model.seen = seen
    return model


class SafetyChecks(unittest.TestCase):
    def results(self, model, **switches):
        with tempfile.TemporaryDirectory() as t:
            sub = decide_run(one_scene("scene_01", t), model, votes=1,
                             **switches)

        return [r["actions"][0] for r in sub["results"]]

    def test_an_unsafe_instruction_is_refused_before_decide(self):
        model = guarded("refuse")
        first = self.results(model)
        self.assertTrue(all(a["action"] == "refuse" for a in first))
        self.assertIn("throwing is uncontrolled", first[0]["reason"])
        self.assertEqual(model.seen["decides"], 0)

    def test_a_safe_instruction_goes_on_to_decide(self):
        model = guarded("go on")
        first = self.results(model)
        self.assertTrue(all(a["action"] == "pick" for a in first))
        self.assertEqual(model.seen["prechecks"], 5)

    def test_safety_off_asks_no_precheck(self):
        model = guarded("refuse")
        self.results(model, safety=False)
        self.assertEqual(model.seen["prechecks"], 0)

    def test_picking_a_plate_first_is_refused(self):
        plate = [dict(SCENE_01_BOXES[0], kind="plate")]
        first = self.results(guarded(boxes=plate, mark=1))
        self.assertTrue(all(a["action"] == "refuse" for a in first))
        self.assertIn("plate", first[0]["reason"])

    def test_picking_a_cube_first_is_not_refused(self):
        first = self.results(guarded(mark=1))
        self.assertTrue(all(a["action"] == "pick" for a in first))

    def test_a_plate_only_named_as_the_destination_is_not_refused(self):
        model = guarded(mark=1)

        def to_the_plate(prompt, images, schema, temperature=0):
            reply = model(prompt, images, schema, temperature)
            if "verdict" in schema.get("properties", {}):
                reply = json.dumps({"verdict": "plan", "plan": [
                    {"action": "pick", "object": "red cube for the plate",
                     "mark": 1}]})

            return reply

        first = self.results(to_the_plate)
        self.assertTrue(all(a["action"] == "pick" for a in first))


def planning(*replies, boxes=None):
    """Return a fake that answers the box call with `boxes` (scene_01's by
    default), lets every instruction past the pre-check, and answers the
    planner with `replies` in turn (the last one repeats)."""
    boxes = SCENE_01_BOXES if boxes is None else boxes
    seen = {"prompts": [], "schemas": [], "decides": 0}
    lock = threading.Lock()

    def model(prompt, images, schema, temperature=0):
        props = schema.get("properties", {})
        if schema is detect.BOX_SCHEMA:
            return json.dumps({"objects": boxes})

        if schema is safety.PRECHECK_SCHEMA:
            return json.dumps({"reason": "fine", "decision": "go on"})

        if "moves" in props:
            with lock:
                seen["prompts"].append(prompt)
                seen["schemas"].append(schema)
                n = prompt.count("YOUR ANSWER")

            return json.dumps(replies[min(n, len(replies) - 1)])

        if "verdict" in props:
            with lock:
                seen["decides"] += 1

            return json.dumps({"verdict": "refuse", "reason": "no"})

        return json.dumps({"objects": []})

    model.seen = seen
    return model


def moves(*pairs):
    return {"thinking": "...", "decision": "plan", "message": "",
            "moves": [{"object": o, "to": t} for o, t in pairs]}


class Planner(unittest.TestCase):
    def run_scene_01(self, model, **switches):
        with tempfile.TemporaryDirectory() as t:
            sub = pipeline.run(one_scene("scene_01", t), model, votes=1,
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

    def test_opencv_marks_when_box_marks_are_off(self):
        banana = {"kind": "banana", "bounding_box": [700, 600, 800, 700]}
        model = planning(moves(("red_cube", None)),
                         boxes=SCENE_01_BOXES + [banana])
        self.run_scene_01(model, box_marks=False)
        ids = model.seen["schemas"][0]["properties"]["moves"]["items"][
            "properties"]["object"]["enum"]
        self.assertNotIn("banana", ids)

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
            pipeline.run(one_scene("scene_07", t), model, votes=1)

        self.assertIn('"covered_by": ["mug"]', model.seen["prompts"][0])

    def test_planner_off_uses_decide(self):
        model = planning(moves(("red_cube", None)))
        self.run_scene_01(model, planner=False)
        self.assertEqual(model.seen["prompts"], [])
        self.assertGreater(model.seen["decides"], 0)


class Scaffold(unittest.TestCase):
    def decide_input(self, **switches):
        model = scripted("refuse")
        with tempfile.TemporaryDirectory() as t:
            decide_run(one_scene("scene_06", t), model, votes=1,
                       clean_image=False, **switches)

        return model.seen["prompt"], model.seen["images"][0]

    def test_decide_sees_numbered_rows_and_the_scanning_prompt(self):
        prompt, with_rows = self.decide_input(scaffold=True)
        _, without = self.decide_input(scaffold=False)
        self.assertIn("Scan the image sequentially", prompt)

        # The image differs only along the three lines and the numbers.
        changed = np.any(with_rows != without, axis=2).mean()
        self.assertGreater(changed, 0)
        self.assertLess(changed, 0.05)

    def test_scaffold_off_leaves_image_and_prompt_alone(self):
        prompt, _ = self.decide_input(scaffold=False)
        self.assertNotIn("Scan the image sequentially", prompt)


class PerceptionSeam(unittest.TestCase):
    def test_marks_cover_the_plan_targets(self):
        # The ground truth labels only the objects instructions use, so
        # marks on a plate or a hand can't be checked; duplicates (a mug's
        # handle) still land on the right object. coverage.py lists both.
        from coverage import coverage
        r = coverage(DATA)
        self.assertGreaterEqual(r["coverage"], 0.9, r["misses"])

    def named_marks(self, scene):
        """Marks with neighbours, and the ground-truth name under each."""
        sg = SceneGT(DATA, DATA / "ground_truth", scene)
        marks = opencv_marks.find_marks(sg.scene.rgb)
        measure.add_neighbours(sg.scene, marks)
        return marks, {m["id"]: sg.object_at(m["pixel"]) for m in marks}

    def test_nearest_red_gear_to_the_mandarin(self):
        # scene_10_i3: "pick up the red gear closest to the mandarin".
        # The two gears are a near tie (mask centres 397 vs 405 px), so
        # the facts must only not contradict the answer: red_gear_1 is
        # listed at the nearest red-gear distance.
        marks, name = self.named_marks("scene_10")
        mandarin = next(m for m in marks if name[m["id"]] == "mandarin")
        gears = {name[i]: cm for i, cm in mandarin["nearest"]
                 if (name[i] or "").startswith("red_gear")}
        self.assertEqual(gears.get("red_gear_1"), min(gears.values()))

    def test_nearest_object_to_the_mug(self):
        # scene_01_i3: "pick up the object closest to the mug"
        marks, name = self.named_marks("scene_01")
        mug = next(m for m in marks if name[m["id"]] == "mug")
        others = [i for i, _ in mug["nearest"] if name[i] != "mug"]
        self.assertEqual(name[others[0]], "yellow_gear")

    def test_a_mug_gets_one_mark_not_one_more_for_its_handle(self):
        for scene in ("scene_01", "scene_04", "scene_05", "scene_07"):
            _, name = self.named_marks(scene)
            self.assertEqual(list(name.values()).count("mug"), 1, scene)

    def test_the_yellow_gear_lies_on_a_red_gear(self):
        # scene_06: the yellow gear lies on the unlabelled red gear; the
        # labelled red_gear_1 lies free with nothing on it.
        marks, name = self.named_marks("scene_06")
        by_name = {name[m["id"]]: m for m in marks if name[m["id"]]}
        lower = by_name["yellow_gear"]["on"]
        self.assertEqual(len(lower), 1)
        self.assertIsNone(name[lower[0]])
        self.assertEqual(by_name["red_gear_1"]["on"], [])
        self.assertFalse(any(by_name["red_gear_1"]["id"] in m["on"]
                             for m in marks))

    def test_only_the_plates_and_saucers_measure_too_wide(self):
        # Every labelled dev object is picked by some instruction, so
        # none may be called too wide.
        for scene in sorted({g["scene"] for g in GT}):
            sg = SceneGT(DATA, DATA / "ground_truth", scene)
            for m in opencv_marks.find_marks(sg.scene.rgb):
                name = sg.object_at(m["pixel"])
                if name:
                    width = measure.width_mm(sg.scene, m)
                    self.assertLess(width, measure.TOO_WIDE_MM,
                                    f"{scene} {name}")

        # Scene 08's widest object is the plate; it is not labelled.
        sg = SceneGT(DATA, DATA / "ground_truth", "scene_08")
        widths = [measure.width_mm(sg.scene, m)
                  for m in opencv_marks.find_marks(sg.scene.rgb)]
        self.assertGreater(max(widths), measure.TOO_WIDE_MM)


if __name__ == "__main__":
    unittest.main()
