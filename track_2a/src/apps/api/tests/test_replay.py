"""Offline integration checks. All socket connections are forbidden."""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ["SIMULATOR_DISABLE_LIVE"] = "1"

from fastapi.testclient import TestClient
from apps.api.main import app, jobs
from simulation.storage import RESULT_FILES, answers, instructions, replay, scene_folder


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.no_network = patch("socket.socket.connect", side_effect=AssertionError("Network forbidden in offline checks"))
        self.no_network.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.no_network.stop()

    def test_catalog_and_images(self):
        response = self.client.get("/api/scenes")
        self.assertEqual(response.status_code, 200)
        scenes = response.json()["scenes"]
        self.assertEqual(len(scenes), 10)
        self.assertEqual(len(scenes[0]["instructions"]), 5)
        self.assertEqual(scenes[0]["instruction_file"], "instructions.txt")
        image = self.client.get("/api/scenes/scene_01/image")
        self.assertEqual(image.headers["content-type"], "image/png")
        self.assertTrue(image.content.startswith(b"\x89PNG"))
        self.assertEqual(self.client.get("/api/scenes/not-a-scene/image").status_code, 404)

    def test_shipped_results_can_be_replayed(self):
        # The repository ships these two; a rename must not hide them.
        for pipeline in ("path_a_graph", "path_b"):
            self.assertTrue(RESULT_FILES[pipeline].is_file(), RESULT_FILES[pipeline])
            self.assertTrue(answers(pipeline), pipeline)

    def test_every_saved_replay_retains_original_answer(self):
        for pipeline, path in RESULT_FILES.items():
            if not path.exists():
                continue
            for answer in answers(pipeline):
                scene = answer["instruction_id"].rsplit("_i", 1)[0]
                result = replay(scene, pipeline, answer["instruction_id"])
                self.assertEqual(result["answer"], answer)
                known = {node["id"] for node in result["objects"]}
                for action in answer["actions"]:
                    if "object" in action:
                        self.assertIn(action["object"], known)
                self.assertEqual(result["geometry_source"], "saved_pixels")

    def test_api_replay_and_missing_result(self):
        response = self.client.post("/api/plans", json={"scene_id": "scene_01", "instruction_id": "scene_01_i1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["answer"]["actions"][0]["object"], "red_cube")
        missing = self.client.post("/api/plans", json={"scene_id": "scene_01", "instruction_id": "scene_02_i1"})
        self.assertEqual(missing.status_code, 400)

    def test_live_cannot_start_in_offline_mode(self):
        with patch("apps.api.main.executor.submit", side_effect=AssertionError("Worker must not start")):
            response = self.client.post("/api/plans", json={"scene_id": "scene_01", "mode": "live", "instruction": "pick up the red cube"})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.client.get("/api/health").json()["live_enabled"])

    def test_scene_paths_are_allowlisted(self):
        for value in ("../ground_truth", "../../.env", "ground_truth", "/tmp"):
            with self.assertRaises(ValueError):
                scene_folder(value)

    def test_instruction_files_support_both_names_and_formats(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(instructions(folder), [])
            (folder / "instruction.txt").write_text("# comment\n\nsimple\tpick up the red cube\nput it in the black tray\n")
            self.assertEqual([item["text"] for item in instructions(folder)],
                             ["pick up the red cube", "put it in the black tray"])
            (folder / "instructions.txt").write_text("preferred plural file\n")
            self.assertEqual(instructions(folder)[0]["text"], "preferred plural file")

    def test_dataset_live_uses_file_text_without_starting_a_worker(self):
        # Only validate the queued payload; the executor is stubbed, so no model can run.
        with patch.dict(os.environ, {"SIMULATOR_DISABLE_LIVE": "0"}), patch("apps.api.main.executor.submit") as submit:
            response = self.client.post("/api/plans", json={"scene_id": "scene_01", "mode": "live",
                "instruction_source": "dataset", "instruction_id": "scene_01_i1", "instruction": "ignored browser text"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(submit.call_args.args[2]["instruction"], instructions(scene_folder("scene_01"))[0]["text"])
        jobs.pop(response.json()["job_id"])

    def test_dataset_live_rejects_instruction_from_another_scene(self):
        with patch.dict(os.environ, {"SIMULATOR_DISABLE_LIVE": "0"}), patch("apps.api.main.executor.submit") as submit:
            response = self.client.post("/api/plans", json={"scene_id": "scene_01", "mode": "live",
                "instruction_source": "dataset", "instruction_id": "scene_02_i1"})
        self.assertEqual(response.status_code, 400)
        submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
