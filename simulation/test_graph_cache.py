"""Offline regression for scene 1's tray-only cached detection."""
import unittest
from simulation.worker import usable_graph


class GraphCacheTests(unittest.TestCase):
    def test_tray_only_detection_is_not_reused(self):
        self.assertFalse(usable_graph({"objects": [
            {"id": "grey_tray", "pickable": False},
            {"id": "black_tray", "pickable": False}]}))
        self.assertFalse(usable_graph(None))
        self.assertFalse(usable_graph({"objects": []}))

    def test_detected_objects_require_geometry(self):
        self.assertFalse(usable_graph({"objects": [{"id": "red_cube"}]}))
        self.assertTrue(usable_graph({"objects": [{"id": "red_cube", "pixel": [0, 0]}]}))
        self.assertTrue(usable_graph({"objects": [{"id": "hand", "pixel": [10, 20], "pickable": False}]}))
