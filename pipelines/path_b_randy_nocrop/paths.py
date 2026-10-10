"""Put the organisers' ``starter`` and ``baseline`` folders on sys.path.

Importing this module first lets the other modules import ``scene_io``
and ``baseline`` from any working directory.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "baseline"))
