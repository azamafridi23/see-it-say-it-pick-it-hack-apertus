"""Put the organisers' ``starter`` and ``baseline`` folders on sys.path.

Importing this module first lets the other modules import ``scene_io``
and ``baseline`` from any working directory.
"""
import sys
from pathlib import Path

# The project root (track_2a/): data/, results/ and .env; the code is in
# src/.
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC / "starter"))
sys.path.insert(0, str(SRC / "baseline"))
