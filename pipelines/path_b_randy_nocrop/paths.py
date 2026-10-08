"""The repository root on sys.path, so the organisers' `starter` and
`baseline` modules import from any working directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "baseline"))
