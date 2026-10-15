#!/usr/bin/env python3
"""Check the setup before a long run: the key, the endpoint, the scenes.

    uv run python scripts/check_setup.py [folder of scenes]

Prints one line per check and exits with 1 if any check fails. The key is
never printed.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines" / "path_a"))


def main() -> int:
    """Run every check and report them."""
    load_dotenv(ROOT.parent / ".env")
    failed = False

    def report(ok: bool, text: str):
        nonlocal failed
        failed |= not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {text}")

    from workcell import CELL, api_key
    base_url, model, key = CELL.base_url, CELL.model, api_key()
    report(bool(key), "LLM_API_KEY is set"
           if key else "LLM_API_KEY is missing: set it or add it to .env")

    if key:
        from openai import OpenAI
        try:
            served = [m.id for m in OpenAI(api_key=key, base_url=base_url,
                                           timeout=30).models.list()]
            report(model in served, f"{base_url} serves {model}"
                   if model in served else
                   f"{base_url} does not serve {model} (it serves {served})")
        except Exception as error:  # any failure means the run would fail
            report(False, f"cannot reach {base_url}: {type(error).__name__}")

    folder = (Path(sys.argv[1]) if len(sys.argv) > 1
              else ROOT.parent / "data" / "heldout")
    scenes = sorted(p.parent for p in folder.glob("*/rgb.png"))
    without = [s.name for s in scenes if not (s / "instructions.txt").exists()]
    calibrated = all((s / "calibration.json").exists()
                     or (folder / "calibration.json").exists()
                     for s in scenes)
    report(bool(scenes) and not without and calibrated,
           f"{folder}: {len(scenes)} scenes" if scenes and not without
           and calibrated else
           f"{folder}: no scenes (folders with rgb.png)" if not scenes else
           f"{folder}: no instructions.txt in {', '.join(without)}"
           if without else f"{folder}: no calibration.json")

    from huggingface_hub import try_to_load_from_cache
    cached = isinstance(try_to_load_from_cache(
        CELL.sam3_model, "model.safetensors",
        revision=CELL.sam3_revision), str)
    print(f"  info SAM 3 weights {'are cached' if cached else
          'download on the first Path A run (3.4 GB)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
