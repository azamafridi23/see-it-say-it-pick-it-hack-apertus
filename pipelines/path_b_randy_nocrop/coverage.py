#!/usr/bin/env python3
"""OpenCV mark coverage (`--opencv-marks`), no model: does every plan
target have a mark on it?

    python pipelines/path_b_randy_nocrop/coverage.py --data dataset

Coverage is the ceiling on grounding: Apertus can only name a mark that
exists. Also lists duplicate marks (one object, several marks) and marks
on unlabelled things. The ground truth only labels objects that
instructions use, so plates, hands and spare gears are unlabelled.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "starter"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import SceneGT  # noqa: E402
from opencv_marks import find_marks  # noqa: E402


def coverage(data):
    """Return coverage of plan targets, plus misses, duplicates, unlabelled."""
    gt_dir = data / "ground_truth"
    gt_file = gt_dir / "ground_truth.json"
    gts = json.loads(gt_file.read_text())["instructions"]
    seen, misses, dupes, unlabelled = {}, [], [], []
    for scene in sorted({g["scene"] for g in gts}):
        sg = SceneGT(data, gt_dir, scene)
        on = [sg.object_at(m["pixel"]) for m in find_marks(sg.scene.rgb)]
        seen[scene] = set(on)
        counts = Counter(o for o in on if o)
        dupes += [f"{scene}: {o} x{n}" for o, n in counts.items() if n > 1]
        if None in on:
            unlabelled.append(f"{scene}: {on.count(None)} on objects "
                              "the ground truth does not label")

    plans = [g for g in gts if g["expected"] == "plan"]
    for g in plans:
        if not set(g["targets"]) & seen[g["scene"]]:
            targets = " / ".join(g["targets"])
            misses.append(f"{g['instruction_id']}: no mark on {targets}")

    return {
        "coverage": 1 - len(misses) / len(plans),
        "misses": misses,
        "duplicates": dupes,
        "unlabelled": unlabelled,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    r = coverage(ap.parse_args().data)
    print(f"coverage {100 * r['coverage']:.0f}% of plan targets have a mark")
    for k in ("misses", "duplicates", "unlabelled"):
        print(f"\n{k} ({len(r[k])}):" + "".join(f"\n  {x}" for x in r[k]))


if __name__ == "__main__":
    main()
