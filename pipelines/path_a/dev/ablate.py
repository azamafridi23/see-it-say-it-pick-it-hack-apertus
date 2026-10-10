#!/usr/bin/env python3
"""Development test, not part of the pipeline: how well are objects found when the fixed list does not know them?

    python pipelines/path_a/dev/ablate.py dataset

Hides some prompts from the fixed list, rebuilds every scene graph, and counts how many of the hidden objects
are still found (same place as in the full graph), and how many objects appear that the full graph does not have.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the pipeline, one folder up

from apertus_list import names_for
from run_sam3 import PROMPTS, detect
from scene_graph import build, scene_dirs

HIDE = {"white part", "metal bracket", "banana", "mug", "mandarin", "orange fruit"}
HIDDEN_LABELS = {"white part", "metal bracket", "banana", "mug", "mandarin"}
VARIANTS = {  # name: (use the "object" catch-all, use Apertus's names, use the leftover check)
    "nothing extra": (False, False, False),
    '"object" prompt only': (True, False, False),
    "Apertus names only": (False, True, False),
    "leftover check only": (False, False, True),
    "all three": (True, True, True),
}


def iou(a, b) -> float:
    i = int((a & b).sum())
    return i / (int(a.sum()) + int(b.sum()) - i)


def main():
    data = Path(sys.argv[1])
    totals = {v: [0, 0, {}] for v in VARIANTS}
    n_hidden = 0
    for d in scene_dirs(data):
        _, full = build(d, detect(d), leftover=False)
        hidden = [o for o in full if o["label"] in HIDDEN_LABELS]
        n_hidden += len(hidden)
        lines = []
        for name, (catch_all, apertus, leftover) in VARIANTS.items():
            prompts = [p for p in PROMPTS if p not in HIDE and (catch_all or p != "object")]
            _, objs = build(d, detect(d, prompts), extra=detect(d, names_for(d)) if apertus else [], leftover=leftover)
            got = {h["id"]: next((o for o in objs if iou(o["mask"], h["mask"]) >= 0.5), None) for h in hidden}
            spurious = [o["id"] for o in objs if not any(iou(o["mask"], f["mask"]) >= 0.5 for f in full)]
            totals[name][0] += sum(v is not None for v in got.values())
            totals[name][1] += len(spurious)
            for v in got.values():
                if v is not None:
                    totals[name][2][v["found_by"]] = totals[name][2].get(v["found_by"], 0) + 1
            if name == "all three":
                lines.append("  " + ", ".join(f"{h} -> {(v['id'] + ' [' + v['found_by'] + ']') if v else 'MISSED'}" for h, v in got.items())
                             + (f" | extra: {spurious}" if spurious else ""))
        if hidden:
            print(d.name + "\n" + "\n".join(lines))
    print(f"\n{n_hidden} hidden objects")
    for name, (found, spurious, by) in totals.items():
        print(f"  {name:22s} found {found:2d}/{n_hidden}   objects not in the full graph: {spurious}   {by or ''}")


if __name__ == "__main__":
    main()
