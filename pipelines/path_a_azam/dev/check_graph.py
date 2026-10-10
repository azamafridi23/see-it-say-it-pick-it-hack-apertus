#!/usr/bin/env python3
"""Development check, not part of the pipeline: compare the scene graphs with the ground-truth masks.

    python pipelines/path_a_azam/dev/check_graph.py dataset

For every object that has a ground-truth mask: is there a graph object whose pixel lies on it, and
does it carry the same name? The masks cover only the objects that are a target of some instruction.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the pipeline, one folder up

from run_sam3 import detect
from scene_graph import build, scene_dirs

SAME = {"small_part": "white_part"}  # ground-truth name -> the name the graph uses


def kind(name: str) -> str:
    base = re.sub(r"_\d+$", "", name)
    return SAME.get(base, base)


def main():
    data = Path(sys.argv[1])
    found = named = exact = total = 0
    for d in scene_dirs(data):
        graph, _ = build(d, detect(d))
        gt = cv2.imread(str(data / "ground_truth" / "masks" / f"{d.name}.png"), cv2.IMREAD_UNCHANGED)
        names = {int(k): v for k, v in json.loads((data / "ground_truth" / "masks" / f"{d.name}.json").read_text())["objects"].items()}
        at = {}
        for o in graph["objects"]:
            if "pixel" in o:
                at.setdefault(names.get(int(gt[o["pixel"][1], o["pixel"][0]])), []).append(o["id"])
        notes = []
        for name in names.values():
            ids = at.get(name, [])
            total += 1
            found += bool(ids)
            named += any(kind(i) == kind(name) for i in ids)
            exact += name in ids or (kind(name) != re.sub(r"_\d+$", "", name) and any(kind(i) == kind(name) for i in ids))
            if not ids:
                notes.append(f"{name}: MISSED")
            elif not any(kind(i) == kind(name) for i in ids):
                notes.append(f"{name}: found as {'/'.join(ids)}")
            elif len(ids) > 1:
                notes.append(f"{name}: found {len(ids)} times ({'/'.join(ids)})")
            elif name not in ids and kind(name) == re.sub(r"_\d+$", "", name):
                notes.append(f"{name}: numbered {ids[0]}")
        rel = [f"{o['id']} on {'+'.join(o['on'])}" for o in graph["objects"] if o.get("on")] \
            + [f"{o['id']} on? {'+'.join(o['maybe_on'])}" for o in graph["objects"] if o.get("maybe_on")]
        hand = [o["id"] for o in graph["objects"] if o.get("touching_hand")]
        print(f"{d.name}: {len(graph['objects'])} objects | " + ("; ".join(notes) or "all targets found and named"))
        print(f"    ids: {', '.join(o['id'] for o in graph['objects'])}")
        if rel or hand:
            print("    " + "; ".join(rel) + (f" | touching hand: {', '.join(hand)}" if hand else ""))
    print(f"\ntargets with a graph pixel on them: {found}/{total}   with the right name: {named}/{total}   same id as the ground truth: {exact}/{total}")


if __name__ == "__main__":
    main()
