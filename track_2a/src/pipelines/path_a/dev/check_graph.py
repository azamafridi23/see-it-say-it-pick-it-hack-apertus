#!/usr/bin/env python3
"""Compare the scene graphs with the ground-truth masks (development).

    python pipelines/path_a/dev/check_graph.py dataset

For every object that has a ground-truth mask: is there a graph object
whose pixel lies on it, and does it carry the same name? The masks cover
only the objects that are a target of some instruction. Not part of the
pipeline: it reads the ground truth.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2

# The pipeline, one folder up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_sam3 import detect  # noqa: E402
from scene_graph import build, scene_dirs  # noqa: E402

# ground-truth name -> the name the graph uses
SAME = {"small_part": "white_part"}


def without_number(name: str) -> str:
    """Strip the trailing number from an id: red_gear_2 -> red_gear."""
    return re.sub(r"_\d+$", "", name)


def kind(name: str) -> str:
    """Strip the number from an id and use the graph's name for the kind."""
    base = without_number(name)
    return SAME.get(base, base)


def _load_truth(data: Path, scene_name: str):
    """Load a scene's ground-truth mask image and its id -> name table."""
    folder = data / "ground_truth" / "masks"
    mask = cv2.imread(str(folder / f"{scene_name}.png"),
                      cv2.IMREAD_UNCHANGED)
    table = json.loads((folder / f"{scene_name}.json").read_text())
    names = {int(key): name for key, name in table["objects"].items()}
    return mask, names


def _note(name, ids):
    """Describe how a ground-truth object was found, or None if exactly."""
    same_kind = any(kind(found) == kind(name) for found in ids)
    if not ids:
        return f"{name}: MISSED"

    if not same_kind:
        return f"{name}: found as {'/'.join(ids)}"

    if len(ids) > 1:
        return f"{name}: found {len(ids)} times ({'/'.join(ids)})"

    if name not in ids and kind(name) == without_number(name):
        return f"{name}: numbered {ids[0]}"

    return None


def main():
    """Check every scene and print the counts."""
    data = Path(sys.argv[1])
    found = named = exact = total = 0
    for scene in scene_dirs(data):
        graph, _ = build(scene, detect(scene))
        truth, names = _load_truth(data, scene.name)

        # Which ground-truth object each graph pixel lands on.
        ids_on = {}
        for entry in graph["objects"]:
            if "pixel" not in entry:
                continue

            u, v = entry["pixel"]
            name = names.get(int(truth[v, u]))
            ids_on.setdefault(name, []).append(entry["id"])

        notes = []
        for name in names.values():
            ids = ids_on.get(name, [])
            same_kind = any(kind(found_id) == kind(name) for found_id in ids)
            renamed = kind(name) != without_number(name)
            total += 1
            found += bool(ids)
            named += same_kind
            exact += name in ids or (renamed and same_kind)
            note = _note(name, ids)
            if note:
                notes.append(note)

        relations = []
        for entry in graph["objects"]:
            if entry.get("on"):
                relations.append(f"{entry['id']} on {'+'.join(entry['on'])}")

        for entry in graph["objects"]:
            if entry.get("maybe_on"):
                relations.append(
                    f"{entry['id']} on? {'+'.join(entry['maybe_on'])}")

        hand = [entry["id"] for entry in graph["objects"]
                if entry.get("touching_hand")]
        ids = ", ".join(entry["id"] for entry in graph["objects"])
        summary = "; ".join(notes) or "all targets found and named"
        print(f"{scene.name}: {len(graph['objects'])} objects | {summary}")
        print(f"    ids: {ids}")
        if relations or hand:
            touching = f" | touching hand: {', '.join(hand)}" if hand else ""
            print("    " + "; ".join(relations) + touching)

    print(f"\ntargets with a graph pixel on them: {found}/{total}   "
          f"with the right name: {named}/{total}   "
          f"same id as the ground truth: {exact}/{total}")


if __name__ == "__main__":
    main()
