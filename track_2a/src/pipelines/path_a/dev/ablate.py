#!/usr/bin/env python3
"""Measure how well unknown objects are found (development, not pipeline).

How well are objects found when the fixed prompt list does not know them?

    python pipelines/path_a/dev/ablate.py dataset

Hides some prompts from the fixed list, rebuilds every scene graph, and
counts how many of the hidden objects are still found (same place as in
the full graph), and how many objects appear that the full graph does not
have.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The pipeline, one folder up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apertus_list import names_for  # noqa: E402
from run_sam3 import detect  # noqa: E402
from scene_graph import build, scene_dirs  # noqa: E402
from workcell import CELL  # noqa: E402

HIDE = {"white part", "metal bracket", "banana", "mug", "mandarin",
        "orange fruit"}
HIDDEN_LABELS = {"white part", "metal bracket", "banana", "mug", "mandarin"}
# name: (use the "object" catch-all, use Apertus's names, use the leftover
# check)
VARIANTS = {
    "nothing extra": (False, False, False),
    '"object" prompt only': (True, False, False),
    "Apertus names only": (False, True, False),
    "leftover check only": (False, False, True),
    "all three": (True, True, True),
}
SAME_OBJECT_IOU = 0.5


def iou(one, two) -> float:
    """Return the intersection over union of two bool masks."""
    overlap = int((one & two).sum())
    return overlap / (int(one.sum()) + int(two.sum()) - overlap)


def _match(obj, candidates):
    """Find the candidate in the same place as ``obj``, or None."""
    for candidate in candidates:
        if iou(candidate["mask"], obj["mask"]) >= SAME_OBJECT_IOU:
            return candidate

    return None


def _run_variant(scene, variant, full, hidden):
    """Rebuild one scene's graph without the hidden prompts.

    Returns:
        For each hidden object, the object found in its place (or None),
        and the ids of objects the full graph does not have.
    """
    catch_all, apertus, leftover = VARIANTS[variant]
    prompts = [prompt for prompt in CELL.sam3_prompts
               if prompt not in HIDE and (catch_all or prompt != "object")]
    extra = detect(scene, names_for(scene)) if apertus else []
    _, objs = build(scene, detect(scene, prompts), extra=extra,
                    leftover=leftover)
    found = {obj["id"]: _match(obj, objs) for obj in hidden}
    spurious = [obj["id"] for obj in objs if _match(obj, full) is None]
    return found, spurious


def _found_line(found, spurious):
    """Describe what one variant found for one scene."""
    parts = []
    for hidden_id, obj in found.items():
        if obj is None:
            parts.append(f"{hidden_id} -> MISSED")
        else:
            parts.append(f"{hidden_id} -> {obj['id']} [{obj['found_by']}]")

    line = "  " + ", ".join(parts)
    if spurious:
        line += f" | extra: {spurious}"

    return line


def main():
    """Run every variant on every scene and print the counts."""
    data = Path(sys.argv[1])
    totals = {variant: [0, 0, {}] for variant in VARIANTS}
    hidden_count = 0
    for scene in scene_dirs(data):
        _, full = build(scene, detect(scene), leftover=False)
        hidden = [obj for obj in full if obj["label"] in HIDDEN_LABELS]
        hidden_count += len(hidden)
        lines = []
        for variant in VARIANTS:
            found, spurious = _run_variant(scene, variant, full, hidden)
            found_count, spurious_count, by_source = totals[variant]
            totals[variant][0] = found_count + sum(
                obj is not None for obj in found.values())
            totals[variant][1] = spurious_count + len(spurious)
            for obj in found.values():
                if obj is not None:
                    source = obj["found_by"]
                    by_source[source] = by_source.get(source, 0) + 1

            if variant == "all three":
                lines.append(_found_line(found, spurious))

        if hidden:
            print(scene.name + "\n" + "\n".join(lines))

    print(f"\n{hidden_count} hidden objects")
    for variant, (found, spurious, by_source) in totals.items():
        print(f"  {variant:22s} found {found:2d}/{hidden_count}   "
              f"objects not in the full graph: {spurious}   "
              f"{by_source or ''}")


if __name__ == "__main__":
    main()
