#!/usr/bin/env python3
"""Path B: Apertus is the only learned model and the main detector;
OpenCV only measures.

See docs/adr/0001 and ARCHITECTURE.md.

    vllm serve swiss-ai/Apertus-v1.5-8B \\
        --chat-template-content-format string
    python pipelines/path_b_randy_nocrop/pipeline.py \\
        --data dataset --out results.json
    python starter/evaluate.py results.json --data dataset

Or a hosted Apertus; the key is read from APERTUS_API_KEY:

    APERTUS_API_KEY=... python pipelines/path_b_randy_nocrop/pipeline.py \\
        --base-url https://hackapertus.livemap.sh/v1 \\
        --model apertus-v1.5-70b

Stages, one module each:
    detect.py     Apertus boxes every object on the whole photo; each box
                  is one object, its corrected centre the target pixel
    measure.py    OpenCV measures: sizes, gaps, trays, what lies on top
    safety.py     per instruction, Apertus refuses an unsafe sentence
    planner.py    Path A's planner on a scene graph of the objects

The older designs stay behind switches, for comparison:
    --opencv-marks   opencv_marks.py: colour blobs as the objects
    --no-boxes       closeups.py: blobs named from close-up crops
    --no-planner     decide.py, mentions.py, zoom.py: Decide with votes

    --mock     no model: refuses everything (checks the loop and the format)
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import paths  # noqa: F401
import closeups
import decide
import detect
import measure
import mentions
import opencv_marks
import planner as path_a_planner
import zoom
from apertus import apertus, ask, mock
from baseline import parse
from safety import precheck, refusal
from scene_io import load_scene


def find_objects(model, sc, boxes, box_marks, closeups_on, widths,
                 neighbours):
    """Return the scene's marks: found, named and measured.

    With `boxes` and `box_marks`, Apertus's boxes are the objects;
    otherwise OpenCV's colour blobs, named by the boxes (`boxes`), by
    close-ups (`closeups_on`) or not at all (Decide's Inventory names
    them).
    """
    from_boxes = boxes and box_marks
    if from_boxes:
        marks = detect.objects(model, sc)
    else:
        marks = opencv_marks.find_marks(sc.rgb)

    for m in marks:
        m["colour"] = measure.mark_colour(sc.rgb, m)

    if widths:
        for m in marks:
            m["width_mm"] = round(measure.width_mm(sc, m))

    if neighbours:
        measure.add_neighbours(sc, marks)

    if from_boxes:
        measure.stacked_on(sc.rgb, marks)
    elif boxes:
        detect.box_names(model, sc, marks)
    elif closeups_on:
        closeups.name_marks(model, sc, marks)

    return marks


def decider(model, sc, marks, reasoning, votes, grid_zoom, clean_image,
            named, extract, scaffold, safety_on):
    """Return solve(iid, text) -> (answer, log record) for Decide mode."""
    schema = (decide.DECIDE_SCHEMA_REASONING if reasoning
              else decide.DECIDE_SCHEMA)
    template = decide.DECIDE_PROMPT + (decide.DECIDE_CHECKS if reasoning
                                       else "")
    if scaffold:
        template = decide.SCAFFOLD_NOTE + template

    lines = decide.mark_lines(marks, sc.size)
    images = [decide.draw_marks(sc.rgb, marks)]
    if clean_image:
        images.append(sc.rgb)

    if scaffold:
        images = [decide.draw_rows(im) for im in images]

    locate = partial(zoom.zoom, model, sc.rgb) if grid_zoom else None

    if named:
        inv_reply = inventory = decide.named_inventory(marks)
    else:
        inv_reply = ask(model, decide.INVENTORY_PROMPT.format(marks=lines),
                        images, decide.INVENTORY_SCHEMA)
        objects = (parse(inv_reply) or {}).get("objects", [])
        inventory = json.dumps(objects)

    def solve(iid, text):
        mentioned = mentions.mentions(model, text, marks) if (
            extract and named) else ""
        prompt = template.format(marks=lines, inventory=inventory,
                                 text=text, mentions=mentioned)
        reply, raw = decide.decide(model, prompt, images, schema, votes)
        logged = [{k: v for k, v in m.items() if k != "outline"}
                  for m in marks]
        record = {"instruction_id": iid, "marks": logged,
                  "inventory": inv_reply, "mentions": mentioned,
                  "decide": raw}
        return decide.answer(iid, reply, marks, locate, safety_on), record

    return solve


def run(data, model, log=None, reasoning=True, votes=5, grid_zoom=True,
        widths=True, clean_image=True, neighbours=True, closeups=True,
        extract=True, scaffold=True, boxes=True, safety=True,
        planner=True, box_marks=True):
    """Answer every instruction of every scene in `data`.

    `model(prompt, images, schema, temperature)` returns the reply text.
    Returns the submission dict.

    The current design, all on by default:
    `box_marks` makes Apertus's boxes the objects (needs `boxes`);
    `safety` refuses before planning when the sentence alone must be
    refused (and, in Decide mode, a first pick of a plate, saucer or
    tray); `planner` answers with Path A's planner (a scene graph of the
    objects, ids, moves, a checker and retries); `widths` and
    `neighbours` measure sizes and gaps.

    The older designs, off by default: `box_marks` off uses OpenCV's
    colour blobs, named by the boxes; `boxes` off names them from
    close-ups instead (`closeups`), or not at all. `planner` off uses
    Decide: `reasoning` adds the checks, examples and a reasoning field;
    `votes` Decide samples; `grid_zoom` locates an unmarked first pick;
    `clean_image` sends the photo without marks too; `extract` lists the
    objects the instruction mentions; `scaffold` draws numbered rows.

    A scene's instructions are answered in parallel.
    """
    results = []
    scenes = sorted(p for p in Path(data).iterdir()
                    if (p / "rgb.png").exists())

    for d in scenes:
        sc = load_scene(d)
        marks = find_objects(model, sc, boxes, box_marks, closeups, widths,
                             neighbours)

        if planner:
            graph = path_a_planner.scene_graph(marks,
                                               measure.black_mat_x(sc.rgb))

            def plan(iid, text):
                return path_a_planner.plan(
                    lambda prompt, schema: ask(model, prompt, [], schema),
                    iid, text, graph)
        else:
            plan = decider(model, sc, marks, reasoning, votes, grid_zoom,
                           clean_image, boxes or closeups, extract,
                           scaffold, safety)

        def solve(instruction):
            # The category is never used: the held-out set has no labels.
            iid, _category, text = instruction
            reason = safety and precheck(model, text)
            if reason:
                return (refusal(iid, reason),
                        {"instruction_id": iid, "precheck": reason})

            return plan(iid, text)

        # ponytail: scenes still go one after another (~25 calls in flight);
        # run scenes in parallel too if the endpoint takes more.
        n = max(1, len(sc.instructions))
        with ThreadPoolExecutor(n) as pool:
            solved = list(pool.map(solve, sc.instructions))

        for result, record in solved:
            results.append(result)
            if log:
                log.write(json.dumps(record) + "\n")

    return {"team": "path_b_randy_nocrop", "results": results}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset"),
                    help="folder of scenes")
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="resize the photo so its long side is at most this")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--parallel", type=int, default=5,
                    help="model calls in flight at once (endpoint limit)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--team", default="path_b_randy_nocrop")

    current = ap.add_argument_group("current design (all on)")
    current.add_argument("--no-safety", action="store_true",
                         help="no pre-check, no plate/saucer rule")
    current.add_argument("--no-widths", action="store_true",
                         help="leave the measured sizes out")
    current.add_argument("--no-neighbours", action="store_true",
                         help="leave the measured gaps out")

    older = ap.add_argument_group("older designs, for comparison")
    older.add_argument("--opencv-marks", action="store_true",
                       help="OpenCV colour blobs as the objects, named by "
                            "boxes")
    older.add_argument("--no-boxes", action="store_true",
                       help="name the blobs from close-ups, not boxes")
    older.add_argument("--no-closeups", action="store_true",
                       help="with --no-boxes: one Inventory call instead")
    older.add_argument("--no-planner", action="store_true",
                       help="Decide with votes instead of Path A's planner")
    older.add_argument("--votes", type=int, default=5,
                       help="Decide samples per instruction (1 = no voting)")
    older.add_argument("--no-reasoning", action="store_true",
                       help="no checks, examples or reasoning field")
    older.add_argument("--no-grid-zoom", action="store_true",
                       help="ask instead of zooming to an unmarked pick")
    older.add_argument("--no-clean-image", action="store_true",
                       help="send only the marked photo (one image per "
                            "call)")
    older.add_argument("--no-extract", action="store_true",
                       help="do not list the objects the instruction "
                            "mentions")
    older.add_argument("--no-scaffold", action="store_true",
                       help="no numbered rows on Decide's images")
    args = ap.parse_args()

    if args.mock:
        model = mock
    else:
        # From the environment, never the command line: keys on the
        # command line end up in the shell history.
        api_key = os.environ.get("APERTUS_API_KEY", "none")
        model = apertus(args.base_url, args.model, args.max_side,
                        args.timeout, api_key, args.parallel)

    t0 = time.time()
    raw = args.out.with_suffix(".raw.jsonl")
    with raw.open("w") as log:
        sub = run(args.data, model, log, reasoning=not args.no_reasoning,
                  votes=args.votes, grid_zoom=not args.no_grid_zoom,
                  widths=not args.no_widths,
                  clean_image=not args.no_clean_image,
                  neighbours=not args.no_neighbours,
                  closeups=not args.no_closeups,
                  boxes=not args.no_boxes,
                  safety=not args.no_safety,
                  planner=not args.no_planner,
                  box_marks=not args.opencv_marks,
                  extract=not args.no_extract,
                  scaffold=not args.no_scaffold)
    sub["team"] = args.team
    args.out.write_text(json.dumps(sub, indent=2))
    print(f"{len(sub['results'])} answers in {args.out} "
          f"in {time.time() - t0:.0f}s; raw replies: {raw}")


if __name__ == "__main__":
    main()
