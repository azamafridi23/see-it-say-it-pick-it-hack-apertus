#!/usr/bin/env python3
"""Path B: Apertus is the only learned model and the main detector;
OpenCV only measures.

See docs/adr/0001 and ARCHITECTURE.md.

    APERTUS_API_KEY=... python pipelines/path_b_randy_nocrop/pipeline.py \\
        --base-url https://hackapertus.livemap.sh/v1 \\
        --model apertus-v1.5-70b --data dataset --out results.json
    python starter/evaluate.py results.json --data dataset

Or a local server:

    vllm serve swiss-ai/Apertus-v1.5-8B \\
        --chat-template-content-format string
    python pipelines/path_b_randy_nocrop/pipeline.py \\
        --data dataset --out results.json

Stages, one module each:
    detect.py     Apertus boxes every object on the whole photo; each box
                  is one object, its corrected centre the target pixel
    measure.py    OpenCV measures: sizes, gaps, trays, what lies on top
    safety.py     per instruction, Apertus refuses an unsafe sentence
    planner.py    Path A's planner on a scene graph of the objects

    --mock     no model: refuses everything (checks the loop and the format)
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paths  # noqa: F401
import detect
import measure
import planner
from apertus import apertus, ask, mock
from safety import precheck, refusal
from scene_io import load_scene


def find_objects(model, sc, widths, gaps):
    """Return the scene's objects: found by Apertus, measured by OpenCV."""
    marks = detect.objects(model, sc)
    for m in marks:
        m["colour"] = measure.mark_colour(sc.rgb, m)

    if widths:
        for m in marks:
            m["width_mm"] = round(measure.width_mm(sc, m))

    if gaps:
        measure.add_gaps(sc, marks)

    measure.stacked_on(sc.rgb, marks)
    return marks


def run(data, model, log=None, widths=True, gaps=True, safety=True,
        thinker=None):
    """Answer every instruction of every scene in `data`.

    `model(prompt, images, schema, temperature)` returns the reply text.
    `widths` and `gaps` measure sizes and gaps for the scene graph;
    `safety` refuses before planning when the sentence alone must be
    refused; `thinker`, a model function with Apertus's thinking on, lets
    the planner think before it answers. Returns the submission dict.

    A scene's instructions are answered in parallel.
    """
    results = []
    scenes = sorted(p for p in Path(data).iterdir()
                    if (p / "rgb.png").exists())

    def answer(prompt, schema):
        return ask(model, prompt, [], schema)

    def think(prompt):
        return ask(thinker, prompt, [], None)

    for d in scenes:
        sc = load_scene(d)
        marks = find_objects(model, sc, widths, gaps)
        graph = planner.scene_graph(marks, measure.black_mat_x(sc.rgb))

        def solve(instruction):
            # The category is never used: the held-out set has no labels.
            iid, _category, text = instruction
            reason = safety and precheck(model, text)
            if reason:
                return (refusal(iid, reason),
                        {"instruction_id": iid, "precheck": reason})

            return planner.plan(answer, iid, text, graph,
                                think if thinker else None)

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
    ap.add_argument("--thinking", action="store_true",
                    help="the planner thinks first (Apertus's thinking "
                         "mode; slower, and worse on the dev scenes)")
    ap.add_argument("--no-safety", action="store_true",
                    help="no pre-check before planning")
    ap.add_argument("--no-widths", action="store_true",
                    help="leave the measured sizes out")
    ap.add_argument("--no-gaps", action="store_true",
                    help="leave the measured gaps out")
    args = ap.parse_args()

    thinker = None
    if args.mock:
        model = mock
    else:
        # From the environment, never the command line: keys on the
        # command line end up in the shell history.
        api_key = os.environ.get("APERTUS_API_KEY", "none")
        model = apertus(args.base_url, args.model, args.max_side,
                        args.timeout, api_key, args.parallel)
        if args.thinking:
            thinker = apertus(args.base_url, args.model, args.max_side,
                              args.timeout, api_key, args.parallel,
                              thinking=True)

    t0 = time.time()
    raw = args.out.with_suffix(".raw.jsonl")
    with raw.open("w") as log:
        sub = run(args.data, model, log, widths=not args.no_widths,
                  gaps=not args.no_gaps, safety=not args.no_safety,
                  thinker=thinker)
    sub["team"] = args.team
    args.out.write_text(json.dumps(sub, indent=2))
    print(f"{len(sub['results'])} answers in {args.out} "
          f"in {time.time() - t0:.0f}s; raw replies: {raw}")


if __name__ == "__main__":
    main()
