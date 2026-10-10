#!/usr/bin/env python3
r"""Path B: from a photo and an instruction to a safe robot action.

Apertus is the only learned model and the main detector; OpenCV only
measures. See ARCHITECTURE.md and docs/adr/0001.

    APERTUS_API_KEY=... python pipelines/path_b_randy_nocrop/pipeline.py \
        --base-url https://hackapertus.livemap.sh/v1 \
        --model apertus-v1.5-70b --data dataset --out results.json
    python starter/evaluate.py results.json --data dataset

Or a local server:

    vllm serve swiss-ai/Apertus-v1.5-8B \
        --chat-template-content-format string
    python pipelines/path_b_randy_nocrop/pipeline.py \
        --data dataset --out results.json

Stages, one module each:
    detect.py     Apertus boxes every object on the whole photo; each box
                  is one object, its corrected centre the target pixel
    measure.py    OpenCV measures sizes, gaps, trays and what lies on top
    safety.py     Apertus refuses an unsafe instruction before planning
    planner.py    Apertus plans on a scene graph; code checks the plan
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
from apertus import ask, server_model
from safety import precheck, refusal
from scene_io import load_scene


def measured_objects(model, scene, widths, gaps):
    """Find the scene's objects with Apertus and measure them with OpenCV.

    Args:
        model: A model function, see ``apertus.server_model()``.
        scene: The scene, as loaded by ``scene_io.load_scene()``.
        widths: Whether to measure each object's width.
        gaps: Whether to measure the gaps between objects.

    Returns:
        The objects, see ``detect.find_objects()``, with ``colour``,
        ``on`` and, when measured, ``width_mm`` and ``gaps``.
    """
    objects = detect.find_objects(model, scene)
    for obj in objects:
        obj["colour"] = measure.colour_name(scene.rgb, obj)

    if widths:
        for obj in objects:
            obj["width_mm"] = round(measure.width_mm(scene, obj))

    if gaps:
        measure.add_gaps(scene, objects)

    measure.add_stacking(scene.rgb, objects)
    return objects


def run(data, model, log=None, widths=True, gaps=True, safety=True,
        thinker=None):
    """Answer every instruction of every scene in a folder.

    A scene's instructions are answered in parallel.

    Args:
        data: The folder of scenes, each with ``rgb.png`` and
            ``instructions.txt``.
        model: A model function, see ``apertus.server_model()``.
        log: A text file that gets one JSON line per instruction, or None.
        widths: Whether to measure the objects' widths.
        gaps: Whether to measure the gaps between objects.
        safety: Whether to run the safety pre-check before planning.
        thinker: A model function with Apertus's thinking on, to let the
            planner think before it answers; None to skip thinking.

    Returns:
        The submission: ``{"team": ..., "results": [...]}``.
    """
    results = []
    folders = sorted(folder for folder in Path(data).iterdir()
                     if (folder / "rgb.png").exists())

    # The planner only needs text in, text out; these bind it to the model.
    def ask_json(prompt, schema):
        return ask(model, prompt, [], schema)

    def ask_thinking(prompt):
        return ask(thinker, prompt, [], None)

    for folder in folders:
        scene = load_scene(folder)
        objects = measured_objects(model, scene, widths, gaps)
        graph = planner.scene_graph(objects,
                                    measure.black_mat_left(scene.rgb))

        def answer_instruction(instruction):
            # The category is never used: the held-out set has no labels.
            instruction_id, _category, text = instruction
            reason = safety and precheck(model, text)
            if reason:
                return (refusal(instruction_id, reason),
                        {"instruction_id": instruction_id,
                         "precheck": reason})

            return planner.plan(ask_json, instruction_id, text, graph,
                                ask_thinking if thinker else None)

        # ponytail: scenes run one after another and the client caps calls
        # in flight at 5; run scenes in parallel if the endpoint allows more.
        workers = max(1, len(scene.instructions))
        with ThreadPoolExecutor(workers) as pool:
            answered = list(pool.map(answer_instruction,
                                     scene.instructions))

        for result, record in answered:
            results.append(result)
            if log:
                log.write(json.dumps(record) + "\n")

    return {"team": "path_b_randy_nocrop", "results": results}


def main():
    """Run the pipeline from the command line and write the results."""
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

    # From the environment, never the command line: keys on the command
    # line end up in the shell history.
    api_key = os.environ.get("APERTUS_API_KEY", "none")
    model = server_model(args.base_url, args.model, args.max_side,
                         args.timeout, api_key, args.parallel)
    thinker = None
    if args.thinking:
        thinker = server_model(args.base_url, args.model, args.max_side,
                               args.timeout, api_key, args.parallel,
                               thinking=True)

    started = time.time()
    log_path = args.out.with_suffix(".raw.jsonl")
    with log_path.open("w") as log:
        sub = run(args.data, model, log, widths=not args.no_widths,
                  gaps=not args.no_gaps, safety=not args.no_safety,
                  thinker=thinker)
    sub["team"] = args.team
    args.out.write_text(json.dumps(sub, indent=2))
    print(f"{len(sub['results'])} answers in {args.out} "
          f"in {time.time() - started:.0f}s; raw replies: {log_path}")


if __name__ == "__main__":
    main()
