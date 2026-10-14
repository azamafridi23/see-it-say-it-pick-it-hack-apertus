#!/usr/bin/env python3
"""Run Path A on a folder of scenes and write the results file.

    python pipelines/path_a/run.py                  # held-out scenes
    python pipelines/path_a/run.py --data dataset   # public, with score

Held-out answers go to results_heldout.json, the file the organisers ask
for; public ones to results/path_a_dev.json, then the score is printed.
Every Apertus reply and correction goes to logs/<results name>.log.jsonl
next to the results file, for the failure report.

    --planner single       one Apertus call instead of the three agents
    --prompt <name>        with --planner single: another version of its
                           rules (files in prompts/)

Per photo: SAM 3 finds the objects, Apertus names what it sees, and code
builds the scene graph (scene_graph.py). Per instruction: Apertus reads
the scene graph and the instruction and answers, and code checks the
answer (agents.py, or planner.py with --planner single).

Needs APERTUS_API_KEY in a .env file. Any OpenAI-compatible server works:
--base-url, --model. The category in instructions.txt is not used.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import planner
from workcell import CELL


def main():
    """Answer every instruction in the chosen scenes and write the results."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--data", type=Path, default=root / "heldout",
                    help="a folder of scenes, or one scene folder")
    ap.add_argument("--out", type=Path,
                    help="default: results_heldout.json for heldout, "
                         "else results/path_a_dev.json")
    ap.add_argument("--model", default=CELL.model)
    ap.add_argument("--base-url", default=CELL.base_url)
    ap.add_argument("--prompt", default=CELL.single_call_prompt,
                    help=f"planner prompt: a name in prompts/ or a path to a "
                         f"text file (default: {CELL.single_call_prompt})")
    ap.add_argument("--safety-prompt",
                    help="graph planner safety prompt name or file")
    ap.add_argument("--reference-prompt",
                    help="graph planner reference prompt name or file")
    ap.add_argument("--constraint-prompt",
                    help="graph planner constraint prompt name or file "
                         "(v3 schema)")
    ap.add_argument("--planner", choices=["single", "graph"],
                    default="graph",
                    help="single: one Apertus call decides everything "
                         "(planner.py). graph: three Apertus agents in a "
                         "fixed LangGraph graph (agents.py)")
    ap.add_argument("--team", default="path_a")
    ap.add_argument("--no-apertus-vision", action="store_true",
                    help="do not ask Apertus to name the objects it sees")
    ap.add_argument("--graphs", type=Path,
                    default=Path(__file__).resolve().parent / "out",
                    help="where the scene graphs are written")
    args = ap.parse_args()

    # The organisers ask for results_heldout.json from the held-out scenes.
    if args.out is None and args.data.name == "heldout":
        args.out = root / "results_heldout.json"
    elif args.out is None:
        args.out = root / "results" / "path_a_dev.json"

    # Every Apertus reply, in a logs/ folder next to the results.
    log_path = args.out.parent / "logs" / f"{args.out.stem}.log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    overrides = {}
    for role in ("safety", "reference", "constraint"):
        prompt = getattr(args, f"{role}_prompt")
        if prompt is not None:
            overrides[role] = prompt

    if args.planner != "graph" and overrides:
        ap.error("Agent prompt overrides require --planner graph.")

    from scene_graph import scene_dirs
    sys.path.insert(0, str(root / "starter"))
    from evaluate import answer_type
    from scene_io import load_scene

    if args.planner == "graph":
        # Only this choice needs the agents, and with them LangGraph.
        from agents import GraphPlanner
        plan = GraphPlanner(args.model, args.base_url,
                            prompt_names=overrides)
    else:
        plan = planner.Planner(args.model, args.base_url, prompt=args.prompt)

    print(f"model: {plan.model}   planner: {args.planner}   "
          f"prompt: {plan.prompt_name}")
    args.graphs.mkdir(exist_ok=True)

    results, started = [], time.time()
    with log_path.open("w") as log_file:
        for scene in scene_dirs(args.data):
            graph = _scene_graph(scene, args)
            for instruction_id, _, text in load_scene(scene).instructions:
                answer, log = plan.answer(instruction_id, text, graph)
                results.append(answer)
                log_file.write(json.dumps({**log, "answer": answer}) + "\n")
                print(_progress_line(instruction_id, text, answer, log,
                                     answer_type))

    submission = {"team": args.team, "results": results}
    args.out.write_text(json.dumps(submission, indent=2))
    print(f"\n{len(results)} answers in {args.out} in "
          f"{time.time() - started:.0f}s; log: {log_path}", flush=True)

    # Public scenes: score right away. Held-out scenes have no answers.
    if (args.data / "ground_truth").exists():
        evaluate = root / "starter" / "evaluate.py"
        subprocess.run([sys.executable, str(evaluate), str(args.out),
                        "--data", str(args.data)])


def _scene_graph(scene: Path, args) -> dict:
    """Build one scene's graph and save it, with a drawing, for checking."""
    from apertus_list import names_for
    from run_sam3 import detect
    from scene_graph import build, render

    if args.no_apertus_vision:
        extra = []
    else:
        extra = detect(scene, names_for(scene, args.model, args.base_url))

    graph, objs = build(scene, detect(scene), extra=extra)
    graph_file = args.graphs / f"{scene.name}_graph.json"
    graph_file.write_text(json.dumps(graph, indent=2))
    render(scene, objs, args.graphs / f"{scene.name}_graph.jpg")
    return graph


def _progress_line(instruction_id, text, answer, log, answer_type) -> str:
    """Describe one answer for the console."""
    kind, error = answer_type(answer)
    actions = answer["actions"]
    if kind == "plan":
        shown = ", ".join(f"{action['action']}({action['object']})"
                          for action in actions)
    elif kind:
        message = actions[0].get("question") or actions[0].get("reason")
        shown = f"{kind}: {message}"
    else:
        shown = f"INVALID ({error})"

    line = f"{instruction_id}  {text!r}\n    -> {shown}"
    # Corrections by the single-call planner.
    tries = sum("agent" not in attempt for attempt in log["attempts"])
    if tries > 1:
        line += f"   [{tries} attempts]"

    return line


if __name__ == "__main__":
    main()
