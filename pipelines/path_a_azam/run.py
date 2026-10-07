#!/usr/bin/env python3
"""The whole pipeline on a folder of scenes: photo -> scene graph -> Apertus -> results file.

    python pipelines/path_a_azam/run.py --data dataset --out results/path_a_azam.json
    python starter/evaluate.py results/path_a_azam.json --data dataset

    --prompt planner_v2    try another version of the rules given to Apertus (files in prompts/)

Per photo:        SAM 3 finds the objects, Apertus names what it sees, code builds the scene graph (scene_graph.py).
Per instruction:  Apertus reads the scene graph and the instruction and answers; code checks the answer (planner.py).

Needs HF_TOKEN (SAM 3) and API_KEY (Apertus) in a .env file. Any OpenAI-compatible server works: --base-url, --model.
Writes the results file and <out>.log.jsonl: every reply from Apertus and every correction, for the failure report.
The category in instructions.txt is not used.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import planner
from apertus_list import names_for
from run_sam3 import detect
from scene_graph import build, render, scene_dirs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "starter"))
from evaluate import answer_type  # noqa: E402
from scene_io import load_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="a folder of scenes, or one scene folder")
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--model", default=planner.MODEL)
    ap.add_argument("--base-url", default=planner.BASE_URL)
    ap.add_argument("--prompt", default=planner.DEFAULT_PROMPT,
                    help=f"planner prompt: a name in prompts/ or a path to a text file (default: {planner.DEFAULT_PROMPT})")
    ap.add_argument("--team", default="team")
    ap.add_argument("--no-apertus-vision", action="store_true", help="do not ask Apertus to name the objects it sees")
    ap.add_argument("--graphs", type=Path, default=Path(__file__).resolve().parent / "out", help="where the scene graphs are written")
    args = ap.parse_args()

    plan = planner.Planner(args.model, args.base_url, prompt=args.prompt)
    print(f"model: {plan.model}   planner prompt: {plan.prompt_name}")
    args.graphs.mkdir(exist_ok=True)
    results, t0 = [], time.time()
    with args.out.with_suffix(".log.jsonl").open("w") as log_file:
        for d in scene_dirs(args.data):
            extra = [] if args.no_apertus_vision else detect(d, names_for(d, args.model, args.base_url))
            graph, objs = build(d, detect(d), extra=extra)
            (args.graphs / f"{d.name}_graph.json").write_text(json.dumps(graph, indent=2))
            render(d, objs, args.graphs / f"{d.name}_graph.jpg")
            for iid, _, text in load_scene(d).instructions:
                answer, log = plan.answer(iid, text, graph)
                kind, err = answer_type(answer)
                results.append(answer)
                log_file.write(json.dumps({**log, "answer": answer}) + "\n")
                tries = len(log["attempts"])
                shown = ", ".join(f"{a['action']}({a['object']})" for a in answer["actions"]) if kind == "plan" \
                    else f"{kind}: {answer['actions'][0].get('question') or answer['actions'][0].get('reason')}" if kind \
                    else f"INVALID ({err})"
                print(f"{iid}  {text!r}\n    -> {shown}" + (f"   [{tries} attempts]" if tries > 1 else ""))
    args.out.write_text(json.dumps({"team": args.team, "results": results}, indent=2))
    print(f"\n{len(results)} answers in {args.out} in {time.time() - t0:.0f}s; log: {args.out.with_suffix('.log.jsonl')}")


if __name__ == "__main__":
    main()
