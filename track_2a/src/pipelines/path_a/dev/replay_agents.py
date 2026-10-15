"""Evaluate agent prompts on saved scene graphs, without vision models.

python pipelines/path_a/dev/replay_agents.py --graphs pipelines/path_a/out \
    --data dataset --out results/agents_v3.json

The saved graphs are the only perception input. Instructions are read
without their category labels. This calls Apertus unless --dry-run is
passed; it does not change graphs or load ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import GraphPlanner  # noqa: E402
from workcell import CELL  # noqa: E402


def cases(graphs_dir: Path, data_dir: Path):
    """Yield every instruction with its saved scene graph.

    Args:
        graphs_dir: The folder of ``<scene>_graph.json`` files.
        data_dir: The folder of scenes, for the instructions.

    Yields:
        The instruction id, the instruction without its category, the
        scene graph, and the graph file's SHA-256.

    Raises:
        ValueError: No graphs, or a graph names another scene.
    """
    files = sorted(graphs_dir.glob("*_graph.json"))
    if not files:
        raise ValueError(f"No *_graph.json files in {graphs_dir}; generate "
                         f"scene graphs first.")

    for path in files:
        raw = path.read_bytes()
        graph = json.loads(raw)
        name = path.name.removesuffix("_graph.json")
        if graph.get("scene") != name:
            raise ValueError(f"Scene name mismatch in {path}.")

        rows = (data_dir / name / "instructions.txt").read_text() \
            .splitlines()
        instructions = [line.strip() for line in rows
                        if line.strip()
                        and not line.lstrip().startswith("#")]
        graph_hash = hashlib.sha256(raw).hexdigest()
        for number, line in enumerate(instructions, 1):
            _, separator, text = line.partition("\t")
            if not separator:
                text = line

            yield f"{name}_i{number}", text.strip(), graph, graph_hash


def main():
    """Replay the agents on the saved graphs and write the results."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("results/agents_v3.json"))
    parser.add_argument("--model", default=CELL.model)
    parser.add_argument("--base-url", default=CELL.base_url)
    parser.add_argument("--team", default="team")
    parser.add_argument("--dry-run", action="store_true",
                        help="check input files and count instructions "
                             "without model calls")
    for role in ("safety", "reference", "constraint"):
        parser.add_argument(f"--{role}-prompt",
                            help=f"{role} prompt name or path; must follow "
                                 f"the current schema")

    args = parser.parse_args()
    # Read inputs before constructing a client or opening output files.
    try:
        inputs = list(cases(args.graphs, args.data))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(f"{len(inputs)} instructions in "
              f"{len({g['scene'] for _, _, g, _ in inputs})} frozen scene "
              f"graphs; no model calls.")
        return

    overrides = {role: value
                 for role in ("safety", "reference", "constraint")
                 if (value := getattr(args, f"{role}_prompt")) is not None}
    planner = GraphPlanner(model=args.model, base_url=args.base_url,
                           prompt_names=overrides)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        with args.out.with_suffix(".log.jsonl").open("w") as log_file:
            for iid, instruction, graph, graph_hash in inputs:
                start = time.monotonic()
                answer, log = planner.answer(iid, instruction, graph)
                results.append(answer)
                log_file.write(json.dumps(
                    {**log, "graph_sha256": graph_hash,
                     "elapsed_seconds": round(time.monotonic() - start, 3),
                     "answer": answer}) + "\n")
                log_file.flush()
                print(f"{iid}: {answer['actions'][0]['action']}")
    finally:
        planner.client.close()

    args.out.write_text(json.dumps({"team": args.team, "results": results},
                                   indent=2))
    print(f"Wrote {len(results)} answers to {args.out}")


if __name__ == "__main__":
    main()
