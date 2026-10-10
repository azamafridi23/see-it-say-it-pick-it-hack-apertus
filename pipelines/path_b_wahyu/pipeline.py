"""Run with python -m pipelines.path_b_wahyu.pipeline --data dataset."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .multimodal import llm_processing
from .scene import load_scenes, save_json


def finalizing_output_format(scene, llm_output):
    """Map local numeric IDs into submission IDs and derive the first target pixel."""
    output = []
    for idx, instruction in enumerate(llm_output, 1):
        if instruction["instruction_id"] != idx:
            raise ValueError("Instruction order mismatch")
        actions = instruction["actions_data"]
        output.append(
            {
                "instruction_id": f"{scene}_i{idx}",
                "target_pixel": actions[0].get("pixel"),
                "actions": actions,
            }
        )
    return output


def pipeline(
    dataset_dir_path,
    output_path="result_wahyu.json",
    *,
    model=None,
    debug_dir=None,
    selected_scenes=None,
    workers=1,
    cache_dir=None,
):
    """Process scenes concurrently, checkpoint ordered results, and report partial failures."""
    scenes = load_scenes(dataset_dir_path)
    if selected_scenes:
        unknown = set(selected_scenes) - scenes.keys()
        if unknown:
            raise ValueError(f"Unknown scenes: {sorted(unknown)}")
        scenes = {s: v for s, v in scenes.items() if s in selected_scenes}
    if not scenes:
        raise ValueError("No scenes found")
    if workers < 1:
        raise ValueError("workers must be positive")
    result = {"team": "path_b_wahyu", "results": []}
    errors = {}

    def process(item):
        name, scene = item
        print(f"Processing {name}", flush=True)
        try:
            predictions = llm_processing(
                scene["instruction_data"],
                scene["image_scene_path"],
                model=model,
                debug_dir=Path(debug_dir) / name if debug_dir else None,
                cache_dir=cache_dir,
            )
            issues = {
                str(p["instruction_id"]): p["processing_error"]
                for p in predictions
                if "processing_error" in p
            }
            return name, finalizing_output_format(name, predictions), issues
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"{name}: {error}", flush=True)
            count = len(
                [s for s in scene["instruction_data"].splitlines() if s.strip()]
            )
            entries = [
                {
                    "instruction_id": f"{name}_i{i}",
                    "target_pixel": None,
                    "actions": [
                        {
                            "action": "ask_user",
                            "question": "Scene processing failed; please retry perception before moving the robot.",
                        }
                    ],
                }
                for i in range(1, count + 1)
            ]
            return name, entries, {"scene": error}

    # Workers may finish out of order; only this thread writes ordered checkpoints.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for name, entries, issues in executor.map(process, scenes.items()):
            result["results"].extend(entries)
            if issues:
                errors[name] = issues
                print(f"{name}: {len(issues)} processing error(s)", flush=True)
            save_json(result, output_path)
            save_json(errors, str(output_path) + ".errors.json")
            print(f"Finished {name}", flush=True)
    # Preserve partial results without reporting a failed run as successful.
    if errors:
        raise RuntimeError(
            f"{len(errors)} scene(s) have processing errors; see {output_path}.errors.json"
        )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="result_wahyu.json")
    ap.add_argument("--model")
    ap.add_argument("--debug-dir", default="pipelines/path_b_wahyu/out")
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cache-dir", default="pipelines/path_b_wahyu/out/cache")
    args = ap.parse_args()
    pipeline(
        args.data,
        args.out,
        model=args.model,
        debug_dir=args.debug_dir,
        selected_scenes=args.scenes,
        workers=args.workers,
        cache_dir=args.cache_dir,
    )
