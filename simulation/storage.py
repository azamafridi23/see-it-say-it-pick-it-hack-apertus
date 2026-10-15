"""Read-only demo data. No model imports and no ground-truth access."""
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The files each pipeline's run.py writes for the public scenes. The
# single-call planner has none in the repository; to replay it, run
# pipelines/path_a/run.py --planner single --data dataset --out results/path_a_single_dev.json
RESULT_FILES = {
    "path_a_graph": ROOT / "results/path_a_dev.json",
    "path_a": ROOT / "results/path_a_single_dev.json",
    "path_b": ROOT / "results/path_b_dev.json",
}


def scene_folder(scene_id: str) -> Path:
    # Use an allowlist rather than accepting arbitrary paths from HTTP.
    folders = {p.name: p for p in (ROOT / "dataset").iterdir() if p.is_dir() and (p / "rgb.png").is_file()}
    if scene_id not in folders:
        raise ValueError("Unknown scene")
    return folders[scene_id]


def image_size(folder: Path) -> list[int]:
    with (folder / "rgb.png").open("rb") as stream:
        header = stream.read(24)
    return list(struct.unpack(">II", header[16:24]))


def instruction_file(folder: Path) -> Path | None:
    for name in ("instructions.txt", "instruction.txt"):
        path = folder / name
        if path.is_file():
            return path
    return None


def instructions(folder: Path) -> list[dict]:
    items = []
    source = instruction_file(folder)
    if source is None:
        return items
    for line in source.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        _, separator, text = line.partition("\t")
        items.append({"id": f"{folder.name}_i{len(items) + 1}", "text": text.strip() if separator else line.strip()})
    return items


def answers(pipeline: str) -> list[dict]:
    path = RESULT_FILES[pipeline]
    if not path.is_file():
        return []
    return json.loads(path.read_text()).get("results", [])


def saved_objects(scene_id: str, pipeline: str) -> list[dict]:
    """Recover known positions from saved actions; missing geometry stays schematic."""
    nodes = {}
    for answer in answers(pipeline):
        if not answer["instruction_id"].startswith(scene_id + "_i"):
            continue
        first_pick = True
        for action in answer["actions"]:
            oid = action.get("object")
            if not oid:
                continue
            node = nodes.setdefault(oid, {"id": oid, "type": oid.replace("_", " "), "pixel": None})
            pixel = action.get("pixel")
            if action["action"] == "pick" and first_pick:
                pixel = pixel or answer.get("target_pixel")
                first_pick = False
            if pixel:
                node["pixel"] = pixel
    return list(nodes.values())


def replay(scene_id: str, pipeline: str, instruction_id: str) -> dict:
    folder = scene_folder(scene_id)
    options = instructions(folder)
    instruction = next((i for i in options if i["id"] == instruction_id), None)
    answer = next((a for a in answers(pipeline) if a["instruction_id"] == instruction_id), None)
    if instruction is None or answer is None:
        raise ValueError("No saved result for this instruction and pipeline. Choose another pipeline or use Live.")
    # run.py writes its log to results/logs/<results name>.log.jsonl.
    log_path = ROOT / "results/logs" / f"{RESULT_FILES[pipeline].stem}.log.jsonl"
    elapsed = None
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("instruction_id") == instruction_id:
                elapsed = record.get("elapsed_seconds")
                break
    return {"scene_id": scene_id, "pipeline": pipeline, "mode": "replay", "instruction": instruction["text"],
            "answer": answer, "objects": saved_objects(scene_id, pipeline), "image_size": image_size(folder),
            "geometry_source": "saved_pixels", "elapsed_seconds": elapsed}
