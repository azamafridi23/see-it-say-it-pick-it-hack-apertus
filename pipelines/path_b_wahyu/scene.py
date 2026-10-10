import json
from pathlib import Path


def extract_instructions(instruction_path: Path) -> list[dict]:
    """
    Parse instructions.txt.

    Expected format:
        # <category><TAB><instruction>
        simple      pick up the mandarin
        relative    pick up the object to the left of the mug
    """


    all_instructions = ""
    with instruction_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip comment/header
            if line.startswith("#"):
                continue

            # Prefer TAB separator because dataset format uses TAB
            parts = line.split("\t", maxsplit=1)

            # Fallback in case file uses spaces instead
            if len(parts) != 2:
                parts = line.split(maxsplit=1)

            if len(parts) != 2:
                print(
                    f"Warning: could not parse line in "
                    f"{instruction_path}: {line}"
                )
                continue

            category, instruction = parts

            all_instructions += f"- {instruction.strip()}\n"
    
    return all_instructions


def load_scenes(scene_main_folder_path: str | Path) -> dict:
    """
    Load instructions from all scene_xx directories.

    Returns:
        {
            "scene_01": [...],
            "scene_02": [...],
            ...
        }
    """

    scene_main_folder_path = Path(scene_main_folder_path)

    scenes = {}

    # Only take directories like scene_01, scene_02, ...
    scene_dirs = sorted(
        path
        for path in scene_main_folder_path.iterdir()
        if path.is_dir() and path.name.startswith("scene_")
    )

    for scene_dir in scene_dirs:
        instruction_file = scene_dir / "instructions.txt"

        if not instruction_file.exists():
            print(
                f"Warning: instructions.txt not found in "
                f"{scene_dir}"
            )
            continue

        instructions = extract_instructions(
            instruction_file
        )

        scenes[scene_dir.name] = {
            "image_scene_path": scene_dir / "rgb.png",
            "instruction_data": instructions
        }

    return scenes


def save_json(data: dict, output_path: str | Path):
    """
    Save parsed scenes into JSON file.
    """

    output_path = Path(output_path)

    # Automatically create output directory
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":
    dataset_path = Path("./dataset")
    output_path = Path("./results/scenes.json")

    scenes = load_scenes(dataset_path)
    # print(scenes)

    # save_json(
    #     scenes,
    #     output_path
    # )