#!/usr/bin/env python3
"""Ask Apertus to list the objects it sees in a scene photo, in its words.

The names become extra SAM 3 prompts, so that objects outside the
detector's fixed list can still be found.

    python pipelines/path_a/apertus_list.py dataset   # every scene, 70B
    python pipelines/path_a/apertus_list.py dataset/scene_08 \
        --model apertus-v1.5-8b

Needs APERTUS_API_KEY in the .env file at the repository root. Any
OpenAI-compatible server works (--base-url). Writes
pipelines/path_a/out/apertus_objects_<model>.json: per scene, the raw
reply and the parsed list.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

from workcell import CELL

load_dotenv()

PROMPT = (
    "This photo shows a robot work table from above: a grey plastic tray "
    "on the left and a black mat on the right.\n"
    "List every separate physical object lying in the tray or on the mat. "
    "Do not list the tray, the mat, or the table frame.\n"
    "If the same kind of object appears several times, give one entry and "
    "its count.\n"
    "Name each object with a short phrase a person would use, including "
    "its colour, for example \"red gear\".\n"
    "\n"
    "Answer with only a JSON list, nothing else:\n"
    "[{\"name\": \"<short phrase>\", \"count\": <number>}]")

# Experiment (--with-list): see [apertus vision experiment] in workcell.ini.
PROMPT_WITH_LIST = (
    "This photo shows a robot work table from above: a grey plastic tray "
    "on the left and a black mat on the right.\n"
    "List every separate physical object lying in the tray or on the mat. "
    "Do not list the tray, the mat, or the table frame.\n"
    "\n"
    "We already know these object types: {known}.\n"
    "- If an object is one of these types, use exactly that name and set "
    "\"known\" to true.\n"
    "- If an object is none of these types, describe it with your own short "
    "phrase including its colour, and set \"known\" to false.\n"
    "- Only list what you can actually see in this photo. Most photos "
    "contain only a few of the known types.\n"
    "- If the same kind of object appears several times, give one entry and "
    "its count.\n"
    "\n"
    "Answer with only a JSON list, nothing else:\n"
    "[{{\"name\": \"<name>\", \"count\": <number>, "
    "\"known\": <true or false>}}]")

CACHE = Path(__file__).resolve().parent / "out" / "cache"


def photo_key(photo: Path) -> str:
    """Fingerprint the photo's content, as run_sam3.py does.

    Args:
        photo: The photo's path.

    Returns:
        The first 10 hex digits of the photo's SHA-1.
    """
    return hashlib.sha1(Path(photo).read_bytes()).hexdigest()[:10]


def encode(path: Path, max_side: int) -> str:
    """Shrink a photo and encode it as base64 JPEG for the API.

    Args:
        path: The photo's path.
        max_side: The longest side after shrinking, in pixels.

    Returns:
        The JPEG as base64 text.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)

    _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(jpeg.tobytes()).decode()


def parse(text: str):
    """Find the first JSON list in a reply.

    Args:
        text: The reply, possibly wrapped in a code fence.

    Returns:
        The list, or None if there is none.
    """
    unfenced = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\[.*\]", unfenced, re.S)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def list_objects(client: OpenAI, model: str, photo: Path,
                 max_side: int = 1024,
                 known=None) -> tuple[str, list | None]:
    """Ask Apertus which objects are in a photo.

    Args:
        client: The OpenAI-compatible client.
        model: The model name.
        photo: The photo's path.
        max_side: The longest side of the photo sent, in pixels.
        known: A list of names switches to the with-list experiment.

    Returns:
        Apertus's reply and the parsed list (None if unparsed).
    """
    if known is None:
        text = PROMPT
    else:
        text = PROMPT_WITH_LIST.format(known=", ".join(known))

    # Here, not at the top: planner.py does not need this file.
    from planner import with_patience
    image_url = f"data:image/jpeg;base64,{encode(photo, max_side)}"
    content = [{"type": "image_url", "image_url": {"url": image_url}},
               {"type": "text", "text": text}]
    response = with_patience(
        client.chat.completions.create, model=model, temperature=0,
        max_tokens=500, messages=[{"role": "user", "content": content}])
    reply = response.choices[0].message.content or ""
    return reply, parse(reply)


def names_for(scene: Path, model: str = CELL.model,
              base_url: str = CELL.base_url) -> list[str]:
    """Get the object names Apertus gives for a scene.

    They are used as extra SAM 3 prompts. Cached per photo and model.

    Args:
        scene: The scene folder.
        model: The model name.
        base_url: The server's OpenAI-compatible URL.

    Returns:
        The names, lower case and without duplicates; [] when Apertus is
        unreachable or there is no key.
    """
    scene = Path(scene)
    key = photo_key(scene / "rgb.png")
    path = CACHE / f"{scene.name}_{key}_apertus_{model}.json"
    if path.exists():
        objs = json.loads(path.read_text())["objects"]
    else:
        api_key = os.environ.get("APERTUS_API_KEY")
        if not api_key:
            print("  no APERTUS_API_KEY: skipping Apertus's object names")
            return []

        try:
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
            reply, objs = list_objects(client, model, scene / "rgb.png")
        except Exception as error:
            print(f"  Apertus unreachable ({type(error).__name__}): "
                  f"skipping its object names")
            return []

        CACHE.mkdir(parents=True, exist_ok=True)
        saved = {"objects": objs, "reply": reply}
        path.write_text(json.dumps(saved, indent=2))

    names = []
    for obj in objs or []:
        if not isinstance(obj, dict):
            continue

        name = obj.get("name")
        if isinstance(name, str) and 2 < len(name.strip()) <= 60:
            names.append(name.strip().lower())

    return list(dict.fromkeys(names))


def _shown(objs, reply) -> str:
    """Describe a parsed list for the console."""
    if objs is None:
        return f"UNPARSED: {reply[:200]!r}"

    parts = []
    for obj in objs:
        new = " [NEW]" if obj.get("known") is False else ""
        parts.append(f"{obj.get('count', 1)}x {obj.get('name')}{new}")

    return ", ".join(parts)


def main():
    """List the objects in one scene or every scene, and save the replies."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path,
                    help="a scene folder, or a folder of scenes")
    ap.add_argument("--model", default=CELL.model)
    ap.add_argument("--base-url", default=CELL.base_url)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--with-list", action="store_true",
                    help="experiment: tell Apertus the names the detector "
                         "knows")
    ap.add_argument("--hide", nargs="*", default=[],
                    help="with --with-list: names to leave out of the list")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "out")
    args = ap.parse_args()
    api_key = os.environ.get("APERTUS_API_KEY")
    if not api_key:
        raise SystemExit("APERTUS_API_KEY is not set: put "
                         "APERTUS_API_KEY=... in the .env file at the "
                         "repository root")

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=120)
    if (args.data / "rgb.png").exists():
        scenes = [args.data]
    else:
        scenes = sorted(folder for folder in args.data.iterdir()
                        if (folder / "rgb.png").exists())

    known = None
    if args.with_list:
        known = [name for name in CELL.known_names
                 if name not in args.hide]

    results = {}
    for scene in scenes:
        started = time.time()
        reply, objs = list_objects(client, args.model, scene / "rgb.png",
                                   args.max_side, known)
        results[scene.name] = {"objects": objs, "reply": reply}
        print(f"{scene.name} ({time.time() - started:.0f}s): "
              f"{_shown(objs, reply)}")

    args.out.mkdir(exist_ok=True)
    suffix = ""
    if args.with_list:
        suffix += "_with-list"

    if args.hide:
        suffix += "_hidden"

    path = args.out / f"apertus_objects_{args.model}{suffix}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
