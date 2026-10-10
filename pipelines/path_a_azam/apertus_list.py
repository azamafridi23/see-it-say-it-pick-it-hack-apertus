#!/usr/bin/env python3
"""Apertus lists the objects it sees in a scene photo, in its own words. The names become extra SAM 3 prompts,
so that objects outside the detector's fixed list can still be found.

    python pipelines/path_a_azam/apertus_list.py dataset                          every scene, 70B model
    python pipelines/path_a_azam/apertus_list.py dataset/scene_08 --model apertus-v1.5-8b

Needs API_KEY in the .env file at the repository root. Any OpenAI-compatible server works (--base-url).
Writes pipelines/path_a_azam/out/apertus_objects_<model>.json: per scene, the raw reply and the parsed list.
"""
from __future__ import annotations

import argparse, base64, hashlib, json, os, re, time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT = """This photo shows a robot work table from above: a grey plastic tray on the left and a black mat on the right.
List every separate physical object lying in the tray or on the mat. Do not list the tray, the mat, or the table frame.
If the same kind of object appears several times, give one entry and its count.
Name each object with a short phrase a person would use, including its colour, for example "red gear".

Answer with only a JSON list, nothing else:
[{"name": "<short phrase>", "count": <number>}]"""

# Experiment (--with-list): tell Apertus the names the detector already knows. It made the lists worse: on the ten
# public scenes it listed about 18 known objects that were not in the photo (1 without the list), and with a name
# hidden from the list it reported the mug as a new object in only 2 of 6 scenes (6 of 6 without the list).
KNOWN = ["red gear", "yellow gear", "red cube", "mug", "mandarin", "banana", "plate", "saucer",
         "grey metal part", "white part", "hand"]
PROMPT_WITH_LIST = """This photo shows a robot work table from above: a grey plastic tray on the left and a black mat on the right.
List every separate physical object lying in the tray or on the mat. Do not list the tray, the mat, or the table frame.

We already know these object types: {known}.
- If an object is one of these types, use exactly that name and set "known" to true.
- If an object is none of these types, describe it with your own short phrase including its colour, and set "known" to false.
- Only list what you can actually see in this photo. Most photos contain only a few of the known types.
- If the same kind of object appears several times, give one entry and its count.

Answer with only a JSON list, nothing else:
[{{"name": "<name>", "count": <number>, "known": <true or false>}}]"""

MODEL = "apertus-v1.5-70b"
BASE_URL = "https://hackapertus.livemap.sh/v1"
CACHE = Path(__file__).resolve().parent / "out" / "cache"


def photo_key(photo: Path) -> str:
    """A short fingerprint of the photo's content (the same one run_sam3.py uses)."""
    return hashlib.sha1(Path(photo).read_bytes()).hexdigest()[:10]


def encode(path: Path, max_side: int) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    scale = min(1.0, max_side / max(img.shape[:2]))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()


def parse(text: str):
    """The first JSON list in the reply, or None."""
    m = re.search(r"\[.*\]", re.sub(r"```(?:json)?", "", text), re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def list_objects(client: OpenAI, model: str, photo: Path, max_side: int = 1024, known=None) -> tuple[str, list | None]:
    """Apertus's reply and the parsed list. known = a list of names switches to the with-list experiment."""
    text = PROMPT if known is None else PROMPT_WITH_LIST.format(known=", ".join(known))
    from planner import with_patience  # here, not at the top: planner.py does not need this file
    r = with_patience(client.chat.completions.create, model=model, temperature=0, max_tokens=500, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode(photo, max_side)}"}},
        {"type": "text", "text": text}]}])
    reply = r.choices[0].message.content or ""
    return reply, parse(reply)


def names_for(scene: Path, model: str = MODEL, base_url: str = BASE_URL) -> list[str]:
    """The object names Apertus gives for a scene, to use as extra SAM 3 prompts. Cached; [] when Apertus is unreachable."""
    path = CACHE / f"{Path(scene).name}_{photo_key(Path(scene) / 'rgb.png')}_apertus_{model}.json"
    if path.exists():
        objs = json.loads(path.read_text())["objects"]
    else:
        if not os.environ.get("API_KEY"):
            print("  no API_KEY: skipping Apertus's object names")
            return []
        try:
            reply, objs = list_objects(OpenAI(api_key=os.environ["API_KEY"], base_url=base_url, timeout=120), model, Path(scene) / "rgb.png")
        except Exception as e:
            print(f"  Apertus unreachable ({type(e).__name__}): skipping its object names")
            return []
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"objects": objs, "reply": reply}, indent=2))
    names = [o.get("name") for o in objs or [] if isinstance(o, dict)]
    return list(dict.fromkeys(n.strip().lower() for n in names if isinstance(n, str) and 2 < len(n.strip()) <= 60))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="a scene folder, or a folder of scenes")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--with-list", action="store_true", help="experiment: tell Apertus the names the detector knows")
    ap.add_argument("--hide", nargs="*", default=[], help="with --with-list: names to leave out of the list")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = ap.parse_args()
    if not os.environ.get("API_KEY"):
        raise SystemExit("API_KEY is not set: put API_KEY=... in the .env file at the repository root")
    client = OpenAI(api_key=os.environ["API_KEY"], base_url=args.base_url, timeout=120)

    scenes = [args.data] if (args.data / "rgb.png").exists() else sorted(p for p in args.data.iterdir() if (p / "rgb.png").exists())
    results = {}
    for d in scenes:
        t0 = time.time()
        reply, objs = list_objects(client, args.model, d / "rgb.png", args.max_side, [k for k in KNOWN if k not in args.hide] if args.with_list else None)
        results[d.name] = {"objects": objs, "reply": reply}
        shown = ", ".join(f"{o.get('count', 1)}x {o.get('name')}{' [NEW]' if o.get('known') is False else ''}" for o in objs) if objs is not None else f"UNPARSED: {reply[:200]!r}"
        print(f"{d.name} ({time.time() - t0:.0f}s): {shown}")
    args.out.mkdir(exist_ok=True)
    path = args.out / f"apertus_objects_{args.model}{'_with-list' if args.with_list else ''}{'_hidden' if args.hide else ''}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
