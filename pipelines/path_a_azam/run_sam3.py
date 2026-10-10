#!/usr/bin/env python3
"""Try SAM 3 on one scene: one text prompt per object type, all instances back as masks.

    python pipelines/path_a_azam/run_sam3.py dataset/scene_06
    python pipelines/path_a_azam/run_sam3.py dataset/scene_06 --prompts "red gear" "yellow gear" --threshold 0.4

Needs HF_TOKEN in the .env file at the repository root: facebook/sam3 is a gated model.

Writes pipelines/path_a_azam/out/<scene>_overlay.jpg (masks drawn on the photo, to check by eye) and
out/<scene>_detections.json (label, score, box, an interior pixel, mask area).
Pixels are [u, v] in the full-resolution photo, as in the challenge's target_pixel.

These are the raw detections. scene_graph.py filters and merges them into a scene graph.
"""
from __future__ import annotations

import argparse, hashlib, json, os, time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops MPS lacks run on the CPU

from dotenv import load_dotenv

load_dotenv()  # HF_TOKEN from the nearest .env, searching upwards from this file; before transformers is imported

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# the objects named in dataset/README.md, plus a hand for the unsafe cases.
# "metal bracket" is the grey metal parts: "grey metal part" also returned the grey tray and the rails.
# "orange fruit" and "small yellow gear" repeat "mandarin" and "yellow gear", which each missed in some scenes.
# "object" is a catch-all: it finds the small white part whole, and anything not named here.
# "grey tray" and "black mat" are the two surfaces of the table: their outlines have holes where objects lie.
PROMPTS = ["red gear", "yellow gear", "small yellow gear", "red cube", "mug", "mandarin", "orange fruit", "banana",
           "plate", "saucer", "metal bracket", "white part", "grey tray", "black mat", "hand", "object"]
THRESHOLD = 0.5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = Path(__file__).resolve().parent / "out" / "cache"

_loaded = {}


def load(device: str = DEVICE):
    if device not in _loaded:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN is not set: put HF_TOKEN=... in the .env file at the repository root "
                             "(facebook/sam3 is a gated model)")
        t0 = time.time()
        model = Sam3Model.from_pretrained("facebook/sam3", token=token).to(device).eval()
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=token)
        print(f"model loaded on {device} in {time.time() - t0:.0f}s")
        _loaded[device] = (model, processor)
    return _loaded[device]


def _pack(path: Path, dets: list[dict], shape) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.stack([d["mask"].reshape(-1) for d in dets]) if dets else np.zeros((0, shape[0] * shape[1]), bool)
    np.savez_compressed(path, h=shape[0], w=shape[1], scores=np.array([d["score"] for d in dets], float),
                        boxes=np.array([d["box_xyxy"] for d in dets], int).reshape(-1, 4), masks=np.packbits(flat, axis=1))


def _unpack(path: Path, label: str) -> list[dict]:
    z = np.load(path)
    h, w = int(z["h"]), int(z["w"])
    masks = np.unpackbits(z["masks"], axis=1)[:, : h * w].reshape(-1, h, w).astype(bool)
    return [{"label": label, "score": float(s), "box_xyxy": b.tolist(), "mask": m}
            for s, b, m in zip(z["scores"], z["boxes"], masks)]


def detect(scene: Path, prompts=PROMPTS, threshold: float = THRESHOLD, device: str = DEVICE, cache: bool = True) -> list[dict]:
    """Every instance SAM 3 finds for every prompt: [{"label", "score", "box_xyxy", "mask"}], mask = bool [H, W].
    Cached per photo, prompt and threshold, so the scene graph can be rebuilt without the model."""
    scene = Path(scene)
    photo = scene / "rgb.png"
    image, out = None, []
    for prompt in prompts:
        key = hashlib.sha1(json.dumps([prompt, threshold, photo.stat().st_size]).encode()).hexdigest()[:10]
        path = CACHE / f"{scene.name}_{prompt.replace(' ', '-')}_{key}.npz"
        if cache and path.exists():
            out += _unpack(path, prompt)
            continue
        model, processor = load(device)
        image = image or Image.open(photo).convert("RGB")
        t1 = time.time()
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist())[0]
        masks = res["masks"].cpu().numpy().astype(bool)
        print(f"  {scene.name} {prompt!r}: {len(masks)} found in {time.time() - t1:.1f}s")
        dets = [{"label": prompt, "score": float(score), "box_xyxy": [round(x) for x in box], "mask": mask}
                for mask, box, score in zip(masks, res["boxes"].cpu().tolist(), res["scores"].cpu().tolist()) if mask.any()]
        if cache:
            _pack(path, dets, (image.height, image.width))
        out += dets
    return out


def photo_key(photo: Path) -> str:
    """A short fingerprint of the photo's content. Caches use it, so two scenes with the same folder name never mix."""
    return hashlib.sha1(Path(photo).read_bytes()).hexdigest()[:10]


def interior_pixel(mask: np.ndarray) -> list[int]:
    """The pixel deepest inside the mask: safer than the box centre for a banana or a mug handle."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v, u = np.unravel_index(int(dist.argmax()), dist.shape)
    return [int(u), int(v)]


def draw(canvas: np.ndarray, mask: np.ndarray, pixel, text: str, colour) -> None:
    """Tint the mask and label it at its pixel, in place. canvas is BGR."""
    canvas[mask] = (0.5 * canvas[mask] + 0.5 * np.array(colour)).astype(np.uint8)
    cv2.circle(canvas, tuple(pixel), 12, (255, 255, 255), -1)
    cv2.putText(canvas, text, (pixel[0] + 18, pixel[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 9)
    cv2.putText(canvas, text, (pixel[0] + 18, pixel[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="scene folder with rgb.png")
    ap.add_argument("--prompts", nargs="+", default=PROMPTS)
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="minimum detection score")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    dets = detect(args.scene, args.prompts, args.threshold, args.device, cache=not args.no_cache)
    canvas = cv2.imread(str(args.scene / "rgb.png"), cv2.IMREAD_COLOR)
    rng = np.random.default_rng(0)
    rows = []
    for d in dets:
        pixel = interior_pixel(d["mask"])
        rows.append({"label": d["label"], "score": round(d["score"], 3), "box_xyxy": d["box_xyxy"],
                     "pixel": pixel, "area_px": int(d["mask"].sum())})
        draw(canvas, d["mask"], pixel, f"{d['label']} {d['score']:.2f}", rng.integers(60, 256, 3).tolist())

    args.out.mkdir(exist_ok=True)
    name = args.scene.name
    cv2.imwrite(str(args.out / f"{name}_overlay.jpg"), canvas)
    (args.out / f"{name}_detections.json").write_text(json.dumps(rows, indent=2))
    print(f"{len(rows)} detections -> {args.out / f'{name}_detections.json'}, {args.out / f'{name}_overlay.jpg'}")


if __name__ == "__main__":
    main()
