#!/usr/bin/env python3
"""Find every object in a photo with SAM 3, one text prompt per object type.

    python pipelines/path_a/run_sam3.py dataset/scene_06
    python pipelines/path_a/run_sam3.py dataset/scene_06 \
        --prompts "red gear" "yellow gear" --threshold 0.4

Downloads Meta's SAM 3 weights on first use, without a Hugging Face token:
facebook/sam3 is gated, so the default is an ungated copy with
byte-identical weights. SAM3_MODEL=facebook/sam3 (with HF_TOKEN) loads the
official repository.

Writes pipelines/path_a/out/<scene>_overlay.jpg (masks drawn on the photo,
to check by eye) and out/<scene>_detections.json (label, score, box, an
interior pixel, mask area). Pixels are [u, v] in the full-resolution
photo, as in the challenge's target_pixel.

These are the raw detections. scene_graph.py filters and merges them into
a scene graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Ops that Apple's MPS lacks run on the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dotenv import load_dotenv  # noqa: E402

# SAM3_MODEL / HF_TOKEN from the nearest .env, searching upwards from this
# file; before transformers is imported.
load_dotenv()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import Sam3Model, Sam3Processor  # noqa: E402

from workcell import CELL  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = Path(__file__).resolve().parent / "out" / "cache"
# The pinned ungated copy from workcell.ini, unless SAM3_MODEL names
# another repository (facebook/sam3 needs HF_TOKEN).
SAM3_MODEL = os.environ.get("SAM3_MODEL") or CELL.sam3_model
SAM3_REVISION = (CELL.sam3_revision if SAM3_MODEL == CELL.sam3_model
                 else None)

_loaded = {}


def load(device: str = DEVICE):
    """Load SAM 3 once per device.

    Args:
        device: ``"mps"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        The model and its processor.
    """
    if device not in _loaded:
        # Only needed for the gated facebook/sam3.
        token = os.environ.get("HF_TOKEN") or None
        started = time.time()
        model = Sam3Model.from_pretrained(SAM3_MODEL, revision=SAM3_REVISION,
                                          token=token)
        model = model.to(device).eval()
        processor = Sam3Processor.from_pretrained(
            SAM3_MODEL, revision=SAM3_REVISION, token=token)
        print(f"model loaded on {device} in {time.time() - started:.0f}s")
        _loaded[device] = (model, processor)

    return _loaded[device]


def _cache_path(scene: Path, prompt: str, threshold: float) -> Path:
    """Name the cache file for one scene and prompt."""
    photo = scene / "rgb.png"
    settings = json.dumps([prompt, threshold, photo.stat().st_size])
    key = hashlib.sha1(settings.encode()).hexdigest()[:10]
    return CACHE / f"{scene.name}_{prompt.replace(' ', '-')}_{key}.npz"


def _pack(path: Path, dets: list[dict], shape) -> None:
    """Save detections compactly: masks as packed bits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = shape
    if dets:
        flat = np.stack([det["mask"].reshape(-1) for det in dets])
    else:
        flat = np.zeros((0, height * width), bool)

    scores = np.array([det["score"] for det in dets], float)
    boxes = np.array([det["box_xyxy"] for det in dets], int).reshape(-1, 4)
    np.savez_compressed(path, h=height, w=width, scores=scores, boxes=boxes,
                        masks=np.packbits(flat, axis=1))


def _unpack(path: Path, label: str) -> list[dict]:
    """Load detections saved by ``_pack()``."""
    saved = np.load(path)
    height, width = int(saved["h"]), int(saved["w"])
    bits = np.unpackbits(saved["masks"], axis=1)[:, : height * width]
    masks = bits.reshape(-1, height, width).astype(bool)
    return [{"label": label, "score": float(score), "box_xyxy": box.tolist(),
             "mask": mask}
            for score, box, mask in zip(saved["scores"], saved["boxes"],
                                        masks)]


def _run_prompt(image, prompt: str, threshold: float,
                device: str) -> list[dict]:
    """Run SAM 3 on one photo for one prompt."""
    model, processor = load(device)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    sizes = inputs.get("original_sizes").tolist()
    result = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=CELL.sam3_mask_threshold,
        target_sizes=sizes)[0]
    masks = result["masks"].cpu().numpy().astype(bool)
    boxes = result["boxes"].cpu().tolist()
    scores = result["scores"].cpu().tolist()

    dets = []
    for mask, box, score in zip(masks, boxes, scores):
        if not mask.any():
            continue

        dets.append({"label": prompt, "score": float(score),
                     "box_xyxy": [round(value) for value in box],
                     "mask": mask})

    return dets


def detect(scene: Path, prompts=CELL.sam3_prompts,
           threshold: float = CELL.sam3_threshold, device: str = DEVICE,
           cache: bool = True) -> list[dict]:
    """Find every instance SAM 3 sees for every prompt.

    Cached per photo, prompt and threshold, so the scene graph can be
    rebuilt without the model.

    Args:
        scene: The scene folder, with rgb.png.
        prompts: The text prompts.
        threshold: The minimum detection score.
        device: Where to run SAM 3.
        cache: Whether to read and write the cache.

    Returns:
        One dict per instance: ``label``, ``score``, ``box_xyxy`` and
        ``mask`` (bool [H, W]).
    """
    scene = Path(scene)
    image, out = None, []
    for prompt in prompts:
        path = _cache_path(scene, prompt, threshold)
        if cache and path.exists():
            out += _unpack(path, prompt)
            continue

        image = image or Image.open(scene / "rgb.png").convert("RGB")
        started = time.time()
        dets = _run_prompt(image, prompt, threshold, device)
        print(f"  {scene.name} {prompt!r}: {len(dets)} found in "
              f"{time.time() - started:.1f}s")
        if cache:
            _pack(path, dets, (image.height, image.width))

        out += dets

    return out


def photo_key(photo: Path) -> str:
    """Fingerprint the photo's content.

    Caches use it, so two scenes with the same folder name never mix.

    Args:
        photo: The photo's path.

    Returns:
        The first 10 hex digits of the photo's SHA-1.
    """
    return hashlib.sha1(Path(photo).read_bytes()).hexdigest()[:10]


def interior_pixel(mask: np.ndarray) -> list[int]:
    """Find the pixel deepest inside a mask.

    Safer than the box centre for a banana or a mug handle.

    Args:
        mask: A bool mask.

    Returns:
        The pixel as [u, v].
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v, u = np.unravel_index(int(distance.argmax()), distance.shape)
    return [int(u), int(v)]


def draw(canvas: np.ndarray, mask: np.ndarray, pixel, text: str,
         colour) -> None:
    """Tint a mask and label it at its pixel, in place.

    Args:
        canvas: The BGR image to draw on.
        mask: The bool mask.
        pixel: Where the label goes, [u, v].
        text: The label.
        colour: The tint, BGR.
    """
    tinted = 0.5 * canvas[mask] + 0.5 * np.array(colour)
    canvas[mask] = tinted.astype(np.uint8)
    cv2.circle(canvas, tuple(pixel), 12, (255, 255, 255), -1)
    corner = (pixel[0] + 18, pixel[1] + 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, corner, font, 1.4, (0, 0, 0), 9)
    cv2.putText(canvas, text, corner, font, 1.4, (255, 255, 255), 3)


def main():
    """Run SAM 3 on one scene and save the overlay and the detections."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="scene folder with rgb.png")
    ap.add_argument("--prompts", nargs="+", default=CELL.sam3_prompts)
    ap.add_argument("--threshold", type=float, default=CELL.sam3_threshold,
                    help="minimum detection score")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    dets = detect(args.scene, args.prompts, args.threshold, args.device,
                  cache=not args.no_cache)
    canvas = cv2.imread(str(args.scene / "rgb.png"), cv2.IMREAD_COLOR)
    rng = np.random.default_rng(0)
    rows = []
    for det in dets:
        pixel = interior_pixel(det["mask"])
        rows.append({"label": det["label"], "score": round(det["score"], 3),
                     "box_xyxy": det["box_xyxy"], "pixel": pixel,
                     "area_px": int(det["mask"].sum())})
        label = f"{det['label']} {det['score']:.2f}"
        draw(canvas, det["mask"], pixel, label,
             rng.integers(60, 256, 3).tolist())

    args.out.mkdir(exist_ok=True)
    name = args.scene.name
    overlay = args.out / f"{name}_overlay.jpg"
    detections = args.out / f"{name}_detections.json"
    cv2.imwrite(str(overlay), canvas)
    detections.write_text(json.dumps(rows, indent=2))
    print(f"{len(rows)} detections -> {detections}, {overlay}")


if __name__ == "__main__":
    main()
