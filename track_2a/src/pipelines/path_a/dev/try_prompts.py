"""Compare SAM 3 prompt wordings on one scene (development).

For each prompt: the count, the scores, and which ground-truth object each
hit lands on.

    python pipelines/path_a/dev/try_prompts.py dataset/scene_06 \
        "red gear" "gear"
"""
import json
import sys
from pathlib import Path

# The pipeline, one folder up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from run_sam3 import DEVICE, interior_pixel, load  # noqa: E402

TOP = 7  # hits shown per prompt


def main():
    """Run each prompt and print what its hits land on."""
    scene, prompts = Path(sys.argv[1]), sys.argv[2:]
    model, processor = load(DEVICE)
    image = Image.open(scene / "rgb.png").convert("RGB")
    truth_folder = scene.parent / "ground_truth" / "masks"
    truth = cv2.imread(str(truth_folder / f"{scene.name}.png"),
                       cv2.IMREAD_UNCHANGED)
    table = json.loads((truth_folder / f"{scene.name}.json").read_text())
    names = {int(key): name for key, name in table["objects"].items()}

    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        sizes = inputs.get("original_sizes").tolist()
        result = processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5,
            target_sizes=sizes)[0]
        masks = result["masks"].cpu().numpy().astype(bool)
        scores = result["scores"].cpu().tolist()

        hits = []
        for mask, score in zip(masks, scores):
            if not mask.any():
                continue

            u, v = interior_pixel(mask)
            landed_on = names.get(int(truth[v, u]), "-")
            hits.append((round(score, 2), landed_on, int(mask.sum())))

        hits.sort(reverse=True)
        shown = "  ".join(f"{score}:{name}({area // 1000}k)"
                          for score, name, area in hits[:TOP])
        print(f"{prompt!r}: {len(hits)}  {shown}")


if __name__ == "__main__":
    main()
