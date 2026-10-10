"""Compare prompt wordings on one scene: count, scores and which ground-truth object each hit lands on."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the pipeline, one folder up
from dotenv import load_dotenv
load_dotenv()
import os, cv2, numpy as np, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from run_sam3 import interior_pixel

scene, prompts = Path(sys.argv[1]), sys.argv[2:]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = os.environ["HF_TOKEN"]
model = Sam3Model.from_pretrained("facebook/sam3", token=tok).to(dev).eval()
proc = Sam3Processor.from_pretrained("facebook/sam3", token=tok)
image = Image.open(scene / "rgb.png").convert("RGB")
gtd = scene.parent / "ground_truth" / "masks"
gt = cv2.imread(str(gtd / f"{scene.name}.png"), cv2.IMREAD_UNCHANGED)
names = {int(k): v for k, v in json.loads((gtd / f"{scene.name}.json").read_text())["objects"].items()}
for p in prompts:
    inp = proc(images=image, text=p, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inp)
    r = proc.post_process_instance_segmentation(out, threshold=0.5, mask_threshold=0.5,
                                                target_sizes=inp.get("original_sizes").tolist())[0]
    hits = []
    for m, s in zip(r["masks"].cpu().numpy().astype(bool), r["scores"].cpu().tolist()):
        if m.any():
            u, v = interior_pixel(m)
            hits.append((round(s, 2), names.get(int(gt[v, u]), "-"), int(m.sum())))
    hits.sort(reverse=True)
    print(f"{p!r}: {len(hits)}  " + "  ".join(f"{s}:{n}({a//1000}k)" for s, n, a in hits[:7]))
