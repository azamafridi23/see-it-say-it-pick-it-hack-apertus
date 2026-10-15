#!/usr/bin/env python3
"""Download the open-weight models Path A runs locally.

    python src/scripts/download_models.py

SAM 3 (pinned to the commit in workcell.ini) and the depth model go into
the Hugging Face cache, so a run needs no download. The Docker build runs
this once; on a machine without Docker it is optional, because the first
run downloads them too.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"
                       / "path_a"))

from workcell import CELL  # noqa: E402


def main():
    """Download SAM 3 and the depth model."""
    # Only what transformers loads: the configs, the tokenizer and the
    # safetensors weights (the repositories hold other formats too).
    files = ["*.json", "*.txt", "*.safetensors"]
    snapshot_download(CELL.sam3_model, revision=CELL.sam3_revision,
                      allow_patterns=files)
    snapshot_download(CELL.depth_model, allow_patterns=files)
    print(f"downloaded {CELL.sam3_model} and {CELL.depth_model}")


if __name__ == "__main__":
    main()
