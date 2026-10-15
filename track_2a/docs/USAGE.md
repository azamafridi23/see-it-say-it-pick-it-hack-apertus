# See It, Say It, Pick It: how to run

Our entry for the Hack Apertus 2026 ZHAW challenge: one photo of a robot table and one English sentence go
in, a safe robot action comes out (a plan, a question, or a refusal). Scoring rules:
[EVALUATION.md](EVALUATION.md). The write-up is in [technical_report.md](../technical_report.md).

- **Path A** (our submission): SAM 3 finds the objects, code builds a scene graph, and three Apertus 1.5
  (70B) agents decide to plan, ask or refuse. Every tuned value is in
  [`workcell.ini`](../src/pipelines/path_a/workcell.ini). Details:
  [ARCHITECTURE.md](../src/pipelines/path_a/ARCHITECTURE.md),
  [MULTI_AGENT.md](../src/pipelines/path_a/MULTI_AGENT.md).
- **Path B**: Apertus finds the objects itself, OpenCV measures them, and one Apertus call plans. Details:
  [ARCHITECTURE.md](../src/pipelines/path_b/ARCHITECTURE.md).

All commands below run from `track_2a/`, the project root.

## 1. With Docker (what the judges run)

You need Docker and an Apertus key. Put the key in the environment or in a `.env` file:

```
cp .env.example .env          # then fill in LLM_API_KEY
make run
```

`make run` builds the image (the packages and the model weights are inside it, so the first build takes
a while) and runs Path A on the five held-out scenes in `data/heldout/`. The answers go to
`results/results_heldout.json`.

| What | Command |
|---|---|
| The held-out scenes | `make run` |
| Any folder of scenes | `make run DATA=path/to/folder` → `results/<folder>_path_a.json` |
| Only build the image | `make build` |

The settings follow the Hack Apertus template: `LLM_API_KEY` (the key), `LLM_BASE_URL` (the endpoint)
and `LLM_NAME` (the model). Only the key is required; the endpoint and the model default to the hosted
hackathon endpoint and `apertus-v1.5-70b`. Variables set in the shell take precedence over `.env`.

The image runs SAM 3 on the CPU: it uses PyTorch's CPU-only build, which keeps the image small and
the build fast, and a full held-out run still fits in about 10 minutes. Give Docker Desktop at least
8 GB of memory (Settings › Resources).

## 2. Without Docker

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python 3.12
itself. Without uv, use Python 3.12 and `pip install -r requirements.txt`.

```
make setup        # installs the packages and creates .env
make check        # checks the key, the endpoint and the scenes
```

No Hugging Face token is needed: SAM 3 downloads from [`jetjodh/sam3`](https://huggingface.co/jetjodh/sam3),
an ungated copy byte-identical to `facebook/sam3`, on the first run (3.4 GB). On an Apple-silicon Mac
or an NVIDIA GPU, SAM 3 uses the GPU automatically.

Every command names its task, then the pipeline: `-a` for Path A, `-b` for Path B.

| Task | Path A (our submission) | Path B |
|---|---|---|
| The held-out scenes (`data/heldout/`) | `make heldout-a` | `make heldout-b` |
| Any folder of scenes | `make run-a DATA=path/to/folder` | `make run-b DATA=path/to/folder` |
| The public scenes, then the score | `make dev-a` | `make dev-b` |
| Offline tests (no key, no downloads) | `make test-a` | `make test-b` |

Without `make`, run the script each target names, for example
`uv run python src/pipelines/path_a/run.py --data data/heldout`. `run.py --help` lists the options.

## 3. See the plans: the Pick Lab app (optional)

Watch a plan carried out by a simulated arm: an animated 2D arm, a top view, a 3D view, and the target
pixel on the photo, for every public scene. It runs on your machine, not in Docker, and needs Node.js 20.9+.

```
make app          # then open http://localhost:3000
```

- **Replay** (the default) shows our saved answers for the public scenes: pick a scene and an
  instruction, click **Load saved plan**, then **Play**. No key, no Apertus calls, no SAM 3. It replays
  `results/path_a_dev.json` and `results/path_b_dev.json`, the files `make dev-a` and `make dev-b` write.
- **Live** runs Path A or Path B on any instruction you type, with the key in `.env`.

Stop it with Ctrl+C. More: [src/apps/README.md](../src/apps/README.md).

## Good to know

- **New scenes** use the same layout as `data/heldout/`: one folder per scene with `rgb.png` and
  `instructions.txt`, and `calibration.json` next to them. See
  [data/scenes/README.md](../data/scenes/README.md). A `ground_truth/` folder next to the scenes, as in
  `data/dataset/`, makes the run print its score.
- **Path A caches each photo's scene graph** in `src/pipelines/path_a/out/`, so a second run on the same
  photos only calls Apertus. `make clean` deletes the cache.
- **Scores vary slightly from run to run** because the endpoint is not fully deterministic.

## Layout

| Path | What it is |
|---|---|
| `README.md`, `technical_report.md` | The track description and our write-up |
| `Makefile`, `Dockerfile` | `make run` builds and runs the container |
| `src/pipelines/path_a/`, `src/pipelines/path_b/` | One folder per solution, each with its own `run.py` |
| `src/apps/`, `src/simulation/` | The Pick Lab app |
| `src/starter/`, `src/schema/`, `src/baseline/`, `src/examples/` | The organisers' files, unchanged |
| `data/dataset/`, `data/heldout/` | The organisers' public and held-out scenes |
| `data/scenes/` | Where new scenes go |
| `results/` | Results files and logs |
| `docs/` | The organisers' notes, this guide, and the Path B failure report |

## Licence

Code: Apache 2.0. Data in `data/dataset/`: CC BY 4.0. SAM 3 is Meta's model, used under the SAM License,
which permits redistribution; the `jetjodh/sam3` copy includes that licence.
