# See It, Say It, Pick It

Our entry for the Hack Apertus 2026 ZHAW challenge: one photo of a robot table and one English sentence go
in, a safe robot action comes out (a plan, a question, or a refusal). Scoring rules: [EVALUATION.md](EVALUATION.md).

- **Path A** (our submission): SAM 3 finds the objects, code builds a scene graph, and three Apertus 1.5
  (70B) agents decide to plan, ask or refuse. Every tuned value is in
  [`workcell.ini`](pipelines/path_a/workcell.ini). Details: [ARCHITECTURE.md](pipelines/path_a/ARCHITECTURE.md),
  [MULTI_AGENT.md](pipelines/path_a/MULTI_AGENT.md).
- **Path B**: Apertus finds the objects itself, OpenCV measures them, and one Apertus call plans. Details:
  [ARCHITECTURE.md](pipelines/path_b/ARCHITECTURE.md).

## Results on the public dev set (10 scenes, 50 instructions)

Scored with the organisers' `starter/evaluate.py`.

| Pipeline | Grounding (35) | Safety (30) | Script total (65) | Runs |
|---|---|---|---|---|
| Path A, three agents | 34.1 | 30.0 | **63.0** (range 62.5–64.1) | 9 |
| Path B | 31.5 | 24.3 | **54.0** (range 52.7–55.8) | 4 |
| Organisers' baseline | | | 21.7 | 1 |

Grounding and safety are from each pipeline's best run. Path A's runs differ only on scene 09 i4, where the
endpoint sometimes asks instead of planning. Both pipelines were tuned on these scenes, so held-out scores
will be lower.

## How to run

### 1. What you need

- **An Apertus key.** Both pipelines call the hosted Apertus 1.5 (70B) endpoint.
- **For Path A, a machine that can run SAM 3** (3.4 GB, downloaded on the first run):
  - Apple-silicon Mac: what we used; about 70 s a scene.
  - NVIDIA GPU: used automatically (CUDA).
  - CPU only: works, but much slower.
- **Path B needs no GPU** and no downloads: about 20 s a scene.

### 2. Install: pick your case

Run these from the repository root.

**A. macOS or Linux with [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended).**
uv installs Python 3.12 itself, and `make` is already there.

```
make setup
```

**B. uv but no `make` (for example Windows)**

```
uv sync --frozen
cp .env.example .env          # Windows: copy .env.example .env
```

**C. No uv:** use Python 3.12.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
cp .env.example .env
```

Then, in every case, put your key in `.env` as `APERTUS_API_KEY`. The endpoint and model are already set
there. No Hugging Face token is needed: SAM 3 downloads from
[`jetjodh/sam3`](https://huggingface.co/jetjodh/sam3), an ungated copy byte-identical to `facebook/sam3`.

Check the setup before a long run:

| A (make) | B (uv, no make) | C (no uv) |
|---|---|---|
| `make check` | `uv run python scripts/check_setup.py` | `.venv/bin/python scripts/check_setup.py` |

### 3. Run: pick a pipeline and a task

There are two pipelines (see the top of this page):

- **Path A** is our submission: SAM 3 plus three Apertus agents.
- **Path B** is our second approach: Apertus finds the objects itself.

**Naming convention:** every command names its task, then the pipeline: `-a` for Path A, `-b` for
Path B. For example, `make heldout-a` runs Path A on the held-out scenes. `make run` is short for
`make run-a`, our submission.

| Task | Path A (our submission) | Path B |
|---|---|---|
| The held-out scenes (`heldout/`) | `make heldout-a` | `make heldout-b` |
| New photos in [`scenes/`](scenes/README.md) | `make run` or `make run-a` | `make run-b` |
| Any other folder of scenes | `make run-a DATA=path/to/folder` | `make run-b DATA=path/to/folder` |
| The public scenes, then the score | `make dev-a` | `make dev-b` |
| Offline tests (no key, no downloads) | `make test-a` | `make test-b` |

**Where the answers go:** a folder's answers go to `results/<folder>_path_a.json` or
`results/<folder>_path_b.json`, and the public scenes' to `results/path_a_dev.json` or
`results/path_b_dev.json`. The held-out answers go to `results_heldout.json` (Path A, the file we
submit) or `results_heldout_path_b.json`.

The rest of the commands are for both pipelines:

| Command | What it does |
|---|---|
| `make check` | Checks the key, the endpoint and `heldout/` (add `DATA=...` for another folder) |
| `make app` | Starts the Pick Lab app on http://localhost:3000 (needs Node.js) |
| `make clean` | Deletes the cached scene graphs |
| `make help` | Lists every command |

**Without `make` (cases B and C):** each `make` command runs one pipeline script, so you can run that
script yourself. Write `uv run python` (case B) or `.venv/bin/python` (case C) in place of `python`:

| Task | Path A | Path B |
|---|---|---|
| The held-out scenes | `python pipelines/path_a/run.py --data heldout` | `python pipelines/path_b/run.py --data heldout` |
| Any folder of scenes | `python pipelines/path_a/run.py --data scenes --out results/scenes_path_a.json` | `python pipelines/path_b/run.py --data scenes --out results/scenes_path_b.json` |
| The public scenes, then the score | `python pipelines/path_a/run.py --data dataset` | `python pipelines/path_b/run.py --data dataset` |
| Offline tests | `python -m unittest discover -s pipelines/path_a -p 'test_*.py'` | `python -m unittest discover -s pipelines/path_b/tests` |

`run.py --help` lists each pipeline's options.

### Good to know

- **New scenes** use the same layout as `heldout/`: one folder per scene with `rgb.png` and
  `instructions.txt`, and `calibration.json` next to them. See [scenes/README.md](scenes/README.md). A
  `ground_truth/` folder next to the scenes, as in `dataset/`, makes the run print its score.
- **Path A caches each photo's scene graph** in `pipelines/path_a/out/`, so a second run on the same photos
  only calls Apertus. `make clean` deletes the cache.
- **Scores vary slightly from run to run** because the endpoint is not fully deterministic (see the
  ranges above).
- **The Pick Lab app** replays saved results with an animated 2D arm and the target pixels on the photo,
  and can run either pipeline live. Replay makes no Apertus calls. See [apps/README.md](apps/README.md).

## Repository

| Path | What it is |
|---|---|
| `pipelines/path_a/`, `pipelines/path_b/` | One folder per solution, each with its own `run.py` and architecture notes |
| `scenes/` | Where new scenes go for `make run` |
| `results/` | One results file and log per pipeline |
| `apps/`, `simulation/` | The Pick Lab app |
| `docs/` | The organisers' notes (`CHALLENGE.md`, `DATA.md`, `safety_primer.md`) and ours on Apertus's limits |
| `dataset/`, `heldout/`, `starter/`, `schema/`, `baseline/`, `examples/` | The organisers' files, unchanged |

## Licence

Code: Apache 2.0. Data in `dataset/`: CC BY 4.0. SAM 3 is Meta's model, used under the SAM
License, which permits redistribution; the `jetjodh/sam3` copy includes that licence.
