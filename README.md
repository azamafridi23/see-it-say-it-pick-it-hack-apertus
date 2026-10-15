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

### Setup (both paths)

Use Python 3.12. From the repository root:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Put your Apertus key in `.env` as `APERTUS_API_KEY`. `.env` already points to the hosted endpoint
(`API_BASE_URL=https://hackapertus.livemap.sh/v1`, `API_MODEL=apertus-v1.5-70b`).

### Path A

```
.venv/bin/python pipelines/path_a/run.py
```

It runs the three-agent planner on the held-out scenes and writes `results_heldout.json`, the file the
organisers ask for. To score it on the public scenes instead, add `--data dataset`: it then writes
`results/path_a_dev.json` and prints the score.

No Hugging Face token is needed: on the first run, Meta's SAM 3 weights (3.4 GB) download from
[`jetjodh/sam3`](https://huggingface.co/jetjodh/sam3), an ungated copy whose weights are byte-identical to
`facebook/sam3`. A run takes about 12 minutes for 10 new scenes on an Apple-silicon
Mac, plus the download on the first run.

### Path B

```
.venv/bin/python pipelines/path_b/run.py
```

### Local robot action app

The Next.js + Python **Pick Lab** playground lives in `apps/`. It includes saved-result replay, an animated 2D arm, a photo view with target pixels, and explicit live inference for Path A and Path B. See [apps/README.md](apps/README.md) for setup and limitations. Start both services from this repository's root with `npm run dev` after installing the app dependencies. Replay makes no Apertus API calls.

| Path | What it is |
|---|---|
| `pipelines/path_a/`, `pipelines/path_b/` | One folder per solution, each with its own `run.py`, `requirements.txt` and architecture notes |
| `results/` | One results file and log per pipeline |
| `docs/` | The organisers' notes (`CHALLENGE.md`, `DATA.md`, `safety_primer.md`) and ours on Apertus's limits |
| `dataset/`, `starter/`, `schema/`, `baseline/`, `examples/` | The organisers' files, unchanged |
It runs on the held-out scenes and writes `results_heldout_path_b.json`. With `--data dataset` it writes
`results/path_b_dev.json` and prints the score. It needs no Hugging Face token and no local model, and takes
about 2.5 minutes for 10 scenes.

## Licence

Code: Apache 2.0. Data in `dataset/`: CC BY 4.0. SAM 3 is Meta's model, used under the SAM
License, which permits redistribution; the `jetjodh/sam3` copy includes that licence.
