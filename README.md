# See It, Say It, Pick It

Our entry for the Hack Apertus 2026 ZHAW challenge: one photo of a robot table and one English
sentence go in, a safe robot action comes out (a plan, a question, or a refusal).

The organisers' original brief is in [docs/CHALLENGE.md](docs/CHALLENGE.md), and the scoring rules
are in [EVALUATION.md](EVALUATION.md).

## Results on the 10 public scenes

| Pipeline | Approach | Grounding (35) | Safety (30) | Script total (65) |
|---|---|---|---|---|
| `path_a_azam`, multi-agent | SAM 3 + scene graph + three Apertus 1.5 (70B) agents | 34.1 | 30.0 | **64.1** |
| `path_a_azam`, single call | SAM 3 + scene graph + one Apertus 1.5 (70B) call | 33.2 | 25.6 | **58.9** |
| `path_a_randy` | to be added | | | |
| `path_b` | Apertus boxes + OpenCV measurements + `path_a_azam`'s single-call planner, Apertus 1.5 (70B) | 31.5 | 24.3 | **55.8** |
| `path_b_wahyu` | Apertus alone, to be added | | | |

The multi-agent version gave 64.1 on three runs out of three; the single call varies between 58.9 and 60.5.
The public answers are public, and the multi-agent prompts were tuned on them, so the held-out scenes are the
real test.
`path_b` gave 55.8 and 54.2 on two fresh runs; its box correction, stacking thresholds and part sizes were
fitted on the public scenes too.

### Every experiment, by branch

All with Apertus 1.5 70B on the ten public scenes. Each score is one full run of the 50 instructions.

| Branch | What it tried | Scores |
|---|---|---|
| `path-a/azam` (merged into `main`) | First pipeline: one Apertus call plans everything | 60.5, 60.5 |
| `path-a/azam-robustness` | Same planner, plus retry on a busy server, caches keyed by photo content, and a worded reason for unpickable objects | 58.9, 58.9 |
| `path-a/azam-depth-size` | Pickable decided by measured size, with heights from a depth model. Dropped: the organisers confirmed pickable is a property of the object, not a size rule | 52.2, 56.5 |
| `path-a/azam-multi-agent` | Three Apertus agents (safety, reference, constraint) in a fixed LangGraph graph, with the moves ordered by code | 64.1, 64.1 |
| `path-a/azam-multi-agent-prompt-validation` | Reply validation, tests and a replay tool, with new v3 prompts and a policy of asking about uncertain stacking | 53.4, 53.4 |
| `path-a/azam-multi-agent-combined` | **Current.** The validation, tests and replay tool, with the v2 prompts and the one-question constraint agent | **64.1, 64.1, 64.1** |

Smaller comparisons, run on the combined branch:

| Variation | Scores |
|---|---|
| The v3 safety prompt in place of v2 | 62.5, 62.5, 62.5 |
| The v3 reference prompt in place of v2 | 59.6, 59.6 |
| Single call with the `planner_v2` prompt in place of `planner_v1` | 58.9 |
| Single call with a "why not pickable" field added to the scene description | 52.8, 52.8, 51.4 |
| Multi-agent before two redesigns: with a planning agent, then with a constraint agent judging the whole plan | 45.0, 40.3; then 54.9, 54.9 |

What these runs taught us is written up in
[MULTI_AGENT.md](pipelines/path_a_azam/MULTI_AGENT.md) and [ARCHITECTURE.md](pipelines/path_a_azam/ARCHITECTURE.md).
The `path-a/azam-robustness`, `path-a/azam-depth-size` and `path-a/azam-multi-agent-combined` branches may exist
only on the author's machine until pushed.

## How `path_a_azam` works

1. **SAM 3** finds every object in the photo and its outline.
2. **Apertus** lists what it sees, so objects outside the fixed prompt list are found too.
3. **Code** builds a scene graph: each object's position, size, and what lies on what.
4. **Apertus** reads the scene graph and the instruction and decides: plan, ask, or refuse.
5. **Code** checks the answer against the scene graph.

Step 4 comes in two versions:

- **Single call** (the default): one Apertus call decides everything, and is asked to correct itself if the
  check fails.
- **Multi-agent** (`--planner graph`): three Apertus agents in a fixed LangGraph graph. A safety agent judges
  the movement, a reference agent works out which objects are meant, code orders the moves, and a constraint
  agent checks whether the instruction forbids any of them.

Apertus chooses the object; the pixel always comes from SAM 3. Details:
[ARCHITECTURE.md](pipelines/path_a_azam/ARCHITECTURE.md) for the pipeline and
[MULTI_AGENT.md](pipelines/path_a_azam/MULTI_AGENT.md) for the multi-agent planner.

## How `path_b` works

Apertus is the only learned model and the main detector; OpenCV only measures.

1. **Apertus** finds every object on the whole photo in one call: a kind and a bounding box each.
2. **Code** corrects the boxes (they sit about 10% too high) and takes each box centre as the
   object's pixel.
3. **OpenCV** only measures: sizes, gaps, which tray each object is in, which object lies on top
   where two boxes overlap, and which kind of part a part is, by its size.
4. **Code** builds a scene graph, like `path_a_azam`'s.
5. **Apertus** checks the sentence alone and refuses unsafe or self-contradicting instructions.
6. **Apertus** reads the scene graph and the instruction and decides: plan, ask, or refuse, using
   `path_a_azam`'s single-call planner.
7. **Code** checks the answer against the scene graph and asks Apertus to correct it if needed.

It needs Python 3.10+ with `numpy` and `opencv-python`; no GPU, and no SAM 3. The key is `API_KEY` in
`.env`, as for `path_a_azam`. Every value particular to this camera, table and object set is in
[`workcell.ini`](pipelines/path_b/workcell.ini).

```
.venv/bin/pip install -r pipelines/path_b/requirements.txt
.venv/bin/python pipelines/path_b/run.py --data dataset --out results/path_b.json \
    --base-url https://hackapertus.livemap.sh/v1 --model apertus-v1.5-70b
.venv/bin/python starter/evaluate.py results/path_b.json --data dataset
.venv/bin/python -m unittest discover pipelines/path_b/tests   # offline, no API key needed
```

Details: [pipelines/path_b/ARCHITECTURE.md](pipelines/path_b/ARCHITECTURE.md).

## Setup

Python 3.12. From the repository root:

```
python -m venv .venv
.venv/bin/pip install -r pipelines/path_a_azam/requirements.txt
cp .env.example .env
```

Then fill in `.env`:

- `HF_TOKEN`: a Hugging Face token. SAM 3 is a gated model, so request access at
  https://huggingface.co/facebook/sam3 first.
- `API_KEY`: the key for your Apertus server.

## Run

```
.venv/bin/python pipelines/path_a_azam/run.py --data dataset --out results/path_a_azam.json
.venv/bin/python starter/evaluate.py results/path_a_azam.json --data dataset
```

For the multi-agent planner, add `--planner graph`:

```
.venv/bin/python pipelines/path_a_azam/run.py --data dataset --out results/path_a_azam_graph.json --planner graph
.venv/bin/python starter/evaluate.py results/path_a_azam_graph.json --data dataset
```

`--data` takes any folder of scenes, so the held-out run is the same command with `--data heldout`.

SAM 3 and the depth model run on your own machine (Apple GPU, CUDA or CPU). Apertus is reached
through any OpenAI-compatible server: set it with `--base-url` and `--model`.

The rules given to Apertus are text files in `pipelines/path_a_azam/prompts/`. `--prompt planner_v2`
picks a version; the default is `planner_v1`.

The run also writes `results/path_a_azam.log.jsonl`, with every Apertus reply and correction.

## Repository layout

| Path | What it is |
|---|---|
| `pipelines/` | One folder per solution, each with its own `run.py` and `requirements.txt` |
| `results/` | One results file and log per pipeline |
| `dataset/`, `starter/`, `schema/`, `baseline/`, `examples/`, `docs/` | The organisers' files, unchanged |

## Licence

Code: Apache 2.0. Data in `dataset/`: CC BY 4.0. SAM 3 is used under Meta's SAM License.
