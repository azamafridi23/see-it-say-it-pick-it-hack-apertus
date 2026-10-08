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
| `path_b_wahyu` | Apertus alone, to be added | | | |

The single call's score varies between runs (58.9 to 60.5); the multi-agent version gave 64.1 on both of its
runs. The public answers are public, and the multi-agent prompts were tuned on them, so the held-out scenes are
the real test.

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
