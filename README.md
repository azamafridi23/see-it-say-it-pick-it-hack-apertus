# See It, Say It, Pick It

Our entry for the Hack Apertus 2026 ZHAW challenge: one photo of a robot table and one English
sentence go in, a safe robot action comes out (a plan, a question, or a refusal).

The organisers' original brief is in [docs/CHALLENGE.md](docs/CHALLENGE.md), and the scoring rules
are in [EVALUATION.md](EVALUATION.md).

## Results on the 10 public scenes

| Pipeline | Approach | Grounding (35) | Safety (30) | Script total (65) |
|---|---|---|---|---|
| `path_a`, multi-agent | SAM 3 + scene graph + three Apertus 1.5 (70B) agents | 34.1 | 30.0 | **64.1** |
| `path_a`, single call | SAM 3 + scene graph + one Apertus 1.5 (70B) call | 33.2 | 25.6 | **58.9** |
| `path_b` | Apertus boxes + OpenCV measurements + `path_a`'s single-call planner, Apertus 1.5 (70B) | 31.5 | 24.3 | **55.8** |
| `path_b_wahyu` | Apertus alone, to be added | | | |

The multi-agent version gave 64.1 on three runs out of three (2026-10-08) and 62.5 on two fresh runs on
2026-10-10, with identical scene graphs and prompts: the endpoint now asks the user on scene 09 i4 instead of
planning. The single call varies between 58.9 and 60.5.
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
[MULTI_AGENT.md](pipelines/path_a/MULTI_AGENT.md) and [ARCHITECTURE.md](pipelines/path_a/ARCHITECTURE.md).
The `path-a/azam-robustness`, `path-a/azam-depth-size` and `path-a/azam-multi-agent-combined` branches may exist
only on the author's machine until pushed.

## How `path_a` works

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
[ARCHITECTURE.md](pipelines/path_a/ARCHITECTURE.md) for the pipeline and
[MULTI_AGENT.md](pipelines/path_a/MULTI_AGENT.md) for the multi-agent planner.

## How `path_b` works

Apertus is the only learned model and the main detector; OpenCV only measures.

1. **Apertus** finds every object on the whole photo in one call: a kind and a bounding box each.
2. **Code** corrects the boxes (they sit about 10% too high) and takes each box centre as the
   object's pixel.
3. **OpenCV** only measures: sizes, gaps, which tray each object is in, which object lies on top
   where two boxes overlap, and which kind of part a part is, by its size.
4. **Code** builds a scene graph, like `path_a`'s.
5. **Apertus** checks the sentence alone and refuses unsafe or self-contradicting instructions.
6. **Apertus** reads the scene graph and the instruction and decides: plan, ask, or refuse, using
   `path_a`'s single-call planner.
7. **Code** checks the answer against the scene graph and asks Apertus to correct it if needed.

Every value particular to this camera, table and object set is in
[`workcell.ini`](pipelines/path_b/workcell.ini). Details:
[pipelines/path_b/ARCHITECTURE.md](pipelines/path_b/ARCHITECTURE.md).

## Running the pipelines (guide for the jury)

Each pipeline reads one folder of scenes and writes one results file in the challenge's format, scored by
the organisers' `starter/evaluate.py`. The held-out run is the same command with `--data heldout`.

| | `path_a` | `path_b` |
|---|---|---|
| Vision model | SAM 3 + a depth model, on your machine (Apple GPU, CUDA or CPU) | none: Apertus finds the objects |
| Hugging Face token | yes: SAM 3 is a gated model | no |
| Apertus server | any OpenAI-compatible server; default: the hosted endpoint | any OpenAI-compatible server; default: a local one |
| Offline tests | `test_agents.py` | `tests/`, no key needed |

### Before you start (both pipelines)

Use Python > 3.12. From the repository root:

```
python3 -m venv .venv
cp .env.example .env
```

Fill in `APERTUS_API_KEY` in `.env`: the key for your Apertus server. Never commit `.env`.

Apertus runs on any OpenAI-compatible server. `.env` names it in `API_BASE_URL` and `API_MODEL`, already set
to the hosted endpoint; `--base-url` and `--model` override them for one run:

- **The hosted endpoint** (the default in `.env.example`): `https://hackapertus.livemap.sh/v1`, model
  `apertus-v1.5-70b`.
- **Your own machine,** with vLLM. The 8B model needs one GPU; the 70B needs four
  (`--tensor-parallel-size 4`):

  ```
  vllm serve swiss-ai/Apertus-v1.5-8B --chat-template-content-format string
  ```

  then set `API_BASE_URL=http://localhost:8000/v1` and `API_MODEL=swiss-ai/Apertus-v1.5-8B` in `.env`. Our scores were measured with
  the 70B model.

### Path A (our submission: the multi-agent planner)

1. Install its dependencies, and add `HF_TOKEN` to `.env`. SAM 3 is a gated model, so request access at
   https://huggingface.co/facebook/sam3 first.

   ```
   .venv/bin/pip install -r pipelines/path_a/requirements.txt
   ```

2. Run it with the three Apertus agents (`--planner graph`):

   ```
   .venv/bin/python pipelines/path_a/run.py --data heldout --out results_heldout.json --planner graph
   ```

3. Score it (on a folder with answers, such as `dataset`):

   ```
   .venv/bin/python starter/evaluate.py results_heldout.json --data heldout
   ```

On an Apple-silicon Mac it takes about 12 minutes for 10 new scenes, and about 6 when the scene graphs are
already cached (`pipelines/path_a/out/`). The run also writes `results_heldout.log.jsonl`, with every Apertus
reply and correction. The agents' rules
are text files in `pipelines/path_a/prompts/`; how they work together is in
[MULTI_AGENT.md](pipelines/path_a/MULTI_AGENT.md). Offline tests:
`.venv/bin/python -m unittest discover -s pipelines/path_a -p 'test_*.py'`.

### Path B

1. Install its dependencies. No Hugging Face token is needed:

   ```
   .venv/bin/pip install -r pipelines/path_b/requirements.txt
   ```

2. Run it. The server and model come from `.env`:

   ```
   .venv/bin/python pipelines/path_b/run.py --data dataset --out results/path_b.json
   ```

3. Score it:

   ```
   .venv/bin/python starter/evaluate.py results/path_b.json --data dataset
   ```

The run takes about 2.5 minutes for the 10 public scenes on the hosted 70B, and also writes
`<out>.raw.jsonl` with every Apertus reply. `--help` lists the options, such as `--workcell` for another
setup's settings and `--thinking` for Apertus's thinking mode (off by default: slower, and worse on the
public scenes). Offline tests, no key needed:
`.venv/bin/python -m unittest discover pipelines/path_b/tests`.

## Repository layout

| Path | What it is |
|---|---|
| `pipelines/path_a/`, `pipelines/path_b/` | One folder per solution, each with its own `run.py`, `requirements.txt` and architecture notes |
| `results/` | One results file and log per pipeline |
| `docs/` | The organisers' notes (`CHALLENGE.md`, `DATA.md`, `safety_primer.md`) and ours on Apertus's limits |
| `dataset/`, `starter/`, `schema/`, `baseline/`, `examples/` | The organisers' files, unchanged |

## Licence

Code: Apache 2.0. Data in `dataset/`: CC BY 4.0. SAM 3 is used under Meta's SAM License.
