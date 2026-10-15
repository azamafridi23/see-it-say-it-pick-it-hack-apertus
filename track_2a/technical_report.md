# Technical report — See It, Say It, Pick It

One photo of a robot table and one English instruction go in; a safe robot action comes out: a
pick-and-place plan with the target pixel, a question for the person, or a refusal. How to run it:
[docs/USAGE.md](docs/USAGE.md), or simply `make run`.

- **Track:** Track 2A — ZHAW: See It, Say It, Pick It: Vision-Language Grounding for a Real Robot Arm
- **Event:** Online
- **Team:** Asian Prompters — Randy Christian Saputra, Azam Afridi, [member], [member]
- **Demo:** [link to the video]
- **Slides:** [docs/slides.pdf](docs/slides.pdf)

## 1. Summary

Each photo is first turned into a scene graph: Meta's SAM 3 segments every object, a depth model tells
which of two touching objects is higher, and our code measures each object's position, size in
millimetres, what lies on what, and whether the gripper can pick it. Three Apertus 1.5 70B agents then
read the graph and the instruction: a safety agent checks the requested movement, a reference agent
chooses the objects and their destinations, and a constraint agent checks that nothing forbidden moves.
Code orders the moves and checks every reply. On the ten public scenes, this pipeline (Path A) scores
**63.0 of the 65 automatic points** on average over nine runs (62.5 to 64.1), against 54.0 for our
second approach (Path B) and 21.7 for the organisers' baseline.

## 2. Architecture

```
photo ──► Apertus names the objects ──► SAM 3 masks ──► depth estimate ──► scene graph (code)
                                                                                 │
instruction ──► safety agent ──► reference agent ──► move ordering (code) ──► constraint agent ──► answer
```

| Step | Model or code | Where it runs |
|---|---|---|
| Name the objects in the photo; each name becomes an extra SAM 3 prompt | Apertus 1.5 70B | hosted endpoint |
| Segment every object, from 16 fixed prompts plus Apertus's names | Meta SAM 3 (`jetjodh/sam3`, pinned) | local: CPU in Docker; the Apple or NVIDIA GPU without Docker |
| Estimate which of two touching objects is higher | Depth Anything V2 Small | local |
| Build the scene graph: ids, pixels, sizes in mm, stacking, gaps, pickability | our code (`scene_graph.py`) | local |
| Safety, reference and constraint decisions | three Apertus 1.5 70B agents | hosted endpoint |
| Order the moves, check every reply against a JSON schema and the graph | our code (`agents.py`, `planning.py`) | local |

The code is in `src/pipelines/path_a/`; [ARCHITECTURE.md](src/pipelines/path_a/ARCHITECTURE.md) and
[MULTI_AGENT.md](src/pipelines/path_a/MULTI_AGENT.md) describe it in detail. Every tuned value (prompts,
thresholds, model names) is in [workcell.ini](src/pipelines/path_a/workcell.ini). Path B, our comparison,
lets Apertus draw the boxes itself, measures them with OpenCV and plans in one call
(`src/pipelines/path_b/`).

## 3. Use of Apertus

- **Model:** Apertus 1.5 70B (`apertus-v1.5-70b`)
- **How it is used:** inference, as agents with tool-free structured outputs
- **Where it runs:** the hosted hackathon endpoint (any OpenAI-compatible server works; set
  `LLM_BASE_URL` and `LLM_NAME`)

Apertus answers at temperature 0 with its replies constrained to a JSON schema per agent. A reply that
breaks the schema or the scene graph is sent back with the error, up to two times; an agent that still
fails makes the pipeline refuse, so an invalid reply never becomes a robot move. The prompts are in
`src/pipelines/path_a/prompts/`. SAM 3 and Depth Anything are open-weight models that run locally;
they only perceive, and every decision is Apertus's.

## 4. Data

- **Public scenes** (`data/dataset/`, 10 scenes, 50 instructions, with ground truth) and **held-out
  scenes** (`data/heldout/`, 5 scenes, 40 instructions): provided by the organisers (ZHAW), CC BY 4.0.
  Photos from a Zivid camera above the table; no people other than one hand.
- **Models:** SAM 3 under Meta's SAM License (redistribution permitted), Depth Anything V2 Small under
  Apache 2.0. The weights are downloaded, never committed.

## 5. Evaluation

The organisers' `src/starter/evaluate.py` scores grounding (35 points: is the first target pixel on the
right object?) and safety (30 points: are questions and refusals correct, without refusing valid
instructions?).

| Setup | Grounding (35) | Safety (30) | Automatic total (65) |
|---|---|---|---|
| Organisers' baseline (Apertus alone) | | | 21.7 |
| Path B: Apertus detects, one planning call (4 runs) | 31.5 | 24.3 | 54.0 (52.7–55.8) |
| Path A: one Apertus call on the scene graph | | | 58.9–60.5 |
| **Path A: three agents (ours, 9 runs)** | 34.1 | 30.0 | **63.0 (62.5–64.1)** |

In our fresh audit run, Path A refused or questioned all 10 instructions that needed it correctly and
refused 1 of the 40 valid instructions by mistake. The held-out scenes have no released ground truth.

## 6. Limitations

- **Unusable replies.** In 09_i4, all three reference-agent replies stopped before the last required
  field and filled the rest with blank space, so the pipeline refused a valid instruction.
- **Spatial words.** Apertus sometimes reads “under” as a lower position in the image.
- **Near ties.** In 10_i3, two gears were 16 and 17 mm from the mandarin; the graph's edge-gap measure
  chose the wrong one, and the agent did not ask.
- **Run-to-run variation.** At temperature 0 the endpoint still varies on a few instructions.
- **Development set.** Both pipelines were tuned on the ten public scenes, so held-out scores will be
  lower; the photos carry no measured depth, so heights come from a monocular estimate.

## 7. Reproducibility

`make run` builds the Docker image and runs Path A on `data/heldout/`; the answers go to
`results/results_heldout.json`. It needs Docker and `LLM_API_KEY`; the model weights are inside the
image. Measured on an Apple M1 Max:

| Setup | Five held-out scenes, 40 instructions |
|---|---|
| Without Docker, Apple GPU (one instruction at a time) | 715 s |
| Without Docker, CPU only (one instruction at a time) | 549 s |
| `make run` in Docker (CPU, 10 cores, 16 GB) | 773 s from a clean checkout: 213 s to build the image, about 560 s to run |

Most of the time is spent waiting for Apertus. The endpoint is not fully deterministic, so a re-run can
change a few answers.

## 8. Next steps

- Ask the person when two objects are almost equally close.
- Retry a reply that turns into blank space as a fresh request, and make the last schema field optional.
- Use measured depth where the camera records it.
- Run Apertus locally with vLLM, so the whole pipeline stays on our own hardware.

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced. The code is
also under Apache 2.0 ([LICENSE](../LICENSE)).

## References

- The challenge: [docs/CHALLENGE.md](docs/CHALLENGE.md), [docs/DATA.md](docs/DATA.md),
  [docs/EVALUATION.md](docs/EVALUATION.md)
- Meta SAM 3: https://huggingface.co/facebook/sam3 (ungated copy: https://huggingface.co/jetjodh/sam3)
- Depth Anything V2: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- Apertus: https://www.swiss-ai.org/apertus
