# Path B: OpenCV measurements + Apertus

For implementation details and solved bottlenecks, read [ARCHITECTURE.md](ARCHITECTURE.md).
For a new coding session, start with [DOCUMENTATION_LOG.md](DOCUMENTATION_LOG.md).

To install dependencies, run all ten dataset scenes with fresh perception, and
score the results in one command from the repository root:

```sh
make -C pipelines/path_b_wahyu
```

This uses the root `.venv` and requires `API_KEY` in `.env`. Results, the evaluation
report, robot-coordinate CSV and scene traces are saved in `out/make_test/`.
Each default run uses a new perception cache. Use `make evaluate` from this
directory to rescore saved results, or `make help` to see options.

Run from the repository root (Python 3.11+):

```sh
python -m pip install -r pipelines/path_b_wahyu/requirements.txt
python -m pipelines.path_b_wahyu.pipeline --data dataset --out result_wahyu.json --workers 3
python starter/evaluate.py result_wahyu.json --data dataset --report pipelines/path_b_wahyu/out/evaluation.json
```

Set `API_KEY` in the root `.env`. `APERTUS_BASE_URL` and `APERTUS_MODEL` can select
an OpenAI-compatible local endpoint instead of the existing hackathon endpoint
and `apertus-v1.5-70b`. `--model` overrides the model environment variable.
`--scenes scene_01 scene_02` limits the run; `--debug-dir` selects the trace folder.
Workers process scenes concurrently while results retain dataset order.

The vision step uses one original photo with a short Apertus prompt to obtain
identities and coarse normalized boxes. `measure.py` segments separate color
regions, including dim neutral components. `grounding.py` matches compatible
proposals to the boxes, or segments a local crop, choosing a solid foreground
point away from holes and boundaries. Oversized merged proposals are rejected.
Normalized boxes are converted once; final action pixels are original pixels.

Unexplained neutral proposals receive a separate component recognition pass on
close-up images. Overlapping manipulation objects receive a stacking recognition
pass. Dominant foreground color can reconcile gear/mug occlusion direction when
both objects have measured proposals. Detected tray rectangles correct inconsistent
support labels. If the dark tray is omitted, a reference is added only when a large
dark support region is visible beside the detected grey tray. These are explicit
heuristics for the challenge's fixed workcell, rather than a universal detector.

Apertus plans each instruction independently from the same scene inventory.
The input includes measured distances, left/right ordering, eligible pick IDs,
blockers, and direct-name/ordinal selection constraints. The model chooses IDs;
code supplies their pixels. Simple noun cardinality is checked against the
inventory, so repeated indistinguishable instances require clarification.
Unresolved demonstratives such as “that one” also require clarification.

`planner.py` serializes grouped transfers to a shared destination into pick/place
pairs and clears visible incidental blockers before moving their supports.
Explicit blocker-removal actions are retained. `safety.py` rejects explicit
uncontrolled motion, taking objects from a hand, contradictory relocation demands,
and extracting a support while expressly forbidding movement of its blocker.
Validation rejects invented IDs, invalid pixels, picks outside trays, human-held
objects, ungraspable containers, unresolved blockers and invalid gripper sequences.
Invalid model replies receive correction attempts; repeated invalid replies stop
early. A complete JSON value is parsed without absorbing trailing explanation.

The output follows `schema/result.schema.json`: exactly one entry per instruction,
with the first pick as `target_pixel`, or null for asking/refusing. Checkpoints and
an adjacent `.errors.json` are saved after each scene. Instruction-level processing
failures preserve other answers and produce an explicit ask_user fallback plus a
nonzero exit status. Optional refinement failures are recorded separately in traces.
A genuine ambiguity is distinct from an infrastructure/validation fallback.

`out/<scene>/` contains masks, candidate boxes, close-up tiles and `trace.json` with
raw replies, validation errors and accepted inventory/plans. `--cache-dir` selects
the raw full-photo perception cache, keyed by image bytes, perception prompt, model
and endpoint. Grounding, close-up refinement and planning are recomputed each run.
Use a new empty cache directory to measure a fully fresh run.

Inference never reads ground truth, CSV expected answers, reference plans or object
masks. Category prefixes are stripped from instructions and never given to the model. The scorer reads evaluation metadata only after inference.

Offline regression checks:

```sh
python -m unittest pipelines.path_b_wahyu.test_pipeline -v
```

These cover all ten scenes' candidate foreground points, output schema, gripper
sequences, blocker handling, holes, ordinal direction, ambiguity and motion gates.
They do not substitute for the official semantic accuracy evaluation.

Color segmentation can merge touching same-color gears, miss dim parts and split
dishes. Perception can confuse small parts or stacking direction; the extra passes
and geometry checks reduce these errors but cannot prove physical support from RGB.
Pixels and tray centers are image references, not collision-free grasp/placement
poses. Robot coordinates can be computed separately with `starter/scene_io.py`.

Measured public development results (10 scenes, 50 instructions):

| Metric | Existing `results/path_b_wahyu.json` | Optimized run |
|---|---:|---:|
| First pick on correct object | 16/40 | 40/40 |
| Correct ask/refuse decisions | 4/10 | 10/10 |
| Automatic score | 24.5/65 | 65/65 |

The final run has 50 schema-valid unique answers, zero processing errors, and
22 passing offline regression tests at the time of the benchmark. The current
cleanup suite contains 25 offline tests, including SDK/cache/checkpoint integration
coverage. Results are in `out/results_optimized.json`,
with the official report and robot-coordinate CSV in `out/verified_final/` during that run; those diagnostic files were subsequently
removed at the user's request. The optimized result remains available.
`benchmark.json` records the comparison and result digest. The run reused cached
raw full-photo perception, while grounding, close-up refinement and all planning
were rerun. The automatic scorer checks the first target and decision type;
the remaining 35 jury points, full plan review, held-out scenes and physical robot
execution have not been validated by this score.

Code organization: `multimodal.py` now only composes the stages. SDK/JSON transport
lives in `apertus_client.py`, visual stages in `perception.py`, target selection in
`instruction_planning.py`, and state checks in `validation.py`. Geometry and
transfer compilation remain in `grounding.py` and `planner.py`. Legacy helper
imports from `multimodal.py` are retained for existing notebooks.

Formatting during cleanup used Black's default 88-column style and isort's Black
profile. Prompt contents were preserved; replay of ten recorded scenes verified
identical output/errors and all 49 SDK request payloads. This check did not make
live model calls. A separate root `report.json` observed during cleanup records
55.8/65, illustrating why cached benchmark results should not be treated as a
fresh-run guarantee.
