# Path B Documentation Log

Last updated: 2026-10-10. Scope: `pipelines/path_b_wahyu/`.

## Read this first in a new session

This file is the repository handoff memory for Path B. The root `AGENTS.md` points
coding sessions here. Read this file, then `ARCHITECTURE.md` if needed, then only
modules relevant to the task. Do not rescan the entire repository by default.
These notes are context, not immutable truth: inspect current files/tests when
something conflicts, and update this log after meaningful changes.

The user communicates in Indonesian. Code comments and documentation are in
English. The latest request was to clean up the Path B code, explain architecture
and solved bottlenecks, and leave useful memory for future coding sessions.

## Current implementation map

| Need | Read |
|---|---|
| Run ten scenes or evaluate saved results | `Makefile`, `README.md`, `pipeline.py` |
| Understand the complete flow and remaining bottlenecks | `ARCHITECTURE.md` |
| Load dataset instructions without leaking category labels | `scene.py` |
| Single-scene composition and endpoint configuration | `multimodal.py` |
| SDK calls, JSON parsing, image encoding, repair attempts | `apertus_client.py` |
| Raw perception cache or close-up recognition | `perception.py`, `template_prompt.py` |
| Masks, contours, interior pixels | `measure.py` |
| Box conversion, local foreground, tray correction, measured spatial facts | `grounding.py` |
| Prompted plan and target-constraint enforcement | `instruction_planning.py` |
| Blockers and grouped transfers | `planner.py` |
| Explicit unsafe motion, ambiguity, named/ordinal/neighbor constraints | `safety.py` |
| Pixel/inventory/gripper validation | `validation.py` |
| Regression and offline SDK orchestration checks | `test_pipeline.py` |
| Historical measured scores and preserved result checksum | `benchmark.json` |

The SDK and validators formerly lived inside `multimodal.py`. Compatibility
re-exports remain there, but new imports should use their owning modules.
`template_prompt` and `PLANNING_PROMPT` refer to the same planning prompt.

## Repository context needed for this pipeline

- Root contains `dataset/`, `schema/`, `starter/`, `examples/`, and other pipelines.
- Public dataset: ten `scene_01` through `scene_10` folders, five instructions each,
  RGB photographs at 2448 × 2048 and shared camera calibration.
- Input: each scene's `rgb.png` and `instructions.txt`. Prefix labels are stripped;
  unlabelled instructions are also supported.
- Submission contract: `schema/result.schema.json`.
- Official scorer: `starter/evaluate.py`.
- Pixel-to-robot utility: `starter/scene_io.py`; this pipeline does not drive a robot.
- Example format: `examples/example_results.json` (actual filename; not the typo
  `exmaple_results.json` used in the initial request).
- Path A and dataset/evaluator code are outside the cleanup scope.

## Commands

From the repository root:

```sh
# Install dependencies, run all ten scenes with fresh perception, then score.
make -C pipelines/path_b_wahyu

# Offline checks using an existing environment; no model calls.
make -C pipelines/path_b_wahyu unit

# Rescore the latest Make output without repeating model inference.
make -C pipelines/path_b_wahyu evaluate

# Limit an investigation to a scene and explicit output/cache directories.
python -m pipelines.path_b_wahyu.pipeline --data dataset --scenes scene_07 \
  --out /tmp/path_b_scene07.json --debug-dir /tmp/path_b_scene07_trace \
  --cache-dir /tmp/path_b_scene07_cache

# Score the preserved optimized artifact without recreating diagnostic files.
python starter/evaluate.py pipelines/path_b_wahyu/out/results_optimized.json --data dataset
```

Make variables: `PYTHON`, `VENV`, `WORKERS`, `DATA`, `OUT`, `CACHE`.
Default VENV is root `.venv`; WORKERS is 3; OUT is this folder's `out/make_test`.
Make creates a new timestamp/PID cache path per test; specifying CACHE opts into
reuse. Direct CLI defaults: one worker, root `result_wahyu.json`, this folder's
`out` traces, and `out/cache`.

The root `.env` supplies `API_KEY` and optional `APERTUS_MODEL` / `APERTUS_BASE_URL`.
Default endpoint: `https://hackapertus.livemap.sh/v1`; default model:
`apertus-v1.5-70b`. A local OpenAI-compatible endpoint can replace it. Never put
keys in this log or print `.env` contents. Dependencies are in `requirements.txt`.

During this cleanup, `.venv` was absent. Tests used the already-existing temporary
`/private/tmp/path-b-venv/bin/python`. Do not assume that temporary environment
exists in another session; prefer Make setup or the user's active environment.
Black/isort were locally available and used for formatting; they are development
tools, not required by inference.

## Important behavior to preserve

1. All instructions are independent snapshots of the original scene.
2. Normalize model boxes once from 0–1000; never rescale measured action pixels.
3. `[u, v]` is column/row from top-left. Boxes use exclusive right/bottom bounds.
4. Proposal category hints are not verified object identities. Apertus provides
   identities; OpenCV grounds them and contributes visible geometric evidence.
5. Inference never reads ground truth, masks, reference plans, or expected answers.
6. Pick IDs must exist, be eligible, not be human-held, and respect visible blockers.
   Tray/dish IDs are destinations rather than graspable targets under challenge rules.
7. A single gripper cannot pick twice before placing. Grouped same-destination
   transfers are compiled into pairs, including a broadcast destination.
8. Incidental blockers are cleared before touching the transfer's supports;
   explicit blocker actions remain in the model sequence. Protected blockers
   cause refusal rather than automatic removal.
9. Named/ordinal/immediate-neighbor constraints come from measured inventory.
   Unresolved demonstratives ask the user rather than guessing.
10. Repeated invalid replies stop early. Processing failures produce explicit
    fallbacks and nonzero pipeline exit status, preserving useful other answers.
11. Raw full-photo perception alone is cached. Current grounding, refinements,
    and all instruction planning are rerun. Prompt/image/model/endpoint changes
    alter the cache key; invalid cache contents currently fail the scene.
12. Prompts are user-content task instructions in the tested SDK request shape.
    Whitespace changes can change payloads/cache behavior; formatting must not
    accidentally alter prompt bytes.

## Verified results and provenance

| Evidence | Recorded result | What it means |
|---|---|---|
| Existing `results/path_b_wahyu.json` | 24.5/65, 16/40 grounding, 4/10 safety decisions | Stored pre-optimization comparison; not a new live baseline run |
| Preserved `out/results_optimized.json` | 65/65, 40/40 grounding, 10/10 safety, 50 schema-valid IDs | Verified public-development result using cached raw full-photo perception |
| Root `report.json` observed before cleanup | 55.8/65; grounding 0.90, caught 0.90, false-alarm rate 0.10 | Separate later local run; never substitute it for the preserved benchmark |
| Before-cleanup offline suite | 22 tests passed | Existing regression baseline |
| Cleanup replay | Ten recorded scenes; 49 SDK payloads and outputs/errors identical | Refactor equivalence without a live endpoint |

Preserved optimized-result SHA256:
`76826da724473e3662ad5a24726149c42311a6e3168f22b84cf0a571b9867cd5`.
`benchmark.json` records historical optimization provenance; its original test
count belongs to that run, not a promise about the current test-suite size.

The cleanup adds three offline orchestration checks: scene/cache reuse, bounded
invalid-reply correction, and ordered checkpoints after partial scene failure.
Current suite: 25 tests. Validation details are recorded in the dated entry below.

A 65/65 preserved artifact does not guarantee the same score with a fresh cache,
new server/model version, or held-out scenes. The score checks the first target
and decision type, not all plan steps or physical robot execution. No live-model
inference is required merely to validate a behavior-preserving code cleanup.

## Artifact ownership and cleanup history

The user requested deletion of `out/` contents except the final result. That was
completed, preserving `out/results_optimized.json`. Subsequent user execution
recreated `out/cache/` and `out/scene_*/` diagnostics. Those files were present when
this cleanup started; they were not deleted again.

Root `report.json`, `result_wahyu.json`, and `result_wahyu.json.errors.json` were
also present as untracked local artifacts. Preserve them and unrelated working-tree
changes unless the user requests their modification. Path B output folders are
gitignored. Historical `out/verified_final/` reports were removed by the earlier
cleanup; recreate reports with the scorer rather than assuming old paths exist.

## Known limits and targeted next work

- Model identity/stack recognition remains variable; small parts, handles, dish
  centers, and partly hidden objects can be confused.
- Color/shape thresholds and dark-tray recovery assume this fixed workcell;
  different lighting, cameras, or tray arrangements need validation.
- Dominant-color overlap is an occlusion heuristic, not physical proof of support.
- Missing human-contact flags default to false unless other evidence marks them.
- Same-color contours can still merge; neutral surfaces can fragment.
- Regex constraints cover specific English forms, not arbitrary paraphrases.
- Only the first pick and decision type are automatically scored. Full action
  semantics, collision checking, grasp poses, held-out performance, and physical
  robot execution remain unverified.
- API latency remains the main practical bottleneck. Cache/concurrency/retry limits
  reduce work, but no measured latency benchmark exists. Sequential instructions
  and ordered scene checkpoint consumption can still delay completion visibility.
- Optional semantic refinement failures are logged and retain prior information;
  transport errors propagate to scene fallback. Inspect both `.errors.json` and
  `trace.json/refinement_errors` when diagnosing a run.

## Change log

### 2026-10-10 — Measured grounding and planning optimization

Integrated `measure.py` into Path B; separated recognition from ID-based planning;
fixed the coordinate-frame mismatch; added component/stack close-ups, support
reconciliation, spatial constraints, ambiguity/safety gates, gripper validation,
transfer compilation, blocker clearing, bounded repair, cache reuse, ordered
checkpoints, per-instruction failure isolation, and regression tests. Public score
improved from the stored 24.5/65 result to the preserved 65/65 cached-perception run.

### 2026-10-10 — Make workflow and output cleanup

Added one-command ten-scene testing with setup, fresh cache, scoring, reports,
and robot-coordinate CSV. Cleaned `out/` at the user's request, retaining only the
final optimized result at that time. Later runs may recreate diagnostics.

### 2026-10-10 — Code cleanup and session documentation

Split the former large `multimodal.py` into client, perception, instruction-planning,
and validation modules. Kept legacy helper imports working, centralized all prompt
strings without changing their contents, named geometry helpers clearly, removed
redundant preprocessing/dead debug work, normalized imports/formatting, and added
English boundary/state comments and docstrings. Removed the inactive scene demo.

Created `ARCHITECTURE.md`, this handoff log, and a root `AGENTS.md` pointer.
Added offline integration coverage. No live model request or user-output rewrite
was needed for cleanup validation.

Validation completed:

- 25 offline tests passed, including `make unit` with the existing temporary venv.
- Ten recorded-scene replays matched before/after output/errors and all 49 request payloads.
- Detector candidates and masks matched for all ten scenes and the optional LAB
  background-subtraction branch.
- Black/isort checks, Python compilation, Make dry-run, and `git diff --check` passed.
- Official rescoring of the unchanged preserved artifact returned 65/65; its SHA256
  still matches `benchmark.json`. This is artifact verification, not fresh inference.

## Update checklist for the next session

Record changes to modules, commands, constraints, cache behavior, or output paths.
State which checks actually ran, which artifacts were scored, and whether inference
used a fresh or reused cache. Do not report old scores as new measurements. Keep
secrets out of notes. Inspect only affected files before broadening repository work.
