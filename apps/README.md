# Pick Lab — local hackathon demo

Next.js animates robot actions; FastAPI reads replay files and runs the existing Python pipelines on explicit live requests. Both services live in this repository. Nothing is deployed.

## Start

From the repository root (Node.js 20.9+ and [uv](https://docs.astral.sh/uv/)):

```sh
make app
```

It installs both pipelines and the API's packages (`uv sync --group app`), then the web packages
(`npm ci`), and starts both services (`npm run dev`).

Open **http://localhost:3000**. The backend listens on `127.0.0.1:8000`. The launcher uses `.venv/bin/python` if present; `SIMULATOR_PYTHON` can select another interpreter. Stop both with Ctrl+C.

Replay requires only the API dependencies above, not SAM 3, PyTorch, or an Apertus key. The app opens a saved Path A multi-agent result. Select an example, click **Load saved plan**, then **Play** or step through it. It replays `results/path_a_dev.json` (Path A, multi-agent) and `results/path_b_dev.json` (Path B), the files each pipeline's `run.py --data dataset` writes. The single-call planner has no saved results; `simulation/storage.py` names the file it would replay. Missing replays are shown as unavailable rather than fabricated.

### Instruction sources

- **Type manually:** enter your own instruction. Selecting this source switches to Live; no API call happens until you click **Generate a plan**. Your draft is preserved when you switch sources or scenes.
- **From dataset:** select a line from the selected scene's `instructions.txt` (or `instruction.txt` if the plural filename is absent). The app displays the file path and instruction preview. Supports plain text lines and the existing category + tab + instruction format. Use this source with either Replay or Live. Live reads the selected instruction from the file on the backend.

Selecting Replay switches the source to dataset because saved results belong to specific dataset instruction IDs. Scenes without an instruction file still support manual Live input.

## Live inference — run when you are ready

1. `make app` already installs both pipelines' packages (`requirements.txt`).

2. Configure `APERTUS_API_KEY`, `API_BASE_URL`, and `API_MODEL` in the root `.env`, following `.env.example`. Path A also needs `HF_TOKEN` and access to SAM 3. Secrets are read only in Python; they are never sent to the browser. Path A settings now pass through the app adapter without modifying the original pipeline defaults.

   The app's Path B adapter defaults to the hosted Apertus endpoint and `apertus-v1.5-70b`, matching `.env.example`. Port 8000 belongs to this simulator's backend, not an inference server. If using local Apertus, run it on a different port and set `API_BASE_URL` to that server's `/v1` URL.
3. Select **Live Apertus**, enter an instruction, and click **Generate a plan**. Merely selecting Live makes no model call. A live request processes one scene and one instruction, reusing its cached perception graph when available. Path B still runs its instruction safety check each time. Cached graphs live under `artifacts/simulator/graphs/` and are keyed by photo content, calibration, and (for Path B) workcell settings. If detection code/model configuration changes, clear this ignored cache before rerunning.

Live jobs run one at a time, isolated in subprocesses to avoid collisions between the existing same-named Python modules. Results persist under `artifacts/simulator/jobs/`. The UI shows the current stage and elapsed time; `progress.json` records stage checkpoints. A `200 OK` from `/api/jobs/...` only means the status request succeeded, not that inference completed. Path B has a 10-minute overall limit, with 90-second model requests and two retries for retryable errors. Path A has a 30-minute limit. The app's Path B adapter surfaces model request failures rather than allowing the CLI's empty-response fallback to hide them. Backend job status is in memory and is lost on backend restart. This is a local demo, not a public multi-user server.

Unexpected Python failures, including `SystemExit`, write a structured error and `diagnostics.json` with exception type, stage, and stack locations. Exception text and model responses are not included in diagnostics. A worker terminated before it can write a response instead reports its exit code or signal. Error banners identify the requested pipeline and job ID. Missing keys for the hosted endpoint and missing `HF_TOKEN` for new Path A perception fail before model requests. Path B does not require `HF_TOKEN`.

The app supplies Path B with a verified TLS context that adds `certifi` CA roots to Python's system/custom trust store. This supports Python installations with an empty default CA store without disabling certificate or hostname verification.

You can enforce replay-only operation:

```sh
SIMULATOR_DISABLE_LIVE=1 npm run dev
```

This disables Live in the UI and rejects live requests in both the API and worker.

## Views and playback

- **Robot:** a side-view schematic with an animated two-link arm, gripper, colored objects, and trays. Pick-and-hold leaves the object in the gripper. Place uses the destination object/tray, not a new picked object.
- **Top view:** a schematic overhead layout with separate grey and black trays, object labels, and an animated gripper. Initial tray membership follows the live scene graph's `in` field; pixel coordinates determine relative positions within each tray. Unknown membership (including older saved replay data) appears in a clearly labeled unassigned area. Pick/place playback stays synchronized when switching views. Placing into a tray chooses an illustrative available slot, not a certified robot destination. Side view also respects known black-tray membership.
- **Scene:** the original photo with available object pixels overlaid. This is the reference view for grounding. It remains the initial photograph throughout playback.
- **Playback:** play/pause, step, reset, and speed controls. Ask/refuse outcomes display their message and disable movement. JSON inspection/download preserves the original challenge answer alongside display metadata.

Each instruction starts from the original scene. Clarification is supplied by submitting a new, complete Live instruction; there is no conversational memory. The demo does not estimate actual joint angles, certify a placement point, execute hardware, or provide physics. Side-view positions and tray slots are illustrative. Replay reconstructs available object positions from saved actions, so it may omit objects never referenced by any saved action; absent coordinates use schematic positions. No ground-truth files are read.

## Local checks (no Apertus calls)

```sh
npm run typecheck
npm run build
# Engine checks use Node.js 22.13+ for its built-in TypeScript stripping API.
npm run test --workspace apps/web
.venv/bin/python -m pip install -r apps/api/requirements-dev.txt
SIMULATOR_DISABLE_LIVE=1 .venv/bin/python -m unittest discover -s apps/api/tests
```

Checks block socket connections and exercise only dataset loading, saved replay, scene images, and the live-disable guard. No live pipeline was invoked during development.

## Structure

```text
apps/web/src/app/          # Next.js page and visual styles
apps/web/src/components/   # SVG robot renderer
apps/web/src/lib/          # API client, shared types, deterministic playback state
apps/api/main.py          # FastAPI scene/replay/live-job endpoints
simulation/storage.py    # Saved outputs and read-only dataset access
simulation/worker.py     # Explicit live adapter; one pipeline per process
scripts/dev.mjs          # Launch both local services
```

Next.js proxies `/api/*` to the Python service; the browser uses one origin. Override the server-only `SIMULATOR_API_URL` if needed. For separate terminals, use `npm run dev:web` and `.venv/bin/python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000`. Production frontend: `npm run build`, then `npm run start --workspace apps/web`, with the Python service running separately.
