"""Local simulator API. Replay never imports a pipeline or calls Apertus."""
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

from simulation.storage import PROJECT, ROOT, RESULT_FILES, answers, image_size, instruction_file, instructions, replay, scene_folder

app = FastAPI(title="Pick Lab local API")
jobs: dict[str, dict] = {}
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)


class PlanRequest(BaseModel):
    scene_id: str
    pipeline: Literal["path_a_graph", "path_a", "path_b"] = "path_a_graph"
    mode: Literal["replay", "live"] = "replay"
    instruction_id: str | None = None
    instruction: str = Field(default="", max_length=2000)
    instruction_source: Literal["manual", "dataset"] = "manual"


@app.get("/api/health")
def health():
    return {"status": "ok", "live_enabled": os.environ.get("SIMULATOR_DISABLE_LIVE") != "1"}


@app.get("/api/scenes")
def scenes():
    saved = {pipeline: {a["instruction_id"] for a in answers(pipeline)} for pipeline in RESULT_FILES}
    result = []
    for folder in sorted((PROJECT / "data" / "dataset").iterdir()):
        if not folder.is_dir() or not (folder / "rgb.png").is_file():
            continue
        items = instructions(folder)
        for item in items:
            item["replays"] = [p for p, ids in saved.items() if item["id"] in ids]
        source = instruction_file(folder)
        result.append({"id": folder.name, "image_size": image_size(folder), "instructions": items,
                       "instruction_file": source.name if source else None})
    return {"scenes": result, "pipelines": [{"id": p, "has_replay": path.is_file()} for p, path in RESULT_FILES.items()]}


@app.get("/api/scenes/{scene_id}/image")
def scene_image(scene_id: str):
    try:
        return FileResponse(scene_folder(scene_id) / "rgb.png", media_type="image/png")
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


def run_job(job_id: str, payload: dict):
    work = ROOT / "artifacts/simulator/jobs" / job_id
    source, output = work / "request.json", work / "response.json"
    with lock:
        jobs[job_id] = {"status": "running", "started": time.time()}
    try:
        work.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(payload))
        # Independent Python processes isolate path_a/path_b's same-named modules.
        # Never return raw model/server errors: they can include request details.
        subprocess.run([sys.executable, str(ROOT / "simulation/worker.py"), str(source), str(output)],
                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                       timeout=600 if payload["pipeline"] == "path_b" else 1800)
        result = json.loads(output.read_text())
        with lock:
            jobs[job_id] = {"status": "complete", "result": result}
    except subprocess.TimeoutExpired:
        with lock:
            jobs[job_id] = {"status": "failed", "error": "Live inference exceeded its time limit. Path B allows 10 minutes; Path A allows 30 minutes. Check the endpoint or use Replay."}
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            reason = f"worker exited with code {error.returncode}" if error.returncode >= 0 else f"worker was terminated by signal {-error.returncode}"
        else:
            reason = type(error).__name__
        message = f"{payload['pipeline']} failed: {reason}. No readable worker error was returned."
        if output.is_file():
            try:
                message = json.loads(output.read_text()).get("error", message)
            except ValueError:
                pass
        with lock:
            jobs[job_id] = {"status": "failed", "error": message}
    with lock:
        jobs[job_id]["pipeline"] = payload["pipeline"]
        jobs[job_id]["job_id"] = job_id


@app.post("/api/plans")
def plan(request: PlanRequest):
    try:
        scene_folder(request.scene_id)
        if request.mode == "replay":
            return {"status": "complete", "result": replay(request.scene_id, request.pipeline, request.instruction_id or "")}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if os.environ.get("SIMULATOR_DISABLE_LIVE") == "1":
        raise HTTPException(403, "Live inference is disabled in this session.")
    if request.instruction_source == "dataset":
        selected = next((item for item in instructions(scene_folder(request.scene_id))
                         if item["id"] == request.instruction_id), None)
        if selected is None:
            raise HTTPException(400, "Choose an instruction from the selected scene's dataset file.")
        request.instruction = selected["text"]
    if not request.instruction.strip():
        raise HTTPException(400, "Enter an instruction first.")
    with lock:
        if any(job["status"] in ("queued", "running") for job in jobs.values()):
            raise HTTPException(409, "A live request is already running. Wait for it to finish.")
        # Bound in-memory job retention. Results remain in the artifacts folder.
        if len(jobs) >= 100:
            jobs.pop(next(iter(jobs)))
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "queued"}
    executor.submit(run_job, job_id, request.model_dump())
    return {"status": "queued", "job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job not found. The backend may have restarted.")
        status = dict(jobs[job_id])
    if status["status"] == "running":
        status["elapsed_seconds"] = round(time.time() - status["started"])
        progress_path = ROOT / "artifacts/simulator/jobs" / job_id / "progress.json"
        try:
            status["stage"] = json.loads(progress_path.read_text())["stage"]
        except (OSError, ValueError, KeyError):
            status["stage"] = "Starting the pipeline"
    return status
