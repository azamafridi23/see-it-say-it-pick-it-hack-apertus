"""Explicit live-only entry point. One pipeline per process, one instruction per request."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import socket
import ssl
import traceback
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_APERTUS_URL = "https://hackapertus.livemap.sh/v1"
DEFAULT_APERTUS_MODEL = "apertus-v1.5-70b"


class ModelRequestError(RuntimeError):
    """Safe user-facing request failure; no headers or response bodies."""


class ConfigurationError(RuntimeError):
    """Missing settings, explained without including their values."""


def require_endpoint_key(base_url: str, api_key: str | None):
    if urlparse(base_url).hostname == "hackapertus.livemap.sh" and (not api_key or api_key == "none"):
        raise ConfigurationError("APERTUS_API_KEY is missing for the hosted Apertus endpoint. Set APERTUS_API_KEY in the repository's .env; API_KEY alone is not read by this pipeline.")


def verified_tls_context() -> ssl.SSLContext:
    import certifi
    context = ssl.create_default_context()
    # Add certifi roots while retaining configured system/custom CA roots.
    context.load_verify_locations(cafile=certifi.where())
    return context


def checked_model(client):
    def call(*args, **kwargs):
        try:
            return client(*args, **kwargs)
        except urllib.error.HTTPError as error:
            detail = {401: "Check APERTUS_API_KEY.", 403: "Check your endpoint access.",
                      404: "Check API_BASE_URL and API_MODEL.", 429: "The endpoint is busy; try again later."}.get(error.code, "The endpoint returned an error; try again later.")
            raise ModelRequestError(f"Apertus request failed (HTTP {error.code}). {detail}") from None
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                message = "Apertus request timed out. The endpoint did not respond within the request time limit."
            elif isinstance(reason, ssl.SSLCertVerificationError):
                message = "Cannot verify the Apertus endpoint's TLS certificate. Check the Python certificate setup."
            elif isinstance(reason, socket.gaierror):
                message = "Cannot resolve the Apertus endpoint hostname. Check API_BASE_URL and your network's DNS."
            elif isinstance(reason, ConnectionRefusedError):
                message = "The Apertus endpoint refused the connection. Check that the model server is running at API_BASE_URL."
            else:
                message = "Cannot connect to the Apertus endpoint. Check network access and API_BASE_URL."
            raise ModelRequestError(message) from None
        except TimeoutError:
            raise ModelRequestError("Apertus request timed out. The endpoint did not respond within the request time limit.") from None
        except json.JSONDecodeError:
            raise ModelRequestError("Apertus returned a response body that is not valid JSON. Check that API_BASE_URL points to an OpenAI-compatible model server.") from None
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ModelRequestError("Apertus returned an unexpected response structure. Expected choices[0].message.content from the model server.") from None
        except (OSError, ValueError) as error:
            raise ModelRequestError(f"Apertus response failed ({type(error).__name__}). Check endpoint connectivity and model configuration.") from None
    return call


def report(source: Path, stage: str):
    path = source.parent / "progress.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stage": stage}))
    temporary.replace(path)


def usable_graph(graph) -> bool:
    """A tray-only graph is a failed detection, not reusable scene geometry."""
    return isinstance(graph, dict) and any(
        node.get("pixel") and node.get("id") not in ("grey_tray", "black_tray")
        for node in graph.get("objects", [])
    )


def run(payload: dict, progress=lambda stage: None) -> dict:
    if os.environ.get("SIMULATOR_DISABLE_LIVE") == "1":
        raise RuntimeError("Live inference is disabled in this session.")
    from simulation.storage import image_size, scene_folder
    folder = scene_folder(payload["scene_id"])
    selected = payload["pipeline"]
    started = time.monotonic()
    progress("Loading pipeline dependencies")
    instruction_id = f"{folder.name}_live"
    cache = ROOT / "artifacts/simulator/graphs"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((folder / "rgb.png").read_bytes()).hexdigest()[:16]
    # Include calibration and workcell configuration so geometry is not reused across setups.
    calibration = folder / "calibration.json"
    if not calibration.exists():
        calibration = folder.parent / "calibration.json"
    settings = calibration.read_bytes()
    if selected == "path_b":
        settings += (ROOT / "pipelines/path_b/workcell.ini").read_bytes()
    config_digest = hashlib.sha256(settings).hexdigest()[:12]
    cache_path = cache / f"{folder.name}_{'path_b' if selected == 'path_b' else 'path_a'}_{digest}_{config_digest}.json"
    graph = json.loads(cache_path.read_text()) if cache_path.exists() else None
    if not usable_graph(graph):
        graph = None
    if selected == "path_b":
        sys.path.insert(0, str(ROOT / "pipelines/path_b"))
        import pipeline
        from apertus import ApertusClient, ask
        import apertus
        import measure
        import planner
        from safety import precheck, refusal
        import workcell
        from scene_io import load_scene

        # Keep the CLI's retry policy unchanged; interactive requests use a shorter budget.
        apertus.RETRY_WAITS = (1, 2)
        base_url = pipeline.env_value("API_BASE_URL") or DEFAULT_APERTUS_URL
        api_key = pipeline.api_key()
        require_endpoint_key(base_url, api_key)
        model = checked_model(ApertusClient(base_url,
                              pipeline.env_value("API_MODEL") or DEFAULT_APERTUS_MODEL,
                              1024, 90, api_key, ssl_context=verified_tls_context()))
        if graph is None:
            progress("Detecting objects with Apertus")
            scene = load_scene(folder)
            cell = workcell.load()
            objects = pipeline.measured_objects(model, scene, cell, True, True)
            if not objects:
                progress("Retrying object detection: no objects found on the first pass")
                objects = pipeline.measured_objects(model, scene, cell, True, True)
            if not objects:
                raise ModelRequestError("Object detection found no usable objects. No scene cache was saved. Generate again to retry detection.")
            graph = planner.scene_graph(objects, measure.black_mat_left(scene.rgb, cell), cell)
            for node, detected in zip(graph["objects"], objects):
                node["box"] = detected.box
            cache_path.write_text(json.dumps(graph))
        progress("Checking instruction safety with Apertus")
        reason = precheck(model, payload["instruction"])
        if reason:
            answer = refusal(instruction_id, reason)
        else:
            progress("Planning actions with Apertus")
            answer, _ = planner.plan(lambda prompt, schema: ask(model, prompt, [], schema),
                                     instruction_id, payload["instruction"], graph)
    else:
        sys.path.insert(0, str(ROOT / "pipelines/path_a"))
        import planner
        model_name = os.environ.get("API_MODEL") or planner.MODEL
        base_url = os.environ.get("API_BASE_URL") or planner.BASE_URL
        require_endpoint_key(base_url, os.environ.get("APERTUS_API_KEY"))
        if graph is None:
            if not os.environ.get("HF_TOKEN"):
                raise ConfigurationError("HF_TOKEN is missing. Path A needs a Hugging Face token with access to SAM 3. Set HF_TOKEN in .env, or select Path B which does not use SAM 3.")
            progress("Loading vision models and scene geometry")
            from apertus_list import names_for
            from run_sam3 import detect
            from scene_graph import build
            progress("Recognizing objects with Apertus and SAM 3")
            extra = detect(folder, names_for(folder, model_name, base_url))
            graph, _ = build(folder, detect(folder), extra=extra)
            if not usable_graph(graph):
                raise ModelRequestError("Object detection found no usable objects. No scene cache was saved. Generate again to retry detection.")
            cache_path.write_text(json.dumps(graph))
        if selected == "path_a_graph":
            from agents import GraphPlanner
            plan = GraphPlanner(model=model_name, base_url=base_url)
        else:
            plan = planner.Planner(model=model_name, base_url=base_url)
        progress("Planning and validating actions with Apertus")
        answer, _ = plan.answer(instruction_id, payload["instruction"], graph)
    progress("Preparing the simulation result")
    nodes = [{**node, "type": node.get("type") or node.get("label") or node["id"],
              "pixel": node.get("pixel")} for node in graph["objects"]]
    return {"scene_id": folder.name, "pipeline": selected, "mode": "live", "instruction": payload["instruction"],
            "answer": answer, "objects": nodes, "image_size": image_size(folder),
            "geometry_source": "pipeline_graph", "elapsed_seconds": round(time.monotonic() - started, 2)}


def execute(source: Path, output: Path) -> int:
    selected = "unknown pipeline"
    try:
        payload = json.loads(source.read_text())
        selected = payload.get("pipeline", selected)
        result = run(payload, lambda stage: report(source, stage))
    except (ModelRequestError, ConfigurationError) as error:
        result = {"error": str(error)}
    except ImportError as error:
        path = "path_b" if selected == "path_b" else "path_a"
        module = error.name or "a pipeline module"
        result = {"error": f"Cannot import {module} for {selected}. Install dependencies in the app's Python environment: .venv/bin/python -m pip install -r pipelines/{path}/requirements.txt. If already installed, check package compatibility. See apps/README.md."}
    except (Exception, SystemExit) as error:
        try:
            stage = json.loads((source.parent / "progress.json").read_text()).get("stage", "starting")
        except (OSError, ValueError):
            stage = "starting"
        # Persist only exception type and stack locations, never exception text,
        # source lines, model replies, request bodies, or credentials.
        diagnostic = {"pipeline": selected, "stage": stage, "exception_type": type(error).__name__,
                      "frames": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                                 for frame in traceback.extract_tb(error.__traceback__)]}
        (source.parent / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2))
        result = {"error": f"{selected} stopped during {stage} ({type(error).__name__}). See this job's diagnostics.json for the failing code location."}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result))
    temporary.replace(output)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    source, output = map(Path, sys.argv[1:3])
    sys.exit(execute(source, output))
