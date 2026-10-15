"""Fresh Path B run with auditable model, detection, and planner traces."""
import dataclasses
import datetime
import json
from pathlib import Path
import ssl
import sys
import threading
import time
import traceback
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipelines/path_b'))
import pipeline
import apertus
import detect
import planner
import safety
import certifi

OUT = ROOT / 'report/data'
OUT.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
context = threading.local()
active_scene = None
traces = []
scenes = []

base_url = pipeline.env_value('API_BASE_URL')
model_name = pipeline.env_value('API_MODEL')
key = pipeline.api_key()
if not base_url or not model_name or key == 'none':
    raise SystemExit('Required Apertus configuration is missing.')
client = apertus.ApertusClient(base_url, model_name, 1024, 90, key, 5,
    ssl_context=ssl.create_default_context(cafile=certifi.where()))
original_post = client._post

def post(body):
    response = original_post(body)
    context.response = {'finish_reason': response.get('choices', [{}])[0].get('finish_reason'),
                        'usage': response.get('usage')}
    return response
client._post = post

def traced_model(prompt, images, schema, temperature=0):
    stage = 'detection' if schema is detect.BOX_SCHEMA else ('safety' if schema is safety.PRECHECK_SCHEMA else 'planning')
    record = {'scene': active_scene, 'stage': stage, 'prompt': prompt, 'image_count': len(images)}
    started = time.monotonic()
    context.response = {}
    try:
        reply = client(prompt, images, schema, temperature)
        record.update(reply=reply, **context.response)
        return reply
    except Exception as error:
        record['error'] = {'type': type(error).__name__}
        if isinstance(error, urllib.error.HTTPError): record['error']['http_status'] = error.code
        raise OSError(f'Model request failed ({type(error).__name__})') from None
    finally:
        record['elapsed_seconds'] = round(time.monotonic()-started,3)
        with lock:
            traces.append(record)
            with (OUT/'model_calls.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')

original_measured = pipeline.measured_objects
original_graph = planner.scene_graph
observed = {}

def measured(*args, **kwargs):
    objects = original_measured(*args, **kwargs)
    observed['measured_objects'] = [dataclasses.asdict(o) for o in objects]
    return objects

def graph(*args, **kwargs):
    value = original_graph(*args, **kwargs)
    observed['graph'] = value
    return value
pipeline.measured_objects = measured
planner.scene_graph = graph

subset = OUT/'single-scene-input'
subset.mkdir(exist_ok=True)
cal = subset/'calibration.json'
if not cal.exists(): cal.symlink_to(ROOT/'dataset/calibration.json')
results = []
started = time.monotonic()
metadata = {'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'model':model_name, 'base_url':base_url, 'data':'dataset/scene_01..scene_10',
 'max_side':1024,'temperature':0,'max_tokens':apertus.MAX_TOKENS,'parallel':5,
 'timeout_seconds':90,'thinking':False,'safety':True,'widths':True,'gaps':True,
 'y_scale':1.065,'y_shift_px':40,'fresh_detection':True}
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))

try:
    with (OUT/'planner_attempts.jsonl').open('w') as log:
        for folder in sorted((ROOT/'dataset').glob('scene_*')):
            if not (folder/'rgb.png').exists(): continue
            active_scene = folder.name
            observed.clear()
            link = subset/folder.name
            if not link.exists(): link.symlink_to(folder)
            t = time.monotonic()
            print(f'{active_scene}: fresh detection + all instructions',flush=True)
            sub = pipeline.run(subset,traced_model,log=log)
            log.flush()
            results.extend(sub['results'])
            item = {'scene':active_scene,'elapsed_seconds':round(time.monotonic()-t,3),
                    **observed,'results':sub['results']}
            scenes.append(item)
            (OUT/f'{active_scene}.json').write_text(json.dumps(item,indent=2))
            (OUT/'results.json').write_text(json.dumps({'team':'path_b_failure_audit','results':results},indent=2))
            link.unlink()
            print(f'{active_scene}: finished in {item["elapsed_seconds"]:.1f}s, {len(sub["results"])} answers, {len(observed.get("measured_objects",[]))} objects',flush=True)
finally:
    metadata.update(elapsed_seconds=round(time.monotonic()-started,3),
        scenes_completed=len(scenes),instructions_completed=len(results),
        calls=len(traces),request_errors=sum('error' in x for x in traces))
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
