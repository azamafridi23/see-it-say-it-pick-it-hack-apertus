"""Evaluate fresh results and audit raw detections against annotated target masks."""
import collections
import json
from pathlib import Path
import re
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'report/data'
sys.path.insert(0,str(ROOT/'starter'))
from evaluate import answer_type

calls=[json.loads(x) for x in (DATA/'model_calls.jsonl').read_text().splitlines()]
truth=json.loads((ROOT/'dataset/ground_truth/ground_truth.json').read_text())['instructions']
results={x['instruction_id']:x for x in json.loads((DATA/'results.json').read_text())['results']}
evaluation=json.loads((DATA/'evaluation.json').read_text())
annotations=[]
discarded=[]
scenes=[]

def canonical(name):
    name=re.sub(r'_\d+$','',name.replace(' ','_'))
    return 'small_part' if name=='small_white_part' else name

def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0

allowed={'red cube','red gear','yellow gear','mug','mandarin','banana','plate','saucer','grey part','white part','small white part','hand'}
for scene_path in sorted(DATA.glob('scene_*.json')):
    scene=json.loads(scene_path.read_text())
    name=scene['scene']
    labels=cv2.imread(str(ROOT/f'dataset/ground_truth/masks/{name}.png'),cv2.IMREAD_UNCHANGED)
    h,w=labels.shape
    meta=json.loads((ROOT/f'dataset/ground_truth/masks/{name}.json').read_text())['objects']
    detection=next(x for x in calls if x['scene']==name and x['stage']=='detection')
    raw_text=detection.get('reply','')
    try: raw=json.loads(raw_text)
    except json.JSONDecodeError:
        match=re.search(r'\{.*\}',raw_text,re.S)
        raw=json.loads(match.group()) if match else {}
    accepted=[]
    for index,entry in enumerate(raw.get('objects',[])):
        box=entry.get('box') or entry.get('bounding_box') or []
        reason=None
        if entry.get('kind') not in allowed: reason='kind not allowed'
        elif len(box)!=4: reason='invalid box'
        else:
            x1,y1,x2,y2=box
            if (x2-x1)*(y2-y1)/1e6>0.25: reason='tray-sized box'
            elif (min(x1,y1)<=2 or max(x2,y2)>=998) and entry['kind']!='hand': reason='photo-edge box'
        if reason:
            discarded.append({'scene':name,'proposal_index':index,'reason':reason,**entry})
            continue
        raw_box=[x1*w/1000,y1*h/1000,x2*w/1000,y2*h/1000]
        corrected=[x1*w/1000,min(h,1.065*y1*h/1000+40),x2*w/1000,min(h,1.065*y2*h/1000+40)]
        accepted.append({'raw_kind':entry['kind'],'raw_box':raw_box,'box':corrected,
             'raw_center':[round((raw_box[0]+raw_box[2])/2),round((raw_box[1]+raw_box[3])/2)],
             'corrected_center':[(round(corrected[0])+round(corrected[2]))//2,(round(corrected[1])+round(corrected[3]))//2]})
    accepted.sort(key=lambda p:p['box'][0]+p['box'][2])
    measured=scene['measured_objects']
    assert len(measured)==len(accepted),(name,len(measured),len(accepted))
    for entry,obj in zip(accepted,measured):
        entry.update(final_kind=obj['kind'],corrected_center=obj['pixel'])
    masks=[]
    for label,objname in meta.items():
        ys,xs=np.where(labels==int(label))
        masks.append({'label':int(label),'name':objname,'box':[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]})
    # Match by overlap, independently of labels, so type errors remain observable.
    edges=sorted([(iou(pred['box'],gt['box']),i,j) for i,pred in enumerate(accepted) for j,gt in enumerate(masks)],reverse=True)
    used_pred=set(); used_gt=set()
    for overlap,i,j in edges:
        if overlap<0.25: continue
        if i in used_pred or j in used_gt: continue
        used_pred.add(i); used_gt.add(j)
        pred,gt=accepted[i],masks[j]
        def hit(px):
            x,y=px
            return 0<=x<w and 0<=y<h and labels[y,x]==gt['label']
        def on(px):
            x,y=px
            return meta.get(str(int(labels[y,x])),'background') if 0<=x<w and 0<=y<h else 'outside image'
        annotations.append({'scene':name,'ground_truth':gt['name'],'gt_box':gt['box'],
            **pred,'bbox_iou':round(overlap,3),'raw_center_hit':bool(hit(pred['raw_center'])),
            'corrected_center_hit':bool(hit(pred['corrected_center'])),
            'raw_center_on':on(pred['raw_center']),'corrected_center_on':on(pred['corrected_center']),
            'raw_label_correct':canonical(pred['raw_kind'])==canonical(gt['name']),
            'final_label_correct':canonical(pred['final_kind'])==canonical(gt['name']),
            'raw_y_center_error_px':round((pred['raw_box'][1]+pred['raw_box'][3]-gt['box'][1]-gt['box'][3])/2,1)})
    scenes.append({'scene':name,'raw_proposals':len(raw.get('objects',[])),
      'accepted_objects':len(accepted),'annotated_targets':len(masks),'matched_targets':len(used_gt),
      'unmatched_targets':[m['name'] for j,m in enumerate(masks) if j not in used_gt],
      'elapsed_seconds':scene['elapsed_seconds']})

failures=[r for r in evaluation['instructions'] if r['score']==0]
attempts=[json.loads(x) for x in (DATA/'planner_attempts.jsonl').read_text().splitlines()]
rejected=[{'instruction_id':x['instruction_id'],'attempt_index':i+1,**a} for x in attempts for i,a in enumerate(x.get('attempts',[])) if a.get('error')]
summary={'instructions':len(results),'scenes':len(scenes),'grounding_hits':sum(x['score'] for x in evaluation['instructions'] if x['expected']=='plan'),
 'plan_instructions':sum(x['expected']=='plan' for x in truth),
 'nonplan_correct':sum(x['score'] for x in evaluation['instructions'] if x['expected']!='plan'),
 'nonplan_instructions':sum(x['expected']!='plan' for x in truth),
 'plan_answers_on_plan_instructions':sum(x['expected']=='plan' and x['answer']=='plan' for x in evaluation['instructions']),
 'false_nonplan_answers':sum(x['expected']=='plan' and x['answer']!='plan' for x in evaluation['instructions']),
 'wrong_pixels_on_plan_answers':sum(x['expected']=='plan' and x['answer']=='plan' and x['score']==0 for x in evaluation['instructions']),
 'schema_errors':sum(bool(x.get('error')) for x in evaluation['instructions']),
 'request_errors':sum('error' in x for x in calls),
 'truncated_responses':sum(x.get('finish_reason')=='length' for x in calls),
 'rejected_planner_attempts':len(rejected),
 'automatic_score':evaluation['automatic_score'],
 'annotated_targets':sum(x['annotated_targets'] for x in scenes),
 'matched_annotated_targets':len(annotations),
 'raw_center_hits':sum(x['raw_center_hit'] for x in annotations),
 'corrected_center_hits':sum(x['corrected_center_hit'] for x in annotations),
 'raw_label_errors':sum(not x['raw_label_correct'] for x in annotations),
 'final_label_errors':sum(not x['final_label_correct'] for x in annotations),
 'median_raw_y_error_px':float(np.median([x['raw_y_center_error_px'] for x in annotations])),
 'discarded_proposals':len(discarded),
 'category_results':{c:{'correct':sum(x['score'] for x in evaluation['instructions'] if x['category']==c),'total':sum(x['category']==c for x in evaluation['instructions'])} for c in sorted({x['category'] for x in truth})}}
audit={'summary':summary,'scenes':scenes,'failures':failures,'annotated_detection_audit':annotations,'discarded_proposals':discarded,'rejected_planner_attempts':rejected}
(DATA/'audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(summary,indent=2))
print('Failures:')
for r in failures: print(json.dumps(r))
print('Rejected attempts:')
for r in rejected: print(json.dumps(r))
