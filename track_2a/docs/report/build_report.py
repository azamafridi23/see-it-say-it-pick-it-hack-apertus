"""Build a one-page report with evidence crops and a reusable English source."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.utils import ImageReader
from reportlab.lib.enums import TA_LEFT

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
audit=json.loads((OUT/'data/audit.json').read_text())
meta=json.loads((OUT/'data/metadata.json').read_text())
font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
font=ImageFont.truetype(font_path,36)
small=ImageFont.truetype(font_path,30)
RED='#ff4a58';GREEN='#33e393';CYAN='#45e1ff';YELLOW='#ffd449'

def crop(scene,box):
    image=Image.open(ROOT/f'dataset/{scene}/rgb.png').convert('RGB')
    image=image.crop(box).resize((900,600))
    draw=ImageDraw.Draw(image)
    sx=900/(box[2]-box[0]);sy=600/(box[3]-box[1])
    def xy(p):return ((p[0]-box[0])*sx,(p[1]-box[1])*sy)
    def rect(b,color,width=4):
        draw.rectangle([xy(b[:2]),xy(b[2:])],outline=color,width=width)
    def label(text,p,color):
        x,y=xy(p); bb=draw.textbbox((x,y),text,font=font)
        draw.rectangle((bb[0]-5,bb[1]-4,bb[2]+5,bb[3]+4),fill='#10202c')
        draw.text((x,y),text,font=font,fill=color)
    def cross(p,color):
        x,y=xy(p);draw.line((x-10,y-10,x+10,y+10),fill=color,width=5);draw.line((x-10,y+10,x+10,y-10),fill=color,width=5)
    return image,draw,xy,rect,label,cross

# Green boxes are actual ground-truth mask extents, not model output.
item=next(x for x in audit['annotated_detection_audit'] if x['scene']=='scene_08' and x['ground_truth']=='yellow_gear')
im,d,xy,rect,label,cross=crop('scene_08',(620,1100,1160,1460))
rect(item['raw_box'],RED);rect(item['gt_box'],GREEN)
cross(item['raw_center'],RED);cross(item['corrected_center'],CYAN)
label('Raw: background',(630,1110),RED)
label('Corrected: on gear',(630,1420),CYAN)
im.save(FIG/'01_coordinate_error.png')

im,d,xy,rect,label,cross=crop('scene_08',(510,320,1260,820))
rect([698,482,1029,711],GREEN)
label('No model proposal',(530,760),GREEN)
im.save(FIG/'02_missing_small_part.png')

# Yellow dashed outline is a manually highlighted visible support region;
# the grey part has no ground-truth mask in this scene.
im,d,xy,rect,label,cross=crop('scene_09',(250,580,1360,1320))
b=[380,680,1070,1000]
p1,p2=xy(b[:2]),xy(b[2:])
for y in (p1[1],p2[1]):
    for x in range(int(p1[0]),int(p2[0]),26):d.line((x,y,min(x+14,p2[0]),y),fill=YELLOW,width=4)
for x in (p1[0],p2[0]):
    for y in range(int(p1[1]),int(p2[1]),26):d.line((x,y,x,min(y+14,p2[1])),fill=YELLOW,width=4)
label('Grey support omitted',(280,1210),YELLOW)
raw=next(x for x in audit['annotated_detection_audit'] if x['scene']=='scene_09' and x['ground_truth']=='red_cube')
rect(raw['raw_box'],RED)
im.save(FIG/'03_missing_grey_part.png')

content={
 'title':'Apertus failure modes in Path B',
 'method':'Fresh run on all 10 development scenes and 50 instructions, 14 October 2026. Model: apertus-v1.5-70b at the hosted hackathon endpoint; temperature 0, 1024-pixel image limit, 800-token replies, thinking off. Standard Path B uses Apertus for detection, safety and planning, plus OpenCV measurements and its existing coordinate correction. No SAM 3 or scene-graph cache was used.',
 'captions':[
 'Scene 08: yellow gear. Raw centre (869, 1285) is on background; corrected centre (869, 1408) is on the gear.',
 'Scene 08: the small white part is visible inside the ground-truth box but has no detection. Instruction 08_i2 is falsely refused.',
 'Scene 09: grey support beneath the cube is omitted. Missing graph evidence causes 09_i3 to ask and 09_i4 to refuse.'
 ],
 'findings':[
 ('Coordinate bias and naming', 'Across 31 spatially matched annotated targets, raw box centres miss 7 masks (24/31 hits); corrected centres hit 31/31. Median raw vertical centre error is -100.5 px relative to mask-box centres. The existing y\' = 1.065y + 40 correction masks this weakness. In scene 04, all three parts are called "small white part"; size measurements fix their types. Eight tray-sized proposals are discarded.'),
 ('Missing objects propagate into refusals', 'The detector omits the small part in scene 08 and the partly occluded grey part in scene 09 (figures above). These omissions explain three of the five false ask/refuse answers. The small part is the only unmatched object among the 32 mask-annotated targets; the grey support is not mask-annotated.'),
 ('Instruction semantics fail despite valid JSON', '08_i1 refuses "the gear in the grey tray" while explicitly acknowledging a yellow gear there. 09_i5 executes pick(mandarin), place(plate), ignoring "and leave it where it is"; ground truth requires refusal. All 50 final answers pass the schema, so valid structure does not guarantee semantic correctness.'),
 ('Spatial ambiguity also reflects pipeline limits', '10_i3 asks which red gear is closest to the mandarin. Both rounded gap measurements are 0 mm in the graph, although ground truth selects red_gear_1. This is an upstream measurement ambiguity; the run does not isolate it as a model-only reasoning failure.')
 ],
 'limits':'One run on development data; the existing coordinate correction was fitted to these scenes. Detection audit: greedy one-to-one matching by corrected box IoU &gt;= 0.25, independent of class. Only 32 objects have masks; no full-scene recall estimate. Grounding scores check the first target, not every action or destination. No Path A comparison. All 105 model calls completed without transport errors or token truncation.',
 'recommendation':'Retain mask-aware localization checks, review omitted/occluded objects, and validate contradictions and spatial ties explicitly before robot execution.'
}
(OUT/'report_content.json').write_text(json.dumps(content,indent=2))
md='# '+content['title']+'\n\n'+content['method']+'\n\nGrounding: 35/40; non-plan decisions: 9/10; official automatic score: 54.25/65. Run time: 160.1 s.\n\n'
for i,(filename,caption) in enumerate(zip(['01_coordinate_error.png','02_missing_small_part.png','03_missing_grey_part.png'],content['captions']),1):
    md+=f'![Figure {i}](figures/{filename})\n\n{caption}\n\n'
for title,body in content['findings']:md+=f'**{title}.** {body}\n\n'
md+='**Limitations.** '+content['limits']+'\n\n**Implication.** '+content['recommendation']+'\n'
(OUT/'apertus_failure_modes.md').write_text(md)

W,H=595.276,841.89
c=canvas.Canvas(str(OUT/'apertus_failure_modes.pdf'),pagesize=(W,H))
c.setTitle(content['title']);c.setAuthor('Path B experiment audit')
c.setFillColor(HexColor('#ffffff'));c.rect(0,0,W,H,fill=1,stroke=0)
NAVY=HexColor('#152e43');GRAY=HexColor('#536574');LIGHT=HexColor('#edf3f7');TEAL=HexColor('#006e74')
LEFT=34;WIDTH=W-2*LEFT
styles={
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10.2,leading=13.7,textColor=NAVY),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.5,leading=11,textColor=GRAY),
 'caption':ParagraphStyle('caption',fontName='Helvetica',fontSize=8.7,leading=11.2,textColor=NAVY),
}
def para(text,x,y,width,style='body'):
    p=Paragraph(text,styles[style]);w,h=p.wrap(width,1000);p.drawOn(c,x,y-h);return y-h

c.setFillColor(NAVY);c.rect(0,H-9,W,9,fill=1,stroke=0)
c.setFont('Helvetica-Bold',21);c.drawString(LEFT,H-44,content['title'])
c.setFont('Helvetica',9);c.setFillColor(GRAY);c.drawString(LEFT,H-60,'10 SCENES  /  50 INSTRUCTIONS  /  FRESH PATH B EXPERIMENT')
y=para(content['method'],LEFT,H-75,WIDTH)-12

metrics=[('35 / 40','Correct first-target grounding'),('9 / 10','Correct ask / refuse decisions'),('54.25 / 65','Official automatic score')]
gap=10;tile=(WIDTH-2*gap)/3
for i,(value,label_) in enumerate(metrics):
    x=LEFT+i*(tile+gap)
    c.setFillColor(LIGHT);c.roundRect(x,y-49,tile,49,5,fill=1,stroke=0)
    c.setFillColor(TEAL);c.setFont('Helvetica-Bold',20);c.drawString(x+10,y-23,value)
    c.setFillColor(GRAY);c.setFont('Helvetica',8);c.drawString(x+10,y-39,label_)
y-=66
c.setFillColor(NAVY);c.setFont('Helvetica-Bold',11);c.drawString(LEFT,y,'Detection evidence')
y-=15
figwidth=(WIDTH-2*gap)/3;figheight=figwidth*2/3
names=['01_coordinate_error.png','02_missing_small_part.png','03_missing_grey_part.png']
cap_bottom=[]
for i,(name,caption) in enumerate(zip(names,content['captions'])):
    x=LEFT+i*(figwidth+gap)
    c.drawImage(ImageReader(FIG/name),x,y-figheight,width=figwidth,height=figheight)
    cap_bottom.append(para(f'<b>{i+1}.</b> '+caption,x,y-figheight-6,figwidth,'caption'))
y=min(cap_bottom)-7
y=para('Green: ground-truth mask bounds. Red: raw Apertus box/centre. Cyan: corrected centre. Yellow dashes: manual grey-support highlight, not a ground-truth mask.',LEFT,y,WIDTH,'small')-12

c.setFillColor(NAVY);c.setFont('Helvetica-Bold',11);c.drawString(LEFT,y,'Observed failures and attribution')
y-=10
for title,body in content['findings']:
    y=para('<b>'+title+'.</b> '+body,LEFT,y,WIDTH)-7

y=para('<b>Implication.</b> '+content['recommendation'],LEFT,y,WIDTH)-9
y=para('<b>Scope and limits.</b> '+content['limits'],LEFT,y,WIDTH,'small')
assert y>40, f'Content overflows one page: bottom={y}'
c.setStrokeColor(HexColor('#d6e0e8'));c.line(LEFT,29,W-LEFT,29)
c.setFillColor(GRAY);c.setFont('Helvetica',7.2)
c.drawString(LEFT,18,'Evidence: report/data/{metadata, model_calls, evaluation, audit}.json*  |  Runtime: 160.1 s')
c.drawRightString(W-LEFT,18,'1 / 1')
c.showPage();c.save()
print('PDF generated; content bottom:',round(y,1),'pt')
