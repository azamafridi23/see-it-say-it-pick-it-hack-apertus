# `path_b`: architecture

Path B uses **Apertus as the only learned model, and as the main detector**:
Apertus finds and names every object. OpenCV only helps by measuring (sizes,
gaps, trays, which object lies on top). It never decides what is an object.
See [docs/adr/0001-opencv-allowed-in-path-b.md](../../docs/adr/0001-opencv-allowed-in-path-b.md).

"No crop" means Apertus sees the **whole photo once per scene** and boxes
every object. Earlier versions cropped one close-up per OpenCV blob. The
planner is `path_a`'s, ported: the same prompt, schema, checker and
retries, fed with a scene graph built from Apertus's boxes instead of SAM 3.

The code is in this folder, one module per job:

| File | Job |
|---|---|
| `pipeline.py` | wiring and the command line only |
| `detect.py` | steps 1–2: Apertus finds the objects; boxes corrected into objects with a target pixel |
| `measure.py` | step 3: OpenCV measuring (sizes, gaps, tray, what lies on top) |
| `planner.py` | steps 4, 6, 7: scene graph, Path A's planner, checker |
| `safety.py` | step 5: the pre-check |
| `apertus.py` | `ApertusClient`: retries, calls in flight; and a safe `ask()` |
| `workcell.ini` | **every value particular to this setup**: object kinds, the box correction, colour hues, stacking, tray and part-size thresholds, each with where it comes from |
| `workcell.py` | loads `workcell.ini` into a `Workcell` dataclass; defines `DetectedObject`, the record of one found object |
| `paths.py` | puts the organisers' `starter` and `baseline` on the path |
| `tests/test_pipeline.py` | the tests, at the `run()` seam and the client; offline, no API key needed |

The older designs in the results table (OpenCV colour blobs, close-up
naming, the voting Decide stage) were removed from the code once this
design beat them; they are in the git history at commit `8ce5000`.

## Flow

```
photo ──► 1 Detect (Apertus) ──► 2 Correct (code) ──► 3 Measure (OpenCV) ──► 4 Scene graph (code)
                                                                                    │
instruction ──► 5 Pre-check (Apertus) ──► 6 Planner (Apertus) ◄─────────────────────┘
                                                │  ▲
                                                ▼  │ error, up to 2 retries
                                          7 Checker (code) ──► target_pixel + moves
```

Steps 1–4 run once per scene. Steps 5–7 run once per instruction, and a
scene's instructions run in parallel. The whole dev set takes under 3
minutes on the hosted 70B.

| Step | Who | What it does |
|---|---|---|
| 1 Detect | **Apertus** | Finds every object: a name and a box |
| 2 Correct | code | Fixes the boxes' y offset; the box centre is the target pixel |
| 3 Measure | OpenCV + calibration | Size in mm, gaps, which tray, which object lies on top |
| 4 Scene graph | code | Ids, types, `pickable`, `in`, `on` / `covered_by`, `touching`, gaps |
| 5 Pre-check | **Apertus** (text only) | Refuses unsafe or self-contradicting sentences |
| 6 Planner | **Apertus** (text only) | plan / ask / refuse, as moves `{object, to}` |
| 7 Checker | code | Rejects rule-breaking answers and sends the error back |

## 1 Detect (Apertus)

One call per scene sends the whole photo (longer side 1024 px) and asks for
every object of these kinds:

> red cube, red gear, yellow gear, mug, mandarin, banana, plate, saucer, grey
> part, white part, small white part, hand

Each object comes back with a bounding box in **coordinates normalised to
0–1000**. Code converts them to pixels: `x_px = x / 1000 × W`,
`y_px = y / 1000 × H`.

- **Loose JSON.** Apertus's reply is only required to be some JSON object. A
  strict schema made it stop early: it found 4 of 6 objects on scene 08.
- **Filters in code:**
  - kinds that are not on the list are dropped (it once answered "black part");
  - boxes covering more than 25% of the photo are dropped (those are the
    trays; the plate, the largest object, covers about 12%).
- **Every remaining box is one object.**

On the 10 dev scenes the boxes found 50 of 52 objects, including objects
that touch or lie on each other: the cube on two gears, the gear under the
mug. A colour-blob detector merges or misses those.

## 2 Correct (code)

Apertus's boxes sit too high. Fitted against the organisers' ground-truth
masks (32 objects, 10 scenes):

```
true y = 1.065 × Apertus y + 40 px     median edge error 103 → 18 px
true x = Apertus x                      median edge error 17 px
```

The corrected **box centre is the target pixel**. On the dev scenes it lands
on an object of the right kind 29 of 32 times. The misses are two grey parts
that Apertus called "small white part", and a gear hidden under a mug, whose
centre falls on the mug.

## 3 Measure (OpenCV + calibration)

The camera calibration (`pixel_to_table`) turns pixels into millimetres on
the table. For each object:

- **size in mm:** the box's shorter side;
- **gaps** to every other object, box edge to box edge;
- **tray:** the black mat is the largest dark region. An object right of its
  left edge is on `black_tray`, any other is in `grey_tray`.
- **lies on:** where two boxes overlap by at least 40% of the smaller one,
  OpenCV compares colours:
  - each object's own colour: the median Lab colour of the part of its box no
    other box covers;
  - the overlap's colour.

  The object whose colour is nearer the overlap's lies on top. If both are
  about equally near (within 15 Lab units), as with the red cube on red
  gears, they are left as `touching`. On the dev scenes, real stacks overlap
  by 46–91% and neighbours side by side by at most 33%.

Apertus itself was asked for "what lies on what" too. It got most cases
wrong, and asking made its boxes worse, so this is measured instead.

## 4 Scene graph (code)

The objects become the JSON that `path_a`'s planner reads. One object
(illustrative values):

```json
{"id": "red_gear_1", "type": "red gear", "colour": "red",
 "position": [503, 1053], "size_mm": 88, "pickable": true,
 "in": "grey_tray", "covered_by": ["mug"],
 "nearest_gap_mm": [["mug", 0], ["red_cube", 54], ["yellow_gear", 190]]}
```

- **ids:** the type, numbered left to right when there are several
  (`red_gear_1`, `red_gear_2`).
- **pickable:** false for plates, saucers and hands. The organisers confirmed
  that only the plate and the saucer are too wide for the gripper; gears and
  mugs measure about 90 mm but can be picked. The two trays are added as
  places, with `pickable` false.
- **on / covered_by:** from step 3.
- **touching:** objects with a 0 mm gap that aren't already `on` or
  `covered_by`. Path A gets the direction from a depth model (`maybe_on`);
  Path B has no depth, so it only says the two touch.
- **touching_hand:** within 3 mm of an object named `hand`.
- **left_to_right, top_to_bottom, nearest_gap_mm:** so Apertus can read off
  "leftmost" or "closest to" without doing arithmetic.

## 5 Pre-check (Apertus, text only)

One short call per instruction, on the sentence alone, before the planner.
Apertus refuses if the sentence:
- asks for an uncontrolled movement (throwing, pushing, sliding, knocking,
  pulling out from under);
- takes something from a person's hand;
- needs a plate, saucer or tray picked up;
- contradicts itself.

Otherwise it answers "go on", even when the sentence is vague. That way a
"which one?" question can't hide an unsafe verb.

The reply puts the reason first and the decision last: after an enum field,
Apertus pads with whitespace instead of writing the next field. On its own,
the pre-check refused 5 of the 6 sentences it should, and none of the 40 plan
instructions.

## 6 Planner (Apertus) and 7 Checker (code)

Ported from `pipelines/path_a/planner.py`. The prompt explains:
- the scene description;
- what the robot can do: one move at a time, and objects that cover the
  target are moved aside first;
- how to read the instruction, with the verb first;
- when to plan, ask or refuse.

Its examples use invented objects. The reply is constrained:

```json
{"thinking": "...", "decision": "plan | ask_user | refuse",
 "moves": [{"object": "<id>", "to": "<id or null>"}], "message": "..."}
```

The object ids are an enum of this scene's ids, so Apertus can't name an
object that isn't there. The planner sees only the scene graph, not the
photo, and runs at temperature 0 with no voting.

The checker rejects an answer when:
- a moved object is not pickable;
- a hand is on it;
- something still covers it;
- it is put down on itself;
- it is left in the gripper before the next move;
- it is handed to a person.

The error goes back to Apertus ("red_gear_1 is covered by mug. Move mug
aside first, or decide refuse…"), up to 2 times. After that, the answer is a
refusal.

Each move becomes `pick(object)` then `place(to)`. The target pixel is the
first picked object's corrected box centre.

## Switches

| Switch | Effect |
|---|---|
| `--workcell PATH` | another workcell's settings (default: `workcell.ini`); a new table or camera needs a new file, not code changes |
| `--no-safety` | no pre-check before planning |
| `--no-widths` | leave the measured sizes out of the scene graph |
| `--no-gaps` | leave the measured gaps out (no `nearest_gap_mm`, no `touching`) |
| `--thinking` | the planner first thinks in Apertus's thinking mode, then answers (off: worse and 7× slower, see below) |

## Results on the 10 public scenes

Hosted `apertus-v1.5-70b`, scored with `starter/evaluate.py`:

| Variant | Grounding (35) | Safety (30) | Script total (65) |
|---|---|---|---|
| Organisers' baseline | 19.2 | 2.5 | 21.7 |
| OpenCV blobs, close-up names, Decide | 21.0 | 16.8 | 37.8 |
| OpenCV blobs named by boxes, Decide | 23.6 | 13.9 | 37.6 |
| + pre-check | 23.6 | 15.7 | 39.4 |
| + Path A's planner | 25.4 | 18.0 | 43.4 |
| **Apertus finds the objects (this pipeline)** | **28.0–29.8** | **23.6–24.3** | **51.6–54.0** |
| same, with `--thinking` | 27.1 | 21.6 | 48.7 |

The pipeline row is three runs (54.0, 53.2, 51.6): the endpoint answers a
little differently from run to run. Best run by category: simple 88%,
relative 93%, multistep 70%, ambiguous 100%, impossible 75%, unsafe 100%.
Asked or refused right: 9 of 10. Plans refused by mistake: 4 of 40.

**Thinking mode.** Apertus 1.5 can think before it answers
(`chat_template_kwargs: {"enable_thinking": true}`); a JSON schema makes it
skip the thinking, so `--thinking` thinks in free text first and passes its
conclusion to the normal JSON call. It helped multistep instructions (90%)
but broke simple ones: the thinking settled on a wrong plan, or found
"several objects to the left" and asked. A third of the thinking calls
ran past 4,000 tokens, and the run took 17 minutes instead of 2.5.

**Caveat:** the box correction and both stacking thresholds were fitted on
these same 10 scenes, so the held-out score will be lower. Box detection
also varies a little between runs (the scene 08 white part was missing in
one run).

## Known weaknesses

The 7 instructions still lost:

| Instruction | Cause |
|---|---|
| 04 i1 pick up the white part | Apertus calls the grey parts "small white part" too, so it asks which one |
| 05 i4 move everything… | Apertus invented a "small white part" box on the robot rail at the photo's bottom edge, and the plan starts with it |
| 06 i4 the red gear under the yellow gear | asked which red gear |
| 08 i2 put the small white part on the plate | the box centre missed the part |
| 09 i3, 09 i4 (the grey part) | Apertus didn't find the flat grey part on the grey tray |
| 09 i5 put it on the plate and leave it where it is | the contradiction was missed |

Possible fixes:
- drop boxes touching the photo's bottom edge (the rail);
- a measured hint for grey vs white parts;
- move a box-centre pixel that falls off the object onto the object, with
  OpenCV helping inside the box.

The full log of Apertus's weaknesses, with evidence, is in
[docs/apertus-difficulties.md](../../docs/apertus-difficulties.md).

## Run

```
pip install -r requirements.txt
python pipelines/path_b/run.py --data dataset --out results/path_b.json
python starter/evaluate.py results/path_b.json --data dataset
```

The server, model and key are read from `API_BASE_URL`, `API_MODEL` and
`APERTUS_API_KEY`, in the environment or in the repository's
`.env` file (see `.env.example`); `--base-url` and `--model` override the
first two.
`--data` takes any folder of scenes, so the held-out run is the same command
with `--data heldout`. With no server set
anywhere, it uses the hosted endpoint and `apertus-v1.5-70b`, as Path A does. The tests run offline, without a key:

```
python -m unittest discover pipelines/path_b/tests
```
