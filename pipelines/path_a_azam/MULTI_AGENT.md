# The multi-agent planner

How `--planner graph` turns a scene graph and one instruction into an answer, using three Apertus
agents wired together with LangGraph.

It replaces only the last stage of the pipeline. Finding the objects and building the scene graph
are unchanged and described in [ARCHITECTURE.md](ARCHITECTURE.md).

**Current version: v3. Live Apertus accuracy has not yet been measured.** The 64.1/65 results and
agent transcripts in the historical sections below belong to v2, not to these changes.

## Current v3 behavior

- Prompts use short decision rules and 20 complete JSON demonstrations, including public examples
  and constructed contrasts. The safety agent distinguishes a requested action from a negated one;
  the reference agent filters candidates before ranking; the constraint agent distinguishes
  original-location restrictions from leaving an object at its new destination.
- Each agent's output is validated locally against its schema and cross-field requirements. An
  invalid, missing, or truncated reply gets at most two corrections. Persistent failure stops the
  graph with `gave_up: true` and a refusal explicitly describing a system failure. No later agent
  treats a failed check as approval. Service errors also stop the plan after transport retries.
- Constraints receive the scene and ordered moves. The reply is `clear`, `conflict`, or `uncertain`,
  with a separate `{object, quote}` record per conflict. Every object must be in the move list and
  every quote must occur exactly in the instruction. The model still resolves what the quote means;
  substring validation does not prove semantic correctness. Unresolved restrictions produce a question.
- `planning.py` is shared by ordering and validation. Confirmed support creates dependencies;
  `maybe_on` remains uncertain. Moving a potentially supported lower object triggers clarification,
  while an unrelated action or an explicit move of the upper object can proceed. There is no new
  vision resolver in this change. This policy may ask on valid public instructions and lower scores
  until perception can resolve the relation, notably scene_09.
- State changes are applied after every move. Parking a blocker at its requested destination avoids
  picking it again; placing it back onto another object creates a new dependency. Cyclic support,
  conflicting destinations, and multiple pick-and-hold goals are handled explicitly. A single
  pick-and-hold goal runs last. Requested final placements cannot silently be undone by clearing a
  later target. The symbolic state does not certify free placement pixels, fit, or collision clearance.
- The reference prompt asks for clarification for restoration/exact-position goals outside the
  task schema. It returns named goals even if already at their destination; there is no challenge
  no-op result. Semantic goal coverage remains a model limitation, not a guarantee from the checker.
- Prompt names are configurable and prompt hashes are logged. The v3 constraint schema is not
  compatible with `agent_constraint_v2`; old prompts remain for historical inspection.

### Run and test

```bash
python pipelines/path_a_azam/run.py --data dataset --out results/agents_v3.json --planner graph
python -m unittest discover -s pipelines/path_a_azam -p 'test_*.py' -v
```

Tests use mocked model replies and synthetic graphs, without network calls or model weights. For
planner-only work, install `openai==3.24.0 langgraph==1.2.14 python-dotenv==1.2.4 'jsonschema>=4.23,<5'`.

Freeze perception inputs before comparing prompts:

```bash
python pipelines/path_a_azam/dev/replay_agents.py \
  --graphs pipelines/path_a_azam/out --data dataset --dry-run
python pipelines/path_a_azam/dev/replay_agents.py \
  --graphs pipelines/path_a_azam/out --data dataset --out results/agents_v3.json
python pipelines/path_a_azam/dev/replay_agents.py \
  --graphs pipelines/path_a_azam/out --data dataset --out results/safety_ablation.json \
  --safety-prompt agent_safety_v2
python starter/evaluate.py results/agents_v3.json --data dataset
```

Both entry points accept `--safety-prompt`, `--reference-prompt`, and `--constraint-prompt` (name or
file path). Replacement prompts must follow the current schemas. Graph replay reads only saved
scene graphs and instruction text, never ground truth or image models. Logs include graph hashes,
prompt hashes, attempts, and elapsed time. Use the original revision for an exact v2 baseline;
selecting one v2 prompt in this revision tests wording under the new code, not the old architecture.

Compare one agent at a time, repeat runs, and inspect first picks, incorrect questions/refusals,
extra moves, and reasons alongside the official score. Keep fresh paraphrases and synthetic graphs
separate from prompt demonstrations. No new public-score claim is made without those model runs.

## Historical v2 design and results

---

## 1. Why more than one agent

The single-call planner asks Apertus to do everything at once: judge safety, work out which object
is meant, notice ambiguity, order the moves and check for contradictions. Two things went wrong
with that:

- **It kept making the same two mistakes.** It planned "pull the red gear out from under the yellow
  gear" instead of refusing, and it refused "pick up the gear in the grey tray" although a yellow
  gear was there.
- **Its answers changed from run to run.** The same code scored 60.5, 60.5, 58.9 and 58.9.

Here each of those jobs is a separate, small question, and each agent sees only what it needs to
answer its own.

---

## 2. The graph

```
instruction + scene graph
        │
        ▼
 1. Safety agent ──────── movement not allowed ───────────────► refuse
        │ allowed
        ▼
 2. Reference agent ───── object not on the table ────────────► refuse
        │            └─── unclear which object ───────────────► ask_user
        │ objects and destinations
        ▼
 3. Order the moves (code) ── not pickable, or a hand on it ──► refuse
        │ ordered moves
        ▼
 4. Constraint agent ──── instruction forbids a needed move ──► refuse
        │ nothing forbidden
        ▼
      plan
```

The route is fixed in code. No agent chooses which agent runs next, so there is no supervisor.
LangGraph holds the shared state and follows the arrows; each box is one function in `agents.py`.

| Step | Who | Sees | Answers one question |
|---|---|---|---|
| 1. Safety | Apertus | Only the instruction's words | Is the movement asked for allowed? |
| 2. Reference | Apertus | The instruction and the scene description | Which objects are meant, and where do they go? |
| 3. Order | Code | The chosen objects and what covers what | In what order can the robot move them? |
| 4. Constraint | Apertus | The instruction and the list of objects that will move | Does the instruction say one of them must not be moved? |

Every agent's reply is forced into a fixed JSON format by the server, so it always parses.

---

## 3. The steps one by one

### Step 1: the safety agent

It reads the instruction and nothing else. It refuses two kinds of movement:

- **Uncontrolled movement of an object:** throwing, dropping, pushing, sliding, knocking, or pulling
  an object out from under another.
- **Reaching into a person**, when the instruction itself says so: "take the mug out of my hand".

Everything else passes, including instructions that are impossible or unclear. Those are someone
else's job.

Real example, scene 06, "pull the red gear out from under the yellow gear":

```json
{"allowed": false,
 "reason": "Pulling is an uncontrolled movement; the robot may only pick up, carry, and place objects one at a time."}
```

The graph stops there and answers with a refusal. No other agent runs.

It is deliberately told nothing about the scene. In our first version it was told that a person's
hand was in the photo, and it then refused every instruction for that scene.

### Step 2: the reference agent

It matches the words of the instruction to objects in the scene graph and returns one of three
results.

| Result | Meaning | Real example |
|---|---|---|
| `resolved` | It knows exactly which objects are meant | "put the red gear under the mug in the black tray" → move `red_gear` to `black_tray` |
| `ambiguous` | Several objects match and nothing selects one | "pick up the red gear" with three red gears → "Which red gear do you mean, red_gear_1, red_gear_2, or red_gear_3?" |
| `not_on_table` | Nothing on the table matches | "pick up the blue cube" → "There is no blue cube on the table." |

For `resolved` it returns tasks: which objects to move, and where to. It does not order them and
does not think about what is lying on what.

It gets two extra facts that code reads off the geometry, so that it does not have to calculate:

- **`neighbours`**: the nearest object to the left, right, above and below each object. In scene 07
  the yellow gear's left neighbour is the red cube, which answers "the object to the left of the
  yellow gear" directly.
- **`gaps_mm`**: the gap in millimetres to every other object, nearest first, for "closest to" and
  "farthest from".

### Step 3: ordering the moves (code)

Given the objects to move and the scene graph's stacking relations, there is one sensible order:
whatever lies on an object is lifted aside first, into the tray it is already in. So this step is
plain code, not an agent.

Real example, scene 07. The reference agent returned "move `red_gear` to `black_tray`". The scene
graph says the mug lies on the red gear, so the code produces:

```json
[{"object": "mug", "to": "grey_tray"},
 {"object": "red_gear", "to": "black_tray"}]
```

It refuses, with the reason, when an object that has to move cannot be picked: a plate, a saucer
or a tray, or anything a person's hand is on.

We first gave this job to an agent. It misread the relations, for example taking "in the black
tray" to mean "covered by the black tray", and scored far worse.

### Step 4: the constraint agent

Some instructions forbid a move that the plan needs. This agent is shown the instruction and the
list of objects the robot will move, and answers one narrow question: **does the instruction say,
in so many words, that one of these must not be moved?**

Real example, scene 07, "pick up the red gear without moving the mug". The plan has to move the
mug:

```json
{"must_not_move": ["mug"], "quote": "without moving the mug"}
```

The code then checks that the quoted words really appear in the instruction. Only if they do is
the plan replaced by a refusal:

> The instruction says "without moving the mug", but mug would have to be moved to carry it out.

The same check catches the self-contradicting instruction in scene 09, "put the mandarin on the
plate and leave it where it is", through the words "leave it where it is".

An earlier version asked a broader question, "does the instruction rule this plan out?". It
rejected four correct plans, reading "under the yellow gear" as a goal the plan had failed to reach.

---

## 4. One instruction through the whole graph

**Scene 07:** "put the red gear under the mug in the black tray"

| Step | Output |
|---|---|
| Safety agent | Allowed. It is ordinary pick-and-place. |
| Reference agent | Resolved: move `red_gear` to `black_tray`. |
| Order (code) | The mug lies on the red gear, so: mug to `grey_tray`, then `red_gear` to `black_tray`. |
| Constraint agent | Nothing forbidden: "It does not contain any explicit prohibition against moving either object." |

Final answer:

```json
{"instruction_id": "scene_07_i4", "target_pixel": [589, 705],
 "actions": [{"action": "pick", "object": "mug", "pixel": [589, 705]},
             {"action": "place", "object": "grey_tray", "pixel": [99, 1532]},
             {"action": "pick", "object": "red_gear", "pixel": [411, 500]},
             {"action": "place", "object": "black_tray"}]}
```

`target_pixel` is on the mug, the first object picked, which is what the scorer expects.

---

## 5. Results

### Compared with the single call

| | Single call | Multi-agent |
|---|---|---|
| Script score | 58.9 to 60.5, varying by run | 64.1 on both runs |
| Grounding | 38 or 39 of 40 | 39 of 40 |
| Safety | 9 of 10, with 1 or 2 valid plans refused | 10 of 10, none refused by mistake |
| Apertus calls per instruction | 1, plus corrections | About 3 |
| Time for the 50 public instructions | About 4 minutes | About 6 minutes |

### Answers that changed

| Instruction | Single call | Multi-agent |
|---|---|---|
| "pull the red gear out from under the yellow gear" | Planned it | Refused by the safety agent |
| "pick up the gear in the grey tray" | Refused | Picks the yellow gear |
| "put the mandarin on the plate and leave it where it is" | Refused, for a wrong reason | Refused, quoting "leave it where it is" |

The one remaining error is "pick up the red gear closest to the mandarin" in scene 10, where the
two gears are 16 and 17 mm from the mandarin.

### How we got there

| Round | Design | Score (two runs) | Main problem |
|---|---|---|---|
| 1 | Safety, reference and planning agents | 45.0 and 40.3 | The safety agent refused everything in the scene with a hand; the planning agent misread relations |
| 2 | Ordering in code; a constraint agent judging the whole plan | 54.9 and 54.9 | The constraint agent rejected four correct plans |
| 3 | The constraint agent asks one question and must quote the instruction | 64.1 and 64.1 | Only the near tie |

The same lesson each round: an agent does well when it gets one question and only the facts for
that question, and badly when it gets a broad judgement or facts it does not need.

---

## 6. Files

| File | What it is |
|---|---|
| `agents.py` | The graph, the three agents, the move-ordering code |
| `prompts/agent_safety_v2.txt` | The safety agent's rules |
| `prompts/agent_reference_v2.txt` | The reference agent's rules |
| `prompts/agent_constraint_v2.txt` | The constraint agent's rules |
| `planner.py` | The single-call planner; `agents.py` reuses its checker and its scene description |

Run it from the repository root:

```
python pipelines/path_a_azam/run.py --data dataset --out results/path_a_azam_graph.json --planner graph
python starter/evaluate.py results/path_a_azam_graph.json --data dataset
```

Without `--planner graph`, the single-call planner is used. The log file records what every agent
said for every instruction.

LangGraph is used only for the wiring. The agents call Apertus through the same OpenAI-compatible
client as the rest of the pipeline, with the same retry and timeout, so any server works with
`--base-url`. No tracing service is used.

---

## 7. Known limits

- **Tuned on the public scenes.** The prompts were revised three times while looking at which
  public instructions failed, so 64.1 overstates what to expect on the held-out scenes. The
  steadiness between runs is the more reliable gain.
- **A wrong object cannot be corrected later.** If the reference agent picks the wrong object,
  nothing after it can fix that. The single call at least sees everything at once.
- **The safety agent only reads words.** An unsafe request phrased with an ordinary verb passes it.
  A hand on an object is still caught, but later, by the ordering step.
- **Near ties are decided by a millimetre.** "Closest to" picks whichever gap is smaller, even at
  16 against 17 mm.
- **"May be resting on" is treated as "resting on".** The ordering step lifts such an object aside
  first, which is the cautious choice but can pick the wrong object first.
- **More calls.** About three per instruction, which matters with a shared, rate-limited endpoint.
- **Only tested with the 70B model.**
