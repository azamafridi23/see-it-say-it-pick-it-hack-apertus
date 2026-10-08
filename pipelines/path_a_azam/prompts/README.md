# Planner prompts

The rules Apertus is given before it reads a scene graph and an instruction. One text file per version.

    python pipelines/path_a_azam/run.py --data dataset --out results/try_v2.json --prompt planner_v2

Without `--prompt`, the default is used: `planner_v2` (set by `DEFAULT_PROMPT` in `planner.py`).
The version used is written to the log next to every answer.

To add a version: copy the latest file to `planner_v<N>.txt`, edit it, and add a row below. Keep old versions.
The reply format (thinking, decision, moves, message) is fixed by `planner.py`, so a prompt must describe that format.

| Version | What changed | Script score on the public scenes |
|---|---|---|
| `planner_v1` | First version | 60.5, 60.5, 58.9 / 65 (three runs) |
| `planner_v2` | Keeps uncertain support distinct from confirmed stacking, matching the shared checker | Not yet measured |

Apertus's answers vary a little between runs, and one instruction is worth about a point: run a version two
or three times before calling it better. Choosing between many versions on the 50 public instructions fits
the prompt to the public answers; the held-out scenes are the real test.

Tried and dropped: a version that explained a new `why_not_pickable` field in the scene description scored 52.8
and 51.4. The field itself was the cause: `planner_v1` with the field also scored 52.8, and 58.9 without it.

## Multi-agent prompts (`--planner graph`)

One file per agent. `PROMPTS` in `agents.py` defaults to v3. Override individual prompts with
`--safety-prompt`, `--reference-prompt`, and `--constraint-prompt` when using `--planner graph`.
Names and file paths are accepted. Every reply is validated locally and retried on invalid content.

| File | Agent | Note |
|---|---|---|
| `agent_safety_v1`, `v2` | Safety | v2 reads only the instruction's words and allows "move everything" |
| `agent_reference_v1`, `v2` | Reference | v2 uses `neighbours` and `gaps_mm` |
| `agent_constraint_v2` | Constraint | One question: must one of the moved objects not be moved? Must quote the instruction |
| `agent_safety_v3` | Safety | Six complete JSON examples; requested vs negated actions; feasibility left to later checks |
| `agent_reference_v3` | Reference | Eight complete JSON examples; filter then rank; uncertain relations remain uncertain |
| `agent_constraint_v3` | Constraint | Six complete JSON examples; scene and moves as context; clear/conflict/uncertain with object-specific quotes |

Scores on the public scenes, two runs each: 45.0 and 40.3 (first version, with a planning agent), 54.9 and 54.9,
then 64.1 and 64.1 with the files above. Revised three times against the public answers.

Those scores are for v1/v2. **v3 has no live-model score yet.** Its 20 demonstrations include public
instructions and constructed contrast cases; do not count performance on demonstrations as validation.

The constraint schema changed in v3: it uses `status`, `restrictions: [{object, quote}]`, `message`,
and `thinking`. A v2 constraint prompt is not a compatible override. Safety and reference retained
their field names, but the current code's uncertainty and validation policies still apply to old prompts.
See [the current planner documentation](../MULTI_AGENT.md#current-v3-behavior) for migration notes,
offline tests, and `dev/replay_agents.py` to compare prompts against frozen scene graphs.
