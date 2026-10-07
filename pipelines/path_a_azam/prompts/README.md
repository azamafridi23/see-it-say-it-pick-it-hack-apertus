# Planner prompts

The rules Apertus is given before it reads a scene graph and an instruction. One text file per version.

    python pipelines/path_a_azam/run.py --data dataset --out results/try_v2.json --prompt planner_v2

Without `--prompt`, the default is used: `planner_v1` (set by `DEFAULT_PROMPT` in `planner.py`).
The version used is written to the log next to every answer.

To add a version: copy the latest file to `planner_v<N>.txt`, edit it, and add a row below. Keep old versions.
The reply format (thinking, decision, moves, message) is fixed by `planner.py`, so a prompt must describe that format.

| Version | What changed | Script score on the public scenes |
|---|---|---|
| `planner_v1` | First version | 60.5 / 65 (one run) |

Apertus's answers vary a little between runs, and one instruction is worth about a point: run a version two
or three times before calling it better. Choosing between many versions on the 50 public instructions fits
the prompt to the public answers; the held-out scenes are the real test.
