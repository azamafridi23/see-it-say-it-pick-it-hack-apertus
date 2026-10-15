# Your scenes

Put new photos here and run `make run DATA=data/scenes`: our submission
(Path A) in Docker. The answers go to `results/scenes_path_a.json`.
Without Docker, `make run-a DATA=data/scenes` or `make run-b DATA=data/scenes`
(Path B, answers in `results/scenes_path_b.json`). Another folder works too.

One folder per scene, the same layout as `data/heldout/`:

```
scenes/
  calibration.json        the camera and gripper; this one is the challenge's workcell
  my_scene_01/
    rgb.png               the top-down photo
    instructions.txt      one instruction per line: <category><TAB><instruction>
```

In `instructions.txt`, lines starting with `#` are skipped, and the
category (for example `simple`) is not used by the pipelines. The ids in
the answers are `<scene folder>_i1`, `_i2`, ... in line order.

A folder `ground_truth/` next to the scenes, as in `dataset/`, makes the
run print its score.

Everything in this folder except this file and `calibration.json` is
ignored by git.
