# Your scenes

Put new photos here and run `make run` (our submission, Path A) or
`make run-b` (Path B). The answers go to `results/scenes_path_a.json` or
`results/scenes_path_b.json`. Another folder works too:
`make run DATA=path/to/folder`.

One folder per scene, the same layout as `heldout/`:

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
