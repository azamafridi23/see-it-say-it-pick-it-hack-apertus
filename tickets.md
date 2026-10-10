# Tickets: Path B — strongest Apertus-only pipeline

Builds the staged Apertus-only pipeline (Marks → Inventory → Decide → Locate) described in `docs/spec-path-b.md`.

Work the **frontier**: any ticket whose blockers are all done. After the skeleton, tickets 2–6 are parallel.

## 1. Skeleton: one command, scene folder → scored results

**What to build:** one command runs over any folder of scenes and writes a schema-valid results file the official scorer accepts. Basic OpenCV marks with interior pixels; one greedy Inventory call and one greedy Decide call with constrained JSON, plans naming mark ids; Locate by lookup. Supports mock mode, server URL, model name, and a raw-reply log.

**Blocked by:** None — can start immediately.

- [x] One command produces a results file for every instruction in a scenes folder
- [x] Every answer is schema-valid, also when the model replies garbage or times out
- [x] A plan's target pixel is the interior pixel of the first-picked mark, in full resolution
- [x] The pipeline never reads the instruction category
- [x] Pipeline-seam test with a scripted fake model, scored through the official scorer
- [x] Result of the "two images per message" experiment written down — run 2026-10-07 on the hosted endpoint (hackapertus.livemap.sh): both models see images and honour JSON schemas. apertus-v1.5-70b takes two images and compares them. apertus-v1.5-8b accepts two but degenerates into "{" + whitespace under a JSON schema, so run 8b with `--no-clean-image`
- Seen on scene_09 (input for ticket 2): red/orange hue boundary splits the mandarin into 4 marks and a gear into 2; a red gear touching the red cube merges into the cube's blob; blue tape at the image border gets marks

## 2. Mark coverage check and segmentation tuning

**What to build:** a model-free report of the share of the 40 dev targets with a mark inside their ground-truth mask, listing the misses; segmentation tuned until coverage ≥ 90 % or misses are explained.

**Blocked by:** 1

- [x] Coverage report runs without a model
- [x] Coverage ≥ 90 %, or each remaining miss explained — 100 % (40/40)
- Known: 6 harmless duplicates (mug handle ×4 scenes, grey_part_1, small_part ×3), all on the right object. Thresholds were tuned on all 10 scenes, not just 01–07

## 3. Decide: reasoning field and voting

**What to build:** reasoning-before-verdict output, the three-check prompt, invented few-shots; N samples, majority vote, ties to ask/refuse, most common plan sequence. Each a switch.

**Blocked by:** 1

- [x] Vote and tie rule covered by the pipeline-seam test
- [x] Switches off reproduces the skeleton's behaviour (`--votes 1 --no-reasoning`)
- Not yet measured against the real Apertus: whether vLLM's JSON-schema decoding keeps `reasoning` before `verdict`, and how much voting helps

## 4. Locate: grid zoom fallback

**What to build:** when the first pick is unmarked, Apertus narrows labelled grid cells over 2–3 crop rounds; the result is a full-resolution pixel. A switch.

**Blocked by:** 1

- [x] Unmarked-target path covered by the pipeline-seam test
- [x] Pixel returned in full resolution (`--no-grid-zoom` switches it off)
- Known: an object spanning several cells makes "which cell holds the middle" a close call; zoom calls are not in the raw log yet (ticket 7)

## 5. Width facts in the Inventory

**What to build:** each mark's minimum width in mm via the table projection, appended to the Inventory as a fact. A switch.

**Blocked by:** 1

- [x] ~~The dev plate measures wider than 85 mm, gears and cubes narrower~~ revised: only objects > 110 mm (85 mm + 30 %) are called too wide. Table-plane widths read up to ~15 % wide for tall/stacked objects (mugs 85–87, a stacked cube 86, the white part 96, all picked in dev); saucers ~120 and plates ~180 are flagged. `--no-widths` switches it off

## 6. Marked + clean image

**What to build:** send both images per call when the server supports it, else marked only. A switch. (No longer dropped if two images are unsupported: the retry covers that.)

**Blocked by:** 1

- [x] Works with and without two-image support: on by default; if a two-image call fails, it is retried with the marked image only (vLLM needs `--limit-mm-per-prompt` for two). `--no-clean-image` sends one image and skips the retry

## 7. Ablations and failure breakdown

**What to build:** a raw-pixel switch (no marks); one command running each switch configuration, scoring scenes 01–07 and 08–10 separately; a per-stage failure table (missed mark, wrong verdict, wrong object, grid zoom miss).

**Blocked by:** 2, 3, 4, 5

- [ ] Ablation table for tune and check splits
- [ ] Per-stage failure table

## 8. Submission readiness

**What to build:** README reproduction section (< 15 min), the one command verified on a fresh scenes folder (held-out dry run), Apache 2.0 check, final results file.

**Blocked by:** 7

- [ ] A fresh clone reproduces the dev score following the README
