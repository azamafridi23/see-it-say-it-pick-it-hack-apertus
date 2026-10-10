# Spec: Path B — strongest Apertus-only pipeline

Labels: `ready-for-agent`

## Problem Statement

The challenge asks for a head-to-head between **Path A** (vision model + Apertus) and **Path B** (Apertus alone). The organisers' **Baseline** is already a minimal Path B: one plain prompt, no constrained decoding, raw pixel guesses. It drifts on grounding and has no deliberate safety reasoning. I need the strongest Path B that stays within "Apertus is the only learned model" (ADR 0001), so the comparison with Path A is meaningful and the report can say exactly where Apertus struggled — and which assistance fixed what.

## Solution

A staged pipeline in which Apertus makes every choice and deterministic OpenCV only prepares what it sees and converts its choice into a pixel:

1. **Marks** — OpenCV segments coloured blobs, draws a number on each, and records each mark's deepest interior pixel and minimum width in mm.
2. **Inventory** — Apertus sees the marked image (no instruction) and lists the objects keyed by mark number, with stacking/containment relations and any unmarked objects; widths are appended as facts.
3. **Decide** — Apertus sees the marked image, the Inventory and the instruction, reasons first, then returns a **Verdict** (plan / ask / refuse). A **Plan** names objects by mark number. Several samples are voted.
4. **Locate** — the **Target pixel** is the first-picked mark's interior pixel; if that object is unmarked, Apertus narrows it down by grid zoom.

Every technique is a switch, so the pipeline can run from "baseline-equivalent" to "everything on" and the report can show each technique's contribution. One command runs it on any folder of scenes.

## User Stories

1. As a team member, I want one command that runs the pipeline on any folder of scenes, so that the jury can re-run it on the held-out set.
2. As a team member, I want the output to be a results file in the official format, so that the official scorer accepts it unchanged.
3. As a team member, I want every answer constrained to the result schema, so that no answer scores 0 for malformed JSON.
4. As a team member, I want the pipeline to point at any OpenAI-compatible server and model name, so that it runs on whatever local GPU setup we have.
5. As a team member, I want a mock mode that runs the whole loop without a model, so that I can develop and test on a laptop.
6. As a team member, I want OpenCV to find the coloured objects on the dark table and number them on the image, so that Apertus can refer to objects by number instead of guessing coordinates.
7. As a team member, I want each mark's pixel to be the deepest interior point of its blob, so that the pixel lies inside the object even for rings or partially covered objects.
8. As a team member, I want stacked objects of different hues (e.g. a yellow gear on a red gear) to get separate marks, so that "the one on top" can be picked.
9. As a team member, I want each mark's minimum width in mm, measured via the calibration's table projection, so that Apertus knows what exceeds the 85 mm gripper.
10. As a team member, I want a coverage check that reports what fraction of the 40 dev targets has a mark inside its ground-truth mask, so that I know the grounding ceiling before involving the model.
11. As a team member, I want the coverage check to list which targets have no mark, so that I can fix segmentation where it matters.
12. As a team member, I want Apertus to produce an Inventory from the marked image before seeing the instruction, so that its scene description is not biased by what the instruction expects.
13. As a team member, I want the Inventory to record what rests on or sits in what, so that Decide can choose the object on top as the first pick.
14. As a team member, I want the Inventory to list objects that have no mark, so that the plan can still name trays, plates or mugs.
15. As a team member, I want widths shown in the Inventory as facts (e.g. "190 mm wide, > 85 mm gripper"), so that Apertus — not code — decides to refuse.
16. As a team member, I want Decide to write its reasoning before its verdict, so that it thinks before committing.
17. As a team member, I want Decide's prompt to name three checks — present and unambiguous, physically possible, safe — so that the ten no-plan cases are each covered by an explicit question.
18. As a team member, I want Decide to sample several times and take a majority, so that one unlucky sample does not flip a verdict.
19. As a team member, I want plan-vs-not-plan ties to go to ask/refuse, so that the more expensive error (a confident wrong plan) is avoided.
20. As a team member, I want the winning plan to be the most frequent plan sequence among plan votes, so that the plan is as stable as the verdict.
21. As a team member, I want plans to refer to marks by number, so that Locate grounds exactly the object Decide meant.
22. As a team member, I want ask answers to carry a concrete question and refuse answers a concrete reason, so that the jury sees sensible safety behaviour.
23. As a team member, I want Locate to look up the mark's interior pixel when the first pick is marked, so that grounding needs no further model call.
24. As a team member, I want Locate to fall back to grid zoom when the first pick is unmarked, so that grey or poorly segmented objects can still be grounded.
25. As a team member, I want a raw-pixel switch (no marks, Apertus guesses coordinates), so that the report can quantify how much positions drift without assistance.
26. As a team member, I want each technique (marks, inventory, reasoning field, voting, grid zoom, width facts) to be a switch, so that I can run ablations.
27. As a team member, I want to send both the marked and the clean image when the server supports two images per message, so that numbers do not hide what objects look like.
28. As a team member, I want a single-image fallback when two images are not supported, so that the pipeline works on either setup.
29. As a team member, I want every raw model reply logged per instruction and stage, so that I can analyse failures for the report.
30. As a team member, I want the pipeline never to read the category, expected or targets columns, so that dev behaviour matches held-out behaviour.
31. As a team member, I want few-shot examples written about invented scenes, so that prompts do not memorise dev answers.
32. As a team member, I want to tune on scenes 01–07 and check on 08–10 only at milestones, so that I can detect overfitting.
33. As a team member, I want to score any subset of scenes, so that I can report the tune and check splits separately.
34. As a team member, I want the same dev split and scorer as my teammate's independent Path B, so that our numbers are comparable.
35. As a team member, I want the pipeline to keep going when one instruction fails (timeout, bad reply), writing a safe fallback answer, so that one error does not lose the whole run.
36. As a jury member, I want a README section that reproduces the result in under 15 minutes, so that I can verify the submission.
37. As a jury member, I want the plan for each instruction to make physical sense (top object first, one object at a time, trays only as places), so that pipeline logic scores well.
38. As a report reader, I want a per-stage failure breakdown (missed mark, wrong verdict, wrong object chosen, grid zoom miss), so that I see where Apertus had difficulty.

## Implementation Decisions

- **Boundary (ADR 0001):** Apertus is the only learned model. OpenCV may draw, crop, threshold, find contours and measure; it never decides which object, verdict or plan.
- **Modules:**
  - *Marks* (perception, no model): scene image + calibration → list of marks `{id, interior pixel [u, v] in full-resolution coords, min width mm}` and the marked image. Segmentation by HSV saturation with hue splitting; interior pixel from the distance-transform maximum; width from the min-area rectangle with its corners projected via the existing pixel-to-table helper.
  - *Apertus client*: OpenAI-compatible chat call with image(s), JSON-schema constrained decoding, temperature and sample count; raw replies logged.
  - *Inventory stage*: marked image (+ clean image if supported) → structured list keyed by mark id, relations, unmarked objects. Code appends width facts.
  - *Decide stage*: marked image + Inventory + instruction → `{reasoning, verdict, plan | question | reason}`; N samples, majority vote, ties to not-plan, most common plan sequence.
  - *Locate stage*: first pick's mark id → interior pixel; unmarked → grid zoom (labelled grid, Apertus picks a cell, crop, repeat 2–3 rounds); raw-pixel mode as an ablation switch.
  - *Runner*: one command over a scenes folder; writes the official results file plus a raw log; switches as command-line flags; reuses the baseline's server/model/mock flag conventions.
- **Answer mapping:** plan → `pick`/`place` actions with readable object names (from the Inventory) and `target_pixel` from Locate; ask → one `ask_user`; refuse → one `refuse`. Always schema-valid.
- **Coordinates:** all pixels returned in full 2448 × 2048 resolution regardless of the image size sent to the model.
- **Model:** `swiss-ai/Apertus-v1.5-8B` on vLLM by default. Whether it accepts two images in one message is the first experiment; it decides between marked+clean and marked-only.
- **Inputs read:** only the image, calibration, instruction id and instruction text.

## Testing Decisions

- Good tests check external behaviour at the highest seam, not internals: given a scene folder, does the pipeline produce a schema-valid results file that the official scorer accepts, with the expected verdicts and pixels?
- **Seam 1 — whole pipeline:** scene folder → results file, scored by the existing official scorer. The model is replaced by a scripted fake server/client returning fixed replies, so the test runs without a GPU and checks: schema validity, vote/tie behaviour, mark-id → pixel mapping, grid-zoom fallback path, and failure fallback.
- **Seam 2 — perception:** scene → marks, checked against the ground-truth masks (mark coverage: every dev target has a mark inside its mask; reported as a fraction plus the missing list). No model involved.
- No finer-grained unit tests.
- Prior art: the official scorer and its schema checks; the baseline's `--mock` mode as the model-free loop.

## Out of Scope

- Any learned vision model (detector, segmenter, depth model) — that is Path A.
- Fine-tuning or training Apertus.
- Running on the real robot or using depth (the dataset is 2D).
- Merging with or reconciling against the teammate's independent Path B implementation.
- The 1–2 page report, slides and screen recording (they use this pipeline's outputs but are separate deliverables).

## Further Notes

- Timeline: held-out scenes released 14.10 at 12:00, results due 16.10 at 12:00. Build order: skeleton end-to-end → mark coverage check → reasoning field + vote → grid zoom → width facts → raw-pixel ablation.
- Scoring economics that drove the vote rule: a missed no-plan instruction costs ~3 points; a false ask/refuse on a plan costs ~1.5.
- Height bias: widths are measured at table height, so tall objects (mug) read a few percent wide — fine against an 85 mm threshold.
