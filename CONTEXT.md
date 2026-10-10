# See It, Say It, Pick It

Turning one workcell photo and one English instruction into a safe robot action.

## Language

**Path A**:
A pipeline where a separate vision model detects objects and their pixels, builds a scene graph, and Apertus reasons over that graph plus the instruction.

**Path B**:
A pipeline where Apertus alone sees the image, names the target object and its pixel, and produces the action plan. No other learned model is involved; deterministic image processing (drawing, cropping, colour thresholds, contours) may assist, but Apertus makes every choice.
_Avoid_: baseline (the baseline is one specific, minimal Path B)

**Baseline**:
The organisers' reference Path B: a single plain prompt with no constrained decoding. It is the score to beat.

**Target pixel**:
The `[u, v]` pixel of the first object the robot picks, which is not necessarily the object the instruction names.

**Inventory**:
The list of objects Apertus reports seeing in a photo (name, colour, rough position, what each rests on or sits in), produced without seeing the instruction.

**Verdict**:
The decision for one instruction: exactly one of *plan*, *ask* (the instruction is ambiguous) or *refuse* (it is impossible or unsafe).

**Plan**:
An ordered list of pick/place actions over named objects, starting with a pick. It contains no pixels.

## Relationships

- **Path A** and **Path B** are compared head-to-head on the same scenes
- The **Baseline** is one instance of **Path B**
- A **Verdict** of *plan* carries exactly one **Plan**; the **Target pixel** locates that plan's first picked object
