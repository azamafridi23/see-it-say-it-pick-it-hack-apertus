# OpenCV is allowed in Path B; learned models other than Apertus are not

"Apertus alone" is read as: Apertus is the only learned model, and it makes every choice (which object, which verdict, which plan). Deterministic image processing (drawing numbered marks, cropping, colour thresholds, contours, measuring widths via the calibration) may prepare what Apertus sees and turn its choice into a pixel. We picked this over a strict "image in, Apertus out" reading because raw pixel answers from a VLM drift badly on a 2448×2048 photo, and grounding is 35% of the score.

## Consequences

- The strict reading stays available as a switch (raw pixel, no marks) so the report can show what the OpenCV assistance is worth.
- The head-to-head with Path A now compares a learned detector against classical segmentation, both feeding Apertus, not "detector vs nothing". The report must say so.
