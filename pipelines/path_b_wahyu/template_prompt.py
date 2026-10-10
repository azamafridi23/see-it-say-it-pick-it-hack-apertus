"""Separate perception from language planning; coordinates use original pixels."""

PERCEPTION_PROMPT = """
Identify every object in this robot workcell, including partly hidden ones.
Two trays: grey_tray is the light grey rectangular container; black_tray is the
large dark supporting surface. Include both even when empty. Ignore stickers
and camera hardware. A green cup with a handle is mug, a yellow sun-shaped toothed
object is yellow_gear, red toothed discs are red_gear, a red square block is red_cube.
Categories: red_cube, red_gear, yellow_gear, mug, mandarin, banana, plate, saucer,
grey_part, white_part, small_white_part, grey_tray, black_tray, hand.
Return JSON only:
{"objects":[{"id":"mug","name":"mug","bbox":[x1,y1,x2,y2],
"tray":"grey_tray","held_by_human":false}],
"relations":[{"subject":"top_id","relation":"on_top_of","object":"bottom_id"}]}.
Tight bounding boxes use normalized coordinates 0 to 1000; x goes right, y down.
Use unique ids; identical instances numbered left to right. Tray membership is
 grey_tray, black_tray, outside, or unknown; trays use their own name.
List physically stacked objects in relations. Mark objects held by a hand.
"""

template_prompt = """
You plan robot actions using a measured scene inventory.
Return ONLY a JSON array with one result per instruction, preserving integer
instruction_id. Each result: {"instruction_id": 1, "actions_data": [...]}.
Pick/place actions: {"action": "pick|place", "object": "EXACT inventory id"}.
No coordinates: the program supplies measured pixels from inventory ids.
Other actions: {"action":"ask_user","question":"specific question"} or
{"action":"refuse","reason":"specific reason"}, alone per instruction.
Every pick object MUST be in eligible_pick_ids; tray ids are NEVER pick targets.
For "everything", enumerate eligible_pick_ids filtered by source tray/exceptions.
Never invent ids. Every instruction starts from the original scene independently.
Only manipulate objects supported by grey_tray or black_tray. Trays, plate and
saucer exceed the 85 mm gripper opening: they are place destinations, never picks. Use image directions and measured positions for
left/right, ordinal order, closest/farthest (Euclidean distance), and exceptions. Use the provided nearest_to_farthest rankings instead of
estimating distances: ignore the reference itself, filter by requested category,
then take the first for closest or last for farthest.
For "object to the left/right of X", select the first in nearest_left_of or
nearest_right_of for X. A named_target_constraint states the exact matching IDs:
zero matches require refuse, multiple matches require ask_user, one is the target.
A relative_constraint supplies the measured target_id for an immediate left/right
neighbor. Pick that target, clearing its blockers first when needed.
An ordinal_constraint supplies the computed target_id for an ordinal instruction;
use that exact ID. If target_id is null, refuse the nonexistent ordinal instance.
Multiple indistinguishable matches with no qualifier require ask_user.
A confidently absent target, contradictory instruction, impossible containment,
or unsafe throwing, pushing objects off stacks, or taking objects from a human
requires refuse. Uncertain perception requires ask_user. The gripper holds one
object and opens at most 85 mm. Do not refuse ordinary feasible transfers.
For transfer of N objects, return N pick/place pairs, including ALL matches and
respecting exceptions and source tray membership. Each place needs an unmatched
pick; never pick a second object while holding the first. Final pick is allowed.
When the requested object has something on it, first pick and place its blockers
in a safe tray, then move the requested object. Never pull out a supporting object
while leaving its blocker on top. Explicit demands to do so require refuse.
Use scene relations and human-contact flags for stacks and human contact. Do not use
instruction category labels or any expected answers. Return only the JSON array.
Example transfer: {"instruction_id":1,"actions_data":[{"action":"pick","object":"red_cube"},{"action":"place","object":"black_tray"}]}.
"""


COMPONENT_PROMPT = """These are close-ups of unclassified neutral foreground candidates.
Identify ONLY actual mechanical components. Ignore labels, tray rims, mug handles,
camera hardware, dishes, background and hands. Some photos may show the SAME
component; return it only once. Return {"parts":[{"candidate_id":"obj_N",
"name":"small_white_part"}]} or {"parts":[]}.
Use ONE category per name, chosen from grey_part, white_part, small_white_part.
Never combine categories with a slash or pipe.
Grey parts are metal bars/brackets; white_part is the broad white component;
small_white_part is the small white component. Do not guess uncertain candidates."""


STACK_PROMPT = """Inspect these overlapping object close-ups. Which object
physically rests ON TOP OF the other? Distinguish stacking from merely being beside
an object in the image. Return only {"relations":[{"subject":"upper_object_id",
"relation":"on_top_of","object":"lower_object_id"}]}. Use the supplied IDs.
Return an empty list for pairs that are not stacked. Ignore tray support."""


# Preferred uppercase name; the original import remains available for callers.
PLANNING_PROMPT = template_prompt
