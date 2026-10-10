"""Apply instruction constraints, request object IDs, and compile grounded actions."""

import json
from functools import partial

from .grounding import planning_inventory
from .planner import compile_transfer_groups, expand_blockers
from .safety import (
    explicit_refusal,
    named_pick_candidates,
    ordinal_pick_constraint,
    relative_neighbor_constraint,
    stationary_blocker_refusal,
    unresolved_deictic,
)
from .template_prompt import template_prompt
from .validation import validate_plan


def validate_instruction_response(
    raw, inventory, width, height, constraint=None, ordinal=None, relative=None
):
    """Compile feasible actions and verify their target selection against measured facts."""
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                entry["actions_data"] = compile_transfer_groups(
                    entry.get("actions_data")
                )
    result = validate_plan(expand_blockers(raw, inventory), inventory, 1, width, height)
    if constraint:
        actions = result[0]["actions_data"]
        matching = constraint["matching_ids"]
        if len(matching) > 1 and actions[0]["action"] != "ask_user":
            raise ValueError(
                f'Ambiguous {constraint["name"]}: ask_user required; matches {matching}'
            )
        if not matching and actions[0]["action"] != "refuse":
            raise ValueError(f'Absent {constraint["name"]}: refuse required')
        if (
            len(matching) == 1
            and actions[0]["action"] == "pick"
            and actions[0]["object"] != matching[0]
        ):
            raise ValueError(
                f'Exact named target is {matching[0]}, not {actions[0]["object"]}'
            )
    if ordinal:
        first = result[0]["actions_data"][0]
        if ordinal["target_id"] is None:
            if first["action"] != "refuse":
                raise ValueError(
                    "Requested ordinal instance does not exist; refuse required"
                )
        elif first["action"] != "pick" or first["object"] != ordinal["target_id"]:
            raise ValueError(
                f'Ordinal target is {ordinal["target_id"]}; ordering {ordinal["ordered_ids"]}'
            )
    if relative:
        actions = result[0]["actions_data"]
        target = relative["target_id"]
        if target is None:
            if actions[0]["action"] != relative["required_decision"]:
                raise ValueError(
                    f'Relative target unavailable: {relative["required_decision"]} required'
                )
        else:
            allowed = {target}
            for _ in inventory["objects"]:
                allowed.update(
                    r["subject"]
                    for r in inventory["relations"]
                    if r["object"] in allowed
                )
            picks = [a["object"] for a in actions if a["action"] == "pick"]
            if target not in picks or any(oid not in allowed for oid in picks):
                raise ValueError(
                    f"Relative target is {target}; only its blockers may be picked first"
                )
    return result


def plan_instructions(instruction, inventory, request, artifacts, width, height):
    """Plan each instruction against the same original inventory and retain failures."""
    instructions = [
        line.lstrip("- ").strip() for line in instruction.splitlines() if line.strip()
    ]
    results = []
    scene_facts = planning_inventory(inventory)
    for index, text in enumerate(instructions, 1):
        if unresolved_deictic(text):
            results.append(
                {
                    "instruction_id": index,
                    "actions_data": [
                        {
                            "action": "ask_user",
                            "question": "Which visible object do you mean? Please specify its name or position.",
                        }
                    ],
                }
            )
            artifacts[f"clarification_gate_{index}"] = (
                "Unresolved demonstrative/pronoun"
            )
            continue
        reason = explicit_refusal(text) or stationary_blocker_refusal(text, inventory)
        if reason:
            results.append(
                {
                    "instruction_id": index,
                    "actions_data": [{"action": "refuse", "reason": reason}],
                }
            )
            artifacts[f"safety_gate_{index}"] = reason
            continue
        constraint = named_pick_candidates(text, inventory)
        ordinal = ordinal_pick_constraint(text, inventory)
        relative = relative_neighbor_constraint(text, scene_facts)

        # Bind this instruction's constraints without changing its scene inventory.
        validator = partial(
            validate_instruction_response,
            inventory=inventory,
            width=width,
            height=height,
            constraint=constraint,
            ordinal=ordinal,
            relative=relative,
        )

        try:
            prediction = request(
                template_prompt,
                [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "inventory": scene_facts,
                                "named_target_constraint": constraint,
                                "ordinal_constraint": ordinal,
                                "relative_constraint": relative,
                                "instructions": [{"instruction_id": 1, "text": text}],
                            }
                        ),
                    }
                ],
                validator,
                f"plan_{index}",
            )[0]
            prediction["instruction_id"] = index
            results.append(prediction)
        except Exception as exc:
            artifacts.setdefault("instruction_errors", {})[str(index)] = str(exc)
            results.append(
                {
                    "instruction_id": index,
                    "processing_error": str(exc),
                    "actions_data": [
                        {
                            "action": "ask_user",
                            "question": "I could not validate a grounded plan for this instruction. Please retry perception/planning.",
                        }
                    ],
                }
            )
    artifacts["plan"] = results
    return results
