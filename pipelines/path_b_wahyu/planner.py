"""Compile visible blockers before the object transfers selected by Apertus."""

import copy


def expand_blockers(raw, inventory):
    """Copy model actions and insert tray transfers for incidental visible blockers."""
    if not isinstance(raw, list):
        return raw
    by_id = {o["id"]: o for o in inventory["objects"]}
    trays = [
        o for o in inventory["objects"] if o["name"] in ("grey_tray", "black_tray")
    ]
    result = copy.deepcopy(raw)
    for entry in result:
        actions = entry.get("actions_data")
        if not isinstance(actions, list) or any(
            a.get("action") not in ("pick", "place") for a in actions
        ):
            continue
        remaining = {(r["subject"], r["object"]) for r in inventory["relations"]}
        expanded = []
        visiting = set()

        def clear(target):
            if target in visiting:
                raise ValueError("Cyclic stacking relations")
            visiting.add(target)
            blockers = sorted(top for top, bottom in remaining if bottom == target)
            for blocker in blockers:
                obj = by_id[blocker]
                if obj["held_by_human"]:
                    raise ValueError("Cannot clear a human-held blocker")
                clear(blocker)
                # Prefer another visible tray; this is a placement reference, not a pose.
                choices = sorted(
                    trays, key=lambda tray: (tray["name"] == obj["tray"], tray["id"])
                )
                if not choices:
                    raise ValueError("No visible tray for clearing a blocker")
                expanded.extend(
                    [
                        {"action": "pick", "object": blocker},
                        {"action": "place", "object": choices[0]["id"]},
                    ]
                )
                remaining.difference_update(
                    (top, bottom) for top, bottom in list(remaining) if top == blocker
                )
            visiting.remove(target)

        # Clear incidental blockers for the entire transfer before touching supports.
        # Explicitly selected blockers remain in the model's action sequence.
        planned_picks = {a.get("object") for a in actions if a.get("action") == "pick"}
        for action in actions:
            if action.get("action") == "pick" and action.get("object") in by_id:
                target = action["object"]
                blockers = [top for top, bottom in remaining if bottom == target]
                if blockers and all(top not in planned_picks for top in blockers):
                    clear(target)
        holding = None
        for action in actions:
            oid = action.get("object")
            if oid not in by_id:
                expanded.append(
                    action
                )  # The schema validator will report the unknown ID.
                continue
            if action["action"] == "pick":
                if holding is not None:
                    raise ValueError("Cannot expand picks while gripper is occupied")
                clear(oid)
                holding = oid
                remaining.difference_update(
                    (top, bottom) for top, bottom in list(remaining) if top == oid
                )
            elif holding is not None:
                if by_id[oid]["name"] not in (
                    "grey_tray",
                    "black_tray",
                    "plate",
                    "saucer",
                ):
                    remaining.add((holding, oid))
                holding = None
            expanded.append(action)
        entry["actions_data"] = expanded
    return result


def compile_transfer_groups(actions):
    """Serialize a model's grouped picks/places for one shared destination."""
    if not isinstance(actions, list):
        return actions
    split = next(
        (i for i, a in enumerate(actions) if a.get("action") != "pick"), len(actions)
    )
    picks, places = actions[:split], actions[split:]
    if (
        len(picks) > 1
        and len(places) in (1, len(picks))
        and all(a.get("action") == "place" for a in places)
        and len({a.get("object") for a in places}) == 1
    ):
        destinations = places * len(picks) if len(places) == 1 else places
        return [action for pair in zip(picks, destinations) for action in pair]
    return actions
