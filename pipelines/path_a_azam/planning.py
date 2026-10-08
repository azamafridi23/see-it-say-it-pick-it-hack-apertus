"""Shared symbolic state for ordering and checking moves. No model calls or image dependencies."""
from __future__ import annotations


class PlanIssue(ValueError):
    def __init__(self, message: str, decision: str = "refuse"):
        super().__init__(message)
        self.decision = decision


class SceneState:
    """Track current support relations, rather than treating every previously moved object as clear."""

    def __init__(self, graph: dict):
        self.objects = {o["id"]: o for o in graph["objects"]}
        self.covers = {oid: set(o.get("covered_by", [])) for oid, o in self.objects.items()}
        self.uncertain = {oid: set() for oid in self.objects}
        for oid, obj in self.objects.items():
            for lower in obj.get("on", []):
                if lower not in self.objects:
                    raise PlanIssue(f"Scene graph refers to missing object {lower}.")
                self.covers[lower].add(oid)
            for lower in obj.get("maybe_on", []):
                if lower not in self.objects:
                    raise PlanIssue(f"Scene graph refers to missing object {lower}.")
                self.uncertain[lower].add(oid)
        for oid, covering in self.covers.items():
            if covering - self.objects.keys():
                raise PlanIssue(f"Scene graph has unknown blockers for {oid}.")
            self.uncertain[oid] -= covering
        self.held = None

    def require_pickable(self, oid: str):
        if oid not in self.objects:
            raise PlanIssue(f"There is no object {oid} in the scene.")
        obj = self.objects[oid]
        if obj.get("touching_hand") or obj.get("label") == "hand":
            raise PlanIssue(f"A person's hand is on {oid}, so the robot must not move it.")
        if not obj.get("pickable", False):
            raise PlanIssue(f"{oid} cannot be picked up: {obj.get('why_not_pickable') or 'it is not pickable'}.")

    def require_resolved(self, oid: str):
        if self.uncertain[oid]:
            upper = ", ".join(sorted(self.uncertain[oid]))
            raise PlanIssue(f"Is {upper} resting on {oid}, or only beside it? I need to know before moving {oid}.",
                            "ask_user")

    def move(self, oid: str, destination: str | None):
        self.require_pickable(oid)
        if self.held is not None:
            raise PlanIssue(f"{self.held} is still in the gripper. Put it down before another pick.")
        self.require_resolved(oid)
        if self.covers[oid]:
            raise PlanIssue(f"{oid} is covered by {', '.join(sorted(self.covers[oid]))}. Move the covering objects first.")
        if destination == oid:
            raise PlanIssue(f"{oid} cannot be placed on itself.")
        if destination is not None:
            if destination not in self.objects:
                raise PlanIssue(f"There is no destination {destination} in the scene.")
            dest = self.objects[destination]
            if dest.get("label") == "hand" or dest.get("touching_hand"):
                raise PlanIssue(f"The robot must not place objects onto a hand or an object touching a hand: {destination}.")
            # A movable destination needs a clear support surface. Tray/plate placement geometry is
            # not certified by this symbolic check; that remains a perception/execution limitation.
            if dest.get("pickable"):
                uncertain = self.uncertain[destination] - {oid}
                if uncertain:
                    raise PlanIssue(f"Is {destination} clear for placement? Its support relations are uncertain.", "ask_user")
                if self.covers[destination] - {oid}:
                    raise PlanIssue(f"The destination {destination} is covered by another object.", "ask_user")
        for relations in (self.covers, self.uncertain):
            for upper in relations.values():
                upper.discard(oid)
        if destination is None:
            self.held = oid
        else:
            self.covers[destination].add(oid)


def order_moves(tasks: list, graph: dict) -> dict:
    """Clear confirmed blockers; ask about task-relevant uncertainty; never invent support edges."""
    moves = []
    try:
        state = SceneState(graph)
        goals = {}
        for task in tasks:
            for oid in task["objects"]:
                if oid in goals and goals[oid] != task["to"]:
                    raise PlanIssue(f"The request gives multiple destinations for {oid}. Which final destination should I use?",
                                    "ask_user")
                goals[oid] = task["to"]
        if not goals:
            raise PlanIssue("No object was selected. Which object should I move?", "ask_user")
        if sum(to is None for to in goals.values()) > 1:
            raise PlanIssue("I can hold only one object at a time. Where should I place the others?", "ask_user")
        # A sole pick-and-hold goal must be last, independent of the reference agent's list order.
        requested = sorted(goals, key=lambda oid: goals[oid] is None)
        placed = {}

        def move(oid, destination, ancestors=()):
            state.require_pickable(oid)
            if oid in ancestors:
                raise PlanIssue("The scene has a cycle in its support relations. Please clarify which object is on top.",
                                "ask_user")
            state.require_resolved(oid)
            for upper in sorted(state.covers[oid]):
                parking = state.objects[upper].get("in")
                if parking not in state.objects or state.objects[parking].get("label") not in ("grey tray", "black tray"):
                    raise PlanIssue(f"Where can I place {upper} while uncovering {oid}?", "ask_user")
                move(upper, parking, (*ancestors, oid))
            state.move(oid, destination)
            moves.append({"object": oid, "to": destination})
            placed[oid] = destination

        for oid in requested:
            destination = goals[oid]
            # A blocker parked at its requested destination has already fulfilled that goal.
            if oid in placed and placed[oid] == destination:
                continue
            move(oid, destination)
        # A later uncovering step must not silently undo an earlier requested placement.
        if any(placed.get(oid) != destination for oid, destination in goals.items()):
            raise PlanIssue("The required moves conflict with a requested final placement. Please clarify the final arrangement.",
                            "ask_user")
    except PlanIssue as exc:
        return {"decision": exc.decision, "moves": [], "message": str(exc)}
    return {"decision": "plan", "moves": moves, "message": ""}


def check(reply: dict, graph: dict) -> str | None:
    """Return the first plan error, using the same state/uncertainty rules as move ordering."""
    if not isinstance(reply, dict):
        return "The answer must be an object."
    decision, moves, message = reply.get("decision"), reply.get("moves"), reply.get("message")
    if not isinstance(moves, list) or not isinstance(message, str):
        return "moves must be a list and message must be a string."
    if decision in ("ask_user", "refuse"):
        return None if message.strip() and not moves else "Questions and refusals need a message and no moves."
    if decision != "plan" or not moves or message.strip():
        return "A plan needs at least one move and an empty message."
    try:
        state = SceneState(graph)
        for n, move in enumerate(moves, 1):
            if not isinstance(move, dict) or set(move) != {"object", "to"}:
                return f"Move {n} needs exactly object and to."
            oid, destination = move["object"], move["to"]
            if not isinstance(oid, str) or (destination is not None and not isinstance(destination, str)):
                return f"Move {n} must use object ids."
            state.move(oid, destination)
            if destination is None and n < len(moves):
                return f"Move {n}: {oid} is still in the gripper. Put it down before the next move."
    except PlanIssue as exc:
        return str(exc)
    return None
