"""System 2. A scripted planner stands in for the LLM/VLM."""

from __future__ import annotations

from robojev.types import Belief, Skill


class ScriptedReasoner:
    def __init__(self, obj: str = "apple", container: str = "fridge") -> None:
        self.obj = obj
        self.container = container
        self.calls = 0

    def plan(self, belief: Belief) -> list[Skill]:
        self.calls += 1
        return scripted_skills(belief, self.obj, self.container)


def scripted_skills(
    belief: Belief, obj: str = "apple", container: str = "fridge"
) -> list[Skill]:
    steps: list[Skill] = []
    at = belief.at
    holding = belief.held_known and belief.held == obj
    loc = belief.loc.get(obj)
    stored = loc == container and not holding

    if stored:
        if belief.open.get(container) is True:
            if at != container:
                steps.append(Skill("move", container))
            steps.append(Skill("close", container))
        return steps

    if not holding:
        if loc and loc not in {at, "gripper"}:
            steps.append(Skill("move", loc))
            at = loc
        steps.append(Skill("pick", obj))

    if belief.open.get(container) is not True:
        if at != container:
            steps.append(Skill("move", container))
            at = container
        steps.append(Skill("open", container))

    if at != container:
        steps.append(Skill("move", container))
    steps.append(Skill("place", obj, container))
    steps.append(Skill("close", container))
    return steps
