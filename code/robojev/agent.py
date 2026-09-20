"""Reasoning / Decision / Action loop, plus a baseline that reasons every step."""

from __future__ import annotations

from robojev.jev import LocalSystemOne, JevSystemOne, should_escalate
from robojev.reasoner import ScriptedReasoner
from robojev.types import Belief, Decision, EpisodeResult, SensorReading, Skill, StepLog, Task
from robojev.world import World

MAX_STEPS = 16
MAX_RETRIES = 2


def update_belief(previous: Belief, reading: SensorReading) -> Belief:
    if not reading.held_known or reading.agreement < 0.5:
        return Belief(
            held=None,
            held_known=False,
            at=reading.robot_at,
            open={name: None for name in previous.open},
            loc={name: None for name in previous.loc},
        )
    doors = dict(previous.open)
    if reading.skill.name in {"open", "close"} and reading.skill.target in doors:
        doors[reading.skill.target] = reading.container_open
    if reading.skill.name == "place" and reading.skill.dest in doors:
        doors[reading.skill.dest] = reading.container_open
    return Belief(
        held=reading.held,
        held_known=True,
        at=reading.robot_at,
        open=doors,
        loc={name: place for name, place in reading.loc.items()},
    )


class RoboJevAgent:
    def __init__(self, decider: LocalSystemOne | JevSystemOne) -> None:
        self.decider = decider

    def run(self, task: Task, reasoner: ScriptedReasoner | None = None, on_start=None, on_step=None) -> EpisodeResult:
        world = World()
        reasoner = reasoner or ScriptedReasoner()
        belief = world.belief()
        plan = reasoner.plan(belief)
        if on_start:
            on_start(world)
        logs: list[StepLog] = []
        seen: dict[str, int] = {}
        retries = 0

        for index in range(1, MAX_STEPS + 1):
            if world.goal_met() or not plan:
                break
            skill = plan.pop(0)
            seen[skill.name] = seen.get(skill.name, 0) + 1
            reading = world.execute(skill, task.faults.get(f"{skill.name}#{seen[skill.name]}"))
            decision = self.decider.judge(reading)
            belief = update_belief(belief, reading)
            gate, detail, plan, retries = self._gate(
                world, reasoner, belief, plan, skill, decision, retries
            )
            logs.append(
                StepLog(index, skill, reading.note, decision, gate, detail)
            )
            if on_step:
                on_step(world, logs[-1])

        return EpisodeResult(
            name=task.name,
            instruction=task.instruction,
            success=world.goal_met(),
            reasoner_calls=reasoner.calls,
            decision_calls=len(logs),
            steps=logs,
            mode="robojev",
        )

    def _gate(
        self,
        world: World,
        reasoner: ScriptedReasoner,
        belief: Belief,
        plan: list[Skill],
        skill: Skill,
        decision: Decision,
        retries: int,
    ) -> tuple[str, str, list[Skill], int]:
        if should_escalate(decision) or retries >= MAX_RETRIES:
            clarified = world.belief()
            return (
                "escalate",
                f"confidence {decision.recovery_confidence:.2f}，reasoner rechecked: {clarified.summary()}",
                reasoner.plan(clarified),
                0,
            )
        if decision.recovery == "retry":
            plan.insert(0, skill)
            return "retry", f"failure={decision.failure}，retry the same skill", plan, retries + 1
        if decision.recovery == "replan":
            return (
                "replan",
                f"failure={decision.failure}，hand off to reasoner: {belief.summary()}",
                reasoner.plan(belief),
                0,
            )
        if decision.recovery == "abort":
            plan.clear()
            return "abort", "Decision aborted", plan, 0
        return "continue", "skill done; continue", plan, 0


class BaselineAgent:
    """The contrast in the note: ask the reasoner again after every skill."""

    def run(self, task: Task) -> EpisodeResult:
        world = World()
        reasoner = ScriptedReasoner()
        plan = reasoner.plan(world.belief())
        logs: list[StepLog] = []
        seen: dict[str, int] = {}

        for index in range(1, MAX_STEPS + 1):
            if world.goal_met() or not plan:
                break
            skill = plan.pop(0)
            seen[skill.name] = seen.get(skill.name, 0) + 1
            reading = world.execute(skill, task.faults.get(f"{skill.name}#{seen[skill.name]}"))
            plan = reasoner.plan(world.belief())
            logs.append(
                StepLog(index, skill, reading.note, None, "reason", "replan every step")
            )

        return EpisodeResult(
            name=task.name,
            instruction=task.instruction,
            success=world.goal_met(),
            reasoner_calls=reasoner.calls,
            decision_calls=0,
            steps=logs,
            mode="baseline",
        )
