from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill:
    name: str
    target: str
    dest: str | None = None

    def label(self) -> str:
        if self.dest:
            return f"{self.name}({self.target} -> {self.dest})"
        return f"{self.name}({self.target})"


@dataclass
class Belief:
    held: str | None
    held_known: bool
    at: str
    open: dict[str, bool | None]
    loc: dict[str, str | None]

    def summary(self) -> str:
        held = self.held if self.held_known else "unknown"
        doors = ", ".join(
            f"{name}={'open' if value else 'closed' if value is False else 'unknown'}"
            for name, value in self.open.items()
        )
        locs = ", ".join(f"{name}@{place or 'unknown'}" for name, place in self.loc.items())
        return f"at={self.at} held={held} {doors} {locs}"


@dataclass
class SensorReading:
    """Cheap sensors after one skill. Ground-truth success is not included."""

    skill: Skill
    robot_at: str
    held: str | None
    held_known: bool
    loc: dict[str, str | None]
    container_open: bool | None
    slip: bool
    contact: str
    agreement: float
    note: str

    def state_text(self) -> str:
        held = self.held if self.held_known else "unknown"
        locs = ", ".join(f"{name}={place or 'unknown'}" for name, place in self.loc.items())
        if self.container_open is None:
            door = "unknown"
        else:
            door = "open" if self.container_open else "closed"
        return (
            f"Just executed {self.skill.label()}. "
            f"Robot is at {self.robot_at}. Gripper holds {held}. "
            f"Contact={self.contact}. Slip={self.slip}. "
            f"Relevant container appears {door}. Locations: {locs}. "
            f"Sensor agreement={self.agreement:.2f}. Note: {self.note}."
        )


@dataclass
class Decision:
    model: str
    grasp_ok: float
    skill_done: float
    failure: str
    failure_confidence: float
    recovery: str
    recovery_confidence: float

    @property
    def skill_done_confidence(self) -> float:
        return abs(self.skill_done - 0.5) * 2


@dataclass
class StepLog:
    index: int
    skill: Skill
    note: str
    decision: Decision | None
    gate: str
    detail: str


@dataclass
class EpisodeResult:
    name: str
    instruction: str
    success: bool
    reasoner_calls: int
    decision_calls: int
    steps: list[StepLog] = field(default_factory=list)
    mode: str = "robojev"


@dataclass(frozen=True)
class Task:
    name: str
    instruction: str
    faults: dict[str, str]
