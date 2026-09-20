"""Symbolic tabletop. One task, four skills, injected faults. No simulator."""

from __future__ import annotations

from robojev.types import Belief, SensorReading, Skill


class World:
    def __init__(self, obj: str = "apple", container: str = "fridge") -> None:
        self.obj = obj
        self.container = container
        self.loc: dict[str, str] = {obj: "counter"}
        self.open: dict[str, bool] = {container: False}
        self.at = "counter"
        self.held: str | None = None

    def goal_met(self) -> bool:
        return (
            self.loc.get(self.obj) == self.container
            and self.held is None
            and self.open[self.container] is False
        )

    def belief(self) -> Belief:
        return Belief(
            held=self.held,
            held_known=True,
            at=self.at,
            open={name: value for name, value in self.open.items()},
            loc=dict(self.loc),
        )

    def execute(self, skill: Skill, fault: str | None = None) -> SensorReading:
        if fault == "lid_falls" and skill.name == "place":
            self.open[self.container] = False
        note = self._apply(skill, fault)
        return self._read(skill, note, corrupt=fault == "ambiguous")

    def _apply(self, skill: Skill, fault: str | None) -> str:
        if skill.name == "move":
            self.at = skill.target
            return f"moved to {skill.target}"
        if skill.name == "pick":
            if self.held is not None:
                return "gripper already full"
            if self.loc.get(skill.target) != self.at:
                return f"{skill.target} is not here"
            if fault == "slip":
                return "fingers slipped, object left behind"
            self.held = skill.target
            self.loc[skill.target] = "gripper"
            return f"grasped {skill.target}"
        if skill.name == "open":
            if self.at != skill.target:
                return f"not at {skill.target}"
            self.open[skill.target] = True
            return f"opened {skill.target}"
        if skill.name == "close":
            if self.at != skill.target:
                return f"not at {skill.target}"
            self.open[skill.target] = False
            return f"closed {skill.target}"
        if skill.name == "place":
            dest = skill.dest or ""
            if self.held != skill.target:
                return f"not holding {skill.target}"
            if self.at != dest:
                return f"not at {dest}"
            if dest in self.open and not self.open[dest]:
                return f"{dest} is closed"
            self.loc[skill.target] = dest
            self.held = None
            return f"placed {skill.target} in {dest}"
        return f"unknown skill {skill.name}"

    def _read(self, skill: Skill, note: str, corrupt: bool) -> SensorReading:
        if corrupt:
            return SensorReading(
                skill=skill,
                robot_at=self.at,
                held=None,
                held_known=False,
                loc={name: None for name in self.loc},
                container_open=None,
                slip=False,
                contact="light",
                agreement=0.28,
                note=note + "; cameras disagree",
            )
        door = self._door(skill)
        slip = "slipped" in note
        if self.held == skill.target or note.startswith(("placed", "opened", "closed")):
            contact = "firm"
        elif slip:
            contact = "light"
        else:
            contact = "none"
        return SensorReading(
            skill=skill,
            robot_at=self.at,
            held=self.held,
            held_known=True,
            loc=dict(self.loc),
            container_open=door,
            slip=slip,
            contact=contact,
            agreement=0.93,
            note=note,
        )

    def _door(self, skill: Skill) -> bool | None:
        if skill.name in {"open", "close"} and skill.target in self.open:
            return self.open[skill.target]
        if skill.name == "place" and skill.dest in self.open:
            return self.open[skill.dest]
        return None
