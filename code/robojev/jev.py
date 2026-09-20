"""System 1. Same four questions the note assigns to RoboJev.

Real Jev when TYPESAFE_API_KEY is set. Otherwise a local rule model that
returns the same answer schema: noul, choice, confidence.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from robojev.types import Decision, SensorReading

API_URL = "https://api.typesafe.ai/v1/systemone"

QUESTIONS = {
    "grasp_ok": {
        "type": "noul",
        "instructions": "Is the intended object securely in the gripper after this skill?",
        "criteria": {
            "true": "Gripper holds the target, contact is firm, and slip is false",
            "false": "Gripper is empty, slipped, or the hold is unknown",
        },
    },
    "skill_done": {
        "type": "noul",
        "instructions": "Did this skill achieve its expected effect?",
        "criteria": {
            "true": "The effect is clear from the sensors",
            "false": "The effect is missing, contradicted, or unknown",
        },
    },
    "failure": {
        "type": "choice",
        "instructions": "What failure, if any, just happened?",
        "criteria": {
            "none": "Expected effect is clearly achieved",
            "slip": "Contact was made but the object slipped",
            "miss": "The gripper closed without the object",
            "blocked": "A precondition failed: wrong place, closed container, or not holding it",
            "uncertain": "Sensors disagree or a required reading is missing",
        },
    },
    "recovery": {
        "type": "choice",
        "instructions": (
            "What should the controller do next? "
            "Choose continue only when the skill clearly succeeded."
        ),
        "criteria": {
            "continue": "Execute the next planned skill",
            "retry": "Attempt this same skill again",
            "replan": "The world diverged; ask the reasoner for a new plan",
            "abort": "Stop, the situation is unrecoverable",
        },
    },
}


class LocalSystemOne:
    """Deterministic stand-in. Confidence drops when sensors conflict."""

    model = "robojev-local"

    def judge(self, reading: SensorReading) -> Decision:
        uncertain = reading.agreement < 0.5 or not reading.held_known
        if reading.skill.name in {"open", "close"} and reading.container_open is None:
            uncertain = True
        if uncertain:
            return Decision(
                model=self.model,
                grasp_ok=0.5,
                skill_done=0.5,
                failure="uncertain",
                failure_confidence=reading.agreement,
                recovery="continue",
                recovery_confidence=reading.agreement,
            )

        done, failure, recovery = _outcome(reading)
        grasped = reading.held == reading.skill.target and not reading.slip
        return Decision(
            model=self.model,
            grasp_ok=0.93 if grasped else 0.07,
            skill_done=done,
            failure=failure,
            failure_confidence=reading.agreement,
            recovery=recovery,
            recovery_confidence=reading.agreement,
        )


class JevSystemOne:
    def __init__(self, api_key: str, model: str = "jev-latest") -> None:
        self.api_key = api_key
        self.model = model

    def judge(self, reading: SensorReading) -> Decision:
        body = json.dumps(
            {"model": self.model, "state": reading.state_text(), "questions": QUESTIONS}
        ).encode()
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Jev HTTP {exc.code}: {detail}") from exc
        return _parse(payload)


def build_decider() -> LocalSystemOne | JevSystemOne:
    backend = os.environ.get("ROBOJEV_BACKEND", "auto")
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if backend == "local" or (backend == "auto" and not api_key):
        return LocalSystemOne()
    if not api_key:
        raise RuntimeError("ROBOJEV_BACKEND=jev requires TYPESAFE_API_KEY")
    return JevSystemOne(api_key)


def should_escalate(decision: Decision) -> bool:
    """Confidence gate: easy cases stay local, gray cases go to the reasoner."""

    if decision.recovery not in {"continue", "retry", "replan", "abort"}:
        return True
    if decision.recovery_confidence < 0.6:
        return True
    return 0.35 < decision.skill_done < 0.65


def _outcome(reading: SensorReading) -> tuple[float, str, str]:
    skill = reading.skill
    if skill.name == "move":
        ok = reading.robot_at == skill.target
        return (0.95, "none", "continue") if ok else (0.08, "blocked", "replan")
    if skill.name == "pick":
        if reading.held == skill.target and not reading.slip:
            return 0.95, "none", "continue"
        object_here = reading.loc.get(skill.target) == reading.robot_at
        if reading.slip or (object_here and reading.held != skill.target):
            return 0.08, ("slip" if reading.slip else "miss"), "retry"
        return 0.10, "blocked", "replan"
    if skill.name == "place":
        placed = reading.held is None and reading.loc.get(skill.target) == skill.dest
        if placed:
            return 0.95, "none", "continue"
        if (
            reading.container_open is False
            or reading.robot_at != skill.dest
            or reading.held != skill.target
        ):
            return 0.10, "blocked", "replan"
        return 0.12, "miss", "retry"
    if skill.name == "open":
        ok = reading.container_open is True
        return (0.95, "none", "continue") if ok else (0.10, "blocked", "replan")
    if skill.name == "close":
        ok = reading.container_open is False
        return (0.95, "none", "continue") if ok else (0.10, "blocked", "replan")
    return 0.40, "uncertain", "replan"


def _parse(payload: dict) -> Decision:
    answers = payload["answers"]
    failure = answers["failure"]
    recovery = answers["recovery"]
    return Decision(
        model=str(payload.get("model", "jev")),
        grasp_ok=float(answers["grasp_ok"]["noul"]),
        skill_done=float(answers["skill_done"]["noul"]),
        failure=str(failure["choice"]),
        failure_confidence=float(failure.get("confidence", 0.0)),
        recovery=str(recovery["choice"]),
        recovery_confidence=float(recovery.get("confidence", 0.0)),
    )
