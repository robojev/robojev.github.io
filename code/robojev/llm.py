"""Reasoning layer backed by an OpenAI-compatible chat model.

The decision layer never calls this. It is only used to write or replace a plan.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from robojev.reasoner import scripted_skills
from robojev.types import Belief, Skill

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM = """You are the robot Reasoning layer. You only plan; you do not judge step outcomes.
Output one JSON object and nothing else. Example:
{"skills":[{"name":"pick","target":"apple"},{"name":"move","target":"fridge"},{"name":"open","target":"fridge"},{"name":"place","target":"apple","dest":"fridge"},{"name":"close","target":"fridge"}]}
Allowed skills only:
- move, target is counter or fridge
- pick, target is apple
- open, target is fridge
- close, target is fridge
- place, target must be apple, dest must be fridge
If the apple is already in the fridge and the door is open, output only close.
If the robot is not at the fridge, move to fridge before open/place/close.
If holding the apple at the counter with the door closed: move fridge, open, place, close.
If already at the fridge holding the apple with the door closed: open, place, close.
"""


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @classmethod
    def from_env(cls) -> "LLMClient":
        api_key = os.environ.get("ROBOJEV_API_KEY", "")
        if not api_key:
            raise RuntimeError("ROBOJEV_API_KEY is not set")
        return cls(
            api_key,
            os.environ.get("ROBOJEV_BASE_URL", DEFAULT_BASE_URL),
            os.environ.get("ROBOJEV_MODEL", DEFAULT_MODEL),
        )

    def complete(self, user: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        return str(payload["choices"][0]["message"].get("content") or "")


class LLMReasoner:
    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.model = client.model
        self.calls = 0
        self.events: list[dict[str, str]] = []

    def plan(self, belief: Belief) -> list[Skill]:
        self.calls += 1
        described = describe_belief(belief)
        try:
            raw = self.client.complete(described)
            skills = parse_skills(raw)
            source = "llm"
        except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            skills = scripted_skills(belief)
            raw = ""
            source = f"scripted_fallback ({exc})"
        self.events.append(
            {
                "source": source,
                "belief": described,
                "plan": " -> ".join(skill.label() for skill in skills),
                "raw": raw[:500],
            }
        )
        return skills


def describe_belief(belief: Belief) -> str:
    if not belief.held_known:
        held = "whether the gripper holds the apple is unknown"
    elif belief.held == "apple":
        held = "the gripper is holding the apple"
    else:
        held = "the gripper is empty"
    door = belief.open.get("fridge")
    if door is True:
        door_text = "the fridge door is open"
    elif door is False:
        door_text = "the fridge door is closed"
    else:
        door_text = "the fridge door state is unknown"
    loc = belief.loc.get("apple")
    loc_text = {
        "counter": "the apple is on the counter",
        "fridge": "the apple is in the fridge",
        "gripper": "the apple is in the gripper",
    }.get(loc or "", "the apple location is unknown")
    where = (
        "the robot is at the fridge"
        if belief.at == "fridge"
        else "the robot is at the counter"
    )
    return (
        f"Task: put the apple in the fridge and close it. "
        f"Current: {where}. {held}. {loc_text}. {door_text}."
    )


def parse_skills(text: str) -> list[Skill]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("model did not return JSON")
    data = json.loads(match.group(0))
    skills: list[Skill] = []
    for item in data["skills"]:
        name = str(item["name"])
        target = str(item.get("target", ""))
        if name == "move" and target in {"counter", "fridge"}:
            skills.append(Skill("move", target))
        elif name == "pick" and target == "apple":
            skills.append(Skill("pick", "apple"))
        elif name in {"open", "close"} and target == "fridge":
            skills.append(Skill(name, "fridge"))
        elif name == "place":
            dest = item.get("dest")
            if dest is None and target == "fridge":
                target, dest = "apple", "fridge"
            if target != "apple" or dest != "fridge":
                raise ValueError(f"bad place: {item}")
            skills.append(Skill("place", "apple", "fridge"))
        else:
            raise ValueError(f"bad skill: {item}")
    if not skills:
        raise ValueError("empty plan")
    return skills
