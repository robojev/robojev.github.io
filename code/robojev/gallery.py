"""Run RoboJev and photograph the same state from the simulator."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from robojev.agent import RoboJevAgent
from robojev.demo import TASKS
from robojev.jev import build_decider
from robojev.llm import LLMClient, LLMReasoner
from robojev.scene import KitchenRenderer
from robojev.types import EpisodeResult

GATE_TEXT = {
    "continue": "continue",
    "retry": "retry",
    "replan": "replan",
    "escalate": "escalate",
    "abort": "abort",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/gallery"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    decider = build_decider()
    client = LLMClient.from_env()
    agent = RoboJevAgent(decider)
    camera = KitchenRenderer()
    episodes: list[tuple[EpisodeResult, LLMReasoner]] = []

    for task in TASKS:
        reasoner = LLMReasoner(client)
        folder = args.out / task.name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        frames: list[dict[str, str]] = []

        def on_start(world, folder=folder, frames=frames) -> None:
            relative = f"{task.name}/00_start.png"
            camera.snap(world, None, "", args.out / relative)
            frames.append({"image": relative, "title": "start", "caption": "Episode start."})

        def on_step(world, step, folder=folder, frames=frames) -> None:
            relative = f"{task.name}/{step.index:02d}_{step.skill.name}.png"
            camera.snap(world, step.skill, step.note, args.out / relative)
            decision = step.decision
            frames.append(
                {
                    "image": relative,
                    "title": step.skill.label(),
                    "caption": (
                        f"{step.note}. skill_done={decision.skill_done:.2f}, "
                        f"failure={decision.failure}, recovery={decision.recovery}, "
                        f"confidence={decision.recovery_confidence:.2f}, "
                        f"gate={GATE_TEXT.get(step.gate, step.gate)}."
                    ),
                }
            )

        result = agent.run(task, reasoner, on_start=on_start, on_step=on_step)
        episodes.append((result, reasoner))
        (folder / "trace.json").write_text(
            json.dumps(
                {
                    "task": task.name,
                    "instruction": task.instruction,
                    "success": result.success,
                    "model": client.model,
                    "reasoner_calls": result.reasoner_calls,
                    "decision_calls": result.decision_calls,
                    "plans": reasoner.events,
                    "steps": [
                        {
                            "skill": step.skill.label(),
                            "note": step.note,
                            "gate": step.gate,
                            "failure": None if step.decision is None else step.decision.failure,
                            "recovery": None if step.decision is None else step.decision.recovery,
                            "confidence": None
                            if step.decision is None
                            else step.decision.recovery_confidence,
                        }
                        for step in result.steps
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"{task.name}: success={result.success} "
            f"reasoner={result.reasoner_calls} decisions={result.decision_calls} "
            f"gates={[step.gate for step in result.steps]}"
        )

    _write_html(args.out, episodes, client.model, decider.model)
    _check(episodes)
    print(f"gallery: {args.out / 'index.html'}")


def _write_html(out: Path, episodes, llm_model: str, decision_model: str) -> None:
    gpu = ""
    gpu_path = out / "gpu.txt"
    if gpu_path.exists():
        gpu = gpu_path.read_text(encoding="utf-8").strip()
    sections = []
    for result, reasoner in episodes:
        folder = out / result.name
        cards = sorted(folder.glob("*.png"))
        caption_by_name = {
            f"{result.name}/00_start.png": "Start. Display only; control does not read pixels."
        }
        for step, image in zip(
            result.steps, [path for path in cards if path.name != "00_start.png"]
        ):
            decision = step.decision
            caption_by_name[f"{result.name}/{image.name}"] = (
                f"{step.skill.label()}. {step.note}. "
                f"skill_done={decision.skill_done:.2f}, failure={decision.failure}, "
                f"recovery={decision.recovery}, confidence={decision.recovery_confidence:.2f}, "
                f"gate={GATE_TEXT.get(step.gate, step.gate)}."
            )
        plans = "".join(
            f"<li>{html.escape(event['source'])}: {html.escape(event['plan'])}</li>"
            for event in reasoner.events
        )
        figures = []
        for image in cards:
            key = f"{result.name}/{image.name}"
            figures.append(
                "<figure>"
                f"<img src='{html.escape(key)}' alt='{html.escape(image.stem)}'>"
                f"<figcaption>{html.escape(caption_by_name.get(key, image.name))}</figcaption>"
                "</figure>"
            )
        flag = "success" if result.success else "failed"
        sections.append(
            f"<section><h2>{html.escape(result.instruction)}</h2>"
            f"<p class='meta'>{html.escape(flag)}. reasoner={result.reasoner_calls}, "
            f"decisions={result.decision_calls}.</p>"
            f"<ol>{plans}</ol><div class='strip'>{''.join(figures)}</div></section>"
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RoboJev Gallery</title>
<style>
  body {{ margin: 0; background: #f6f3ec; color: #1d1a16; font: 16px/1.5 Georgia, serif; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; }}
  h1 {{ font-size: 40px; margin-bottom: 8px; }}
  h2 {{ font-size: 26px; margin: 36px 0 4px; }}
  .lede, .meta {{ color: #5c564c; }}
  .strip {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }}
  figure {{ margin: 0; background: white; border-radius: 8px; overflow: hidden; }}
  img {{ width: 100%; display: block; background: #ddd; }}
  figcaption {{ padding: 10px 12px 14px; font: 13px/1.45 sans-serif; }}
</style>
</head>
<body>
<main>
  <h1>RoboJev</h1>
  <p class="lede">Reasoning={html.escape(llm_model)}; Decision={html.escape(decision_model)}.
  Frames are display-only.</p>
  <p class="meta">{html.escape(gpu)}</p>
  {''.join(sections)}
</main>
</body>
</html>
"""
    (out / "index.html").write_text(page, encoding="utf-8")


def _check(episodes: list[tuple[EpisodeResult, LLMReasoner]]) -> None:
    problems = []
    for result, reasoner in episodes:
        gates = [step.gate for step in result.steps]
        if not result.success:
            problems.append(f"{result.name} did not finish")
        if any(event["source"] != "llm" for event in reasoner.events):
            problems.append(f"{result.name} did not use the model: {reasoner.events}")
        if result.name == "slip" and "retry" not in gates:
            problems.append("slip missing retry")
        if result.name == "ambiguous" and "escalate" not in gates:
            problems.append("ambiguous missing escalate")
        if result.name == "lid" and "replan" not in gates:
            problems.append("lid missing replan")
        if result.name == "clean" and any(
            gate in {"escalate", "replan", "retry"} for gate in gates
        ):
            problems.append(f"clean should not interrupt: {gates}")
    if problems:
        raise SystemExit("Self-check failed:\n- " + "\n- ".join(problems))


if __name__ == "__main__":
    main()
