"""Four scripted episodes. They check the loop, not a benchmark suite."""

from __future__ import annotations

from robojev.agent import BaselineAgent, RoboJevAgent
from robojev.jev import build_decider
from robojev.types import EpisodeResult, Task

TASKS = [
    Task("clean", "Put the apple in the fridge and close it", {}),
    Task("slip", "First grasp slips; Decision should retry locally", {"pick#1": "slip"}),
    Task("ambiguous", "Sensors disagree after open; should escalate", {"open#1": "ambiguous"}),
    Task("lid", "Door closes before place; should replan", {"place#1": "lid_falls"}),
]


def main() -> None:
    decider = build_decider()
    agent = RoboJevAgent(decider)
    baseline = BaselineAgent()
    robojev_runs = [agent.run(task) for task in TASKS]
    baseline_runs = [baseline.run(task) for task in TASKS]

    print(f"RoboJev minimal loop    decision={decider.model}")
    print("Task: put the apple in the fridge and close it. Symbolic world, no simulator required.")
    print()
    for result in robojev_runs:
        _print_episode(result)
    _print_table(robojev_runs, baseline_runs)
    _check(robojev_runs, baseline_runs)


def _print_episode(result: EpisodeResult) -> None:
    print(f"[{result.name}] {result.instruction}")
    for step in result.steps:
        if step.decision is None:
            print(f"  {step.index}. {step.skill.label():<22} {step.note}")
            continue
        decision = step.decision
        print(f"  {step.index}. {step.skill.label():<22} {step.note}")
        print(
            f"     skill_done={decision.skill_done:.2f}  "
            f"failure={decision.failure}  "
            f"recovery={decision.recovery} "
            f"conf={decision.recovery_confidence:.2f}  -> {step.gate}"
        )
    flag = "done" if result.success else "failed"
    print(
        f"  {flag}    reasoner={result.reasoner_calls}    "
        f"decisions={result.decision_calls}"
    )
    print()


def _print_table(robojev_runs: list[EpisodeResult], baseline_runs: list[EpisodeResult]) -> None:
    print("Versus replan-every-step, Decision absorbs most calls:")
    print(f"  {'episode':<12} {'RoboJev reas.':>12} {'Baseline reas.':>14} {'decisions':>9}")
    for left, right in zip(robojev_runs, baseline_runs):
        print(
            f"  {left.name:<12} {left.reasoner_calls:>12} "
            f"{right.reasoner_calls:>14} {left.decision_calls:>9}"
        )
    print()


def _check(robojev_runs: list[EpisodeResult], baseline_runs: list[EpisodeResult]) -> None:
    by_name = {result.name: result for result in robojev_runs}
    problems: list[str] = []
    for result in robojev_runs + baseline_runs:
        if not result.success:
            problems.append(f"{result.mode}/{result.name} did not finish the task")
    clean = by_name["clean"]
    slip = by_name["slip"]
    ambiguous = by_name["ambiguous"]
    lid = by_name["lid"]
    if clean.reasoner_calls != 1:
        problems.append(f"clean should not escalate; reasoner calls={clean.reasoner_calls}")
    if slip.reasoner_calls != 1 or not any(step.gate == "retry" for step in slip.steps):
        problems.append("slip should retry in Decision without Reasoning")
    if ambiguous.reasoner_calls != 2 or not any(step.gate == "escalate" for step in ambiguous.steps):
        problems.append("ambiguous should escalate once on low confidence")
    if lid.reasoner_calls != 2 or not any(step.gate == "replan" for step in lid.steps):
        problems.append("lid should replan with high confidence")
    for left, right in zip(robojev_runs, baseline_runs):
        if right.reasoner_calls <= left.reasoner_calls:
            problems.append(
                f"{left.name} baseline reasoner calls ({right.reasoner_calls}) "
                f"is not greater than RoboJev ({left.reasoner_calls})"
            )
    if problems:
        raise SystemExit("Self-check failed:\n- " + "\n- ".join(problems))
    print("Self-check passed.")
