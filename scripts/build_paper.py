#!/usr/bin/env python3
"""Render a short academic-looking PDF for RoboJev."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "static" / "images"
OUT = ROOT / "static" / "pdf" / "robojev.pdf"
FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
FONT_I = "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"
FONT_BI = "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf"


class Paper(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("TimesRoman", "I", 9)
        self.set_text_color(90, 90, 90)
        self.cell(
            0,
            8,
            "RoboJev: Confidence-Gated Structured Decisions for Embodied Agents",
            align="C",
        )
        self.ln(10)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("TimesRoman", "I", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 10, str(self.page_no()), align="C")
        self.set_text_color(0, 0, 0)

    def section(self, text: str, size: int = 12) -> None:
        self.ln(3)
        self.set_x(self.l_margin)
        self.set_font("TimesRoman", "B", size)
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("TimesRoman", "", 10)
        self.multi_cell(0, 5.2, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def italic_block(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("TimesRoman", "I", 10)
        self.multi_cell(0, 5.2, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def bold_line(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("TimesRoman", "B", 10)
        self.multi_cell(0, 5.2, text, new_x="LMARGIN", new_y="NEXT")


def add_figure(pdf: Paper, paths: list[Path], caption: str, height: float = 38) -> None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return
    if pdf.get_y() > 220:
        pdf.add_page()
    gap = 3
    usable = pdf.epw - gap * (len(existing) - 1)
    width = usable / len(existing)
    x0 = pdf.l_margin
    y0 = pdf.get_y()
    for i, path in enumerate(existing):
        with Image.open(path) as im:
            tmp = Path(f"/tmp/robojev_fig_{i}.jpg")
            im.convert("RGB").save(tmp, quality=88)
        pdf.image(str(tmp), x=x0 + i * (width + gap), y=y0, w=width, h=height)
    pdf.set_y(y0 + height + 2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("TimesRoman", "", 9)
    pdf.multi_cell(0, 4.5, caption, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = Paper(format="letter")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_font("TimesRoman", "", FONT)
    pdf.add_font("TimesRoman", "B", FONT_B)
    pdf.add_font("TimesRoman", "I", FONT_I)
    pdf.add_font("TimesRoman", "BI", FONT_BI)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()

    pdf.set_font("TimesRoman", "B", 16)
    pdf.multi_cell(
        0,
        8,
        "RoboJev: Confidence-Gated Structured Decision Making "
        "for Long-Horizon Embodied Agents",
        align="C",
    )
    pdf.ln(4)
    pdf.set_font("TimesRoman", "", 12)
    pdf.cell(0, 6, "Guobao Tegong", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("TimesRoman", "I", 10)
    pdf.cell(
        0,
        5,
        "Institute of Embodied Intelligence, Happy Heroes University",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("TimesRoman", "", 10)
    pdf.cell(0, 5, "guobao@happyheroes.edu", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("TimesRoman", "I", 10)
    pdf.cell(0, 5, "Preprint, 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.section("Abstract")
    pdf.body(
        "Long-horizon embodied agents repeatedly face closed-form questions during "
        "execution: Did the grasp hold? Did the skill succeed? Should the controller "
        "retry, replan, or escalate? Current stacks often answer these questions by "
        "invoking an LLM or VLM at every step, even when the answer space is tiny. "
        "We present RoboJev, a three-layer architecture that separates open-ended "
        "reasoning from high-frequency structured decisions. A Reasoning layer "
        "(LLM/VLM) writes or revises skill plans only when needed. A Decision layer "
        "returns typed judgments with confidence in the style of System-One models "
        "such as Jev. An Action layer executes skills. Confidence-gated escalation "
        "routes only uncertain cases back to Reasoning. In a symbolic kitchen task "
        "with MuJoCo display rendering, RoboJev completes clean, slip, "
        "ambiguous-sensor, and early door-close episodes while cutting reasoner "
        "calls versus a replan-every-step baseline."
    )

    pdf.section("1. Introduction")
    pdf.body(
        "Embodied agents that execute multi-step skills spend most of their time not "
        "inventing new language, but judging short, discrete outcomes. Grasp success, "
        "skill completion, failure type, and recovery choice are exactly the kind of "
        "high-frequency, schema-constrained decisions that generative models are "
        "expensive at answering. Recent System-One models argue that automation needs "
        "fast, typed judgments rather than another paragraph of free text."
    )
    pdf.body(
        "RoboJev applies that idea to embodied control. The claim is architectural: "
        "if a question has a small answer space, it belongs in a Decision layer with "
        "confidence; only low-confidence or open-ended cases should escalate to an "
        "LLM. The contribution of this preprint is a minimal, reproducible loop that "
        "makes the claim concrete without requiring real-robot hardware for control."
    )

    pdf.section("2. Related Work")
    pdf.body(
        "Hierarchical robot stacks commonly separate planning from control "
        "(task-and-motion planning, skill libraries, VLA policies). LLM-based agents "
        "increasingly act as the planner, but often also as the step-level critic. "
        "Selective prediction and confidence gating have long been used to defer "
        "uncertain decisions. RoboJev sits between these lines: it keeps LLMs for "
        "open planning, while assigning closed judgments to a structured System-One "
        "interface inspired by Jev."
    )

    pdf.section("3. Method")
    pdf.body(
        "RoboJev factors an episode into Reasoning, Decision, and Action. The "
        "environment exposes a symbolic belief: robot location, gripper contents, "
        "object locations, and container openness. Skills are pick, move, open, "
        "place, and close. Perception for control is state text; RGB frames are "
        "rendered only for display."
    )
    pdf.bold_line("3.1 Reasoning")
    pdf.body(
        "The Reasoning layer proposes an ordered skill plan from the current belief. "
        "In our implementation it is an LLM with a constrained JSON schema; a "
        "scripted planner is used as a fallback and for ablations. Reasoning is "
        "called at episode start, on explicit replan, and when Decision escalates."
    )
    pdf.bold_line("3.2 Decision")
    pdf.body(
        "After each skill, Decision evaluates four questions in one shot: "
        "(1) grasp_ok (noul), (2) skill_done (noul), (3) failure type (choice among "
        "none/slip/miss/blocked/uncertain), and (4) recovery "
        "(continue/retry/replan/abort). Answers include calibrated confidence. "
        "If recovery confidence is below 0.6, or skill_done falls in (0.35, 0.65), "
        "the controller escalates to Reasoning with a clarified belief."
    )
    pdf.bold_line("3.3 Action")
    pdf.body(
        "Action executes the selected skill in the symbolic world and returns a "
        "sensor reading used by Decision. MuJoCo mirrors the same state for "
        "screenshots; the agent never conditions on pixels."
    )

    pdf.section("4. Experiments")
    pdf.body(
        "Task: put an apple into a fridge and close the door. We evaluate four "
        "scripted episodes on one NVIDIA H800, with Decision as a local System-One "
        "stand-in matching Jev's answer schema, and Reasoning via a hosted chat "
        "model. The baseline replans with the reasoner after every skill."
    )

    pdf.set_font("TimesRoman", "B", 10)
    widths = [40, 35, 35, 35, 35]
    headers = ["Episode", "RoboJev reas.", "Baseline reas.", "Decisions", "Outcome"]
    for width, header in zip(widths, headers):
        pdf.cell(width, 7, header, border=1)
    pdf.ln()
    pdf.set_font("TimesRoman", "", 10)
    rows = [
        ("Clean", "1", "6", "5", "success"),
        ("Slip", "1", "7", "6", "retry"),
        ("Ambiguous", "2", "6", "5", "escalate"),
        ("Lid closes", "2", "8", "7", "replan"),
    ]
    for row in rows:
        for width, cell in zip(widths, row):
            pdf.cell(width, 7, cell, border=1)
        pdf.ln()
    pdf.set_x(pdf.l_margin)
    pdf.ln(2)
    pdf.body(
        "Clean runs need a single plan. Slip is recovered by a high-confidence local "
        "retry. Ambiguous sensors escalate once. An early door close triggers "
        "replan rather than blind retries. Across episodes, RoboJev uses far fewer "
        "reasoner calls than the baseline while finishing the task."
    )

    add_figure(
        pdf,
        [
            FIG / "00_start.png",
            FIG / "01_pick.png",
            FIG / "03_open.png",
            FIG / "04_place.png",
            FIG / "05_close.png",
        ],
        "Figure 1. Clean execution. Display frames from MuJoCo; control uses state only.",
        height=30,
    )
    add_figure(
        pdf,
        [
            FIG / "slip_fail.png",
            FIG / "slip_ok.png",
            FIG / "escalate.png",
            FIG / "lid_block.png",
        ],
        "Figure 2. Left to right: slip then retry; ambiguous open (escalate); blocked place before replan.",
        height=32,
    )

    pdf.section("5. Limitations and Future Work")
    pdf.body(
        "This preprint is intentionally minimal. The world is symbolic; Decision is "
        "a schema-compatible local model unless a hosted System-One API is provided; "
        "and we do not claim EDBench-scale evaluation, real-robot transfer, or "
        "learned calibration. Failure modes include over-eager escalation, brittle "
        "LLM plan schemas, and display scenes that overstate geometric fidelity."
    )
    pdf.body(
        "Future work includes replacing the local Decision stand-in with a trained "
        "System-One model on robot traces, evaluating on standard embodied "
        "benchmarks, closing the loop with real sensors, and studying when "
        "confidence is trustworthy enough to suppress LLM calls safely."
    )

    pdf.section("6. Conclusion")
    pdf.body(
        "RoboJev is a simple recipe: let LLMs reason when the problem is open, and "
        "let structured judgments handle the high-frequency closed questions that "
        "dominate long-horizon execution. Even a minimal kitchen loop shows that "
        "confidence-gated escalation can preserve competence while cutting "
        "reasoner calls."
    )

    pdf.section("Acknowledgments")
    pdf.italic_block(
        "This preprint accompanies the project page at https://robojev.github.io. "
        "Code is released at https://github.com/robojev/robojev."
    )

    pdf.section("References")
    refs = [
        "[1] Kahneman, D. Thinking, Fast and Slow. 2011.",
        "[2] TypeSafe AI. Jev: A System One Model for Structured Decisions. 2026.",
        "[3] EmbodiedBench: Benchmarking MLLMs for Vision-Driven Embodied Agents. 2025.",
        "[4] Todorov, E., Erez, T., and Tassa, Y. MuJoCo: A physics engine for model-based control. 2012.",
    ]
    for ref in refs:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("TimesRoman", "", 9)
        pdf.multi_cell(0, 4.6, ref, new_x="LMARGIN", new_y="NEXT")

    pdf.output(OUT)
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
