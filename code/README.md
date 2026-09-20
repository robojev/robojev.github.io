# RoboJev

**RoboJev: Confidence-Gated Structured Decision Making for Long-Horizon Embodied Agents**

[Project page](https://robojev.github.io) · [Paper (PDF)](https://robojev.github.io/static/pdf/robojev.pdf) · [Code](https://github.com/robojev/robojev.github.io/tree/main/code)

Author: **Guobao Tegong** (Institute of Embodied Intelligence, Happy Heroes University)

## Idea

Long-horizon robot agents keep asking closed questions during execution:

- Did the grasp hold?
- Did the skill finish?
- Retry, replan, or escalate?

Those questions often trigger another LLM/VLM call. RoboJev splits the stack:

| Layer | Role |
| --- | --- |
| Reasoning | LLM/VLM writes or revises a skill plan |
| Decision | Typed System-One judgments with confidence |
| Action | Skill controllers (`pick` / `move` / `open` / `place` / `close`) |

Low confidence escalates to Reasoning. High-confidence retries stay local.

## Install

```bash
pip install -e .
# optional MuJoCo gallery rendering
pip install -e ".[sim]"
```

Python 3.10+ only for the core symbolic loop.

## Quickstart

```bash
python -m robojev
```

This runs four scripted episodes (clean / slip / ambiguous / lid) and compares against a replan-every-step baseline.

Optional LLM reasoning:

```bash
export ROBOJEV_API_KEY=...
export ROBOJEV_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint
export ROBOJEV_MODEL=gpt-4o-mini
python -m robojev.gallery --out outputs/gallery
```

Optional real Jev Decision backend:

```bash
export TYPESAFE_API_KEY=...
python -m robojev
```

## Layout

```
robojev/
  agent.py      # Reasoning / Decision / Action loop
  jev.py        # Decision layer (local or Jev API)
  llm.py        # Reasoning layer (OpenAI-compatible)
  world.py      # Symbolic kitchen
  scene.py      # MuJoCo display renderer
  gallery.py    # Episode runner + HTML gallery
docs/figures/   # Example frames from the paper
```

## Citation

```bibtex
@misc{tegong2026robojev,
  title={RoboJev: Confidence-Gated Structured Decision Making for Long-Horizon Embodied Agents},
  author={Tegong, Guobao},
  year={2026},
  howpublished={\url{https://robojev.github.io}},
  note={Preprint}
}
```

## License

MIT
