"""Prompt settings as an experiment: a factored Thompson-sampling bandit per project.

Several knobs change the writing (how many exemplars, temperature, how many technique cards,
how hard the world pushes, how much of the style guide is shown). Instead of fixing them, each
chapter samples a value for every knob from a Beta posterior, writes with those settings, and
the reward (reader-panel click-through, later real retention) updates the posteriors. Knobs are
treated as independent so a handful of chapters is enough to start learning.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

KNOBS: dict[str, list[Any]] = {
    "exemplars": [2, 4, 6],
    "temperature": [0.75, 0.9, 1.05],
    "techniques": [0, 2, 4],
    "world_agency": [0.4, 0.6, 0.8],
    "guide_lines": [8, 14, 20],
}


class Arm(BaseModel):
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class BanditState(BaseModel):
    arms: dict[str, dict[str, Arm]] = Field(default_factory=dict)  # knob -> value(str) -> Arm
    pending: dict[int, dict[str, Any]] = Field(default_factory=dict)  # chapter -> settings awaiting delayed reward
    history: list[dict[str, Any]] = Field(default_factory=list)


class PromptBandit:
    def __init__(self, path: Path, seed: int | None = None):
        self.path = path
        self.rng = random.Random(seed)
        self.state = BanditState()
        if path.exists():
            try:
                self.state = BanditState(**json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                self.state = BanditState()
        for knob, values in KNOBS.items():
            self.state.arms.setdefault(knob, {})
            for v in values:
                self.state.arms[knob].setdefault(str(v), Arm())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.state.model_dump_json(indent=1), encoding="utf-8")

    def sample(self, chapter: int) -> dict[str, Any]:
        """Thompson sampling: draw from each knob's posteriors, pick the best draw."""
        settings: dict[str, Any] = {}
        for knob, values in KNOBS.items():
            best, best_draw = values[0], -1.0
            for v in values:
                arm = self.state.arms[knob][str(v)]
                draw = self.rng.betavariate(arm.alpha, arm.beta)
                if draw > best_draw:
                    best, best_draw = v, draw
            settings[knob] = best
        self.state.pending[chapter] = dict(settings)
        return settings

    def update(self, chapter: int, reward: float, weight: float = 1.0, kind: str = "immediate") -> bool:
        """reward in 0..1. Immediate rewards (panel score) count once; delayed real retention can
        update the same chapter's settings again with a larger weight."""
        settings = self.state.pending.get(chapter)
        if settings is None:
            return False
        r = max(0.0, min(1.0, reward))
        for knob, v in settings.items():
            arm = self.state.arms.get(knob, {}).get(str(v))
            if arm is None:
                continue
            arm.alpha += weight * r
            arm.beta += weight * (1.0 - r)
            arm.pulls += 1
        self.state.history.append({"chapter": chapter, "reward": round(r, 3), "kind": kind, **settings})
        self.state.history = self.state.history[-500:]
        if kind != "immediate":
            self.state.pending.pop(chapter, None)
        return True

    def best(self) -> dict[str, Any]:
        out = {}
        for knob, values in KNOBS.items():
            out[knob] = max(values, key=lambda v: self.state.arms[knob][str(v)].mean)
        return out

    def report(self) -> str:
        lines = []
        for knob, values in KNOBS.items():
            cells = "  ".join(f"{v}: {self.state.arms[knob][str(v)].mean:.2f} ({self.state.arms[knob][str(v)].pulls})" for v in values)
            lines.append(f"{knob:<13} {cells}")
        if self.state.history:
            lines.append(f"rewards so far: {len(self.state.history)}, last {self.state.history[-1]['reward']}")
        return "\n".join(lines)
