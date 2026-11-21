"""The Suitability Checker: an explicit contract with the audience, enforced every chapter.

A project can declare who it is for — platform, age band, how much heat it allows, and content
lines it will never cross — and this agent measures each chapter against that contract instead
of discovering a mismatch after publication. It is deliberately *measured and transparent*: it
counts and matches, it does not moralise. Editorial judgement about depictions stays with the
Sensitivity Reader; this is about fit with a stated audience.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from ...text import flesch_reading_ease, word_count
from .base import AgentReview, SceneContext, StoryAgent

# Transparent heat markers by level. 1 = mild (romance/tension), 2 = moderate, 3 = explicit.
HEAT_MARKERS: dict[int, set[str]] = {
    1: {"kiss", "kissed", "blush", "embrace", "longing", "desire", "bare", "undress"},
    2: {"moan", "thrust", "naked", "arousal", "panting", "grind", "straddle"},
    3: {"explicit-anatomical-terms-left-to-the-platform"},  # placeholder; real lists are platform-specific and pasted in
}
AGE_READING_EASE = {"all": 80.0, "teen": 60.0, "adult": 40.0}  # minimum Flesch expected for the band


class SuitabilityState(BaseModel):
    history: list[dict] = Field(default_factory=list)


def heat_level(text: str) -> int:
    low = text.lower()
    level = 0
    for lvl, markers in HEAT_MARKERS.items():
        if any(re.search(rf"\b{re.escape(m)}\b", low) for m in markers):
            level = max(level, lvl)
    return level


class SuitabilityChecker(StoryAgent):
    name = "suitability"
    title = "Suitability Checker"
    weight = 0.6
    state_model = SuitabilityState

    def _contract(self) -> dict[str, Any]:
        o = self.options or {}
        return {
            "platform": o.get("platform", ""),
            "age_band": (o.get("age_band") or "teen"),
            "heat_max": int(o.get("heat_max", 1)),
            "content_lines": [str(x).strip() for x in (o.get("content_lines") or []) if str(x).strip()],
            "word_target": int(o.get("word_target", 0) or 0),
            "word_tolerance": float(o.get("word_tolerance", 0.4)),
        }

    def brief(self, ctx: SceneContext) -> str:
        c = self._contract()
        bits = []
        if c["platform"]:
            bits.append(f"platform: {c['platform']}")
        bits.append(f"audience: {c['age_band']}, heat up to level {c['heat_max']}")
        if c["content_lines"]:
            bits.append("never include: " + ", ".join(c["content_lines"][:8]))
        return "Audience contract — " + "; ".join(bits)

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        c = self._contract()
        issues: list[str] = []
        low = text.lower()
        crossed = [line for line in c["content_lines"] if line.lower() in low]
        if crossed:
            issues.append("content line crossed: " + ", ".join(crossed) + " — remove entirely")
        heat = heat_level(text)
        if heat > c["heat_max"]:
            issues.append(f"heat level {heat} exceeds the contract's {c['heat_max']}; dial the intimate content back")
        ease = flesch_reading_ease(text)
        floor = AGE_READING_EASE.get(c["age_band"], 60.0)
        if c["age_band"] in ("all", "teen") and ease < floor - 10:
            issues.append(f"reading ease {ease:.0f} is below the {c['age_band']} band (~{floor:.0f}); simplify sentences and vocabulary")
        wc = word_count(text)
        if c["word_target"]:
            lo = c["word_target"] * (1 - c["word_tolerance"])
            hi = c["word_target"] * (1 + c["word_tolerance"])
            if wc < lo:
                issues.append(f"chapter is short for this serial ({wc} words; aim near {c['word_target']})")
            elif wc > hi:
                issues.append(f"chapter runs long ({wc} words; aim near {c['word_target']})")
        data = {"heat_level": heat, "heat_max": c["heat_max"], "reading_ease": round(ease, 1), "words": wc,
                "crossed_lines": crossed}
        self._last = data
        # A crossed content line or over-heat is blocking only when the agent is configured to block;
        # otherwise it is a strong warning (low score). Weight 0 from the council would make it blocking.
        score = 1.0 if not issues else max(0.0, 1.0 - 0.34 * len(issues))
        if crossed and self.options.get("block", False):
            score = 0.0
        return AgentReview(agent=self.name, score=round(score, 3), issues=issues[:4], data=data)

    def remember(self, text: str, ctx: SceneContext) -> None:
        last = getattr(self, "_last", None)
        if last:
            self.state.history.append({"chapter": ctx.number, **last})
            self.state.history = self.state.history[-200:]

    def report(self) -> str:
        c = self._contract()
        n_cross = sum(1 for h in self.state.history if h.get("crossed_lines"))
        return (f"contract: {c['age_band']}, heat<= {c['heat_max']}; "
                f"{len(self.state.history)} chapters checked, {n_cross} with crossed lines")
