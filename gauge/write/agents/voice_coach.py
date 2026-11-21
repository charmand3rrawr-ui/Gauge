"""The Voice Coach: keeps characters from sounding alike. Measured, no model calls."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..stylometry import (
    attribute_dialogue,
    distinctness,
    merge_profile,
    profile_summary,
    style_vector,
)
from .base import AgentReview, SceneContext, StoryAgent


class VoiceState(BaseModel):
    profiles: dict[str, dict[str, float]] = Field(default_factory=dict)
    samples: dict[str, int] = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)


class VoiceCoach(StoryAgent):
    name = "voice_coach"
    title = "Voice Coach"
    weight = 0.5
    state_model = VoiceState
    threshold = 0.35  # pairwise distance below which two characters "sound alike"

    def _names(self, ctx: SceneContext) -> list[str]:
        try:
            from ..characters import Roster

            return Roster.load(ctx.project.dir).names()
        except Exception:
            return []

    def brief(self, ctx: SceneContext) -> str:
        if len(self.state.profiles) < 2:
            return ""
        _, pairs = distinctness(self.state.profiles)
        close = [(a, b) for a, b, d in pairs if d < self.threshold]
        if not close:
            return ""
        return "\n".join(f"- {a} and {b} have been sounding alike; give one of them a different rhythm or register this chapter" for a, b in close[:2])

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        names = self._names(ctx)
        if len(names) < 2:
            return None
        lines = attribute_dialogue(text, names)
        speaking = {n: ls for n, ls in lines.items() if len(ls) >= 3}
        if len(speaking) < 2:
            return None
        # Judge this chapter's dialogue on its own; accumulated profiles are kept for the brief and report.
        profiles = {n: style_vector(ls) for n, ls in speaking.items()}
        score_d, pairs = distinctness(profiles)
        close = [(a, b, d) for a, b, d in pairs if d < self.threshold]
        issues = [f"{a} and {b} sound alike in this chapter (distance {d:.2f}); change one voice: sentence length, questions vs statements, contractions, swearing"
                  for a, b, d in close[:2]]
        self._last = {n: style_vector(ls) for n, ls in speaking.items()}
        score = max(0.0, min(1.0, 0.5 + score_d)) if close else 1.0
        return AgentReview(agent=self.name, score=score, issues=issues, data={"distinctness": round(score_d, 3), "speakers": list(speaking)})

    def remember(self, text: str, ctx: SceneContext) -> None:
        for n, vec in getattr(self, "_last", {}).items():
            k = self.state.samples.get(n, 0)
            self.state.profiles[n] = merge_profile(self.state.profiles.get(n, {}), vec, k)
            self.state.samples[n] = k + 1
        if self.state.profiles:
            d, _ = distinctness(self.state.profiles)
            self.state.history.append({"chapter": ctx.number, "distinctness": round(d, 3)})
            self.state.history = self.state.history[-100:]

    def report(self) -> str:
        lines = [f"{n}: {profile_summary(p)} ({self.state.samples.get(n, 0)} chapters)" for n, p in self.state.profiles.items()]
        if len(self.state.profiles) >= 2:
            d, pairs = distinctness(self.state.profiles)
            lines.append(f"overall distinctness {d:.2f}; closest: " + ", ".join(f"{a}/{b} {x:.2f}" for a, b, x in pairs[:3]))
        return "\n".join(lines) or "(no dialogue profiled yet)"
