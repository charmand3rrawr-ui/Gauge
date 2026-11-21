"""The Sensory Editor: grounds a scene in the senses, and stops emotion being announced.

Measured, no model calls. It scores each chapter on how many senses it engages, whether feelings are
shown or merely stated, how often filter verbs hold the reader at arm's length, and how specific the
detail is — then hands the reviser concrete fixes. This is the single biggest lever on whether prose
reads as real.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..realism import sensory_profile, sensory_score
from .base import AgentReview, SceneContext, StoryAgent


class SensoryState(BaseModel):
    history: list[dict] = Field(default_factory=list)


class SensoryEditor(StoryAgent):
    name = "sensory"
    title = "Sensory Editor"
    weight = 0.8
    state_model = SensoryState

    def brief(self, ctx: SceneContext) -> str:
        if not self.state.history:
            return "Ground the scene in more than sight: let the reader hear, smell and feel it. Show emotion through the body, not by naming it."
        last = self.state.history[-1]
        if last.get("issues"):
            return "Realism notes from last chapter to fix:\n" + "\n".join(f"- {i}" for i in last["issues"][:3])
        return ""

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        from ...text import word_count
        if word_count(text) < 100:
            return None
        p = sensory_profile(text)
        score = sensory_score(p)
        self._last = {"senses": p.senses_engaged, "telling": p.telling_emotions,
                      "filter_density": p.filter_density, "specifics": p.concrete_specifics, "issues": p.issues}
        return AgentReview(agent=self.name, score=score, issues=p.issues[:3],
                           data={"senses_engaged": p.senses_engaged, "telling_emotions": p.telling_emotions,
                                 "filter_density": p.filter_density, "concrete_specifics": p.concrete_specifics})

    def remember(self, text: str, ctx: SceneContext) -> None:
        last = getattr(self, "_last", None)
        if last:
            self.state.history.append({"chapter": ctx.number, **last})
            self.state.history = self.state.history[-200:]

    def report(self) -> str:
        if not self.state.history:
            return "(no chapters measured yet)"
        rows = self.state.history[-10:]
        avg = lambda k: sum(r.get(k, 0) for r in rows) / len(rows)  # noqa: E731
        return (f"last {len(rows)} chapters: {avg('senses'):.1f} senses engaged, "
                f"{avg('telling'):.1f} told-emotions, {avg('filter_density'):.2f} filter words/sentence")
