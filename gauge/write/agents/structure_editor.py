"""The Structure Editor: try-fail cost, conflict, cliffhanger variety and beat placement (section D).

Measured, no model calls. It tracks the ending type of recent chapters to catch repetition, flags a
chapter with no conflict or a win that cost nothing, and checks each chapter against the structural beat
its arc position implies. Complements the Showrunner (promises) and the Pacing Director (scene shape).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..structure import report
from .base import AgentReview, SceneContext, StoryAgent


class StructureState(BaseModel):
    recent_cliffs: list[str] = Field(default_factory=list)


class StructureEditor(StoryAgent):
    name = "structure"
    title = "Structure Editor"
    weight = 0.6
    state_model = StructureState

    def _total(self, ctx: SceneContext) -> int:
        try:
            return len(ctx.project.state.outline) or ctx.project.state.target_chapters
        except Exception:
            return 0

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        from ...text import word_count
        if word_count(text) < 150:
            return None
        rep = report(text, number=ctx.number, total=self._total(ctx), recent_cliffs=self.state.recent_cliffs)
        self._ct = rep.cliff_type
        score = 1.0 if not rep.issues else max(0.5, 1.0 - 0.12 * len(rep.issues))
        return AgentReview(agent=self.name, score=round(score, 3), issues=rep.issues[:3],
                           data={"cliff_type": rep.cliff_type, "conflict": rep.conflict,
                                 "beat": rep.expected_beat, "tension_shape": rep.tension_shape})

    def remember(self, text: str, ctx: SceneContext) -> None:
        ct = getattr(self, "_ct", None)
        if ct:
            self.state.recent_cliffs.append(ct)
            self.state.recent_cliffs = self.state.recent_cliffs[-30:]

    def report(self) -> str:
        if not self.state.recent_cliffs:
            return "(no chapters tracked yet)"
        return "recent endings: " + ", ".join(self.state.recent_cliffs[-8:])
