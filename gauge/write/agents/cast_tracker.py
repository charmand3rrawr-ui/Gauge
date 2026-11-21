"""The Cast Tracker: keeps the ensemble alive and in motion (roadmap section B).

Measured, no model calls. It remembers who has appeared and flags a major character who has dropped out
of the story, then folds in the ensemble checks (missing wants/needs, frozen relationships, unused
voices). It complements the Voice Coach (how characters sound) and the character agents (who they are).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..ensemble import report
from .base import AgentReview, SceneContext, StoryAgent


class CastState(BaseModel):
    last_seen: dict[str, int] = Field(default_factory=dict)  # name -> chapter number last appeared
    chapters: int = 0


class CastTracker(StoryAgent):
    name = "cast_tracker"
    title = "Cast Tracker"
    weight = 0.6
    state_model = CastState

    def _roster(self, ctx: SceneContext):
        try:
            from ..characters import Roster
            return Roster.load(ctx.project.dir)
        except Exception:
            return None

    def _gaps(self, number: int, roster) -> dict[str, int]:
        # Every major character gets a gap: chapters since last seen (the whole story so far if never seen).
        gaps = {}
        for c in roster.ordered():
            if c.role in ("protagonist", "antagonist", "supporting"):
                gaps[c.name] = number - self.state.last_seen.get(c.name, 0)
        return gaps

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        roster = self._roster(ctx)
        if roster is None or not roster.names():
            return None
        rep = report(text, roster, appearance_gap=self._gaps(ctx.number, roster))
        self._appeared = rep.appeared
        score = 1.0 if not rep.issues else max(0.5, 1.0 - 0.12 * len(rep.issues))
        return AgentReview(agent=self.name, score=round(score, 3), issues=rep.issues[:3],
                           data={"appeared": rep.appeared, "absent_major": rep.absent_major})

    def remember(self, text: str, ctx: SceneContext) -> None:
        for name in getattr(self, "_appeared", []):
            self.state.last_seen[name] = ctx.number
        self.state.chapters = max(self.state.chapters, ctx.number)

    def report(self) -> str:
        if not self.state.last_seen:
            return "(no cast tracked yet)"
        rows = sorted(self.state.last_seen.items(), key=lambda kv: kv[1])
        return "last seen — " + ", ".join(f"{n}: ch{c}" for n, c in rows[:8])
