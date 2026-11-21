"""The Arc Keeper: stop a long serial from drifting away from the story it promised.

A serial writes chapter by chapter, and past thirty or so chapters it quietly forgets its own arc.
The bible declares a long arc (a handful of beats from "discovers the power" to the final turn); this
agent tracks how far the story has actually travelled along it, tells the planner and drafter which
beat is due, and penalises a stretch of chapters that advance the plot locally while the big arc
stands still. It is measured and cheap — keyword overlap against the arc beats, no model calls — so it
runs every chapter without cost. The long payoff is what retains readers; this guards it.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..muse import _tokens
from .base import AgentReview, SceneContext, StoryAgent

_STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "her", "his", "their",
         "they", "she", "he", "it", "is", "are", "was", "were", "be", "as", "at", "by", "that", "this"}


def _keywords(text: str) -> set[str]:
    return {t for t in _tokens(text) if len(t) > 3 and t not in _STOP}


def beat_overlap(chapter_text: str, beat: str) -> float:
    """Share of the arc beat's keywords that appear in the chapter."""
    kb = _keywords(beat)
    if not kb:
        return 0.0
    kc = _keywords(chapter_text)
    return len(kb & kc) / len(kb)


class ArcState(BaseModel):
    reached: int = 0  # index of the next arc beat still to be delivered
    chapters_since_progress: int = 0
    history: list[dict] = Field(default_factory=list)


class ArcKeeper(StoryAgent):
    name = "arc_keeper"
    title = "Arc Keeper"
    weight = 1.0
    state_model = ArcState
    advance_threshold = 0.34   # keyword overlap at which a beat counts as delivered
    stall_limit = 4            # chapters of no arc progress before it complains

    def _arc(self, ctx: SceneContext) -> list[str]:
        try:
            return list(ctx.project.state.bible.arc or [])
        except Exception:
            return []

    def _target(self, arc: list[str]) -> Optional[str]:
        if 0 <= self.state.reached < len(arc):
            return arc[self.state.reached]
        return None

    def plan_notes(self, project) -> str:
        arc = list(getattr(project.state.bible, "arc", []) or [])
        target = self._target(arc)
        if not target:
            return ""
        note = f"Arc progress: {self.state.reached}/{len(arc)} beats delivered. The next arc beat due is: {target}"
        if self.state.chapters_since_progress >= self.stall_limit:
            note += f" — the big arc has not moved in {self.state.chapters_since_progress} chapters; advance it now."
        return note

    def brief(self, ctx: SceneContext) -> str:
        target = self._target(self._arc(ctx))
        if not target:
            return ""
        return f"This chapter should move the overall arc toward: {target}"

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        arc = self._arc(ctx)
        target = self._target(arc)
        if not target:
            return None
        ov = beat_overlap(text, target)
        advanced = ov >= self.advance_threshold
        self._advanced = advanced  # consumed in remember()
        stall = self.state.chapters_since_progress + (0 if advanced else 1)
        issues: list[str] = []
        score = 1.0
        if not advanced and stall >= self.stall_limit:
            score = max(0.3, 1.0 - 0.15 * (stall - self.stall_limit + 1))
            issues.append(f"the overall arc has stalled for {stall} chapters; this chapter should deliver: {target}")
        elif not advanced:
            issues.append(f"little progress on the current arc beat ({ov:.0%} match): {target}")
        return AgentReview(agent=self.name, score=round(score, 3), issues=issues,
                           data={"reached": self.state.reached, "beats": len(arc), "beat_overlap": round(ov, 3), "stall": stall})

    def remember(self, text: str, ctx: SceneContext) -> None:
        arc = self._arc(ctx)
        if not self._target(arc):
            return
        advanced = getattr(self, "_advanced", False)
        if advanced:
            self.state.reached = min(self.state.reached + 1, len(arc))
            self.state.chapters_since_progress = 0
        else:
            self.state.chapters_since_progress += 1
        self.state.history.append({"chapter": ctx.number, "reached": self.state.reached,
                                   "stall": self.state.chapters_since_progress})
        self.state.history = self.state.history[-200:]

    def report(self) -> str:
        return (f"arc {self.state.reached} beats delivered; "
                f"{self.state.chapters_since_progress} chapters since the arc last advanced")
