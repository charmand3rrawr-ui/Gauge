"""The Craft Editor: the line-level craft checks of roadmap section A, in one measured agent.

Clichés, same-y sentence openings, weak verb+adverb pairs, on-the-nose dialogue, talking-heads runs,
tense drift, head-hopping and period anachronisms — caught deterministically and handed to the reviser.
No model calls. It complements the Line Editor (filter words, rhythm) and the Sensory Editor (grounding).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..prose import profile, score
from .base import AgentReview, SceneContext, StoryAgent


class CraftState(BaseModel):
    history: list[dict] = Field(default_factory=list)


class CraftEditor(StoryAgent):
    name = "craft_editor"
    title = "Craft Editor"
    weight = 0.6
    state_model = CraftState

    def _names(self, ctx: SceneContext) -> list[str]:
        try:
            from ..characters import Roster
            return Roster.load(ctx.project.dir).names()
        except Exception:
            return []

    def _era(self, ctx: SceneContext) -> str:
        try:
            return str(getattr(ctx.project.state.bible, "era", "") or self.options.get("era", ""))
        except Exception:
            return ""

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        from ...text import word_count
        if word_count(text) < 120:
            return None
        p = profile(text, names=self._names(ctx), era=self._era(ctx), limited_pov=bool(self.options.get("limited_pov")))
        s = score(p)
        self._last = {"cliches": len(p.cliches), "weak_pairs": len(p.weak_pairs), "on_the_nose": p.on_the_nose,
                      "issues": p.issues}
        return AgentReview(agent=self.name, score=s, issues=p.issues[:3],
                           data={"cliches": len(p.cliches), "weak_pairs": len(p.weak_pairs),
                                 "on_the_nose": p.on_the_nose, "opener_variety": p.opener_variety})

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
        return f"last {len(rows)} chapters: {avg('cliches'):.1f} clichés, {avg('weak_pairs'):.1f} weak pairs, {avg('on_the_nose'):.1f} on-the-nose lines"
