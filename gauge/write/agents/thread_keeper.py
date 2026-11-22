"""The Thread Keeper: weaves and resolves subplots, and links foreshadow to payoff (roadmap 34, 37).

Measured, no model calls. The Showrunner tracks the big promises; this tracks the smaller weave — the
B-plots that must keep breathing and resolve, and the details planted to pay off later. It flags a subplot
that has gone quiet for too long (a dropped thread), a payoff that lands with no earlier setup (unearned),
and reminds the planner which plants are still waiting to be paid.

Declare the threads in config under ``agents`` (nothing declared = the agent stays silent):
``subplots`` as ``label: keyword, keyword``; ``foreshadow`` as ``plant -> payoff``; ``subplot_gap`` is how
many chapters a subplot may be absent before it is called dormant (default 5).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..threads import (
    dormant_subplots,
    foreshadow_events,
    parse_foreshadow,
    parse_subplots,
    subplot_hits,
)
from .base import AgentReview, SceneContext, StoryAgent


class ThreadState(BaseModel):
    subplot_last_seen: dict[str, int] = Field(default_factory=dict)  # label -> chapter last present
    planted: list[str] = Field(default_factory=list)                 # foreshadow plants seen (by plant text)
    paid: list[str] = Field(default_factory=list)                    # plants whose payoff has landed
    chapters: int = 0


class ThreadKeeper(StoryAgent):
    name = "thread_keeper"
    title = "Thread Keeper"
    weight = 0.6
    state_model = ThreadState

    def _subplots(self):
        return parse_subplots(self.options.get("subplots", []))

    def _foreshadow(self):
        return parse_foreshadow(self.options.get("foreshadow", []))

    def _gap(self) -> int:
        try:
            return max(2, int(self.options.get("subplot_gap", 5) or 5))
        except (TypeError, ValueError):
            return 5

    # -- plan notes ----------------------------------------------------------
    def plan_notes(self, project) -> str:
        subplots = self._subplots()
        notes: list[str] = []
        if subplots:
            labels = [sp.label for sp in subplots]
            dormant = dormant_subplots(self.state.subplot_last_seen, labels, self.state.chapters + 1, self._gap())
            if dormant:
                notes.append("Dormant subplots to weave back in: " + "; ".join(dormant[:6]))
        pending = [f.plant for f in self._foreshadow() if f.plant in self.state.planted and f.plant not in self.state.paid]
        if pending:
            notes.append("Planted but unpaid — consider paying one off: " + "; ".join(pending[:6]))
        return "\n".join(notes)

    # -- brief ---------------------------------------------------------------
    def brief(self, ctx: SceneContext) -> str:
        lines: list[str] = []
        subplots = self._subplots()
        if subplots:
            dormant = dormant_subplots(self.state.subplot_last_seen, [sp.label for sp in subplots], ctx.number, self._gap())
            if dormant:
                lines.append("- a dropped thread needs a beat: " + "; ".join(dormant[:4]))
        pending = [f.plant for f in self._foreshadow() if f.plant in self.state.planted and f.plant not in self.state.paid]
        if pending:
            lines.append("- foreshadow waiting to pay off (land one when it fits): " + "; ".join(pending[:4]))
        return "\n".join(lines)

    # -- review --------------------------------------------------------------
    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        from ...text import word_count
        subplots = self._subplots()
        pairs = self._foreshadow()
        if not subplots and not pairs:
            return None
        if word_count(text) < 120:
            return None
        issues: list[str] = []

        hits = subplot_hits(text, subplots)
        self._hits_now = hits
        # Dormant threads: declared subplots not seen for `gap` chapters and not present now either.
        labels = [sp.label for sp in subplots]
        dormant = [d for d in dormant_subplots(self.state.subplot_last_seen, labels, ctx.number, self._gap()) if d not in hits]
        for d in dormant[:2]:
            last = self.state.subplot_last_seen.get(d, 0)
            where = f"last seen in chapter {last}" if last else "never yet on the page"
            issues.append(f"subplot '{d}' has gone dark ({where}); weave it back in or resolve it before it reads as dropped")

        plants_now, payoffs_now = foreshadow_events(text, pairs)
        self._plants_now, self._payoffs_now = plants_now, payoffs_now
        for plant in sorted(payoffs_now):
            if plant not in self.state.planted and plant not in plants_now:
                pay = next((f.payoff for f in pairs if f.plant == plant), plant)
                issues.append(f"the payoff '{pay}' lands with no earlier setup; plant '{plant}' in a previous chapter so it reads as earned")

        score = 1.0 if not issues else max(0.5, 1.0 - 0.15 * len(issues))
        return AgentReview(agent=self.name, score=round(score, 3), issues=issues[:3],
                           data={"subplots_present": sorted(hits), "planted": sorted(plants_now), "paid_off": sorted(payoffs_now)})

    # -- remember ------------------------------------------------------------
    def remember(self, text: str, ctx: SceneContext) -> None:
        for label in getattr(self, "_hits_now", set()):
            self.state.subplot_last_seen[label] = ctx.number
        for plant in getattr(self, "_plants_now", set()):
            if plant not in self.state.planted:
                self.state.planted.append(plant)
        for plant in getattr(self, "_payoffs_now", set()):
            if plant not in self.state.paid:
                self.state.paid.append(plant)
        self.state.chapters = max(self.state.chapters, ctx.number)

    # -- report --------------------------------------------------------------
    def report(self) -> str:
        if not self.state.chapters:
            return "(no chapters tracked yet)"
        bits: list[str] = []
        subplots = self._subplots()
        if subplots:
            rows = [f"{sp.label}: ch{self.state.subplot_last_seen.get(sp.label, 0) or '-'}" for sp in subplots]
            bits.append("subplots last seen — " + ", ".join(rows[:10]))
        pairs = self._foreshadow()
        if pairs:
            pending = [f.plant for f in pairs if f.plant in self.state.planted and f.plant not in self.state.paid]
            bits.append(f"foreshadow paid {len(self.state.paid)}/{len(pairs)}")
            if pending:
                bits.append("still pending: " + "; ".join(pending[:5]))
        return "; ".join(bits) or "(tracking)"
