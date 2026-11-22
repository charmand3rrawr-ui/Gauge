"""The Continuity Auditor: keeps the physical world consistent in space and time.

The Archivist guards facts and the emotion auditor guards feelings; this guards logistics. It flags a
chapter whose time of day contradicts itself with no passage of time, an explicit clock that runs
backwards or contradicts the light (2 a.m. sunlight), and a season or time of day that flips between
consecutive chapters without time having moved. Measured, no model calls. It complements
rather than duplicates the World agent, which invents the world; this one checks that the world holds
still when it should.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..realism import clock_check, logistics_check
from .base import AgentReview, SceneContext, StoryAgent


class ContinuityState(BaseModel):
    history: list[dict] = Field(default_factory=list)  # {chapter, time_of_day, seasons}


class ContinuityAuditor(StoryAgent):
    name = "continuity"
    title = "Continuity Auditor"
    weight = 0.7
    state_model = ContinuityState

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        from ...text import word_count
        if word_count(text) < 100:
            return None
        lc = logistics_check(text)
        issues = list(lc.issues)
        cc = clock_check(text)                       # explicit times: 3 p.m., 14:30, midnight
        issues += cc.issues
        # Cross-chapter: a season flip between consecutive chapters is suspicious unless time clearly passed.
        if self.state.history:
            prev = self.state.history[-1]
            prev_seasons = set(prev.get("seasons", []))
            if prev_seasons and lc.seasons and prev_seasons.isdisjoint(lc.seasons):
                issues.append(f"the season changed from {', '.join(prev_seasons)} to {', '.join(lc.seasons)} since last chapter; "
                              "confirm enough time has passed, or fix it")
        clock = [f"{t // 60:02d}:{t % 60:02d}" for t, _ in cc.times]
        self._last = {"time_of_day": lc.time_of_day, "seasons": lc.seasons, "issues": issues, "clock": clock}
        score = 1.0 if not issues else max(0.4, 1.0 - 0.2 * len(issues))
        return AgentReview(agent=self.name, score=round(score, 3), issues=issues[:3],
                           data={"time_of_day": lc.time_of_day, "seasons": lc.seasons,
                                 "clock": f"{clock[0]} to {clock[-1]}" if clock else ""})

    def remember(self, text: str, ctx: SceneContext) -> None:
        last = getattr(self, "_last", None)
        if last:
            self.state.history.append({"chapter": ctx.number, "time_of_day": last["time_of_day"], "seasons": last["seasons"],
                                       "clock": last.get("clock", [])})
            self.state.history = self.state.history[-200:]

    def report(self) -> str:
        if not self.state.history:
            return "(no chapters tracked yet)"
        last = self.state.history[-1]
        clock = last.get("clock") or []
        return (f"last chapter: time of day {last['time_of_day']}, season {', '.join(last['seasons']) or 'unspecified'}"
                + (f", clock {clock[0]} to {clock[-1]}" if clock else ""))
