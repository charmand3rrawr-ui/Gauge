"""The World Auditor: keeps the slow physical state of the world consistent across the serial (roadmap C).

Measured, no model calls. It runs the deterministic world-state checks and carries the cross-chapter
memory the single-chapter checks need: an absolute day on the timeline, the established season, who is
hurt, and which props have been destroyed. The Continuity Auditor guards time-of-day within a chapter;
this guards weeks and months of story. Author-declared world rules and place distances sharpen it and are
passed in through options (from the project bible / config).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..world_state import (
    check_rules,
    parse_rules,
    read_injuries,
    read_props,
    read_timeline,
    read_travel,
    read_weather,
)
from .base import AgentReview, SceneContext, StoryAgent


class WorldState(BaseModel):
    day: float = 0.0                                   # absolute day offset reached so far
    season: str = ""                                   # established season, if ever stated unambiguously
    injured: dict[str, int] = Field(default_factory=dict)   # character -> chapter hurt (still unrecovered)
    destroyed: list[str] = Field(default_factory=list)      # props destroyed/lost, not seen used since
    chapters: int = 0


class WorldAuditor(StoryAgent):
    name = "world_auditor"
    title = "World Auditor"
    weight = 0.7
    state_model = WorldState

    def _rules(self):
        return parse_rules(self.options.get("world_rules", ""))

    def _known_places(self) -> set[str]:
        return {p.strip() for p in self.options.get("known_places", []) if str(p).strip()}

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        from ...text import word_count
        if word_count(text) < 100:
            return None
        issues: list[str] = []

        tl = read_timeline(text)
        issues += tl.issues

        wx = read_weather(text, established_season=self.state.season or None)
        issues += wx.issues

        inj = read_injuries(text)
        # A character hurt in an earlier chapter, now doing vigorous action with no recovery shown.
        for name, hurt_ch in self.state.injured.items():
            if name in inj.vigorous and not inj.recovery_cue and 0 < (ctx.number - hurt_ch) <= 2:
                issues.append(f"{name} was badly hurt in chapter {hurt_ch} and is in hard action now "
                              "with no recovery or time shown; add healing, a limp, or passed time")

        props = read_props(text)
        for p in props.used:
            if p in self.state.destroyed:
                issues.append(f"the {p} was destroyed or lost earlier but is in use again; explain or fix it")

        tr = read_travel(text, known_places=self._known_places())
        issues += tr.issues

        rc = check_rules(text, self._rules())
        issues += rc.violations

        self._read = {"days": tl.days_elapsed, "flashback": tl.has_flashback_cue, "seasons": wx.seasons,
                      "injured": inj.injured, "recovery": inj.recovery_cue, "vigorous": sorted(inj.vigorous),
                      "destroyed": sorted(props.destroyed), "used": sorted(props.used)}
        score = 1.0 if not issues else max(0.4, 1.0 - 0.15 * len(issues))
        return AgentReview(agent=self.name, score=round(score, 3), issues=issues[:3],
                           data={"day": round(self.state.day + tl.days_elapsed, 1),
                                 "season": wx.seasons[0] if len(wx.seasons) == 1 else (self.state.season or "unset"),
                                 "open_injuries": len(self.state.injured),
                                 "rule_violations": len(rc.violations)})

    def remember(self, text: str, ctx: SceneContext) -> None:
        r = getattr(self, "_read", None)
        if not r:
            return
        self.state.day = round(self.state.day + r["days"], 2)
        if len(r["seasons"]) == 1:
            self.state.season = r["seasons"][0]
        # New injuries open; recovery this chapter clears old ones.
        for name in r["injured"]:
            self.state.injured[name] = ctx.number
        if r["recovery"]:
            self.state.injured = {}
        # Props: destroyed ones go on the watch-list; a destroyed-then-used prop is cleared (the slip is
        # already flagged) so it is not reported forever.
        for p in r["destroyed"]:
            if p not in self.state.destroyed:
                self.state.destroyed.append(p)
        self.state.destroyed = [p for p in self.state.destroyed if p not in r["used"]][-50:]
        self.state.chapters = max(self.state.chapters, ctx.number)

    def report(self) -> str:
        if not self.state.chapters:
            return "(no chapters tracked yet)"
        bits = [f"day ~{self.state.day:g}", f"season {self.state.season or 'unset'}"]
        if self.state.injured:
            bits.append("open injuries: " + ", ".join(f"{n} (ch{c})" for n, c in self.state.injured.items()))
        if self.state.destroyed:
            bits.append("destroyed props watched: " + ", ".join(self.state.destroyed[:6]))
        return "; ".join(bits)
