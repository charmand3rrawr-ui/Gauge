"""The Reader Panel: a simulated audience, calibrated against real reactions when you have them."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import AgentReview, SceneContext, StoryAgent, as_str_list, clamp01, given01

DEFAULT_PERSONAS = {
    "binge": "the binge reader: reads at 2am, forgives flaws, needs a reason to click Next every single chapter",
    "skimmer": "the skimmer: reads dialogue and short paragraphs only, bails at the first wall of description",
    "lore": "the lore-hound: tracks every rule and number, loves systems, hates contradictions",
    "commenter": "the commenter: reads to react, posts predictions and complaints, notices pacing and character moments",
    "hostile": "the hostile reader: looks for plot holes, convenient coincidences and out-of-character choices, and will say so publicly",
}

SYSTEM = """You are a panel of web novel readers. Each of you reads the same chapter and reacts honestly, in your
own persona. You are not editors; you are the audience. Say what you would actually do and post."""


class Prediction(BaseModel):
    chapter: int
    click_next: float
    per_persona: dict[str, float] = Field(default_factory=dict)
    comments: list[str] = Field(default_factory=list)
    stop_points: list[str] = Field(default_factory=list)


class PanelState(BaseModel):
    personas: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_PERSONAS))
    predictions: list[Prediction] = Field(default_factory=list)
    actuals: dict[int, dict[str, float]] = Field(default_factory=dict)  # chapter -> {"views":..,"comments":..}
    bias: float = 0.0  # learned offset between predicted click_next and observed retention


class ReaderPanel(StoryAgent):
    name = "reader_panel"
    title = "Reader Panel"
    weight = 2.0
    state_model = PanelState

    def _personas_text(self) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in self.state.personas.items())

    def brief(self, ctx: SceneContext) -> str:
        if ctx.number <= 1 or not ctx.prev_tail:
            return ""
        d = self.ask_json("brief", f"""The readers just finished the previous chapter, which ended:
{ctx.prev_tail}

The next chapter is planned as: {ctx.plan_text}

Personas:
{self._personas_text()}

Respond with ONLY JSON: {{"hopes": ["what each persona hopes happens next, 1 line each"], "fears": ["what would make a persona quit"]}}""",
                         system=SYSTEM, temperature=0.6)
        hopes, fears = as_str_list(d.get("hopes"), 5), as_str_list(d.get("fears"), 3)
        if not hopes and not fears:
            return ""
        out = []
        if hopes:
            out.append("readers are hoping for: " + "; ".join(hopes))
        if fears:
            out.append("readers will quit if: " + "; ".join(fears))
        return "\n".join(f"- {o}" for o in out)

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        d = self.ask_json("review", f"""Each persona reads chapter {ctx.number} below and reports.

Personas:
{self._personas_text()}

Respond with ONLY JSON: {{"readers": [{{"persona": "<key>", "click_next": 0-1 probability they read the next chapter,
"stop_at": "exact words where they would have closed the tab, or empty", "comment": "what they would post, or empty",
"mood": "one word"}}]}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.4, max_tokens=900)
        self._last = None
        readers = d.get("readers") if isinstance(d.get("readers"), list) else []
        if not readers:
            return self.no_answer()
        per: dict[str, float] = {}
        comments, stops = [], []
        for r in readers:
            if not isinstance(r, dict):
                continue
            key = str(r.get("persona", "reader"))[:20]
            p_next = given01(r.get("click_next"))
            if p_next is not None:  # a persona that gave no probability comments but does not vote
                per[key] = p_next
            if r.get("comment"):
                comments.append(f"{key}: {str(r['comment'])[:160]}")
            if r.get("stop_at"):
                stops.append(f"{key} would stop at \"{str(r['stop_at'])[:100]}\"")
        if not per:
            return self.no_answer()
        click = sum(per.values()) / len(per)
        adjusted = clamp01(click + self.state.bias)
        issues = stops[:3]
        if adjusted < 0.6:
            issues.append(f"only {adjusted:.0%} of the panel would click Next; raise the stakes of the ending or cut the slow middle")
        self._last = Prediction(chapter=ctx.number, click_next=adjusted, per_persona=per, comments=comments[:6], stop_points=stops[:5])
        return AgentReview(agent=self.name, score=adjusted, issues=issues,
                           notes="; ".join(comments[:3]), data={"click_next": round(adjusted, 3), "per_persona": per})

    def remember(self, text: str, ctx: SceneContext) -> None:
        last = getattr(self, "_last", None)
        if last is not None and last.chapter == ctx.number:
            self.state.predictions = [p for p in self.state.predictions if p.chapter != ctx.number] + [last]
            self.state.predictions = self.state.predictions[-200:]

    # ---- calibration against reality -----------------------------------------
    def record_actual(self, chapter: int, views: Optional[float] = None, comments: Optional[float] = None) -> None:
        d = self.state.actuals.get(chapter, {})
        if views is not None:
            d["views"] = float(views)
        if comments is not None:
            d["comments"] = float(comments)
        self.state.actuals[chapter] = d
        self._recalibrate()

    def _recalibrate(self) -> None:
        """Retention proxy: views of chapter n+1 / views of chapter n. Bias = mean(actual - predicted)."""
        pairs = []
        preds = {p.chapter: p for p in self.state.predictions}
        for n, p in preds.items():
            a, b = self.state.actuals.get(n, {}).get("views"), self.state.actuals.get(n + 1, {}).get("views")
            if a and b and a > 0:
                pairs.append((p.click_next - self.state.bias, min(1.0, b / a)))
        if len(pairs) >= 3:
            self.state.bias = round(sum(actual - pred for pred, actual in pairs) / len(pairs), 3)

    def calibration(self) -> dict[str, Any]:
        preds = {p.chapter: p for p in self.state.predictions}
        pairs = []
        for n, p in preds.items():
            a, b = self.state.actuals.get(n, {}).get("views"), self.state.actuals.get(n + 1, {}).get("views")
            if a and b and a > 0:
                pairs.append((p.click_next, min(1.0, b / a)))
        corr = None
        if len(pairs) >= 3:
            import numpy as np

            x, y = np.asarray([p for p, _ in pairs]), np.asarray([a for _, a in pairs])
            if x.std() > 0 and y.std() > 0:
                corr = float(np.corrcoef(x, y)[0, 1])
        return {"pairs": len(pairs), "bias": self.state.bias, "correlation": None if corr is None else round(corr, 3)}

    def report(self) -> str:
        lines = [f"Personas: {', '.join(self.state.personas)}", f"Calibration: {self.calibration()}", "Recent predictions:"]
        for p in self.state.predictions[-8:]:
            lines.append(f"  ch {p.chapter}: click_next {p.click_next:.2f}  " + "  ".join(f"{k} {v:.1f}" for k, v in p.per_persona.items()))
            for c in p.comments[:2]:
                lines.append(f"      \"{c}\"")
        return "\n".join(lines)
