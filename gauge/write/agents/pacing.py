"""The Pacing Director: varies chapter shapes across the serial. Measured, no LLM calls."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ...features import chapter_features
from ...models import ChapterRecord
from .base import AgentReview, SceneContext, StoryAgent

SHAPES = ("action", "quiet", "reveal", "progression", "dialogue")


def classify(f: dict[str, float]) -> str:
    if f.get("reveal_density", 0) > 25 and f.get("cliff_reveal", 0) > 0.3:
        return "reveal"
    if f.get("progression_density", 0) > 25:
        return "progression"
    if f.get("tension_mean", 0) > 18:
        return "action"
    if f.get("dialogue_ratio", 0) > 0.5:
        return "dialogue"
    return "quiet"


class PacingState(BaseModel):
    shapes: list[str] = Field(default_factory=list)
    words: list[int] = Field(default_factory=list)
    recommended: str = ""


class PacingDirector(StoryAgent):
    name = "pacing"
    title = "Pacing Director"
    weight = 0.5
    state_model = PacingState

    def recommend(self) -> str:
        recent = self.state.shapes[-3:]
        if not recent:
            return "action"
        counts = {s: recent.count(s) for s in SHAPES}
        if recent[-1] == "action" and counts["action"] >= 2:
            return "quiet"
        if recent[-1] == "quiet":
            return "action" if counts["reveal"] else "reveal"
        if counts["dialogue"] >= 2:
            return "action"
        least = min(SHAPES, key=lambda s: (counts[s], SHAPES.index(s)))
        return least

    def plan_notes(self, project) -> str:
        rec = self.recommend()
        self.state.recommended = rec
        if not self.state.shapes:
            return ""
        return f"- recent chapter shapes: {' -> '.join(self.state.shapes[-4:])}; the next chapter should be a {rec} chapter"

    def brief(self, ctx: SceneContext) -> str:
        rec = self.state.recommended or self.recommend()
        self.state.recommended = rec
        hint = {"action": "fast scenes, short paragraphs, physical stakes", "quiet": "a breath: character, texture, a small warm beat, then one turn",
                "reveal": "withhold until two-thirds, then land a reveal in a one-line paragraph",
                "progression": "visible gain: a skill, a rank, a resource, earned on the page", "dialogue": "a confrontation in talk; subtext over statement"}[rec]
        tail = f"; last chapters ran {', '.join(self.state.shapes[-3:])}" if self.state.shapes else ""
        return f"- shape this chapter as {rec}: {hint}{tail}"

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        f = chapter_features(ChapterRecord(novel_key="draft", index=ctx.number, text=text))
        shape = classify(f)
        rec = self.state.recommended or "action"
        self._last = (shape, int(f["words"]))
        if shape == rec or not self.state.shapes:
            return AgentReview(agent=self.name, score=1.0, data={"shape": shape})
        same_run = self.state.shapes[-2:] == [shape, shape]
        score = 0.6 if same_run else 0.85
        issues = [f"third {shape} chapter in a row; this one was meant to be {rec}"] if same_run else []
        return AgentReview(agent=self.name, score=score, issues=issues, data={"shape": shape, "wanted": rec})

    def remember(self, text: str, ctx: SceneContext) -> None:
        shape, n = getattr(self, "_last", (classify(chapter_features(ChapterRecord(novel_key="d", index=ctx.number, text=text))), len(text.split())))
        self.state.shapes.append(shape)
        self.state.words.append(n)
        self.state.shapes, self.state.words = self.state.shapes[-200:], self.state.words[-200:]
        self.state.recommended = ""

    def report(self) -> str:
        return "Shapes: " + " -> ".join(self.state.shapes[-12:]) + f"\nNext: {self.recommend()}"
