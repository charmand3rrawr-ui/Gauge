"""The Theme Keeper: central question, motifs and images, threaded so the serial rhymes."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import AgentReview, SceneContext, StoryAgent, as_str_list

SYSTEM = """You keep the theme of a long serial alive without ever stating it. You work in motifs, recurring images and
the central question, placed lightly, paid off rarely."""


class ThemeState(BaseModel):
    central_question: str = ""
    motifs: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    occurrences: dict[str, list[int]] = Field(default_factory=dict)
    rotation: int = 0


class ThemeKeeper(StoryAgent):
    name = "theme_keeper"
    title = "Theme Keeper"
    weight = 0.5
    state_model = ThemeState

    def setup(self, project) -> None:
        if self.state.motifs:
            return
        d = self.ask_json("setup", f"""From this bible, name the central question, 3-5 motifs (concrete recurring things) and
2-4 recurring images. Respond with ONLY JSON: {{"central_question": "...", "motifs": [...], "images": [...]}}

BIBLE:
{project.bible_markdown()}""", system=SYSTEM, temperature=0.7)
        self.state.central_question = str(d.get("central_question", ""))[:200]
        self.state.motifs = as_str_list(d.get("motifs"), 6, 60)
        self.state.images = as_str_list(d.get("images"), 5, 80)
        self.save()

    def _pick(self, number: int) -> str:
        if not self.state.motifs:
            return ""
        # Least recently used motif first.
        return min(self.state.motifs, key=lambda m: (self.state.occurrences.get(m, [0])[-1] if self.state.occurrences.get(m) else 0))

    def brief(self, ctx: SceneContext) -> str:
        if not self.state.motifs:
            return ""
        motif = self._pick(ctx.number)
        out = [f"thread this motif lightly, without comment: {motif}"]
        if self.state.central_question and ctx.number % 5 == 0:
            out.append(f"this chapter may brush the central question ({self.state.central_question}) through action, never speech")
        return "\n".join(f"- {o}" for o in out)

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        if not self.state.motifs:
            return None
        low = text.lower()
        present = [m for m in self.state.motifs if m.lower().split()[0] in low]
        d = self.ask_json("review", f"""Motifs: {', '.join(self.state.motifs)}. Central question: {self.state.central_question}
Does chapter {ctx.number} carry any motif or touch the question, without stating it? Respond with ONLY JSON:
{{"motifs_present": [...], "theme_touched": true/false, "notes": "one line"}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.2, max_tokens=300)
        if not any(k in d for k in ("motifs_present", "theme_touched", "notes")):
            self.no_answer()  # the motifs found in the text above stand on their own
        present = sorted(set(present) | set(as_str_list(d.get("motifs_present"), 6, 60)))
        score = 1.0 if present else 0.7
        issues = [] if present else [f"no motif appears; place one of {', '.join(self.state.motifs[:3])} in a detail"]
        self._last_present = present
        return AgentReview(agent=self.name, score=score, issues=issues, notes=str(d.get("notes", ""))[:160],
                           data={"motifs": present})

    def remember(self, text: str, ctx: SceneContext) -> None:
        for m in getattr(self, "_last_present", []):
            self.state.occurrences.setdefault(m, []).append(ctx.number)

    def report(self) -> str:
        return (f"Question: {self.state.central_question}\nMotifs: " + "; ".join(f"{m} (ch {', '.join(map(str, self.state.occurrences.get(m, [])[-5:]))})"
                for m in self.state.motifs) + f"\nImages: {', '.join(self.state.images)}")
