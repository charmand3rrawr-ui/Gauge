"""The Sensitivity Reader: flags content against platform rules. Warns by default; blocks only if asked."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import AgentReview, SceneContext, StoryAgent

SYSTEM = """You review fiction for a publishing platform. You flag content categories and severity precisely and without
moralising; the author decides what to do with the flags."""
DEFAULT_CATEGORIES = ["graphic violence", "sexual content", "sexual content involving minors", "self-harm or suicide",
                      "hate or slurs", "drug use", "real people", "copyrighted text"]


class SensState(BaseModel):
    flags: dict[int, list[dict]] = Field(default_factory=dict)


class SensitivityReader(StoryAgent):
    name = "sensitivity"
    title = "Sensitivity Reader"
    weight = 0.0  # never changes the score unless block_on_flags is set
    state_model = SensState

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        cats = self.options.get("categories") or DEFAULT_CATEGORIES
        rules = self.options.get("rules", "")
        d = self.ask_json("review", f"""Flag content in chapter {ctx.number}. Categories: {', '.join(cats)}.
{('Platform rules: ' + rules) if rules else ''}
Respond with ONLY JSON: {{"flags": [{{"category": "...", "severity": "mild|moderate|severe", "quote": "<= 12 words", "note": "one line"}}],
"ok_for_general": true/false}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.1, max_tokens=500)
        self._last = None
        if not isinstance(d.get("flags"), list) and not isinstance(d.get("ok_for_general"), bool):
            return self.no_answer()  # unread is not "clean": no verdict, and nothing is recorded for the chapter
        flags = [f for f in (d.get("flags") or []) if isinstance(f, dict) and f.get("category")] if isinstance(d.get("flags"), list) else []
        flags = [{"category": str(f["category"])[:40], "severity": str(f.get("severity", "mild"))[:10], "quote": str(f.get("quote", ""))[:80],
                  "note": str(f.get("note", ""))[:120]} for f in flags][:8]
        self._last = flags
        severe = [f for f in flags if f["severity"] == "severe"]
        block = bool(self.options.get("block_on_flags")) and severe
        issues = [f"content flag ({f['severity']} {f['category']}): \"{f['quote']}\" - {f['note']}" for f in severe] if block else []
        return AgentReview(agent=self.name, score=0.0 if block else 1.0, issues=issues,
                           notes=f"{len(flags)} flag(s)" if flags else "", data={"flags": flags, "ok_for_general": bool(d.get("ok_for_general", True))})

    def remember(self, text: str, ctx: SceneContext) -> None:
        last = getattr(self, "_last", [])
        if last is not None:
            self.state.flags[ctx.number] = last

    def report(self) -> str:
        return "\n".join(f"ch {n}: " + "; ".join(f"{f['severity']} {f['category']}" for f in fl) if fl else f"ch {n}: clean"
                         for n, fl in sorted(self.state.flags.items())[-15:]) or "(no chapters reviewed yet)"
