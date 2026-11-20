"""The Comedian: comic beats for comic stories. Attaches itself only when the genre asks for it."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import AgentReview, SceneContext, StoryAgent, as_str_list

SYSTEM = """You are a comedy writer for serial fiction. You find the joke that comes from character and situation, never
from the narrator winking. Rule of three, callbacks, deadpan, escalation."""
COMIC_WORDS = ("comedy", "humor", "humour", "comic", "funny", "satire", "parody", "wry", "farce")


def wants_comedy(genres: list[str], tone: str = "") -> bool:
    hay = " ".join(genres + [tone]).lower()
    return any(w in hay for w in COMIC_WORDS)


class ComedyState(BaseModel):
    callbacks: list[str] = Field(default_factory=list)  # running gags
    laughs: list[int] = Field(default_factory=list)


class Comedian(StoryAgent):
    name = "comedian"
    title = "Comedian"
    weight = 0.5
    state_model = ComedyState

    def brief(self, ctx: SceneContext) -> str:
        d = self.ask_json("brief", f"""Plan comic beats for chapter {ctx.number}: {ctx.plan_text}
Cast: {ctx.cast_text[:600]}
Running gags available for a callback: {'; '.join(self.state.callbacks[-5:]) or '(none yet)'}
Respond with ONLY JSON: {{"beats": ["2-3 specific comic beats that come from character or situation"]}}""",
                         system=SYSTEM, temperature=0.9, max_tokens=300)
        beats = as_str_list(d.get("beats"), 3)
        return "\n".join(f"- comic beat: {b}" for b in beats)

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        d = self.ask_json("review", f"""Count the genuine laughs in chapter {ctx.number} and name what fell flat.
Respond with ONLY JSON: {{"laughs": int, "flat": ["lines that tried and failed"], "notes": "one line", "new_running_gag": "a line or bit worth calling back later, or empty"}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.3, max_tokens=300)
        try:
            laughs = int(d.get("laughs", 0))
        except (TypeError, ValueError):
            laughs = 0
        self._last = (laughs, str(d.get("new_running_gag", "") or ""))
        issues = [f"flat joke: {f}" for f in as_str_list(d.get("flat"), 2)]
        if laughs == 0:
            issues.append("no laugh in a comic serial; let a character's flaw collide with the situation at least once")
        return AgentReview(agent=self.name, score=min(1.0, 0.5 + 0.2 * laughs), issues=issues, notes=str(d.get("notes", ""))[:160], data={"laughs": laughs})

    def remember(self, text: str, ctx: SceneContext) -> None:
        laughs, gag = getattr(self, "_last", (0, ""))
        self.state.laughs.append(laughs)
        if gag and gag not in self.state.callbacks:
            self.state.callbacks.append(gag[:120])
        self.state.callbacks = self.state.callbacks[-12:]

    def report(self) -> str:
        return f"Laughs per chapter: {self.state.laughs[-12:]}\nRunning gags: " + "; ".join(self.state.callbacks)
