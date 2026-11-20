"""The Schemer: the opposition as a planner with hidden information, trying to win."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import AgentReview, SceneContext, StoryAgent, as_str_list

SYSTEM = """You are the mind behind the story's opposition: the antagonist and the forces aligned with them. You are
intelligent, patient and resourceful. You plan several moves ahead, you act on what you actually know (not what
the reader knows), and you adapt when the protagonist surprises you. You want to win."""


class SchemeState(BaseModel):
    antagonist: str = ""
    goal: str = ""
    resources: list[str] = Field(default_factory=list)
    knows: list[str] = Field(default_factory=list)  # what the opposition knows about the protagonist
    scheme: list[str] = Field(default_factory=list)  # planned steps
    done: list[str] = Field(default_factory=list)
    exposure: str = ""  # what the protagonist knows about the scheme
    moves: list[dict] = Field(default_factory=list)  # per chapter


class Schemer(StoryAgent):
    name = "schemer"
    title = "Schemer"
    weight = 1.0
    state_model = SchemeState

    def _state_text(self) -> str:
        s = self.state
        return (f"Antagonist: {s.antagonist or '(unknown)'}\nGoal: {s.goal}\nResources: {'; '.join(s.resources)}\n"
                f"Knows about the protagonist: {'; '.join(s.knows)}\nScheme (remaining): {' -> '.join(s.scheme) or '(none)'}\n"
                f"Done: {'; '.join(s.done[-5:])}\nWhat the protagonist knows of this: {s.exposure or 'nothing'}")

    def setup(self, project) -> None:
        if self.state.antagonist:
            return
        d = self.ask_json("setup", f"""Define the opposition for this serial from its bible.

BIBLE:
{project.bible_markdown()}

Respond with ONLY JSON: {{"antagonist": "name of the main opposing force or person", "goal": "what they want, concretely",
"resources": ["3-5 assets"], "knows": ["2-4 things they know about the protagonist"],
"scheme": ["4-7 ordered steps of their plan"], "exposure": "what the protagonist knows of the scheme at the start"}}""",
                         system=SYSTEM, temperature=0.8, max_tokens=800)
        s = self.state
        s.antagonist = str(d.get("antagonist", ""))[:80]
        s.goal = str(d.get("goal", ""))[:200]
        s.resources = as_str_list(d.get("resources"), 6)
        s.knows = as_str_list(d.get("knows"), 6)
        s.scheme = as_str_list(d.get("scheme"), 8)
        s.exposure = str(d.get("exposure", ""))[:200]
        self.save()

    def brief(self, ctx: SceneContext) -> str:
        if not self.state.antagonist:
            return ""
        d = self.ask_json("brief", f"""{self._state_text()}

The author plans chapter {ctx.number}: {ctx.plan_text}
Cast in the scene:
{ctx.cast_text[:800] or '(unknown)'}

What does the opposition do this chapter? Respond with ONLY JSON: {{"visible": "what the reader/protagonist can notice",
"hidden": "what happens off-page", "next_move": "the scheme step being executed", "wants_protagonist_to": "the reaction you are trying to provoke"}}""",
                         system=SYSTEM, temperature=0.7)
        if not d:
            return ""
        self._last_brief = d
        parts = [f"the opposition ({self.state.antagonist}) this chapter: visible: {d.get('visible', '')}",
                 f"hidden: {d.get('hidden', '')}", f"it wants the protagonist to {d.get('wants_protagonist_to', '')}"]
        return "\n".join(f"- {p}" for p in parts if p.strip() and not p.endswith(": "))

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        if not self.state.antagonist:
            return None
        d = self.ask_json("review", f"""{self._state_text()}

Read chapter {ctx.number}. Does the opposition behave as an intelligent actor with these resources and this knowledge?
Respond with ONLY JSON: {{"consistent": true/false, "antagonist_smart": true/false (not stupid, not omniscient),
"notes": "one line", "exposure_change": "what the protagonist now knows of the scheme, if changed"}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.2)
        if not d:
            return None
        score = 1.0
        issues = []
        if d.get("consistent") is False:
            score -= 0.3
            issues.append(f"the opposition acts against what it knows/has: {d.get('notes', '')}")
        if d.get("antagonist_smart") is False:
            score -= 0.3
            issues.append("the antagonist is written as stupid or omniscient; give them a smart move that fits their resources")
        return AgentReview(agent=self.name, score=score, issues=issues, notes=str(d.get("notes", ""))[:200],
                           data={"exposure": str(d.get("exposure_change", ""))[:120]})

    def remember(self, text: str, ctx: SceneContext) -> None:
        if not self.state.antagonist:
            return
        d = self.ask_json("remember", f"""{self._state_text()}

Chapter {ctx.number} happened. Update the scheme. Respond with ONLY JSON: {{"step_done": "scheme step completed or empty",
"new_steps": ["steps added or changed because of what happened"], "exposure": "what the protagonist now knows",
"resources_gained": [], "resources_lost": []}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.3)
        s = self.state
        step = str(d.get("step_done", "") or "")
        if step:
            s.done.append(step[:160])
            s.scheme = [x for x in s.scheme if x.lower()[:25] != step.lower()[:25]]
        for ns in as_str_list(d.get("new_steps"), 4):
            if ns not in s.scheme:
                s.scheme.append(ns)
        if d.get("exposure"):
            s.exposure = str(d["exposure"])[:200]
        for r in as_str_list(d.get("resources_gained"), 3):
            if r not in s.resources:
                s.resources.append(r)
        for r in as_str_list(d.get("resources_lost"), 3):
            s.resources = [x for x in s.resources if r.lower()[:20] not in x.lower()]
        s.moves.append({"chapter": ctx.number, "visible": getattr(self, "_last_brief", {}).get("visible", ""), "done": step})
        s.moves = s.moves[-60:]

    def report(self) -> str:
        return self._state_text() + "\nRecent moves: " + "; ".join(f"ch {m['chapter']}: {m.get('done') or m.get('visible')}" for m in self.state.moves[-5:])
