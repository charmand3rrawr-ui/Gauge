"""Planner: builds the bible and rolling chapter outlines."""
from __future__ import annotations

import logging
from typing import Optional

from ..learn.exemplars import ExemplarLibrary
from ..llm.client import LLM, generate_json
from . import prompts
from .project import Bible, ChapterPlan, Project
from .schemas import BibleOut, OutlineOut, schema_of

log = logging.getLogger(__name__)


class Planner:
    def __init__(self, llm: LLM, exemplars: Optional[ExemplarLibrary], style_guide: str):
        self.llm = llm
        self.exemplars = exemplars
        self.system = prompts.SYSTEM_AUTHOR.format(style_guide=style_guide or "(no data yet)")

    def build_bible(self, project: Project) -> Bible:
        prompt = prompts.BIBLE_PROMPT.format(idea=project.state.seed_idea, genres=", ".join(project.state.genres))
        data = generate_json(self.llm, prompt, system=self.system, schema=schema_of(BibleOut), temperature=0.9,
                             max_tokens=1600, purpose="bible")
        bible = Bible(
            premise=str(data.get("premise", "")), protagonist=str(data.get("protagonist", "")),
            world=str(data.get("world", "")), stakes=str(data.get("stakes", "")),
            promise=str(data.get("promise", "")), tone=str(data.get("tone", "")),
            pov=str(data.get("pov", "first person, past tense")),
            characters={str(k): str(v) for k, v in (data.get("characters") or {}).items()} if isinstance(data.get("characters"), dict) else {},
            rules=[str(r) for r in data.get("rules", [])],
            arc=[str(a) for a in data.get("arc", [])],
        )
        project.state.bible = bible
        project.save()
        return bible

    def _ending_exemplars(self, project: Project, k: int) -> str:
        if not self.exemplars:
            return "(none yet - ingest some novels and run `gauge train`)"
        q = f"{project.state.bible.premise} {project.state.bible.promise}"
        found = self.exemplars.search(q, kind="ending", k=k, genres=project.state.genres)
        return "\n\n".join(f"[{e.novel_title}, ch {e.chapter_index}]\n...{e.text}" for e in found) or "(none)"

    def extend_outline(self, project: Project, count: int = 5, exemplar_k: int = 3) -> list[ChapterPlan]:
        start = (project.state.outline[-1].number + 1) if project.state.outline else 1
        end = start + count - 1
        prompt = prompts.OUTLINE_PROMPT.format(
            start=start, end=end, bible=project.bible_markdown(),
            summary=project.state.running_summary or "(nothing yet - this is the beginning)",
            threads="\n".join(f"- {t}" for t in project.state.open_threads) or "(none)",
            ending_exemplars=self._ending_exemplars(project, exemplar_k),
        )
        data = generate_json(self.llm, prompt, system=self.system, schema=schema_of(OutlineOut), temperature=0.8,
                             max_tokens=1800, purpose="outline")
        plans: list[ChapterPlan] = []
        for i, c in enumerate(data.get("chapters", [])):
            try:
                number = int(c.get("number", start + i))
            except (TypeError, ValueError):
                number = start + i
            if number < start:
                number = start + i
            plans.append(ChapterPlan(
                number=number, title=str(c.get("title", f"Chapter {number}")), goal=str(c.get("goal", "")),
                hook=str(c.get("hook", "")), turn=str(c.get("turn", "")),
                cliffhanger=str(c.get("cliffhanger", "")), beats=[str(b) for b in c.get("beats", [])],
            ))
        plans = plans[:count]
        existing = {c.number for c in project.state.outline}
        project.state.outline.extend(p for p in plans if p.number not in existing)
        project.state.outline.sort(key=lambda c: c.number)
        project.save()
        return plans

    def ensure_plan(self, project: Project, number: int) -> ChapterPlan:
        plan = project.plan_for(number)
        tries = 0
        while plan is None and tries < 3:
            self.extend_outline(project)
            plan = project.plan_for(number)
            tries += 1
        if plan is None:
            plan = ChapterPlan(number=number, title=f"Chapter {number}", goal="advance the arc",
                               hook="open mid-action", turn="a reversal", cliffhanger="a new threat")
            project.state.outline.append(plan)
            project.save()
        return plan
