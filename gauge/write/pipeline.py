"""The chapter loop: plan -> draft -> critique -> revise -> accept -> update memory."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import GaugeConfig
from ..db import Database
from ..learn.trainer import Trainer
from ..llm.client import LLM, extract_json, make_llm
from ..util import utcnow
from . import prompts
from .critic import Critic, Critique
from .drafter import Drafter
from .planner import Planner
from .project import Project

log = logging.getLogger(__name__)


@dataclass
class ChapterResult:
    number: int
    title: str
    path: Path
    score: float
    revisions: int
    critique: Critique


class WritePipeline:
    def __init__(self, cfg: GaugeConfig, db: Database, llm: Optional[LLM] = None, trainer: Optional[Trainer] = None):
        self.cfg = cfg
        self.db = db
        self.llm = llm or make_llm(cfg.llm)
        self.trainer = trainer or Trainer(cfg, db)
        self.model = self.trainer.load_model()
        self.exemplars = self.trainer.load_exemplars()
        if not self.exemplars.items:
            self.exemplars = None  # type: ignore[assignment]
        self.guide = self.trainer.load_guide()
        from ..learn.style import compact_guide_for_prompt

        compact = compact_guide_for_prompt(self.guide) if self.guide else ""
        self.planner = Planner(self.llm, self.exemplars, compact)
        self.drafter = Drafter(self.llm, self.exemplars, compact, cfg.write.exemplars_in_prompt)
        self.critic = Critic(self.llm, self.model)

    # ---- project setup -----------------------------------------------------
    def new_project(self, name: str, idea: str, genres: list[str], words: Optional[int] = None) -> Project:
        p = Project.create(self.cfg.projects_dir, name, idea, genres, words or self.cfg.write.target_chapter_words)
        self.planner.build_bible(p)
        self.planner.extend_outline(p, count=5)
        return p

    def open_project(self, name: str) -> Project:
        if not Project.exists(self.cfg.projects_dir, name):
            raise FileNotFoundError(f"no project named {name!r}; create one with `gauge new`")
        return Project(self.cfg.projects_dir, name)

    # ---- one chapter -------------------------------------------------------
    def write_next(self, project: Project) -> ChapterResult:
        n = project.next_chapter_number()
        plan = self.planner.ensure_plan(project, n)
        prev = project.read_chapter(n - 1) if n > 1 else None
        words = project.state.target_chapter_words
        text = self.drafter.draft(project, plan, prev, words)
        crit = self.critic.critique(text, plan)
        revisions = 0
        best_text, best = text, crit
        while crit.score < self.cfg.write.min_acceptable_score and revisions < self.cfg.write.max_revisions:
            revisions += 1
            log.info("ch %d score %.2f < %.2f; revising (%d)", n, crit.score, self.cfg.write.min_acceptable_score, revisions)
            text = self.drafter.revise(text, self.critic.problems(crit), crit.measured_issues, words)
            crit = self.critic.critique(text, plan)
            if crit.score > best.score:
                best_text, best = text, crit
        meta = {
            "title": plan.title, "score": round(best.score, 3), "measured": round(best.measured, 3),
            "judged": round(best.judged, 3), "rubric": best.rubric, "revisions": revisions,
            "words": len(best_text.split()), "written_at": utcnow().isoformat(),
        }
        path = project.write_chapter(n, plan.title, best_text, meta)
        self._update_memory(project, n, plan.title, best_text)
        project.save()
        self.db.record_generated(project.name, n, best.score, best.features)
        return ChapterResult(number=n, title=plan.title, path=path, score=best.score, revisions=revisions, critique=best)

    def _update_memory(self, project: Project, n: int, title: str, text: str) -> None:
        prompt = prompts.SUMMARY_PROMPT.format(summary=project.state.running_summary or "(none)", number=n,
                                               title=title, text=text)
        try:
            data = extract_json(self.llm.generate(prompt, json_mode=True, temperature=0.2, max_tokens=600))
            if data.get("summary"):
                project.state.running_summary = str(data["summary"])
            threads = data.get("open_threads")
            if isinstance(threads, list):
                project.state.open_threads = [str(t) for t in threads][:8]
        except Exception as e:
            log.warning("summary update failed: %s; appending raw tail", e)
            project.state.running_summary = (project.state.running_summary + f"\nCh {n}: " + " ".join(text.split()[:60]))[-2000:]

    def write_many(self, project: Project, count: int) -> list[ChapterResult]:
        return [self.write_next(project) for _ in range(count)]

    def status(self, project: Project) -> dict[str, object]:
        gens = self.db.generated(project.name)
        return {
            "project": project.name, "written": project.state.written, "outlined": len(project.state.outline),
            "avg_score": round(sum(g["score"] for g in gens) / len(gens), 3) if gens else None,
            "dir": str(project.dir), "open_threads": project.state.open_threads,
        }
