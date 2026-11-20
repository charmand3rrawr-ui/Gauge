"""The chapter loop: plan -> draft -> critique -> revise -> accept -> update memory."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import GaugeConfig
from ..db import Database
from ..learn.trainer import Trainer
from ..llm.client import LLM, extract_json, make_llm
from ..util import utcnow
from . import prompts
from .characters import (
    Character,
    CharacterAgent,
    CharacterCreator,
    CharacterReview,
    PersonalityMatrix,
    Roster,
    SceneBrief,
    render_cast,
)
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
    cast: list[str] = field(default_factory=list)
    character_reviews: list[CharacterReview] = field(default_factory=list)


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
        if self.cfg.write.characters_enabled:
            self.sync_characters(p)
        self.planner.extend_outline(p, count=5)
        return p

    # ---- characters --------------------------------------------------------
    def roster(self, project: Project) -> Roster:
        return Roster.load(project.dir)

    def sync_characters(self, project: Project) -> list[str]:
        """Create agents for every character named in the bible that has no sheet yet."""
        roster = self.roster(project)
        creator = CharacterCreator(self.llm)
        b = project.state.bible
        wanted: list[tuple[str, str, str]] = []
        if b.protagonist:
            pname = b.protagonist.split(",")[0].split(":")[0].split(" - ")[0].strip()
            pname = " ".join(pname.split()[:3]) if pname else "Protagonist"
            wanted.append((pname, "protagonist", b.protagonist))
        for name, desc in b.characters.items():
            role = "antagonist" if any(w in desc.lower() for w in ("antagonist", "villain", "rival", "enemy")) else "supporting"
            wanted.append((name.strip(), role, desc))
        created = []
        for name, role, concept in wanted:
            if not name or roster.get(name):
                continue
            c = creator.from_concept(name, concept, role, project.bible_markdown(), roster.names())
            c.created_chapter = project.state.written
            roster.add(c)
            created.append(c.name)
        if created:
            roster.save()
        return created

    def create_character(self, project: Project, name: str, concept: str, role: str = "supporting",
                         seed: Optional[int] = None) -> Character:
        roster = self.roster(project)
        seed_matrix = PersonalityMatrix.random(seed) if seed is not None else None
        c = CharacterCreator(self.llm).from_concept(name, concept, role, project.bible_markdown(), roster.names(), seed_matrix)
        c.created_chapter = project.state.written
        roster.add(c)
        roster.save()
        return c

    def _activate_cast(self, project: Project, plan, prev: Optional[str], roster: Roster) -> list[SceneBrief]:
        texts = [plan.title, plan.goal, plan.hook, plan.turn, plan.cliffhanger, " ".join(plan.beats)]
        if prev:
            texts.append(" ".join(prev.split()[-400:]))
        cast = roster.present_in(texts, limit=self.cfg.write.max_active_characters)
        briefs: list[SceneBrief] = []
        for c in cast:
            scene = {"number": plan.number, "title": plan.title, "goal": plan.goal, "hook": plan.hook, "turn": plan.turn,
                     "cliffhanger": plan.cliffhanger, "beats": plan.beats, "summary": project.state.running_summary,
                     "prev_tail": " ".join(prev.split()[-250:]) if prev else "",
                     "others": [o.name for o in cast if o is not c]}
            briefs.append(CharacterAgent(self.llm, c).brief(scene))
        return briefs

    def _character_reviews(self, text: str, number: int, roster: Roster, cast_names: list[str]) -> list[CharacterReview]:
        if not self.cfg.write.character_review:
            return []
        out = []
        for name in cast_names:
            c = roster.get(name)
            if c is not None and c.mentioned_in(text):
                out.append(CharacterAgent(self.llm, c).review(text, number))
        return out

    @staticmethod
    def _consistency(reviews: list[CharacterReview]) -> float:
        return sum(r.consistency for r in reviews) / len(reviews) if reviews else 1.0

    @staticmethod
    def _character_problems(reviews: list[CharacterReview]) -> list[str]:
        out = []
        for r in reviews:
            for i in r.issues[:2]:
                out.append(f"{r.name} out of character at \"{i.quote}\": {i.problem}. Instead: {i.fix}")
        return out

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

        # Activate the characters who are in this scene: each thinks about it as themselves.
        roster = self.roster(project) if self.cfg.write.characters_enabled else Roster(project.dir)
        briefs = self._activate_cast(project, plan, prev, roster) if roster.characters else []
        cast_names = [b.name for b in briefs]
        cast_text = render_cast(briefs, roster) if briefs else ""

        text = self.drafter.draft(project, plan, prev, words, cast=cast_text)
        crit, reviews = self._critique(text, plan, n, roster, cast_names)
        revisions = 0
        best_text, best, best_reviews = text, crit, reviews
        while crit.score < self.cfg.write.min_acceptable_score and revisions < self.cfg.write.max_revisions:
            revisions += 1
            log.info("ch %d score %.2f < %.2f; revising (%d)", n, crit.score, self.cfg.write.min_acceptable_score, revisions)
            problems = self.critic.problems(crit) + self._character_problems(reviews)
            text = self.drafter.revise(text, problems, crit.measured_issues, words)
            crit, reviews = self._critique(text, plan, n, roster, cast_names)
            if crit.score > best.score:
                best_text, best, best_reviews = text, crit, reviews

        # Everyone who was on the page remembers it, in their own terms.
        present = [c for c in roster.characters.values() if c.mentioned_in(best_text)]
        for c in present:
            CharacterAgent(self.llm, c).remember(best_text, n, roster)
        if roster.characters:
            roster.save()

        meta = {
            "title": plan.title, "score": round(best.score, 3), "measured": round(best.measured, 3),
            "judged": round(best.judged, 3), "rubric": best.rubric, "revisions": revisions,
            "words": len(best_text.split()), "written_at": utcnow().isoformat(),
            "cast": cast_names, "character_consistency": round(self._consistency(best_reviews), 3),
        }
        path = project.write_chapter(n, plan.title, best_text, meta)
        self._update_memory(project, n, plan.title, best_text)
        project.save()
        self.db.record_generated(project.name, n, best.score, best.features)
        return ChapterResult(number=n, title=plan.title, path=path, score=best.score, revisions=revisions, critique=best,
                             cast=cast_names, character_reviews=best_reviews)

    def _critique(self, text: str, plan, n: int, roster: Roster, cast_names: list[str]):
        crit = self.critic.critique(text, plan)
        reviews = self._character_reviews(text, n, roster, cast_names) if cast_names else []
        if reviews:
            w = self.cfg.write.character_weight
            crit.score = (1 - w) * crit.score + w * self._consistency(reviews)
        return crit, reviews

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
