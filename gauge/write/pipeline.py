"""The chapter loop: plan -> draft -> critique -> revise -> accept -> update memory."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import GaugeConfig
from ..db import Database
from ..learn.trainer import Trainer
from ..llm.client import LLM, CallLog, generate_json, make_llm
from ..llm.parallel import run_parallel
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
from .schemas import SummaryOut, schema_of

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
    timings: dict[str, float] = field(default_factory=dict)  # seconds per step
    llm_calls: int = 0
    llm_seconds: float = 0.0
    resumed: bool = False


class WritePipeline:
    def __init__(self, cfg: GaugeConfig, db: Database, llm: Optional[LLM] = None, trainer: Optional[Trainer] = None,
                 fast_llm: Optional[LLM] = None):
        self.cfg = cfg
        self.db = db
        self.calls = CallLog()
        self.llm = llm or make_llm(cfg.llm, "main", self.calls)
        if llm is not None:
            llm.calls = self.calls  # share one log so timings cover everything
        # Small model for JSON work (briefs, reviews, memory, critic, summaries); main model for prose.
        self.fast_llm = fast_llm or (self.llm if llm is not None and cfg.llm.provider == "fake" else make_llm(cfg.llm, "fast", self.calls))
        if fast_llm is not None:
            fast_llm.calls = self.calls
        self._ctx: dict[str, object] = {"project": "", "chapter": None}
        self.calls.sink = self._log_call
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
        self.critic = Critic(self.fast_llm, self.model)

    def _log_call(self, rec) -> None:
        self.db.log_call(rec.purpose, rec.model, rec.prompt_tokens, rec.output_tokens, rec.duration_ms, rec.ok, rec.retries,
                         str(self._ctx.get("project") or ""), self._ctx.get("chapter"))  # type: ignore[arg-type]

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
        def one(c: Character) -> SceneBrief:
            scene = {"number": plan.number, "title": plan.title, "goal": plan.goal, "hook": plan.hook, "turn": plan.turn,
                     "cliffhanger": plan.cliffhanger, "beats": plan.beats, "summary": project.state.running_summary,
                     "prev_tail": " ".join(prev.split()[-250:]) if prev else "",
                     "others": [o.name for o in cast if o is not c]}
            return CharacterAgent(self.fast_llm, c).brief(scene)

        return [b for b in run_parallel(one, cast, self.cfg.llm.parallel, "scene brief") if b is not None]

    def _worth_attention(self, c: Character, text: str) -> bool:
        """Protagonist always; others only when they are really in the chapter, not name-dropped."""
        return c.role == "protagonist" and c.mentioned_in(text) or c.mentions(text) >= self.cfg.write.min_mentions_for_review

    def _character_reviews(self, text: str, number: int, roster: Roster, cast_names: list[str],
                           only: Optional[set[str]] = None) -> list[CharacterReview]:
        if not self.cfg.write.character_review:
            return []
        chars = [c for c in (roster.get(n) for n in cast_names) if c is not None and self._worth_attention(c, text)
                 and (only is None or c.name in only)]
        reviews = run_parallel(lambda c: CharacterAgent(self.fast_llm, c).review(text, number), chars,
                               self.cfg.llm.parallel, "character review")
        return [r for r in reviews if r is not None]

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
    def write_next(self, project: Project, fresh: bool = False) -> ChapterResult:
        n = project.next_chapter_number()
        self._ctx = {"project": project.name, "chapter": n}
        timings: dict[str, float] = {}
        calls_before = len(self.calls.records)
        secs_before = self.calls.total_seconds()
        t_all = time.monotonic()

        plan = self.planner.ensure_plan(project, n)
        prev = project.read_chapter(n - 1) if n > 1 else None
        words = project.state.target_chapter_words
        roster = self.roster(project) if self.cfg.write.characters_enabled else Roster(project.dir)

        # Resume from a checkpointed draft if the last run died after generating it.
        ckpt = None if fresh else project.load_checkpoint(n)
        resumed = False
        if ckpt and ckpt.get("plan_title") == plan.title:
            text = ckpt["text"]
            cast_names = list(ckpt.get("cast", []))
            revisions = int(ckpt.get("revisions", 0))
            resumed = True
            log.info("ch %d: resuming from checkpoint (%s, %d words)", n, ckpt.get("stage"), len(text.split()))
        else:
            # Activate the characters who are in this scene: each thinks about it as themselves.
            t0 = time.monotonic()
            briefs = self._activate_cast(project, plan, prev, roster) if roster.characters else []
            timings["briefs"] = time.monotonic() - t0
            cast_names = [b.name for b in briefs]
            cast_text = render_cast(briefs, roster) if briefs else ""
            t0 = time.monotonic()
            text = self.drafter.draft(project, plan, prev, words, cast=cast_text)
            timings["draft"] = time.monotonic() - t0
            revisions = 0
            if self.cfg.write.checkpoint_drafts:
                project.save_checkpoint(n, "draft", text, {"plan_title": plan.title, "cast": cast_names, "revisions": 0})

        t0 = time.monotonic()
        crit, reviews = self._critique(text, plan, n, roster, cast_names)
        timings["critique"] = time.monotonic() - t0
        best_text, best, best_reviews = text, crit, reviews
        t_rev = 0.0
        while crit.score < self.cfg.write.min_acceptable_score and revisions < self.cfg.write.max_revisions:
            revisions += 1
            log.info("ch %d score %.2f < %.2f; revising (%d)", n, crit.score, self.cfg.write.min_acceptable_score, revisions)
            problems = self.critic.problems(crit) + self._character_problems(reviews)
            t1 = time.monotonic()
            text = self.drafter.revise(text, problems, crit.measured_issues, words)
            if self.cfg.write.checkpoint_drafts:
                project.save_checkpoint(n, f"revision-{revisions}", text, {"plan_title": plan.title, "cast": cast_names,
                                                                             "revisions": revisions})
            # Only characters who had complaints are asked again; the others' verdicts stand.
            flagged = {r.name for r in reviews if r.issues}
            crit, new_reviews = self._critique(text, plan, n, roster, cast_names, only=flagged or None, previous=reviews)
            reviews = new_reviews
            t_rev += time.monotonic() - t1
            if crit.score > best.score:
                best_text, best, best_reviews = text, crit, reviews
        if t_rev:
            timings["revise"] = t_rev

        # Everyone who was really on the page remembers it, in their own terms (in parallel).
        t0 = time.monotonic()
        present = [c for c in roster.characters.values() if self._worth_attention(c, best_text)]
        run_parallel(lambda c: CharacterAgent(self.fast_llm, c).remember(best_text, n, roster), present,
                     self.cfg.llm.parallel, "character memory")
        if roster.characters:
            roster.save()
        self._update_memory(project, n, plan.title, best_text)
        timings["memory"] = time.monotonic() - t0

        meta = {
            "title": plan.title, "score": round(best.score, 3), "measured": round(best.measured, 3),
            "judged": round(best.judged, 3), "rubric": best.rubric, "revisions": revisions,
            "words": len(best_text.split()), "written_at": utcnow().isoformat(),
            "cast": cast_names, "character_consistency": round(self._consistency(best_reviews), 3),
            "resumed": resumed, "seconds": round(time.monotonic() - t_all, 1),
        }
        path = project.write_chapter(n, plan.title, best_text, meta)
        project.clear_checkpoint(n)
        project.save()
        self.db.record_generated(project.name, n, best.score, best.features)
        timings["total"] = time.monotonic() - t_all
        self._ctx = {"project": project.name, "chapter": None}
        return ChapterResult(number=n, title=plan.title, path=path, score=best.score, revisions=revisions, critique=best,
                             cast=cast_names, character_reviews=best_reviews, timings=timings,
                             llm_calls=len(self.calls.records) - calls_before,
                             llm_seconds=round(self.calls.total_seconds() - secs_before, 1), resumed=resumed)

    def _critique(self, text: str, plan, n: int, roster: Roster, cast_names: list[str],
                  only: Optional[set[str]] = None, previous: Optional[list[CharacterReview]] = None):
        crit = self.critic.critique(text, plan)
        reviews = self._character_reviews(text, n, roster, cast_names, only=only) if cast_names else []
        if only is not None and previous:
            redone = {r.name for r in reviews}
            reviews = reviews + [r for r in previous if r.name not in redone]
        if reviews:
            w = self.cfg.write.character_weight
            crit.score = (1 - w) * crit.score + w * self._consistency(reviews)
        return crit, reviews

    def _update_memory(self, project: Project, n: int, title: str, text: str) -> None:
        prompt = prompts.SUMMARY_PROMPT.format(summary=project.state.running_summary or "(none)", number=n,
                                               title=title, text=text)
        try:
            data = generate_json(self.fast_llm, prompt, schema=schema_of(SummaryOut), temperature=0.2, max_tokens=600,
                                 purpose="summary")
            if data.get("summary"):
                project.state.running_summary = str(data["summary"])
            threads = data.get("open_threads")
            if isinstance(threads, list):
                project.state.open_threads = [str(t) for t in threads][:8]
        except Exception as e:
            log.warning("summary update failed: %s; appending raw tail", e)
            project.state.running_summary = (project.state.running_summary + f"\nCh {n}: " + " ".join(text.split()[:60]))[-2000:]

    def write_many(self, project: Project, count: int, fresh: bool = False) -> list[ChapterResult]:
        return [self.write_next(project, fresh=fresh and i == 0) for i in range(count)]

    def status(self, project: Project) -> dict[str, object]:
        gens = self.db.generated(project.name)
        return {
            "project": project.name, "written": project.state.written, "outlined": len(project.state.outline),
            "avg_score": round(sum(g["score"] for g in gens) / len(gens), 3) if gens else None,
            "dir": str(project.dir), "open_threads": project.state.open_threads,
        }
