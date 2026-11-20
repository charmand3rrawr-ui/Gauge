"""The chapter loop: plan -> draft -> critique -> revise -> accept -> update memory."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import GaugeConfig
from ..db import Database
from ..learn.techniques import TechniqueLibrary
from ..learn.trainer import Trainer
from ..llm.client import LLM, CallLog, generate_json, make_llm
from ..llm.parallel import run_parallel
from ..llm.tokens import truncate_to_tokens
from ..util import utcnow
from . import prompts
from .agents import AgentCouncil, AgentReview, SceneContext, build_council
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
from .muse import Muse, NoveltyReport, NoveltyTracker, direction_record
from .planner import Planner
from .project import Project
from .schemas import ContinueSummaryOut, SummaryOut, schema_of
from .world import WorldAgent, WorldBrief, WorldReview, WorldState, load_world, save_world

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
    world_review: Optional[WorldReview] = None
    novelty: Optional[NoveltyReport] = None
    agent_reviews: list[AgentReview] = field(default_factory=list)
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
        self.techniques = TechniqueLibrary(self.trainer.embedder)
        self.techniques.load(cfg.models_dir)
        self.novelty = NoveltyTracker()
        hint = "\n".join(t.card() for t in self.techniques.items[:12]) if self.techniques.items else ""
        self.muse = Muse(self.fast_llm, hint)
        if cfg.write.muse_enabled:
            self.planner.direction_for = self._pick_direction

    def _log_call(self, rec) -> None:
        self.db.log_call(rec.purpose, rec.model, rec.prompt_tokens, rec.output_tokens, rec.duration_ms, rec.ok, rec.retries,
                         str(self._ctx.get("project") or ""), self._ctx.get("chapter"))  # type: ignore[arg-type]

    # ---- project setup -----------------------------------------------------
    def new_project(self, name: str, idea: str, genres: list[str], words: Optional[int] = None,
                    premise_doc: str = "", continue_from: Optional[str] = None) -> Project:
        """Create a project from a one-line idea, a full premise document, and/or an existing manuscript."""
        p = Project.create(self.cfg.projects_dir, name, idea or premise_doc[:200], genres,
                           words or self.cfg.write.target_chapter_words, premise_doc=premise_doc)
        if continue_from:
            self.continue_from(p, continue_from)
        self.planner.build_bible(p)
        if self.cfg.write.characters_enabled:
            self.sync_characters(p)
        if self.cfg.write.world_enabled:
            self.build_world(p)
        self.setup_agents(p)
        self.planner.extend_outline(p, count=5)
        return p

    def continue_from(self, project: Project, path: str) -> int:
        """Import an existing manuscript as the chapters already written, then summarise it so the
        story continues from there instead of starting over."""
        from ..ingest.local import import_path

        imported = import_path(path, library=False)
        if not imported:
            raise ValueError(f"no chapters found in {path}")
        chapters = imported[0].chapters
        for ch in chapters:
            project.write_chapter(ch.index, ch.title or f"Chapter {ch.index}", ch.text, {"title": ch.title, "imported": True,
                                                                                            "words": ch.words})
        project.state.continued_from = str(path)
        # Summarise with as much of the manuscript as fits: the last chapters matter most.
        room = self.fast_llm.context_window() - 1200
        text = "\n\n".join(f"## {c.title or 'Chapter ' + str(c.index)}\n{c.text}" for c in chapters)
        text = truncate_to_tokens(text, room, from_end=True)
        prompt = prompts.CONTINUE_SUMMARY_PROMPT.format(count=len(chapters), text=text)
        try:
            d = generate_json(self.fast_llm, prompt, schema=schema_of(ContinueSummaryOut), temperature=0.2, max_tokens=900,
                              purpose="continue:summary")
        except Exception as e:
            log.warning("manuscript summary failed: %s", e)
            d = {}
        project.state.running_summary = str(d.get("summary") or " ".join(chapters[-1].text.split()[-200:]))
        project.state.open_threads = [str(t) for t in d.get("open_threads", [])][:8] if isinstance(d.get("open_threads"), list) else []
        if not project.state.premise_doc:
            cast = "\n".join(f"- {k}: {v}" for k, v in (d.get("characters") or {}).items()) if isinstance(d.get("characters"), dict) else ""
            notes = "\n".join(f"- {w}" for w in d.get("world_notes", [])) if isinstance(d.get("world_notes"), list) else ""
            project.state.premise_doc = (f"{d.get('premise', '')}\n\nCHARACTERS SO FAR:\n{cast}\n\nWORLD RULES SEEN:\n{notes}\n\n"
                                         f"STORY SO FAR:\n{project.state.running_summary}").strip()
        project.save()
        return len(chapters)

    # ---- the writing room --------------------------------------------------
    def council(self, project: Project) -> AgentCouncil:
        return build_council(self.cfg, self.fast_llm, project, embedder=self.trainer.embedder)

    def setup_agents(self, project: Project) -> list[str]:
        c = self.council(project)
        c.setup(project)
        return c.names()

    # ---- world -------------------------------------------------------------
    def world(self, project: Project) -> Optional[WorldState]:
        return load_world(project.dir) if self.cfg.write.world_enabled else None

    def build_world(self, project: Project) -> WorldState:
        w = load_world(project.dir) or WorldState()
        agent = WorldAgent(self.fast_llm, w, self.cfg.write.world_agency)
        agent.build(project.bible_markdown(), project.state.premise_doc)
        save_world(project.dir, w)
        return w

    def _world_brief(self, project: Project, plan, w: WorldState, briefs: list[SceneBrief]) -> WorldBrief:
        cast_summary = "\n".join(f"- {b.name}: wants {b.goal}; mood {b.mood}" + (f"; friction: {b.conflict_with_plan}" if b.conflict_with_plan else "")
                                 for b in briefs) or ""
        conflicts = sum(1 for b in briefs if b.conflict_with_plan)
        scene = {"number": plan.number, "title": plan.title, "goal": plan.goal, "hook": plan.hook, "turn": plan.turn,
                 "cliffhanger": plan.cliffhanger, "beats": plan.beats}
        return WorldAgent(self.fast_llm, w, self.cfg.write.world_agency).brief(scene, cast_summary, conflicts)

    # ---- creativity --------------------------------------------------------
    def _pick_direction(self, project: Project) -> str:
        notes = ""
        try:
            notes = self.council(project).plan_notes(project)
        except Exception as e:
            log.warning("council plan notes failed: %s", e)
        d = self.muse.choose(project.bible_markdown(), project.state.running_summary, project.state.open_threads,
                             project.state.past_directions, self.cfg.write.muse_directions)
        if d is None:
            return notes
        project.state.past_directions.append(d.pitch)
        project.state.past_directions = project.state.past_directions[-20:]
        log.info("muse chose: %s (%s)", d.title, direction_record(d))
        text = f"{d.title}: {d.pitch}" + (f" Fresh because {d.fresh_because}" if d.fresh_because else "")
        return text + (f"\nConstraints from the writing room:\n{notes}" if notes else "")

    def _technique_cards(self, plan) -> str:
        if not self.techniques.items or self.cfg.write.techniques_in_prompt <= 0:
            return ""
        found = self.techniques.search(f"{plan.goal} {plan.hook} {plan.turn} {plan.cliffhanger}", k=self.cfg.write.techniques_in_prompt)
        return "\n".join(f"- {t.card()}" for t in found)

    def _avoid_list(self, project: Project, n: int) -> tuple[str, list[str]]:
        previous = project.previous_texts(n)
        if not previous:
            return "", previous
        phrases = self.novelty.overused_phrases(previous)
        openings = [" ".join(p.split()[:7]) for p in previous[-3:]]
        parts = []
        if phrases:
            parts.append("phrases already worn out: " + ", ".join(f'"{p}"' for p in phrases[:10]))
        if openings:
            parts.append("recent chapter openings (open differently): " + " | ".join(f'"{o}..."' for o in openings))
        return "\n".join(f"- {x}" for x in parts), previous

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
        world_state = self.world(project)
        previous_texts = project.previous_texts(n)
        council = self.council(project)
        ctx = SceneContext(project=project, number=n, plan=plan, prev_tail=" ".join(prev.split()[-250:]) if prev else "",
                           summary=project.state.running_summary, genres=project.state.genres)

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
            world_text = ""
            if world_state is not None:
                t0 = time.monotonic()
                world_text = self._world_brief(project, plan, world_state, briefs).render()
                timings["world"] = time.monotonic() - t0
            avoid_text, _prev_texts = self._avoid_list(project, n)
            ctx.cast_text, ctx.world_text = cast_text, world_text
            t0 = time.monotonic()
            council_text = council.brief(ctx) if council.agents else ""
            timings["council"] = time.monotonic() - t0
            t0 = time.monotonic()
            text = self.drafter.draft(project, plan, prev, words, cast=cast_text, world=world_text,
                                      techniques=self._technique_cards(plan), avoid=avoid_text, council=council_text)
            timings["draft"] = time.monotonic() - t0
            revisions = 0
            if self.cfg.write.checkpoint_drafts:
                project.save_checkpoint(n, "draft", text, {"plan_title": plan.title, "cast": cast_names, "revisions": 0})

        t0 = time.monotonic()
        crit, reviews, wrev, nov, agent_revs = self._critique(text, plan, n, roster, cast_names, world_state, previous_texts,
                                                              council=council, ctx=ctx)
        timings["critique"] = time.monotonic() - t0
        best_text, best, best_reviews, best_wrev, best_nov, best_agents = text, crit, reviews, wrev, nov, agent_revs
        t_rev = 0.0
        while crit.score < self.cfg.write.min_acceptable_score and revisions < self.cfg.write.max_revisions:
            revisions += 1
            log.info("ch %d score %.2f < %.2f; revising (%d)", n, crit.score, self.cfg.write.min_acceptable_score, revisions)
            problems = (self.critic.problems(crit) + self._character_problems(reviews) + self._world_problems(wrev)
                        + (nov.notes if nov else []) + AgentCouncil.problems(agent_revs))
            t1 = time.monotonic()
            text = self.drafter.revise(text, problems, crit.measured_issues, words)
            if self.cfg.write.checkpoint_drafts:
                project.save_checkpoint(n, f"revision-{revisions}", text, {"plan_title": plan.title, "cast": cast_names,
                                                                             "revisions": revisions})
            # Only characters who had complaints are asked again; the others' verdicts stand.
            flagged = {r.name for r in reviews if r.issues}
            crit, new_reviews, wrev, nov, agent_revs = self._critique(text, plan, n, roster, cast_names, world_state, previous_texts,
                                                                      only=flagged or None, previous=reviews, council=council, ctx=ctx)
            reviews = new_reviews
            t_rev += time.monotonic() - t1
            if crit.score > best.score:
                best_text, best, best_reviews, best_wrev, best_nov, best_agents = text, crit, reviews, wrev, nov, agent_revs
        if t_rev:
            timings["revise"] = t_rev

        # Everyone who was really on the page remembers it, in their own terms (in parallel).
        t0 = time.monotonic()
        present = [c for c in roster.characters.values() if self._worth_attention(c, best_text)]
        run_parallel(lambda c: CharacterAgent(self.fast_llm, c).remember(best_text, n, roster), present,
                     self.cfg.llm.parallel, "character memory")
        if roster.characters:
            roster.save()
        if world_state is not None:
            WorldAgent(self.fast_llm, world_state, self.cfg.write.world_agency).remember(best_text, n)
            save_world(project.dir, world_state)
            promotable = [f.tag for f in world_state.promotable()]
            if promotable:
                log.info("recurring extras worth promoting to characters: %s (gauge world promote)", ", ".join(promotable))
        self._update_memory(project, n, plan.title, best_text)
        if council.agents:
            # Re-run the cheap deterministic parts on the accepted text so memories match what was saved.
            council.remember(best_text, ctx)
        timings["memory"] = time.monotonic() - t0
        new_title = council.retitle(best_text, plan, n) if council.agents else None
        if new_title:
            log.info("packager retitled ch %d: %r -> %r", n, plan.title, new_title)
            plan.title = new_title

        meta = {
            "title": plan.title, "score": round(best.score, 3), "measured": round(best.measured, 3),
            "judged": round(best.judged, 3), "rubric": best.rubric, "revisions": revisions,
            "words": len(best_text.split()), "written_at": utcnow().isoformat(),
            "cast": cast_names, "character_consistency": round(self._consistency(best_reviews), 3),
            "world_consistency": round(best_wrev.consistency, 3) if best_wrev else None,
            "novelty": round(best_nov.score, 3) if best_nov else None,
            "direction": project.state.current_direction[:120],
            "agents": AgentCouncil.summary(best_agents),
            "content_flags": next((r.data.get("flags", []) for r in best_agents if r.agent == "sensitivity"), []),
            "resumed": resumed, "seconds": round(time.monotonic() - t_all, 1),
        }
        path = project.write_chapter(n, plan.title, best_text, meta)
        project.clear_checkpoint(n)
        project.save()
        self.db.record_generated(project.name, n, best.score, best.features)
        timings["total"] = time.monotonic() - t_all
        self._ctx = {"project": project.name, "chapter": None}
        return ChapterResult(number=n, title=plan.title, path=path, score=best.score, revisions=revisions, critique=best,
                             cast=cast_names, character_reviews=best_reviews, world_review=best_wrev, novelty=best_nov,
                             agent_reviews=best_agents, timings=timings,
                             llm_calls=len(self.calls.records) - calls_before,
                             llm_seconds=round(self.calls.total_seconds() - secs_before, 1), resumed=resumed)

    def _critique(self, text: str, plan, n: int, roster: Roster, cast_names: list[str],
                  world_state: Optional[WorldState] = None, previous_texts: Optional[list[str]] = None,
                  only: Optional[set[str]] = None, previous: Optional[list[CharacterReview]] = None,
                  council: Optional[AgentCouncil] = None, ctx: Optional[SceneContext] = None):
        crit = self.critic.critique(text, plan)
        reviews = self._character_reviews(text, n, roster, cast_names, only=only) if cast_names else []
        if only is not None and previous:
            redone = {r.name for r in reviews}
            reviews = reviews + [r for r in previous if r.name not in redone]
        if reviews:
            w = self.cfg.write.character_weight
            crit.score = (1 - w) * crit.score + w * self._consistency(reviews)
        wrev = None
        if world_state is not None and self.cfg.write.world_weight > 0:
            wrev = WorldAgent(self.fast_llm, world_state, self.cfg.write.world_agency).review(text, n)
            w = self.cfg.write.world_weight
            crit.score = (1 - w) * crit.score + w * wrev.consistency
        nov = None
        if previous_texts and self.cfg.write.novelty_weight > 0:
            nov = self.novelty.report(text, previous_texts)
            w = self.cfg.write.novelty_weight
            crit.score = (1 - w) * crit.score + w * nov.score
        agent_revs: list[AgentReview] = []
        if council is not None and council.agents and ctx is not None:
            agent_revs = council.review(text, ctx)
            cs = council.score(agent_revs)
            if cs is not None:
                w = council.weight
                crit.score = (1 - w) * crit.score + w * cs
        return crit, reviews, wrev, nov, agent_revs

    @staticmethod
    def _world_problems(wrev: Optional[WorldReview]) -> list[str]:
        if not wrev:
            return []
        return [f"world rule broken at \"{v['quote']}\": {v['rule']}. Fix: {v['fix']}" for v in wrev.violations[:3]]

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
