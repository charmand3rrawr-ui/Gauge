"""The chapter loop: plan -> draft -> critique -> revise -> accept -> update memory."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import GaugeConfig
from ..db import Database
from ..learn.bandit import PromptBandit
from ..learn.finetune import record_self_pair, save_accepted_copy
from ..learn.lessons import distill, prompt_lines
from ..learn.project_model import ProjectModel
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
from .improv import render_improv, run_improv
from .muse import Muse, NoveltyReport, NoveltyTracker, direction_record
from .persona import load_persona
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
    words: int = 0  # how long the chapter came out
    words_asked: int = 0  # how long it was asked to be


MIN_CHAPTER_WORDS, MAX_CHAPTER_WORDS = 100, 20000


class WritePipeline:
    def __init__(self, cfg: GaugeConfig, db: Database, llm: Optional[LLM] = None, trainer: Optional[Trainer] = None,
                 fast_llm: Optional[LLM] = None):
        self.cfg = cfg
        if cfg.seed is not None:
            from ..util import seed_all
            seed_all(cfg.seed)
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
        self._project_model_cache: dict[int, ProjectModel] = {}
        self.calls.sink = self._log_call
        # Spread prose generation over your workers that serve a model (opt-in, token required).
        if llm is None and cfg.compute.pool_generation and cfg.compute.token and db.workers():
            from ..compute import Pool, PoolLLM

            self.llm = PoolLLM(self.llm, Pool(db, cfg.compute.token), self.calls)
        self.trainer = trainer or Trainer(cfg, db)
        self.model = self.trainer.load_model()
        self.exemplars = self.trainer.load_exemplars()
        if not self.exemplars.items:
            self.exemplars = None  # type: ignore[assignment]
        self.guide = self.trainer.load_guide()
        from ..learn.style import compact_guide_for_prompt

        compact = compact_guide_for_prompt(self.guide) if self.guide else ""
        self.persona = load_persona(cfg.models_dir)
        persona_block = self.persona.prompt_block(cfg.write.persona_blend) if self.persona else ""
        try:
            lessons = prompt_lines(distill(db), cfg.learn.lessons_in_prompt) if cfg.learn.lessons_in_prompt > 0 else ""
        except Exception:
            lessons = ""
        if lessons:
            compact = (compact + "\n\nLessons from your own earlier chapters:\n" + lessons).strip()
        if persona_block:
            compact = (persona_block + "\n\n" + compact).strip()
        if cfg.write.rebel:
            compact = (prompts.REBEL_BLOCK.format(level=cfg.write.rebel_level) + "\n\n" + compact).strip()
        self.guide_compact = compact
        self.planner = Planner(self.llm, self.exemplars, compact)
        self.drafter = Drafter(self.llm, self.exemplars, compact, cfg.write.exemplars_in_prompt)
        self.critic = Critic(self.fast_llm, self.model, rebel=cfg.write.rebel, retention_weight=cfg.write.retention_weight, realism_weight=cfg.write.realism_weight)
        self.techniques = TechniqueLibrary(self.trainer.embedder)
        self.techniques.load(cfg.models_dir)
        self.novelty = NoveltyTracker()
        hint = "\n".join(t.card() for t in self.techniques.items[:12]) if self.techniques.items else ""
        self.muse = Muse(self.fast_llm, hint, rebel_level=cfg.write.rebel_level if cfg.write.rebel else 0.0)
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
        if self.cfg.research.enabled and self.cfg.research.dossier_on_create:
            try:
                self.build_dossier(p)
            except Exception as e:
                log.warning("project dossier failed: %s", e)
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

    # ---- learning from your own serial ------------------------------------
    def bandit(self, project: Project) -> Optional[PromptBandit]:
        return PromptBandit(project.dir / "bandit.json", seed=self.cfg.seed) if self.cfg.learn.bandit_enabled else None

    def _apply_settings(self, settings: dict) -> dict:
        """Apply a bandit draw to the knobs for this chapter; return the previous values."""
        prev = {"exemplars": self.drafter.k, "temperature": self.cfg.llm.temperature, "techniques": self.cfg.write.techniques_in_prompt,
                "world_agency": self.cfg.write.world_agency}
        self.drafter.k = int(settings.get("exemplars", prev["exemplars"]))
        self.cfg.llm.temperature = float(settings.get("temperature", prev["temperature"]))
        self.cfg.write.techniques_in_prompt = int(settings.get("techniques", prev["techniques"]))
        self.cfg.write.world_agency = float(settings.get("world_agency", prev["world_agency"]))
        return prev

    def _restore_settings(self, prev: dict) -> None:
        self.drafter.k = prev["exemplars"]
        self.cfg.llm.temperature = prev["temperature"]
        self.cfg.write.techniques_in_prompt = prev["techniques"]
        self.cfg.write.world_agency = prev["world_agency"]

    def project_model(self, project: Project) -> ProjectModel:
        return ProjectModel(project.dir, "retention")

    def record_feedback(self, project: Project, chapter: int, views: Optional[float] = None, comments: Optional[float] = None,
                        likes: Optional[float] = None) -> dict:
        """Real reader numbers for a posted chapter: feed the panel, the bandit (delayed reward) and the project model."""
        self.db.record_feedback(project.state.name, chapter, views, comments, likes)
        fb = self.db.feedback(project.state.name)
        out: dict = {"recorded": chapter}
        council = self.council(project)
        panel = council.get("reader_panel")
        if panel is not None:
            panel.record_actual(chapter, views, comments)
            panel.save()
            out["panel"] = panel.calibration()
        b = self.bandit(project)
        if b is not None:
            prev = fb.get(chapter - 1, {}).get("views")
            if prev and views:
                retention = max(0.0, min(1.0, views / prev))
                if b.update(chapter - 1, retention, weight=2.0, kind="retention"):
                    out["bandit_reward"] = {"chapter": chapter - 1, "retention": round(retention, 3)}
            b.save()
        pm = self.project_model(project)
        fitted = pm.fit(project, fb, min_chapters=self.cfg.learn.project_model_min_chapters)
        out["project_model"] = {"n": fitted.n_train, "r2_cv": round(fitted.r2_cv, 3) if fitted.r2_cv == fitted.r2_cv else None} if fitted else "not enough labelled chapters"
        return out

    def _improv(self, project: Project, plan, roster: Roster, cast_names: list[str], world_state, level: float) -> str:
        if not self.cfg.write.improv_enabled or not cast_names:
            return ""
        cast = [c for c in (roster.get(n) for n in cast_names) if c is not None]
        situation = f"{plan.goal}. It opens: {plan.hook}. The turn: {plan.turn}."
        transcript = run_improv(self.fast_llm, cast, situation, rounds=self.cfg.write.improv_rounds, world=world_state,
                                world_level=level, max_cast=self.cfg.write.improv_max_cast)
        return render_improv(transcript)

    # ---- the writing room --------------------------------------------------
    def council(self, project: Project) -> AgentCouncil:
        return build_council(self.cfg, self.fast_llm, project, embedder=self.trainer.embedder, scout=self.scout)

    @property
    def scout(self):
        if not self.cfg.research.enabled:
            return None
        if getattr(self, "_scout", None) is None:
            from ..research.scout import Scout

            self._scout = Scout(self.cfg, self.db, sources=getattr(self, "_scout_sources", None))
        return self._scout

    def use_research_sources(self, sources: dict) -> None:
        """Inject sources (tests, offline runs) before the first council is built."""
        self._scout_sources = sources
        self._scout = None

    def build_dossier(self, project: Project, topics: Optional[list[str]] = None, per_topic_sources: tuple = ("wikipedia",)) -> Optional[Path]:
        """A starting dossier for a new project: look up the bible's proper nouns and trades."""
        if self.scout is None:
            return None
        from ..write.agents.scout import extract_topics

        topics = topics or extract_topics(project.bible_markdown(), project.state.premise_doc)
        queries = [{"query": t, "source": src, "why": "project background"} for t in topics for src in per_topic_sources]
        saved_max = self.cfg.research.max_queries_per_chapter
        self.cfg.research.max_queries_per_chapter = max(saved_max, len(queries))
        try:
            dossier = self.scout.research(queries, project=project.state.name, chapter=0)
        finally:
            self.cfg.research.max_queries_per_chapter = saved_max
        return self.scout.file(project.dir, dossier, "project dossier")

    def setup_agents(self, project: Project) -> list[str]:
        c = self.council(project)
        c.setup(project)
        return c.names()

    # ---- world -------------------------------------------------------------
    def world(self, project: Project) -> Optional[WorldState]:
        return load_world(project.dir) if self.cfg.write.world_enabled else None

    def build_world(self, project: Project) -> WorldState:
        w = load_world(project.dir) or WorldState()
        agent = WorldAgent(self.fast_llm, w, self.cfg.write.world_agency, realism=self.cfg.write.world_realism)
        agent.build(project.bible_markdown(), project.state.premise_doc)
        save_world(project.dir, w)
        return w

    def _world_brief(self, project: Project, plan, w: WorldState, briefs: list[SceneBrief]) -> WorldBrief:
        cast_summary = "\n".join(f"- {b.name}: wants {b.goal}; mood {b.mood}" + (f"; friction: {b.conflict_with_plan}" if b.conflict_with_plan else "")
                                 for b in briefs) or ""
        conflicts = sum(1 for b in briefs if b.conflict_with_plan)
        scene = {"number": plan.number, "title": plan.title, "goal": plan.goal, "hook": plan.hook, "turn": plan.turn,
                 "cliffhanger": plan.cliffhanger, "beats": plan.beats}
        return WorldAgent(self.fast_llm, w, self.cfg.write.world_agency, realism=self.cfg.write.world_realism).brief(scene, cast_summary, conflicts)

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

    def _carry_character(self, c: Character) -> None:
        """When carry_characters is on, seed a new character from their cross-book memory."""
        if not self.cfg.write.carry_characters:
            return
        try:
            from .character_memory import GlobalCharacterMemory

            if GlobalCharacterMemory(self.cfg.models_dir).carry_into(c):
                log.info("carried cross-book memory into %s", c.name)
        except Exception as e:
            log.warning("could not carry character memory for %s: %s", c.name, e)

    def character_creator(self) -> CharacterCreator:
        """The creator, taught by the character school. GaugeLM's context is too small for examples, so
        with it the school teaches through training instead of through the prompt."""
        from ..learn.character_school import CharacterSchool

        small = self.cfg.llm.provider == "native"
        return CharacterCreator(self.llm, CharacterSchool(self.cfg.models_dir),
                                examples=0 if small else self.cfg.write.character_examples,
                                checklist=not small, repair=self.cfg.write.character_repair and not small)

    @staticmethod
    def _genres(project: Project) -> list[str]:
        return [str(x) for x in (project.state.genres or [])]

    def direct_next(self, project: Project, text: str) -> int:
        """The author says what should happen in the next chapter. The plans for chapters not yet written
        are dropped so they are made again around the instruction. An empty text takes the instruction back.
        Returns the number of the chapter it will apply to."""
        n = project.next_chapter_number()
        text = " ".join((text or "").split())[:1200]
        changed = text != project.state.author_direction
        project.state.author_direction = text
        if changed:
            project.state.outline = [c for c in project.state.outline if c.number < n]
            project.clear_checkpoint(n)
        project.save()
        return n

    def _prepare_selection(self, project: Project, roster: Roster, previous_texts: list[str], plan) -> None:
        """Write several, keep the best: tell the drafter how many attempts to make and what the story's
        canon is, so it can judge them without a model (see selection.py)."""
        from ..features import chapter_features
        from ..models import ChapterRecord
        from .selection import known_names

        d = self.drafter
        d.best_of = max(1, int(self.cfg.write.best_of))
        d.choices = []
        if d.best_of <= 1:
            return
        d.known_names = known_names(project, roster, previous_texts[-6:])

        def measure(text: str) -> float:
            feats = chapter_features(ChapterRecord(novel_key="draft", index=plan.number, title=plan.title, text=text))
            return float(self.critic._measured(feats)[0])

        d.measure = measure

    def sync_characters(self, project: Project) -> list[str]:
        """Create agents for every character named in the bible that has no sheet yet."""
        roster = self.roster(project)
        creator = self.character_creator()
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
            c = creator.from_concept(name, concept, role, project.bible_markdown(), roster.names(),
                                     genres=self._genres(project))
            c.created_chapter = project.state.written
            self._carry_character(c)
            roster.add(c)
            created.append(c.name)
        if created:
            roster.save()
        return created

    def create_character(self, project: Project, name: str, concept: str, role: str = "supporting",
                         seed: Optional[int] = None) -> Character:
        roster = self.roster(project)
        seed_matrix = PersonalityMatrix.random(seed) if seed is not None else None
        c = self.character_creator().from_concept(name, concept, role, project.bible_markdown(), roster.names(), seed_matrix,
                                                  genres=self._genres(project))
        c.created_chapter = project.state.written
        self._carry_character(c)
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
    def write_next(self, project: Project, fresh: bool = False, words: Optional[int] = None) -> ChapterResult:
        """`words` sets how long this chapter and the ones after it should be (kept with the story)."""
        if words:
            project.state.target_chapter_words = max(MIN_CHAPTER_WORDS, min(MAX_CHAPTER_WORDS, int(words)))
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
                           summary=project.state.running_summary, genres=project.state.genres,
                           extra={"graph": roster.graph_text() if roster.characters else "", "triangles": roster.triangles() if roster.characters else []})
        bandit = self.bandit(project)
        settings = bandit.sample(n) if bandit is not None else {}
        prev_settings = self._apply_settings(settings) if settings else None
        if settings and self.exemplars and self.cfg.write.world_enabled and world_state is not None:
            pass

        # Resume from a checkpointed draft if the last run died after generating it.
        ckpt = None if fresh else project.load_checkpoint(n)
        resumed = False
        first_draft = ""
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
            world_level = 0.0
            if world_state is not None:
                t0 = time.monotonic()
                wb = self._world_brief(project, plan, world_state, briefs)
                world_text, world_level = wb.render(), wb.intervention_level
                timings["world"] = time.monotonic() - t0
            t0 = time.monotonic()
            improv_text = self._improv(project, plan, roster, cast_names, world_state, world_level)
            if improv_text:
                timings["improv"] = time.monotonic() - t0
                cast_text = (cast_text + "\n\n" + improv_text).strip()
            avoid_text, _prev_texts = self._avoid_list(project, n)
            if self.cfg.write.grounding and self.scout is not None:
                try:
                    from .grounding import build_dossier
                    setting = world_state.current_location if world_state else ""
                    dossier = build_dossier(self.scout, plan.plan_text if hasattr(plan, "plan_text") else plan.title, setting)
                    if dossier.notes:
                        world_text = (world_text + "\n\n" + dossier.render()).strip()
                except Exception as e:
                    log.warning("grounding dossier failed for ch %d: %s", n, e)
            ctx.cast_text, ctx.world_text = cast_text, world_text
            t0 = time.monotonic()
            council_text = council.brief(ctx) if council.agents else ""
            asked = project.state.author_direction.strip()
            if asked:
                council_text = (f"THE AUTHOR ASKED FOR THIS IN THIS CHAPTER (do it; it outranks the plan and every other note): "
                                f"{asked}\n{council_text}").strip()
            timings["council"] = time.monotonic() - t0
            self._prepare_selection(project, roster, previous_texts, plan)
            t0 = time.monotonic()
            text = self.drafter.draft(project, plan, prev, words, cast=cast_text, world=world_text,
                                      techniques=self._technique_cards(plan), avoid=avoid_text, council=council_text)
            if self.cfg.write.reach_length:
                text, carried = self.drafter.reach_length(text, words, project, plan)
                if carried:
                    log.info("ch %d: carried on %d time(s) to reach %d words (now %d)", n, carried, words, len(text.split()))
            timings["draft"] = time.monotonic() - t0
            first_draft = text
            revisions = 0
            if self.cfg.write.checkpoint_drafts:
                project.save_checkpoint(n, "draft", text, {"plan_title": plan.title, "cast": cast_names, "revisions": 0})

        t0 = time.monotonic()
        crit, reviews, wrev, nov, agent_revs = self._critique(text, plan, n, roster, cast_names, world_state, previous_texts,
                                                              council=council, ctx=ctx)
        timings["critique"] = time.monotonic() - t0
        first_score = crit.score
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
            WorldAgent(self.fast_llm, world_state, self.cfg.write.world_agency, realism=self.cfg.write.world_realism).remember(best_text, n)
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
        if self.cfg.write.ab_titles:
            try:
                from .variants import VariantLog, best_title

                chosen, ranked = best_title(self.llm, plan.plan_text if hasattr(plan, "plan_text") else plan.title, best_text, n=5)
                if chosen and ranked:
                    VariantLog(project.dir).record("title", n, chosen, ranked[1:], ranked[0].score)
                    log.info("A/B title for ch %d: %r (from %d candidates)", n, chosen, len(ranked))
                    plan.title = chosen
            except Exception as e:
                log.warning("A/B titling failed for ch %d: %s", n, e)

        if self.cfg.write.reach_length and len(best_text.split()) < words * 0.9:
            # A revision can come back shorter than what it revised; the chapter is carried on once more.
            best_text, _ = self.drafter.reach_length(best_text, words, project, plan, passes=2)
        meta = {
            "title": plan.title, "score": round(best.score, 3), "measured": round(best.measured, 3),
            "judged": round(best.judged, 3), "rubric": best.rubric, "revisions": revisions,
            "words": len(best_text.split()), "words_asked": words, "written_at": utcnow().isoformat(),
            "best_of": self.drafter.best_of, "choices": self.drafter.choices[-12:],
            "cast": cast_names, "character_consistency": round(self._consistency(best_reviews), 3),
            "world_consistency": round(best_wrev.consistency, 3) if best_wrev else None,
            "novelty": round(best_nov.score, 3) if best_nov else None,
            "direction": project.state.current_direction[:120],
            "author_direction": project.state.author_direction[:400],
            "agents": AgentCouncil.summary(best_agents),
            "content_flags": next((r.data.get("flags", []) for r in best_agents if r.agent == "sensitivity"), []),
            "resumed": resumed, "seconds": round(time.monotonic() - t_all, 1),
        }
        path = project.write_chapter(n, plan.title, best_text, meta)
        project.clear_checkpoint(n)
        project.state.author_direction = ""  # it was for this chapter; the next one starts clean
        project.save()
        self.db.record_generated(project.name, n, best.score, best.features)
        # Learning from this chapter: what the room complained about, a preference pair if revising
        # helped, a copy of what was accepted (so your later edits can be learned from), and the
        # bandit's immediate reward (reader-panel click-through when available, else the score).
        notes = [(r.agent, i) for r in best_agents for i in r.issues[:2]] + [(f"character:{r.name}", i.problem) for r in best_reviews for i in r.issues[:1]]
        if best_wrev:
            notes += [("world", v["rule"]) for v in best_wrev.violations[:2]]
        if best_nov:
            notes += [("novelty", x) for x in best_nov.notes[:2]]
        if notes:
            self.db.add_notes(project.name, n, notes)
        if self.cfg.learn.self_preference_pairs and first_draft and revisions and best_text != first_draft:
            record_self_pair(self.cfg.models_dir / "datasets", project.name, n, f"Chapter {n} of {project.name}: {plan.goal}",
                             first_draft, best_text, first_score, best.score)
        save_accepted_copy(project, n, best_text)
        if bandit is not None:
            panel_rev = next((r for r in best_agents if r.agent == "reader_panel"), None)
            reward = panel_rev.score if panel_rev is not None else best.score
            bandit.update(n, reward, weight=1.0, kind="immediate")
            bandit.save()
            meta_settings = settings
        else:
            meta_settings = {}
        if prev_settings is not None:
            self._restore_settings(prev_settings)
        if meta_settings:
            # record the settings used in the chapter's front matter for later analysis
            try:
                front = path.read_text(encoding="utf-8")
                path.write_text(front.replace("---\n\n# Chapter", f"settings: {json.dumps(meta_settings)}\n---\n\n# Chapter", 1), encoding="utf-8")
            except OSError:
                pass
        timings["total"] = time.monotonic() - t_all
        self._ctx = {"project": project.name, "chapter": None}
        return ChapterResult(number=n, title=plan.title, path=path, score=best.score, revisions=revisions, critique=best,
                             cast=cast_names, character_reviews=best_reviews, world_review=best_wrev, novelty=best_nov,
                             agent_reviews=best_agents, timings=timings,
                             llm_calls=len(self.calls.records) - calls_before,
                             llm_seconds=round(self.calls.total_seconds() - secs_before, 1), resumed=resumed,
                             words=len(best_text.split()), words_asked=words)

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
            wrev = WorldAgent(self.fast_llm, world_state, self.cfg.write.world_agency, realism=self.cfg.write.world_realism).review(text, n)
            w = self.cfg.write.world_weight
            crit.score = (1 - w) * crit.score + w * wrev.consistency
        nov = None
        if previous_texts and self.cfg.write.novelty_weight > 0:
            nov = self.novelty.report(text, previous_texts)
            w = self.cfg.write.novelty_weight
            crit.score = (1 - w) * crit.score + w * nov.score
        pm = self._project_model_cache.get(id(ctx.project)) if ctx is not None else None
        if ctx is not None and pm is None:
            pm = self.project_model(ctx.project)
            self._project_model_cache[id(ctx.project)] = pm
        if pm is not None and pm.model is not None:
            own = pm.score01(crit.features)
            if own is not None:
                crit.score = 0.6 * crit.score + 0.4 * own  # your readers outrank the corpus
                crit.measured_issues.append(f"project model (your readers): {own:.2f}")
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
        out = []
        for i in range(count):
            try:
                out.append(self.write_next(project, fresh=fresh and i == 0))
            except Exception as e:
                import traceback

                try:
                    self.db.add_journal("write", f"{type(e).__name__}: {e}", traceback.format_exc())
                except Exception:
                    pass
                raise
        return out

    def status(self, project: Project) -> dict[str, object]:
        gens = self.db.generated(project.name)
        return {
            "project": project.name, "written": project.state.written, "outlined": len(project.state.outline),
            "avg_score": round(sum(g["score"] for g in gens) / len(gens), 3) if gens else None,
            "dir": str(project.dir), "open_threads": project.state.open_threads,
        }
