"""The council: runs every enabled agent at each moment of the chapter loop, in parallel, and
merges their verdicts into one score and one list of problems for the reviser."""
from __future__ import annotations

import logging
from typing import Any, Optional

from ...llm.client import LLM
from ...llm.parallel import run_parallel
from .arc_keeper import ArcKeeper
from .archivist import Archivist
from .base import AgentReview, SceneContext, StoryAgent
from .comedian import Comedian, wants_comedy
from .continuity_auditor import ContinuityAuditor
from .craft_editor import CraftEditor
from .editor import LineEditor
from .librarian import Librarian
from .pacing import PacingDirector
from .packager import Packager
from .readability import ReadabilityCoach
from .readers import ReaderPanel
from .realism_editor import SensoryEditor
from .schemer import Schemer
from .scout import ScoutAgent
from .sensitivity import SensitivityReader
from .showrunner import Showrunner
from .suitability import SuitabilityChecker
from .theme import ThemeKeeper
from .voice_coach import VoiceCoach

log = logging.getLogger(__name__)

ALL_AGENTS: dict[str, type[StoryAgent]] = {
    ReaderPanel.name: ReaderPanel, Showrunner.name: Showrunner, Archivist.name: Archivist, Schemer.name: Schemer,
    LineEditor.name: LineEditor, Packager.name: Packager, ThemeKeeper.name: ThemeKeeper, SensitivityReader.name: SensitivityReader,
    PacingDirector.name: PacingDirector, Librarian.name: Librarian, Comedian.name: Comedian, ScoutAgent.name: ScoutAgent,
    VoiceCoach.name: VoiceCoach, ReadabilityCoach.name: ReadabilityCoach, SuitabilityChecker.name: SuitabilityChecker,
    ArcKeeper.name: ArcKeeper, SensoryEditor.name: SensoryEditor, ContinuityAuditor.name: ContinuityAuditor, CraftEditor.name: CraftEditor,
}
LEAN = {"archivist", "showrunner", "line_editor", "pacing", "librarian", "packager", "voice_coach", "readability", "suitability", "arc_keeper", "sensory", "continuity"}


def build_council(cfg, llm: LLM, project, embedder=None, scout=None) -> AgentCouncil:
    """Instantiate the agents the config enables for this project."""
    a = cfg.agents
    if a.profile == "off":
        return AgentCouncil([], cfg.llm.parallel, a.weight)
    enabled = {name for name in ALL_AGENTS if getattr(a, name, True)}
    if a.profile == "lean":
        enabled &= LEAN
    if "comedian" in enabled and not wants_comedy(project.state.genres, project.state.bible.tone):
        enabled.discard("comedian")
    agents: list[StoryAgent] = []
    for name in ALL_AGENTS:
        if name not in enabled:
            continue
        cls = ALL_AGENTS[name]
        opts: dict[str, Any] = {}
        if name == "sensitivity":
            opts = {"rules": a.sensitivity_rules, "block_on_flags": a.block_on_sensitivity, "categories": a.sensitivity_categories or None}
        if name == "suitability":
            opts = {"platform": a.suitability_platform, "age_band": a.suitability_age_band, "heat_max": a.suitability_heat_max,
                    "content_lines": a.suitability_content_lines, "block": a.suitability_block,
                    "word_target": cfg.write.target_chapter_words}
        if name == "scout":
            if scout is None or not cfg.research.enabled:
                continue
            opts = {"max_queries": cfg.research.max_queries_per_chapter, "fact_check": cfg.research.fact_check}
        try:
            if name == "librarian":
                agent = cls(llm, project.dir, opts, embedder=embedder)
            elif name == "scout":
                agent = cls(llm, project.dir, opts, scout=scout)
            else:
                agent = cls(llm, project.dir, opts)
        except Exception as e:  # one broken agent must not stop the book
            log.warning("could not start agent %s: %s", name, e)
            continue
        agents.append(agent)
    return AgentCouncil(agents, cfg.llm.parallel, a.weight)


class AgentCouncil:
    def __init__(self, agents: list[StoryAgent], parallel: int = 4, weight: float = 0.25):
        self.agents = agents
        self.parallel = parallel
        self.weight = weight

    def get(self, name: str) -> Optional[StoryAgent]:
        return next((a for a in self.agents if a.name == name), None)

    def names(self) -> list[str]:
        return [a.name for a in self.agents]

    def setup(self, project) -> None:
        run_parallel(lambda a: a.setup(project), self.agents, self.parallel, "agent setup")
        self.save()

    def save(self) -> None:
        for a in self.agents:
            try:
                a.save()
            except Exception as e:
                log.warning("could not save agent %s: %s", a.name, e)

    # ---- moments -----------------------------------------------------------
    def plan_notes(self, project) -> str:
        notes = run_parallel(lambda a: a.plan_notes(project), self.agents, self.parallel, "plan notes")
        return "\n".join(n for n in notes if n)

    def brief(self, ctx: SceneContext) -> str:
        outs = run_parallel(lambda a: (a.title, a.brief(ctx)), self.agents, self.parallel, "agent brief")
        sections = [f"### {title}\n{text}" for item in outs if item for title, text in [item] if text]
        return "\n\n".join(sections)

    def review(self, text: str, ctx: SceneContext) -> list[AgentReview]:
        outs = run_parallel(lambda a: a.review(text, ctx), self.agents, self.parallel, "agent review")
        return [r for r in outs if r is not None]

    def remember(self, text: str, ctx: SceneContext) -> None:
        run_parallel(lambda a: a.remember(text, ctx), self.agents, self.parallel, "agent memory")
        self.save()

    # ---- merging -----------------------------------------------------------
    def score(self, reviews: list[AgentReview]) -> Optional[float]:
        weights = {a.name: a.weight for a in self.agents}
        pairs = [(r.score, weights.get(r.agent, 1.0)) for r in reviews if weights.get(r.agent, 1.0) > 0]
        blocking = [r for r in reviews if weights.get(r.agent, 1.0) == 0 and r.score == 0.0]
        if blocking:
            return 0.0
        if not pairs:
            return None
        return sum(s * w for s, w in pairs) / sum(w for _, w in pairs)

    @staticmethod
    def problems(reviews: list[AgentReview], per_agent: int = 2) -> list[str]:
        out = []
        for r in reviews:
            for i in r.issues[:per_agent]:
                out.append(f"[{r.agent}] {i}")
        return out

    @staticmethod
    def summary(reviews: list[AgentReview]) -> dict[str, Any]:
        return {r.agent: {"score": round(r.score, 3), **{k: v for k, v in r.data.items() if not isinstance(v, (list, dict)) or k == "flags"}}
                for r in reviews}

    def retitle(self, text: str, plan, number: int) -> Optional[str]:
        p = self.get("packager")
        return p.retitle(text, plan, number) if isinstance(p, Packager) else None
