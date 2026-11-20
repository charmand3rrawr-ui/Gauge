"""Shared shape of every story agent.

An agent is an independent intelligence with its own point of view, its own persistent state
in ``projects/<slug>/agents/<name>.json``, and up to four moments in the chapter loop:

* ``plan_notes(project)``   - advice to the planner before the outline is extended
* ``brief(ctx)``            - advice to the drafter before the chapter is written
* ``review(text, ctx)``     - a verdict on the draft (0..1 score + concrete issues)
* ``remember(text, ctx)``   - update its own state once the chapter is accepted

Agents never see each other's prompts; the council merges their outputs. Any of them may be
switched off in ``config.json`` under ``agents``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ...llm.client import LLM, generate_json

log = logging.getLogger(__name__)


@dataclass
class SceneContext:
    project: Any  # Project
    number: int
    plan: Any  # ChapterPlan
    prev_tail: str = ""
    summary: str = ""
    cast_text: str = ""
    world_text: str = ""
    genres: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def plan_text(self) -> str:
        p = self.plan
        return " ".join([p.title, p.goal, p.hook, p.turn, p.cliffhanger, " ".join(p.beats)])


class AgentReview(BaseModel):
    agent: str
    score: float = 1.0  # 0..1, 1 = no complaints
    issues: list[str] = Field(default_factory=list)  # concrete, actionable, go into the revision prompt
    notes: str = ""
    data: dict[str, Any] = Field(default_factory=dict)  # agent-specific numbers for front matter / reports


class StoryAgent:
    name: str = "agent"
    title: str = "Agent"
    weight: float = 1.0  # relative weight inside the council's share of the chapter score
    state_model: type[BaseModel] = BaseModel

    def __init__(self, llm: LLM, project_dir: Path, options: Optional[dict[str, Any]] = None):
        self.llm = llm
        self.dir = project_dir / "agents"
        self.path = self.dir / f"{self.name}.json"
        self.options = options or {}
        self.state = self.state_model()
        if self.path.exists():
            try:
                self.state = self.state_model(**json.loads(self.path.read_text(encoding="utf-8")))
            except Exception as e:  # a hand-edited file must never kill a run
                log.warning("%s: could not load state (%s); starting fresh", self.name, e)

    # ---- persistence -------------------------------------------------------
    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")

    # ---- the four moments (defaults: do nothing) ---------------------------
    def setup(self, project: Any) -> None:
        """Called once when the agent is first attached to a project (after the bible exists)."""

    def plan_notes(self, project: Any) -> str:
        return ""

    def brief(self, ctx: SceneContext) -> str:
        return ""

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        return None

    def remember(self, text: str, ctx: SceneContext) -> None:
        return None

    def report(self) -> str:
        """Human-readable state for `gauge agents show`."""
        return self.state.model_dump_json(indent=2)

    # ---- helpers -----------------------------------------------------------
    def marker(self, op: str) -> str:
        """First line of every prompt: lets the offline FakeLLM route, and makes logs greppable."""
        return f"[agent:{self.name}:{op}]"

    def ask_json(self, op: str, prompt: str, schema: Optional[dict[str, Any]] = None, system: str = "",
                 temperature: float = 0.3, max_tokens: int = 700) -> dict[str, Any]:
        try:
            return generate_json(self.llm, f"{self.marker(op)}\n{prompt}", system=system, schema=schema,
                                 temperature=temperature, max_tokens=max_tokens, purpose=f"{self.name}:{op}")
        except Exception as e:
            log.warning("%s.%s failed: %s", self.name, op, e)
            return {}

    def ask_text(self, op: str, prompt: str, system: str = "", temperature: float = 0.6, max_tokens: int = 500) -> str:
        try:
            return self.llm.generate(f"{self.marker(op)}\n{prompt}", system=system, temperature=temperature,
                                     max_tokens=max_tokens, purpose=f"{self.name}:{op}").strip()
        except Exception as e:
            log.warning("%s.%s failed: %s", self.name, op, e)
            return ""


def as_str_list(v: Any, limit: int = 8, maxlen: int = 300) -> list[str]:
    if isinstance(v, list):
        return [str(x)[:maxlen] for x in v if str(x).strip()][:limit]
    if isinstance(v, str) and v.strip():
        return [v.strip()[:maxlen]]
    return []


def clamp01(v: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def score10(v: Any, default: float = 7.0) -> float:
    try:
        return max(0.0, min(10.0, float(v))) / 10.0
    except (TypeError, ValueError):
        return default / 10.0
