"""The World as an agent.

Characters decide what they want; the **World** decides what is true around them. It owns the
physics and the rules of the setting (gravity, magic costs, how the city's law works), the
clock and the weather, the places, the ongoing events nobody in the cast controls, and the
consequences that are still rolling from earlier chapters. It is in every scene.

How much it *does* in a scene depends on what is happening: a quiet talk gets ambience and a
passerby; an escalation gets the weather turning, a crowd reacting, an institution noticing.
Its hands are the nameless people: a vendor who refuses to serve, a child who points, a guard
who looks away. Extras who keep showing up can be promoted to full characters.
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .. import lexicon
from ..llm.client import LLM, generate_json
from ..text import words
from . import prompts
from .schemas import WorldBriefOut, WorldBuildOut, WorldMemoryOut, WorldReviewOut, schema_of

log = logging.getLogger(__name__)


class BackgroundFigure(BaseModel):
    tag: str  # "the one-armed noodle vendor"
    trait: str = ""
    first_seen: int = 0
    last_seen: int = 0
    appearances: int = 0
    actions: list[str] = Field(default_factory=list)


class WorldState(BaseModel):
    name: str = ""
    physics_rules: list[str] = Field(default_factory=list)  # what cannot happen, what always costs
    power_rules: list[str] = Field(default_factory=list)  # magic / tech / system rules
    social_rules: list[str] = Field(default_factory=list)  # law, money, institutions, taboos
    time: str = ""  # "night of day 3"
    weather: str = ""
    season: str = ""
    current_location: str = ""
    locations: dict[str, str] = Field(default_factory=dict)  # name -> one line of texture
    active_events: list[str] = Field(default_factory=list)  # things happening in the world regardless of the cast
    consequences: list[str] = Field(default_factory=list)  # unpaid bills from earlier chapters
    background_figures: list[BackgroundFigure] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)  # {"chapter": n, "events": [...]}
    last_chapter: int = 0

    def figure(self, tag: str) -> Optional[BackgroundFigure]:
        t = tag.lower().strip()
        for f in self.background_figures:
            if f.tag.lower().strip() == t:
                return f
        return None

    def promotable(self, min_appearances: int = 3) -> list[BackgroundFigure]:
        return [f for f in self.background_figures if f.appearances >= min_appearances]

    def describe(self) -> str:
        parts = [f"World: {self.name or '(unnamed)'}"]
        if self.physics_rules:
            parts.append("Physics: " + "; ".join(self.physics_rules))
        if self.power_rules:
            parts.append("Power rules: " + "; ".join(self.power_rules))
        if self.social_rules:
            parts.append("Society: " + "; ".join(self.social_rules))
        parts.append(f"Now: {self.time or 'unknown time'}, {self.weather or 'weather unknown'}"
                     + (f", {self.season}" if self.season else "") + f". Location: {self.current_location or 'unknown'}")
        if self.locations:
            parts.append("Places: " + "; ".join(f"{k}: {v}" for k, v in list(self.locations.items())[:10]))
        if self.active_events:
            parts.append("Happening in the world: " + "; ".join(self.active_events[:6]))
        if self.consequences:
            parts.append("Consequences still owed: " + "; ".join(self.consequences[:6]))
        if self.background_figures:
            parts.append("Recurring extras: " + "; ".join(f"{f.tag} ({f.trait}, seen {f.appearances}x)"
                                                         for f in self.background_figures[-8:]))
        recent = self.history[-3:]
        if recent:
            parts.append("Recently: " + " | ".join(f"ch {h['chapter']}: {'; '.join(h.get('events', [])[:3])}" for h in recent))
        return "\n".join(parts)

    def markdown(self) -> str:
        lines = [f"# World: {self.name}", "", "## Rules", "**Physics**"] + [f"- {r}" for r in self.physics_rules]
        lines += ["", "**Power**"] + [f"- {r}" for r in self.power_rules] + ["", "**Society**"] + [f"- {r}" for r in self.social_rules]
        lines += ["", "## Now", f"- time: {self.time}", f"- weather: {self.weather}", f"- season: {self.season}",
                  f"- location: {self.current_location}", "", "## Places"] + [f"- **{k}**: {v}" for k, v in self.locations.items()]
        lines += ["", "## Active events"] + [f"- {e}" for e in self.active_events]
        lines += ["", "## Consequences owed"] + [f"- {c}" for c in self.consequences]
        lines += ["", "## Background figures"] + [f"- **{f.tag}** ({f.trait}) seen {f.appearances}x, last ch {f.last_seen}: "
                                                  + "; ".join(f.actions[-3:]) for f in self.background_figures]
        lines += ["", "## History"] + [f"- ch {h['chapter']}: {'; '.join(h.get('events', []))}" for h in self.history[-20:]]
        return "\n".join(lines) + "\n"


class WorldBrief(BaseModel):
    setting: str = ""
    constraints: list[str] = Field(default_factory=list)
    intervention_level: float = 0.3
    interventions: list[str] = Field(default_factory=list)
    background_figures: list[dict[str, str]] = Field(default_factory=list)  # {tag, action, line}
    consequences_surfacing: list[str] = Field(default_factory=list)
    continuity: list[str] = Field(default_factory=list)

    def render(self) -> str:
        out = ["### The World", f"- setting: {self.setting}"]
        if self.constraints:
            out.append("- rules in play: " + "; ".join(self.constraints))
        lvl = "ambient" if self.intervention_level < 0.35 else ("active" if self.intervention_level < 0.7 else "forceful")
        out.append(f"- the world's presence this scene: {lvl} ({self.intervention_level:.1f})")
        if self.interventions:
            out.append("- the world does: " + "; ".join(self.interventions))
        for f in self.background_figures:
            line = f" says \"{f['line']}\"" if f.get("line") else ""
            out.append(f"- background: {f.get('tag', 'someone')} - {f.get('action', '')}{line}")
        if self.consequences_surfacing:
            out.append("- consequences surfacing: " + "; ".join(self.consequences_surfacing))
        if self.continuity:
            out.append("- must stay true: " + "; ".join(self.continuity))
        return "\n".join(out)


class WorldReview(BaseModel):
    consistency: float = 1.0
    violations: list[dict[str, str]] = Field(default_factory=list)  # {quote, rule, fix}


def intervention_level(plan_text: str, cast_conflicts: int, agency: float, chapter: int) -> float:
    """How hard the world should push this scene. Deterministic for a given chapter.

    Tension in the plan raises it, friction between characters and plan raises it, a little
    seeded variety keeps it from being the same every chapter, and ``agency`` scales it all."""
    ws = [w.lower() for w in words(plan_text)]
    tension = sum(1 for w in ws if w in lexicon.TENSION) / max(1, len(ws))
    base = 0.2 + min(0.5, tension * 6.0) + min(0.2, 0.1 * cast_conflicts)
    jitter = random.Random(chapter * 7919).uniform(-0.1, 0.15)
    return round(max(0.05, min(1.0, (base + jitter) * (0.5 + agency))), 2)


class WorldAgent:
    def __init__(self, llm: LLM, state: WorldState, agency: float = 0.6, realism: bool = False):
        self.llm = llm
        self.w = state
        self.agency = agency
        self.realism = realism  # ground physics/geography/procedure in the real world rather than inventing

    _REALISM_NOTE = ("\nREALISM MODE: this is a real-world setting. Do not invent physics or magic. Ground geography, "
                     "weather, institutions and how things work in the actual world; where a specific real detail is "
                     "needed and unknown, note it as needing research rather than making it up.")

    def system_prompt(self) -> str:
        return prompts.WORLD_SYSTEM.format(world=self.w.describe()) + (self._REALISM_NOTE if self.realism else "")

    # ---- creation ----------------------------------------------------------
    def build(self, bible_md: str, premise: str = "") -> WorldState:
        prompt = prompts.WORLD_BUILD_PROMPT.format(bible=bible_md, premise=premise or "(none beyond the bible)")
        if self.realism:
            prompt += self._REALISM_NOTE
        try:
            d = generate_json(self.llm, prompt, schema=schema_of(WorldBuildOut), temperature=0.8, max_tokens=1400,
                              purpose="world:build")
        except Exception as e:
            log.warning("world build failed: %s", e)
            d = {}
        self.w.name = str(d.get("name") or self.w.name or "the world")
        for k in ("physics_rules", "power_rules", "social_rules", "active_events"):
            vals = d.get(k)
            if isinstance(vals, list):
                setattr(self.w, k, [str(v) for v in vals][:8])
        if isinstance(d.get("locations"), dict):
            self.w.locations = {str(k): str(v) for k, v in list(d["locations"].items())[:12]}
        for k in ("time", "weather", "season", "current_location"):
            if d.get(k):
                setattr(self.w, k, str(d[k]))
        return self.w

    # ---- per scene ---------------------------------------------------------
    def brief(self, scene: dict[str, Any], cast_summary: str = "", conflicts: int = 0) -> WorldBrief:
        plan_text = " ".join(str(scene.get(k, "")) for k in ("title", "goal", "hook", "turn", "cliffhanger")) + " ".join(scene.get("beats", []))
        level = intervention_level(plan_text, conflicts, self.agency, int(scene.get("number", 0)))
        prompt = prompts.WORLD_BRIEF_PROMPT.format(
            number=scene.get("number", "?"), title=scene.get("title", ""), goal=scene.get("goal", ""),
            hook=scene.get("hook", ""), turn=scene.get("turn", ""), cliffhanger=scene.get("cliffhanger", ""),
            beats="; ".join(scene.get("beats", [])), cast=cast_summary or "(no profiled characters)", level=level,
            mode=("stay in the background: texture, weather, one passerby at most" if level < 0.35 else
                  "take part: let the setting, a crowd or an institution act on the scene" if level < 0.7 else
                  "push hard: the world itself turns against or around them; consequences land now"),
        )
        try:
            d = generate_json(self.llm, prompt, system=self.system_prompt(), schema=schema_of(WorldBriefOut),
                              temperature=0.8, max_tokens=700, purpose="world:brief")
        except Exception as e:
            log.warning("world brief failed: %s", e)
            d = {}
        figs = []
        for f in d.get("background_figures", []) if isinstance(d.get("background_figures"), list) else []:
            if isinstance(f, dict) and f.get("tag"):
                figs.append({"tag": str(f["tag"])[:60], "action": str(f.get("action", ""))[:160], "line": str(f.get("line", ""))[:120]})
        # Below the ambient threshold the world keeps its hands off: at most one extra, no interventions.
        if level < 0.35:
            figs = figs[:1]
        return WorldBrief(
            setting=str(d.get("setting", "") or f"{self.w.current_location}, {self.w.time}, {self.w.weather}"),
            constraints=[str(c) for c in d.get("constraints", [])][:5] if isinstance(d.get("constraints"), list) else [],
            intervention_level=level,
            interventions=([str(i) for i in d.get("interventions", [])][:3] if isinstance(d.get("interventions"), list) and level >= 0.35 else []),
            background_figures=figs[:3],
            consequences_surfacing=[str(c) for c in d.get("consequences_surfacing", [])][:3] if isinstance(d.get("consequences_surfacing"), list) else [],
            continuity=[str(c) for c in d.get("continuity", [])][:5] if isinstance(d.get("continuity"), list) else [],
        )

    def review(self, chapter_text: str, number: int) -> WorldReview:
        prompt = prompts.WORLD_REVIEW_PROMPT.format(number=number, text=chapter_text)
        try:
            d = generate_json(self.llm, prompt, system=self.system_prompt(), schema=schema_of(WorldReviewOut),
                              temperature=0.2, max_tokens=600, purpose="world:review")
        except Exception as e:
            log.warning("world review failed: %s", e)
            return WorldReview()
        viols = [{"quote": str(v.get("quote", ""))[:200], "rule": str(v.get("rule", ""))[:200], "fix": str(v.get("fix", ""))[:200]}
                 for v in d.get("violations", []) if isinstance(v, dict)] if isinstance(d.get("violations"), list) else []
        try:
            score = max(0.0, min(10.0, float(d.get("consistency", 10)))) / 10.0
        except (TypeError, ValueError):
            score = 1.0
        return WorldReview(consistency=score, violations=viols[:5])

    def remember(self, chapter_text: str, number: int) -> None:
        prompt = prompts.WORLD_MEMORY_PROMPT.format(number=number, text=chapter_text)
        try:
            d = generate_json(self.llm, prompt, system=self.system_prompt(), schema=schema_of(WorldMemoryOut),
                              temperature=0.2, max_tokens=700, purpose="world:memory")
        except Exception as e:
            log.warning("world memory failed: %s", e)
            d = {"events": [" ".join(chapter_text.split()[-40:])]}
        w = self.w
        w.history.append({"chapter": number, "events": [str(e) for e in d.get("events", [])][:6] if isinstance(d.get("events"), list) else []})
        w.history = w.history[-60:]
        for k in ("time", "weather", "season", "current_location"):
            if d.get(k):
                setattr(w, k, str(d[k]))
        if isinstance(d.get("new_locations"), dict):
            for k, v in list(d["new_locations"].items())[:4]:
                w.locations.setdefault(str(k), str(v))
        for ev in d.get("new_events", []) if isinstance(d.get("new_events"), list) else []:
            if str(ev) not in w.active_events:
                w.active_events.append(str(ev))
        for ev in d.get("ended_events", []) if isinstance(d.get("ended_events"), list) else []:
            w.active_events = [e for e in w.active_events if str(ev).lower() not in e.lower()]
        w.active_events = w.active_events[-10:]
        for c in d.get("new_consequences", []) if isinstance(d.get("new_consequences"), list) else []:
            if str(c) not in w.consequences:
                w.consequences.append(str(c))
        for c in d.get("paid_consequences", []) if isinstance(d.get("paid_consequences"), list) else []:
            w.consequences = [x for x in w.consequences if str(c).lower() not in x.lower()]
        w.consequences = w.consequences[-10:]
        for f in d.get("background_figures", []) if isinstance(d.get("background_figures"), list) else []:
            if not isinstance(f, dict) or not f.get("tag"):
                continue
            fig = w.figure(str(f["tag"]))
            if fig is None:
                fig = BackgroundFigure(tag=str(f["tag"])[:60], trait=str(f.get("trait", ""))[:80], first_seen=number)
                w.background_figures.append(fig)
            fig.appearances += 1
            fig.last_seen = number
            if f.get("action"):
                fig.actions.append(f"ch {number}: {str(f['action'])[:120]}")
                fig.actions = fig.actions[-6:]
        w.background_figures = w.background_figures[-30:]
        w.last_chapter = number

    def chat(self, question: str) -> str:
        prompt = prompts.WORLD_CHAT_PROMPT.format(question=question)
        return self.llm.generate(prompt, system=self.system_prompt(), temperature=0.7, max_tokens=500, purpose="world:chat").strip()


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------
def world_path(project_dir: Path) -> Path:
    return project_dir / "world.json"


def load_world(project_dir: Path) -> Optional[WorldState]:
    p = world_path(project_dir)
    if not p.exists():
        return None
    try:
        return WorldState(**json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:  # hand-edited file with a typo must not kill a run
        log.warning("could not load world.json: %s", e)
        return None


def save_world(project_dir: Path, w: WorldState) -> None:
    world_path(project_dir).write_text(w.model_dump_json(indent=2), encoding="utf-8")
    (project_dir / "world.md").write_text(w.markdown(), encoding="utf-8")


def figure_slug(tag: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")
