"""Critic: combines the learned engagement model (measured) with an LLM rubric (judged), behind a
measured coherence floor that scores unreadable text 0 before either is consulted."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..features import chapter_features
from ..learn.engagement import EngagementModel
from ..llm.client import LLM, generate_json
from ..models import ChapterRecord
from . import prompts
from .project import ChapterPlan
from .prose import COHERENCE_FLOOR, coherence
from .schemas import CriticOut, schema_of

log = logging.getLogger(__name__)


@dataclass
class Critique:
    score: float  # 0..1 blended
    measured: float  # 0..1 from the engagement model (or heuristic fallback)
    judged: float  # 0..1 from the LLM rubric
    rubric: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    measured_issues: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    retention: float = 0.0  # 0..1 retention-curve shape (inert unless retention_weight > 0)
    coherence: float = 1.0  # 0..1 measured: is this prose at all (see prose.coherence)
    readable: bool = True  # False = under the coherence floor: score is 0 and nothing may raise it


# Sensible defaults when no model has been trained yet: what a chapter should look like.
_HEURISTIC_TARGETS = {
    "hook": (0.45, 1.0, "the opening lacks an immediate question or action; start in motion"),
    "cliff": (0.45, 1.0, "the ending resolves too much; cut the last paragraph to one short unresolved line"),
    "dialogue_ratio": (0.20, 0.65, "dialogue share is off; web serials run 25-55% dialogue"),
    "one_line_par_ratio": (0.25, 1.0, "paragraphs are too dense; break into more one-line paragraphs"),
    "sent_len_mean": (8.0, 18.0, "average sentence length is outside the 9-17 word sweet spot"),
    "tension_rising": (0.0, 1e9, "tension falls across the chapter; move the biggest beat later"),
}


class Critic:
    def __init__(self, llm: LLM, model: Optional[EngagementModel], novel_level_feats: Optional[dict[str, float]] = None,
                 rebel: bool = False, retention_weight: float = 0.0, realism_weight: float = 0.0,
                 coherence_floor: float = COHERENCE_FLOOR):
        self.llm = llm
        self.model = model
        self.novel_feats = novel_level_feats or {}
        self.rebel = rebel
        self.retention_weight = max(0.0, min(0.5, retention_weight))
        self.realism_weight = max(0.0, min(0.5, realism_weight))
        self.coherence_floor = max(0.0, min(1.0, coherence_floor))  # 0 = measure but never veto

    def _measured(self, feats: dict[str, float]) -> (float, list[str]):  # type: ignore[valid-type]
        issues: list[str] = []
        for k, (lo, hi, msg) in _HEURISTIC_TARGETS.items():
            v = feats.get(k, 0.0)
            if v < lo or v > hi:
                issues.append(f"{msg} (measured {k}={v:.2f})")
        if self.model is None:
            # No model: score = share of heuristic targets met, softened.
            met = len(_HEURISTIC_TARGETS) - len(issues)
            return 0.3 + 0.7 * met / len(_HEURISTIC_TARGETS), issues
        # Project the chapter into novel-level feature space by treating it as the novel's mean
        # chapter, on top of whatever novel-level context we have.
        novel_like = dict(self.novel_feats)
        for k, v in feats.items():
            novel_like[f"mean_{k}"] = v
        novel_like.setdefault("early_hook", feats.get("hook", 0.0))
        novel_like.setdefault("early_cliff", feats.get("cliff", 0.0))
        s = self.model.score01(novel_like)
        # Add model-driven hints: the biggest negative contributions the writer can act on.
        for name, c in self.model.contributions(novel_like)[:12]:
            if c < 0 and name.startswith("mean_") and name[5:] in feats:
                issues.append(f"{name[5:]} is pulling engagement down ({c:+.2f})")
        return s, issues[:6]

    def _judged(self, text: str, plan: ChapterPlan) -> (float, dict[str, float], str):  # type: ignore[valid-type]
        prompt = prompts.CRITIC_PROMPT.format(hook=plan.hook, turn=plan.turn, cliffhanger=plan.cliffhanger, text=text)
        if self.rebel:
            prompt = prompts.REBEL_CRITIC_LINE + "\n\n" + prompt
        try:
            data = generate_json(self.llm, prompt, schema=schema_of(CriticOut), temperature=0.2, max_tokens=400,
                                 purpose="critic")
        except Exception as e:
            log.warning("critic LLM failed: %s", e)
            return 0.6, {}, ""
        dims = ["hook", "momentum", "payoff", "cliffhanger", "clarity"]
        weights = {"hook": 0.25, "momentum": 0.2, "payoff": 0.15, "cliffhanger": 0.3, "clarity": 0.1}
        rubric = {}
        for d in dims:
            try:
                rubric[d] = float(data.get(d, 5))
            except (TypeError, ValueError):
                rubric[d] = 5.0
        judged = sum(weights[d] * (rubric[d] / 10.0) for d in dims)
        return judged, rubric, str(data.get("notes", ""))

    def critique(self, text: str, plan: ChapterPlan, title: str = "") -> Critique:
        # The floor comes first: text that is not prose is scored 0 on the measurement alone. No rubric
        # call is spent on it, and no judge (a weak one says "fine" to anything) gets a vote.
        co = coherence(text, self.coherence_floor)
        if co.score < self.coherence_floor:
            return Critique(score=0.0, measured=0.0, judged=0.0, measured_issues=co.issues[:4],
                            coherence=co.score, readable=False)
        feats = chapter_features(ChapterRecord(novel_key="draft", index=plan.number, title=title or plan.title, text=text))
        measured, issues = self._measured(feats)
        issues = co.issues + issues  # readable, but with debris the reviser should clean up
        judged, rubric, notes = self._judged(text, plan)
        retention = 0.0
        base = 0.5 * measured + 0.5 * judged
        parts = [(base, 1.0)]
        if self.retention_weight > 0:
            from ..learn.retention import shape
            rs = shape(text)
            retention = rs.score
            issues = (issues + rs.issues)[:8]
            parts = [(base, 1.0 - self.retention_weight), (retention, self.retention_weight)]
        realism = 0.0
        if self.realism_weight > 0:
            from .realism import sensory_profile, sensory_score
            sp = sensory_profile(text)
            realism = sensory_score(sp)
            issues = (issues + sp.issues)[:10]
            base_w = parts[0][1] - self.realism_weight
            parts = [(p, w) for p, w in parts[1:]] + [(base, max(0.0, base_w)), (realism, self.realism_weight)]
        total_w = sum(w for _, w in parts) or 1.0
        score = sum(p * w for p, w in parts) / total_w
        return Critique(score=round(score, 4), measured=measured, judged=judged, rubric=rubric, notes=notes,
                        measured_issues=issues, features=feats, retention=round(retention, 3), coherence=co.score)

    def problems(self, c: Critique) -> list[str]:
        out = []
        if c.notes:
            out.append(c.notes)
        for d, v in sorted(c.rubric.items(), key=lambda kv: kv[1])[:2]:
            if v < 7:
                out.append(f"{d} scored {v:.0f}/10; raise it")
        return out
