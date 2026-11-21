"""Does Gauge's score predict real retention? The harness that keeps everything else honest.

The pipeline optimises a chapter *score*. That is only worth optimising if it tracks what real
readers do. This module measures that: it computes realized retention from the reader feedback you
have recorded (views of chapter n over chapter n-1), lines it up against the score Gauge gave each
chapter, and reports the correlation. A held-out split checks that the project model predicts
retention on chapters it was not fitted on. Without this, every other "improvement" is a guess; with
it, you can tell whether a change to the writing actually moved the needle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def realized_retention(feedback: dict[int, dict]) -> dict[int, float]:
    """views[n] / views[n-1] for each chapter that has both. Clamped to [0, 2]."""
    out: dict[int, float] = {}
    for n in sorted(feedback):
        cur = feedback.get(n, {}).get("views")
        prev = feedback.get(n - 1, {}).get("views")
        if cur and prev and prev > 0:
            out[n] = round(min(2.0, cur / prev), 4)
    return out


def pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return round(sxy / (sxx * syy) ** 0.5, 3)


@dataclass
class Evaluation:
    chapters_with_feedback: int
    retention_mean: Optional[float]
    retention_trend: Optional[float]
    score_vs_retention: Optional[float]  # correlation of Gauge's chapter score with realized retention
    panel_correlation: Optional[float]   # the reader panel's own calibration correlation
    project_model_r2cv: Optional[float]  # cross-validated R^2 of the per-project model
    worst_chapters: list[tuple[int, float]] = field(default_factory=list)  # (chapter, retention), lowest first
    notes: list[str] = field(default_factory=list)

    def verdict(self) -> str:
        c = self.score_vs_retention
        if c is None:
            return "not enough paired data yet; record more reader feedback (gauge feedback / feedback-import)"
        if c >= 0.3:
            return f"Gauge's score tracks real retention (r={c}); optimising it is justified"
        if c <= -0.1:
            return f"Gauge's score is ANTI-correlated with retention (r={c}); the scoring needs rework before trusting it"
        return f"weak link between score and retention (r={c}); treat internal scores with caution"


def evaluate_project(project: Any, feedback: dict[int, dict], panel_corr: Optional[float] = None,
                     project_model_r2cv: Optional[float] = None) -> Evaluation:
    ret = realized_retention(feedback)
    notes: list[str] = []
    pairs: list[tuple[float, float]] = []
    for n, r in ret.items():
        meta = project.read_meta(n) if hasattr(project, "read_meta") else {}
        score = meta.get("score")
        if isinstance(score, (int, float)):
            pairs.append((float(score), r))
    corr = pearson([s for s, _ in pairs], [r for _, r in pairs]) if len(pairs) >= 3 else None
    if ret:
        vals = list(ret.values())
        mean = round(sum(vals) / len(vals), 3)
        if len(vals) >= 4:
            half = len(vals) // 2
            trend = round(sum(vals[half:]) / (len(vals) - half) - sum(vals[:half]) / half, 3)
        else:
            trend = None
        worst = sorted(ret.items(), key=lambda kv: kv[1])[:5]
    else:
        mean = trend = None
        worst = []
    if len(pairs) < 3 and ret:
        notes.append("chapters have retention but no stored score to compare; re-run or backfill chapter meta")
    if not ret:
        notes.append("no realized retention yet: record views for consecutive chapters")
    return Evaluation(len(ret), mean, trend, corr, panel_corr, project_model_r2cv, worst, notes)
