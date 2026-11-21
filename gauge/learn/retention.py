"""Virality as a *retention curve*, not a single number.

A chapter does not go viral because it is uniformly "good"; it goes viral because of its shape:
a hook that grabs in the first lines, a spike in the middle that stops a skimmer from drifting,
and a cliff at the end that forces the next tap. This module scores those three positions
separately so the bandit and critic can optimise the shape, and it measures real chapter-to-
chapter retention across a serial from whatever follower/view deltas the database has.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..features import cliffhanger_score, hook_score, tension_curve
from ..text import head, tail, word_count


@dataclass
class RetentionShape:
    opening: float  # 0..1 strength of the hook
    middle: float   # 0..1 mid-chapter spike (keeps skimmers)
    ending: float   # 0..1 cliff pull
    sag: float      # 0..1 how deep the mid-chapter low is (lower is better); the "boring middle"
    score: float    # 0..1 blended shape quality
    issues: list[str] = field(default_factory=list)


def shape(text: str) -> RetentionShape:
    if word_count(text) < 60:
        return RetentionShape(0.0, 0.0, 0.0, 0.0, 0.0, ["too short to shape"])
    opening = float(hook_score(head(text, 200)).get("hook", 0.0))
    ending = float(cliffhanger_score(tail(text, 200)).get("cliff", 0.0))
    curve = tension_curve(text, 10)
    if curve:
        peak = max(curve)
        mid = curve[len(curve) // 2]
        lo = min(curve[1:-1]) if len(curve) > 2 else min(curve)
        norm = max(peak, 1e-6)
        middle = min(1.0, mid / norm)
        sag = max(0.0, 1.0 - lo / norm)  # 1 = the dullest segment is near-zero tension
    else:
        middle = sag = 0.0
    issues: list[str] = []
    if opening < 0.45:
        issues.append("weak opening hook; start in motion or on a question")
    if middle < 0.4:
        issues.append("the middle sags; land a reversal or revelation around the midpoint")
    if sag > 0.8:
        issues.append("there is a dead stretch in the body; cut or compress the flattest scene")
    if ending < 0.45:
        issues.append("the ending resolves too much; leave one thread pulling into the next chapter")
    # Reward a strong open and close, a lively middle, and penalise a deep sag.
    score = 0.3 * opening + 0.25 * middle + 0.3 * ending + 0.15 * (1.0 - sag)
    return RetentionShape(round(opening, 3), round(middle, 3), round(ending, 3), round(sag, 3), round(score, 3), issues)


def serial_retention(deltas: list[float]) -> dict[str, float]:
    """Given per-chapter reader-count deltas (new followers/views for each chapter in order),
    estimate how well the serial retains: the fraction of chapters that grew, and the trend.
    """
    if not deltas:
        return {"chapters": 0.0, "growth_fraction": 0.0, "trend": 0.0, "mean_delta": 0.0}
    grew = sum(1 for d in deltas if d > 0) / len(deltas)
    mean = sum(deltas) / len(deltas)
    if len(deltas) >= 2:
        first = sum(deltas[: len(deltas) // 2]) / max(1, len(deltas) // 2)
        second = sum(deltas[len(deltas) // 2:]) / max(1, len(deltas) - len(deltas) // 2)
        trend = second - first
    else:
        trend = 0.0
    return {"chapters": float(len(deltas)), "growth_fraction": round(grew, 3),
            "trend": round(trend, 3), "mean_delta": round(mean, 3)}
