"""Reader reaction as a learning signal: sentiment and reply velocity, not just the prose.

The strongest "did it go viral" signal is how readers reacted and how fast. This layer turns a list of
reactions on a chapter (comment text plus timestamps) into a blended reaction score and a handful of
actionable lessons. It reuses the transparent comment lexicon so it never needs a model, and it is
honest about its input: it analyses whatever reactions you feed it. Wiring a real platform's
comment feed into ``load`` is the only step between this and live learning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .comments import analyze_comment


@dataclass
class ReactionScore:
    count: int
    sentiment: float        # -1..+1 (excited/laugh minus complaint/confused)
    velocity: float         # reactions per hour in the first window
    score: float            # 0..1 blended
    breakdown: dict[str, float] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)


def _parse_time(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    if isinstance(v, (int, float)):
        try:
            return datetime.fromtimestamp(float(v))
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def score_reactions(reactions: list[dict[str, Any]], window_hours: float = 24.0) -> ReactionScore:
    """`reactions`: list of {text, at} where `at` is an ISO string, epoch, or datetime (optional)."""
    n = len(reactions)
    if n == 0:
        return ReactionScore(0, 0.0, 0.0, 0.0, {}, ["no reactions yet to learn from"])
    pos = neg = laugh = confused = complaint = 0
    for r in reactions:
        a = analyze_comment(str(r.get("text", "")))
        pos += 1 if a["excited"] else 0
        laugh += 1 if a["laugh"] else 0
        confused += 1 if a["confused"] else 0
        complaint += 1 if a["complaint"] else 0
    neg = confused + complaint
    sentiment = (pos + laugh - neg) / n
    # Velocity: reactions landing within the first window after the earliest timestamp.
    times = [t for t in (_parse_time(r.get("at")) for r in reactions) if t is not None]
    velocity = 0.0
    if len(times) >= 2:
        t0 = min(times)
        in_window = [t for t in times if (t - t0).total_seconds() <= window_hours * 3600]
        velocity = len(in_window) / window_hours
    # Blend: sentiment mapped to 0..1, plus a saturating velocity term, plus engagement volume.
    sent01 = (sentiment + 1) / 2
    vel01 = min(1.0, velocity / 5.0)  # 5 reactions/hour saturates the term
    vol01 = min(1.0, n / 50.0)
    score = round(0.5 * sent01 + 0.3 * vel01 + 0.2 * vol01, 3)
    breakdown = {"excited": pos / n, "laugh": laugh / n, "confused": confused / n, "complaint": complaint / n,
                 "velocity_per_hour": round(velocity, 2)}
    lessons: list[str] = []
    if complaint / n > 0.25:
        lessons.append(f"{complaint}/{n} reactions complained (slow/filler/OOC); tighten the pacing and stay in character")
    if confused / n > 0.2:
        lessons.append(f"{confused}/{n} reactions were confused; clarify the chapter's key turn")
    if (pos + laugh) / n > 0.4:
        lessons.append("readers reacted with excitement; repeat the beats that landed (hook and cliff)")
    if velocity > 3:
        lessons.append("reactions came fast; this chapter hooked the binge readers — note what it did")
    return ReactionScore(n, round(sentiment, 3), round(velocity, 2), score, breakdown, lessons)
