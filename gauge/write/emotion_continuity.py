"""Emotional continuity: flag mood jumps a chapter never earned.

Gauge already tracks each character's mood as a pleasure/arousal/dominance vector that decays toward
their baseline and is pushed by each chapter's emotional event. This auditor reads the recorded
series and flags two failures of continuity:

* an **unearned jump** - the mood moved a long way between appearances while the chapter's emotional
  event was small (the character suddenly feels something nothing caused), and
* **whiplash** - the mood reverses direction hard two chapters running (elated, crushed, elated),
  which reads as the writer forgetting where the character was.

It judges only the recorded data; it is cheap and deterministic, and it complements the Voice Coach
(which guards how characters sound) by guarding how they feel.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContinuityFlag:
    chapter: int
    kind: str  # "unearned_jump" | "whiplash"
    magnitude: float
    detail: str


def _dist(a: dict, b: dict) -> float:
    return sum((float(a.get(k, 0.0)) - float(b.get(k, 0.0))) ** 2 for k in ("pleasure", "arousal", "dominance")) ** 0.5


def audit(history: list[dict], jump_threshold: float = 0.8, event_floor: float = 0.3,
          whiplash_threshold: float = 0.7) -> list[ContinuityFlag]:
    """`history` is a list of {chapter, pleasure, arousal, dominance, event} in chapter order."""
    flags: list[ContinuityFlag] = []
    rows = [r for r in history if "chapter" in r]
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        moved = _dist(prev, cur)
        event = float(cur.get("event", 0.0) or 0.0)
        if moved >= jump_threshold and event < event_floor:
            flags.append(ContinuityFlag(int(cur["chapter"]), "unearned_jump", round(moved, 3),
                                        f"mood moved {moved:.2f} but the chapter's emotional event was only {event:.2f}; "
                                        f"show what caused the shift or soften it"))
        if i >= 2:
            # Reversal: the pleasure axis swung up then down (or vice versa) by a lot both times.
            p0 = float(rows[i - 2].get("pleasure", 0.0))
            p1 = float(prev.get("pleasure", 0.0))
            p2 = float(cur.get("pleasure", 0.0))
            d1, d2 = p1 - p0, p2 - p1
            if d1 * d2 < 0 and min(abs(d1), abs(d2)) >= whiplash_threshold:
                flags.append(ContinuityFlag(int(cur["chapter"]), "whiplash", round(min(abs(d1), abs(d2)), 3),
                                            f"mood whiplash across chapters {int(rows[i-2]['chapter'])}-{int(cur['chapter'])}; "
                                            f"the character swings {d1:+.2f} then {d2:+.2f} with no settling beat"))
    return flags


def summary(history: list[dict]) -> str:
    flags = audit(history)
    if not history:
        return "(no mood recorded yet)"
    head = f"{len(history)} appearances tracked"
    if not flags:
        return head + "; emotional arc is continuous"
    return head + "; " + "; ".join(f"ch {f.chapter}: {f.kind} ({f.magnitude})" for f in flags[:6])
