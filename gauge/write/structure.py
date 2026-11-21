"""Structure checks (roadmap section D): try-fail cost, conflict, cliff variety, beat placement.

Measured, no model calls. These catch the structural failure modes that make a serial sag — the hero
who wins without paying, the chapter with no conflict, the same cliffhanger three times running, the
climax energy spent at the midpoint — and give the planner and reviser something to act on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import lexicon
from ..text import segments, sentences, tail, word_count

# Small lexicons for the try-fail check.
SUCCESS = {"won", "succeeded", "escaped", "saved", "defeated", "solved", "found", "achieved", "triumphed",
           "rescued", "survived", "beat", "outwitted", "secured", "reached", "earned", "claimed"}
SETBACK = {"failed", "lost", "trapped", "caught", "wounded", "cornered", "betrayed", "died", "broke",
           "surrendered", "retreated", "cost", "sacrificed", "too late", "couldn't", "wouldn't", "denied"}
DECISION = {"decided", "chose", "refused", "vowed", "swore", "resolved", "would not", "had to choose"}
ARRIVAL = {"arrived", "stepped into", "the door opened", "reached the", "entered", "appeared", "crossed the threshold"}


def _density(text: str, lex) -> float:
    ws = [w.lower() for w in text.split()]
    n = max(1, len(ws))
    low = text.lower()
    hits = sum(1 for w in ws if w.strip(".,;:!?\"'") in lex) + sum(low.count(p) for p in lex if " " in p)
    return round(hits / n * 1000, 2)  # per 1000 words


def try_fail(text: str) -> dict[str, float]:
    """Success vs cost in a chapter. too_easy = success with no visible setback."""
    s = _density(text, SUCCESS)
    f = _density(text, SETBACK)
    return {"success": s, "setback": f, "too_easy": 1.0 if s >= 3 and f < 0.5 else 0.0}


def conflict_level(text: str) -> float:
    """Conflict/tension density per 1000 words (reuses the tension lexicon)."""
    return _density(text, lexicon.TENSION)


def cliff_type(text: str) -> str:
    """Classify a chapter's ending beat from its last ~60 words."""
    end = tail(text, 60)
    low = end.lower()
    last = (sentences(end) or [""])[-1].strip()
    if last.endswith("?"):
        return "question"
    if any(w in low for w in lexicon.REVEAL):
        return "revelation"
    if any(p in low for p in ARRIVAL):
        return "arrival"
    if any(w in low for w in DECISION):
        return "decision"
    if _density(end, lexicon.TENSION) > 0 or any(w in low for w in lexicon.CLIFFHANGER):
        return "danger"
    return "soft"


# Structural beats by position through the planned arc (0..1).
def expected_beat(number: int, total: int) -> str:
    if total <= 1:
        return "standalone"
    pos = (number - 1) / max(1, total - 1)
    if pos < 0.12:
        return "setup"
    if pos < 0.45:
        return "rising action"
    if pos < 0.55:
        return "midpoint turn"
    if pos < 0.78:
        return "escalation"
    if pos < 0.92:
        return "climax"
    return "resolution"


def tension_shape(text: str) -> str:
    """Rough shape of the tension curve: rising / falling / flat / spiky."""
    curve = [_density(s, lexicon.TENSION) for s in segments(text, 6)]
    if not curve or max(curve) == 0:
        return "flat"
    first, last = curve[0], curve[-1]
    peak = max(curve)
    if peak - min(curve) < 0.5 * (peak or 1):
        return "flat"
    if last > first * 1.3:
        return "rising"
    if first > last * 1.3:
        return "falling"
    return "spiky"


@dataclass
class StructureReport:
    cliff_type: str = "soft"
    conflict: float = 0.0
    try_fail: dict = field(default_factory=dict)
    expected_beat: str = ""
    tension_shape: str = ""
    issues: list[str] = field(default_factory=list)


def report(text: str, number: int = 0, total: int = 0, recent_cliffs: list[str] | None = None) -> StructureReport:
    ct = cliff_type(text)
    conflict = conflict_level(text)
    tf = try_fail(text)
    beat = expected_beat(number, total) if total else ""
    shape = tension_shape(text)
    issues: list[str] = []
    if word_count(text) >= 150:
        if conflict < 1.0:
            issues.append("this chapter has little conflict; a scene needs someone wanting something and something in the way")
        if tf["too_easy"]:
            issues.append("the win comes at no visible cost; make success expensive or partial (try-fail, not try-succeed)")
        if recent_cliffs is not None and ct != "soft":
            run = 1
            for prev in reversed(recent_cliffs):
                if prev == ct:
                    run += 1
                else:
                    break
            if run >= 3:
                issues.append(f"the same cliffhanger type ('{ct}') {run} chapters running; vary how chapters end")
        if beat in ("climax", "resolution") and shape == "flat":
            issues.append(f"this is the {beat} by arc position but the tension is flat; raise the stakes here")
        if beat == "resolution" and ct == "danger":
            issues.append("a danger cliffhanger at the resolution; the arc should be landing, not re-opening")
    return StructureReport(ct, conflict, tf, beat, shape, issues)
