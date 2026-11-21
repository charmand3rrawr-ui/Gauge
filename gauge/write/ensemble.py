"""Ensemble checks (roadmap section B): keep the cast alive, distinct, and in motion.

Measured, no model calls. These read the roster and the chapter to catch the ways an ensemble goes
wrong — a character who vanishes, a relationship that never changes, a protagonist with a goal but no
deeper need, a defined voice that never shows up on the page — and hand concrete notes to the writer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_MAJOR = {"protagonist", "antagonist", "supporting"}


def appeared(text: str, roster: Any, min_mentions: int = 1) -> list[str]:
    """Names of characters who appear in the chapter (by mention count)."""
    out = []
    for c in roster.ordered():
        if c.mentions(text) >= min_mentions:
            out.append(c.name)
    return out


def missing_wants_needs(roster: Any) -> list[str]:
    """Major characters with neither a conscious want nor a deeper need defined — flat on the page."""
    out = []
    for c in roster.ordered():
        if c.role in _MAJOR:
            m = c.matrix
            if not (m.wants or "").strip() and not (m.needs or "").strip():
                out.append(c.name)
    return out


def want_need_same(roster: Any) -> list[str]:
    """Major characters whose want and need are identical — no inner arc to travel."""
    out = []
    for c in roster.ordered():
        if c.role in _MAJOR and c.matrix.wants and c.matrix.needs:
            if c.matrix.wants.strip().lower() == c.matrix.needs.strip().lower():
                out.append(c.name)
    return out


def static_relationships(roster: Any) -> list[tuple[str, str]]:
    """Major-character relationships that have never moved (no history recorded). A story is relationships
    changing; one that is frozen is usually a sign the two characters never really meet on the page."""
    out = []
    majors = [c for c in roster.ordered() if c.role in _MAJOR]
    for c in majors:
        for other in majors:
            if other is c:
                continue
            rel = c.relationships.get(other.name)
            if rel is not None and not rel.history and (c.name, other.name) not in out and (other.name, c.name) not in out:
                out.append((c.name, other.name))
    return out


def unused_voice(text: str, roster: Any) -> list[str]:
    """Characters who appear and have a defined catchphrase or tic, but none of it shows this chapter."""
    low = (text or "").lower()
    out = []
    for c in roster.ordered():
        if c.mentions(text) < 2:
            continue
        markers = [str(p).lower() for p in (c.voice.catchphrases or [])] + [str(t).lower() for t in (c.voice.tics or [])]
        markers = [m for m in markers if len(m) > 3]
        if markers and not any(re.search(re.escape(m[:30]), low) for m in markers):
            out.append(c.name)
    return out


@dataclass
class EnsembleReport:
    appeared: list[str] = field(default_factory=list)
    absent_major: list[str] = field(default_factory=list)      # major characters not in this chapter
    missing_want_need: list[str] = field(default_factory=list)
    want_need_same: list[str] = field(default_factory=list)
    static_relationships: list[tuple[str, str]] = field(default_factory=list)
    unused_voice: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def report(text: str, roster: Any, appearance_gap: dict[str, int] | None = None, gap_limit: int = 5) -> EnsembleReport:
    """One pass. `appearance_gap` maps a name to chapters since last seen (tracked by the Cast Tracker)."""
    here = appeared(text, roster)
    majors = [c.name for c in roster.ordered() if c.role in _MAJOR]
    absent = [n for n in majors if n not in here]
    mwn = missing_wants_needs(roster)
    wns = want_need_same(roster)
    stat = static_relationships(roster)
    unused = unused_voice(text, roster)
    issues: list[str] = []
    if appearance_gap:
        gone = [n for n, g in appearance_gap.items() if g >= gap_limit and n in majors]
        if gone:
            issues.append(f"cast members absent for {gap_limit}+ chapters: {', '.join(gone[:4])}; bring them back or write them out")
    if mwn:
        issues.append(f"major characters with no want or need defined: {', '.join(mwn[:3])}; give them something to pursue")
    if wns:
        issues.append(f"{', '.join(wns)}: want and need are identical; the inner arc has nowhere to go")
    if len(stat) >= 2:
        issues.append(f"relationships that never change: {stat[0][0]}/{stat[0][1]} and others; let them shift")
    if unused:
        issues.append(f"defined voice not used for {', '.join(unused[:3])}; show the catchphrase or tic or drop it")
    return EnsembleReport(here, absent, mwn, wns, stat, unused, issues)
