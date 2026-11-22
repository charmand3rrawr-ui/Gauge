"""Measured subplot and foreshadow tracking (roadmap 34, 37).

All deterministic, no model calls. A long serial drops threads: a B-plot introduced and forgotten, a
detail planted to pay off later that never does, a payoff that lands with no setup. This module tracks the
threads the author declares and flags those failures. The Showrunner guards the big promises; this guards
the weave.

Declaration is light and the matching is keyword/substring, so it is robust and offline:

* subplots — ``label: keyword, keyword, ...`` ; a subplot is "present" in a chapter when any keyword
  appears. A subplot dormant for too many chapters is a dropped thread.
* foreshadow — ``plant text -> payoff text`` ; the plant should appear before the payoff. A payoff with
  no earlier plant is unearned; a plant never paid off is a loose end.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Subplot:
    label: str
    keywords: list[str]


def parse_subplots(raw: list[str]) -> list[Subplot]:
    """One subplot per entry: ``label: keyword, keyword``. An entry with no ':' uses the whole text as its
    single keyword and its own label."""
    out: list[Subplot] = []
    for item in raw or []:
        s = str(item).strip()
        if not s:
            continue
        if ":" in s:
            label, _, kws = s.partition(":")
            keywords = [k.strip().lower() for k in re.split(r"[;,]", kws) if k.strip()]
            label = label.strip()
        else:
            label, keywords = s, [s.lower()]
        if label and keywords:
            out.append(Subplot(label, keywords))
    return out


def subplot_hits(text: str, subplots: list[Subplot]) -> set[str]:
    """The labels of subplots present in this chapter (any keyword matches, case-insensitive)."""
    low = text.lower()
    return {sp.label for sp in subplots if any(k in low for k in sp.keywords)}


@dataclass
class Foreshadow:
    plant: str
    payoff: str


def parse_foreshadow(raw: list[str]) -> list[Foreshadow]:
    """One pair per entry: ``plant text -> payoff text`` (``->`` or ``=>``)."""
    out: list[Foreshadow] = []
    for item in raw or []:
        s = str(item).strip()
        m = re.split(r"\s*(?:->|=>)\s*", s, maxsplit=1)
        if len(m) == 2 and m[0].strip() and m[1].strip():
            out.append(Foreshadow(m[0].strip(), m[1].strip()))
    return out


def foreshadow_events(text: str, pairs: list[Foreshadow]) -> tuple[set[str], set[str]]:
    """(plants present, payoffs present) in this chapter, keyed by the plant text."""
    low = text.lower()
    plants = {f.plant for f in pairs if f.plant.lower() in low}
    payoffs = {f.plant for f in pairs if f.payoff.lower() in low}
    return plants, payoffs


@dataclass
class ThreadReport:
    dormant: list[str] = field(default_factory=list)          # subplot labels gone quiet too long
    unearned_payoffs: list[str] = field(default_factory=list)  # payoff landed with no prior plant
    pending_plants: list[str] = field(default_factory=list)    # planted, not yet paid off


def dormant_subplots(last_seen: dict[str, int], all_labels: list[str], current: int, gap: int) -> list[str]:
    """Subplots not seen for `gap` or more chapters (including ones never seen since they were declared)."""
    out = []
    for label in all_labels:
        seen = last_seen.get(label, 0)
        if current - seen >= gap:
            out.append(label)
    return out
