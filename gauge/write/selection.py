"""Write several, keep the best: choosing between drafts without asking a model.

A small model's output varies a lot from one attempt to the next. Some attempts invent people the
story has never heard of, some loop, some stop short. Asking it several times and keeping the best
attempt is the cheapest way to make a weak writer more consistent, and GaugeLM is fast enough to
afford it.

The judging here uses no language model, so it costs almost nothing:

* **invented names**: capitalised names that are not in the story's canon (the bible, the cast, the
  chapters already written). A new name in every chapter is the commonest way a serial stops being
  consistent;
* **repetition**: the share of the text that is a phrase already used in it;
* **length**: how far it is from the length asked for;
* **measured pull**: the engagement measure Gauge learned from its library (hook, momentum, dialogue,
  ending), when one exists.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from .names import proper_names


@dataclass
class Verdict:
    score: float
    invented: list[str] = field(default_factory=list)
    repeated: float = 0.0
    words: int = 0
    measured: Optional[float] = None

    def line(self) -> str:
        bits = [f"score {self.score:.2f}", f"{self.words} words", f"{self.repeated:.0%} repeated"]
        if self.invented:
            bits.append("invented names: " + ", ".join(self.invented[:5]))
        return ", ".join(bits)


def known_names(project: Any, roster: Any = None, previous: Optional[list[str]] = None) -> set[str]:
    """Every name the story already has: from the bible, the cast (with aliases), and what has been written."""
    names: set[str] = set()

    def add(text: str) -> None:
        for part in str(text or "").replace(",", " ").replace(":", " ").split():
            w = part.strip(".;'\"()[]!?")
            if len(w) >= 3 and w[0].isupper():
                names.add(w)

    try:
        b = project.state.bible
        add(project.state.name)
        add(b.protagonist)
        for name, desc in (b.characters or {}).items():
            add(name)
            names.update(proper_names(str(desc)))
        for attr in ("setting", "world", "premise", "promise"):
            names.update(proper_names(str(getattr(b, attr, "") or "")))
    except AttributeError:
        pass
    if roster is not None:
        try:
            for c in roster.ordered():
                add(c.name)
                for a in c.aliases:
                    add(a)
        except AttributeError:
            pass
    for text in previous or []:
        names.update(proper_names(text, names))
    return names


def repeated_share(text: str, n: int = 4) -> float:
    """How much of the text is a run of `n` words that already appeared in it."""
    ws = text.lower().split()
    if len(ws) < n * 2:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    again = 0
    for i in range(len(ws) - n + 1):
        g = tuple(ws[i:i + n])
        if g in seen:
            again += 1
        seen.add(g)
    return again / (len(ws) - n + 1)


def judge(text: str, known: set[str], target_words: int = 0, measure: Optional[Callable[[str], float]] = None) -> Verdict:
    """One draft's verdict. Higher is better; the parts are kept so the choice can be explained."""
    words = len(text.split())
    if words < 5:
        return Verdict(-1.0, words=words)
    invented = sorted(w for w, c in proper_names(text, known).items() if w not in known and not any(w.startswith(k) or k.startswith(w) for k in known))
    repeated = repeated_share(text)
    measured = None
    if measure is not None:
        try:
            measured = float(measure(text))
        except Exception:
            measured = None
    off = min(1.0, abs(words - target_words) / target_words) if target_words else 0.0
    score = (0.5 if measured is None else measured) - 0.07 * min(len(invented), 6) - 0.6 * repeated - 0.25 * off
    return Verdict(round(score, 4), invented, round(repeated, 4), words, measured)


def pick(candidates: list[str], known: set[str], target_words: int = 0,
         measure: Optional[Callable[[str], float]] = None) -> tuple[int, list[Verdict]]:
    """The index of the best draft, and every draft's verdict."""
    verdicts = [judge(c, known, target_words, measure) for c in candidates]
    best = max(range(len(candidates)), key=lambda i: verdicts[i].score) if candidates else 0
    return best, verdicts
