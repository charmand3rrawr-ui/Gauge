"""Creativity controls for the planner and drafter.

* **Muse**: before the outline is extended, propose several genuinely different directions for
  the next stretch of chapters, rate them for surprise, fit and payoff, penalise anything that
  resembles a direction already taken, and hand the winner to the planner. Divergent first,
  convergent second, instead of letting the model's first idea win every time.
* **NoveltyTracker**: measure a draft against the project's own earlier chapters: n-gram overlap,
  reused openings, phrases that keep coming back. The result feeds the critic (a repetitive
  chapter scores lower) and the drafter (a list of phrases to avoid) so the voice stays alive
  across hundreds of chapters.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..llm.client import LLM, generate_json
from ..text import sentences, words
from . import prompts
from .schemas import MuseDirectionsOut, MuseRatingsOut, schema_of

log = logging.getLogger(__name__)


class Direction(BaseModel):
    title: str = ""
    pitch: str = ""
    fresh_because: str = ""
    risk: str = ""
    surprise: float = 0.0
    fit: float = 0.0
    payoff: float = 0.0
    novelty: float = 1.0

    @property
    def score(self) -> float:
        return 0.35 * self.surprise + 0.3 * self.fit + 0.2 * self.payoff + 1.5 * self.novelty


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in words(text)]


def _ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def overlap(a: str, b: str, n: int = 3) -> float:
    """Share of a's n-grams that appear in b."""
    ga, gb = _ngrams(_tokens(a), n), _ngrams(_tokens(b), n)
    return len(ga & gb) / max(1, len(ga))


class Muse:
    def __init__(self, llm: LLM, techniques_hint: str = ""):
        self.llm = llm
        self.techniques_hint = techniques_hint

    def directions(self, bible_md: str, summary: str, threads: list[str], past: list[str], k: int = 3) -> list[Direction]:
        prompt = prompts.MUSE_PROMPT.format(k=k, bible=bible_md, summary=summary or "(the story is beginning)",
                                            threads="\n".join(f"- {t}" for t in threads) or "(none)",
                                            past="\n".join(f"- {p}" for p in past[-8:]) or "(none yet)",
                                            techniques=self.techniques_hint or "(no technique library yet)")
        try:
            d = generate_json(self.llm, prompt, schema=schema_of(MuseDirectionsOut), temperature=1.0, max_tokens=1200, purpose="muse")
        except Exception as e:
            log.warning("muse failed: %s", e)
            return []
        out = []
        for x in d.get("directions", []) if isinstance(d.get("directions"), list) else []:
            if isinstance(x, dict) and x.get("pitch"):
                out.append(Direction(title=str(x.get("title", ""))[:80], pitch=str(x["pitch"])[:600],
                                     fresh_because=str(x.get("fresh_because", ""))[:300], risk=str(x.get("risk", ""))[:200]))
        return out[:k]

    def rate(self, directions: list[Direction], bible_md: str, summary: str, past: list[str]) -> list[Direction]:
        if not directions:
            return []
        # Deterministic novelty against directions already taken.
        for d in directions:
            d.novelty = 1.0 - max([overlap(d.pitch, p, 2) for p in past] or [0.0])
        listing = "\n".join(f"{i + 1}. {d.title}: {d.pitch}" for i, d in enumerate(directions))
        prompt = prompts.MUSE_PICK_PROMPT.format(directions=listing, bible=bible_md, summary=summary or "(beginning)")
        try:
            r = generate_json(self.llm, prompt, schema=schema_of(MuseRatingsOut), temperature=0.2, max_tokens=500, purpose="muse:rate")
            for s in r.get("scores", []) if isinstance(r.get("scores"), list) else []:
                try:
                    i = int(s.get("index", 0)) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= i < len(directions):
                    for k in ("surprise", "fit", "payoff"):
                        try:
                            setattr(directions[i], k, max(0.0, min(10.0, float(s.get(k, 5)))) / 10.0)
                        except (TypeError, ValueError):
                            pass
        except Exception as e:
            log.warning("muse rating failed: %s; using novelty only", e)
        return sorted(directions, key=lambda d: d.score, reverse=True)

    def choose(self, bible_md: str, summary: str, threads: list[str], past: list[str], k: int = 3) -> Optional[Direction]:
        dirs = self.rate(self.directions(bible_md, summary, threads, past, k), bible_md, summary, past)
        return dirs[0] if dirs else None


class NoveltyReport(BaseModel):
    score: float = 1.0  # 1 = nothing repeated
    ngram_overlap: float = 0.0  # 4-gram overlap with earlier chapters
    opening_repeat: bool = False
    ending_repeat: bool = False
    overused: list[str] = Field(default_factory=list)  # phrases already used in >= 2 earlier chapters
    notes: list[str] = Field(default_factory=list)


_STOP_START = {"the", "a", "an", "and", "but", "then", "she", "he", "i", "it", "they", "we", "you", "there", "when"}


def _phrases(text: str, n: int = 3) -> Counter:
    toks = _tokens(text)
    c: Counter = Counter()
    for i in range(len(toks) - n + 1):
        g = toks[i:i + n]
        if g[0] in _STOP_START and g[-1] in _STOP_START:
            continue
        c[" ".join(g)] += 1
    return c


class NoveltyTracker:
    def __init__(self, min_chapters_for_phrases: int = 2):
        self.min_chapters = min_chapters_for_phrases

    def overused_phrases(self, previous: list[str], top: int = 15) -> list[str]:
        """Phrases (3-grams) that appear in at least ``min_chapters`` earlier chapters."""
        seen_in: Counter = Counter()
        for ch in previous:
            for ph in set(_phrases(ch)):
                seen_in[ph] += 1
        common = [(ph, n) for ph, n in seen_in.items() if n >= self.min_chapters]
        common.sort(key=lambda t: (-t[1], t[0]))
        # Skip the most generic function-word trigrams.
        return [ph for ph, _ in common if not all(w in _STOP_START or len(w) <= 2 for w in ph.split())][:top]

    def report(self, text: str, previous: list[str]) -> NoveltyReport:
        rep = NoveltyReport()
        if not previous:
            return rep
        recent = previous[-6:]
        rep.ngram_overlap = max(overlap(text, p, 4) for p in recent)
        open_now = " ".join(_tokens(" ".join(sentences(text)[:1]))[:6])
        end_now = " ".join(_tokens(" ".join(sentences(text)[-1:]))[-6:])
        for p in recent:
            if open_now and open_now == " ".join(_tokens(" ".join(sentences(p)[:1]))[:6]):
                rep.opening_repeat = True
            if end_now and end_now == " ".join(_tokens(" ".join(sentences(p)[-1:]))[-6:]):
                rep.ending_repeat = True
        over = self.overused_phrases(previous)
        here = _phrases(text)
        rep.overused = [ph for ph in over if here.get(ph, 0) > 0][:10]
        penalty = min(0.6, rep.ngram_overlap * 4.0) + (0.15 if rep.opening_repeat else 0.0) + (0.1 if rep.ending_repeat else 0.0) \
            + min(0.2, 0.04 * len(rep.overused))
        rep.score = round(max(0.0, 1.0 - penalty), 3)
        if rep.ngram_overlap > 0.08:
            rep.notes.append(f"{rep.ngram_overlap:.0%} of phrases repeat earlier chapters; vary sentence shapes and images")
        if rep.opening_repeat:
            rep.notes.append("the opening line has the same shape as a recent chapter; open differently")
        if rep.ending_repeat:
            rep.notes.append("the closing line echoes a recent ending; end on a different kind of beat")
        if rep.overused:
            rep.notes.append("retire these phrases: " + ", ".join(f'"{p}"' for p in rep.overused[:5]))
        return rep


def strip_markup(text: str) -> str:
    return re.sub(r"^---.*?---\s*", "", text, flags=re.S)


def direction_record(d: Direction) -> dict[str, Any]:
    return {"title": d.title, "pitch": d.pitch, "score": round(d.score, 3)}
