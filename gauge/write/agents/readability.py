"""The Readability Coach: makes "smooth to read" a number the reviser optimises.

Measured, no model calls. Good serial prose is not just *simple* — it *varies*: sentence
length should rise and fall (burstiness), paragraphs should breathe, and the syllable load
should stay light enough to skim on a phone. This agent scores those and hands the reviser
concrete cuts.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ...text import dialogue_ratio, flesch_reading_ease, paragraphs, sentences, syllables, words
from .base import AgentReview, SceneContext, StoryAgent


class ReadabilityState(BaseModel):
    history: list[dict] = Field(default_factory=list)


def measure(text: str) -> dict[str, float]:
    """Readability metrics for a chapter. All cheap, all deterministic."""
    sents = sentences(text)
    pars = paragraphs(text)
    s_lens = [len(words(s)) for s in sents if s.strip()]
    p_lens = [len(words(p)) for p in pars if p.strip()]
    all_words = words(text)
    n = max(1, len(all_words))
    mean_s = sum(s_lens) / max(1, len(s_lens))
    # Burstiness: how much sentence length varies, normalised by the mean. ~0.5+ reads well.
    if len(s_lens) > 1 and mean_s > 0:
        var = sum((x - mean_s) ** 2 for x in s_lens) / len(s_lens)
        burstiness = (var ** 0.5) / mean_s
    else:
        burstiness = 0.0
    long_s = sum(1 for x in s_lens if x > 30) / max(1, len(s_lens))  # lungbuster sentences
    syl_per_word = sum(syllables(w) for w in all_words) / n
    big_words = sum(1 for w in all_words if syllables(w) >= 4) / n  # polysyllable density
    short_par = sum(1 for x in p_lens if x <= 12) / max(1, len(p_lens))
    wall_par = sum(1 for x in p_lens if x > 90) / max(1, len(p_lens))  # walls of text
    return {
        "flesch": flesch_reading_ease(text),
        "sent_len_mean": round(mean_s, 2),
        "burstiness": round(burstiness, 3),
        "long_sentence_ratio": round(long_s, 3),
        "syllables_per_word": round(syl_per_word, 3),
        "polysyllable_ratio": round(big_words, 3),
        "short_paragraph_ratio": round(short_par, 3),
        "wall_paragraph_ratio": round(wall_par, 3),
        "dialogue_ratio": round(dialogue_ratio(text), 3),
    }


# (metric, low ok, high ok, issue when below low, issue when above high). None = no bound on that side.
_BANDS: dict[str, tuple] = {
    "flesch": (60.0, None, "prose scores hard to read (Flesch {v:.0f}); shorten sentences and cut long words", None),
    "burstiness": (0.45, None, "sentences are too even in length ({v:.2f}); mix a very short line into runs of longer ones", None),
    "long_sentence_ratio": (None, 0.12, None, "{v:.0%} of sentences run over 30 words; split the longest ones"),
    "syllables_per_word": (None, 1.7, None, "word complexity is high ({v:.2f} syllables/word); swap long words for plain ones"),
    "wall_paragraph_ratio": (None, 0.1, None, "{v:.0%} of paragraphs are walls of text; break them for phone reading"),
    "short_paragraph_ratio": (0.2, None, "too few breathing-room paragraphs ({v:.0%}); add one-line beats between dense blocks", None),
}


def score_and_issues(m: dict[str, float]) -> tuple[float, list[str]]:
    issues: list[str] = []
    met = 0
    for key, (lo, hi, lo_msg, hi_msg) in _BANDS.items():
        v = m.get(key, 0.0)
        ok = True
        if lo is not None and v < lo:
            issues.append(lo_msg.format(v=v))
            ok = False
        if hi is not None and v > hi:
            issues.append(hi_msg.format(v=v))
            ok = False
        if ok:
            met += 1
    score = met / len(_BANDS)
    return round(score, 3), issues


class ReadabilityCoach(StoryAgent):
    name = "readability"
    title = "Readability Coach"
    weight = 0.5
    state_model = ReadabilityState

    def brief(self, ctx: SceneContext) -> str:
        if not self.state.history:
            return ""
        last = self.state.history[-1]
        if last.get("issues"):
            return "Last chapter's readability notes to avoid repeating:\n" + "\n".join(f"- {i}" for i in last["issues"][:3])
        return ""

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        if len(words(text)) < 80:
            return None
        m = measure(text)
        score, issues = score_and_issues(m)
        self._last = {"metrics": m, "issues": issues}
        return AgentReview(agent=self.name, score=score, issues=issues[:3], data=m)

    def remember(self, text: str, ctx: SceneContext) -> None:
        last = getattr(self, "_last", None)
        if not last:
            return
        self.state.history.append({"chapter": ctx.number, **last["metrics"], "issues": last["issues"]})
        self.state.history = self.state.history[-200:]

    def report(self) -> str:
        if not self.state.history:
            return "(no chapters measured yet)"
        rows = self.state.history[-10:]
        avg = lambda k: sum(r.get(k, 0.0) for r in rows) / len(rows)  # noqa: E731
        return (f"last {len(rows)} chapters: Flesch {avg('flesch'):.0f}, burstiness {avg('burstiness'):.2f}, "
                f"dialogue {avg('dialogue_ratio'):.0%}, walls {avg('wall_paragraph_ratio'):.0%}")
