"""Turn chapters and novels into numeric features.

The guiding question is "what makes a reader click Next Chapter?" so the features are
organised around four things:

* **Hook** - how fast the first ~200 words create a question in the reader's head.
* **Momentum** - pacing: sentence length variance, paragraph size, dialogue ratio, tension curve.
* **Payoff** - reveals, progression markers, emotional warmth; the reward loop.
* **Cliffhanger** - how unresolved the last ~200 words leave things.

Everything is transparent and dependency-free. The learner (``gauge.learn``) decides which
of these actually correlate with engagement on the data you ingest.
"""
from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Sequence

from . import lexicon
from .models import ChapterRecord, NovelRecord
from .text import (
    dialogue_ratio,
    flesch_reading_ease,
    head,
    paragraphs,
    segments,
    sentences,
    tail,
    word_count,
    words,
)

TENSION_SEGMENTS = 10


def _lex_density(text: str, lex: Iterable[str]) -> float:
    ws = [w.lower() for w in words(text)]
    if not ws:
        return 0.0
    lexset = set(lex)
    hits = sum(1 for w in ws if w in lexset)
    # multiword entries
    low = text.lower()
    hits += sum(low.count(e) for e in lexset if " " in e)
    return 1000.0 * hits / len(ws)  # per 1000 words


def _safe_mean(xs: Sequence[float], default: float = 0.0) -> float:
    return float(statistics.fmean(xs)) if xs else default


def _safe_std(xs: Sequence[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


# --------------------------------------------------------------------------------------
# Hook / cliffhanger scoring
# --------------------------------------------------------------------------------------
def hook_score(text: str) -> dict[str, float]:
    """Heuristic strength of the opening. Returns sub-features; the composite is ``hook``."""
    opening = head(text, 200)
    ss = sentences(opening)
    first = ss[0] if ss else opening
    first_words = [w.lower() for w in words(first)]
    first_len = len(first_words)

    short_first = 1.0 if 0 < first_len <= 12 else 0.0
    opens_dialogue = 1.0 if opening.lstrip().startswith(("\"", "“", "'", "‘")) else 0.0
    question = 1.0 if "?" in opening[:400] else 0.0
    hook_verb = 1.0 if any(w in lexicon.HOOK_VERBS for w in first_words) else 0.0
    tension = _lex_density(opening, lexicon.TENSION) / 50.0
    in_media_res = 1.0 if first_words and first_words[0] not in {"it", "the", "there", "in", "on"} else 0.0
    negation = 1.0 if re.search(r"\b(never|no one|nobody|nothing|didn't|wasn't|couldn't)\b", opening.lower()) else 0.0

    composite = (
        0.20 * short_first
        + 0.15 * opens_dialogue
        + 0.15 * question
        + 0.20 * hook_verb
        + 0.15 * min(1.0, tension)
        + 0.05 * in_media_res
        + 0.10 * negation
    )
    return {
        "hook": composite,
        "hook_first_sentence_len": float(first_len),
        "hook_opens_dialogue": opens_dialogue,
        "hook_question": question,
        "hook_verb": hook_verb,
        "hook_tension": min(1.0, tension),
    }


def cliffhanger_score(text: str) -> dict[str, float]:
    ending = tail(text, 200)
    ps = paragraphs(ending) or [ending]
    last_par = ps[-1]
    last_ss = sentences(last_par)
    last = last_ss[-1] if last_ss else last_par
    last_words = [w.lower() for w in words(last)]

    short_last_par = 1.0 if word_count(last_par) <= 25 else 0.0
    punchy_last = 1.0 if 0 < len(last_words) <= 10 else 0.0
    ends_question = 1.0 if last.rstrip().endswith("?") else 0.0
    ends_dialogue = 1.0 if last.rstrip().endswith(("\"", "”")) else 0.0
    ellipsis = 1.0 if last.rstrip().endswith(("...", "…", "—", "-")) else 0.0
    cliff_lex = min(1.0, _lex_density(ending, lexicon.CLIFFHANGER) / 60.0)
    reveal_lex = min(1.0, _lex_density(ending, lexicon.REVEAL) / 40.0)
    tension = min(1.0, _lex_density(ending, lexicon.TENSION) / 50.0)
    # A good cliffhanger introduces something NEW at the very end: named entity / proper noun
    # density in the final paragraph is a cheap proxy.
    proper = sum(1 for w in words(last_par) if w[:1].isupper()) / max(1, word_count(last_par))

    composite = (
        0.20 * short_last_par
        + 0.15 * punchy_last
        + 0.10 * ends_question
        + 0.10 * ends_dialogue
        + 0.05 * ellipsis
        + 0.20 * cliff_lex
        + 0.10 * reveal_lex
        + 0.10 * tension
    )
    return {
        "cliff": composite,
        "cliff_short_last_par": short_last_par,
        "cliff_punchy_last": punchy_last,
        "cliff_ends_question": ends_question,
        "cliff_ends_dialogue": ends_dialogue,
        "cliff_lex": cliff_lex,
        "cliff_reveal": reveal_lex,
        "cliff_tension": tension,
        "cliff_proper_noun_ratio": float(proper),
    }


# --------------------------------------------------------------------------------------
# Pacing / momentum
# --------------------------------------------------------------------------------------
def tension_curve(text: str, n: int = TENSION_SEGMENTS) -> list[float]:
    return [_lex_density(seg, lexicon.TENSION) for seg in segments(text, n)]


def pacing_features(text: str) -> dict[str, float]:
    ss = sentences(text)
    ps = paragraphs(text)
    s_lens = [word_count(s) for s in ss] or [0]
    p_lens = [word_count(p) for p in ps] or [0]
    curve = tension_curve(text)
    peak_idx = max(range(len(curve)), key=lambda i: curve[i]) if curve else 0
    rising = 0.0
    if len(curve) >= 2:
        first_half = _safe_mean(curve[: len(curve) // 2])
        second_half = _safe_mean(curve[len(curve) // 2:])
        rising = second_half - first_half
    exclaim = text.count("!") / max(1, len(ss))
    short_sentence_ratio = sum(1 for n in s_lens if n <= 6) / max(1, len(s_lens))
    one_line_pars = sum(1 for n in p_lens if n <= 12) / max(1, len(p_lens))
    return {
        "sent_len_mean": _safe_mean(s_lens),
        "sent_len_std": _safe_std(s_lens),
        "short_sentence_ratio": short_sentence_ratio,
        "par_len_mean": _safe_mean(p_lens),
        "par_len_std": _safe_std(p_lens),
        "one_line_par_ratio": one_line_pars,
        "paragraph_count": float(len(ps)),
        "dialogue_ratio": dialogue_ratio(text),
        "exclaim_per_sentence": exclaim,
        "readability": flesch_reading_ease(text),
        "tension_mean": _safe_mean(curve),
        "tension_peak": max(curve) if curve else 0.0,
        "tension_peak_position": peak_idx / max(1, len(curve) - 1),
        "tension_rising": rising,
        "tension_final": curve[-1] if curve else 0.0,
    }


def payoff_features(text: str) -> dict[str, float]:
    return {
        "reveal_density": _lex_density(text, lexicon.REVEAL),
        "progression_density": _lex_density(text, lexicon.PROGRESSION),
        "warmth_density": _lex_density(text, lexicon.WARMTH),
        "tension_density": _lex_density(text, lexicon.TENSION),
    }


def voice_features(text: str) -> dict[str, float]:
    ws = [w.lower() for w in words(text)]
    n = max(1, len(ws))
    first_person = sum(1 for w in ws if w in {"i", "me", "my", "mine", "myself"}) / n
    third_person = sum(1 for w in ws if w in {"he", "she", "his", "her", "him", "they", "them"}) / n
    unique = len(set(ws)) / n
    content = [w for w in ws if w not in lexicon.STOPWORDS and len(w) > 2]
    long_words = sum(1 for w in content if len(w) >= 9) / max(1, len(content))
    adverbs = sum(1 for w in ws if w.endswith("ly") and len(w) > 4) / n
    return {
        "first_person_ratio": first_person,
        "third_person_ratio": third_person,
        "type_token_ratio": unique,
        "long_word_ratio": long_words,
        "adverb_ratio": adverbs,
    }


def title_features(title: str) -> dict[str, float]:
    t = title.strip()
    low = t.lower()
    numbered = 1.0 if re.search(r"\b(chapter|ch\.?|episode|part)\s*\d+", low) else 0.0
    has_subtitle = 1.0 if re.search(r"[:\-–—]\s*\S", t) else 0.0
    return {
        "title_words": float(word_count(t)),
        "title_numbered_only": 1.0 if numbered and has_subtitle == 0.0 else 0.0,
        "title_has_subtitle": has_subtitle,
        "title_question": 1.0 if "?" in t else 0.0,
    }


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def chapter_features(ch: ChapterRecord) -> dict[str, float]:
    text = ch.text
    f: dict[str, float] = {"words": float(word_count(text)), "log_words": math.log1p(word_count(text))}
    f.update(hook_score(text))
    f.update(cliffhanger_score(text))
    f.update(pacing_features(text))
    f.update(payoff_features(text))
    f.update(voice_features(text))
    f.update(title_features(ch.title))
    return f


def novel_features(novel: NovelRecord, chapter_feats: list[dict[str, float]],
                   chapter_dates: Sequence = ()) -> dict[str, float]:
    """Aggregate chapter features to the novel level and add novel-only signals."""
    f: dict[str, float] = {}
    keys = sorted({k for cf in chapter_feats for k in cf})
    for k in keys:
        vals = [cf[k] for cf in chapter_feats if k in cf]
        f[f"mean_{k}"] = _safe_mean(vals)
        if k in {"words", "hook", "cliff", "dialogue_ratio", "tension_mean"}:
            f[f"std_{k}"] = _safe_std(vals)
    # The first three chapters decide whether a reader stays; weight them separately.
    early = chapter_feats[:3]
    if early:
        f["early_hook"] = _safe_mean([c.get("hook", 0.0) for c in early])
        f["early_cliff"] = _safe_mean([c.get("cliff", 0.0) for c in early])
        f["early_words"] = _safe_mean([c.get("words", 0.0) for c in early])
    f["chapters_analyzed"] = float(len(chapter_feats))
    f["log_chapters"] = math.log1p(novel.chapter_count or len(chapter_feats))

    # Release cadence: chapters per week over the observed span.
    dates = sorted(d for d in chapter_dates if d is not None)
    if len(dates) >= 2:
        span_days = max(1.0, (dates[-1] - dates[0]).total_seconds() / 86400.0)
        f["chapters_per_week"] = 7.0 * (len(dates) - 1) / span_days
        gaps = [(b - a).total_seconds() / 86400.0 for a, b in zip(dates, dates[1:])]
        f["release_gap_std_days"] = _safe_std(gaps)
    else:
        f["chapters_per_week"] = 0.0
        f["release_gap_std_days"] = 0.0

    # Blurb / packaging.
    blurb = novel.blurb or ""
    f["blurb_words"] = float(word_count(blurb))
    f["blurb_question"] = 1.0 if "?" in blurb else 0.0
    f["blurb_first_person"] = voice_features(blurb)["first_person_ratio"] if blurb else 0.0
    f["blurb_progression"] = _lex_density(blurb, lexicon.PROGRESSION) if blurb else 0.0
    f["blurb_tension"] = _lex_density(blurb, lexicon.TENSION) if blurb else 0.0
    f["title_words"] = float(word_count(novel.title))
    f["tag_count"] = float(len(novel.tags) + len(novel.genres))
    for g in lexicon.GENRE_KEYWORDS:
        joined = " ".join(novel.genres + novel.tags).lower()
        f[f"tagged_{g}"] = 1.0 if g in joined else 0.0
    return f


def feature_names(sample: dict[str, float]) -> list[str]:
    return sorted(sample.keys())
