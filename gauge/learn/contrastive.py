"""Contrastive learning: learn from the *difference* between a hit and a miss, not from either alone.

A single good chapter teaches a model what good looks like; a good chapter paired against a weak one
on the same serial teaches it what to *change*. This module takes the measured features of the
best-performing chapters and the worst, reports where they differ most, turns those gaps into
concrete "do more / do less" guidance the writer can act on, and can emit preference pairs
(chosen = the hit, rejected = the miss) for fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Human phrasing and polarity for the features worth contrasting. +1 = more is usually better.
_FEATURE_GUIDE: dict[str, tuple[int, str]] = {
    "hook": (1, "open with more immediate motion or a question"),
    "cliff": (1, "end on a sharper unresolved beat"),
    "dialogue_ratio": (0, "adjust the dialogue-to-prose balance toward the winners"),
    "one_line_par_ratio": (1, "break dense paragraphs into more one-line beats"),
    "tension_peak": (1, "raise the chapter's peak tension"),
    "tension_rising": (1, "move the biggest beat later so tension rises"),
    "reveal_density": (1, "land more reveals per chapter"),
    "progression_density": (1, "show more visible progress for the protagonist"),
    "exclaim_per_sentence": (-1, "lean less on exclamation marks"),
    "adverb_ratio": (-1, "cut adverbs"),
    "long_word_ratio": (-1, "use fewer long words"),
    "sent_len_mean": (0, "move average sentence length toward the winners'"),
    "type_token_ratio": (1, "vary word choice more"),
}


@dataclass
class Contrast:
    feature: str
    high_mean: float
    low_mean: float
    delta: float  # high - low
    guidance: str


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def contrast(high: list[dict], low: list[dict], keys: Optional[list[str]] = None, top: int = 8) -> list[Contrast]:
    """`high`/`low` are lists of chapter feature dicts (from gauge.features.chapter_features)."""
    keys = keys or list(_FEATURE_GUIDE)
    out: list[Contrast] = []
    for k in keys:
        hi, lo = _mean(high, k), _mean(low, k)
        d = hi - lo
        if abs(d) < 1e-6:
            continue
        polarity, phrase = _FEATURE_GUIDE.get(k, (0, f"adjust {k}"))
        # Only advise in the direction the winners went, unless the feature has no fixed polarity.
        if polarity == 0 or (polarity > 0 and d > 0) or (polarity < 0 and d < 0):
            g = phrase
        else:
            g = f"winners actually have {'higher' if d > 0 else 'lower'} {k}; reconsider the usual advice here"
        out.append(Contrast(k, round(hi, 3), round(lo, 3), round(d, 3), g))
    out.sort(key=lambda c: abs(c.delta), reverse=True)
    return out[:top]


def guidance(high: list[dict], low: list[dict], n: int = 6) -> list[str]:
    cs = contrast(high, low, top=n)
    return [f"{c.guidance} (hits {c.high_mean:.2f} vs misses {c.low_mean:.2f})" for c in cs]


def split_by_label(rows: list[tuple[dict, float]], top_frac: float = 0.3) -> tuple[list[dict], list[dict]]:
    """Given (features, label) pairs, return (high, low) feature lists split at the top/bottom fraction."""
    ordered = sorted(rows, key=lambda r: r[1])
    k = max(1, int(len(ordered) * top_frac))
    low = [f for f, _ in ordered[:k]]
    high = [f for f, _ in ordered[-k:]]
    return high, low


def build_pairs(high_texts: list[str], low_texts: list[str], prompt_summary: str = "Write an engaging chapter") -> list[dict]:
    """Preference pairs for fine-tuning: each hit chosen over each miss, capped to avoid blow-up."""
    pairs: list[dict] = []
    for h in high_texts[:10]:
        for low in low_texts[:3]:
            if h and low and h != low:
                pairs.append({"prompt": prompt_summary, "chosen": h, "rejected": low, "source": "contrastive"})
    return pairs
