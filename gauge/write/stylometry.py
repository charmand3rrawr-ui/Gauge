"""Who sounds like whom: attribute dialogue lines to speakers and measure how distinct they are."""
from __future__ import annotations

import math
import re
from typing import Any

_QUOTE = re.compile(r"[\"“]([^\"”]{3,400})[\"”]")
SWEARS = {"damn", "hell", "shit", "fuck", "bastard", "bloody", "crap", "piss"}
CONTRACTIONS = re.compile(r"\b\w+'(?:t|s|re|ll|ve|d|m)\b", re.I)


_VERBS = r"(?:said|asked|whispered|muttered|snapped|shouted|replied|hissed|murmured|called|growled|laughed|answered|added|went on)"
_ARTICLES = {"the", "a", "an", "mr", "mrs", "ms", "dr", "old", "young"}


def _name_index(names: list[str]) -> dict[str, str]:
    """Every token form a speaker can be referred to by -> full name (lower-cased keys)."""
    idx: dict[str, str] = {}
    for n in names:
        idx[n.lower()] = n
        parts = [p for p in n.split() if p.lower() not in _ARTICLES]
        for p in parts:
            if len(p) >= 3:
                idx.setdefault(p.lower(), n)
    return idx


def attribute_dialogue(text: str, names: list[str]) -> dict[str, list[str]]:
    """Map speaker name -> their quoted lines, using tags after/before the quote ("...," the Broker said /
    Rin said, "...") and the only name present in the paragraph as a fallback."""
    idx = _name_index(names)
    if not idx:
        return {}
    alts = "|".join(re.escape(k) for k in sorted(idx, key=len, reverse=True))
    tag_after = re.compile(rf"[\"”]\s*,?\s*(?:{_VERBS}\s+)?(?:the\s+)?({alts})\b", re.I)
    tag_before = re.compile(rf"\b(?:the\s+)?({alts})\s+{_VERBS}\b[^\"“]{{0,40}}[\"“]", re.I)
    out: dict[str, list[str]] = {n: [] for n in names}
    for para in text.split("\n\n"):
        quotes = _QUOTE.findall(para)
        if not quotes:
            continue
        speaker = None
        m = tag_after.search(para) or tag_before.search(para)
        if m:
            speaker = idx.get(m.group(1).lower())
        if speaker is None:
            present = {idx[k] for k in idx if re.search(rf"\b{re.escape(k)}\b", para, re.I)}
            if len(present) == 1:
                speaker = present.pop()
        if speaker in out:
            out[speaker].extend(quotes)
    return out


def style_vector(lines: list[str]) -> dict[str, float]:
    joined = " ".join(lines)
    words = re.findall(r"[A-Za-z']+", joined)
    n = max(1, len(words))
    sents = max(1, len(re.findall(r"[.!?]+", joined)) or len(lines))
    return {
        "words_per_line": len(words) / max(1, len(lines)),
        "words_per_sentence": n / sents,
        "question_ratio": joined.count("?") / max(1, len(lines)),
        "exclaim_ratio": joined.count("!") / max(1, len(lines)),
        "contraction_ratio": len(CONTRACTIONS.findall(joined)) / n,
        "swear_ratio": sum(1 for w in words if w.lower() in SWEARS) / n,
        "long_word_ratio": sum(1 for w in words if len(w) >= 8) / n,
        "ellipsis_ratio": (joined.count("...") + joined.count("…")) / max(1, len(lines)),
    }


_SCALE = {"words_per_line": 15.0, "words_per_sentence": 10.0, "question_ratio": 0.5, "exclaim_ratio": 0.5,
          "contraction_ratio": 0.1, "swear_ratio": 0.03, "long_word_ratio": 0.15, "ellipsis_ratio": 0.3}


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(sum(((a[k] - b[k]) / _SCALE[k]) ** 2 for k in _SCALE) / len(_SCALE))


def distinctness(profiles: dict[str, dict[str, float]]) -> tuple[float, list[tuple[str, str, float]]]:
    """Mean pairwise distance and the closest pairs (who sounds like whom)."""
    names = list(profiles)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], distance(profiles[names[i]], profiles[names[j]])))
    if not pairs:
        return 1.0, []
    pairs.sort(key=lambda t: t[2])
    return sum(d for _, _, d in pairs) / len(pairs), pairs


def merge_profile(old: dict[str, float], new: dict[str, float], weight_old: int) -> dict[str, float]:
    if not old:
        return dict(new)
    return {k: (old[k] * weight_old + new[k]) / (weight_old + 1) for k in new}


def profile_summary(p: dict[str, Any]) -> str:
    return (f"{p['words_per_line']:.0f} w/line, {'questions' if p['question_ratio'] > 0.3 else 'statements'}, "
            f"{'contractions' if p['contraction_ratio'] > 0.05 else 'formal'}, {'swears' if p['swear_ratio'] > 0.005 else 'clean'}")
