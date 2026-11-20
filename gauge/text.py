"""Small, dependency-free text utilities tuned for prose."""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?…])[\"'”’)]*\s+(?=[\"'“‘(A-Z0-9])")
_WORD = re.compile(r"[A-Za-z'’\-]+")
_DIALOGUE = re.compile(r"[\"“][^\"”]{2,}[\"”]")
_VOWELS = re.compile(r"[aeiouy]+")


def paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text)]
    return [p for p in parts if p]


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    out = [s.strip() for s in _SENT_SPLIT.split(text)]
    return [s for s in out if s]


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def dialogue_spans(text: str) -> list[str]:
    return _DIALOGUE.findall(text)


def dialogue_ratio(text: str) -> float:
    total = word_count(text)
    if not total:
        return 0.0
    inside = sum(word_count(s) for s in dialogue_spans(text))
    return min(1.0, inside / total)


def syllables(word: str) -> int:
    w = word.lower().strip("'’")
    if not w:
        return 0
    count = len(_VOWELS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    ws = words(text)
    ss = sentences(text) or [text]
    if not ws:
        return 0.0
    syl = sum(syllables(w) for w in ws)
    score = 206.835 - 1.015 * (len(ws) / len(ss)) - 84.6 * (syl / len(ws))
    return max(-50.0, min(121.0, score))


def segments(text: str, n: int = 10) -> list[str]:
    """Split text into ``n`` roughly equal word-count segments (for tension curves)."""
    ws = text.split()
    if not ws:
        return [""] * n
    size = max(1, len(ws) // n)
    out = [" ".join(ws[i * size:(i + 1) * size]) for i in range(n - 1)]
    out.append(" ".join(ws[(n - 1) * size:]))
    return out


def head(text: str, n_words: int = 200) -> str:
    return " ".join(text.split()[:n_words])


def tail(text: str, n_words: int = 200) -> str:
    return " ".join(text.split()[-n_words:])


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
