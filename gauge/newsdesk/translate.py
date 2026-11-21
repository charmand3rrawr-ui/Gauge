"""Translation, honestly scoped: the main model does it.

GaugeLM, the built-in model, is far too small to become fluent in many languages, and claiming otherwise
would be false. Modern instruction models (the Ollama or OpenAI-compatible model Gauge already uses) are
strong translators, so the desk translates through that. Translations are research material for the
writer; the published piece is original commentary, and it links the original-language source.
"""
from __future__ import annotations

import re
from typing import Any

# Script ranges for a cheap language-family guess (enough to know a translation is needed).
_SCRIPTS = [
    ("cjk", re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")),
    ("cyrillic", re.compile(r"[Ѐ-ӿ]")),
    ("arabic", re.compile(r"[؀-ۿ]")),
    ("hebrew", re.compile(r"[֐-׿]")),
    ("devanagari", re.compile(r"[ऀ-ॿ]")),
    ("greek", re.compile(r"[Ͱ-Ͽ]")),
    ("thai", re.compile(r"[฀-๿]")),
]


def detect_script(text: str) -> str:
    """The dominant non-Latin script, or 'latin'. A hint, not a language detector."""
    best, best_n = "latin", 0
    for name, rx in _SCRIPTS:
        n = len(rx.findall(text or ""))
        if n > best_n:
            best, best_n = name, n
    return best


def translate(llm: Any, text: str, target: str = "English", source_lang: str = "") -> str:
    """Translate `text` into `target` with the configured model. Falls back to the original on failure,
    so a translation outage never blocks the desk — the writer just sees the source language."""
    if not (text or "").strip():
        return ""
    prompt = (f"[translate] Translate the following text into {target}. Preserve names, numbers and meaning "
              f"exactly; do not add commentary. Return only the translation.\n"
              + (f"SOURCE LANGUAGE: {source_lang}\n" if source_lang else "") + f"TEXT:\n{text[:3000]}")
    try:
        out = llm.generate(prompt, temperature=0.1, max_tokens=900, purpose="translate")
        return (out or "").strip() or text
    except Exception:
        return text
