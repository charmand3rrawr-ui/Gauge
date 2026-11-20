"""Token estimation and prompt budgeting without a tokenizer dependency.

English prose runs about 1.3 tokens per word on Llama/Qwen tokenizers; JSON and code run
higher. We over-estimate on purpose: a prompt that is cut by the server is far worse than one
that is a little shorter than it could be.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(_WORD.findall(text))
    # Punctuation-heavy or JSON-heavy text tokenises worse than prose.
    symbols = sum(text.count(ch) for ch in "{}[]\":,")
    return int(words * 1.35 + symbols * 0.5) + 1


def truncate_to_tokens(text: str, max_tokens: int, from_end: bool = False) -> str:
    """Trim ``text`` to roughly ``max_tokens``. ``from_end`` keeps the tail instead of the head."""
    if estimate_tokens(text) <= max_tokens:
        return text
    words = text.split()
    keep = max(1, int(max_tokens / 1.4))
    kept = words[-keep:] if from_end else words[:keep]
    return (("..." + " ") if from_end else "") + " ".join(kept) + ("" if from_end else " ...")


class PromptBudget:
    """Fit optional prompt sections into ``num_ctx - reserved_output - margin``.

    Fixed sections always go in; optional sections are added in priority order while they fit,
    and the last one that does not fit is truncated to the remaining room if ``shrinkable``.
    """

    def __init__(self, num_ctx: int, reserved_output: int, margin: int = 256):
        self.capacity = max(512, num_ctx - reserved_output - margin)
        self.used = 0
        self.dropped: list[str] = []

    def take(self, text: str) -> str:
        """Fixed section: always included, counted against the budget."""
        self.used += estimate_tokens(text)
        return text

    def fit(self, name: str, text: str, shrinkable: bool = True, min_tokens: int = 120) -> str:
        room = self.capacity - self.used
        need = estimate_tokens(text)
        if need <= room:
            self.used += need
            return text
        if shrinkable and room >= min_tokens:
            cut = truncate_to_tokens(text, room)
            self.used += estimate_tokens(cut)
            self.dropped.append(f"{name} (trimmed)")
            return cut
        self.dropped.append(name)
        return ""

    def fit_list(self, name: str, items: list[str], joiner: str = "\n\n", min_items: int = 0) -> str:
        """Add as many whole items as fit (most important first)."""
        kept: list[str] = []
        for i, item in enumerate(items):
            need = estimate_tokens(item) + 2
            if self.used + need > self.capacity and len(kept) >= min_items:
                self.dropped.append(f"{name}[{i}:]")
                break
            self.used += need
            kept.append(item)
        return joiner.join(kept)

    @property
    def over(self) -> bool:
        return self.used > self.capacity

    @property
    def remaining(self) -> int:
        return self.capacity - self.used
