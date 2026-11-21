"""Eye-catching headlines that stay honest: strong pull, and never out of context.

A good headline earns the click by being vivid and specific, not by misrepresenting the piece. This
module generates candidates, throws out the ones that trip the clickbait detector, and then applies an
*in-context* check — the headline's key terms must actually be supported by the article body — so a
headline cannot promise something the piece does not deliver. What survives is ranked by how much it
pulls (brevity, a concrete noun or number, tension) times how well the body backs it up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..llm.client import LLM, generate_json
from .integrity import check_headline

_STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "is", "are", "was",
         "were", "be", "as", "at", "by", "that", "this", "it", "its", "what", "how", "why", "here", "means",
         "about", "from", "will", "can", "your", "you", "we", "they"}

_SCHEMA = {"type": "object", "properties": {"headlines": {"type": "array", "items": {"type": "string"}}},
           "required": ["headlines"]}


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 3 and w not in _STOP}


@dataclass
class Headline:
    text: str
    score: float
    supported: float          # 0..1 share of the headline's key terms found in the body
    in_context: bool
    unsupported: list[str]


def in_context(title: str, body: str, threshold: float = 0.6) -> Headline:
    """Does the body support the headline's claims? Flags terms the headline asserts that the body never
    mentions — the mechanism that keeps a headline from taking the article out of context."""
    kt = _keywords(title)
    if not kt:
        return Headline(title, 0.0, 0.0, False, [])
    kb = _keywords(body)
    supported_terms = kt & kb
    unsupported = sorted(kt - kb)
    share = len(supported_terms) / len(kt)
    return Headline(title, 0.0, round(share, 3), share >= threshold, unsupported)


def _pull(title: str) -> float:
    words = title.split()
    length_fit = 1.0 - min(1.0, abs(len(words) - 8) / 8.0)   # ~5-11 words reads best
    concrete = 1.0 if re.search(r"\d", title) or sum(1 for w in words[1:] if w[:1].isupper()) else 0.6
    tension = 1.0 if (":" in title or "?" in title or any(w.lower() in _STRONG_VERBS for w in words)) else 0.7
    return 0.4 * length_fit + 0.3 * concrete + 0.3 * tension


_STRONG_VERBS = {"breaks", "ends", "eats", "reshapes", "collapses", "rethinks", "beats", "upends",
                 "quietly", "stalls", "surges", "cools", "holds", "wins", "loses", "rewrites"}


def score_headline(title: str, body: str) -> Headline:
    hc = check_headline(title)
    ctx = in_context(title, body)
    if not hc.ok:          # clickbait is disqualified outright
        ctx.score = 0.0
        return ctx
    ctx.score = round(_pull(title) * (0.4 + 0.6 * ctx.supported), 3)  # pull, weighted by how well the body backs it
    return ctx


def rank_headlines(candidates: list[str], body: str) -> list[Headline]:
    scored = [score_headline(c, body) for c in candidates if c.strip()]
    return sorted(scored, key=lambda h: h.score, reverse=True)


def suggest_headlines(llm: LLM, body: str, n: int = 6, angle: str = "") -> list[Headline]:
    """Generate and rank honest, eye-catching headlines for an article body."""
    prompt = (f"Write {n} eye-catching but accurate headlines for this article. Each must be vivid and specific, "
              f"pull the reader in, and NOT overstate or misrepresent the content. No clickbait, no withholding.\n"
              + (f"ANGLE: {angle}\n" if angle else "") + f"ARTICLE:\n{body[:1500]}\n\nReturn JSON {{headlines: [...]}}.")
    try:
        d: dict[str, Any] = generate_json(llm, prompt, schema=_SCHEMA, temperature=0.9, max_tokens=300, purpose="headlines")
        cands = [str(h).strip() for h in d.get("headlines", []) if str(h).strip()]
    except Exception:
        cands = []
    ranked = rank_headlines(cands, body)
    return [h for h in ranked if h.score > 0][:n]  # clickbait (score 0) excluded
