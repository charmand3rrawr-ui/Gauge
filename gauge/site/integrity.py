"""Content-integrity rails for anything the author publishes.

Three checks, all cheap and deterministic: a byline must be present and may not impersonate a real
outlet or a shared pseudonym; a headline is flagged when it uses the manipulative patterns that define
clickbait (a strong, accurate headline passes); and factual-looking claims in a draft are flagged so the
author verifies them before publishing. These do not make the writing bland — they keep it honest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

OPINION_DISCLAIMER = ("Opinion: this is commentary and the view of the named author, not news reporting. "
                      "Claims are the author's own; verify before relying on them.")

# Bylines that hide who is speaking: real outlets and shared pseudonyms. Writing as these is impersonation.
IMPERSONATION_DENYLIST = {
    "tyler durden", "zerohedge", "zero hedge", "anonymous", "staff", "admin", "editor", "the editor",
    "associated press", "ap", "reuters", "bloomberg", "cnn", "fox news", "bbc", "the new york times",
    "nyt", "the guardian", "wall street journal", "wsj", "breitbart", "infowars",
}

# Clickbait / deception patterns. Strong-but-honest headlines do not match these.
_CLICKBAIT = [
    (r"you won'?t believe", "curiosity-gap bait that withholds the point"),
    (r"what happened next", "withholds the actual content"),
    (r"the reason why will (shock|surprise|amaze)", "manufactured shock"),
    (r"\b(shocking|insane|unbelievable|jaw-dropping|mind-blowing)\b", "sensational absolute"),
    (r"\b(doctors|experts|scientists) (hate|don'?t want)", "fake-authority bait"),
    (r"this one (weird )?trick", "classic clickbait formula"),
    (r"\bgone wrong\b", "vague alarm bait"),
    (r"(secret|they don'?t want you to know)", "conspiracy framing"),
    (r"number \d+ will", "listicle withholding bait"),
    (r"!!!|\?\?\?", "manipulative punctuation"),
    (r"\b(destroys?|slams?|obliterates?|annihilates?|eviscerates?)\b", "manufactured-conflict verb"),
    (r"\b(breaking|urgent|alert)\b", "false-urgency label"),
]
_COMPILED_CB = [(re.compile(p, re.I), why) for p, why in _CLICKBAIT]

# Factual-looking spans that an opinion piece should not assert as sourced fact without the author's check.
_CLAIM_PATTERNS = [
    (r"\b\d+(\.\d+)?\s?%|\b\d+(\.\d+)?\s?(?:percent|million|billion|trillion)\b", "a statistic"),
    (r"“[^”]{10,}”", "a direct quote"),
    (r"\b(according to|sources say|reportedly|a study (found|shows)|data shows)\b", "an attributed claim"),
    (r"\b(yesterday|today|this week|last week|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", "a dated event"),
]
_COMPILED_CLAIMS = [(re.compile(p, re.I), what) for p, what in _CLAIM_PATTERNS]


@dataclass
class HeadlineCheck:
    ok: bool
    flags: list[str]


def check_byline(byline: str) -> tuple[bool, str]:
    """A byline must be present and may not impersonate a real outlet or a shared pseudonym."""
    name = (byline or "").strip()
    if not name:
        return False, "a byline is required; publish under a name a reader can hold responsible"
    if name.lower() in IMPERSONATION_DENYLIST:
        return False, f"'{name}' impersonates a real outlet or a shared pseudonym; use your own name or an owned pen name"
    return True, "ok"


def check_headline(title: str) -> HeadlineCheck:
    """Flag clickbait/deception. An empty flag list means the headline is strong-but-honest."""
    flags = [f"{why}: {rx.pattern}" for rx, why in _COMPILED_CB if rx.search(title or "")]
    if (title or "").isupper() and len(title) > 12:
        flags.append("all-caps shouting")
    return HeadlineCheck(ok=not flags, flags=flags)


def flag_claims(text: str) -> list[str]:
    """Factual-looking statements in a draft that the author should verify before publishing."""
    out: list[str] = []
    for rx, what in _COMPILED_CLAIMS:
        for m in rx.finditer(text or ""):
            out.append(f"{what}: …{(text[max(0, m.start() - 20):m.end() + 20]).strip()}…")
            if len(out) >= 20:
                return out
    return out
