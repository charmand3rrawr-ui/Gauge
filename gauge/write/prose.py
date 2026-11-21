"""Line-level craft checks (roadmap section A). Measured, deterministic, no model calls.

Each function catches one recurring prose weakness and returns concrete, countable evidence so the
reviser can act. They are heuristics, not a grammar engine: they flag likely problems to look at, and
err toward quiet rather than nagging. The Craft Editor agent bundles them into one review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text import paragraphs, sentences, words

# 3. Cliché / tired-metaphor lexicon.
CLICHES = [
    "at the end of the day", "needle in a haystack", "tip of the iceberg", "calm before the storm",
    "avoid like the plague", "cold as ice", "heart of gold", "time will tell", "only time will tell",
    "in the nick of time", "weight of the world", "a pit in her stomach", "a pit in his stomach",
    "blood ran cold", "heart skipped a beat", "time stood still", "silence was deafening",
    "nerves of steel", "quiet as a mouse", "white as a sheet", "like a deer in headlights",
    "butterflies in her stomach", "butterflies in his stomach", "sent shivers down", "a chill ran down",
    "last but not least", "few and far between", "dead of night", "crystal clear", "add insult to injury",
]

# 9. Weak verb + -ly adverb ("ran quickly", "said softly"): usually a stronger verb exists.
_WEAK_VERB_ADVERB = re.compile(
    r"\b(ran|walked|went|said|looked|moved|spoke|smiled|laughed|cried|replied|asked|closed|opened|held|"
    r"turned|nodded|grabbed|pushed|pulled|whispered|shouted)\s+(\w+ly)\b", re.I)

# 6. On-the-nose dialogue: feelings/wants stated outright inside quotes.
_ON_NOSE = re.compile(r"\bI\s+(feel|am|'m)\s+(so\s+)?(sad|angry|afraid|scared|happy|jealous|hurt|lonely|"
                      r"furious|anxious|nervous|upset|in love|betrayed)\b|\bI\s+(want|need)\s+you\s+to\b", re.I)

# 10. Period anachronisms by era (extend as needed). Only used when a setting era is given.
ANACHRONISMS = {
    "medieval": {"phone", "car", "computer", "internet", "tv", "radio", "gun", "okay", "ok", "email", "plastic"},
    "victorian": {"phone", "internet", "computer", "tv", "email", "okay", "smartphone", "laptop", "plastic"},
    "1950s": {"internet", "smartphone", "email", "website", "selfie", "google", "laptop", "texting"},
}

_PRESENT_3SG = re.compile(r"\b\w+(?<!ss)(?<!us)(?<!is)s\b")  # crude present-tense singular verb proxy; used only ratio-wise


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in words(text)]


def cliche_hits(text: str) -> list[str]:
    low = text.lower()
    return [c for c in CLICHES if c in low]


def sentence_opener_variety(text: str) -> tuple[float, list[str]]:
    """Share of distinct opening words across sentences (1.0 = all different). Also the overused openers."""
    sents = [s.strip() for s in sentences(text) if s.strip()]
    if len(sents) < 4:
        return 1.0, []
    openers = [s.split()[0].lower().strip(",.;:\"'") for s in sents if s.split()]
    from collections import Counter
    counts = Counter(openers)
    variety = len(counts) / max(1, len(openers))
    overused = [w for w, n in counts.items() if n >= max(3, len(openers) * 0.25)]
    return round(variety, 3), overused


def weak_verb_adverbs(text: str) -> list[str]:
    return [f"{m.group(1)} {m.group(2)}" for m in _WEAK_VERB_ADVERB.finditer(text)]


def on_the_nose_lines(text: str) -> int:
    quotes = re.findall(r"“([^”]+)”|\"([^\"]+)\"", text)
    n = 0
    for a, b in quotes:
        if _ON_NOSE.search(a or b or ""):
            n += 1
    return n


def talking_heads_runs(text: str) -> int:
    """Count runs of 4+ consecutive dialogue-only paragraphs (no action beat between lines)."""
    pars = [p.strip() for p in paragraphs(text) if p.strip()]
    run = longest = 0
    for p in pars:
        is_dialogue = p.lstrip().startswith(("“", '"'))
        has_beat = len(p.split()) > 0 and not is_dialogue
        if is_dialogue and not has_beat:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return 1 if longest >= 4 else 0


def tense_mix(text: str) -> float:
    """Rough past/present balance. ~0 = consistent, ~1 = evenly mixed (a warning sign)."""
    toks = _tokens(text)
    past = sum(1 for t in toks if t.endswith("ed") and len(t) > 3)
    present = len(_PRESENT_3SG.findall(text))
    tot = past + present
    if tot < 20:
        return 0.0
    minority = min(past, present) / tot
    return round(minority * 2, 3)  # 0 when one tense dominates, →1 when 50/50


def pov_heads(text: str, names: list[str]) -> int:
    """Head-hopping proxy: how many different named characters have an interior-state verb attributed to
    them in the chapter ('X felt/thought/knew/realised'). >1 in a limited-POV scene is a warning."""
    heads = set()
    for name in names:
        if re.search(rf"\b{re.escape(name)}\b\s+\w*\s*(felt|thought|knew|realised|realized|wondered|believed)\b", text, re.I):
            heads.add(name)
    return len(heads)


def anachronisms(text: str, era: str) -> list[str]:
    bank = ANACHRONISMS.get((era or "").lower())
    if not bank:
        return []
    toks = set(_tokens(text))
    return sorted(toks & bank)


@dataclass
class CraftProfile:
    cliches: list[str] = field(default_factory=list)
    opener_variety: float = 1.0
    overused_openers: list[str] = field(default_factory=list)
    weak_pairs: list[str] = field(default_factory=list)
    on_the_nose: int = 0
    talking_heads: int = 0
    tense_mix: float = 0.0
    pov_heads: int = 0
    anachronisms: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def profile(text: str, names: list[str] | None = None, era: str = "", limited_pov: bool = False) -> CraftProfile:
    cl = cliche_hits(text)
    variety, overused = sentence_opener_variety(text)
    weak = weak_verb_adverbs(text)
    nose = on_the_nose_lines(text)
    heads_run = talking_heads_runs(text)
    tmix = tense_mix(text)
    pov = pov_heads(text, names or []) if (limited_pov and names) else 0
    anach = anachronisms(text, era)
    issues: list[str] = []
    if len(words(text)) >= 120:
        if cl:
            issues.append(f"clichés to freshen ({len(cl)}): {', '.join(cl[:3])}")
        if variety < 0.6 and overused:
            issues.append(f"too many sentences open the same way ('{overused[0]}'); vary the openings")
        if len(weak) >= 3:
            issues.append(f"weak verb+adverb pairs ({len(weak)}, e.g. '{weak[0]}'); use a stronger verb")
        if nose >= 2:
            issues.append(f"on-the-nose dialogue ({nose}x characters stating feelings outright); let subtext carry it")
        if heads_run:
            issues.append("a long run of talking-heads dialogue; break it with action and the body")
        if tmix > 0.6:
            issues.append("past and present tense are mixed; pick one and hold it")
        if pov > 1:
            issues.append(f"head-hopping: {pov} characters' inner thoughts in one scene; stay in one POV")
        if anach:
            issues.append(f"anachronisms for a {era} setting: {', '.join(anach[:4])}")
    return CraftProfile(cl, variety, overused, weak, nose, heads_run, tmix, pov, anach, issues)


def score(p: CraftProfile) -> float:
    penalties = (0.08 * min(3, len(p.cliches)) + 0.08 * min(3, len(p.weak_pairs) // 2) + 0.1 * min(2, p.on_the_nose)
                 + 0.15 * p.talking_heads + 0.15 * (1 if p.tense_mix > 0.6 else 0)
                 + 0.1 * (1 if p.pov_heads > 1 else 0) + 0.1 * (1 if p.anachronisms else 0)
                 + 0.1 * (1 if p.opener_variety < 0.6 else 0))
    return round(max(0.0, 1.0 - penalties), 3)
