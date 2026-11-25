"""Line-level craft checks (roadmap section A). Measured, deterministic, no model calls.

Each function catches one recurring prose weakness and returns concrete, countable evidence so the
reviser can act. They are heuristics, not a grammar engine: they flag likely problems to look at, and
err toward quiet rather than nagging. The Craft Editor agent bundles them into one review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text import paragraphs, sentences, words

# Coherence floor: below this the text is not prose at all (word salad, markup soup, a stuck loop).
COHERENCE_FLOOR = 0.5
UNREADABLE = "the text is not readable prose"

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


# ---- coherence floor --------------------------------------------------------------------------
# Is this prose at all? Four shape signals, none of which needs a model or an English word list, so
# names from any language and any spaced script pass. Thresholds sit far from where real prose lives:
# the floor is a veto, so it must stay silent on anything a person could have written on purpose.
_OPENERS = "\"'“‘«‹([{*_¿¡—–-"
_CLOSERS = "\"'”’»›)]}*_.,;:!?…—–-"
_WORD_CORE = re.compile(r"[$£€#]?[^\W_]{1,30}(?:[’'./:,…—–-]+[^\W_]{1,30})*[%°]?")
_PAUSE = re.compile(r"[—–―…-]+|\.{2,}")  # a spaced dash or ellipsis: punctuation, not noise
_SCENE_BREAK = re.compile(r"^[ \t]*[*#~=_•·—–-][ \t*#~=_•·—–-]{0,24}$", re.M)
_SPACED_DOTS = re.compile(r"\.(?:[ \t]\.){2,}")
_MARKUP = re.compile(r"[`|#\\<>{}=~^]|\*{2,}|_{2,}")
_UNIT_END = re.compile(r"([.!?…。！？]+)([\"'”’»)\]*_]*)\s+")
_UNSPACED = re.compile(r"[\u0e00-\u0e7f\u3040-\u30ff\u3400-\u9fff]")  # scripts written without spaces
_WINDOW = 25  # words; short enough that a paragraph repeated on purpose is not a loop
_RUN_ON = 150  # words without a full stop before a "sentence" counts as shapeless


@dataclass
class Coherence:
    score: float = 1.0  # 0..1; under COHERENCE_FLOOR the text is not prose
    word_share: float = 1.0  # tokens that are words, not punctuation or markup debris
    sentence_share: float = 1.0  # sentences that start and end like sentences
    markup_density: float = 0.0  # stray markup characters (backticks, pipes, ##, **) per token
    loop_share: float = 0.0  # share of the text stuck repeating a short fragment
    issues: list[str] = field(default_factory=list)


def _ramp(v: float, bad: float, good: float) -> float:
    """0 at ``bad``, 1 at ``good``, linear between (``bad`` may be above ``good``)."""
    return max(0.0, min(1.0, (v - bad) / (good - bad)))


def _is_word(token: str) -> bool:
    return bool(_WORD_CORE.fullmatch(token.lstrip(_OPENERS).rstrip(_CLOSERS)))


def _sentence_share(body: str) -> float:
    """Share of sentence-sized units that open on a capital (or digit, or caseless script), close on
    terminal punctuation and are not a 150-word run with no full stop. A lower-case start is fine after a
    closing quote or an ellipsis: dialogue tags and trailing-off are how fiction is punctuated."""
    flat = re.sub(r"\s+", " ", body).strip() + " "
    good = bad = 0
    pos, carry_on = 0, False
    for m in list(_UNIT_END.finditer(flat)) + [None]:
        unit = flat[pos:m.end(1) if m else len(flat)].strip()
        if not unit:
            break
        n_words = sum(1 for t in unit.split() if _is_word(t))
        first = next((ch for ch in unit.lstrip(_OPENERS + " ") if ch.isalnum()), "")
        lead = unit.lstrip("\"'“‘«‹([{*_¿¡ ")
        # A dash or an ellipsis in front is a dialogue tag or a trailing-off resumed ("—dijo Tomás.").
        starts = carry_on or lead.startswith(("—", "–", "-", "…", "..")) or (bool(first) and not first.islower())
        ends = m is not None or unit[-1] in "\"'”’»)]*_—–-"  # the tail may stop on a quote or a cut-off dash
        if n_words and starts and ends and n_words <= _RUN_ON:
            good += 1
        else:
            bad += max(1, n_words // _RUN_ON)
        if m is None:
            break
        pos = m.end()
        carry_on = bool(m.group(2)) or "…" in m.group(1) or len(m.group(1)) > 1
    return (good + 2) / (good + bad + 2)  # a short text gets the benefit of the doubt


def _loop_share(toks: list[str]) -> float:
    """Share of 25-word windows where fewer than half the words are distinct: a model stuck on a fragment."""
    wins = [toks[i:i + _WINDOW] for i in range(0, len(toks), _WINDOW)]
    wins = [w for w in wins if len(w) >= _WINDOW // 2]
    if not wins:
        return 0.0
    return sum(1 for w in wins if len(set(w)) < 0.5 * len(w)) / len(wins)


def coherence(text: str, floor: float = COHERENCE_FLOOR) -> Coherence:
    """Measured, model-free: does this read as prose at all? Catches what an under-trained or broken
    model emits (punctuation soup, markup debris, word fragments with no sentences, a stuck loop). It does
    not judge quality: flat, clumsy or dull prose scores 1.0, and so does word salad that is punctuated
    like prose. ``floor`` only decides whether the headline issue is added."""
    body = _SPACED_DOTS.sub("…", _SCENE_BREAK.sub("", text))
    tokens = [t for t in body.split() if not _PAUSE.fullmatch(t)]
    letters = sum(1 for ch in body if ch.isalpha())
    if letters and len(_UNSPACED.findall(body)) > 0.3 * letters:
        return Coherence()  # written without spaces between words: these word-shape measures do not apply
    wordlike = [t for t in tokens if _is_word(t)]
    if not wordlike:
        return Coherence(0.0, 0.0, 0.0, 0.0, 0.0, [f"{UNREADABLE}: there are no words in it"] if floor > 0 else [])
    word_share = (len(wordlike) + 3) / (len(tokens) + 3)
    sentence_share = _sentence_share(body)
    markup = len(_MARKUP.findall(body)) / len(tokens)
    loops = _loop_share([t.lstrip(_OPENERS).rstrip(_CLOSERS).lower() for t in wordlike])
    parts = [
        (_ramp(word_share, 0.70, 0.88), f"{1 - word_share:.0%} of the tokens are punctuation or markup debris, not words"),
        (_ramp(sentence_share, 0.50, 0.85), f"only {sentence_share:.0%} of the sentences start with a capital and end with a full stop"),
        (_ramp(markup, 0.35, 0.15), f"stray markup (backticks, pipes, ##, **) on {markup:.0%} of the tokens"),
        (_ramp(loops, 0.35, 0.05), f"{loops:.0%} of the text repeats the same few words over and over"),
    ]
    score = min(h for h, _ in parts)
    issues = [msg for h, msg in parts if h < 1.0]
    if score < floor:
        issues.insert(0, f"{UNREADABLE}; write it again in plain, complete sentences")
    return Coherence(round(score, 3), round(word_share, 3), round(sentence_share, 3), round(markup, 3), round(loops, 3), issues)
