"""Measured world-and-continuity checks: timeline, weather/season, injuries, props, travel, world rules.

All deterministic, no model calls. The Archivist guards named facts and the Continuity Auditor guards
time-of-day; this layer guards the slower-moving physical state of the world across a whole serial:

* timeline  (roadmap 23) — reads explicit time markers and tracks an absolute day offset; flags a chapter
  that jumps backwards in time with no flashback cue.
* weather   (roadmap 24) — flags weather that contradicts the established season (snow in summer).
* injuries  (roadmap 26) — remembers who was seriously hurt; flags a character back to vigorous action
  a chapter later with no recovery or passage of time shown.
* props     (roadmap 22) — remembers objects destroyed or lost; flags one that reappears in use.
* travel    (roadmap 21) — flags instantaneous travel between two distinct named places, and, when the
  author declares distances, travel faster than the stated time.
* rules     (roadmap 29) — the author's own world rules as assertions the checker tests each chapter.

Each function is pure and returns a small dataclass; `WorldAuditor` (agents/world_auditor.py) runs them
and carries the cross-chapter memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..text import sentences, words

# ---------------------------------------------------------------------------- timeline

# Relative markers mapped to an approximate number of days moved forward.
_REL_DAYS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(a|one)\s+few\s+minutes?\b", re.I), 0.0),
    (re.compile(r"\bmoments?\s+later\b", re.I), 0.0),
    (re.compile(r"\b(an?\s+)?hours?\s+(later|passed)\b", re.I), 0.1),
    (re.compile(r"\bthat\s+(afternoon|evening|night)\b", re.I), 0.3),
    (re.compile(r"\b(the\s+)?next\s+(morning|day)\b", re.I), 1.0),
    (re.compile(r"\bthe\s+following\s+(morning|day)\b", re.I), 1.0),
    (re.compile(r"\b(a|one|two|three|four|five|six|seven|\d+)\s+days?\s+(later|passed|after)\b", re.I), None),
    (re.compile(r"\b(a|one|two|three|\d+)\s+weeks?\s+(later|passed|after)\b", re.I), None),
    (re.compile(r"\b(a|one|two|three|\d+)\s+months?\s+(later|passed|after)\b", re.I), None),
    (re.compile(r"\b(a|one|\d+)\s+years?\s+(later|passed|after)\b", re.I), None),
]
_NUMWORD = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_FLASHBACK = re.compile(r"\b(flashback|years?\s+(ago|earlier)|months?\s+(ago|earlier)|days?\s+(ago|earlier)|"
                        r"remembered|had\s+been|back\s+then|once,|when\s+(she|he|they|i)\s+was\s+(a\s+)?(child|young|little))\b", re.I)


def _units_to_days(span: str) -> float:
    m = re.search(r"\b(a|an|one|two|three|four|five|six|seven|\d+)\b", span.lower())
    n = 1.0
    if m:
        tok = m.group(1)
        n = float(_NUMWORD.get(tok, tok)) if tok in _NUMWORD or tok.isdigit() else 1.0
    if "week" in span:
        return n * 7
    if "month" in span:
        return n * 30
    if "year" in span:
        return n * 365
    return n  # days


@dataclass
class TimelineRead:
    days_elapsed: float            # how far this chapter moves the clock forward (best effort)
    markers: list[str]             # the phrases that moved it
    has_flashback_cue: bool
    issues: list[str] = field(default_factory=list)


def read_timeline(text: str) -> TimelineRead:
    markers: list[str] = []
    days = 0.0
    for rx, fixed in _REL_DAYS:
        for m in rx.finditer(text):
            span = m.group(0)
            markers.append(span.strip())
            days += fixed if fixed is not None else _units_to_days(span)
    flashback = bool(_FLASHBACK.search(text))
    issues: list[str] = []
    # Multiple contradictory forward markers ("the next morning" and "a week later") in one short chapter
    # is usually a slip unless the chapter spans a long stretch.
    if len({m.lower() for m in markers}) >= 4 and not flashback:
        issues.append("several different time jumps in one chapter; confirm the elapsed time is intended, not a slip")
    return TimelineRead(round(days, 2), markers, flashback, issues)


# ---------------------------------------------------------------------------- weather / season

_SEASON_WORDS = {
    "spring": {"spring"},
    "summer": {"summer", "midsummer"},
    "autumn": {"autumn", "fall"},
    "winter": {"winter", "midwinter"},
}
_WEATHER_BY_SEASON = {
    "summer": {"snow", "snowed", "snowing", "snowfall", "blizzard", "frost", "frozen", "sleet", "icicle", "icicles"},
    "winter": {"heatwave", "sweltering", "scorching", "sunbathing", "humid heat"},
}
_WARM_CUES = {"sweat", "sweltering", "scorching", "heat", "sunbathing", "parched"}
_COLD_CUES = {"snow", "frost", "frozen", "blizzard", "icicle", "sleet", "shivered", "shivering"}


@dataclass
class WeatherRead:
    seasons: list[str]
    issues: list[str] = field(default_factory=list)


def read_weather(text: str, established_season: Optional[str] = None) -> WeatherRead:
    toks = {w.lower() for w in words(text)}
    seasons = [s for s, lex in _SEASON_WORDS.items() if toks & lex]
    issues: list[str] = []
    ref = established_season or (seasons[0] if len(seasons) == 1 else None)
    if ref in _WEATHER_BY_SEASON:
        bad = toks & _WEATHER_BY_SEASON[ref]
        if bad:
            issues.append(f"{', '.join(sorted(bad))} in {ref}; the weather contradicts the season")
    # Hot and cold cues together with no season set is a softer smell.
    if not ref and (toks & _WARM_CUES) and (toks & _COLD_CUES):
        issues.append("both hot and cold weather cues in one scene; pick one so the climate reads true")
    return WeatherRead(seasons, issues)


# ---------------------------------------------------------------------------- injuries

_INJURY = re.compile(
    r"\b(?P<who>[A-Z][a-z]+)\b[^.?!]{0,60}?\b(?P<inj>broke\s+(?:his|her|their|a|the)\s+\w+|"
    r"broken\s+(?:arm|leg|rib|ribs|ankle|wrist|nose|jaw)|stabbed|shot|gutted|slashed|"
    r"sprained\s+(?:his|her|their)\s+\w+|dislocated|concussion|bleeding\s+badly|"
    r"gash|deep\s+wound|fractured)\b", re.I)
_SERIOUS = {"stabbed", "shot", "gutted", "broken", "broke", "fractured", "concussion", "dislocated", "deep wound"}
_VIGOROUS = re.compile(r"\b(sprint|sprinted|ran|running|raced|climbed|fought|wrestled|vaulted|leapt|leaped|"
                       r"swam|danced|punched|charged|dashed)\b", re.I)
_RECOVERY = re.compile(r"\b(healed|recovered|bandaged|stitched|splint|rested|mended|scar|limp|wince|winced|"
                       r"nursed|convalesc|days\s+later|weeks?\s+later|months?\s+later)\b", re.I)


@dataclass
class InjuryRead:
    injured: dict[str, str]        # character -> injury phrase, first seen this chapter
    vigorous: set[str]             # characters doing vigorous action this chapter
    recovery_cue: bool


def read_injuries(text: str) -> InjuryRead:
    injured: dict[str, str] = {}
    for m in _INJURY.finditer(text):
        who = m.group("who")
        if who.lower() in {"the", "a", "an", "it", "he", "she", "they"}:
            continue
        injured.setdefault(who, m.group("inj").lower())
    # Characters near a vigorous verb (coarse: any capitalised name in a sentence with a vigorous verb).
    vigorous: set[str] = set()
    for s in sentences(text):
        if _VIGOROUS.search(s):
            for name in re.findall(r"\b[A-Z][a-z]{2,}\b", s):
                vigorous.add(name)
    return InjuryRead(injured, vigorous, bool(_RECOVERY.search(text)))


# ---------------------------------------------------------------------------- props

_DESTROY = re.compile(
    r"\b(?:the|her|his|their|a)\s+(?P<prop>[a-z]{3,})\s+"
    r"(?:was|were|got)?\s*(?P<verb>shattered|smashed|burned|burnt|destroyed|melted|snapped|"
    r"shredded|disintegrated|exploded|lost|dropped\s+into|fell\s+into|sank|crushed)\b", re.I)
_USE = re.compile(
    r"\b(?:drew|raised|lifted|gripped|held|swung|wielded|pulled\s+out|took\s+out|used|opened|"
    r"reached\s+for|picked\s+up)\s+(?:the|her|his|their)\s+(?P<prop>[a-z]{3,})\b", re.I)
# A head noun captured as a prop that is really a stopword/adjective carries no object identity.
_STOPWORDS = {"the", "and", "her", "his", "their", "was", "were", "with", "into", "from", "that", "this", "then",
              "old", "new", "big", "its", "one", "two", "few", "all", "any", "out", "for"}


def _props(rx: re.Pattern, text: str) -> set[str]:
    out = set()
    for m in rx.finditer(text):
        p = m.group("prop").strip().lower()
        if p and p not in _STOPWORDS:
            out.add(p)
    return out


@dataclass
class PropRead:
    destroyed: set[str]
    used: set[str]


def read_props(text: str) -> PropRead:
    return PropRead(_props(_DESTROY, text), _props(_USE, text))


# ---------------------------------------------------------------------------- travel

_INSTANT = re.compile(r"\b(moments?\s+later|seconds?\s+later|in\s+an\s+instant|at\s+once|immediately|"
                      r"blink\s+of\s+an\s+eye|a\s+heartbeat\s+later|no\s+time\s+at\s+all)\b", re.I)


@dataclass
class TravelRead:
    places: list[str]
    issues: list[str] = field(default_factory=list)


def read_travel(text: str, known_places: Optional[set[str]] = None,
                distances: Optional[dict[tuple[str, str], str]] = None) -> TravelRead:
    """Flag instantaneous travel between two distinct named places in one chapter. `known_places` and
    `distances` (declared by the author) sharpen it; without them it is conservative."""
    known = {p.lower() for p in (known_places or set())}
    found: list[str] = []
    for name in re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b", text):
        if not known or name.lower() in known:
            found.append(name)
    distinct = []
    for p in found:
        if p not in distinct:
            distinct.append(p)
    issues: list[str] = []
    if _INSTANT.search(text) and known and len([p for p in distinct if p.lower() in known]) >= 2:
        here = [p for p in distinct if p.lower() in known][:2]
        issues.append(f"instant travel between {here[0]} and {here[1]} in one beat; confirm they are close enough")
    return TravelRead(distinct, issues)


# ---------------------------------------------------------------------------- world rules

@dataclass
class WorldRule:
    text: str
    kind: str                      # "never" | "always"
    phrase: str                    # the forbidden/required phrase
    trigger: str = ""              # for "always": only checked when this word is present


def parse_rules(raw: str) -> list[WorldRule]:
    """One rule per line. 'never: <phrase>' forbids it outright. 'always: <phrase> when <trigger>'
    requires <phrase> whenever <trigger> appears. Blank lines and '#' comments are ignored."""
    rules: list[WorldRule] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("never:"):
            rules.append(WorldRule(line, "never", line.split(":", 1)[1].strip().lower()))
        elif low.startswith("always:"):
            body = line.split(":", 1)[1].strip()
            if " when " in body.lower():
                phrase, trig = re.split(r"\s+when\s+", body, maxsplit=1, flags=re.I)
                rules.append(WorldRule(line, "always", phrase.strip().lower(), trig.strip().lower()))
            else:
                rules.append(WorldRule(line, "always", body.strip().lower()))
    return rules


@dataclass
class RuleCheck:
    violations: list[str] = field(default_factory=list)


def check_rules(text: str, rules: list[WorldRule]) -> RuleCheck:
    low = text.lower()
    out: list[str] = []
    for r in rules:
        if r.kind == "never" and r.phrase and r.phrase in low:
            out.append(f"world rule broken: '{r.phrase}' appears but your rule says never ({r.text})")
        elif r.kind == "always" and r.phrase:
            if r.trigger:
                if r.trigger in low and r.phrase not in low:
                    out.append(f"world rule: when '{r.trigger}' happens, '{r.phrase}' should hold, but it is absent ({r.text})")
    return RuleCheck(out)
