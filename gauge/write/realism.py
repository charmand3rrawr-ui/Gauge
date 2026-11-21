"""Measured realism checks: sensory grounding, show-don't-tell, filter words, and physical logistics.

All deterministic, no model calls. These catch the things that make prose read as unreal — a scene with
no texture, emotions announced instead of shown, the reader held at arm's length by filter verbs, and
time of day that contradicts itself — and hand concrete fixes to the reviser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text import sentences, words

SENSES: dict[str, set[str]] = {
    "sight": {"saw", "light", "dark", "bright", "shadow", "glow", "gleam", "glint", "flicker", "blur", "colour",
              "color", "red", "blue", "green", "grey", "gray", "pale", "shine", "loom", "glare", "dim"},
    "sound": {"heard", "hum", "roar", "whisper", "crack", "silence", "echo", "hiss", "clang", "murmur", "thud",
              "scream", "ring", "buzz", "rustle", "snap", "creak", "clatter", "boom", "whine"},
    "smell": {"smell", "smelled", "reek", "reeked", "stink", "scent", "acrid", "musty", "perfume", "smoke",
              "rot", "fragrant", "stench", "odour", "odor"},
    "taste": {"taste", "tasted", "bitter", "sweet", "sour", "salt", "salty", "metallic", "tang", "savour"},
    "touch": {"cold", "warm", "hot", "rough", "smooth", "damp", "wet", "sharp", "slick", "grip", "ache",
              "sting", "soft", "hard", "cool", "sticky", "greasy", "numb", "throb"},
}

# Filter verbs that put a layer between the reader and the scene.
FILTER = {"saw", "heard", "felt", "noticed", "realized", "realised", "watched", "seemed", "wondered",
          "thought", "knew", "decided", "observed", "sensed", "could see", "could hear", "could feel", "looked"}

# Emotions stated outright instead of shown.
_TELLING = re.compile(
    r"\b(felt|was|were|is|are|became|grew)\s+(very\s+|so\s+|really\s+)?"
    r"(sad|angry|afraid|scared|frightened|happy|excited|nervous|anxious|furious|terrified|miserable|"
    r"delighted|embarrassed|ashamed|jealous|lonely|calm|worried|upset|relieved|confident|proud)\b", re.I)
_TELLING_SURGE = re.compile(r"\b(a\s+)?(surge|wave|rush|flood|pang|wave)\s+of\s+(fear|anger|joy|sadness|relief|dread|guilt|love|rage)\b", re.I)

_ABSTRACT = re.compile(r"\b(somehow|something|some kind of|a sense of|a feeling of|indescribable|unexplainable)\b", re.I)

_TIME_DAY = {"dawn", "sunrise", "morning", "noon", "midday", "afternoon", "daylight", "sunlight", "sunny", "sun"}
_TIME_NIGHT = {"midnight", "night", "nightfall", "dusk", "twilight", "stars", "starlight", "moon", "moonlight", "dark"}
_TRANSITION = {"later", "hours", "by the time", "that evening", "that night", "next morning", "the next", "afterward",
               "afterwards", "soon", "eventually", "then", "after a while", "days passed", "by nightfall", "by dawn"}
_SEASONS = {"spring", "summer", "autumn", "fall", "winter"}


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in words(text)]


@dataclass
class SensoryProfile:
    senses_engaged: int
    per_sense: dict[str, float]
    concrete_specifics: int        # proper-noun-ish tokens + digits: named, particular things
    telling_emotions: int
    filter_density: float          # filter verbs per sentence
    abstract_hits: int
    issues: list[str] = field(default_factory=list)


def sensory_profile(text: str) -> SensoryProfile:
    toks = _tokens(text)
    n = max(1, len(toks))
    sents = sentences(text) or [""]
    per_sense = {}
    for sense, lex in SENSES.items():
        per_sense[sense] = round(sum(1 for t in toks if t in lex) / n, 4)
    engaged = sum(1 for s, d in per_sense.items() if d > 0)
    # Named specifics: capitalised words mid-sentence (not the first word) and numbers.
    specifics = len(re.findall(r"(?<=[a-z,;:]\s)[A-Z][a-z]{2,}", text)) + len(re.findall(r"\b\d+\b", text))
    telling = len(_TELLING.findall(text)) + len(_TELLING_SURGE.findall(text))
    filters = sum(1 for t in toks if t in FILTER)
    filter_density = round(filters / len(sents), 3)
    abstract = len(_ABSTRACT.findall(text))
    issues: list[str] = []
    wc = len(toks)
    if wc >= 120:
        if engaged < 2:
            issues.append("the scene engages few senses; add sound, smell or touch, not just sight")
        if telling >= 2:
            issues.append(f"emotion is told, not shown ({telling}x, e.g. 'was afraid'); dramatise it through action and body")
        if filter_density > 0.5:
            issues.append(f"filter words distance the reader ({filter_density:.2f}/sentence: saw/heard/felt); cut them and show the thing directly")
        if specifics < wc / 120:
            issues.append("detail is generic; name particular things (a brand, a street, an exact object) instead of 'a car', 'a building'")
        if abstract >= 2:
            issues.append("vague abstraction ('something', 'a sense of'); replace with the concrete particular")
    return SensoryProfile(engaged, per_sense, specifics, telling, filter_density, abstract, issues)


def sensory_score(p: SensoryProfile) -> float:
    s = 0.0
    s += min(1.0, p.senses_engaged / 3) * 0.4
    s += (1.0 if p.telling_emotions == 0 else max(0.0, 1.0 - 0.2 * p.telling_emotions)) * 0.25
    s += (1.0 if p.filter_density <= 0.5 else max(0.0, 1.0 - (p.filter_density - 0.5))) * 0.2
    s += (1.0 if not p.issues or p.abstract_hits == 0 else 0.5) * 0.15
    return round(s, 3)


@dataclass
class LogisticsCheck:
    time_of_day: str               # "day" | "night" | "mixed" | "unknown"
    seasons: list[str]
    issues: list[str] = field(default_factory=list)


def logistics_check(text: str) -> LogisticsCheck:
    toks = set(_tokens(text))
    low = text.lower()
    has_day = bool(toks & _TIME_DAY)
    has_night = bool(toks & _TIME_NIGHT)
    has_transition = any(t in low for t in _TRANSITION)
    seasons = sorted(toks & _SEASONS)
    issues: list[str] = []
    tod = "unknown"
    if has_day and has_night:
        tod = "mixed"
        if not has_transition:
            issues.append("time of day is inconsistent within the chapter (daylight and night cues with no passage of time); add a transition or pick one")
    elif has_day:
        tod = "day"
    elif has_night:
        tod = "night"
    if len(seasons) >= 2:
        issues.append(f"more than one season is referenced ({', '.join(seasons)}); keep the season consistent")
    return LogisticsCheck(tod, seasons, issues)
