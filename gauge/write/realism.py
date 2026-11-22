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


# ---------------------------------------------------------------------------- explicit clock times (roadmap 27)

_CLOCK_12 = re.compile(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*([ap])\.?\s?m\b\.?", re.I)
_CLOCK_24 = re.compile(r"\b(?:at|by|around|until|till|past|before|after|nearly|almost|was|read|said)\s+([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*[ap]\.?\s?m\b)", re.I)
_CLOCK_WORD = re.compile(r"\b(noon|midday|midnight)\b", re.I)
_QUOTED = re.compile(r"[\"\u201c][^\"\u201c\u201d]{0,600}[\"\u201d]")
# Words that explain a clock running backwards: time passed into another day, or the text looked back.
_CLOCK_EXCUSE = re.compile(
    r"\b(next\s+(day|morning|evening|night)|the\s+following|tomorrow|days?\s+later|weeks?\s+later|hours?\s+later|"
    r"later\s+that|by\s+the\s+time|the\s+day\s+before|yesterday|earlier|ago|had\s+been|remembered|that\s+morning|"
    r"last\s+night|the\s+night\s+before|overnight|slept|woke|dawn\s+came|until\s+morning)\b", re.I)
_SUN_CUES = {"sunlight", "sunny", "sunshine", "daylight", "sunlit"}
_MOON_CUES = {"moonlight", "moonlit", "starlight", "stars"}


@dataclass
class ClockCheck:
    times: list[tuple[int, str]]        # (minutes after midnight, the phrase), in narrative order
    issues: list[str] = field(default_factory=list)


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _say(phrase: str, minutes: int) -> str:
    return phrase if phrase == _fmt(minutes) else f"{phrase} ({_fmt(minutes)})"


def clock_times(text: str) -> list[tuple[int, str, int]]:
    """Explicit clock times in narration, in order: (minutes after midnight, phrase, position). Times
    spoken in dialogue are skipped, since a character naming a time is not the scene's clock."""
    narration = _QUOTED.sub(lambda m: " " * len(m.group(0)), text)   # keep positions stable
    found: list[tuple[int, str, int]] = []
    for m in _CLOCK_12.finditer(narration):
        h, mins, half = int(m.group(1)) % 12, int(m.group(2) or 0), m.group(3).lower()
        found.append(((h + (12 if half == "p" else 0)) * 60 + mins, m.group(0).strip(), m.start()))
    for m in _CLOCK_24.finditer(narration):
        found.append((int(m.group(1)) * 60 + int(m.group(2)), f"{m.group(1)}:{m.group(2)}", m.start(1)))
    for m in _CLOCK_WORD.finditer(narration):
        found.append((0 if m.group(1).lower() == "midnight" else 12 * 60, m.group(1).lower(), m.start()))
    found.sort(key=lambda t: t[2])
    return found


def clock_check(text: str) -> ClockCheck:
    """Flag a clock that runs backwards with nothing to explain it, and a stated time that contradicts the
    light in its own sentence (sunlight at 2 a.m., moonlight at noon)."""
    found = clock_times(text)
    issues: list[str] = []
    for (t1, p1, pos1), (t2, p2, pos2) in zip(found, found[1:]):
        if t2 >= t1:
            continue
        if (t2 + 1440 - t1) <= 180:                 # 11:50 p.m. then 12:10 a.m. is just midnight passing
            continue
        if _CLOCK_EXCUSE.search(text[pos1:pos2]):
            continue
        issues.append(f"the clock runs backwards: {_say(p1, t1)} and then {_say(p2, t2)} with no day change or look back between them")
    for t, phrase, pos in found:
        start = max(text.rfind(".", 0, pos), text.rfind("!", 0, pos), text.rfind("?", 0, pos)) + 1
        after = pos + len(phrase)                    # past the dots inside "a.m."
        ends = [e for e in (text.find(c, after) for c in ".!?") if e != -1]
        sentence = set(_tokens(text[start:min(ends) if ends else len(text)]))
        if (t >= 23 * 60 or t < 4 * 60) and sentence & _SUN_CUES:
            issues.append(f"{_say(phrase, t)} but the same sentence has {', '.join(sorted(sentence & _SUN_CUES))}; the light contradicts the hour")
        elif 10 * 60 <= t <= 15 * 60 and sentence & _MOON_CUES:
            issues.append(f"{_say(phrase, t)} but the same sentence has {', '.join(sorted(sentence & _MOON_CUES))}; the light contradicts the hour")
    return ClockCheck([(t, p) for t, p, _ in found], issues[:4])
