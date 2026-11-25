"""The character blueprint: everything a newly built character needs, as something that can be checked.

A character sheet with a name, a job and "brave but stubborn" is not a character. This module writes
down what a finished one has, in nine sections, and for each item says what it is and the test it must
pass. The same list drives four things, so they cannot drift apart:

* ``breakdown_markdown()`` - the breakdown, to read;
* ``blank_sheet()`` - a fill-in sheet an author completes to teach Gauge by example;
* ``parse_sheet()`` / ``sheet_text()`` - reading a filled sheet, and writing a character back out as one;
* ``audit()`` - which items a given character has, which are missing and which are too thin to count.

Minor characters need less: each item names the roles it is required for.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .characters import TRAIT_AXES, Character, character_from_sheet

MAJOR = ("protagonist", "antagonist", "supporting")
EVERYONE = ("protagonist", "antagonist", "supporting", "minor")
LEADS = ("protagonist", "antagonist")


@dataclass(frozen=True)
class Item:
    key: str                 # the line's name on a sheet, and the lookup key
    label: str
    what: str                # what it is
    test: str                # the test a good answer passes
    roles: tuple[str, ...] = MAJOR
    kind: str = "text"       # text | list | axes | relationships | flag
    min_words: int = 3       # for text: fewer words than this is "thin"
    min_items: int = 1       # for lists


SECTIONS: list[tuple[str, str, list[Item]]] = [
    ("Identity", "Who this is in one breath. If this section is vague, everything after it will be too.", [
        Item("name", "Name", "What they are called.",
             "Pronounceable, distinct from the rest of the cast (no two names starting the same way).", EVERYONE, min_words=1),
        Item("aliases", "Other names", "What different people call them: a nickname, a title, a surname.",
             "Each alias says something about who uses it.", LEADS, "list"),
        Item("role", "Role in the story", "Protagonist, antagonist, supporting or minor.",
             "One of the four; it decides how much of the rest is needed.", EVERYONE, min_words=1),
        Item("concept", "Concept", "The character in one vivid line.",
             "A specific person in a specific bind, not a job title: 'a courier who owes the wrong people'.", EVERYONE, min_words=4),
    ]),
    ("Outside", "What a stranger would notice. Readers remember one detail, so make it a particular one.", [
        Item("appearance", "Appearance", "How they look, in two concrete sentences.",
             "Contains one detail nobody else in the cast has, and nothing a police sketch would say.", MAJOR, min_words=8),
        Item("habits", "Habits and tells", "What their body does: a ritual, a nervous tell, how they take up a room.",
             "Could be filmed. Something a scene can show instead of naming a feeling.", MAJOR, "list"),
    ]),
    ("Past", "Only the past that still acts on the present. Not a biography.", [
        Item("backstory", "Backstory", "The few events that made them who they are now.",
             "Every sentence explains something they do today. Holds one thing they hide.", MAJOR, min_words=15),
        Item("wound", "Wound", "The specific hurt they have not got over.",
             "An event, not an adjective: 'her mother left without a word', not 'abandonment issues'.", MAJOR, min_words=4),
    ]),
    ("Engine", "What makes them act. This is the section a plot is built from; it has to hold tension.", [
        Item("want", "Want", "The concrete thing they are trying to get.",
             "A camera could see them get it. It is what they would tell you if you asked.", MAJOR, min_words=3),
        Item("need", "Need", "What would actually mend them.",
             "Different from the want, and in its way. They would deny it.", MAJOR, min_words=3),
        Item("lie", "The lie they believe", "The false belief they live by.",
             "The wound taught it; the flaw protects it; the arc is about losing it.", LEADS, min_words=4),
        Item("flaw", "Flaw", "How they get in their own way.",
             "Causes real trouble on the page. Not a charming quirk, not a strength in disguise.", MAJOR, min_words=3),
        Item("stakes", "Stakes", "What they lose if they fail.",
             "Personal and specific. Said in words they would use.", LEADS, min_words=4),
        Item("fears", "Fears", "What they are afraid of.", "At least one they would never admit.", MAJOR, "list"),
        Item("values", "Values", "What they hold to.", "Two of them should be able to collide.", MAJOR, "list", min_items=2),
        Item("lines", "Lines they will not cross", "What they refuse to do, whatever it costs.",
             "The story should be able to push them right up to one.", MAJOR, "list"),
    ]),
    ("Temperament", "The fifteen trait values from 0 to 1. A character with every dial at the middle is nobody.", [
        Item("traits", "Trait values", "Openness, conscientiousness, extraversion and the rest, each 0 to 1.",
             "At least three sit at an extreme (0.2 or lower, 0.8 or higher), and two of the extremes pull against each other.",
             MAJOR, "axes"),
        Item("contradiction", "Contradiction", "The thing about them that does not fit.",
             "Surprising and true: the enforcer who tends orchids. It makes them a person, not a type.", LEADS, min_words=4),
    ]),
    ("Voice", "How they sound. Cover the names in a page of dialogue and you should still know who is speaking.", [
        Item("register", "Register", "Formal, casual, clipped, florid, street, archaic...", "One word a reader would agree with.",
             MAJOR, min_words=1),
        Item("sentence_length", "Sentence length", "Short, medium or long.", "Matches how they think.", MAJOR, min_words=1),
        Item("vocabulary", "Vocabulary", "Where their words come from.",
             "Names a source (a trade, a region, a book) and something they never say.", MAJOR, min_words=3),
        Item("tics", "Verbal tics", "Habits of speech.", "Tied to a feeling: what they do when scared, lying, cornered.", MAJOR, "list"),
        Item("catchphrases", "Catchphrases", "Lines that are theirs.", "Rare. One is plenty.", LEADS, "list"),
        Item("swears", "Swears", "Yes or no.", "Decided on purpose.", EVERYONE, "flag"),
        Item("humor_style", "Humour", "How they are funny, or fail to be.", "Deflecting, dry, cruel, none...", MAJOR, min_words=1),
        Item("avoids", "Will not talk about", "Subjects they steer away from.", "At least one leads back to the wound.", MAJOR, "list"),
    ]),
    ("Capability", "What they can do, so that problems are solved by who they are and not by luck.", [
        Item("skills", "Skills", "What they are good at.",
             "Each comes with a limit or a price. Something they are bad at is worth more than another talent.", MAJOR, "list"),
    ]),
    ("Situation", "Where they stand when the story finds them.", [
        Item("mood", "Mood", "Their state at the start.", "One word, not 'fine'.", MAJOR, min_words=1),
        Item("goal", "Goal right now", "What they are doing this week.", "Smaller than the want, and on the way to it.", MAJOR, min_words=3),
        Item("knows", "Knows", "Facts they hold that matter to the plot.", "At least one that someone else wants.", LEADS, "list"),
        Item("secrets", "Secrets", "What they are hiding, and from whom.", "Would change a relationship if it came out.", MAJOR, "list"),
    ]),
    ("Connections and change", "Nobody is a character alone, and nobody interesting stays the same.", [
        Item("relationships", "Relationships", "How they stand with the other people in the cast.",
             "A line of stance for each: 'older brother she resents and would die for'. At least one is unresolved.",
             MAJOR, "relationships"),
        Item("arc", "Arc", "Where they start, what presses on them, and where it could take them.",
             "Names the change, and what it would cost. Could also end badly.", LEADS, min_words=8),
    ]),
]

ITEMS: dict[str, Item] = {it.key: it for _title, _why, items in SECTIONS for it in items}
# Other spellings an author might use on a sheet.
ALIASES = {"wants": "want", "needs": "need", "lines_wont_cross": "lines", "lines they will not cross": "lines",
           "will not": "lines", "avoids_talking_about": "avoids", "will not talk about": "avoids", "tone": "register",
           "humour": "humor_style", "humor": "humor_style", "other names": "aliases", "nicknames": "aliases",
           "habits and tells": "habits", "tells": "habits", "the lie they believe": "lie", "misbelief": "lie",
           "verbal tics": "tics", "goal right now": "goal", "sentence length": "sentence_length", "trait values": "traits"}


# ---------------------------------------------------------------------------- reading a character
def value_of(c: Character, key: str) -> Any:
    """The value a blueprint item refers to on a character."""
    m, v, st = c.matrix, c.voice, c.state
    table: dict[str, Any] = {
        "name": c.name, "aliases": c.aliases, "role": c.role, "concept": c.concept, "appearance": c.appearance,
        "habits": c.habits, "backstory": c.backstory, "wound": m.wound, "want": m.wants, "need": m.needs, "lie": c.lie,
        "flaw": m.flaw, "stakes": c.stakes, "fears": m.fears, "values": m.values, "lines": m.lines_wont_cross,
        "traits": m.axes(), "contradiction": c.contradiction, "register": v.tone, "sentence_length": v.sentence_length,
        "vocabulary": v.vocabulary, "tics": v.tics, "catchphrases": v.catchphrases, "swears": v.swears,
        "humor_style": v.humor_style, "avoids": v.avoids_talking_about, "skills": c.skills, "mood": st.mood,
        "goal": st.goal, "knows": st.knows, "secrets": st.secrets,
        "relationships": {o: r.stance for o, r in c.relationships.items()}, "arc": c.arc,
    }
    return table[key]


def _words(s: Any) -> int:
    return len(str(s or "").split())


_PLACEHOLDER = {"", "steady", "casual", "medium", "fine", "unknown", "none", "n/a", "tbd", "?"}


@dataclass
class Finding:
    key: str
    label: str
    status: str            # ok | thin | missing
    note: str = ""


@dataclass
class Audit:
    name: str
    role: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def required(self) -> list[Finding]:
        return self.findings

    @property
    def score(self) -> float:
        """Share of the required items that are properly filled; a thin one counts for half."""
        if not self.findings:
            return 1.0
        got = sum(1.0 if f.status == "ok" else 0.5 if f.status == "thin" else 0.0 for f in self.findings)
        return round(got / len(self.findings), 3)

    @property
    def missing(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "missing"]

    @property
    def thin(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "thin"]

    def gaps(self) -> list[str]:
        return [f.key for f in self.findings if f.status != "ok"]

    def summary(self) -> str:
        done = sum(1 for f in self.findings if f.status == "ok")
        line = f"{self.name} ({self.role}): {done} of {len(self.findings)} items complete ({self.score:.0%})"
        if self.missing:
            line += "; missing: " + ", ".join(f.label for f in self.missing[:8])
        if self.thin:
            line += "; too thin: " + ", ".join(f.label for f in self.thin[:6])
        return line


def audit(c: Character, cast_size: int = 0) -> Audit:
    """Check a character against the blueprint for its role. `cast_size` is how many other characters
    exist: a relationship is only required when there is someone to have one with."""
    role = c.role if c.role in EVERYONE else "supporting"
    out = Audit(c.name, role)
    for _title, _why, items in SECTIONS:
        for it in items:
            if role not in it.roles:
                continue
            v = value_of(c, it.key)
            status, note = "ok", ""
            if it.kind == "flag":
                pass                                              # yes or no are both answers
            elif it.kind == "list":
                real = [x for x in (v or []) if str(x).strip()]
                if not real:
                    status = "missing"
                elif len(real) < it.min_items:
                    status, note = "thin", f"needs at least {it.min_items}"
            elif it.kind == "axes":
                vals = list((v or {}).values())
                extremes = sum(1 for x in vals if x <= 0.2 or x >= 0.8)
                if not vals or all(abs(x - 0.5) < 1e-9 for x in vals):
                    status = "missing"
                elif extremes < 3:
                    status, note = "thin", f"only {extremes} extreme value(s); a character needs at least three"
            elif it.kind == "relationships":
                rels = {o: s for o, s in (v or {}).items()}
                if not rels:
                    status = "missing" if cast_size > 0 else "ok"
                elif not any(str(s).strip() for s in rels.values()):
                    status, note = "thin", "the relationships have numbers but no stance in words"
            else:
                text = str(v or "").strip()
                if text.lower() in _PLACEHOLDER and it.key not in ("name", "role", "sentence_length", "register", "humor_style"):
                    status = "missing"
                elif not text:
                    status = "missing"
                elif _words(text) < it.min_words:
                    status, note = "thin", f"{_words(text)} word(s); say more"
            out.findings.append(Finding(it.key, it.label, status, note))
    # The one cross-check that matters most: a want and a need that are the same thing give no story.
    w, n = c.matrix.wants.strip().lower(), c.matrix.needs.strip().lower()
    if w and n and (w == n or w in n or n in w):
        for f in out.findings:
            if f.key == "need":
                f.status, f.note = "thin", "the need repeats the want; it should be different and in its way"
    return out


# ---------------------------------------------------------------------------- the breakdown, to read
def breakdown_markdown() -> str:
    lines = ["# What a newly built character needs", "",
             "Nine sections. Each item says what it is and the test a good answer passes. Leads (the protagonist and "
             "the antagonist) need all of it; supporting characters most of it; minor characters only what is marked.", ""]
    for title, why, items in SECTIONS:
        lines += [f"## {title}", "", why, ""]
        for it in items:
            who = "everyone" if it.roles == EVERYONE else "leads" if it.roles == LEADS else "leads and supporting"
            lines.append(f"- **{it.label}** ({who}). {it.what} *Test:* {it.test}")
        lines.append("")
    return "\n".join(lines)


def checklist(role: str = "supporting") -> str:
    """The blueprint as a compact checklist for a prompt: what this role needs, and the test for each."""
    role = role if role in EVERYONE else "supporting"
    out = []
    for title, _why, items in SECTIONS:
        need = [it for it in items if role in it.roles and it.kind != "flag"]
        if need:
            out.append(f"{title}: " + " ".join(f"{it.label.lower()} ({it.test.rstrip('.')})." for it in need))
    return "\n".join(out)


# ---------------------------------------------------------------------------- the fill-in sheet
SHEET_HEADER = "# Gauge character sheet"


def blank_sheet(name: str = "", role: str = "supporting") -> str:
    """A sheet for an author to fill in. One `key: value` per line; lists are separated by `;`; lines
    starting with `#` are notes and are ignored when the sheet is read."""
    lines = [SHEET_HEADER,
             "# Fill in what you can. One 'key: value' per line. Separate the items of a list with ;",
             "# Lines starting with # are notes for you; Gauge ignores them.",
             "# A complete, specific sheet teaches Gauge more than ten vague ones.", ""]
    for title, why, items in SECTIONS:
        lines += [f"## {title}", f"# {why}"]
        for it in items:
            lines.append(f"# {it.label}: {it.what} Test: {it.test}")
            if it.key == "name":
                lines.append(f"name: {name}")
            elif it.key == "role":
                lines.append(f"role: {role}")
            elif it.kind == "axes":
                lines.append("# Give each a value from 0 to 1 (0 = not at all, 1 = extremely). Leave a line out to keep it at 0.5.")
                lines += [f"{axis}: " for axis in TRAIT_AXES]
            elif it.kind == "relationships":
                lines.append("# One line per person:   relationship: Name = how they stand with them")
                lines.append("# Optional, after a | :   trust 0.9, affection 0.8, respect 0.6, fear 0.1")
                lines.append("relationship: ")
            elif it.kind == "flag":
                lines.append(f"{it.key}: no")
            else:
                lines.append(f"{it.key}: ")
        lines.append("")
    return "\n".join(lines)


def _split_list(v: str) -> list[str]:
    return [x.strip() for x in re.split(r"[;\n]", v) if x.strip()]


def _axis_value(v: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", v)
    if not m:
        words = {"very low": 0.05, "low": 0.2, "medium": 0.5, "middle": 0.5, "high": 0.8, "very high": 0.95}
        return words.get(v.strip().lower())
    x = float(m.group(0))
    if x > 10:
        x /= 100.0            # written as a percentage
    elif x > 1:
        x /= 10.0             # written out of ten
    return max(0.0, min(1.0, x))


def parse_sheet(text: str) -> dict[str, Any]:
    """Read a filled-in sheet (or a JSON character) into the dict `character_from_sheet` understands."""
    stripped = text.lstrip("﻿").strip()
    if stripped.startswith("{"):
        d = json.loads(stripped)
        return d if isinstance(d, dict) else {}
    flat: dict[str, str] = {}
    rels: dict[str, dict[str, Any]] = {}
    axes: dict[str, float] = {}
    current: Optional[str] = None
    for raw in stripped.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            current = None if not line.strip() else current
            continue
        m = re.match(r"^\s*([A-Za-z][A-Za-z _'-]{0,40}?)\s*:\s*(.*)$", line)
        key = m.group(1).strip().lower() if m else None
        if key in TRAIT_AXES and re.match(r"\s*-?\d", m.group(2)):
            pass                                        # "humor: 0.7" is the trait; "humor: dry" is the style
        elif key:
            key = ALIASES.get(key, key.replace(" ", "_"))
        if m and key in ("relationship", "relationships"):
            body = m.group(2).strip()
            if "=" in body:
                who, _, rest = body.partition("=")
                stance, _, nums = rest.partition("|")
                if who.strip():
                    rel: dict[str, Any] = {"stance": stance.strip()}
                    for dim, raw in re.findall(r"(trust|affection|respect|fear)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", nums.lower()):
                        rel[dim] = _axis_value(raw)
                    rels[who.strip()] = rel
            current = None
        elif m and key in TRAIT_AXES:
            val = _axis_value(m.group(2))
            if val is not None:
                axes[key] = val
            current = None
        elif m and key in ITEMS:
            flat[key] = m.group(2).strip()
            current = key
        elif current:                                   # a continuation line of the last value
            flat[current] = (flat[current] + " " + line.strip()).strip()
    get = flat.get
    lst = lambda k: _split_list(get(k, ""))  # noqa: E731
    matrix: dict[str, Any] = dict(axes)
    matrix.update({"wants": get("want", ""), "needs": get("need", ""), "wound": get("wound", ""), "flaw": get("flaw", ""),
                   "values": lst("values"), "fears": lst("fears"), "lines_wont_cross": lst("lines")})
    swears = get("swears", "").strip().lower() in ("yes", "y", "true", "1", "often", "sometimes")
    return {
        "name": get("name", ""), "aliases": lst("aliases"), "role": (get("role", "") or "supporting").lower().split()[0].strip(".,|"),
        "concept": get("concept", ""), "appearance": get("appearance", ""), "backstory": get("backstory", ""),
        "habits": lst("habits"), "skills": lst("skills"), "contradiction": get("contradiction", ""), "lie": get("lie", ""),
        "stakes": get("stakes", ""), "arc": get("arc", ""), "matrix": matrix,
        "voice": {"register": get("register", "") or "casual", "sentence_length": get("sentence_length", "") or "medium",
                  "vocabulary": get("vocabulary", ""), "tics": lst("tics"), "catchphrases": lst("catchphrases"), "swears": swears,
                  "humor_style": get("humor_style", ""), "avoids_talking_about": lst("avoids")},
        "state": {"mood": get("mood", "") or "steady", "goal": get("goal", ""), "knows": lst("knows"), "secrets": lst("secrets")},
        "relationships": rels,
    }


def character_from_text(text: str) -> Character:
    """A filled-in sheet, or a JSON character, as a Character."""
    d = parse_sheet(text)
    if "matrix" in d and "voice" in d and isinstance(d.get("state"), dict) and "mood_vec" in d["state"]:
        return Character(**d)                           # a character file Gauge wrote itself
    name = str(d.get("name") or "").strip()
    if not name:
        raise ValueError("the sheet has no name")
    return character_from_sheet(name, d, str(d.get("role") or "supporting"), str(d.get("concept") or ""))


def sheet_text(c: Character, notes: bool = False) -> str:
    """A character written out as a fill-in sheet: the same format `parse_sheet` reads."""
    lines = [SHEET_HEADER]
    for title, _why, items in SECTIONS:
        lines += ["", f"## {title}"]
        for it in items:
            v = value_of(c, it.key)
            if notes:
                lines.append(f"# {it.label}: {it.test}")
            if it.kind == "axes":
                lines += [f"{axis}: {val:.2f}" for axis, val in v.items()]
            elif it.kind == "relationships":
                for who, r in c.relationships.items():
                    nums = f"trust {r.trust:.2f}, affection {r.affection:.2f}, respect {r.respect:.2f}, fear {r.fear:.2f}"
                    lines.append(f"relationship: {who} = {r.stance} | {nums}")
            elif it.kind == "flag":
                lines.append(f"{it.key}: {'yes' if v else 'no'}")
            elif it.kind == "list":
                lines.append(f"{it.key}: {'; '.join(str(x) for x in v)}")
            else:
                lines.append(f"{it.key}: {v}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- the sheet as JSON
def sheet_json(c: Character) -> dict[str, Any]:
    """A character as the JSON object the creation prompt asks for: what an example looks like to the
    model, and the answer a training row teaches. Leaves out everything a story adds later (memories,
    mood history, the chapter they were created in)."""
    m, v, st = c.matrix, c.voice, c.state
    matrix: dict[str, Any] = {axis: round(val, 2) for axis, val in m.axes().items()}
    matrix.update({"values": list(m.values), "fears": list(m.fears), "wants": m.wants, "needs": m.needs, "wound": m.wound,
                   "flaw": m.flaw, "lines_wont_cross": list(m.lines_wont_cross)})
    return {
        "name": c.name, "aliases": list(c.aliases), "role": c.role, "concept": c.concept, "appearance": c.appearance,
        "backstory": c.backstory, "habits": list(c.habits), "skills": list(c.skills), "contradiction": c.contradiction,
        "lie": c.lie, "stakes": c.stakes, "arc": c.arc, "matrix": matrix,
        "voice": {"register": v.tone, "sentence_length": v.sentence_length, "vocabulary": v.vocabulary, "tics": list(v.tics),
                  "catchphrases": list(v.catchphrases), "swears": v.swears, "humor_style": v.humor_style,
                  "avoids_talking_about": list(v.avoids_talking_about)},
        "state": {"mood": st.mood, "goal": st.goal, "knows": list(st.knows), "secrets": list(st.secrets)},
        "relationships": {o: {"trust": round(r.trust, 2), "affection": round(r.affection, 2), "respect": round(r.respect, 2),
                              "fear": round(r.fear, 2), "stance": r.stance} for o, r in c.relationships.items()},
    }


def _listy(v: Any, cap: int = 6) -> list[str]:
    if isinstance(v, str):
        v = _split_list(v)
    return [str(x).strip() for x in (v or []) if str(x).strip()][:cap] if isinstance(v, (list, tuple)) else []


def apply_item(c: Character, key: str, value: Any) -> bool:
    """Set one blueprint item on a character from a loosely typed value. True when something was set."""
    from .characters import Relationship

    it = ITEMS.get(ALIASES.get(key, key))
    if it is None or value in (None, "", [], {}):
        return False
    k = it.key
    if it.kind == "axes":
        if not isinstance(value, dict):
            return False
        done = False
        for axis, raw in value.items():
            if axis in TRAIT_AXES:
                val = _axis_value(str(raw))
                if val is not None:
                    setattr(c.matrix, axis, val)
                    done = True
        return done
    if it.kind == "relationships":
        if not isinstance(value, dict):
            return False
        for who, raw in value.items():
            stance = str(raw.get("stance", "") if isinstance(raw, dict) else raw).strip()
            if not stance or str(who).strip().lower() == c.name.lower():
                continue
            rel = c.relationships.get(str(who)) or Relationship()
            rel.stance = stance
            c.relationships[str(who)] = rel
        return bool(c.relationships)
    if it.kind == "flag":
        c.voice.swears = bool(value) if not isinstance(value, str) else value.strip().lower() in ("yes", "y", "true", "1")
        return True
    if it.kind == "list":
        items = _listy(value)
        if not items:
            return False
        target = {"aliases": (c, "aliases"), "habits": (c, "habits"), "skills": (c, "skills"), "fears": (c.matrix, "fears"),
                  "values": (c.matrix, "values"), "lines": (c.matrix, "lines_wont_cross"), "tics": (c.voice, "tics"),
                  "catchphrases": (c.voice, "catchphrases"), "avoids": (c.voice, "avoids_talking_about"),
                  "knows": (c.state, "knows"), "secrets": (c.state, "secrets")}[k]
        setattr(target[0], target[1], items)
        return True
    text = " ".join(str(x) for x in value) if isinstance(value, (list, tuple)) else str(value).strip()
    if not text:
        return False
    target = {"name": (c, "name"), "role": (c, "role"), "concept": (c, "concept"), "appearance": (c, "appearance"),
              "backstory": (c, "backstory"), "wound": (c.matrix, "wound"), "want": (c.matrix, "wants"),
              "need": (c.matrix, "needs"), "lie": (c, "lie"), "flaw": (c.matrix, "flaw"), "stakes": (c, "stakes"),
              "contradiction": (c, "contradiction"), "register": (c.voice, "tone"),
              "sentence_length": (c.voice, "sentence_length"), "vocabulary": (c.voice, "vocabulary"),
              "humor_style": (c.voice, "humor_style"), "mood": (c.state, "mood"), "goal": (c.state, "goal"),
              "arc": (c, "arc")}[k]
    if k in ("name", "role"):
        return False                                    # a repair never renames or recasts
    setattr(target[0], target[1], text)
    return True


def repair_request(a: Audit) -> list[tuple[Item, str]]:
    """The items an audit found wanting, each with what to ask for."""
    out = []
    for f in a.findings:
        if f.status == "ok" or f.key in ("name", "role"):
            continue
        it = ITEMS[f.key]
        shape = {"list": "a list of short strings", "axes": "an object of trait name to a value from 0 to 1",
                 "relationships": "an object of other character name to one line of stance"}.get(it.kind, "a string")
        why = "missing" if f.status == "missing" else f"too thin ({f.note})" if f.note else "too thin"
        out.append((it, f"\"{it.key}\": {shape}. {it.what} Test: {it.test} [{why}]"))
    return out
