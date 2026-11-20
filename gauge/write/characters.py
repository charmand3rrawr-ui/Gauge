"""Characters as independent agents.

Each character owns a **personality matrix** (trait axes, values, fears, want/need, wound, flaw,
hard lines), a **voice** fingerprint, a **relationship matrix** towards the other characters,
and a **state** (mood, goal, what they know, memories by chapter). All of it lives in
``projects/<slug>/characters/<name>.json`` and is readable in ``characters.md``.

A ``CharacterAgent`` wraps one character and the local LLM, with a system prompt that is *only*
that person: not the author, not the plot. When the planner puts a character in a scene the
agent is activated to produce a **scene brief** (what they want, feel, would do and say, what
they would refuse). The drafter writes with those briefs in hand; after drafting each present
character **reviews** the chapter for out-of-character moments, and then **remembers** it,
updating mood, knowledge and relationships for next time. You can also simply talk to them
(``gauge character chat``).
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..llm.client import LLM, generate_json
from ..text import tail
from . import prompts
from .schemas import BriefOut, MemoryOut, ReviewOut, SheetOut, schema_of

log = logging.getLogger(__name__)

TRAIT_AXES = (
    "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
    "dominance", "warmth", "impulsivity", "honesty", "courage", "humor", "ambition", "loyalty",
    "curiosity", "cynicism",
)


_NOT_NAMES = {"the", "mr", "mrs", "ms", "miss", "dr", "doctor", "sir", "lady", "lord", "master", "mister",
              "captain", "professor", "old", "young", "big", "little", "saint", "st", "king", "queen", "prince", "princess"}


def _clamp(v: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "character"


# --------------------------------------------------------------------------------------
# Personality matrix
# --------------------------------------------------------------------------------------
class PersonalityMatrix(BaseModel):
    """Fifteen 0..1 axes plus the narrative anchors that make a person a character."""

    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    dominance: float = 0.5
    warmth: float = 0.5
    impulsivity: float = 0.5
    honesty: float = 0.5
    courage: float = 0.5
    humor: float = 0.5
    ambition: float = 0.5
    loyalty: float = 0.5
    curiosity: float = 0.5
    cynicism: float = 0.5
    values: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    wants: str = ""  # conscious goal
    needs: str = ""  # what would actually fix them
    wound: str = ""
    flaw: str = ""
    lines_wont_cross: list[str] = Field(default_factory=list)

    @classmethod
    def random(cls, seed: Optional[int] = None) -> PersonalityMatrix:
        rng = random.Random(seed)
        # Beta(2,2) keeps most people near the middle with real outliers.
        return cls(**{axis: round(rng.betavariate(2, 2), 2) for axis in TRAIT_AXES})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PersonalityMatrix:
        d = d or {}
        kw: dict[str, Any] = {axis: _clamp(d.get(axis, 0.5)) for axis in TRAIT_AXES}
        for k in ("values", "fears", "lines_wont_cross"):
            v = d.get(k, [])
            kw[k] = [str(x) for x in v] if isinstance(v, list) else [s.strip() for s in str(v).split(";") if s.strip()]
        for k in ("wants", "needs", "wound", "flaw"):
            kw[k] = str(d.get(k, "") or "")
        return cls(**kw)

    def axes(self) -> dict[str, float]:
        return {a: getattr(self, a) for a in TRAIT_AXES}

    def tendencies(self) -> list[str]:
        """Deterministic behavioural rules derived from the axes. These are what make the
        character *independent*: they hold whatever the plot would prefer."""
        a = self.axes()
        hi, lo = 0.7, 0.3
        t: list[str] = []
        if a["impulsivity"] >= hi and a["conscientiousness"] <= 0.5:
            t.append("acts before planning; regrets later, rarely apologises first")
        if a["conscientiousness"] >= hi:
            t.append("prepares, keeps promises, notices when others cut corners")
        if a["dominance"] >= hi:
            t.append("takes charge of a room within a minute; interrupts; gives orders as suggestions")
        elif a["dominance"] <= lo:
            t.append("defers in groups, then acts alone; hates being the centre of attention")
        if a["warmth"] >= hi:
            t.append("notices other people's pain first and moves toward it")
        elif a["warmth"] <= lo:
            t.append("reads people accurately but does not comfort them")
        if a["honesty"] >= hi:
            t.append("cannot hold a lie for long; blunt when cornered")
        elif a["honesty"] <= lo:
            t.append("lies smoothly and small; tells the truth only when it costs nothing")
        if a["courage"] <= lo:
            t.append("freezes or flees under sudden threat, then is ashamed")
        elif a["courage"] >= hi:
            t.append("moves toward danger; underestimates it")
        if a["neuroticism"] >= hi:
            t.append("catastrophises; sleeps badly; shows stress in the hands and the voice")
        elif a["neuroticism"] <= lo:
            t.append("stays level when others panic; can seem cold")
        if a["humor"] >= hi:
            t.append("deflects with jokes, especially at the worst moment")
        if a["cynicism"] >= hi:
            t.append("assumes an angle behind every kindness")
        elif a["cynicism"] <= lo:
            t.append("extends trust too early and is hurt by it")
        if a["loyalty"] >= hi:
            t.append("will break rules for their people and knows it")
        if a["ambition"] >= hi:
            t.append("measures every situation by what it can get them")
        if a["curiosity"] >= hi:
            t.append("asks the question everyone else is avoiding")
        if a["extraversion"] <= lo:
            t.append("speaks in short bursts; needs silence to recover")
        if a["agreeableness"] <= lo:
            t.append("argues on principle; concedes rarely and never in public")
        if a["openness"] <= lo:
            t.append("distrusts the new; prefers the known cost")
        return t

    def describe(self) -> str:
        words = {0: "very low", 1: "low", 2: "middling", 3: "high", 4: "very high"}
        axes = ", ".join(f"{k} {words[min(4, int(v * 5))]}" for k, v in self.axes().items())
        parts = [f"Traits: {axes}."]
        if self.values:
            parts.append("Values: " + "; ".join(self.values) + ".")
        if self.fears:
            parts.append("Fears: " + "; ".join(self.fears) + ".")
        if self.wants:
            parts.append(f"Wants: {self.wants}")
        if self.needs:
            parts.append(f"Needs (but does not know it): {self.needs}")
        if self.wound:
            parts.append(f"Wound: {self.wound}")
        if self.flaw:
            parts.append(f"Flaw: {self.flaw}")
        if self.lines_wont_cross:
            parts.append("Will not: " + "; ".join(self.lines_wont_cross) + ".")
        tend = self.tendencies()
        if tend:
            parts.append("Tendencies: " + "; ".join(tend) + ".")
        return "\n".join(parts)


class Voice(BaseModel):
    tone: str = "casual"  # register: formal / casual / clipped / florid / street / archaic ...
    sentence_length: str = "medium"  # short / medium / long
    vocabulary: str = ""  # e.g. "mechanic's jargon, no abstractions"
    tics: list[str] = Field(default_factory=list)  # verbal habits
    catchphrases: list[str] = Field(default_factory=list)
    swears: bool = False
    humor_style: str = ""
    avoids_talking_about: list[str] = Field(default_factory=list)

    def describe(self) -> str:
        bits = [f"{self.tone} register, {self.sentence_length} sentences"]
        if self.vocabulary:
            bits.append(f"vocabulary: {self.vocabulary}")
        if self.tics:
            bits.append("tics: " + ", ".join(self.tics))
        if self.catchphrases:
            bits.append("catchphrases: " + " / ".join(f'"{c}"' for c in self.catchphrases))
        bits.append("swears" if self.swears else "does not swear")
        if self.humor_style:
            bits.append(f"humour: {self.humor_style}")
        if self.avoids_talking_about:
            bits.append("avoids: " + ", ".join(self.avoids_talking_about))
        return "; ".join(bits) + "."


class Relationship(BaseModel):
    trust: float = 0.5
    affection: float = 0.5
    respect: float = 0.5
    fear: float = 0.0
    stance: str = ""  # one line: "older brother she resents and would die for"
    history: list[str] = Field(default_factory=list)

    def describe(self, other: str) -> str:
        return (f"{other}: trust {self.trust:.1f}, affection {self.affection:.1f}, respect {self.respect:.1f}, "
                f"fear {self.fear:.1f}" + (f"; {self.stance}" if self.stance else "")
                + (f"; recent: {self.history[-1]}" if self.history else ""))


class Memory(BaseModel):
    chapter: int
    events: list[str] = Field(default_factory=list)  # from the character's own point of view
    feelings: str = ""


class CharacterState(BaseModel):
    mood: str = "steady"
    goal: str = ""  # current short-term goal
    location: str = ""
    knows: list[str] = Field(default_factory=list)  # facts this character knows
    secrets: list[str] = Field(default_factory=list)  # facts they hide from others
    condition: list[str] = Field(default_factory=list)  # injuries, resources, debts
    arc_progress: str = ""  # how far along their change they are
    last_seen_chapter: int = 0
    memories: list[Memory] = Field(default_factory=list)


class Character(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: str = "supporting"  # protagonist / antagonist / supporting / minor
    concept: str = ""  # one line
    appearance: str = ""
    backstory: str = ""
    matrix: PersonalityMatrix = Field(default_factory=PersonalityMatrix)
    voice: Voice = Field(default_factory=Voice)
    state: CharacterState = Field(default_factory=CharacterState)
    relationships: dict[str, Relationship] = Field(default_factory=dict)
    created_chapter: int = 0

    # ---- detection ---------------------------------------------------------
    def _pattern(self) -> re.Pattern:
        names = [self.name] + [a for a in self.aliases if a]
        parts = self.name.split()
        first = parts[0] if parts else self.name
        if (len(parts) > 1 and len(first) >= 3 and first not in names and first[0].isupper()
                and first.lower().strip(".") not in _NOT_NAMES):
            names.append(first)
        alts = "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))
        return re.compile(rf"(?<![A-Za-z])(?:{alts})(?![A-Za-z])")

    def mentioned_in(self, text: str) -> bool:
        return bool(text) and bool(self._pattern().search(text))

    def mentions(self, text: str) -> int:
        return len(self._pattern().findall(text or ""))

    # ---- prompts -----------------------------------------------------------
    def system_prompt(self) -> str:
        rel = "\n".join(f"- {r.describe(o)}" for o, r in self.relationships.items()) or "- (no one yet)"
        recent = self.state.memories[-3:]
        mem = "\n".join(f"- ch {m.chapter}: {'; '.join(m.events)}" + (f" ({m.feelings})" if m.feelings else "")
                        for m in recent) or "- (nothing yet)"
        return prompts.CHARACTER_SYSTEM.format(
            name=self.name, role=self.role, concept=self.concept or "(no concept)",
            backstory=self.backstory or "(unknown)", appearance=self.appearance or "(unspecified)",
            matrix=self.matrix.describe(), voice=self.voice.describe(),
            mood=self.state.mood, goal=self.state.goal or "(none stated)", location=self.state.location or "(unknown)",
            knows="; ".join(self.state.knows) or "(nothing recorded)",
            secrets="; ".join(self.state.secrets) or "(none)",
            condition="; ".join(self.state.condition) or "(fine)",
            arc=self.state.arc_progress or "(at the start)", relationships=rel, memories=mem,
        )

    def card(self) -> str:
        """Short description for the author-side prompts (drafter)."""
        return (f"{self.name} ({self.role}): {self.concept}. Voice: {self.voice.describe()} "
                f"Mood: {self.state.mood}. Wants: {self.matrix.wants or '?'}")

    def sheet_markdown(self) -> str:
        m = self.matrix
        lines = [f"## {self.name}" + (f" ({', '.join(self.aliases)})" if self.aliases else ""),
                 f"*{self.role}* - {self.concept}", "",
                 f"**Appearance:** {self.appearance}", f"**Backstory:** {self.backstory}", "",
                 "**Matrix:** " + ", ".join(f"{k} {v:.2f}" for k, v in m.axes().items()), "",
                 f"**Wants:** {m.wants}  \n**Needs:** {m.needs}  \n**Wound:** {m.wound}  \n**Flaw:** {m.flaw}",
                 "**Values:** " + "; ".join(m.values), "**Fears:** " + "; ".join(m.fears),
                 "**Will not:** " + "; ".join(m.lines_wont_cross), "",
                 "**Tendencies:**"] + [f"- {t}" for t in m.tendencies()] + ["",
                 f"**Voice:** {self.voice.describe()}", "",
                 f"**State:** mood {self.state.mood}; goal {self.state.goal}; location {self.state.location}; "
                 f"arc {self.state.arc_progress}; last seen ch {self.state.last_seen_chapter}",
                 "**Knows:** " + "; ".join(self.state.knows), "**Secrets:** " + "; ".join(self.state.secrets), ""]
        if self.relationships:
            lines += ["**Relationships:**"] + [f"- {r.describe(o)}" for o, r in self.relationships.items()] + [""]
        if self.state.memories:
            lines += ["**Memories:**"] + [f"- ch {x.chapter}: {'; '.join(x.events)}" for x in self.state.memories[-8:]] + [""]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Roster: persistence + who is in the scene
# --------------------------------------------------------------------------------------
class Roster:
    def __init__(self, project_dir: Path):
        self.dir = project_dir / "characters"
        self.characters: dict[str, Character] = {}

    @classmethod
    def load(cls, project_dir: Path) -> Roster:
        r = cls(project_dir)
        if r.dir.exists():
            for p in sorted(r.dir.glob("*.json")):
                try:
                    c = Character(**json.loads(p.read_text(encoding="utf-8")))
                    r.characters[_slug(c.name)] = c
                except Exception as e:  # a hand-edited file with a typo must not kill the run
                    log.warning("could not load character %s: %s", p, e)
        return r

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for slug, c in self.characters.items():
            (self.dir / f"{slug}.json").write_text(c.model_dump_json(indent=2), encoding="utf-8")
        md = ["# Characters", ""] + [c.sheet_markdown() for c in self.ordered()]
        (self.dir.parent / "characters.md").write_text("\n".join(md), encoding="utf-8")

    def add(self, c: Character) -> Character:
        self.characters[_slug(c.name)] = c
        # Make sure everyone has a relationship slot for everyone else.
        for other in self.characters.values():
            if other is c:
                continue
            c.relationships.setdefault(other.name, Relationship())
            other.relationships.setdefault(c.name, Relationship())
        return c

    def remove(self, name: str) -> bool:
        c = self.get(name)
        if c is None:
            return False
        del self.characters[_slug(c.name)]
        p = self.dir / f"{_slug(c.name)}.json"
        if p.exists():
            p.unlink()
        return True

    def get(self, name: str) -> Optional[Character]:
        s = _slug(name)
        if s in self.characters:
            return self.characters[s]
        low = name.lower()
        for c in self.characters.values():
            first = c.name.split()[0].lower() if c.name.split() else ""
            if low == c.name.lower() or low in (a.lower() for a in c.aliases) or (low == first and first not in _NOT_NAMES):
                return c
        return None

    def names(self) -> list[str]:
        return [c.name for c in self.ordered()]

    def ordered(self) -> list[Character]:
        rank = {"protagonist": 0, "antagonist": 1, "supporting": 2, "minor": 3}
        return sorted(self.characters.values(), key=lambda c: (rank.get(c.role, 9), c.name))

    def present_in(self, texts: list[str], limit: int = 5, always_protagonist: bool = True) -> list[Character]:
        """Characters named in the scene plan / previous tail, most-mentioned first."""
        joined = "\n".join(t for t in texts if t)
        scored = [(c.mentions(joined), c) for c in self.characters.values()]
        present = [c for n, c in sorted(scored, key=lambda t: -t[0]) if n > 0]
        if always_protagonist:
            for c in self.ordered():
                if c.role == "protagonist" and c not in present:
                    present.insert(0, c)
        return present[:limit]


# --------------------------------------------------------------------------------------
# Agent: the character thinking for themselves
# --------------------------------------------------------------------------------------
class SceneBrief(BaseModel):
    name: str
    goal: str = ""
    mood: str = ""
    actions: list[str] = Field(default_factory=list)
    dialogue: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    hides: list[str] = Field(default_factory=list)
    refuses: list[str] = Field(default_factory=list)
    conflict_with_plan: str = ""

    def render(self) -> str:
        out = [f"### {self.name}", f"- wants in this scene: {self.goal}", f"- mood: {self.mood}"]
        if self.actions:
            out.append("- would do: " + "; ".join(self.actions))
        if self.dialogue:
            out.append("- might say: " + " | ".join(f'"{d}"' for d in self.dialogue))
        if self.notices:
            out.append("- notices: " + "; ".join(self.notices))
        if self.hides:
            out.append("- hides: " + "; ".join(self.hides))
        if self.refuses:
            out.append("- will NOT: " + "; ".join(self.refuses))
        if self.conflict_with_plan:
            out.append(f"- friction with the plan: {self.conflict_with_plan}")
        return "\n".join(out)


class ReviewIssue(BaseModel):
    quote: str = ""
    problem: str = ""
    fix: str = ""


class CharacterReview(BaseModel):
    name: str
    consistency: float = 1.0  # 0..1
    issues: list[ReviewIssue] = Field(default_factory=list)


def _as_list(v: Any, limit: int = 6) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()][:limit]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


class CharacterAgent:
    def __init__(self, llm: LLM, character: Character):
        self.llm = llm
        self.c = character

    def brief(self, scene: dict[str, Any]) -> SceneBrief:
        prompt = prompts.SCENE_BRIEF_PROMPT.format(
            number=scene.get("number", "?"), title=scene.get("title", ""), goal=scene.get("goal", ""),
            hook=scene.get("hook", ""), turn=scene.get("turn", ""), cliffhanger=scene.get("cliffhanger", ""),
            beats="; ".join(scene.get("beats", [])), summary=scene.get("summary", "") or "(the story is beginning)",
            prev_tail=scene.get("prev_tail", "") or "(no previous chapter)",
            others=", ".join(scene.get("others", [])) or "(no one else named)",
        )
        try:
            d = generate_json(self.llm, prompt, system=self.c.system_prompt(), schema=schema_of(BriefOut),
                              temperature=0.7, max_tokens=600, purpose="character:brief")
        except Exception as e:
            log.warning("scene brief failed for %s: %s", self.c.name, e)
            d = {}
        b = SceneBrief(
            name=self.c.name, goal=str(d.get("goal", self.c.state.goal or self.c.matrix.wants)),
            mood=str(d.get("mood", self.c.state.mood)), actions=_as_list(d.get("actions")),
            dialogue=_as_list(d.get("dialogue"), 4), notices=_as_list(d.get("notices"), 4),
            hides=_as_list(d.get("hides"), 3), refuses=_as_list(d.get("refuses"), 4),
            conflict_with_plan=str(d.get("conflict_with_plan", "") or ""),
        )
        # Hard lines from the matrix are enforced even if the model forgot them.
        for line in self.c.matrix.lines_wont_cross:
            if line and line not in b.refuses:
                b.refuses.append(line)
        return b

    def review(self, chapter_text: str, number: int) -> CharacterReview:
        prompt = prompts.CHARACTER_REVIEW_PROMPT.format(number=number, name=self.c.name, text=chapter_text)
        try:
            d = generate_json(self.llm, prompt, system=self.c.system_prompt(), schema=schema_of(ReviewOut),
                              temperature=0.2, max_tokens=700, purpose="character:review")
        except Exception as e:
            log.warning("character review failed for %s: %s", self.c.name, e)
            return CharacterReview(name=self.c.name)
        issues = []
        for it in d.get("issues", []) if isinstance(d.get("issues"), list) else []:
            if isinstance(it, dict):
                issues.append(ReviewIssue(quote=str(it.get("quote", ""))[:200], problem=str(it.get("problem", ""))[:300],
                                          fix=str(it.get("fix", ""))[:300]))
        score = _clamp(d.get("consistency", 10), 0, 10, 10) / 10.0
        return CharacterReview(name=self.c.name, consistency=score, issues=issues[:5])

    def remember(self, chapter_text: str, number: int, roster: Optional[Roster] = None) -> None:
        prompt = prompts.CHARACTER_MEMORY_PROMPT.format(number=number, name=self.c.name, text=chapter_text,
                                                        others=", ".join(roster.names()) if roster else "")
        try:
            d = generate_json(self.llm, prompt, system=self.c.system_prompt(), schema=schema_of(MemoryOut),
                              temperature=0.2, max_tokens=700, purpose="character:memory")
        except Exception as e:
            log.warning("memory update failed for %s: %s; recording raw tail", self.c.name, e)
            d = {"events": [" ".join(tail(chapter_text, 40).split())]}
        st = self.c.state
        st.memories.append(Memory(chapter=number, events=_as_list(d.get("events"), 6), feelings=str(d.get("feelings", "") or "")))
        st.memories = st.memories[-40:]
        if d.get("mood"):
            st.mood = str(d["mood"])
        if d.get("goal"):
            st.goal = str(d["goal"])
        if d.get("location"):
            st.location = str(d["location"])
        if d.get("arc_progress"):
            st.arc_progress = str(d["arc_progress"])
        for fact in _as_list(d.get("learned"), 6):
            if fact not in st.knows:
                st.knows.append(fact)
        st.knows = st.knows[-40:]
        for fact in _as_list(d.get("new_secrets"), 3):
            if fact not in st.secrets:
                st.secrets.append(fact)
        for cond in _as_list(d.get("condition"), 5):
            if cond not in st.condition:
                st.condition.append(cond)
        st.condition = st.condition[-8:]
        st.last_seen_chapter = number
        rel_changes = d.get("relationships")
        if isinstance(rel_changes, dict):
            for other, ch in rel_changes.items():
                if not isinstance(ch, dict):
                    continue
                other_name = other
                if roster is not None:
                    oc = roster.get(str(other))
                    if oc is None or oc is self.c:
                        continue
                    other_name = oc.name
                rel = self.c.relationships.setdefault(other_name, Relationship())
                for k in ("trust", "affection", "respect", "fear"):
                    if k in ch:
                        setattr(rel, k, _clamp(getattr(rel, k) + _clamp(ch[k], -1, 1, 0.0)))
                if ch.get("stance"):
                    rel.stance = str(ch["stance"])
                if ch.get("note"):
                    rel.history.append(f"ch {number}: {ch['note']}")
                    rel.history = rel.history[-10:]

    def chat(self, question: str, history: Optional[list[tuple[str, str]]] = None) -> str:
        convo = "\n".join(f"Author: {q}\n{self.c.name}: {a}" for q, a in (history or []))
        prompt = prompts.CHARACTER_CHAT_PROMPT.format(name=self.c.name, history=convo or "(start of conversation)", question=question)
        return self.llm.generate(prompt, system=self.c.system_prompt(), temperature=0.8, max_tokens=500,
                                 purpose="character:chat").strip()


def render_cast(briefs: list[SceneBrief], roster: Roster) -> str:
    if not briefs:
        return "(no profiled characters in this scene)"
    out = []
    for b in briefs:
        c = roster.get(b.name)
        out.append((c.card() + "\n" if c else "") + b.render())
    return "\n\n".join(out)


# --------------------------------------------------------------------------------------
# Creation
# --------------------------------------------------------------------------------------
def character_from_sheet(name: str, d: dict[str, Any], role: str = "supporting", concept: str = "") -> Character:
    d = d or {}
    voice_d = d.get("voice") if isinstance(d.get("voice"), dict) else {}
    voice = Voice(
        tone=str(voice_d.get("register", voice_d.get("tone", "casual")) or "casual"),
        sentence_length=str(voice_d.get("sentence_length", "medium") or "medium"),
        vocabulary=str(voice_d.get("vocabulary", "") or ""), tics=_as_list(voice_d.get("tics"), 4),
        catchphrases=_as_list(voice_d.get("catchphrases"), 3), swears=bool(voice_d.get("swears", False)),
        humor_style=str(voice_d.get("humor_style", "") or ""), avoids_talking_about=_as_list(voice_d.get("avoids_talking_about"), 4),
    )
    matrix_d = dict(d.get("matrix") or {}) if isinstance(d.get("matrix"), dict) else {}
    for k in ("values", "fears", "wants", "needs", "wound", "flaw", "lines_wont_cross"):
        if k in d and k not in matrix_d:
            matrix_d[k] = d[k]
    state_d = d.get("state") if isinstance(d.get("state"), dict) else {}
    full_name = str(d.get("name") or name)
    aliases = [a for a in _as_list(d.get("aliases"), 6)
               if len(a) >= 3 and a.lower().strip(".") not in _NOT_NAMES and a.lower() != full_name.lower()][:4]
    c = Character(
        name=full_name, aliases=aliases,
        role=str(d.get("role") or role), concept=str(d.get("concept") or concept),
        appearance=str(d.get("appearance", "") or ""), backstory=str(d.get("backstory", "") or ""),
        matrix=PersonalityMatrix.from_dict(matrix_d), voice=voice,
        state=CharacterState(mood=str(state_d.get("mood", "steady") or "steady"), goal=str(state_d.get("goal", "") or ""),
                             knows=_as_list(state_d.get("knows"), 8), secrets=_as_list(state_d.get("secrets"), 4)),
    )
    rels = d.get("relationships")
    if isinstance(rels, dict):
        for other, r in rels.items():
            if isinstance(r, dict):
                c.relationships[str(other)] = Relationship(
                    trust=_clamp(r.get("trust", 0.5)), affection=_clamp(r.get("affection", 0.5)),
                    respect=_clamp(r.get("respect", 0.5)), fear=_clamp(r.get("fear", 0.0), default=0.0),
                    stance=str(r.get("stance", "") or ""))
            elif isinstance(r, str):
                c.relationships[str(other)] = Relationship(stance=r)
    return c


class CharacterCreator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def from_concept(self, name: str, concept: str, role: str = "supporting", bible_md: str = "",
                     others: Optional[list[str]] = None, seed_matrix: Optional[PersonalityMatrix] = None) -> Character:
        prompt = prompts.CHARACTER_CREATE_PROMPT.format(
            name=name, concept=concept, role=role, bible=bible_md or "(no bible yet)",
            others=", ".join(others or []) or "(none yet)",
            seed=("Start from these fixed trait values and do not change them: " + json.dumps(seed_matrix.axes()))
            if seed_matrix else "Choose trait values (0-1) that make this person specific, with at least three extremes.",
        )
        try:
            d = generate_json(self.llm, prompt, schema=schema_of(SheetOut), temperature=0.9, max_tokens=1400,
                              purpose="character:create")
        except Exception as e:
            log.warning("character sheet generation failed for %s: %s; using a rolled matrix", name, e)
            d = {}
        c = character_from_sheet(name, d, role, concept)
        if seed_matrix is not None:
            for axis, v in seed_matrix.axes().items():
                setattr(c.matrix, axis, v)
        elif not d:
            c.matrix = PersonalityMatrix.random()
        return c

    def roll(self, name: str, role: str = "supporting", seed: Optional[int] = None, concept: str = "") -> Character:
        """A character with a random matrix and no LLM involved; flesh out later with `from_concept`."""
        return Character(name=name, role=role, concept=concept, matrix=PersonalityMatrix.random(seed))
