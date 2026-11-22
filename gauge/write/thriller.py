"""Measured craft checks for crime, thriller and mystery fiction.

All deterministic, no model calls. This layer raises the craft of dark fiction — suspense, a fairly
clued mystery, a consistent and genuinely menacing antagonist, and prose whose procedural/forensic/
psychological register reads expert rather than vague.

Scope and intent: this improves *storytelling* — tension, structure, characterisation and verisimilitude.
It deliberately does NOT produce operational real-world instructions (how to deceive people, commit
crimes, evade investigation, or build weapons). The register check rewards the *presence* of plausible,
widely-known vocabulary so the fiction reads authentic; it never supplies methods. Good thrillers are made
by dread, fair clues and consistent character, not by accurate crime manuals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text import paragraphs, words

THRILLER_WORDS = ("thriller", "crime", "mystery", "detective", "noir", "suspense", "horror", "whodunit",
                  "police procedural", "psychological", "serial killer")


def wants_thriller(genres: list[str], tone: str = "") -> bool:
    hay = " ".join(genres + [tone]).lower()
    return any(w in hay for w in THRILLER_WORDS)


# ---------------------------------------------------------------------------- suspense / tension

_DREAD = {"wrong", "too quiet", "too still", "shouldn't", "something", "watched", "watching", "followed",
          "cold", "dread", "uneasy", "prickled", "hairs", "silence", "shadow", "waited", "hidden",
          "warning", "danger", "threat", "afraid", "fear", "trap", "alone", "dark", "empty", "missing"}
_CLOCK = re.compile(r"\b(deadline|before\s+(?:midnight|dawn|sunrise|morning|noon|it'?s\s+too\s+late)|"
                    r"running\s+out\s+of\s+time|countdown|minutes?\s+left|hours?\s+left|ticking|"
                    r"only\s+\w+\s+(?:minutes?|hours?|days?)\s+(?:left|remain)|by\s+the\s+time)\b", re.I)
_STAKES = re.compile(r"\b(kill|killed|die|death|dead|murder|hostage|bomb|missing|abducted|kidnap|"
                     r"bleed|bleeding|survive|alive|execute|victim|body|corpse|drown|suffocat)\w*\b", re.I)
# A chapter that ends on a hook: a question, a reveal, an interruption, a cliff.
_END_HOOK = re.compile(r"(\?|—|\.\.\.|\bthen\s+(?:the|she|he|they)\b|\bbut\s+(?:the|she|he|they)\b|"
                       r"\btoo\s+late\b|\bwasn'?t\s+alone\b|\bgone\b|\bwrong\b)\s*[\"'”’]?\s*$", re.I)


@dataclass
class SuspenseProfile:
    dread_cues: int
    ticking_clock: bool
    stakes: int
    ends_on_hook: bool
    quiet_paragraphs: int           # consecutive low-tension paragraphs (talking heads / lull)
    issues: list[str] = field(default_factory=list)


def suspense_profile(text: str) -> SuspenseProfile:
    toks = [w.lower() for w in words(text)]
    n = max(1, len(toks))
    dread = sum(1 for t in toks if t in _DREAD)
    clock = bool(_CLOCK.search(text))
    stakes = len(_STAKES.findall(text))
    paras = [p for p in paragraphs(text) if p.strip()]
    tail = paras[-1] if paras else text
    ends_hook = bool(_END_HOOK.search(tail.strip()))
    # Longest run of paragraphs with no dread/stakes cue at all.
    longest_quiet = run = 0
    for p in paras:
        pt = p.lower()
        if any(d in pt for d in _DREAD) or _STAKES.search(p):
            run = 0
        else:
            run += 1
            longest_quiet = max(longest_quiet, run)
    issues: list[str] = []
    if len(toks) >= 150:
        if dread < max(1, n // 400) and stakes == 0:
            issues.append("the chapter carries little tension; plant dread (something watched, something wrong) or raise the stakes")
        if not ends_hook:
            issues.append("the chapter does not end on a hook; close on a question, a reveal, or a turn that pulls the reader on")
        if longest_quiet >= 6:
            issues.append(f"a long lull ({longest_quiet} paragraphs with no tension); thread menace or a ticking clock through it")
    return SuspenseProfile(dread, clock, stakes, ends_hook, longest_quiet, issues)


def suspense_score(p: SuspenseProfile) -> float:
    s = 0.0
    s += min(1.0, p.dread_cues / 4) * 0.35
    s += (1.0 if p.stakes > 0 else 0.0) * 0.2
    s += (1.0 if p.ends_on_hook else 0.0) * 0.25
    s += (0.1 if p.ticking_clock else 0.0)
    s += (1.0 if p.quiet_paragraphs < 6 else 0.4) * 0.1
    return round(min(1.0, s), 3)


# ---------------------------------------------------------------------------- expert register (verisimilitude only)

# Widely-known vocabulary that makes each domain *read* authentic. Presence, not instruction: these are the
# words a knowledgeable author uses, not a method. No synthesis, dosages, defeat-the-lock, or build steps.
REGISTER: dict[str, set[str]] = {
    "forensic": {"forensic", "autopsy", "coroner", "medical examiner", "rigor mortis", "livor", "lividity",
                 "time of death", "ballistics", "trajectory", "cartridge", "casing", "trace", "fibre", "fiber",
                 "dna", "latent", "fingerprint", "luminol", "blood spatter", "spatter", "toxicology", "postmortem",
                 "decomposition", "entomology", "chain of custody", "evidence", "swab", "tox screen"},
    "criminology": {"motive", "opportunity", "means", "modus operandi", "mo", "signature", "staging", "undoing",
                    "victimology", "profile", "profiler", "offender", "cold case", "jurisdiction", "warrant",
                    "miranda", "interrogation", "alibi", "witness", "perpetrator", "spree", "pattern", "escalation",
                    "cooling-off", "trophy", "souvenir", "disorganised", "disorganized", "organised", "organized"},
    "psychology": {"compartmentalise", "compartmentalize", "dissociation", "projection", "narcissis", "empathy",
                   "remorse", "antisocial", "psychopath", "sociopath", "grandiosity", "manipulat", "gaslight",
                   "trauma", "trigger", "compulsion", "rationalise", "rationalize", "denial", "transference",
                   "affect", "flat affect", "delusion", "paranoia", "obsession", "fixation", "cognitive"},
    "deception": {"tell", "tells", "microexpression", "baseline", "pretext", "cover story", "misdirection",
                  "sleight", "feint", "double bluff", "red herring", "plausible", "inconsistency", "contradiction",
                  "lie", "deceive", "conceal", "disguise", "alias", "forged", "forgery", "cover", "ruse"},
    "engineering": {"tolerance", "load-bearing", "shear", "torque", "circuit", "voltage", "current", "relay",
                    "actuator", "schematic", "pressure", "valve", "hydraulic", "failsafe", "fail-safe", "redundancy",
                    "calibrat", "mechanism", "gear", "bearing", "weld", "fatigue", "stress", "specification", "blueprint"},
    # Espionage/spy-thriller texture — the words a le Carré or Clancy page uses. Plot-device terms only
    # (honeypot, dead drop, legend); NOT how to actually run any of it against real people.
    "tradecraft": {"tradecraft", "dead drop", "brush pass", "cutout", "legend", "cover", "cover identity",
                   "safe house", "handler", "asset", "mole", "double agent", "sleeper", "exfil", "exfiltration",
                   "extraction", "surveillance detection route", "sdr", "countersurveillance", "tail", "watchers",
                   "burned", "blown", "flap", "defector", "walk-in", "cipher", "one-time pad", "cache", "signal site",
                   "honeypot", "honey trap", "kompromat", "false flag", "non-official cover", "noc", "station chief",
                   "exfil route", "chatter", "source", "tasking", "debrief", "pocket litter"},
    # Special-operations / military texture — ground an operation, not a manual.
    "special_ops": {"fireteam", "overwatch", "breach", "stack", "rules of engagement", "roe", "exfil", "callsign",
                    "rally point", "lz", "hlz", "casevac", "medevac", "qrf", "sitrep", "frago", "oscar mike",
                    "danger close", "cover and concealment", "suppressive fire", "fields of fire", "fire discipline",
                    "overmatch", "hostiles", "friendlies", "comms", "exfil bird", "fast rope", "hard target",
                    "soft target", "actionable", "op order", "no-go", "go code", "stand down", "weapons free"},
    # Firearms texture — the sensory vocabulary of a gun scene (report, recoil, discipline), never ballistics
    # for real use, zeroing instructions, or anything that improves real-world lethality.
    "firearms": {"magazine", "chamber", "chambered", "slide", "bolt", "muzzle", "report", "recoil", "sling",
                 "optic", "red dot", "iron sights", "safety", "trigger discipline", "muzzle discipline", "stance",
                 "grip", "cycle", "misfire", "jam", "stovepipe", "reload", "spent casing", "suppressor", "crack", "burst", "controlled pair", "holster", "drawstroke"},
    # Close-quarters texture — the feel of a fight, not instruction on how to hurt anyone.
    "close_quarters": {"clinch", "grapple", "control", "leverage", "off-balance", "disarm", "takedown", "choke",
                       "pivot", "footwork", "guard", "strike", "parry", "blade", "edge", "reversal", "pin",
                       "distance", "angle", "close the gap", "break contact"},
}


@dataclass
class RegisterProfile:
    hits: dict[str, int]
    vague: int                      # vague placeholders where expertise is claimed
    issues: list[str] = field(default_factory=list)


# Storytelling guidance per domain: the tropes a chapter should *dramatise*, not methods to perform.
# These shape scenes (show the cover being tested, the fight turning on one mistake), never real technique.
TROPES: dict[str, list[str]] = {
    "tradecraft": ["ground the operation in texture: a dead drop, a cover that gets tested, a handler who wants more than the asset will give",
                   "show the cost of the double life, not just the gadgetry; loyalty, paranoia, who can be trusted",
                   "a honeypot or kompromat is a plot device about leverage and betrayal — play the human stakes, not a how-to"],
    "special_ops": ["let competence show through small procedure and restraint, then one thing goes wrong",
                    "the team is characters first; give them a cost, a loss, a decision no training covers",
                    "tension from the plan meeting reality, not from a gear list"],
    "firearms": ["render a gun scene through the senses — the report, the recoil, the after-silence — and its consequence",
                 "discipline and reluctance read as expertise more than marksmanship does"],
    "close_quarters": ["a fight is drama: stakes, a turn, a cost — exhaustion and fear, not choreography",
                       "let it turn on one mistake or one improvised choice, and end it fast"],
    "psychology": ["show the interior logic that makes a monster coherent from the inside, without endorsing it"],
}


def register_palette(domain: str) -> list[str]:
    """The vocabulary palette for a domain — a genre thesaurus for the author, sorted. Terms, not methods."""
    return sorted(REGISTER.get(domain, set()))


_VAGUE_EXPERT = re.compile(r"\b(did\s+(?:some|the)\s+(?:science|tests?|forensics?|calculations?|spy\s+stuff|"
                           r"ninja\s+stuff|combat\s+stuff|tradecraft)|"
                           r"some\s+(?:kind\s+of\s+)?(?:technical|scientific|medical|tactical|special-?ops)\s+\w+|"
                           r"used\s+(?:his|her|their)\s+training|years\s+of\s+training\s+kicked\s+in|"
                           r"fiddled\s+with|messed\s+with|did\s+something\s+to|worked\s+his\s+magic|"
                           r"complicated\s+stuff|technical\s+stuff)\b", re.I)
_CLAIMS_EXPERTISE = re.compile(r"\b(examined|analysed|analyzed|investigat|autops|diagnos|engineer|calculat|"
                               r"forensic|dissect|profiled|interrogat|infiltrat|exfiltrat|surveil)\w*\b", re.I)


def register_profile(text: str, domains: list[str] | None = None) -> RegisterProfile:
    """How authentically the prose carries each requested domain's register. Rewards specific vocabulary;
    flags vague placeholders ('did some science stuff') where the scene claims expertise."""
    low = text.lower()
    doms = domains or list(REGISTER)
    hits = {d: sum(1 for term in REGISTER.get(d, set()) if term in low) for d in doms}
    vague = len(_VAGUE_EXPERT.findall(text))
    issues: list[str] = []
    claims = bool(_CLAIMS_EXPERTISE.search(text))
    if vague:
        issues.append(f"expertise is hand-waved ({vague}x, e.g. 'did some forensics'); use the specific, concrete register instead")
    if claims and sum(hits.values()) == 0 and len(words(text)) >= 150:
        issues.append("the scene claims technical/forensic work but carries none of its vocabulary; ground it in specifics")
    return RegisterProfile(hits, vague, issues)


# ---------------------------------------------------------------------------- antagonist menace & signature

_MENACE_ONPAGE = re.compile(r"\b(smiled|calm|patient|unhurried|precise|methodical|watched|studied|"
                            r"whispered|softly|gentle|polite|courteous)\b", re.I)


def signature_terms(text: str, lexicon: set[str] | None = None) -> set[str]:
    """Distinctive MO/signature tokens present in a chapter — used to check a serial antagonist stays
    consistent across chapters. `lexicon` (author-declared) narrows it; otherwise it is the forensic/
    criminology register found in the text."""
    low = text.lower()
    if lexicon:
        return {t for t in lexicon if t.lower() in low}
    base = REGISTER["forensic"] | REGISTER["criminology"]
    return {t for t in base if t in low}


# ---------------------------------------------------------------------------- fair-play mystery

@dataclass
class FairPlay:
    missing_clues: list[str]        # declared key clues never shown to the reader before the reveal
    unresolved_herrings: list[str]  # red herrings raised but never dispelled


def fair_play(declared_clues: list[str], declared_herrings: list[str],
              seen_clues: set[str], resolved_herrings: set[str]) -> FairPlay:
    missing = [c for c in declared_clues if c not in seen_clues]
    unresolved = [h for h in declared_herrings if h not in resolved_herrings]
    return FairPlay(missing, unresolved)


def clues_present(text: str, declared: list[str]) -> set[str]:
    """Which declared clues (substring, case-insensitive) appear in this chapter."""
    low = text.lower()
    return {c for c in declared if c and c.lower() in low}


# ---------------------------------------------------------------------------- secrets (the mirror of clues)

@dataclass
class Secret:
    phrase: str
    chapter: int                    # the chapter the secret may first surface; before it, a leak


def parse_secrets(raw: list[str]) -> list[Secret]:
    """Secrets that must stay hidden until a chapter — a mole's name, a cover blown, the twist. One per
    entry, 'phrase @ 12' (reveal in chapter 12). Entries without '@' are treated as reveal at chapter 0
    (always allowed) and ignored by the leak check."""
    out: list[Secret] = []
    for item in raw or []:
        s = str(item).strip()
        if not s:
            continue
        if "@" in s:
            phrase, _, ch = s.rpartition("@")
            try:
                out.append(Secret(phrase.strip(), int(ch.strip())))
            except ValueError:
                out.append(Secret(s, 0))
        else:
            out.append(Secret(s, 0))
    return out


def leaked_secrets(text: str, secrets: list[Secret], current_chapter: int) -> list[str]:
    """Declared secrets that surface in this chapter before the chapter they are meant to be revealed."""
    low = text.lower()
    return [s.phrase for s in secrets if s.chapter > current_chapter and s.phrase and s.phrase.lower() in low]


# ---------------------------------------------------------------------------- covers / legends

@dataclass
class Cover:
    character: str                  # the real identity
    alias: str                      # the cover name / legend they operate under


def parse_covers(raw: list[str]) -> list[Cover]:
    """A character's cover identity, one per entry: 'RealName = Alias' (e.g. 'Mara Vane = Lena Koss')."""
    out: list[Cover] = []
    for item in raw or []:
        s = str(item).strip()
        if "=" not in s:
            continue
        real, _, alias = s.partition("=")
        real, alias = real.strip(), alias.strip()
        if real and alias:
            out.append(Cover(real, alias))
    return out


def cover_collisions(covers: list[Cover]) -> list[str]:
    """Aliases that two different characters both operate under — usually an accidental clash, not a
    deliberate shared legend; worth one warning so the author confirms it is intended."""
    by_alias: dict[str, set[str]] = {}
    for c in covers:
        by_alias.setdefault(c.alias.lower(), set()).add(c.character)
    return [alias for alias, chars in by_alias.items() if len(chars) > 1]


def covers_present(text: str, covers: list[Cover]) -> set[str]:
    """Which declared aliases appear in this chapter (for the cover ledger)."""
    low = text.lower()
    return {c.alias for c in covers if c.alias and c.alias.lower() in low}
