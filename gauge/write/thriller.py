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
}


@dataclass
class RegisterProfile:
    hits: dict[str, int]
    vague: int                      # vague placeholders where expertise is claimed
    issues: list[str] = field(default_factory=list)


_VAGUE_EXPERT = re.compile(r"\b(did\s+(?:some|the)\s+(?:science|tests?|forensics?|calculations?)|"
                           r"some\s+(?:kind\s+of\s+)?(?:technical|scientific|medical)\s+\w+|"
                           r"fiddled\s+with|messed\s+with|did\s+something\s+to|worked\s+his\s+magic|"
                           r"complicated\s+stuff|technical\s+stuff)\b", re.I)
_CLAIMS_EXPERTISE = re.compile(r"\b(examined|analysed|analyzed|investigat|autops|diagnos|engineer|calculat|"
                               r"forensic|dissect|profiled|interrogat)\w*\b", re.I)


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
