"""A declared editorial stance — transparent, consistent, and honest about being a point of view.

No writer, and no model, produces "pure logic" free of values. The honest version of a moderate,
evidence-first platform is to *declare* the lens and apply it consistently: fiscal prudence and
compassion as stated principles, leaning conservative or liberal only where the evidence and those
principles point rather than by tribe, and labelling opinion as opinion. This module holds that stance,
turns it into guidance for the drafter, and checks a finished piece for one-sidedness — so "balanced
and logical" is enforced by steelmanning the other side, not merely claimed. The stance shapes opinion
and analysis; it never licenses bending the facts, which stay under the site's verify-and-no-fabricate
rails.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_COUNTER_MARKERS = [
    "to be fair", "admittedly", "critics", "opponents", "the other side", "the counterargument",
    "the strongest case against", "the case against", "yes, but", "on the other hand", "detractors",
    "skeptics", "those who disagree", "the conservative case", "the liberal case", "the left", "the right",
    "fairly, ", "in fairness", "steelman", "the best argument against", "concede",
]


@dataclass
class EditorialStance:
    name: str = "Evidence-first centrist"
    principles: list[str] = field(default_factory=lambda: [
        "fiscal prudence: public money is other people's money; weigh cost against result",
        "compassion: policy is judged by what it does to real people, especially the worst off",
        "follow the evidence, not the tribe",
    ])
    lean_rule: str = ("lean conservative or liberal only where the evidence and these principles point that way; "
                      "never by default, and say plainly which way a given piece lands and why")
    tone: str = "punchy, wry and comedic, plain-spoken; land a joke without faking a fact"
    rails: list[str] = field(default_factory=lambda: [
        "label opinion as opinion and satire as satire; never present a joke as a factual claim",
        "state the strongest version of the opposing view before answering it (steelman, don't strawman)",
        "separate what is known (and sourced) from what is argued",
        "no invented facts, quotes or statistics",
    ])

    def to_dict(self) -> dict:
        return {"name": self.name, "principles": self.principles, "lean_rule": self.lean_rule,
                "tone": self.tone, "rails": self.rails}


def default_stance() -> EditorialStance:
    return EditorialStance()


def stance_guidance(stance: EditorialStance, comedic: bool = True) -> str:
    tone = stance.tone if comedic else stance.tone.replace("and comedic, ", "").replace("punchy, wry", "measured")
    lines = [f"EDITORIAL STANCE: {stance.name}.", "Principles:"]
    lines += [f"- {p}" for p in stance.principles]
    lines += [f"Lean: {stance.lean_rule}.", f"Tone: {tone}.", "Rules:"]
    lines += [f"- {r}" for r in stance.rails]
    return "\n".join(lines)


def balance_check(text: str, min_words: int = 120) -> list[str]:
    """Flag a piece that argues only one side. Over `min_words`, it should engage the opposing view."""
    if len((text or "").split()) < min_words:
        return []
    low = (text or "").lower()
    if any(m in low for m in _COUNTER_MARKERS):
        return []
    return ["one-sided: the piece never engages an opposing view; add the strongest counterargument and answer it"]


def load_stance(path: Path) -> EditorialStance:
    if Path(path).exists():
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            return EditorialStance(name=d.get("name", "Evidence-first centrist"),
                                   principles=d.get("principles") or default_stance().principles,
                                   lean_rule=d.get("lean_rule") or default_stance().lean_rule,
                                   tone=d.get("tone") or default_stance().tone,
                                   rails=d.get("rails") or default_stance().rails)
        except Exception:
            pass
    return default_stance()


def save_stance(path: Path, stance: EditorialStance) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stance.to_dict(), indent=2), encoding="utf-8")
    return Path(path)


def editorial_page_body(stance: EditorialStance) -> str:
    """Transparent 'where we stand' text for the site, so readers know the lens."""
    parts = [f"This platform writes from a declared editorial stance: {stance.name}. It is a point of view, "
             "not a claim of neutrality. We tell you the lens so you can weigh what we write.", "", "Our principles:"]
    parts += [f"- {p}" for p in stance.principles]
    parts += ["", f"How we lean: {stance.lean_rule}.", "",
              "Opinion is labelled opinion. Facts are facts and are checked; where we argue, we show the strongest "
              "opposing case first. We would rather be fair than flattering to our own side."]
    return "\n".join(parts)
