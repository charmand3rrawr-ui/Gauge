"""The Packager: chapter titles, blurb, tags. The hooks outside the prose."""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from ...features import title_features
from ...text import word_count
from .base import StoryAgent, as_str_list

SYSTEM = """You package serial fiction so strangers click: chapter titles that promise without spoiling, blurbs that ask a
question the reader must have answered, tags readers actually search for."""
GENERIC = re.compile(r"^\s*(chapter|ch\.?|part|episode)\s*\d+\s*$", re.I)


class PackState(BaseModel):
    titles: dict[int, str] = Field(default_factory=dict)
    candidates: dict[int, list[str]] = Field(default_factory=dict)
    blurb: str = ""
    tagline: str = ""
    tags: list[str] = Field(default_factory=list)


def title_score(t: str) -> float:
    f = title_features(t)
    n = f["title_words"]
    s = 0.0
    s += 0.4 if 1 <= n <= 6 else 0.0
    s += 0.2 if f["title_question"] else 0.0
    s -= 0.5 if f["title_numbered_only"] else 0.0
    s += 0.2 if any(w[:1].isupper() and len(w) > 3 for w in t.split()[1:]) else 0.0  # a concrete noun
    s -= 0.3 if t.lower().startswith(("the ", "a ")) and n <= 2 else 0.0
    return s


class Packager(StoryAgent):
    name = "packager"
    title = "Packager"
    weight = 0.0  # advises; does not score chapters
    state_model = PackState

    def retitle(self, text: str, plan, number: int) -> Optional[str]:
        """Suggest a better chapter title when the plan's title is generic. Returns None to keep it."""
        if plan.title and not GENERIC.match(plan.title) and word_count(plan.title) <= 7:
            return None
        d = self.ask_json("titles", f"""Propose 3 chapter titles for this chapter: 1-5 words, concrete, a promise or a question,
no spoilers of the ending, no "Chapter N".

CHAPTER (excerpt):
{' '.join(text.split()[:500])}

Respond with ONLY JSON: {{"titles": ["...", "...", "..."]}}""", system=SYSTEM, temperature=0.9, max_tokens=120)
        cands = [c.strip(" \"'.") for c in as_str_list(d.get("titles"), 5, 80) if c.strip()]
        cands = [c for c in cands if not GENERIC.match(c) and 1 <= word_count(c) <= 7]
        if not cands:
            return None
        best = max(cands, key=title_score)
        self.state.candidates[number] = cands
        self.state.titles[number] = best
        return best

    def package(self, project) -> dict:
        d = self.ask_json("blurb", f"""Write the store-front for this serial.

BIBLE:
{project.bible_markdown()}

STORY SO FAR:
{project.state.running_summary or '(beginning)'}

Respond with ONLY JSON: {{"blurb": "80-140 words, opens with a question or a hard fact, ends on the promise, no spoilers past chapter 3",
"tagline": "<= 12 words", "tags": ["6-10 tags readers search for"]}}""", system=SYSTEM, temperature=0.8, max_tokens=500)
        if d.get("blurb"):
            self.state.blurb = str(d["blurb"])[:1200]
        if d.get("tagline"):
            self.state.tagline = str(d["tagline"])[:120]
        tags = as_str_list(d.get("tags"), 10, 40)
        if tags:
            self.state.tags = tags
        self.save()
        return {"blurb": self.state.blurb, "tagline": self.state.tagline, "tags": self.state.tags}

    def report(self) -> str:
        lines = [f"Tagline: {self.state.tagline}", f"Tags: {', '.join(self.state.tags)}", "Blurb:", self.state.blurb, "Titles chosen:"]
        lines += [f"  ch {n}: {t}  (from {', '.join(self.state.candidates.get(n, []))})" for n, t in sorted(self.state.titles.items())[-10:]]
        return "\n".join(lines)
