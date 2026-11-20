"""Drafter: writes and revises chapter prose."""
from __future__ import annotations

import logging
import re
from typing import Optional

from ..learn.exemplars import ExemplarLibrary
from ..llm.client import LLM
from ..text import tail
from . import prompts
from .project import ChapterPlan, Project

log = logging.getLogger(__name__)


def clean_prose(text: str) -> str:
    text = text.strip()
    # Strip a leading title/heading the model may have added despite instructions.
    text = re.sub(r"^\s*(#+\s*)?chapter\s+\d+[^\n]*\n+", "", text, flags=re.I)
    text = re.sub(r"^\s*\*\*[^\n]{0,80}\*\*\s*\n+", "", text)
    # Remove trailing author notes.
    text = re.sub(r"\n+(---+|\*\*\*+)\s*\n.*$", "", text, flags=re.S)
    text = re.sub(r"\n+(author'?s? note|a/n)[:\s].*$", "", text, flags=re.I | re.S)
    return text.strip()


class Drafter:
    def __init__(self, llm: LLM, exemplars: Optional[ExemplarLibrary], style_guide: str, exemplars_in_prompt: int = 4):
        self.llm = llm
        self.exemplars = exemplars
        self.k = exemplars_in_prompt
        self.system = prompts.SYSTEM_AUTHOR.format(style_guide=style_guide or "(no data yet)")

    def _exemplars(self, project: Project, plan: ChapterPlan, kind: str) -> str:
        if not self.exemplars:
            return "(none yet)"
        q = f"{plan.hook if kind == 'opening' else plan.cliffhanger} {project.state.bible.promise}"
        found = self.exemplars.search(q, kind=kind, k=self.k, genres=project.state.genres)
        return "\n\n".join(f"[{e.novel_title}]\n{e.text}" for e in found) or "(none)"

    def draft(self, project: Project, plan: ChapterPlan, prev_text: Optional[str], words: int,
              cast: str = "") -> str:
        prompt = prompts.DRAFT_PROMPT.format(
            number=plan.number, title_novel=project.state.name, words=words, bible=project.bible_markdown(),
            summary=project.state.running_summary or "(this is the first chapter)",
            prev_tail=tail(prev_text, 250) if prev_text else "(there is no previous chapter)",
            ch_title=plan.title, goal=plan.goal, hook=plan.hook, turn=plan.turn,
            cliffhanger=plan.cliffhanger, beats="; ".join(plan.beats),
            opening_exemplars=self._exemplars(project, plan, "opening"),
            ending_exemplars=self._exemplars(project, plan, "ending"),
            cast=cast or "(no profiled characters in this scene)",
        )
        max_tokens = int(words * 1.8) + 400
        raw = self.llm.generate(prompt, system=self.system, max_tokens=max_tokens)
        return clean_prose(raw)

    def revise(self, text: str, problems: list[str], measured: list[str], words: int) -> str:
        prompt = prompts.REVISE_PROMPT.format(
            problems="\n".join(f"- {p}" for p in problems) or "- (none)",
            measured="\n".join(f"- {m}" for m in measured) or "- (none)",
        ) + "\n\nCHAPTER:\n" + text
        raw = self.llm.generate(prompt, system=self.system, max_tokens=int(words * 1.8) + 400, temperature=0.7)
        revised = clean_prose(raw)
        # Guard against a model that returns a fragment instead of the full chapter.
        return revised if len(revised.split()) >= 0.6 * len(text.split()) else text
