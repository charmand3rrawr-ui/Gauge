"""Drafter: writes and revises chapter prose, within an explicit token budget.

Every prompt is assembled against ``num_ctx``: the fixed parts (plan, previous tail, bible,
summary) go in first, then exemplars and cast briefs are added while they fit. Nothing is ever
left for the server to truncate, because servers truncate from the front, which is where the
system prompt and bible live.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from ..learn.exemplars import ExemplarLibrary
from ..llm.client import LLM
from ..llm.tokens import PromptBudget, estimate_tokens, truncate_to_tokens
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


def output_tokens_for(words: int) -> int:
    return int(words * 1.8) + 400


class Drafter:
    def __init__(self, llm: LLM, exemplars: Optional[ExemplarLibrary], style_guide: str, exemplars_in_prompt: int = 4):
        self.llm = llm
        self.exemplars = exemplars
        self.k = exemplars_in_prompt
        self.system = prompts.SYSTEM_AUTHOR.format(style_guide=style_guide or "(no data yet)")
        self.last_budget: Optional[PromptBudget] = None

    def _exemplar_items(self, project: Project, plan: ChapterPlan, kind: str) -> list[str]:
        if not self.exemplars:
            return []
        q = f"{plan.hook if kind == 'opening' else plan.cliffhanger} {project.state.bible.promise}"
        found = self.exemplars.search(q, kind=kind, k=self.k, genres=project.state.genres)
        return [f"[{e.novel_title}]\n{e.text}" for e in found]

    def build_draft_prompt(self, project: Project, plan: ChapterPlan, prev_text: Optional[str], words: int,
                           cast: str = "") -> tuple[str, int]:
        """Return (prompt, max_output_tokens), fitted to the model's context window."""
        max_out = output_tokens_for(words)
        budget = PromptBudget(self.llm.context_window(), max_out)
        budget.take(self.system)
        # Fixed skeleton: the template itself plus the plan.
        skeleton = prompts.DRAFT_PROMPT.format(
            number=plan.number, title_novel=project.state.name, words=words, bible="", summary="", prev_tail="",
            ch_title=plan.title, goal=plan.goal, hook=plan.hook, turn=plan.turn, cliffhanger=plan.cliffhanger,
            beats="; ".join(plan.beats), opening_exemplars="", ending_exemplars="", cast="")
        budget.take(skeleton)
        # Priority order: previous tail (continuity) > bible > summary > cast > exemplars.
        prev_tail = budget.fit("prev_tail", tail(prev_text, 250) if prev_text else "(there is no previous chapter)")
        bible = budget.fit("bible", project.bible_markdown(), min_tokens=200)
        summary = budget.fit("summary", project.state.running_summary or "(this is the first chapter)")
        cast_text = budget.fit("cast", cast or "(no profiled characters in this scene)", min_tokens=150)
        openings = budget.fit_list("opening_exemplars", self._exemplar_items(project, plan, "opening"))
        endings = budget.fit_list("ending_exemplars", self._exemplar_items(project, plan, "ending"))
        if budget.dropped:
            log.info("draft prompt trimmed to fit ctx=%d: %s", self.llm.context_window(), ", ".join(budget.dropped))
        self.last_budget = budget
        prompt = prompts.DRAFT_PROMPT.format(
            number=plan.number, title_novel=project.state.name, words=words, bible=bible or "(bible omitted: context too small)",
            summary=summary or "(summary omitted)", prev_tail=prev_tail or "(previous chapter omitted)",
            ch_title=plan.title, goal=plan.goal, hook=plan.hook, turn=plan.turn, cliffhanger=plan.cliffhanger,
            beats="; ".join(plan.beats), opening_exemplars=openings or "(none)", ending_exemplars=endings or "(none)",
            cast=cast_text or "(no profiled characters in this scene)",
        )
        return prompt, max_out

    def draft(self, project: Project, plan: ChapterPlan, prev_text: Optional[str], words: int, cast: str = "") -> str:
        prompt, max_out = self.build_draft_prompt(project, plan, prev_text, words, cast)
        raw = self.llm.generate(prompt, system=self.system, max_tokens=max_out, purpose="draft")
        return clean_prose(raw)

    def revise(self, text: str, problems: list[str], measured: list[str], words: int) -> str:
        max_out = output_tokens_for(words)
        head = prompts.REVISE_PROMPT.format(
            problems="\n".join(f"- {p}" for p in problems) or "- (none)",
            measured="\n".join(f"- {m}" for m in measured) or "- (none)",
        )
        room = self.llm.context_window() - max_out - estimate_tokens(self.system) - estimate_tokens(head) - 256
        body = text
        if estimate_tokens(body) > room:
            # A chapter too long to revise whole is revised from the start; the tail is kept verbatim.
            log.warning("chapter too long for a whole-text revision at ctx=%d; revising the first part only", self.llm.context_window())
            body = truncate_to_tokens(text, max(300, room))
        raw = self.llm.generate(head + "\n\nCHAPTER:\n" + body, system=self.system, max_tokens=max_out, temperature=0.7,
                                purpose="revise")
        revised = clean_prose(raw)
        # Guard against a model that returns a fragment instead of the full chapter.
        return revised if len(revised.split()) >= 0.6 * len(text.split()) else text
