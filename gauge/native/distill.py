"""Distillation: teach GaugeLM to behave like the big local model on Gauge's own tasks.

A small model learns behaviour far faster from a stronger model's answers than from raw prose.
This module draws prompts shaped like the pipeline's real work (outline a chapter from a bible,
draft a scene from a plan, score a chapter, brief a character, retitle) using material from your
corpus, asks the configured big provider (Ollama / OpenAI-compatible) to answer, and stores the
pairs in ``datasets/distill.jsonl``, which the native trainer upweights as instruction rows."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

from ..db import Database
from ..llm.client import LLM
from ..text import head, tail

log = logging.getLogger(__name__)

TASKS = [
    ("continue", "Continue this serial chapter for about {n} words, keeping the voice and ending on an unresolved beat:\n\n{excerpt}", 220),
    ("opening", "Write the opening 120 words of a serial chapter in the style of this excerpt, starting in motion:\n\n{excerpt}", 180),
    ("ending", "Rewrite the ending of this passage so it stops on an unresolved line that forces the next chapter:\n\n{excerpt}", 160),
    ("title", "Give this chapter a title of 1-5 words that promises without spoiling:\n\n{excerpt}", 12),
    ("score", "Score this chapter 1-10 for hook, momentum and cliffhanger, as three lines 'name: number', then one line of advice:\n\n{excerpt}", 60),
    ("outline", "Outline the next chapter after this passage: give title, hook, turn and cliffhanger as four short lines:\n\n{excerpt}", 90),
    ("brief", "A character in this passage is about to act. In three lines, say what they want, what they fear and what they would refuse:\n\n{excerpt}", 80),
    ("dialogue", "Write six lines of dialogue between two people from this passage, each in a distinct voice, no narration:\n\n{excerpt}", 150),
]


def sample_excerpts(db: Database, n: int, rng: random.Random, words: int = 220) -> list[str]:
    """n random windows of prose, spread over novels and chapters (a small corpus is re-sampled)."""
    novels = [nv for nv in db.novels() if db.chapter_count(nv.key)]
    if not novels:
        return []
    out: list[str] = []
    attempts = 0
    while len(out) < n and attempts < n * 5:
        attempts += 1
        novel = rng.choice(novels)
        chapters = db.chapters(novel.key)
        ch = rng.choice(chapters)
        ws = ch.text.split()
        if len(ws) < 80:
            continue
        start = rng.randrange(0, max(1, len(ws) - words))
        out.append(" ".join(ws[start:start + words]))
    return out


def distill(db: Database, teacher: LLM, out_path: Path, rows: int = 200, seed: int = 0,
            progress: Optional[Any] = None) -> dict[str, int]:
    rng = random.Random(seed)
    excerpts = sample_excerpts(db, rows, rng)
    if not excerpts:
        return {"rows": 0, "reason": "no chapters in the corpus"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, ex in enumerate(excerpts):
            kind, template, max_words = TASKS[i % len(TASKS)]
            prompt = template.format(excerpt=ex, n=max_words)
            try:
                answer = teacher.generate(prompt, temperature=0.8, max_tokens=int(max_words * 1.6) + 30, purpose="distill").strip()
            except Exception as e:
                log.warning("distill: teacher failed (%s)", e)
                failed += 1
                continue
            if not answer:
                failed += 1
                continue
            f.write(json.dumps({"task": kind, "prompt": prompt, "response": answer}, ensure_ascii=False) + "\n")
            written += 1
            if progress and written % 10 == 0:
                progress({"written": written, "failed": failed})
    return {"rows": written, "failed": failed, "path": str(out_path)}


def distill_summary(out_path: Path) -> dict[str, int]:
    if not out_path.exists():
        return {}
    counts: dict[str, int] = {}
    for line in out_path.open(encoding="utf-8"):
        try:
            counts[json.loads(line).get("task", "?")] = counts.get(json.loads(line).get("task", "?"), 0) + 1
        except json.JSONDecodeError:
            continue
    return counts


__all__ = ["distill", "distill_summary", "sample_excerpts", "TASKS", "head", "tail"]
