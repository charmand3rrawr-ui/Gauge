"""Cross-project author memory: what the room keeps complaining about, distilled into lessons."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..db import Database

_NORM = re.compile(r"[^a-z ]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "this", "that", "it", "at", "for", "with", "on", "as",
         "be", "by", "chapter", "line", "would", "should", "her", "his", "she", "he", "they", "its", "than", "too", "very"}

THEMES = {
    "openings": ("open", "opening", "first line", "hook"),
    "endings": ("ending", "cliffhanger", "last paragraph", "resolves"),
    "pacing": ("slow middle", "pacing", "momentum", "drags", "stall", "same shape"),
    "repetition": ("repeat", "worn", "phrases", "echoes", "again"),
    "character voice": ("out of character", "voice", "sound alike", "too calm", "would not say"),
    "world rules": ("world rule", "continuity", "contradiction", "canon", "weather", "it is night"),
    "prose": ("filter words", "adverb", "dialogue tags", "telling", "intensifiers", "rhythm"),
    "payoffs": ("promise", "payoff", "overdue", "stalling", "nothing the reader"),
    "readers": ("click next", "would stop", "skimmer", "panel"),
}


def classify(issue: str) -> str:
    low = issue.lower()
    for theme, keys in THEMES.items():
        if any(k in low for k in keys):
            return theme
    return "other"


def distill(db: Database, project: str | None = None, top: int = 8) -> dict[str, Any]:
    notes = db.notes(project)
    if not notes:
        return {"total": 0, "themes": [], "lessons": []}
    themes = Counter(classify(n["issue"]) for n in notes)
    by_agent = Counter(n["agent"] for n in notes)
    phrases: Counter = Counter()
    for n in notes:
        words = [w for w in _NORM.sub(" ", n["issue"].lower()).split() if w not in _STOP and len(w) > 3]
        for i in range(len(words) - 1):
            phrases[" ".join(words[i:i + 2])] += 1
    total = len(notes)
    lessons = []
    for theme, count in themes.most_common(top):
        if theme == "other" or count < 2:
            continue
        share = count / total
        lessons.append({"theme": theme, "count": count, "share": round(share, 2), "advice": ADVICE.get(theme, "")})
    return {"total": total, "themes": themes.most_common(), "by_agent": by_agent.most_common(), "phrases": phrases.most_common(10),
            "lessons": lessons}


ADVICE = {
    "openings": "your openings keep getting flagged: start every chapter inside an action or a line of dialogue, never with weather or waking up",
    "endings": "your endings resolve too much: cut the last paragraph to one unresolved line",
    "pacing": "your middles sag: put the chapter's biggest turn at two-thirds and cut the scene before it by a third",
    "repetition": "you reuse phrases and shapes: vary sentence rhythm and retire the worn phrases in the avoid list",
    "character voice": "characters drift out of voice: reread their sheet before dialogue-heavy scenes and let them refuse things",
    "world rules": "world continuity slips: check time, weather and place against the world brief before writing",
    "prose": "prose-level habits: fewer filter words and intensifiers, plain dialogue tags, show the noticing",
    "payoffs": "promises go unpaid: advance or pay at least one ledger item every chapter",
    "readers": "readers drop off: the panel's stop points cluster; cut or sharpen those passages",
}


def render(summary: dict[str, Any]) -> str:
    lines = ["# Lessons from your chapters", "", f"Based on {summary.get('total', 0)} notes from the writing room.", ""]
    for les in summary.get("lessons", []):
        lines.append(f"- **{les['theme']}** ({les['count']} notes, {les['share']:.0%}): {les['advice']}")
    if summary.get("by_agent"):
        lines += ["", "Most vocal: " + ", ".join(f"{a} ({n})" for a, n in summary["by_agent"][:5])]
    if summary.get("phrases"):
        lines += ["", "Recurring complaints: " + "; ".join(p for p, _ in summary["phrases"][:6])]
    return "\n".join(lines) + "\n"


def write_lessons(db: Database, path: Path, project: str | None = None) -> dict[str, Any]:
    summary = distill(db, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(summary), encoding="utf-8")
    return summary


def prompt_lines(summary: dict[str, Any], n: int = 5) -> str:
    return "\n".join(f"- {les['advice']}" for les in summary.get("lessons", [])[:n])
