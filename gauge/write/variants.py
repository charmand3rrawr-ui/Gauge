"""A/B hooks and titles: generate several, score them, keep the ledger so Gauge learns its hit rate.

The strongest cheap lever on a serial is the first line and the title: readers decide in both. This
module asks the model for several candidate openings and titles, ranks them with the same measured
features the critic trusts (hook strength, title shape), and logs the chosen one against its rivals
so that, over time, Gauge can see whether its own ranking actually predicts the winners.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..features import hook_score
from ..llm.client import LLM, generate_json
from ..text import word_count
from ..util import utcnow

_VARIANTS_SCHEMA = {
    "type": "object",
    "properties": {"variants": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
    "required": ["variants"],
}


@dataclass
class Scored:
    text: str
    score: float
    why: dict[str, float]


def _title_score(title: str) -> dict[str, float]:
    w = word_count(title)
    length_fit = 1.0 - min(1.0, abs(w - 3) / 6.0)  # 2-5 word titles read best on a feed
    has_concrete = 1.0 if any(c.isupper() for c in title[1:]) or any(ch.isdigit() for ch in title) else 0.6
    punch = 1.0 if w <= 5 and not title.endswith(".") else 0.7
    return {"length_fit": round(length_fit, 3), "concreteness": has_concrete, "punch": punch,
            "score": round(0.5 * length_fit + 0.25 * has_concrete + 0.25 * punch, 3)}


def rank_hooks(candidates: list[str]) -> list[Scored]:
    out = [Scored(c, float(hook_score(c).get("hook", 0.0)), {k: round(float(v), 3) for k, v in hook_score(c).items()})
           for c in candidates if c.strip()]
    return sorted(out, key=lambda s: s.score, reverse=True)


def rank_titles(candidates: list[str]) -> list[Scored]:
    out = []
    for c in candidates:
        if not c.strip():
            continue
        why = _title_score(c)
        out.append(Scored(c.strip(), why["score"], why))
    return sorted(out, key=lambda s: s.score, reverse=True)


def generate_hooks(llm: LLM, plan_text: str, n: int = 4, context: str = "") -> list[str]:
    prompt = (f"Generate opening hooks for a web-serial chapter. Give {n} different first lines, each a single "
              f"sentence that drops the reader into motion or a question.\nChapter plan: {plan_text}\n{context}")
    try:
        data = generate_json(llm, prompt, schema=_VARIANTS_SCHEMA, purpose="variants:hooks", temperature=1.0, max_tokens=400)
        return [str(x).strip() for x in data.get("variants", []) if str(x).strip()][:n]
    except Exception:
        return []


def generate_titles(llm: LLM, plan_text: str, chapter_text: str = "", n: int = 5) -> list[str]:
    prompt = (f"Generate chapter titles for this web-serial chapter. Give {n} short, punchy titles (2-5 words), "
              f"no spoilers.\nChapter plan: {plan_text}\nOpening: {chapter_text[:400]}")
    try:
        data = generate_json(llm, prompt, schema=_VARIANTS_SCHEMA, purpose="variants:titles", temperature=1.0, max_tokens=200)
        return [str(x).strip() for x in data.get("variants", []) if str(x).strip()][:n]
    except Exception:
        return []


class VariantLog:
    """Append-only record of A/B choices under projects/<slug>/variants.jsonl."""

    def __init__(self, project_dir: Path):
        self.path = Path(project_dir) / "variants.jsonl"

    def record(self, kind: str, number: int, chosen: str, alternatives: list[Scored], chosen_score: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"at": utcnow().isoformat(timespec="seconds"), "kind": kind, "chapter": number, "chosen": chosen,
               "chosen_score": round(chosen_score, 3),
               "alternatives": [{"text": s.text, "score": s.score} for s in alternatives[:8]]}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def hit_rate(self) -> Optional[float]:
        """Of the logged choices that carry an observed `won` flag, how often the top-ranked variant won.
        Returns None until outcomes are recorded (needs real reader data fed back in)."""
        judged = [r for r in self.rows() if "won" in r]
        if not judged:
            return None
        return round(sum(1 for r in judged if r.get("won")) / len(judged), 3)


def best_hook(llm: LLM, plan_text: str, n: int = 4, context: str = "") -> tuple[Optional[str], list[Scored]]:
    ranked = rank_hooks(generate_hooks(llm, plan_text, n, context))
    return (ranked[0].text if ranked else None), ranked


def best_title(llm: LLM, plan_text: str, chapter_text: str = "", n: int = 5) -> tuple[Optional[str], list[Scored]]:
    ranked = rank_titles(generate_titles(llm, plan_text, chapter_text, n))
    return (ranked[0].text if ranked else None), ranked
