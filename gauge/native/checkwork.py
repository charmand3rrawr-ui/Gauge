"""Checking GaugeLM's work: another model reads what it wrote, marks it, and points at the faults.

GaugeLM cannot tell whether what it wrote is any good. A stronger model can. A check takes a few
passages of a chapter (the opening, the ending, and some from between), and has each read by one or
two checkers. Each checker gives:

* marks out of 10 for clarity, grammar, sense (does it hold together) and interest;
* the worst faults, each with the exact words, what is wrong, and how to put it right;
* the passage rewritten with those faults put right.

What is done with that:

* the marks are averaged and remembered, so the trend over many chapters shows whether GaugeLM's
  writing is improving. They are a model's opinion: useful as a trend, not as a measurement;
* a fault is only reported when the quoted words really are in the passage (checkers make things up
  too), so every fault shown can be found in the text;
* the corrected passage becomes a lesson ("rewrite this so it is clear and correct") that GaugeLM
  trains on, but only from a checker whose answers may be kept as lessons.

Who checks: the free hosted helpers switched on for checking (`llm/helpers.py`), then Claude if it is
teaching, then the free models on this computer. GaugeLM never checks itself.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from ..llm.client import extract_json
from . import school

log = logging.getLogger(__name__)

ASPECTS = ("clarity", "grammar", "sense", "interest")
CHECK_PROMPT = (
    "A student writer wrote this passage of a story:\n\n{passage}\n\n"
    "Check it. Reply with one JSON object and nothing else, with exactly these keys:\n"
    '"marks": an object with a whole number from 1 (very poor) to 10 (excellent) for each of "clarity", "grammar", '
    '"sense" (does it hold together and mean something) and "interest";\n'
    '"problems": a list of up to five of the worst faults, each an object with "quote" (the exact words from the '
    'passage, a few words long), "problem" (what is wrong, in one plain sentence) and "fix" (how to put it right);\n'
    '"corrected": the passage rewritten with those faults put right, keeping what happens and about the same length.')
CORRECT_TASK = "Rewrite this passage so that it is clear and correct, keeping what happens:\n{passage}"


def pieces(text: str, n: int = 3, words: int = 160) -> list[str]:
    """Up to n passages of about `words` words: the opening, the ending, and evenly spaced ones between.
    Cut at paragraph ends where the text has them."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks: list[str] = []
    cur: list[str] = []
    count = 0
    for p in paras:
        cur.append(p)
        count += len(p.split())
        if count >= words:
            blocks.append("\n\n".join(cur))
            cur, count = [], 0
    if cur and (count >= words // 3 or not blocks):
        blocks.append("\n\n".join(cur))
    blocks = [" ".join(b.split()[:int(words * 1.8)]) if len(b.split()) > words * 1.8 else b for b in blocks]
    blocks = [b for b in blocks if len(b.split()) >= 20]
    if len(blocks) <= n:
        return blocks
    if n == 1:
        return [blocks[0]]
    picks = sorted({round(i * (len(blocks) - 1) / (n - 1)) for i in range(n)})
    return [blocks[i] for i in picks]


def checkers(cfg: Any) -> list[school.Teacher]:
    """Who can check right now, the ones most worth asking first: hosted helpers switched on for checking, Claude
    when it is teaching, then the free models on this computer."""
    out = school.helper_teachers(cfg, "check")
    lead = school.claude_teacher(cfg)
    if lead is not None and school._answers(lead):
        out.append(lead)
    seen = {t.name for t in out}
    try:
        local = [t for t in school.teachers(cfg) if not t.lead and not t.helper]
    except Exception as e:
        log.warning("could not look for local checkers: %s", e)
        local = []
    return out + [t for t in local if t.name not in seen]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower().replace("’", "'"))).strip()


def read_reply(reply: str, passage: str) -> Optional[dict[str, Any]]:
    """A checker's reply, kept only as far as it can be trusted: marks within 1 to 10, faults whose quoted words
    are really in the passage, and a correction that is a rewrite of the passage rather than something else."""
    try:
        d = extract_json(reply)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    marks: dict[str, float] = {}
    raw = d.get("marks") if isinstance(d.get("marks"), dict) else {}
    for a in ASPECTS:
        try:
            marks[a] = max(1.0, min(10.0, float(raw[a])))
        except (KeyError, TypeError, ValueError):
            continue
    if not marks:
        return None
    body = _norm(passage)
    problems = []
    for item in (d.get("problems") or [])[:5]:
        if not isinstance(item, dict):
            continue
        quote, problem = str(item.get("quote") or "").strip(), str(item.get("problem") or "").strip()
        if quote and problem and len(_norm(quote)) >= 3 and _norm(quote) in body:      # the fault can be found in the text
            problems.append({"quote": quote[:200], "problem": problem[:300], "fix": str(item.get("fix") or "").strip()[:300]})
    corrected = str(d.get("corrected") or "").strip()
    n_old, n_new = len(passage.split()), len(corrected.split())
    if not (0.5 * n_old <= n_new <= 1.7 * n_old) or _norm(corrected) == body:
        corrected = ""
    return {"marks": marks, "problems": problems, "corrected": corrected}


def check_text(cfg: Any, text: str, pool: Optional[list[school.Teacher]] = None, source: str = "", step: int = 0,
               n_pieces: Optional[int] = None, panel: Optional[int] = None) -> dict[str, Any]:
    """Have what GaugeLM wrote checked. Returns the averaged marks, the faults found, who checked, and how many
    corrections were kept as lessons."""
    pool = checkers(cfg) if pool is None else pool
    if not pool:
        return {"ok": False, "marks": {}, "overall": None, "problems": [], "checked_by": [], "rows": 0,
                "message": "Nothing could check it: no free helper is connected and no model is running on this computer."}
    parts = pieces(text, int(n_pieces or cfg.helpers.check_pieces))
    if not parts:
        return {"ok": False, "marks": {}, "overall": None, "problems": [], "checked_by": [], "rows": 0,
                "message": "There is too little text to check."}
    panel = max(1, min(len(pool), int(panel or cfg.helpers.check_panel)))
    datasets = Path(cfg.models_dir) / "datasets"
    sums: dict[str, list[float]] = {a: [] for a in ASPECTS}
    problems: list[dict[str, str]] = []
    by: list[str] = []
    rows, out_of_action = 0, []
    for i, passage in enumerate(parts):
        asked = 0
        for t in [p for p in pool if all(p is not g for g in out_of_action)]:
            if asked >= panel:
                break
            try:
                reply = t.ask(CHECK_PROMPT.format(passage=passage), temperature=0.2,
                              max_tokens=int(len(passage.split()) * 2.2) + 300, json_mode=True, purpose="check")
            except Exception as e:
                log.warning("check: %s could not check a passage (%s)", t.name, e)
                if type(e).__name__ in ("HelperBusy", "ClaudeLimit", "ClaudeRefused"):
                    out_of_action.append(t)
                continue
            got = read_reply(reply, passage)
            if got is None:
                continue
            asked += 1
            if t.name not in by:
                by.append(t.name)
            for a, v in got["marks"].items():
                sums[a].append(v)
            for p in got["problems"]:
                if all(_norm(p["quote"]) != _norm(q["quote"]) for q in problems):
                    problems.append({**p, "by": t.name, "piece": str(i + 1)})
            if got["corrected"] and t.teach:
                school._append(datasets / school.FILES["checks"], {
                    "task": "correct", "prompt": CORRECT_TASK.format(passage=passage), "response": got["corrected"],
                    "student": passage, "teacher": t.name, "licence": t.licence, "source": source})
                rows += 1
    marks = {a: round(sum(v) / len(v), 1) for a, v in sums.items() if v}
    if not marks:
        return {"ok": False, "marks": {}, "overall": None, "problems": [], "checked_by": by, "rows": 0,
                "message": "The check gave nothing usable: no checker answered in a form Gauge could read."}
    overall = round(sum(marks.values()) / len(marks), 1)
    state = school._load(datasets)
    hist = state.setdefault("checks", [])
    hist.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": int(step), "overall": overall, "source": source, **marks})
    state["checks"] = hist[-400:]
    school._save(datasets, state)
    msg = (f"Checked by {' and '.join(by)}: {overall} out of 10 ("
           + ", ".join(f"{a} {marks[a]:g}" for a in ASPECTS if a in marks) + ").")
    if problems:
        worst = problems[0]
        msg += f" {len(problems)} fault{'' if len(problems) == 1 else 's'} found; the first: “{worst['quote']}” - {worst['problem']}"
    if rows:
        msg += f" {rows} corrected passage{'' if rows == 1 else 's'} kept as lesson{'' if rows == 1 else 's'} for GaugeLM."
    return {"ok": True, "marks": marks, "overall": overall, "problems": problems, "checked_by": by, "rows": rows,
            "pieces": len(parts), "message": msg}


def trend(datasets_dir: Path) -> dict[str, Any]:
    """How the checkers' marks have moved: the first few checks against the latest few."""
    hist = school._load(Path(datasets_dir)).get("checks") or []
    if not hist:
        return {"checks": 0, "line": "GaugeLM's work has not been checked yet."}
    k = max(1, min(5, len(hist) // 2))
    first = sum(h["overall"] for h in hist[:k]) / k
    now = sum(h["overall"] for h in hist[-k:]) / k
    line = f"Checked {len(hist)} time{'' if len(hist) == 1 else 's'}: {now:.1f} out of 10 lately"
    line += f", {first:.1f} at first." if len(hist) >= 2 else "."
    return {"checks": len(hist), "first": round(first, 1), "now": round(now, 1), "line": line, "history": hist}


def report(result: dict[str, Any]) -> str:
    """A check, written out for the author to read."""
    if not result.get("ok"):
        return str(result.get("message", ""))
    lines = [result["message"].split(" 1 fault")[0].split(f" {len(result['problems'])} fault")[0], ""]
    for p in result["problems"]:
        lines.append(f"“{p['quote']}”")
        lines.append(f"    {p['problem']}" + (f" Fix: {p['fix']}" if p.get("fix") else "") + f"  ({p['by']})")
    if not result["problems"]:
        lines.append("No fault was pointed at that could be found in the text.")
    return "\n".join(lines)


def save_report(project_dir: Path, chapter: int, result: dict[str, Any]) -> Optional[Path]:
    """Keep the check beside the story, so it can be read later."""
    try:
        d = Path(project_dir) / "checks"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"chapter_{int(chapter):04d}.json"
        f.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        return f
    except OSError as e:
        log.warning("could not save the check: %s", e)
        return None
