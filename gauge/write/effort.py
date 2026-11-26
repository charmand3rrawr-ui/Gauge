"""Thinking effort: one dial for how hard Gauge works on each chapter.

A model does not get cleverer by being asked nicely, but the same model does better work when it is
given more attempts, more rounds of criticism and more alternatives to choose between. Gauge already
had a dozen settings that each buy a little quality for a little time. This puts them on one dial.

| level   | what it does                                                                          |
| ------- | ------------------------------------------------------------------------------------- |
| quick   | one attempt, no revision, no alternatives: a first draft, fast                         |
| normal  | the defaults: revise up to twice, three directions considered, the scene played out    |
| deep    | two attempts at each piece, three revisions, five directions, titles and hooks tested  |
| maximum | three attempts, five revisions, six directions, a higher bar before a chapter passes   |

More effort is more time, roughly in proportion to the number in `cost`. It raises the floor (fewer
bad chapters) more than the ceiling: it cannot make a model know what it does not know.
"""
from __future__ import annotations

from typing import Any

LEVELS: dict[str, dict[str, Any]] = {
    "quick": {"best_of": 1, "max_revisions": 0, "min_acceptable_score": 0.5, "muse_enabled": False, "muse_directions": 2,
              "ab_titles": False, "ab_hooks": False, "improv_enabled": False, "improv_rounds": 1, "cost": 0.4},
    "normal": {"best_of": 1, "max_revisions": 2, "min_acceptable_score": 0.62, "muse_enabled": True, "muse_directions": 3,
               "ab_titles": False, "ab_hooks": False, "improv_enabled": True, "improv_rounds": 2, "cost": 1.0},
    "deep": {"best_of": 2, "max_revisions": 3, "min_acceptable_score": 0.68, "muse_enabled": True, "muse_directions": 5,
             "ab_titles": True, "ab_hooks": True, "improv_enabled": True, "improv_rounds": 3, "cost": 2.5},
    "maximum": {"best_of": 3, "max_revisions": 5, "min_acceptable_score": 0.72, "muse_enabled": True, "muse_directions": 6,
                "ab_titles": True, "ab_hooks": True, "improv_enabled": True, "improv_rounds": 3, "cost": 5.0},
}
ORDER = ["quick", "normal", "deep", "maximum"]
SAYS = {"quick": "a fast first draft: one attempt, no revision",
        "normal": "the usual care: up to two revisions, three directions considered",
        "deep": "two attempts at each piece, three revisions, five directions, titles and openings tested",
        "maximum": "three attempts at each piece, five revisions, six directions, and a higher bar to pass"}


def apply(cfg: Any, level: str) -> dict[str, Any]:
    """Set the dial. With GaugeLM as the writer the number of attempts never goes below four (it is fast
    and uneven) and the writing room stays lean; with any other writer deep and maximum use the full room."""
    level = str(level).strip().lower()
    if level not in LEVELS:
        raise ValueError(f"effort is one of: {', '.join(ORDER)}")
    want = LEVELS[level]
    w = cfg.write
    for key, value in want.items():
        if key != "cost":
            setattr(w, key, value)
    native = cfg.llm.provider == "native"
    if native:
        w.best_of = max(4, want["best_of"] * 2) if level != "quick" else 2
    elif level in ("deep", "maximum"):
        cfg.agents.profile = "full"
    w.effort = level
    return {"level": level, "cost": want["cost"], "attempts": w.best_of, "revisions": w.max_revisions,
            "message": f"Thinking effort is {level}: {SAYS[level]}. Expect a chapter to take about "
                       f"{want['cost']:g} times as long as at normal."}


def current(cfg: Any) -> str:
    """The level the settings match now, or 'custom' when they have been changed by hand since."""
    level = str(getattr(cfg.write, "effort", "normal") or "normal")
    want = LEVELS.get(level)
    if want is None:
        return "custom"
    w = cfg.write
    same = all(getattr(w, k) == v for k, v in want.items() if k not in ("cost", "best_of"))
    return level if same else "custom"
