"""Distil the learned model into a human-readable style guide the LLM can follow.

The guide is regenerated every training cycle, so what the writer is told to do tracks what
the data currently says works. It is also saved to ``models/style_guide.md`` for you to read.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .engagement import EngagementModel

# Plain-English descriptions for the features that matter most to a writer.
_DESCRIPTIONS = {
    "mean_hook": "open every chapter with an immediate question, action or line of dialogue",
    "early_hook": "the first three chapters must hook hardest of all",
    "mean_cliff": "end chapters unresolved: a reveal, a threat, or a line that demands the next chapter",
    "early_cliff": "end chapters 1-3 on strong cliffhangers",
    "mean_dialogue_ratio": "dialogue share of the text",
    "mean_short_sentence_ratio": "share of very short sentences (punch)",
    "mean_one_line_par_ratio": "share of one-line paragraphs (white space, speed)",
    "mean_sent_len_mean": "average sentence length",
    "mean_par_len_mean": "average paragraph length",
    "mean_words": "chapter length in words",
    "mean_readability": "readability (higher = simpler prose)",
    "mean_tension_rising": "tension should climb through the chapter rather than peak early",
    "mean_tension_final": "tension in the final tenth of the chapter",
    "mean_reveal_density": "frequency of reveals / twists",
    "mean_progression_density": "frequency of visible progression (levels, skills, ranks, power)",
    "mean_warmth_density": "frequency of warm, emotional payoff beats",
    "mean_first_person_ratio": "first-person narration",
    "mean_third_person_ratio": "third-person narration",
    "mean_adverb_ratio": "adverb use",
    "chapters_per_week": "release cadence (chapters per week)",
    "release_gap_std_days": "irregularity of release schedule",
    "blurb_question": "the blurb poses a direct question",
    "blurb_progression": "the blurb signals progression / power fantasy",
    "blurb_words": "blurb length",
    "mean_title_has_subtitle": "chapter titles carry a subtitle, not just a number",
}


def _quantile_targets(novel_feats: dict[str, dict[str, float]], keys: Sequence[str],
                      top_keys: Sequence[str]) -> dict[str, float]:
    """Median feature value among the top-engagement novels: concrete numbers to aim for."""
    out = {}
    for k in keys:
        vals = [novel_feats[n][k] for n in top_keys if n in novel_feats and k in novel_feats[n]]
        if vals:
            out[k] = float(np.median(vals))
    return out


def distil_style_guide(model: EngagementModel, novel_feats: dict[str, dict[str, float]],
                       top_novel_keys: Sequence[str], top_n: int = 12) -> str:
    imps = model.importances(top=60)
    positives = [(k, w) for k, w in imps if w > 0 and k in _DESCRIPTIONS][:top_n]
    negatives = [(k, w) for k, w in imps if w < 0 and k in _DESCRIPTIONS][:6]
    targets = _quantile_targets(novel_feats, [k for k, _ in positives + negatives], top_novel_keys)

    lines: list[str] = []
    lines.append("# What the data says keeps readers clicking Next")
    lines.append("")
    lines.append(f"Learned from {model.n_train} novels. Target: {model.target}. "
                 f"Cross-validated R2: {model.r2_cv:.2f}. Treat this as evidence, not law.")
    lines.append("")
    lines.append("## Do more of this (strongest first)")
    for k, w in positives:
        tgt = targets.get(k)
        tgt_s = f" Aim for about {tgt:.2f}." if tgt is not None and "ratio" not in k else (
            f" Top novels average {tgt:.0%}." if tgt is not None else "")
        if k in {"mean_words", "chapters_per_week", "blurb_words", "mean_sent_len_mean", "mean_par_len_mean"} and tgt is not None:
            tgt_s = f" Top novels average {tgt:.0f}."
        lines.append(f"- **{_DESCRIPTIONS[k]}** (weight {w:+.2f}).{tgt_s}")
    if negatives:
        lines.append("")
        lines.append("## Do less of this")
        for k, w in negatives:
            lines.append(f"- {_DESCRIPTIONS[k]} (weight {w:+.2f})")
    lines.append("")
    lines.append("## Non-negotiables for a serial chapter")
    lines.append("- The first line creates a question. The last line refuses to answer one.")
    lines.append("- Every chapter changes something: a fact, a relationship, a power level, a plan.")
    lines.append("- Promise early, pay off often, and open a bigger promise each time you pay one off.")
    return "\n".join(lines)


def compact_guide_for_prompt(guide_md: str, max_lines: int = 18) -> str:
    """Trim the guide to the bullet points for use inside a system prompt."""
    bullets = [ln for ln in guide_md.splitlines() if ln.startswith("- ")]
    return "\n".join(bullets[:max_lines])
