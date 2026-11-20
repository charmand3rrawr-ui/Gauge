"""Human-readable reports on what Gauge has learned."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from .db import Database
from .learn.engagement import TARGETS, EngagementModel


def corpus_summary(db: Database) -> dict[str, object]:
    novels = db.novels()
    by_source: dict[str, int] = {}
    labelled = 0
    for n in novels:
        by_source[n.source] = by_source.get(n.source, 0) + 1
        if n.followers is not None or n.views is not None or n.rating is not None:
            labelled += 1
    return {"novels": len(novels), "labelled": labelled, "chapters": db.total_chapters(), "by_source": by_source,
            "follows": len(db.follows())}


def leaderboard(db: Database, target: str, top: int = 15) -> list[dict[str, object]]:
    fn = TARGETS[target]
    rows = []
    nf = db.features("novel")
    for n in db.novels():
        y = fn(n)
        if y is None:
            continue
        f = nf.get(n.key, {})
        rows.append({"title": n.title[:40], "source": n.source, target: round(y, 2), "followers": n.followers,
                     "rating": n.rating, "chapters": n.chapter_count, "hook": round(f.get("mean_hook", 0), 2),
                     "cliff": round(f.get("mean_cliff", 0), 2), "words": int(f.get("mean_words", 0)),
                     "ch/wk": round(f.get("chapters_per_week", 0), 1)})
    rows.sort(key=lambda r: r[target], reverse=True)
    return rows[:top]


def explain_novel(db: Database, model: Optional[EngagementModel], key: str) -> str:
    n = db.get_novel(key)
    if n is None:
        return f"unknown novel {key}"
    f = db.feature_for(key, "novel")
    if f is None:
        return f"{n.title}: no features yet (run `gauge analyze`)"
    lines = [f"{n.title} by {n.author} [{n.source}]", f"  followers={n.followers} rating={n.rating} views={n.views} "
             f"chapters={n.chapter_count}", f"  mean hook={f.get('mean_hook', 0):.2f} mean cliff={f.get('mean_cliff', 0):.2f} "
             f"words/ch={f.get('mean_words', 0):.0f} dialogue={f.get('mean_dialogue_ratio', 0):.0%} "
             f"ch/week={f.get('chapters_per_week', 0):.1f}"]
    if model is not None:
        lines.append(f"  predicted {model.target}={model.predict(f):.2f}  (score01={model.score01(f):.2f})")
        lines.append("  biggest drivers:")
        for name, c in model.contributions(f)[:8]:
            lines.append(f"    {name:<28} {c:+.3f}")
    return "\n".join(lines)


def compare_to_corpus(feats: dict[str, float], corpus: dict[str, dict[str, float]],
                      keys: Sequence[str] = ("hook", "cliff", "dialogue_ratio", "words", "one_line_par_ratio",
                                             "sent_len_mean", "tension_rising")) -> list[dict[str, object]]:
    """Where does one chapter sit against every ingested chapter? Percentiles per feature."""
    import numpy as np

    out = []
    for k in keys:
        vals = np.asarray([c[k] for c in corpus.values() if k in c])
        if len(vals) == 0:
            continue
        v = feats.get(k, 0.0)
        pct = float((vals < v).mean())
        out.append({"feature": k, "value": round(v, 3), "corpus_median": round(float(np.median(vals)), 3),
                    "percentile": round(100 * pct)})
    return out
