"""Prediction ledger: honest self-evaluation of the engagement model over time.

Each training cycle records the model's prediction for every novel alongside what was observed
at that moment. Later, predictions old enough are compared with what the novel's numbers became.
If the correlation is near zero, the model has learned nothing you should trust, and the report
says so."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..db import Database
from .engagement import TARGETS, EngagementModel


def record_predictions(db: Database, model: EngagementModel, novels, novel_feats: dict[str, dict[str, float]]) -> int:
    fn = TARGETS[model.target] if model.target in TARGETS else None
    n = 0
    for novel in novels:
        f = novel_feats.get(novel.key)
        if not f:
            continue
        observed = fn(novel) if fn else None
        db.record_prediction(novel.key, model.target, float(model.predict(f)), None if observed is None else float(observed))
        n += 1
    return n


def evaluate(db: Database, target: str, older_than_days: int = 7) -> dict[str, Any]:
    """Compare predictions made at least N days ago with the novels' current numbers."""
    fn = TARGETS.get(target)
    rows = db.predictions(target, older_than_days)
    if not rows or fn is None:
        return {"pairs": 0}
    latest: dict[str, dict[str, Any]] = {}
    for r in rows:  # keep the oldest prediction per novel for the fairest test
        latest.setdefault(r["novel_key"], r)
    preds, actual, growth_pred, growth_act = [], [], [], []
    for key, r in latest.items():
        novel = db.get_novel(key)
        if novel is None:
            continue
        now = fn(novel)
        if now is None:
            continue
        preds.append(r["predicted"])
        actual.append(float(now))
        if r["observed_then"] is not None:
            growth_pred.append(r["predicted"] - r["observed_then"])
            growth_act.append(float(now) - r["observed_then"])
    out: dict[str, Any] = {"pairs": len(preds)}
    if len(preds) >= 3:
        p, a = np.asarray(preds), np.asarray(actual)
        out["correlation"] = round(float(np.corrcoef(p, a)[0, 1]), 3) if p.std() > 0 and a.std() > 0 else None
        out["mae"] = round(float(np.abs(p - a).mean()), 3)
    if len(growth_pred) >= 3:
        gp, ga = np.asarray(growth_pred), np.asarray(growth_act)
        out["growth_correlation"] = round(float(np.corrcoef(gp, ga)[0, 1]), 3) if gp.std() > 0 and ga.std() > 0 else None
    return out
