"""A per-project engagement model: what *your* readers reward, chapter by chapter.

Features are the same chapter features as the corpus model. Labels come from the numbers you
record with `gauge feedback` (views, comments, likes). Two targets are available: log views and
retention (views of the next chapter over this one). Once enough chapters are labelled, the
critic blends this model in ahead of the corpus model.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from ..features import chapter_features
from ..models import ChapterRecord
from .engagement import EngagementModel


def retention_labels(feedback: dict[int, dict]) -> dict[int, float]:
    out = {}
    for n, f in feedback.items():
        nxt = feedback.get(n + 1, {})
        a, b = f.get("views"), nxt.get("views")
        if a and b and a > 0:
            out[n] = min(1.5, b / a)
    return out


def build_table(project, feedback: dict[int, dict], target: str = "retention"):
    labels = retention_labels(feedback) if target == "retention" else {
        n: math.log1p(f["views"]) for n, f in feedback.items() if f.get("views")}
    if target == "comments":
        labels = {n: math.log1p(f["comments"]) for n, f in feedback.items() if f.get("comments") is not None}
    rows, ys, names = [], [], None
    for n, y in sorted(labels.items()):
        text = project.read_chapter(n)
        if not text:
            continue
        f = chapter_features(ChapterRecord(novel_key="own", index=n, title="", text=text))
        if names is None:
            names = sorted(f)
        rows.append([float(f.get(k, 0.0)) for k in names])
        ys.append(float(y))
    if not rows:
        return np.zeros((0, 0)), np.zeros(0), names or []
    return np.asarray(rows), np.asarray(ys), names


class ProjectModel:
    def __init__(self, project_dir: Path, target: str = "retention"):
        self.path = project_dir / "models" / f"project_{target}.json"
        self.target = target
        self.model: Optional[EngagementModel] = EngagementModel.load(self.path)

    def fit(self, project, feedback: dict[int, dict], min_chapters: int = 30, alpha: float = 2.0) -> Optional[EngagementModel]:
        X, y, names = build_table(project, feedback, self.target)
        if len(y) < max(5, min_chapters):
            return None
        self.model = EngagementModel.fit(X, y, names, f"project:{self.target}", alpha=alpha, try_sklearn=False)
        self.model.save(self.path)
        return self.model

    def score01(self, feats: dict[str, float]) -> Optional[float]:
        return self.model.score01(feats) if self.model else None

    def explain(self) -> str:
        return self.model.explain() if self.model else "(no project model yet; record feedback for more chapters)"
