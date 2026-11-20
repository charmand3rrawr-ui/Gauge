"""Engagement model: which measurable properties of a novel predict that readers stay?

A ridge regression over standardised features. Deliberately simple: with a few hundred novels
you cannot support anything fancier without fooling yourself, and a linear model gives
*readable* coefficients ("cliffhanger strength is worth +0.31 log-followers"). If scikit-learn
is installed a gradient-boosted model is fitted alongside for better scoring; the ridge model
is always kept for explanation.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import NovelRecord

TARGETS = {
    "log_followers": lambda n: math.log1p(n.followers) if n.followers is not None else None,
    "log_views": lambda n: math.log1p(n.views) if n.views is not None else None,
    "log_avg_views": lambda n: math.log1p(n.average_views) if n.average_views is not None else None,
    "rating": lambda n: n.rating,
    "log_favorites": lambda n: math.log1p(n.favorites) if n.favorites is not None else None,
    "user_rating": lambda n: n.user_rating,
}


def build_training_table(novels: Sequence[NovelRecord], novel_feats: dict[str, dict[str, float]],
                         target: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Returns X, y, feature_names, novel_keys for novels that have both features and a target."""
    fn = TARGETS[target]
    rows, ys, keys = [], [], []
    names = sorted({k for f in novel_feats.values() for k in f})
    for n in novels:
        f = novel_feats.get(n.key)
        y = fn(n)
        if f is None or y is None:
            continue
        rows.append([float(f.get(k, 0.0)) for k in names])
        ys.append(float(y))
        keys.append(n.key)
    if not rows:
        return np.zeros((0, len(names))), np.zeros(0), names, keys
    return np.asarray(rows, dtype=np.float64), np.asarray(ys, dtype=np.float64), names, keys


@dataclass
class EngagementModel:
    target: str
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    y_mean: float
    y_std: float
    n_train: int = 0
    r2_train: float = 0.0
    r2_cv: float = 0.0
    alpha: float = 1.0
    gbm: object = field(default=None, repr=False)

    # ---- fitting -----------------------------------------------------------
    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, names: list[str], target: str,
            alpha: float = 1.0, try_sklearn: bool = True) -> EngagementModel:
        if len(y) < 5:
            raise ValueError(f"need at least 5 labelled novels, have {len(y)}")
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        Xs = (X - mean) / std
        y_mean, y_std = float(y.mean()), float(y.std() or 1.0)
        ys = (y - y_mean) / y_std
        coef = cls._ridge(Xs, ys, alpha)
        pred = Xs @ coef
        r2 = 1.0 - float(((ys - pred) ** 2).sum() / max(1e-9, ((ys - ys.mean()) ** 2).sum()))
        r2_cv = cls._cv_r2(Xs, ys, alpha)
        model = cls(target=target, feature_names=names, mean=mean, std=std, coef=coef, intercept=0.0,
                    y_mean=y_mean, y_std=y_std, n_train=len(y), r2_train=r2, r2_cv=r2_cv, alpha=alpha)
        if try_sklearn and len(y) >= 40:
            try:
                from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

                gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                subsample=0.8, random_state=7)
                gbm.fit(Xs, ys)
                model.gbm = gbm
            except Exception:  # sklearn missing or failed: ridge alone is fine
                model.gbm = None
        return model

    @staticmethod
    def _ridge(Xs: np.ndarray, ys: np.ndarray, alpha: float) -> np.ndarray:
        d = Xs.shape[1]
        A = Xs.T @ Xs + alpha * np.eye(d)
        return np.linalg.solve(A, Xs.T @ ys)

    @classmethod
    def _cv_r2(cls, Xs: np.ndarray, ys: np.ndarray, alpha: float, k: int = 5) -> float:
        n = len(ys)
        if n < 10:
            return float("nan")
        rng = np.random.default_rng(13)
        idx = rng.permutation(n)
        folds = np.array_split(idx, k)
        preds = np.zeros(n)
        for f in folds:
            train = np.setdiff1d(idx, f)
            coef = cls._ridge(Xs[train], ys[train], alpha)
            preds[f] = Xs[f] @ coef
        return 1.0 - float(((ys - preds) ** 2).sum() / max(1e-9, ((ys - ys.mean()) ** 2).sum()))

    # ---- scoring -----------------------------------------------------------
    def _vec(self, feats: dict[str, float]) -> np.ndarray:
        x = np.asarray([float(feats.get(k, 0.0)) for k in self.feature_names])
        return (x - self.mean) / self.std

    def predict(self, feats: dict[str, float]) -> float:
        """Predicted target in original units (e.g. log followers)."""
        xs = self._vec(feats)
        z = float(xs @ self.coef)
        if self.gbm is not None:
            z = 0.5 * z + 0.5 * float(self.gbm.predict(xs[None, :])[0])
        return z * self.y_std + self.y_mean

    def score01(self, feats: dict[str, float]) -> float:
        """Squash prediction to 0..1 relative to the training distribution (percentile-ish)."""
        z = (self.predict(feats) - self.y_mean) / (self.y_std or 1.0)
        return float(1.0 / (1.0 + math.exp(-z)))

    def contributions(self, feats: dict[str, float]) -> list[tuple[str, float]]:
        xs = self._vec(feats)
        contrib = xs * self.coef * self.y_std
        order = np.argsort(-np.abs(contrib))
        return [(self.feature_names[i], float(contrib[i])) for i in order]

    def importances(self, top: int = 25) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.coef))
        return [(self.feature_names[i], float(self.coef[i] * self.y_std)) for i in order[:top]]

    # ---- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target": self.target, "feature_names": self.feature_names, "mean": self.mean.tolist(),
            "std": self.std.tolist(), "coef": self.coef.tolist(), "intercept": self.intercept,
            "y_mean": self.y_mean, "y_std": self.y_std, "n_train": self.n_train,
            "r2_train": self.r2_train, "r2_cv": self.r2_cv, "alpha": self.alpha,
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        if self.gbm is not None:
            try:
                import pickle

                (path.with_suffix(".gbm.pkl")).write_bytes(pickle.dumps(self.gbm))
            except Exception:
                pass

    @classmethod
    def load(cls, path: Path) -> Optional[EngagementModel]:
        if not path.exists():
            return None
        p = json.loads(path.read_text(encoding="utf-8"))
        m = cls(target=p["target"], feature_names=p["feature_names"], mean=np.asarray(p["mean"]),
                std=np.asarray(p["std"]), coef=np.asarray(p["coef"]), intercept=p["intercept"],
                y_mean=p["y_mean"], y_std=p["y_std"], n_train=p["n_train"], r2_train=p["r2_train"],
                r2_cv=p["r2_cv"], alpha=p.get("alpha", 1.0))
        gbm_path = path.with_suffix(".gbm.pkl")
        if gbm_path.exists():
            try:
                import pickle

                m.gbm = pickle.loads(gbm_path.read_bytes())
            except Exception:
                m.gbm = None
        return m

    def explain(self, top: int = 15) -> str:
        lines = [f"target={self.target}  n={self.n_train}  R2(train)={self.r2_train:.2f}  "
                 f"R2(5-fold)={self.r2_cv:.2f}"]
        for name, w in self.importances(top):
            lines.append(f"  {'+' if w >= 0 else '-'} {name:<28} {w:+.3f}")
        return "\n".join(lines)
