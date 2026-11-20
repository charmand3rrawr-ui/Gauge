"""Exemplar library: the openings and endings of chapters from novels that readers love.

Stored as small JSON + a numpy matrix of embeddings so retrieval is instant on a laptop.
Used as few-shot context when planning and drafting: *show* the model what a hook that worked
looks like, instead of describing it.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..llm.embeddings import Embedder
from ..text import head, tail


@dataclass
class Exemplar:
    novel_key: str
    novel_title: str
    chapter_index: int
    kind: str  # "opening" | "ending"
    text: str
    engagement: float  # the target value of the novel (e.g. log followers)
    local_score: float  # hook or cliff score of this specific passage
    genres: list[str]


class ExemplarLibrary:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.items: list[Exemplar] = []
        self.matrix: Optional[np.ndarray] = None

    # ---- build -------------------------------------------------------------
    def build(self, candidates: Sequence[Exemplar], keep: int = 200) -> None:
        # Favour passages that are both from popular novels AND locally strong.
        ranked = sorted(candidates, key=lambda e: e.engagement * (0.5 + e.local_score), reverse=True)
        # Diversity: at most 3 passages per novel per kind.
        seen: dict[str, int] = {}
        chosen: list[Exemplar] = []
        for e in ranked:
            k = f"{e.novel_key}:{e.kind}"
            if seen.get(k, 0) >= 3:
                continue
            seen[k] = seen.get(k, 0) + 1
            chosen.append(e)
            if len(chosen) >= keep:
                break
        self.items = chosen
        self.matrix = self.embedder.embed([e.text for e in chosen]) if chosen else None

    # ---- query -------------------------------------------------------------
    def search(self, query: str, kind: Optional[str] = None, k: int = 4,
               genres: Optional[Sequence[str]] = None) -> list[Exemplar]:
        if not self.items or self.matrix is None:
            return []
        q = self.embedder.embed([query])[0]
        sims = self.matrix @ q
        if genres:
            gl = {g.lower() for g in genres}
            boost = np.asarray([0.1 if gl & {x.lower() for x in e.genres} else 0.0 for e in self.items])
            sims = sims + boost
        order = np.argsort(-sims)
        out = []
        for i in order:
            e = self.items[int(i)]
            if kind and e.kind != kind:
                continue
            out.append(e)
            if len(out) >= k:
                break
        return out

    # ---- persistence -------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "exemplars.json").write_text(
            json.dumps([asdict(e) for e in self.items], ensure_ascii=False, indent=0), encoding="utf-8")
        if self.matrix is not None:
            np.save(directory / "exemplars.npy", self.matrix)
        (directory / "exemplars.meta.json").write_text(
            json.dumps({"embedder": self.embedder.describe(), "dim": int(self.embedder.dim)}), encoding="utf-8")

    def load(self, directory: Path) -> bool:
        p = directory / "exemplars.json"
        if not p.exists():
            return False
        self.items = [Exemplar(**d) for d in json.loads(p.read_text(encoding="utf-8"))]
        m = directory / "exemplars.npy"
        if m.exists():
            self.matrix = np.load(m)
            meta_p = directory / "exemplars.meta.json"
            if meta_p.exists():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                if meta.get("embedder") != self.embedder.describe():
                    # Embedder changed since the library was built; re-embed.
                    self.matrix = self.embedder.embed([e.text for e in self.items]) if self.items else None
        elif self.items:
            self.matrix = self.embedder.embed([e.text for e in self.items])
        return True


def make_candidates(novel_key: str, novel_title: str, genres: list[str], engagement: float,
                    chapters: Sequence, chapter_feats: Sequence[dict[str, float]]) -> list[Exemplar]:
    out = []
    for ch, f in zip(chapters, chapter_feats):
        out.append(Exemplar(novel_key, novel_title, ch.index, "opening", head(ch.text, 180),
                            engagement, f.get("hook", 0.0), genres))
        out.append(Exemplar(novel_key, novel_title, ch.index, "ending", tail(ch.text, 180),
                            engagement, f.get("cliff", 0.0), genres))
    return out
