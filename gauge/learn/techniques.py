"""Technique absorption: learn *craft moves*, not just numbers, from new content.

The engagement model learns what correlates with readers staying. The exemplar library shows
the writer what good openings and endings look like. Neither says *how* a chapter earned its
effect. The absorber reads the strongest chapters in the corpus (newest first, so the library
keeps changing as you ingest) and asks the local model to name the moves: the misdirect, the
reversal placed at the two-thirds mark, the one-line paragraph that lands a reveal, the
dialogue beat that hides exposition. Each move becomes a **technique card** with a short
example, embedded for retrieval. The drafter is handed two or three cards that fit the planned
chapter, with one instruction: use a move like this, in your own way.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..db import Database
from ..llm.client import LLM, generate_json
from ..llm.embeddings import Embedder
from ..llm.tokens import truncate_to_tokens
from ..models import ChapterRecord, NovelRecord
from .engagement import TARGETS

log = logging.getLogger(__name__)

KINDS = ("hook", "structure", "reversal", "reveal", "dialogue", "imagery", "pacing", "ending", "humour", "emotion", "worldbuilding")

TECHNIQUE_PROMPT = """Extract the craft techniques that make this serial chapter work for readers.
Name concrete, reusable moves: not themes, not plot summary. For each move give a 1-2 sentence
description a writer could apply to a different story, and quote or paraphrase the shortest
possible example (<= 40 words).

Chapter from "{title}" ({genres}), chapter {index}:
{text}

Respond with ONLY JSON: {{"techniques": [{{"kind": "one of {kinds}", "name": "3-6 words",
"description": "how the move works and why it hooks", "example": "<= 40 words"}}, ... 3 to 6 items]}}"""


@dataclass
class Technique:
    kind: str
    name: str
    description: str
    example: str
    novel_key: str
    novel_title: str
    chapter_index: int
    engagement: float

    def card(self) -> str:
        return f"[{self.kind}] {self.name}: {self.description} e.g. \"{self.example}\" ({self.novel_title})"


class TechniqueLibrary:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.items: list[Technique] = []
        self.matrix: Optional[np.ndarray] = None

    def add(self, techs: list[Technique]) -> None:
        if not techs:
            return
        self.items.extend(techs)
        new = self.embedder.embed([f"{t.kind} {t.name} {t.description}" for t in techs])
        self.matrix = new if self.matrix is None else np.vstack([self.matrix, new])

    def search(self, query: str, k: int = 3, exclude_kinds: Optional[set[str]] = None,
               diversify: bool = True) -> list[Technique]:
        if not self.items or self.matrix is None:
            return []
        q = self.embedder.embed([query])[0]
        sims = self.matrix @ q + 0.05 * np.asarray([t.engagement for t in self.items]) / max(1.0, max(t.engagement for t in self.items))
        order = np.argsort(-sims)
        out: list[Technique] = []
        kinds_seen: set[str] = set()
        for i in order:
            t = self.items[int(i)]
            if exclude_kinds and t.kind in exclude_kinds:
                continue
            if diversify and t.kind in kinds_seen:
                continue
            kinds_seen.add(t.kind)
            out.append(t)
            if len(out) >= k:
                break
        return out

    def kinds_count(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for t in self.items:
            c[t.kind] = c.get(t.kind, 0) + 1
        return c

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "techniques.json").write_text(json.dumps([asdict(t) for t in self.items], ensure_ascii=False, indent=0), encoding="utf-8")
        if self.matrix is not None:
            np.save(directory / "techniques.npy", self.matrix)
        (directory / "techniques.meta.json").write_text(json.dumps({"embedder": self.embedder.describe()}), encoding="utf-8")

    def load(self, directory: Path) -> bool:
        p = directory / "techniques.json"
        if not p.exists():
            return False
        self.items = [Technique(**d) for d in json.loads(p.read_text(encoding="utf-8"))]
        m = directory / "techniques.npy"
        meta = directory / "techniques.meta.json"
        same_embedder = meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("embedder") == self.embedder.describe()
        if m.exists() and same_embedder and self.items:
            self.matrix = np.load(m)
        elif self.items:
            self.matrix = self.embedder.embed([f"{t.kind} {t.name} {t.description}" for t in self.items])
        return True


def extract_techniques(llm: LLM, novel: NovelRecord, ch: ChapterRecord, engagement: float, max_tokens_in: int = 2500) -> list[Technique]:
    prompt = TECHNIQUE_PROMPT.format(title=novel.title, genres=", ".join(novel.genres + novel.tags) or "web fiction",
                                     index=ch.index, text=truncate_to_tokens(ch.text, max_tokens_in), kinds="|".join(KINDS))
    d = generate_json(llm, prompt, temperature=0.4, max_tokens=900, purpose="absorb")
    out = []
    for t in d.get("techniques", []) if isinstance(d.get("techniques"), list) else []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        kind = str(t.get("kind", "structure")).lower().strip()
        if kind not in KINDS:
            kind = "structure"
        out.append(Technique(kind=kind, name=str(t["name"])[:60], description=str(t.get("description", ""))[:400],
                             example=str(t.get("example", ""))[:300], novel_key=novel.key, novel_title=novel.title,
                             chapter_index=ch.index, engagement=engagement))
    return out[:6]


class Absorber:
    """Incrementally turn the best unread chapters into technique cards."""

    def __init__(self, db: Database, llm: LLM, embedder: Embedder, models_dir: Path, target: str = "log_followers"):
        self.db = db
        self.llm = llm
        self.library = TechniqueLibrary(embedder)
        self.library.load(models_dir)
        self.models_dir = models_dir
        self.target = target

    def candidates(self, limit: int) -> list[tuple[NovelRecord, ChapterRecord, float]]:
        fn = TARGETS[self.target]
        done = self.db.absorbed_keys()
        novels = self.db.novels()
        labelled = [(fn(n), n) for n in novels if fn(n) is not None]
        fallback = sorted(v for v, _ in labelled)[len(labelled) // 4] if labelled else 1.0
        ranked = sorted(novels, key=lambda n: (fn(n) if fn(n) is not None else fallback, n.fetched_at), reverse=True)
        out: list[tuple[NovelRecord, ChapterRecord, float]] = []
        for n in ranked:
            eng = fn(n) if fn(n) is not None else fallback
            # Early chapters teach hooks; recent chapters teach how the author keeps readers. Take both ends.
            chapters = self.db.chapters(n.key)
            picks = chapters[:3] + chapters[-2:] if len(chapters) > 5 else chapters
            for ch in picks:
                key = f"{n.key}#{ch.index}"
                if key in done or len(ch.text.split()) < 250:
                    continue
                out.append((n, ch, float(eng)))
                if len(out) >= limit:
                    return out
        return out

    def run(self, max_chapters: int = 10) -> dict[str, int]:
        cands = self.candidates(max_chapters)
        added = 0
        for novel, ch, eng in cands:
            try:
                techs = extract_techniques(self.llm, novel, ch, eng)
            except Exception as e:
                log.warning("absorb failed for %s#%d: %s", novel.key, ch.index, e)
                techs = []
            self.library.add(techs)
            self.db.mark_absorbed(f"{novel.key}#{ch.index}", len(techs))
            added += len(techs)
        if cands:
            self.library.save(self.models_dir)
        return {"chapters_absorbed": len(cands), "techniques_added": added, "techniques_total": len(self.library.items)}
