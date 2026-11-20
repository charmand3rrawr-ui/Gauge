"""The Research Librarian: your notes and lore, retrieved when a scene needs them. No LLM for briefs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from ...llm.embeddings import Embedder, HashingEmbedder
from .base import SceneContext, StoryAgent


class LibState(BaseModel):
    indexed: dict[str, float] = Field(default_factory=dict)  # file -> mtime


class Librarian(StoryAgent):
    name = "librarian"
    title = "Research Librarian"
    weight = 0.0
    state_model = LibState

    def __init__(self, llm, project_dir: Path, options=None, embedder: Optional[Embedder] = None):
        super().__init__(llm, project_dir, options)
        self.project_dir = project_dir
        self.embedder = embedder or HashingEmbedder(256)
        self.chunks: list[tuple[str, str]] = []  # (source, text)
        self.matrix: Optional[np.ndarray] = None
        self.index()

    @property
    def research_dir(self) -> Path:
        return self.project_dir / "research"

    def index(self) -> int:
        chunks: list[tuple[str, str]] = []
        premise = self.project_dir / "premise.md"
        sources = [premise] if premise.exists() else []
        if self.research_dir.exists():
            sources += sorted(p for p in self.research_dir.rglob("*") if p.suffix.lower() in {".md", ".txt"})
        for p in sources:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            words = text.split()
            for i in range(0, len(words), 180):
                chunk = " ".join(words[i:i + 220])
                if len(chunk.split()) >= 20:
                    chunks.append((p.name, chunk))
        self.chunks = chunks
        self.matrix = self.embedder.embed([c for _, c in chunks]) if chunks else None
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        if not self.chunks or self.matrix is None:
            return []
        q = self.embedder.embed([query])[0]
        order = np.argsort(-(self.matrix @ q))
        return [self.chunks[int(i)] for i in order[:k]]

    def brief(self, ctx: SceneContext) -> str:
        hits = self.search(ctx.plan_text, k=3)
        if not hits:
            return ""
        return "\n".join(f"- from your notes ({src}): {' '.join(text.split()[:70])}..." for src, text in hits)

    def ask(self, question: str) -> str:
        hits = self.search(question, k=4)
        if not hits:
            return "No notes match. Put .md or .txt files in the project's research/ folder."
        return "\n\n".join(f"[{src}] {text}" for src, text in hits)

    def report(self) -> str:
        return f"{len(self.chunks)} indexed chunks from premise.md and {self.research_dir}"
