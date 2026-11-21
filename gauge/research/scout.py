"""The Scout: finds material that makes a scene concrete and a world believable.

Given questions ("how were medieval couriers paid", "Lisbon in November", "what does a burn
ward smell like"), it searches the open sources, caches every answer in SQLite, and files what
it finds under ``projects/<slug>/research/scout/`` as Markdown with attribution. The Librarian
indexes that folder, so everything the Scout learns is available to every later chapter.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from ..config import GaugeConfig, IngestConfig
from ..db import Database
from ..ingest.http import Fetcher
from ..util import utcnow
from .sources import (
    ArxivSource,
    ClimateSource,
    GutenbergSource,
    MediaWikiSource,
    SearxSource,
    Snippet,
)

log = logging.getLogger(__name__)

SOURCE_HINTS = {
    "wikipedia": "facts: history, technology, professions, biology, geography, institutions",
    "wikivoyage": "place texture: streets, food, markets, climate, how people move around",
    "gutenberg": "period prose and vocabulary from public-domain books (diaries, letters, novels)",
    "arxiv": "hard science and engineering for sci-fi plausibility",
    "climate": "real weather for a real place and month ('Lisbon in November')",
    "searx": "general web search (only if you run a SearXNG instance)",
}


def build_sources(cfg: GaugeConfig, fetcher: Optional[Fetcher] = None) -> dict[str, Any]:
    """Instantiate the enabled sources. ``GAUGE_OFFLINE=1`` disables all of them (airplane mode, tests)."""
    import os

    if os.environ.get("GAUGE_OFFLINE"):
        return {}
    r = cfg.research
    # robots.txt is honoured by default. Wikipedia/Wikivoyage go through api.wikimedia.org, whose
    # robots.txt permits clients; Gutendex allows our user agent. Open-Meteo and arXiv disallow all
    # automated access in robots.txt even for their APIs, so they only work if you set
    # research.respect_robots_txt=false yourself (their API terms do allow identified, slow clients).
    f = fetcher or Fetcher(IngestConfig(user_agent=cfg.ingest.user_agent + " (research)", request_delay_seconds=r.request_delay_seconds,
                                        respect_robots_txt=r.respect_robots_txt))
    out: dict[str, Any] = {}
    if "wikipedia" in r.sources:
        out["wikipedia"] = MediaWikiSource(f, "wikipedia", "en")
    if "wikivoyage" in r.sources:
        out["wikivoyage"] = MediaWikiSource(f, "wikivoyage", "en")
    if "gutenberg" in r.sources:
        out["gutenberg"] = GutenbergSource(f)
    if "arxiv" in r.sources:
        out["arxiv"] = ArxivSource(f)
    if "climate" in r.sources:
        out["climate"] = ClimateSource(f)
    if "searx" in r.sources and r.searx_url:
        out["searx"] = SearxSource(f, r.searx_url)
    return out


@dataclass
class Finding:
    query: str
    source: str
    snippets: list[Snippet]
    cached: bool = False
    why: str = ""


@dataclass
class Dossier:
    findings: list[Finding] = field(default_factory=list)

    def snippets(self) -> list[Snippet]:
        return [s for f in self.findings for s in f.snippets]

    def cards(self, max_items: int = 6, max_words: int = 80) -> str:
        return "\n".join(f"- {s.card(max_words)}" for s in self.snippets()[:max_items])

    def markdown(self, title: str) -> str:
        lines = [f"# {title}", "", f"_Collected by the Scout on {utcnow().date().isoformat()}. Facts and texture only; never copy sentences into the story._", ""]
        for f in self.findings:
            lines.append(f"## {f.query}  ({f.source})")
            if f.why:
                lines.append(f"_{f.why}_")
            lines.append("")
            for s in f.snippets:
                lines.append(f"**{s.title}** - {s.url}  ")
                lines.append(f"{s.text}")
                lines.append(f"<sub>{s.license}</sub>")
                lines.append("")
        return "\n".join(lines)


class Scout:
    def __init__(self, cfg: GaugeConfig, db: Database, sources: Optional[dict[str, Any]] = None):
        self.cfg = cfg
        self.db = db
        self.sources = sources if sources is not None else build_sources(cfg)

    # ---- searching with cache ------------------------------------------------
    def search(self, query: str, source: str = "wikipedia", k: Optional[int] = None, why: str = "") -> Finding:
        k = k or self.cfg.research.results_per_query
        src = self.sources.get(source)
        if src is None:
            # Unknown or disabled source: fall back to the first available one.
            if not self.sources:
                return Finding(query, source, [], why=why)
            source, src = next(iter(self.sources.items()))
        key = hashlib.sha1(f"{source}|{query.lower().strip()}|{k}".encode()).hexdigest()
        cached = self.db.research_cached(key, max_age_days=self.cfg.research.cache_days)
        if cached is not None:
            return Finding(query, source, [Snippet(**d) for d in cached], cached=True, why=why)
        try:
            snippets = src.search(query, k)
        except Exception as e:  # a source outage must never stop a chapter
            log.warning("scout: %s failed for %r: %s", source, query, e)
            snippets = []
        self.db.research_store(key, source, query, [s.to_dict() for s in snippets])
        return Finding(query, source, snippets, why=why)

    def research(self, queries: list[dict[str, str]], project: Optional[str] = None, chapter: Optional[int] = None) -> Dossier:
        """queries: [{"query": ..., "source": ..., "why": ...}]"""
        d = Dossier()
        for q in queries[: self.cfg.research.max_queries_per_chapter]:
            if not q.get("query"):
                continue
            f = self.search(str(q["query"]), str(q.get("source", "wikipedia")), why=str(q.get("why", "")))
            d.findings.append(f)
            self.db.research_log(project or "", chapter, f.query, f.source, len(f.snippets), f.cached)
        return d

    # ---- filing into the project ---------------------------------------------
    @staticmethod
    def research_dir(project_dir: Path) -> Path:
        return project_dir / "research" / "scout"

    def file(self, project_dir: Path, dossier: Dossier, name: str) -> Optional[Path]:
        if not dossier.snippets():
            return None
        d = self.research_dir(project_dir)
        d.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "notes"
        path = d / f"{slug}.md"
        path.write_text(dossier.markdown(name), encoding="utf-8")
        self._update_bibliography(project_dir, dossier)
        return path

    def _update_bibliography(self, project_dir: Path, dossier: Dossier) -> None:
        bib = project_dir / "research" / "sources.md"
        existing = bib.read_text(encoding="utf-8") if bib.exists() else "# Sources consulted\n\n"
        for s in dossier.snippets():
            line = f"- [{s.source}] {s.title}: {s.url} ({s.license})\n"
            if line not in existing:
                existing += line
        bib.parent.mkdir(parents=True, exist_ok=True)
        bib.write_text(existing, encoding="utf-8")

    # ---- public-domain books into the library ------------------------------------
    def import_gutenberg(self, runner, query: str, max_books: int = 1) -> list[str]:
        """Pull public-domain novels into the corpus so analysis and technique absorption can
        learn from them. Returns the keys of the imported novels."""
        src = self.sources.get("gutenberg")
        if src is None:
            raise ValueError("gutenberg source is disabled in research.sources")
        from ..ingest.local import split_single_file
        from ..models import ChapterRecord, NovelRecord

        keys = []
        for book in src.find_books(query, k=max_books):
            text = src.book_text(book)
            if len(text.split()) < 2000:
                continue
            chapters = split_single_file(text)
            novel = NovelRecord(source="gutenberg", source_id=str(book["id"]), url=f"https://www.gutenberg.org/ebooks/{book['id']}",
                                title=book["title"], author=", ".join(book["authors"]), genres=[s.split(" -- ")[0] for s in book["subjects"][:4]],
                                tags=["public domain"], status="completed", views=book.get("downloads"), chapter_count=len(chapters))
            self.db.upsert_novel(novel)
            for i, (title, body) in enumerate(chapters, 1):
                self.db.upsert_chapter(ChapterRecord(novel_key=novel.key, index=i, source_chapter_id=str(i), title=title, text=body))
            keys.append(novel.key)
            log.info("imported %s (%d chapters) from Project Gutenberg", novel.title, len(chapters))
        return keys

    @staticmethod
    def expired(at: str, days: int) -> bool:
        try:
            return datetime.fromisoformat(at) < utcnow() - timedelta(days=days)
        except ValueError:
            return True
