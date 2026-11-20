"""Incremental ingestion. Fetches novels and only the chapters it has not seen yet, so the same
command keeps a corpus current as new chapters are released ("future" chapters)."""
from __future__ import annotations

import logging
from typing import Optional

from ..config import GaugeConfig
from ..db import Database
from ..models import NovelRecord
from ..util import utcnow
from .base import Source, chapter_batches
from .http import Fetcher
from .royalroad import RoyalRoad
from .scribblehub import ScribbleHub

log = logging.getLogger(__name__)


def build_sources(cfg: GaugeConfig, fetcher: Optional[Fetcher] = None) -> list[Source]:
    f = fetcher or Fetcher(cfg.ingest)
    return [RoyalRoad(f), ScribbleHub(f)]


def source_for_url(sources: list[Source], url: str) -> Optional[Source]:
    for s in sources:
        if s.matches(url):
            return s
    return None


class IngestRunner:
    def __init__(self, cfg: GaugeConfig, db: Database, sources: Optional[list[Source]] = None):
        self.cfg = cfg
        self.db = db
        self.sources = sources or build_sources(cfg)

    def discover(self, source_name: str, listing: str, pages: int = 1) -> list[str]:
        src = next((s for s in self.sources if s.name == source_name), None)
        if src is None:
            raise ValueError(f"unknown source {source_name}; have {[s.name for s in self.sources]}")
        urls = src.discover(listing, pages)
        for u in urls:
            self.db.follow(u, source_name)
        return urls

    def follow(self, url: str) -> str:
        src = source_for_url(self.sources, url)
        if src is None:
            raise ValueError(f"no connector matches {url}")
        self.db.follow(url, src.name)
        return src.name

    def ingest_url(self, url: str, max_chapters: Optional[int] = None) -> dict[str, int]:
        src = source_for_url(self.sources, url)
        if src is None:
            raise ValueError(f"no connector matches {url}")
        novel = src.fetch_novel(url)
        if novel is None:
            log.warning("could not fetch %s", url)
            return {"novels": 0, "chapters": 0}
        self.db.upsert_novel(novel)
        refs = src.list_chapters(novel)
        known = self.db.known_chapter_ids(novel.key)
        budget = max_chapters if max_chapters is not None else self.cfg.ingest.max_chapters_per_novel
        remaining = max(0, budget - len(known))
        todo = chapter_batches(refs, known, remaining)
        got = 0
        for ref in todo:
            ch = src.fetch_chapter(novel, ref)
            if ch is None:
                continue
            self.db.upsert_chapter(ch)
            got += 1
        self.db.touch_follow(url)
        log.info("%s: %d new chapters (%d total listed)", novel.title, got, len(refs))
        return {"novels": 1, "chapters": got}

    def run(self, max_novels: Optional[int] = None) -> dict[str, int]:
        started = utcnow()
        totals = {"novels": 0, "chapters": 0, "errors": 0}
        follows = self.db.follows()
        limit = max_novels if max_novels is not None else self.cfg.ingest.max_novels_per_run
        for url, _src in follows[:limit]:
            try:
                r = self.ingest_url(url)
                totals["novels"] += r["novels"]
                totals["chapters"] += r["chapters"]
            except Exception as e:  # keep the loop alive; one bad page must not stop the daemon
                log.exception("ingest failed for %s: %s", url, e)
                totals["errors"] += 1
        self.db.log_run("ingest", started, totals)
        return totals

    def import_local(self, path: str) -> NovelRecord:
        from .local import import_path

        novel, chapters = import_path(path)
        self.db.upsert_novel(novel)
        for ch in chapters:
            self.db.upsert_chapter(ch)
        return novel
