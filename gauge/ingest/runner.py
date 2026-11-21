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
            self._sanitize_chapter(ch, novel.key)
            self.db.upsert_chapter(ch)
            if ch.comment_texts:
                self.db.add_comments(novel.key, ch.index, ch.comment_texts)
            got += 1
        self.db.touch_follow(url)
        log.info("%s: %d new chapters (%d total listed)", novel.title, got, len(refs))
        return {"novels": 1, "chapters": got}

    def _sanitize_chapter(self, ch, novel_key: str) -> None:
        """Untrusted text is data: strip hidden characters, neutralise instruction-shaped spans, log what was found."""
        from ..security.sanitize import scrub

        action = getattr(getattr(self.cfg, "security", None), "injection_action", "scrub")
        r = scrub(ch.text, action)
        ch.text = r.text
        cleaned_comments = []
        flagged = list(r.findings)
        for cm in ch.comment_texts or []:
            rc = scrub(str(cm.get("text", "")), action)
            if rc.text:
                cleaned_comments.append({**cm, "text": rc.text})
            flagged += rc.findings
        ch.comment_texts = cleaned_comments
        if flagged and max(f["severity"] for f in flagged) >= 2:
            self._audit("injection", f"{novel_key}#{ch.index}: " + "; ".join(f"{f['label']} ({f['match'][:40]})" for f in flagged[:3]),
                        2, {"count": len(flagged)})

    def _audit(self, kind: str, detail: str, sev: int, data: Optional[dict] = None) -> None:
        try:
            from ..security.audit import AuditLog

            AuditLog(self.cfg.security_dir / "audit.jsonl").event(kind, detail, sev, data)
        except Exception as e:  # auditing must never break ingestion
            log.debug("audit failed: %s", e)

    def run(self, max_novels: Optional[int] = None) -> dict[str, int]:
        started = utcnow()
        totals = {"novels": 0, "chapters": 0, "errors": 0}
        follows = self.db.follows()
        limit = max_novels if max_novels is not None else self.cfg.ingest.max_novels_per_run
        for url, src in follows[:limit]:
            try:
                if src == "local":
                    for r in self.import_local(url):
                        totals["novels"] += 1
                        totals["chapters"] += int(r["new"]) + int(r["changed"])
                    self.db.touch_follow(url)
                    continue
                r = self.ingest_url(url)
                totals["novels"] += r["novels"]
                totals["chapters"] += r["chapters"]
            except Exception as e:  # keep the loop alive; one bad page must not stop the daemon
                log.exception("ingest failed for %s: %s", url, e)
                totals["errors"] += 1
        self.db.log_run("ingest", started, totals)
        return totals

    # ---- local folders -----------------------------------------------------
    def watch(self, path: str) -> str:
        """Register a local novel or library folder; every run re-scans it for new/edited chapters."""
        from pathlib import Path

        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        self.db.follow(str(p), "local")
        return str(p)

    def import_local(self, path: str, library: Optional[bool] = None,
                     meta_override: Optional[dict[str, object]] = None) -> list[dict[str, object]]:
        """Import a novel or library from disk. Incremental: chapters whose text is unchanged are
        skipped, edited chapters are replaced, chapters removed on disk are removed here."""
        from .local import import_path

        results: list[dict[str, object]] = []
        for imp in import_path(path, meta_override=meta_override, library=library):
            n = imp.novel
            self.db.upsert_novel(n)
            have = self.db.chapter_hashes(n.key)
            new = changed = 0
            for ch in imp.chapters:
                prev = have.get(ch.index)
                if prev == ch.content_hash:
                    continue
                self._sanitize_chapter(ch, n.key)
                self.db.upsert_chapter(ch)
                if prev is None:
                    new += 1
                else:
                    changed += 1
            removed = 0
            if have and max(have) > len(imp.chapters):
                removed = self.db.delete_chapters_from(n.key, len(imp.chapters) + 1)
            if changed:
                # Edited chapters need fresh features; drop their cached rows so analyze() recomputes.
                for ch in imp.chapters:
                    if have.get(ch.index) not in (None, ch.content_hash):
                        self.db.delete_feature(f"{n.key}#{ch.index}", "chapter")
            results.append({"key": n.key, "title": n.title, "chapters": len(imp.chapters), "new": new,
                            "changed": changed, "removed": removed, "labelled": _is_labelled(n),
                            "warnings": imp.warnings})
            log.info("%s: %d chapters (%d new, %d changed, %d removed)", n.title, len(imp.chapters), new, changed, removed)
        return results


def _is_labelled(n: NovelRecord) -> bool:
    return any(v is not None for v in (n.followers, n.views, n.average_views, n.rating, n.favorites, n.user_rating))
