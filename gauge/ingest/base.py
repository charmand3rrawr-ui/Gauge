"""Source interface. Implement these four methods to add a new web novel site."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Protocol

from ..models import ChapterRecord, NovelRecord


@dataclass
class ChapterRef:
    index: int
    source_chapter_id: str
    url: str
    title: str = ""
    published_at: Optional[object] = None  # datetime


class Source(Protocol):
    name: str

    def matches(self, url: str) -> bool: ...

    def discover(self, listing: str, pages: int) -> list[str]:
        """Return novel URLs from a site listing (e.g. 'best-rated')."""
        ...

    def fetch_novel(self, url: str) -> Optional[NovelRecord]: ...

    def list_chapters(self, novel: NovelRecord) -> list[ChapterRef]: ...

    def fetch_chapter(self, novel: NovelRecord, ref: ChapterRef) -> Optional[ChapterRecord]: ...


def chapter_batches(refs: Iterable[ChapterRef], known_ids: set, limit: int) -> list[ChapterRef]:
    """Pick chapters we have not stored yet, preserving order, capped at ``limit``."""
    out = [r for r in refs if r.source_chapter_id not in known_ids]
    return out[:limit]
