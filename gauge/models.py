"""Core data records shared across ingestion, analysis and writing."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .util import utcnow


class NovelRecord(BaseModel):
    source: str  # "royalroad", "scribblehub", "local", ...
    source_id: str  # id on the source site, or a slug for local imports
    url: str = ""
    title: str
    author: str = ""
    blurb: str = ""
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    status: str = ""  # ongoing / completed / hiatus
    # Engagement metrics. Any may be None when a source does not expose it.
    followers: Optional[int] = None
    favorites: Optional[int] = None
    views: Optional[int] = None
    average_views: Optional[int] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    comments: Optional[int] = None
    chapter_count: Optional[int] = None
    pages: Optional[int] = None
    # Your own 1-5 verdict for local novels; lets Gauge learn *your* taste (target 'user_rating').
    user_rating: Optional[float] = None
    first_published: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=utcnow)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.source_id}"


class ChapterRecord(BaseModel):
    novel_key: str
    index: int  # 1-based position in the novel
    source_chapter_id: str = ""
    url: str = ""
    title: str = ""
    text: str
    published_at: Optional[datetime] = None
    words: int = 0
    # Chapter-level engagement where available (Royal Road exposes neither publicly; local
    # imports and some sources do). Kept optional so the schema is uniform.
    views: Optional[int] = None
    comments: Optional[int] = None
    content_hash: str = ""
    comment_texts: list[dict] = Field(default_factory=list)  # reader comments scraped with the chapter (not stored in chapters table)
    fetched_at: datetime = Field(default_factory=utcnow)

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        if not self.words:
            self.words = len(self.text.split())
        if not self.content_hash:
            self.content_hash = hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:16]


class FeatureVector(BaseModel):
    """A named, flat dict of numeric features for one chapter or one novel."""

    owner_key: str
    level: str  # "chapter" | "novel"
    values: dict[str, float]
