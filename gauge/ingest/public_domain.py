"""Fetch public-domain and openly-licensed books into the training library.

Gutenberg (full text) reuses the existing GutenbergSource; Wikisource is a best-effort page fetch via the
Wikimedia API. Everything fetched here is tagged with a source whose licence the corpus compiler treats
as safe to train on. Standard Ebooks and Creative-Commons works are imported through `gauge import
--source-tag` (they ship as EPUBs/files), and the registry below marks those tags permitted too. Live
fetching needs the network, so it runs on the author's machine or the daemon, not in a sandbox.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..models import ChapterRecord, NovelRecord


@dataclass
class FetchedBook:
    source: str
    source_id: str
    title: str
    author: str
    text: str
    extra: dict = field(default_factory=dict)


def fetch_gutenberg(fetcher: Any, query: str, limit: int = 5) -> list[FetchedBook]:
    """Full public-domain book texts from Project Gutenberg (boilerplate stripped)."""
    from ..research.sources import GutenbergSource
    gs = GutenbergSource(fetcher)
    out: list[FetchedBook] = []
    for book in gs.find_books(query, k=limit):
        text = gs.book_text(book)
        if text and len(text.split()) > 200:
            out.append(FetchedBook("gutenberg", str(book.get("id", "")), book.get("title", "Untitled"),
                                   ", ".join(book.get("authors", [])), text))
    return out


_WIKI_MARKUP = [
    (re.compile(r"\{\{[^{}]*\}\}"), ""),           # templates
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S), ""),  # references
    (re.compile(r"<[^>]+>"), ""),                   # html tags
    (re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]"), r"\1"),  # links -> label
    (re.compile(r"'{2,}"), ""),                      # bold/italic
    (re.compile(r"(?m)^=+\s*(.*?)\s*=+\s*$"), r"\1"),  # headings -> plain
]


def _strip_wikitext(text: str) -> str:
    for rx, repl in _WIKI_MARKUP:
        text = rx.sub(repl, text)
    from ..text import normalize_whitespace
    return normalize_whitespace(text)


def fetch_wikisource(fetcher: Any, title: str, lang: str = "en") -> list[FetchedBook]:
    """Best-effort: one Wikisource page's text via the Wikimedia API. Multi-part works need the parts named."""
    import json
    from urllib.parse import quote
    url = f"https://api.wikimedia.org/core/v1/wikisource/{lang}/page/{quote(title)}"
    raw = fetcher.get(url)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    source = data.get("source") or data.get("content") or ""
    text = _strip_wikitext(source)
    if len(text.split()) < 100:
        return []
    return [FetchedBook("wikisource", title, data.get("title", title), "", text)]


def store_books(db: Any, books: list[FetchedBook]) -> int:
    """Upsert fetched books into the library as novels (one chapter each). Returns how many were stored."""
    n = 0
    for b in books:
        novel = NovelRecord(source=b.source, source_id=b.source_id, title=b.title, author=b.author)
        db.upsert_novel(novel)
        db.upsert_chapter(ChapterRecord(novel_key=novel.key, index=0, source_chapter_id="0", title=b.title, text=b.text))
        n += 1
    return n
