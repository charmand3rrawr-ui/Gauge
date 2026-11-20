"""Import novels from local files: a folder of chapter files, one big text file, or an EPUB.

Metadata and engagement numbers can be supplied in a ``novel.json`` sidecar::

    {"title": "...", "author": "...", "genres": ["LitRPG"], "tags": ["Progression"],
     "followers": 12000, "rating": 4.6, "views": 2500000, "blurb": "..."}

This is how you feed Gauge your own drafts, or exports from sites without a connector.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..models import ChapterRecord, NovelRecord
from ..text import normalize_whitespace

_CHAPTER_HEAD = re.compile(r"^\s*(?:#+\s*)?(chapter|ch\.?|episode|part)\s+(\d+)[^\n]*$", re.I | re.M)


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]


def split_single_file(text: str) -> list[tuple[str, str]]:
    """Split one document into (title, body) chapters on 'Chapter N' style headings."""
    text = normalize_whitespace(text)
    heads = list(_CHAPTER_HEAD.finditer(text))
    if len(heads) < 2:
        return [("Chapter 1", text)]
    out = []
    for i, m in enumerate(heads):
        start = m.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append((m.group(0).strip().lstrip("# "), text[start:end].strip()))
    return [c for c in out if c[1]]


def _read_epub(path: Path) -> list[tuple[str, str]]:
    try:
        from ebooklib import ITEM_DOCUMENT, epub  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install ebooklib   (or: pip install gauge-novel[epub])") from e
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path))
    chapters = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        title_el = soup.find(["h1", "h2", "h3"])
        title = title_el.get_text(strip=True) if title_el else item.get_name()
        body = "\n\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        if len(body.split()) > 150:
            chapters.append((title, normalize_whitespace(body)))
    return chapters


def import_path(path: Path | str, slug: Optional[str] = None) -> tuple[NovelRecord, list[ChapterRecord]]:
    path = Path(path)
    meta = {}
    chapters: list[tuple[str, str]] = []
    if path.is_dir():
        side = path / "novel.json"
        if side.exists():
            meta = json.loads(side.read_text(encoding="utf-8"))
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in {".txt", ".md"}], key=_natural_key)
        for f in files:
            body = normalize_whitespace(f.read_text(encoding="utf-8", errors="replace"))
            if body:
                chapters.append((f.stem.replace("_", " ").replace("-", " "), body))
        base_name = path.name
    elif path.suffix.lower() == ".epub":
        chapters = _read_epub(path)
        base_name = path.stem
        side = path.with_suffix(".json")
        if side.exists():
            meta = json.loads(side.read_text(encoding="utf-8"))
    else:
        chapters = split_single_file(path.read_text(encoding="utf-8", errors="replace"))
        base_name = path.stem
        side = path.with_suffix(".json")
        if side.exists():
            meta = json.loads(side.read_text(encoding="utf-8"))

    slug = slug or re.sub(r"[^a-z0-9]+", "-", (meta.get("title") or base_name).lower()).strip("-")
    novel = NovelRecord(
        source="local", source_id=slug, url=str(path.resolve()),
        title=meta.get("title", base_name.replace("_", " ")), author=meta.get("author", ""),
        blurb=meta.get("blurb", ""), genres=meta.get("genres", []), tags=meta.get("tags", []),
        status=meta.get("status", ""), followers=meta.get("followers"), favorites=meta.get("favorites"),
        views=meta.get("views"), rating=meta.get("rating"), rating_count=meta.get("rating_count"),
        comments=meta.get("comments"), chapter_count=len(chapters),
    )
    per_chapter = meta.get("chapters", {})  # optional {"1": {"views": 100, "comments": 3}}
    recs = []
    for i, (title, body) in enumerate(chapters, start=1):
        extra = per_chapter.get(str(i), {})
        recs.append(ChapterRecord(novel_key=novel.key, index=i, source_chapter_id=str(i), title=title,
                                  text=body, views=extra.get("views"), comments=extra.get("comments")))
    return novel, recs
