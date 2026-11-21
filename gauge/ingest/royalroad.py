"""Royal Road connector.

Royal Road exposes rich public engagement metrics on each fiction page (followers, favorites,
views, ratings, pages) which makes it the best single source for learning what readers
actually keep reading. Only public pages are read; robots.txt and a request delay are honoured.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..models import ChapterRecord, NovelRecord
from ..text import normalize_whitespace
from .base import ChapterRef
from .http import Fetcher

log = logging.getLogger(__name__)
BASE = "https://www.royalroad.com"
_ID = re.compile(r"/fiction/(\d+)")
_NUM = re.compile(r"[\d,]+")

LISTINGS = {
    "best-rated": "/fictions/best-rated",
    "trending": "/fictions/trending",
    "rising-stars": "/fictions/rising-stars",
    "popular": "/fictions/active-popular",
    "new": "/fictions/new-releases",
    "complete": "/fictions/complete",
}


def _int(s: str) -> Optional[int]:
    m = _NUM.search(s or "")
    return int(m.group(0).replace(",", "")) if m else None


def _float(s: str) -> Optional[float]:
    m = re.search(r"\d+(?:\.\d+)?", s or "")
    return float(m.group(0)) if m else None


def parse_fiction_page(html: str, url: str) -> NovelRecord:
    soup = BeautifulSoup(html, "lxml")
    sid = _ID.search(url)
    source_id = sid.group(1) if sid else url
    title_el = soup.select_one("h1") or soup.select_one(".fic-title h1")
    title = title_el.get_text(strip=True) if title_el else "Untitled"
    author_el = soup.select_one(".fic-title h4 a") or soup.select_one("h4 span a")
    author = author_el.get_text(strip=True) if author_el else ""
    blurb_el = soup.select_one(".description")
    blurb = normalize_whitespace(blurb_el.get_text("\n")) if blurb_el else ""
    tags = [a.get_text(strip=True) for a in soup.select(".tags a, a.fiction-tag")]
    status = ""
    for lab in soup.select(".fiction-info .label, .label-default"):
        t = lab.get_text(strip=True).lower()
        if t in {"ongoing", "completed", "hiatus", "stub", "dropped"}:
            status = t
            break

    stats = {}
    for ul in soup.select(".fiction-stats ul.list-unstyled, .fiction-stats"):
        items = [li.get_text(" ", strip=True) for li in ul.select("li")]
        for k, v in zip(items, items[1:]):
            stats[k.lower().rstrip(" :")] = v
    rating = None
    score_el = soup.select_one(".fiction-stats [data-content*='/ 5'], .fiction-stats [aria-label*='stars']")
    if score_el:
        rating = _float(score_el.get("data-content") or score_el.get("aria-label") or "")
    if rating is None:
        for k, v in stats.items():
            if k.startswith("overall score"):
                rating = _float(v)

    n = NovelRecord(
        source="royalroad", source_id=source_id, url=url, title=title, author=author, blurb=blurb,
        tags=tags, status=status,
        followers=_int(stats.get("followers", "")),
        favorites=_int(stats.get("favorites", "")),
        views=_int(stats.get("total views", "")),
        average_views=_int(stats.get("average views", "")),
        rating=rating,
        rating_count=_int(stats.get("ratings", "")),
        pages=_int(stats.get("pages", "")),
    )
    refs = parse_chapter_list(html, url)
    n.chapter_count = len(refs) or None
    dates = [r.published_at for r in refs if r.published_at]
    if dates:
        n.first_published = min(dates)  # type: ignore[type-var]
        n.last_updated = max(dates)  # type: ignore[type-var]
    return n


def parse_chapter_list(html: str, novel_url: str) -> list[ChapterRef]:
    soup = BeautifulSoup(html, "lxml")
    refs: list[ChapterRef] = []
    rows = soup.select("table#chapters tbody tr")
    for i, tr in enumerate(rows, start=1):
        a = tr.select_one("td a[href]")
        if not a:
            continue
        href = urljoin(BASE, a["href"])
        m = re.search(r"/chapter/(\d+)", href)
        cid = m.group(1) if m else href
        t = tr.select_one("time")
        published = None
        if t is not None:
            unix = t.get("unixtime")
            if unix and unix.isdigit():
                published = datetime.fromtimestamp(int(unix), tz=timezone.utc).replace(tzinfo=None)
            elif t.get("datetime"):
                try:
                    published = datetime.fromisoformat(t["datetime"].replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    published = None
        refs.append(ChapterRef(index=i, source_chapter_id=cid, url=href,
                               title=a.get_text(strip=True), published_at=published))
    return refs


def parse_chapter_page(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one(".chapter-inner.chapter-content, .chapter-content")
    if body is None:
        return ""
    for bad in body.select("script, style, .ad, [class*='ad-']"):
        bad.decompose()
    text = "\n\n".join(p.get_text(" ", strip=True) for p in body.find_all(["p", "h1", "h2", "h3", "li"]))
    if not text.strip():
        text = body.get_text("\n", strip=True)
    return normalize_whitespace(text)


def parse_chapter_comments(html: str, limit: int = 60) -> list[dict[str, str]]:
    """Reader comments rendered on a chapter page (author, text, time). Best effort: layouts change."""
    soup = BeautifulSoup(html, "lxml")
    out: list[dict[str, str]] = []
    for node in soup.select(".comment, .comment-item, [class*='comment-']"):
        body = node.select_one(".comment-body, .comment-content, .message, .text, p")
        if body is None:
            continue
        text = normalize_whitespace(body.get_text(" ", strip=True))
        if len(text) < 2:
            continue
        author_el = node.select_one(".comment-author, .username, .name, a[href*='/profile/']")
        time_el = node.select_one("time")
        out.append({"author": author_el.get_text(strip=True) if author_el else "", "text": text[:2000],
                    "at": (time_el.get("datetime") or time_el.get("title") or "") if time_el else ""})
        if len(out) >= limit:
            break
    # Deduplicate nested matches (a wrapper and its inner node can both match).
    seen, uniq = set(), []
    for c in out:
        key = c["text"][:120]
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def parse_listing(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for a in soup.select(".fiction-list-item h2 a[href], .fiction-list-item .fiction-title a[href]"):
        urls.append(urljoin(BASE, a["href"]))
    return list(dict.fromkeys(urls))


class RoyalRoad:
    name = "royalroad"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def matches(self, url: str) -> bool:
        return "royalroad.com/fiction/" in url

    def discover(self, listing: str = "best-rated", pages: int = 1) -> list[str]:
        path = LISTINGS.get(listing, listing)
        out: list[str] = []
        for p in range(1, pages + 1):
            html = self.fetcher.get(f"{BASE}{path}?page={p}")
            if not html:
                break
            out.extend(parse_listing(html))
        return list(dict.fromkeys(out))

    def fetch_novel(self, url: str) -> Optional[NovelRecord]:
        html = self.fetcher.get(url)
        if not html:
            return None
        self._last_html = html
        return parse_fiction_page(html, url)

    def list_chapters(self, novel: NovelRecord) -> list[ChapterRef]:
        html = getattr(self, "_last_html", None) or self.fetcher.get(novel.url)
        return parse_chapter_list(html, novel.url) if html else []

    def fetch_chapter(self, novel: NovelRecord, ref: ChapterRef) -> Optional[ChapterRecord]:
        html = self.fetcher.get(ref.url)
        if not html:
            return None
        text = parse_chapter_page(html)
        if len(text) < 200:
            return None
        comments = parse_chapter_comments(html)
        return ChapterRecord(novel_key=novel.key, index=ref.index, source_chapter_id=ref.source_chapter_id,
                             url=ref.url, title=ref.title, text=text, published_at=ref.published_at,
                             comments=len(comments) or None, comment_texts=comments)
