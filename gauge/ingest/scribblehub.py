"""Scribble Hub connector (public series pages and chapters)."""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..models import ChapterRecord, NovelRecord
from ..text import normalize_whitespace
from .base import ChapterRef
from .http import Fetcher

log = logging.getLogger(__name__)
BASE = "https://www.scribblehub.com"
_ID = re.compile(r"/series/(\d+)")

LISTINGS = {
    "popular": "/series-ranking/?sort=1&order=1",
    "rating": "/series-ranking/?sort=1&order=4",
    "favorites": "/series-ranking/?sort=1&order=2",
    "trending": "/series-ranking/?sort=5&order=1",
}


def _int(s: str) -> Optional[int]:
    s = (s or "").lower().replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*([km])?", s)
    if not m:
        return None
    v = float(m.group(1))
    v *= {"k": 1e3, "m": 1e6}.get(m.group(2) or "", 1)
    return int(v)


def parse_series_page(html: str, url: str) -> NovelRecord:
    soup = BeautifulSoup(html, "lxml")
    sid = _ID.search(url)
    title_el = soup.select_one(".fic_title")
    author_el = soup.select_one(".auth_name_fic")
    blurb_el = soup.select_one(".wi_fic_desc")
    genres = [a.get_text(strip=True) for a in soup.select(".fic_genre")]
    tags = [a.get_text(strip=True) for a in soup.select(".stag")]
    stats = {}
    for item in soup.select(".fic_stats .st_item"):
        txt = item.get_text(" ", strip=True)
        m = re.match(r"([\d.,kKmM]+)\s*(.*)", txt)
        if m:
            stats[m.group(2).lower()] = m.group(1)
    rating_el = soup.select_one("#ratefic_user span, .rating_total")
    rating = None
    if rating_el:
        m = re.search(r"\d+(?:\.\d+)?", rating_el.get_text())
        rating = float(m.group(0)) if m else None
    status = ""
    st = soup.select_one(".widget_fic_similar li:-soup-contains('Status'), .rnd_stats")
    if st:
        low = st.get_text(" ", strip=True).lower()
        for s in ("ongoing", "completed", "hiatus", "dropped"):
            if s in low:
                status = s
    return NovelRecord(
        source="scribblehub", source_id=sid.group(1) if sid else url, url=url,
        title=title_el.get_text(strip=True) if title_el else "Untitled",
        author=author_el.get_text(strip=True) if author_el else "",
        blurb=normalize_whitespace(blurb_el.get_text("\n")) if blurb_el else "",
        genres=genres, tags=tags, status=status,
        views=_int(stats.get("views", "")),
        favorites=_int(stats.get("favorites", "")),
        followers=_int(stats.get("readers", "")),
        chapter_count=_int(stats.get("chapters", "")),
        rating_count=_int(stats.get("reviews", "")),
        rating=rating,
    )


def parse_toc(html: str) -> list[ChapterRef]:
    soup = BeautifulSoup(html, "lxml")
    out: list[ChapterRef] = []
    for li in soup.select(".toc_ol li, ol.toc_ol li"):
        a = li.select_one("a.toc_a, a[href]")
        if not a:
            continue
        href = urljoin(BASE, a["href"])
        m = re.search(r"/chapter/(\d+)", href)
        published = None
        t = li.select_one(".fic_date_pub")
        if t is not None and t.get("title"):
            for fmt in ("%b %d, %Y %I:%M %p", "%b %d, %Y"):
                try:
                    published = datetime.strptime(t["title"], fmt)
                    break
                except ValueError:
                    continue
        out.append(ChapterRef(index=0, source_chapter_id=m.group(1) if m else href, url=href,
                              title=a.get_text(strip=True), published_at=published))
    # Scribble Hub lists newest first; return oldest first and number them.
    out.reverse()
    for i, r in enumerate(out, start=1):
        r.index = i
    return out


def parse_chapter_page(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("#chp_raw, .chp_raw")
    if body is None:
        return ""
    for bad in body.select("script, style, .wi_authornotes"):
        bad.decompose()
    text = "\n\n".join(p.get_text(" ", strip=True) for p in body.find_all(["p", "h1", "h2", "h3"]))
    if not text.strip():
        text = body.get_text("\n", strip=True)
    return normalize_whitespace(text)


def parse_listing(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = [urljoin(BASE, a["href"]) for a in soup.select(".search_title a[href], .search_main_box a.fp_title[href]")]
    return list(dict.fromkeys(u for u in urls if "/series/" in u))


class ScribbleHub:
    name = "scribblehub"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def matches(self, url: str) -> bool:
        return "scribblehub.com/series/" in url

    def discover(self, listing: str = "popular", pages: int = 1) -> list[str]:
        path = LISTINGS.get(listing, listing)
        out: list[str] = []
        for p in range(1, pages + 1):
            html = self.fetcher.get(f"{BASE}{path}&pg={p}")
            if not html:
                break
            out.extend(parse_listing(html))
        return list(dict.fromkeys(out))

    def fetch_novel(self, url: str) -> Optional[NovelRecord]:
        html = self.fetcher.get(url)
        if not html:
            return None
        self._last_html = html
        return parse_series_page(html, url)

    def list_chapters(self, novel: NovelRecord) -> list[ChapterRef]:
        html = getattr(self, "_last_html", None) or self.fetcher.get(novel.url)
        if not html:
            return []
        refs = parse_toc(html)
        # The series page shows only the first TOC page; fetch the rest via the site's ajax endpoint.
        page = 2
        while True:
            more = self.fetcher.post(f"{BASE}/wp-admin/admin-ajax.php",
                                     {"action": "wi_getreleases_pagination", "pagenum": str(page),
                                      "mypostid": novel.source_id})
            if not more:
                break
            extra = parse_toc(more)
            if not extra:
                break
            known = {r.source_chapter_id for r in refs}
            new = [r for r in extra if r.source_chapter_id not in known]
            if not new:
                break
            refs = new + refs
            page += 1
            if page > 60:
                break
        for i, r in enumerate(refs, start=1):
            r.index = i
        return refs

    def fetch_chapter(self, novel: NovelRecord, ref: ChapterRef) -> Optional[ChapterRecord]:
        html = self.fetcher.get(ref.url)
        if not html:
            return None
        text = parse_chapter_page(html)
        if len(text) < 200:
            return None
        return ChapterRecord(novel_key=novel.key, index=ref.index, source_chapter_id=ref.source_chapter_id,
                             url=ref.url, title=ref.title, text=text, published_at=ref.published_at)
