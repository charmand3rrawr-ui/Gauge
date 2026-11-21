"""Open sources the Scout can search. No keys, no accounts, polite rate limits, cached.

Every source returns ``Snippet`` objects: a short, attributed piece of text with a licence note,
so the drafter can be told exactly what it is allowed to do with it (facts and texture, never
copying). Parsing is kept in pure functions so it can be tested without a network.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from urllib.parse import quote, urlencode
from xml.etree import ElementTree as ET

from ..ingest.http import Fetcher
from ..text import normalize_whitespace

log = logging.getLogger(__name__)


@dataclass
class Snippet:
    source: str  # wikipedia / wikivoyage / gutenberg / arxiv / climate / searx
    title: str
    url: str
    text: str
    license: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def card(self, max_words: int = 90) -> str:
        words = self.text.split()
        body = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
        return f"[{self.source}: {self.title}] {body} ({self.url})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


# --------------------------------------------------------------------------------------
# Wikipedia / Wikivoyage through the Wikimedia Core REST API (api.wikimedia.org)
# --------------------------------------------------------------------------------------
# The wiki sites' own robots.txt disallow "/w/" and "/api/" for generic clients; the developer
# API host api.wikimedia.org carries no such restriction, so that is what we use.
def parse_core_search(payload: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    for p in payload.get("pages", []):
        title = p.get("title", "")
        if not title:
            continue
        excerpt = normalize_whitespace(re.sub(r"<[^>]+>", "", p.get("excerpt", "") or ""))
        out.append({"title": title, "key": p.get("key", title.replace(" ", "_")), "excerpt": excerpt,
                    "description": (p.get("description") or "").strip()})
    return out


def lead_paragraphs(html: str, n: int = 2, min_words: int = 15) -> str:
    """The first real paragraphs of a wiki page's HTML: the lead, without infobox and hatnotes."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for bad in soup.select("table, .infobox, .hatnote, .mw-empty-elt, sup, style, script, .navbox, .sidebar, figure"):
        bad.decompose()
    paras = []
    for p in soup.find_all("p"):
        t = normalize_whitespace(p.get_text(" ", strip=True))
        if len(t.split()) >= min_words:
            paras.append(t)
        if len(paras) >= n:
            break
    return "\n\n".join(paras)


# Kept for compatibility with older tests / tools that parse Action API payloads.
def parse_mediawiki_search(payload: dict[str, Any]) -> list[str]:
    return [h.get("title", "") for h in payload.get("query", {}).get("search", []) if h.get("title")]


def parse_mediawiki_extracts(payload: dict[str, Any]) -> dict[str, str]:
    pages = payload.get("query", {}).get("pages", {})
    out = {}
    for p in (pages.values() if isinstance(pages, dict) else pages):
        title, extract = p.get("title", ""), p.get("extract", "")
        if title and extract:
            out[title] = normalize_whitespace(re.sub(r"<[^>]+>", " ", extract))
    return out


class MediaWikiSource:
    def __init__(self, fetcher: Fetcher, project: str = "wikipedia", lang: str = "en", name: Optional[str] = None,
                 license_note: str = "CC BY-SA 4.0: use facts and texture, never copy sentences", fetch_lead: bool = True):
        self.fetcher = fetcher
        self.project = project
        self.lang = lang
        self.name = name or project
        self.license_note = license_note
        self.fetch_lead = fetch_lead
        self.base = f"https://api.wikimedia.org/core/v1/{project}/{lang}"
        self.site = f"https://{lang}.{project}.org/wiki/"

    def _json(self, url: str) -> Optional[dict[str, Any]]:
        raw = self.fetcher.get(url)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def search(self, query: str, k: int = 3) -> list[Snippet]:
        found = self._json(f"{self.base}/search/page?{urlencode({'q': query, 'limit': str(k)})}")
        hits = parse_core_search(found or {})[:k]
        out = []
        for h in hits:
            text = ""
            if self.fetch_lead:
                html = self.fetcher.get(f"{self.base}/page/{quote(h['key'])}/html")
                if html:
                    text = lead_paragraphs(html)
            if not text:
                text = " ".join(x for x in (h["description"], h["excerpt"]) if x)
            if text:
                out.append(Snippet(self.name, h["title"], self.site + quote(h["key"]), _words(text, 220), self.license_note,
                                   {"description": h["description"]}))
        return out


# --------------------------------------------------------------------------------------
# Project Gutenberg via Gutendex
# --------------------------------------------------------------------------------------
def parse_gutendex(payload: dict[str, Any]) -> list[dict[str, Any]]:
    books = []
    for b in payload.get("results", []):
        fmts = b.get("formats", {})
        text_url = next((u for k, u in fmts.items() if k.startswith("text/plain")), None)
        books.append({"id": b.get("id"), "title": b.get("title", ""), "authors": [a.get("name", "") for a in b.get("authors", [])],
                      "subjects": b.get("subjects", []), "year": (b.get("authors") or [{}])[0].get("death_year"),
                      "text_url": text_url, "downloads": b.get("download_count", 0)})
    return books


def strip_gutenberg_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.I)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, re.I)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start() - (start.end() if start else 0)] if start else text[: end.start()]
    return normalize_whitespace(text)


def best_passages(text: str, query: str, k: int = 2, window: int = 120) -> list[str]:
    """Paragraph windows with the most query-word hits: period texture on the thing asked about."""
    words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", query)}
    paras = [p for p in text.split("\n\n") if len(p.split()) >= 25]
    scored = []
    for p in paras:
        low = p.lower()
        hits = sum(low.count(w) for w in words)
        if hits:
            scored.append((hits, p))
    scored.sort(key=lambda t: -t[0])
    return [_words(p, window) for _, p in scored[:k]]


class GutenbergSource:
    name = "gutenberg"

    def __init__(self, fetcher: Fetcher, max_bytes: int = 600_000):
        self.fetcher = fetcher
        self.max_bytes = max_bytes

    def find_books(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        raw = self.fetcher.get(f"https://gutendex.com/books?{urlencode({'search': query, 'languages': 'en'})}")
        if not raw:
            return []
        try:
            return parse_gutendex(json.loads(raw))[:k]
        except json.JSONDecodeError:
            return []

    def book_text(self, book: dict[str, Any]) -> str:
        if not book.get("text_url"):
            return ""
        raw = self.fetcher.get(book["text_url"])
        return strip_gutenberg_boilerplate(raw[: self.max_bytes]) if raw else ""

    def search(self, query: str, k: int = 2) -> list[Snippet]:
        out: list[Snippet] = []
        for book in self.find_books(query, k=3):
            text = self.book_text(book)
            if not text:
                continue
            for passage in best_passages(text, query, k=1):
                out.append(Snippet(self.name, book["title"], f"https://www.gutenberg.org/ebooks/{book['id']}", passage,
                                   "public domain (US): period texture; quote sparingly, transform freely",
                                   {"authors": book["authors"], "subjects": book["subjects"][:5]}))
            if len(out) >= k:
                break
        return out[:k]


# --------------------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------------------
def parse_arxiv(xml_text: str) -> list[dict[str, str]]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for e in root.findall("a:entry", ns):
        title = normalize_whitespace(e.findtext("a:title", default="", namespaces=ns) or "")
        summary = normalize_whitespace(e.findtext("a:summary", default="", namespaces=ns) or "")
        link = e.findtext("a:id", default="", namespaces=ns) or ""
        if title and summary:
            out.append({"title": title, "summary": summary, "url": link})
    return out


class ArxivSource:
    name = "arxiv"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def search(self, query: str, k: int = 2) -> list[Snippet]:
        raw = self.fetcher.get(f"http://export.arxiv.org/api/query?{urlencode({'search_query': 'all:' + query, 'max_results': str(k)})}")
        if not raw:
            return []
        return [Snippet(self.name, e["title"], e["url"], _words(e["summary"], 160), "abstract: use the science, cite nothing verbatim")
                for e in parse_arxiv(raw)[:k]]


# --------------------------------------------------------------------------------------
# Real climate for real places (Open-Meteo, free, no key)
# --------------------------------------------------------------------------------------
def parse_geocode(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    res = payload.get("results") or []
    if not res:
        return None
    r = res[0]
    return {"name": r.get("name"), "country": r.get("country"), "lat": r.get("latitude"), "lon": r.get("longitude"),
            "elevation": r.get("elevation"), "timezone": r.get("timezone")}


def summarize_climate(payload: dict[str, Any], place: str) -> str:
    d = payload.get("daily", {})
    tmax, tmin, rain = d.get("temperature_2m_max") or [], d.get("temperature_2m_min") or [], d.get("precipitation_sum") or []
    if not tmax:
        return ""
    avg = lambda xs: sum(x for x in xs if x is not None) / max(1, len([x for x in xs if x is not None]))  # noqa: E731
    wet_days = sum(1 for r in rain if r and r > 1.0)
    return (f"{place}: typical highs {avg(tmax):.0f} C, lows {avg(tmin):.0f} C, {wet_days} of {len(rain)} days with rain, "
            f"{sum(r for r in rain if r):.0f} mm total in the period")


class ClimateSource:
    name = "climate"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def search(self, query: str, k: int = 1) -> list[Snippet]:
        """query: 'Lisbon in November' / 'weather Lisbon' -> climate for that place and month (last year)."""
        m = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", query, re.I)
        place = re.sub(r"\b(weather|climate|in|during|typical|what|is|the|like)\b", " ", re.sub(m.group(0), " ", query, flags=re.I) if m else query, flags=re.I)
        place = re.sub(r"\s+", " ", place).strip(" ?.,")
        if not place:
            return []
        raw = self.fetcher.get(f"https://geocoding-api.open-meteo.com/v1/search?{urlencode({'name': place, 'count': '1'})}")
        if not raw:
            return []
        try:
            geo = parse_geocode(json.loads(raw))
        except json.JSONDecodeError:
            geo = None
        if not geo:
            return []
        month = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"].index(m.group(0).lower()) + 1 if m else 6
        year = 2024
        start, end = f"{year}-{month:02d}-01", f"{year}-{month:02d}-28"
        raw2 = self.fetcher.get("https://archive-api.open-meteo.com/v1/archive?" + urlencode({
            "latitude": geo["lat"], "longitude": geo["lon"], "start_date": start, "end_date": end,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", "timezone": "auto"}))
        if not raw2:
            return []
        try:
            summary = summarize_climate(json.loads(raw2), f"{geo['name']}, {geo['country']} in {start[:7]}")
        except json.JSONDecodeError:
            return []
        if not summary:
            return []
        return [Snippet(self.name, f"Climate: {geo['name']}", "https://open-meteo.com/", summary + f". Elevation {geo.get('elevation')} m, timezone {geo.get('timezone')}.",
                        "open data (CC BY 4.0)")]


# --------------------------------------------------------------------------------------
# Optional: a self-hosted SearXNG metasearch instance
# --------------------------------------------------------------------------------------
def parse_searx(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in payload.get("results", []) if r.get("url")]


class SearxSource:
    name = "searx"

    def __init__(self, fetcher: Fetcher, base_url: str):
        self.fetcher = fetcher
        self.base = base_url.rstrip("/")

    def search(self, query: str, k: int = 3) -> list[Snippet]:
        raw = self.fetcher.get(f"{self.base}/search?{urlencode({'q': query, 'format': 'json'})}")
        if not raw:
            return []
        try:
            hits = parse_searx(json.loads(raw))
        except json.JSONDecodeError:
            return []
        return [Snippet(self.name, h["title"], h["url"], _words(h["content"], 120), "web result: verify before relying on it") for h in hits[:k]]


class FakeSource:
    """Deterministic source for tests and offline runs."""

    def __init__(self, name: str = "wikipedia", canned: Optional[dict[str, list[Snippet]]] = None):
        self.name = name
        self.canned = canned or {}
        self.queries: list[str] = []

    def search(self, query: str, k: int = 3) -> list[Snippet]:
        self.queries.append(query)
        for key, snips in self.canned.items():
            if key.lower() in query.lower():
                return snips[:k]
        return [Snippet(self.name, f"About {query}", f"https://example.org/{quote(query)}",
                        f"Facts about {query}: couriers were paid per delivery and carried sealed bags; the guild fined late runners.",
                        "test")][:k]
