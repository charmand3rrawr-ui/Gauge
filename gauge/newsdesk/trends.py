"""Find what the world is reading, in many languages, from sources that permit it.

Two signals, both fetched through Gauge's polite Fetcher (robots.txt honoured, rate-limited):

* **Wikipedia's most-viewed pages per language** — the Wikimedia pageviews API is public and built for
  exactly this. Read across `en`, `es`, `ja`, `de`, ... it is a strong, multilingual "what people are
  looking up today" signal.
* **RSS/Atom feeds you choose** — publishers syndicate headlines, short summaries and links through RSS
  precisely so they can be read by other software. We take the headline, summary and link — never the
  full article — as research and attribution.

Nothing here scrapes article bodies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional
from xml.etree import ElementTree as ET

# Pages that are always "popular" and are not topics.
_WIKI_SKIP = re.compile(r"^(Main_Page|Special:|Wikipedia:|Portal:|File:|Help:|Talk:|Template:|Category:|User:|-$|"
                        r"Hauptseite|Wikipedia$|Página_principal|Accueil_principal|メインページ|Заглавная_страница|"
                        r"Pagina_principale|Strona_główna|Spezial:|Especial:|Spécial:|特別:)", re.I)


@dataclass
class Topic:
    title: str
    lang: str
    score: float          # 0..1, normalised within its source
    source: str           # "wikipedia" | "rss"
    url: str = ""
    langs: list[str] = field(default_factory=list)  # every language the topic trended in


@dataclass
class SourceItem:
    title: str
    summary: str
    url: str
    lang: str = ""
    feed: str = ""
    translated: str = ""   # summary translated into the desk's language, for research


def wikimedia_top(fetcher: Any, lang: str, day: Optional[date] = None, limit: int = 25) -> list[Topic]:
    """Top viewed articles on `lang`.wikipedia for a day (default: yesterday, the latest complete day)."""
    day = day or (date.today() - timedelta(days=1))
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia/all-access/"
           f"{day.year:04d}/{day.month:02d}/{day.day:02d}")
    raw = fetcher.get(url)
    if not raw:
        return []
    try:
        arts = json.loads(raw)["items"][0]["articles"]
    except (KeyError, IndexError, ValueError, TypeError):
        return []
    arts = [a for a in arts if not _WIKI_SKIP.match(str(a.get("article", "")))][:limit]
    if not arts:
        return []
    top = max(float(a.get("views", 0)) for a in arts) or 1.0
    return [Topic(title=str(a["article"]).replace("_", " "), lang=lang, score=round(float(a.get("views", 0)) / top, 4),
                  source="wikipedia", url=f"https://{lang}.wikipedia.org/wiki/{a['article']}", langs=[lang]) for a in arts]


def _text(el: Optional[ET.Element]) -> str:
    return re.sub(r"<[^>]+>", "", (el.text or "")).strip() if el is not None else ""


def rss_items(fetcher: Any, feed_url: str, limit: int = 20, lang: str = "") -> list[SourceItem]:
    """Headlines, summaries and links from an RSS 2.0 or Atom feed. Never fetches the linked articles."""
    raw = fetcher.get(feed_url)
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out: list[SourceItem] = []
    atom = "{http://www.w3.org/2005/Atom}"
    for it in root.iter("item"):  # RSS 2.0
        out.append(SourceItem(_text(it.find("title")), _text(it.find("description"))[:600], _text(it.find("link")),
                              lang=lang, feed=feed_url))
    for it in root.iter(f"{atom}entry"):  # Atom
        link = it.find(f"{atom}link")
        href = link.get("href", "") if link is not None else ""
        summ = it.find(f"{atom}summary") if it.find(f"{atom}summary") is not None else it.find(f"{atom}content")
        out.append(SourceItem(_text(it.find(f"{atom}title")), _text(summ)[:600], href, lang=lang, feed=feed_url))
    return [i for i in out if i.title][:limit]


def _norm(title: str) -> str:
    return re.sub(r"[^\w]+", " ", title.lower()).strip()


def discover(fetcher: Any, langs: list[str], feeds: list[str], day: Optional[date] = None,
             per_lang: int = 10) -> list[Topic]:
    """Merge trending topics across languages and feeds. A topic trending in several languages ranks higher."""
    merged: dict[str, Topic] = {}
    for lang in langs:
        for t in wikimedia_top(fetcher, lang, day, per_lang):
            key = _norm(t.title)
            if key in merged:
                merged[key].score = round(merged[key].score + t.score, 4)
                merged[key].langs.append(lang)
            else:
                merged[key] = t
    for feed in feeds:
        items = rss_items(fetcher, feed, per_lang)
        for i, item in enumerate(items):
            key = _norm(item.title)
            score = round(1.0 - i / max(1, len(items)), 4)
            if key not in merged:
                merged[key] = Topic(item.title, item.lang, score, "rss", item.url, [item.lang] if item.lang else [])
    return sorted(merged.values(), key=lambda t: (len(t.langs), t.score), reverse=True)
