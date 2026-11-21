"""Build a static site the author hosts: a landing page, the serial's chapters, and a labelled blog.

Plain, fast, dependency-free HTML. Every page carries a disclosed byline (the build refuses an
impersonating one), opinion pages wear a visible Opinion banner, and the metadata (OpenGraph title and
description) is drawn from the real content, so a thumbnail represents the piece rather than baiting a
click. The output is a folder of files the author uploads to any host.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from ..util import utcnow
from .integrity import OPINION_DISCLAIMER, check_byline, check_headline


@dataclass
class Article:
    slug: str
    title: str
    body: str
    byline: str
    kind: str = "post"          # post | opinion | chapter | update
    summary: str = ""
    published_at: str = ""
    thumbnail: str = ""         # a path/URL to an image the author provides; never required


_STYLE = """:root{--fg:#1a1a1a;--bg:#fff;--muted:#666;--rule:#e5e5e5;--accent:#8a1b1b}
@media(prefers-color-scheme:dark){:root{--fg:#e8e8e8;--bg:#141414;--muted:#9a9a9a;--rule:#2a2a2a;--accent:#e07a7a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 Georgia,serif}
.wrap{max-width:720px;margin:0 auto;padding:16px}a{color:var(--accent)}header{border-bottom:1px solid var(--rule);margin-bottom:24px}
h1{font-size:2rem;line-height:1.2}.byline{color:var(--muted);font-size:.9rem}.opinion{border:1px solid var(--accent);
border-radius:6px;padding:8px 12px;margin:16px 0;font-size:.9rem;color:var(--accent)}.card{border-bottom:1px solid var(--rule);
padding:16px 0}.card h2{margin:.2rem 0;font-size:1.3rem}.muted{color:var(--muted);font-size:.85rem}footer{color:var(--muted);
font-size:.8rem;border-top:1px solid var(--rule);margin-top:32px;padding-top:12px}"""


def _para(body: str) -> str:
    blocks = re.split(r"\n\s*\n", body.strip())
    return "\n".join(f"<p>{html.escape(b).strip()}</p>" for b in blocks if b.strip())


def _page(title: str, description: str, body_html: str, image: str = "") -> str:
    og = [f'<meta property="og:title" content="{html.escape(title)}">',
          f'<meta property="og:description" content="{html.escape(description[:200])}">']
    if image:
        og.append(f'<meta property="og:image" content="{html.escape(image)}">')
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title><meta name=\"description\" content=\"{html.escape(description[:200])}\">"
            + "".join(og) + f"<style>{_STYLE}</style></head><body><div class=\"wrap\">{body_html}"
            f"<footer>Published by the author. Built with Gauge.</footer></div></body></html>")


def _article_html(a: Article) -> str:
    banner = f'<div class="opinion">{html.escape(OPINION_DISCLAIMER)}</div>' if a.kind == "opinion" else ""
    when = a.published_at or utcnow().date().isoformat()
    head = (f'<header><p class="muted"><a href="../index.html">&larr; home</a></p>'
            f'<h1>{html.escape(a.title)}</h1>'
            f'<p class="byline">By {html.escape(a.byline)} &middot; {html.escape(when)}'
            f'{" &middot; Opinion" if a.kind == "opinion" else ""}</p></header>')
    return _page(a.title, a.summary or a.title, head + banner + _para(a.body), a.thumbnail)


def _index_html(site_title: str, articles: list[Article]) -> str:
    cards = []
    for a in articles:
        tag = " &middot; Opinion" if a.kind == "opinion" else (f" &middot; {a.kind}" if a.kind != "post" else "")
        cards.append(f'<div class="card"><h2><a href="articles/{html.escape(a.slug)}.html">{html.escape(a.title)}</a></h2>'
                     f'<p class="muted">By {html.escape(a.byline)} &middot; {html.escape(a.published_at or "")}{tag}</p>'
                     f'<p>{html.escape(a.summary)}</p></div>')
    body = f'<header><h1>{html.escape(site_title)}</h1></header>' + "".join(cards)
    return _page(site_title, f"{site_title}: fiction and commentary.", body)


def _rss(site_title: str, articles: list[Article], base_url: str) -> str:
    items = []
    for a in articles[:50]:
        link = f"{base_url.rstrip('/')}/articles/{a.slug}.html" if base_url else f"articles/{a.slug}.html"
        items.append(f"<item><title>{html.escape(a.title)}</title><link>{html.escape(link)}</link>"
                     f"<description>{html.escape(a.summary or a.title)}</description>"
                     f"<pubDate>{html.escape(a.published_at or '')}</pubDate></item>")
    return ('<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
            f'<title>{html.escape(site_title)}</title><link>{html.escape(base_url)}</link>'
            f'<description>{html.escape(site_title)}</description>' + "".join(items) + "</channel></rss>")


def build_site(site_title: str, articles: list[Article], out_dir: Path, base_url: str = "") -> list[Path]:
    """Render the static site. Refuses to build if any article has an impersonating or missing byline, or
    an opinion piece carries a clickbait headline. Returns the files written."""
    for a in articles:
        ok, why = check_byline(a.byline)
        if not ok:
            raise PermissionError(f"{a.slug}: {why}")
        if a.kind == "opinion":
            hc = check_headline(a.title)
            if not hc.ok:
                raise PermissionError(f"{a.slug}: clickbait headline refused ({'; '.join(hc.flags)}); make it accurate")
    out = Path(out_dir)
    (out / "articles").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ordered = sorted(articles, key=lambda a: a.published_at or "", reverse=True)
    idx = out / "index.html"
    idx.write_text(_index_html(site_title, ordered), encoding="utf-8")
    written.append(idx)
    for a in ordered:
        p = out / "articles" / f"{a.slug}.html"
        p.write_text(_article_html(a), encoding="utf-8")
        written.append(p)
    feed = out / "feed.xml"
    feed.write_text(_rss(site_title, ordered, base_url), encoding="utf-8")
    written.append(feed)
    return written
