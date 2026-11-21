"""Build a static site the author hosts: a landing page, the serial's chapters, and a labelled blog.

Plain, fast, dependency-free HTML. Every page carries a disclosed byline (the build refuses an
impersonating one), opinion pages wear a visible Opinion banner, and the metadata (OpenGraph title and
description) is drawn from the real content, so a thumbnail represents the piece rather than baiting a
click. The build also emits a sitemap and robots.txt (SEO), per-author pages, a client-side search index
and page, and accessible markup (a skip link, a main landmark, alt text on images). The output is a
folder of files the author uploads to any host.
"""
from __future__ import annotations

import html
import json
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
    thumbnail_alt: str = ""     # description of the image, for screen readers


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "author"


_STYLE = """:root{--fg:#1a1a1a;--bg:#fff;--muted:#555;--rule:#e5e5e5;--accent:#8a1b1b}
@media(prefers-color-scheme:dark){:root{--fg:#e8e8e8;--bg:#141414;--muted:#a6a6a6;--rule:#2a2a2a;--accent:#e07a7a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 Georgia,serif}
.wrap{max-width:720px;margin:0 auto;padding:16px}a{color:var(--accent)}header{border-bottom:1px solid var(--rule);margin-bottom:24px}
h1{font-size:2rem;line-height:1.2}.byline{color:var(--muted);font-size:.9rem}.opinion{border:1px solid var(--accent);
border-radius:6px;padding:8px 12px;margin:16px 0;font-size:.9rem;color:var(--accent)}.card{border-bottom:1px solid var(--rule);
padding:16px 0}.card h2{margin:.2rem 0;font-size:1.3rem}.muted{color:var(--muted);font-size:.85rem}footer{color:var(--muted);
font-size:.8rem;border-top:1px solid var(--rule);margin-top:32px;padding-top:12px}img{max-width:100%;height:auto;border-radius:6px}
.skip{position:absolute;left:-999px}.skip:focus{left:8px;top:8px;background:var(--accent);color:#fff;padding:8px;z-index:9}
input[type=search]{font:inherit;padding:8px;width:100%;border:1px solid var(--rule);border-radius:6px;background:var(--bg);color:var(--fg)}"""


def _para(body: str) -> str:
    blocks = re.split(r"\n\s*\n", body.strip())
    return "\n".join(f"<p>{html.escape(b).strip()}</p>" for b in blocks if b.strip())


def _page(title: str, description: str, body_html: str, image: str = "", canonical: str = "", depth: str = "") -> str:
    og = [f'<meta property="og:title" content="{html.escape(title)}">',
          f'<meta property="og:description" content="{html.escape(description[:200])}">']
    if image:
        og.append(f'<meta property="og:image" content="{html.escape(image)}">')
    can = f'<link rel="canonical" href="{html.escape(canonical)}">' if canonical else ""
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title><meta name=\"description\" content=\"{html.escape(description[:200])}\">"
            + "".join(og) + can + f"<style>{_STYLE}</style></head><body>"
            f"<a class=\"skip\" href=\"#main\">Skip to content</a>"
            f"<main id=\"main\"><div class=\"wrap\">{body_html}"
            f"<footer><a href=\"{depth}search.html\">Search</a> &middot; Published by the author. Built with Gauge."
            f"</footer></div></main></body></html>")


def _byline_link(byline: str, depth: str) -> str:
    return f'<a href="{depth}authors/{_slug(byline)}.html">{html.escape(byline)}</a>'


def _article_html(a: Article, base_url: str = "") -> str:
    banner = f'<div class="opinion">{html.escape(OPINION_DISCLAIMER)}</div>' if a.kind == "opinion" else ""
    when = a.published_at or utcnow().date().isoformat()
    img = (f'<img src="{html.escape(a.thumbnail)}" alt="{html.escape(a.thumbnail_alt or a.title)}">'
           if a.thumbnail else "")
    head = (f'<header><p class="muted"><a href="../index.html">&larr; home</a></p>'
            f'<h1>{html.escape(a.title)}</h1>'
            f'<p class="byline">By {_byline_link(a.byline, "../")} &middot; {html.escape(when)}'
            f'{" &middot; Opinion" if a.kind == "opinion" else ""}</p></header>')
    canonical = f"{base_url.rstrip('/')}/articles/{a.slug}.html" if base_url else ""
    return _page(a.title, a.summary or a.title, head + banner + img + _para(a.body), a.thumbnail, canonical, depth="../")


def _card(a: Article, depth: str) -> str:
    tag = " &middot; Opinion" if a.kind == "opinion" else (f" &middot; {a.kind}" if a.kind != "post" else "")
    return (f'<div class="card"><h2><a href="{depth}articles/{html.escape(a.slug)}.html">{html.escape(a.title)}</a></h2>'
            f'<p class="muted">By {_byline_link(a.byline, depth)} &middot; {html.escape(a.published_at or "")}{tag}</p>'
            f'<p>{html.escape(a.summary)}</p></div>')


def _index_html(site_title: str, articles: list[Article], base_url: str = "") -> str:
    body = f'<header><h1>{html.escape(site_title)}</h1></header>' + "".join(_card(a, "") for a in articles)
    canonical = f"{base_url.rstrip('/')}/index.html" if base_url else ""
    return _page(site_title, f"{site_title}: fiction and commentary.", body, canonical=canonical)


def _author_html(site_title: str, author: str, articles: list[Article]) -> str:
    body = (f'<header><p class="muted"><a href="../index.html">&larr; home</a></p>'
            f'<h1>{html.escape(author)}</h1><p class="muted">Writing on {html.escape(site_title)}.</p></header>'
            + "".join(_card(a, "../") for a in articles))
    return _page(f"{author} — {site_title}", f"Articles by {author}.", body, depth="../")


def _search_html(site_title: str) -> str:
    body = ('<header><p class="muted"><a href="index.html">&larr; home</a></p><h1>Search</h1></header>'
            '<p><input type="search" id="q" placeholder="Search titles and summaries" aria-label="Search"></p>'
            '<div id="results" aria-live="polite"></div>'
            '<script>let I=[];fetch("search-index.json").then(r=>r.json()).then(d=>{I=d;});'
            'document.getElementById("q").addEventListener("input",e=>{const q=e.target.value.toLowerCase().trim();'
            'const out=document.getElementById("results");if(!q){out.innerHTML="";return;}'
            'const hits=I.filter(a=>(a.title+" "+a.summary+" "+a.body).toLowerCase().includes(q)).slice(0,20);'
            'out.innerHTML=hits.map(a=>`<div class=card><h2><a href="articles/${a.slug}.html">${a.title}</a></h2>`+'
            '`<p class=muted>${a.summary||""}</p></div>`).join("")||"<p class=muted>No matches.</p>";});</script>')
    return _page(f"Search — {site_title}", f"Search {site_title}.", body)


def _search_index(articles: list[Article]) -> str:
    return json.dumps([{"slug": a.slug, "title": a.title, "summary": a.summary,
                        "body": re.sub(r"\s+", " ", a.body)[:600]} for a in articles], ensure_ascii=False)


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


def _sitemap(base_url: str, paths: list[str]) -> str:
    base = base_url.rstrip("/") if base_url else ""
    urls = "".join(f"<url><loc>{html.escape(base + '/' + p if base else p)}</loc></url>" for p in paths)
    return f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'


def accessibility_report(html_text: str) -> list[str]:
    """Flag missing accessibility basics in a rendered page. Empty list = the basics are present."""
    issues = []
    if 'lang="' not in html_text:
        issues.append("missing lang attribute on <html>")
    if 'id="main"' not in html_text and "<main" not in html_text:
        issues.append("no main landmark")
    if "skip" not in html_text.lower():
        issues.append("no skip-to-content link")
    for m in re.finditer(r"<img\b[^>]*>", html_text):
        if "alt=" not in m.group(0):
            issues.append("image without alt text")
            break
    return issues


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
    (out / "authors").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ordered = sorted(articles, key=lambda a: a.published_at or "", reverse=True)
    paths = ["index.html", "search.html"]

    def _w(rel: str, text: str) -> None:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        written.append(p)

    _w("index.html", _index_html(site_title, ordered, base_url))
    for a in ordered:
        _w(f"articles/{a.slug}.html", _article_html(a, base_url))
        paths.append(f"articles/{a.slug}.html")
    # Per-author pages.
    by_author: dict[str, list[Article]] = {}
    for a in ordered:
        by_author.setdefault(a.byline, []).append(a)
    for author, arts in by_author.items():
        _w(f"authors/{_slug(author)}.html", _author_html(site_title, author, arts))
        paths.append(f"authors/{_slug(author)}.html")
    _w("search.html", _search_html(site_title))
    _w("search-index.json", _search_index(ordered))
    _w("feed.xml", _rss(site_title, ordered, base_url))
    _w("sitemap.xml", _sitemap(base_url, paths))
    _w("robots.txt", (f"User-agent: *\nAllow: /\nSitemap: {base_url.rstrip('/')}/sitemap.xml\n" if base_url
                      else "User-agent: *\nAllow: /\n"))
    return written
