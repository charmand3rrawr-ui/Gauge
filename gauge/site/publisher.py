"""Build a static site the author hosts: a landing page, the serial's chapters, and a labelled blog.

Plain, fast, dependency-free HTML. Every page carries a disclosed byline (the build refuses an
impersonating one), opinion pages wear a visible Opinion banner, and the metadata (OpenGraph title and
description) is drawn from the real content, so a thumbnail represents the piece rather than baiting a
click. The build also emits a sitemap and robots.txt (SEO), per-author pages, a client-side search index
and page, and accessible markup (a skip link, a main landmark, alt text on images).

Themes: a handful of named palettes, each with a light and a dark variant that meet WCAG AA contrast. The
page follows the reader's system preference and offers a light/dark toggle that remembers the choice.

Feeds: RSS 2.0 (``feed.xml``) and Atom 1.0 (``atom.xml``) for everything, plus per-section RSS feeds for
the serial (``feeds/serial.xml``) and the opinion blog (``feeds/opinion.xml``). Opinion is labelled as
opinion in the feed too, and every item carries its byline. The output is a folder of files the author
uploads to any host.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------- themes

_SERIF = "Georgia,'Times New Roman',serif"
# Each theme: a font stack and light/dark tokens. Text, muted text and accent all clear WCAG AA (4.5:1)
# against the background in both modes; tests/test_site.py checks this for every theme.
THEMES: dict[str, dict] = {
    "classic": {"font": _SERIF,
                "light": {"fg": "#1a1a1a", "bg": "#ffffff", "muted": "#555555", "rule": "#e5e5e5", "accent": "#8a1b1b"},
                "dark": {"fg": "#e8e8e8", "bg": "#141414", "muted": "#a6a6a6", "rule": "#2a2a2a", "accent": "#e07a7a"}},
    "paper": {"font": "'Iowan Old Style','Palatino Linotype',Palatino," + _SERIF,
              "light": {"fg": "#2b2622", "bg": "#f7f1e3", "muted": "#5e554b", "rule": "#e2d8c3", "accent": "#8f3417"},
              "dark": {"fg": "#e9e0d2", "bg": "#1e1a16", "muted": "#b3a797", "rule": "#352e26", "accent": "#e59468"}},
    "slate": {"font": "system-ui,-apple-system,'Segoe UI',Roboto,sans-serif",
              "light": {"fg": "#1c2430", "bg": "#f6f8fa", "muted": "#51606f", "rule": "#dde3ea", "accent": "#1b56ad"},
              "dark": {"fg": "#dfe6ee", "bg": "#12161c", "muted": "#9aa8b8", "rule": "#262e39", "accent": "#84b4ff"}},
    "noir": {"font": _SERIF,
             "light": {"fg": "#0d0d0d", "bg": "#fafafa", "muted": "#4a4a4a", "rule": "#d9d9d9", "accent": "#a8001c"},
             "dark": {"fg": "#f2f2f2", "bg": "#0b0b0b", "muted": "#b0b0b0", "rule": "#2b2b2b", "accent": "#ff6b7d"}},
}
DEFAULT_THEME = "classic"

_BASE_CSS = """*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 var(--font);overflow-wrap:break-word}
.wrap{max-width:720px;margin:0 auto;padding:16px}a{color:var(--accent)}header{border-bottom:1px solid var(--rule);margin-bottom:24px}
h1{font-size:2rem;line-height:1.2;text-wrap:balance}.byline{color:var(--muted);font-size:.9rem}.opinion{border:1px solid var(--accent);
border-radius:6px;padding:8px 12px;margin:16px 0;font-size:.9rem;color:var(--accent)}.card{border-bottom:1px solid var(--rule);
padding:16px 0}.card h2{margin:.2rem 0;font-size:1.3rem}.muted{color:var(--muted);font-size:.85rem}footer{color:var(--muted);
font-size:.8rem;border-top:1px solid var(--rule);margin-top:32px;padding-top:12px}img{max-width:100%;height:auto;border-radius:6px}
.skip{position:absolute;left:-999px}.skip:focus{left:8px;top:8px;background:var(--accent);color:var(--bg);padding:8px;z-index:9}
.bar{display:flex;justify-content:flex-end;min-height:30px}.toggle{font:inherit;font-size:.8rem;background:none;color:var(--muted);
border:1px solid var(--rule);border-radius:6px;padding:4px 10px;cursor:pointer}.toggle:hover{color:var(--fg);border-color:var(--muted)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input[type=search]{font:inherit;padding:8px;width:100%;border:1px solid var(--rule);border-radius:6px;background:var(--bg);color:var(--fg)}"""

# Applied before first paint so a remembered choice never flashes the other theme.
_THEME_BOOT_JS = ('(function(){try{var t=localStorage.getItem("gauge-theme");'
                  'if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t);}catch(e){}})();')
# The toggle is hidden until this runs, so without JavaScript there is no dead button (the system
# preference still applies). The label names the action; aria-pressed reports the state.
_THEME_TOGGLE_JS = ('(function(){var b=document.getElementById("theme-toggle");if(!b)return;var r=document.documentElement;'
                    'function dark(){var t=r.getAttribute("data-theme");if(t)return t==="dark";'
                    'return !!(window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches);}'
                    'function paint(){var d=dark();b.setAttribute("aria-pressed",d?"true":"false");'
                    'b.textContent=d?"Light mode":"Dark mode";}'
                    'b.addEventListener("click",function(){var n=dark()?"light":"dark";r.setAttribute("data-theme",n);'
                    'try{localStorage.setItem("gauge-theme",n);}catch(e){}paint();});b.hidden=false;paint();})();')


def _tokens(t: dict[str, str]) -> str:
    return "".join(f"--{k}:{v};" for k, v in t.items())


def theme_css(theme: str = DEFAULT_THEME) -> str:
    """The stylesheet for a named theme: light tokens by default, dark tokens when the system asks for dark
    (unless the reader chose light) or when the reader chose dark with the toggle."""
    t = THEMES.get(theme) or THEMES[DEFAULT_THEME]
    light, dark = _tokens(t["light"]), _tokens(t["dark"])
    return (f":root{{--font:{t['font']};{light}color-scheme:light}}"
            f"@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{{dark}color-scheme:dark}}}}"
            f":root[data-theme=dark]{{{dark}color-scheme:dark}}\n" + _BASE_CSS)


def _hex_luminance(h: str) -> float:
    h = h.lstrip("#")

    def ch(i: int) -> float:
        c = int(h[i:i + 2], 16) / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(0) + 0.7152 * ch(2) + 0.0722 * ch(4)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two #rrggbb colours."""
    la, lb = _hex_luminance(a), _hex_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def theme_contrast_issues(theme: str, minimum: float = 4.5) -> list[str]:
    """Token pairs in a theme that fall below a contrast minimum. Empty = the theme is readable."""
    t = THEMES[theme]
    out = []
    for mode in ("light", "dark"):
        for key in ("fg", "muted", "accent"):
            r = contrast_ratio(t[mode][key], t[mode]["bg"])
            if r < minimum:
                out.append(f"{theme}/{mode}: {key} on bg is {r:.2f}:1 (needs {minimum}:1)")
    return out


# ---------------------------------------------------------------------------- pages

def _para(body: str) -> str:
    blocks = re.split(r"\n\s*\n", body.strip())
    return "\n".join(f"<p>{html.escape(b).strip()}</p>" for b in blocks if b.strip())


def _page(title: str, description: str, body_html: str, image: str = "", canonical: str = "", depth: str = "",
          theme: str = DEFAULT_THEME, site_title: str = "") -> str:
    og = [f'<meta property="og:title" content="{html.escape(title)}">',
          f'<meta property="og:description" content="{html.escape(description[:200])}">']
    if image:
        og.append(f'<meta property="og:image" content="{html.escape(image)}">')
    can = f'<link rel="canonical" href="{html.escape(canonical)}">' if canonical else ""
    feed_title = html.escape(site_title or title)
    feeds = (f'<link rel="alternate" type="application/rss+xml" title="{feed_title} (RSS)" href="{depth}feed.xml">'
             f'<link rel="alternate" type="application/atom+xml" title="{feed_title} (Atom)" href="{depth}atom.xml">')
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title><meta name=\"description\" content=\"{html.escape(description[:200])}\">"
            + "".join(og) + can + feeds + f"<script>{_THEME_BOOT_JS}</script><style>{theme_css(theme)}</style></head><body>"
            f"<a class=\"skip\" href=\"#main\">Skip to content</a>"
            f"<main id=\"main\"><div class=\"wrap\">"
            f"<div class=\"bar\"><button id=\"theme-toggle\" class=\"toggle\" type=\"button\" aria-pressed=\"false\" hidden>Dark mode</button></div>"
            f"{body_html}"
            f"<footer><a href=\"{depth}search.html\">Search</a> &middot; <a href=\"{depth}feed.xml\">Feed</a> &middot; "
            f"Published by the author. Built with Gauge.</footer></div></main>"
            f"<script>{_THEME_TOGGLE_JS}</script></body></html>")


def _byline_link(byline: str, depth: str) -> str:
    return f'<a href="{depth}authors/{_slug(byline)}.html">{html.escape(byline)}</a>'


def _article_html(a: Article, base_url: str = "", theme: str = DEFAULT_THEME, site_title: str = "") -> str:
    banner = f'<div class="opinion">{html.escape(OPINION_DISCLAIMER)}</div>' if a.kind == "opinion" else ""
    when = a.published_at or utcnow().date().isoformat()
    img = (f'<img src="{html.escape(a.thumbnail)}" alt="{html.escape(a.thumbnail_alt or a.title)}">'
           if a.thumbnail else "")
    head = (f'<header><p class="muted"><a href="../index.html">&larr; home</a></p>'
            f'<h1>{html.escape(a.title)}</h1>'
            f'<p class="byline">By {_byline_link(a.byline, "../")} &middot; {html.escape(when)}'
            f'{" &middot; Opinion" if a.kind == "opinion" else ""}</p></header>')
    canonical = f"{base_url.rstrip('/')}/articles/{a.slug}.html" if base_url else ""
    return _page(a.title, a.summary or a.title, head + banner + img + _para(a.body), a.thumbnail, canonical, depth="../",
                 theme=theme, site_title=site_title)


def _card(a: Article, depth: str) -> str:
    tag = " &middot; Opinion" if a.kind == "opinion" else (f" &middot; {a.kind}" if a.kind != "post" else "")
    return (f'<div class="card"><h2><a href="{depth}articles/{html.escape(a.slug)}.html">{html.escape(a.title)}</a></h2>'
            f'<p class="muted">By {_byline_link(a.byline, depth)} &middot; {html.escape(a.published_at or "")}{tag}</p>'
            f'<p>{html.escape(a.summary)}</p></div>')


def _index_html(site_title: str, articles: list[Article], base_url: str = "", theme: str = DEFAULT_THEME) -> str:
    body = f'<header><h1>{html.escape(site_title)}</h1></header>' + "".join(_card(a, "") for a in articles)
    canonical = f"{base_url.rstrip('/')}/index.html" if base_url else ""
    return _page(site_title, f"{site_title}: fiction and commentary.", body, canonical=canonical, theme=theme,
                 site_title=site_title)


def _author_html(site_title: str, author: str, articles: list[Article], theme: str = DEFAULT_THEME) -> str:
    body = (f'<header><p class="muted"><a href="../index.html">&larr; home</a></p>'
            f'<h1>{html.escape(author)}</h1><p class="muted">Writing on {html.escape(site_title)}.</p></header>'
            + "".join(_card(a, "../") for a in articles))
    return _page(f"{author} — {site_title}", f"Articles by {author}.", body, depth="../", theme=theme, site_title=site_title)


def _search_html(site_title: str, theme: str = DEFAULT_THEME) -> str:
    body = ('<header><p class="muted"><a href="index.html">&larr; home</a></p><h1>Search</h1></header>'
            '<p><input type="search" id="q" placeholder="Search titles and summaries" aria-label="Search"></p>'
            '<div id="results" aria-live="polite"></div>'
            '<script>let I=[];fetch("search-index.json").then(r=>r.json()).then(d=>{I=d;});'
            'document.getElementById("q").addEventListener("input",e=>{const q=e.target.value.toLowerCase().trim();'
            'const out=document.getElementById("results");if(!q){out.innerHTML="";return;}'
            'const hits=I.filter(a=>(a.title+" "+a.summary+" "+a.body).toLowerCase().includes(q)).slice(0,20);'
            'out.innerHTML=hits.map(a=>`<div class=card><h2><a href="articles/${a.slug}.html">${a.title}</a></h2>`+'
            '`<p class=muted>${a.summary||""}</p></div>`).join("")||"<p class=muted>No matches.</p>";});</script>')
    return _page(f"Search — {site_title}", f"Search {site_title}.", body, theme=theme, site_title=site_title)


def _search_index(articles: list[Article]) -> str:
    return json.dumps([{"slug": a.slug, "title": a.title, "summary": a.summary,
                        "body": re.sub(r"\s+", " ", a.body)[:600]} for a in articles], ensure_ascii=False)


# ---------------------------------------------------------------------------- feeds

_XML_BAD = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_KIND_LABEL = {"opinion": "Opinion", "chapter": "Chapter", "update": "Update", "post": "Post"}


def _x(text: str) -> str:
    """Escape text for XML and drop control characters XML 1.0 forbids."""
    return html.escape(_XML_BAD.sub("", text or ""), quote=True)


def _parse_when(s: str) -> Optional[datetime]:
    """An article's published_at as an aware UTC datetime, or None when it is empty or unparseable."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _now() -> datetime:
    """Build time as an aware UTC datetime, to the second (feeds need an explicit offset)."""
    return utcnow().replace(tzinfo=timezone.utc, microsecond=0)


def _link(a: Article, base_url: str) -> str:
    return f"{base_url.rstrip('/')}/articles/{a.slug}.html" if base_url else f"articles/{a.slug}.html"


def _feed_summary(a: Article) -> str:
    # Opinion is labelled as opinion wherever it travels, including a feed reader.
    text = a.summary or a.title
    return f"[Opinion] {text}" if a.kind == "opinion" else text


def _rss(site_title: str, articles: list[Article], base_url: str, self_path: str = "feed.xml",
         description: str = "") -> str:
    """RSS 2.0. Dates are RFC 822; each item has a guid, a category for its kind, and its byline."""
    base = base_url.rstrip("/")
    items = []
    for a in articles[:50]:
        link = _link(a, base_url)
        when = _parse_when(a.published_at)
        items.append(
            f"<item><title>{_x(a.title)}</title><link>{_x(link)}</link>"
            f'<guid isPermaLink="{"true" if base_url else "false"}">{_x(link)}</guid>'
            f"<description>{_x(_feed_summary(a))}</description>"
            f"<category>{_x(_KIND_LABEL.get(a.kind, a.kind.title()))}</category>"
            f"<dc:creator>{_x(a.byline)}</dc:creator>"
            + (f"<pubDate>{format_datetime(when)}</pubDate>" if when else "") + "</item>")
    self_link = (f'<atom:link href="{_x(base + "/" + self_path)}" rel="self" type="application/rss+xml"/>' if base else "")
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>'
            f"<title>{_x(site_title)}</title><link>{_x(base_url or 'index.html')}</link>"
            f"<description>{_x(description or site_title)}</description><language>en</language>"
            f"<lastBuildDate>{format_datetime(_now())}</lastBuildDate>{self_link}" + "".join(items) + "</channel></rss>")


def _atom(site_title: str, articles: list[Article], base_url: str, self_path: str = "atom.xml") -> str:
    """Atom 1.0. Dates are RFC 3339; ids are the permalink, or a stable urn when no base URL is set."""
    base = base_url.rstrip("/")
    now = _now()
    site_id = base + "/" if base else f"urn:gauge:{_slug(site_title)}"
    entries = []
    newest = None
    for a in articles[:50]:
        when = _parse_when(a.published_at)
        stamp = (when or now).isoformat()
        if when and (newest is None or when > newest):
            newest = when
        link = _link(a, base_url)
        entry_id = link if base else f"urn:gauge:{_slug(site_title)}:{a.slug}"
        entries.append(
            f"<entry><title>{_x(a.title)}</title><id>{_x(entry_id)}</id>"
            f'<link rel="alternate" href="{_x(link)}"/><updated>{stamp}</updated>'
            + (f"<published>{stamp}</published>" if when else "")
            + f"<author><name>{_x(a.byline)}</name></author>"
            f'<category term="{_x(_KIND_LABEL.get(a.kind, a.kind.title()))}"/>'
            f"<summary>{_x(_feed_summary(a))}</summary></entry>")
    links = (f'<link rel="self" href="{_x(base + "/" + self_path)}"/><link rel="alternate" href="{_x(base + "/")}"/>' if base else "")
    return ('<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
            f"<title>{_x(site_title)}</title><id>{_x(site_id)}</id><updated>{(newest or now).isoformat()}</updated>{links}"
            + "".join(entries) + "</feed>")


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


def build_site(site_title: str, articles: list[Article], out_dir: Path, base_url: str = "",
               theme: str = DEFAULT_THEME) -> list[Path]:
    """Render the static site. Refuses to build if any article has an impersonating or missing byline, or
    an opinion piece carries a clickbait headline. Returns the files written."""
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}; choose one of: {', '.join(sorted(THEMES))}")
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

    _w("index.html", _index_html(site_title, ordered, base_url, theme))
    for a in ordered:
        _w(f"articles/{a.slug}.html", _article_html(a, base_url, theme, site_title))
        paths.append(f"articles/{a.slug}.html")
    # Per-author pages.
    by_author: dict[str, list[Article]] = {}
    for a in ordered:
        by_author.setdefault(a.byline, []).append(a)
    for author, arts in by_author.items():
        _w(f"authors/{_slug(author)}.html", _author_html(site_title, author, arts, theme))
        paths.append(f"authors/{_slug(author)}.html")
    _w("search.html", _search_html(site_title, theme))
    _w("search-index.json", _search_index(ordered))
    # Feeds: everything in RSS and Atom, plus a feed each for the serial and the opinion blog.
    _w("feed.xml", _rss(site_title, ordered, base_url))
    _w("atom.xml", _atom(site_title, ordered, base_url))
    serial = [a for a in ordered if a.kind == "chapter"]
    opinion = [a for a in ordered if a.kind == "opinion"]
    if serial:
        _w("feeds/serial.xml", _rss(f"{site_title}: the serial", serial, base_url, "feeds/serial.xml",
                                    f"New chapters of {site_title}."))
    if opinion:
        _w("feeds/opinion.xml", _rss(f"{site_title}: opinion", opinion, base_url, "feeds/opinion.xml",
                                     f"Opinion columns from {site_title}. Opinion, not news reporting."))
    _w("sitemap.xml", _sitemap(base_url, paths))
    _w("robots.txt", (f"User-agent: *\nAllow: /\nSitemap: {base_url.rstrip('/')}/sitemap.xml\n" if base_url
                      else "User-agent: *\nAllow: /\n"))
    return written
