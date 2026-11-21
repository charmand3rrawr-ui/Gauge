"""Roadmap items 78 (SEO/sitemap), 79 (accessibility), 82 (search), 83 (multi-author)."""

from gauge.site.publisher import Article, accessibility_report, build_site

ARTS = [
    Article("about", "Rin Writes", "Welcome to the site.\n\nNew chapters weekly.", "Jane Doe", kind="post",
            summary="A rain-soaked serial.", published_at="2026-11-01", thumbnail="cover.png", thumbnail_alt="a rainy dock"),
    Article("op1", "Serials reshape publishing", "The argument, made here.", "Al Rivers", kind="opinion",
            summary="An op-ed.", published_at="2026-11-02"),
    Article("ch1", "Chapter 1", "Rin ran through the rain.", "Jane Doe", kind="chapter", summary="Chapter one",
            published_at="2026-11-03"),
]


def _build(tmp_path, base_url="https://example.test"):
    return build_site("Rin Writes", ARTS, tmp_path / "site", base_url=base_url), tmp_path / "site"


def test_seo_sitemap_and_robots(tmp_path):
    _, site = _build(tmp_path)
    sm = (site / "sitemap.xml").read_text()
    assert "https://example.test/index.html" in sm and "https://example.test/articles/op1.html" in sm
    robots = (site / "robots.txt").read_text()
    assert "Sitemap: https://example.test/sitemap.xml" in robots
    # Canonical link present on an article page.
    assert 'rel="canonical" href="https://example.test/articles/op1.html"' in (site / "articles" / "op1.html").read_text()


def test_accessibility(tmp_path):
    _, site = _build(tmp_path)
    for page in (site / "index.html", site / "articles" / "about.html", site / "search.html"):
        assert accessibility_report(page.read_text()) == []  # lang, main landmark, skip link, alt text
    about = (site / "articles" / "about.html").read_text()
    assert 'alt="a rainy dock"' in about  # author-provided alt text on the image
    # A missing-alt image is caught by the checker.
    assert "image without alt text" in accessibility_report('<html lang="en"><main id="main">skip<img src="x"></main></html>')


def test_client_side_search(tmp_path):
    _, site = _build(tmp_path)
    assert (site / "search.html").exists() and "search-index.json" in (site / "search.html").read_text()
    import json
    idx = json.loads((site / "search-index.json").read_text())
    titles = {e["title"] for e in idx}
    assert "Serials reshape publishing" in titles and all("slug" in e and "body" in e for e in idx)


def test_multi_author_pages(tmp_path):
    files, site = _build(tmp_path)
    assert (site / "authors" / "jane-doe.html").exists() and (site / "authors" / "al-rivers.html").exists()
    jane = (site / "authors" / "jane-doe.html").read_text()
    assert "Chapter 1" in jane and "Rin Writes" in jane and "Serials reshape publishing" not in jane  # only Jane's
    # Bylines on the index link to the author page.
    assert 'authors/jane-doe.html' in (site / "index.html").read_text()


def test_build_still_guards_byline_and_clickbait(tmp_path):
    import pytest
    with pytest.raises(PermissionError):
        build_site("S", [Article("a", "T", "b", "Tyler Durden")], tmp_path / "s1")
    with pytest.raises(PermissionError):
        build_site("S", [Article("a", "You won't believe this SHOCKING truth", "b", "Jane Doe", kind="opinion")], tmp_path / "s2")
