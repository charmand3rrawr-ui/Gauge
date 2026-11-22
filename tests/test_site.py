"""Batch K: the author site publisher and its content-integrity rails."""

import pytest
from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.llm import FakeLLM
from gauge.market.policy import REFUSED, check_action
from gauge.site.integrity import check_byline, check_headline, flag_claims
from gauge.site.opinion import draft_opinion
from gauge.site.publisher import Article, build_site


def test_policy_refuses_news_and_clickbait_and_impersonation():
    for bad in ("fabricate_news", "clickbait_deception", "impersonate_outlet"):
        assert check_action(bad)[0] is False and bad in REFUSED
    for good in ("publish_own_site", "opinion_labeled", "honest_headline"):
        assert check_action(good)[0] is True


def test_byline_guard():
    assert check_byline("Jane Doe")[0] is True
    assert check_byline("")[0] is False
    for impostor in ("Tyler Durden", "ZeroHedge", "Reuters", "staff", "The Editor"):
        assert check_byline(impostor)[0] is False


def test_headline_clickbait_detector():
    assert check_headline("The Fed held rates steady; what changed").ok is True
    for bait in ("You won't believe what happened next", "SHOCKING truth they don't want you to know",
                 "This one weird trick destroys inflation", "BREAKING!!!"):
        assert check_headline(bait).ok is False and check_headline(bait).flags


def test_flag_claims():
    flags = flag_claims('Inflation hit 9.1% yesterday, and a study found "most people agree".')
    kinds = " ".join(flags)
    assert "statistic" in kinds and ("quote" in kinds or "attributed" in kinds or "dated" in kinds)
    assert flag_claims("This is pure argument with no specific factual claims to check.") == []


def test_draft_opinion_is_labelled_and_guarded():
    d = draft_opinion(FakeLLM(), "Serial fiction is reshaping publishing", "Jane Doe")
    assert d.byline == "Jane Doe" and "Opinion:" in d.body and d.title
    assert "By Jane Doe" in d.body
    with pytest.raises(PermissionError):
        draft_opinion(FakeLLM(), "anything", "Tyler Durden")  # impersonation byline refused


def test_build_site_renders_and_labels_opinion(tmp_path):
    arts = [
        Article("about", "Ten Seconds", "A courier who can rewind.\n\nNew chapters weekly.", "Jane Doe",
                kind="post", summary="A rain-soaked serial.", published_at="2026-11-01"),
        Article("op1", "Serials are reshaping publishing", "Opinion body argued here.", "Jane Doe",
                kind="opinion", summary="An op-ed.", published_at="2026-11-02"),
    ]
    written = build_site("Rin Writes", arts, tmp_path / "site", base_url="https://me")
    assert (tmp_path / "site" / "index.html").exists() and (tmp_path / "site" / "feed.xml").exists()
    op_html = (tmp_path / "site" / "articles" / "op1.html").read_text(encoding="utf-8")
    assert "Opinion" in op_html and "Jane Doe" in op_html and "authors/jane-doe.html" in op_html  # labelled, bylined, linked
    index = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "Ten Seconds" in index and "og:title" in index
    assert len(written) >= 4 and any(p.name == "sitemap.xml" for p in written)


def test_build_site_refuses_impersonation_and_clickbait(tmp_path):
    with pytest.raises(PermissionError):
        build_site("S", [Article("a", "T", "b", "Tyler Durden")], tmp_path / "s1")
    with pytest.raises(PermissionError):
        build_site("S", [Article("a", "You won't believe this SHOCKING truth", "b", "Jane Doe", kind="opinion")], tmp_path / "s2")


def test_cli_site(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.llm.provider = "fake"
    cfg.save()
    from gauge.write.project import Project
    p = Project.create(cfg.projects_dir, "Ten Seconds", "a courier who can rewind", ["Sci-fi"], 300)
    p.write_chapter(1, "The Seal", "word " * 80, {"title": "The Seal", "score": 0.7, "written_at": "2026-11-01"})
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", data, "site", "opinion", "Ten Seconds",
                             "--thesis", "Serials reshape publishing", "--byline", "Jane Doe"])
    assert r.exit_code == 0 and "Opinion" in r.output
    r = runner.invoke(main, ["--data-dir", data, "site", "opinion", "Ten Seconds", "--thesis", "x", "--byline", "ZeroHedge"])
    assert r.exit_code == 1 and "refused" in r.output
    r = runner.invoke(main, ["--data-dir", data, "site", "build", "Ten Seconds", "--byline", "Jane Doe", "--url", "https://me"])
    assert r.exit_code == 0 and "built" in r.output
    assert (cfg.projects_dir / "ten-seconds" / "site" / "index.html").exists()


# ------------------------------------------------------------------ themes and the light/dark toggle
def _arts():
    return [
        Article("about", "Ten Seconds", "A courier who can rewind.", "Jane Doe", kind="post",
                summary="A rain-soaked serial.", published_at="2026-11-01"),
        Article("chapter-001", "Chapter 1: The Seal", "Rain fell.", "Jane Doe", kind="chapter",
                summary="Chapter 1 of Ten Seconds", published_at="2026-11-03"),
        Article("op1", "Serials are reshaping publishing", "Opinion body argued here.", "Jane Doe",
                kind="opinion", summary="An op-ed & a half.", published_at="2026-11-02T09:30:00"),
    ]


def test_every_theme_meets_contrast_in_both_modes():
    from gauge.site.publisher import THEMES, theme_contrast_issues
    assert set(THEMES) >= {"classic", "paper", "slate", "noir"}
    for name in THEMES:
        assert theme_contrast_issues(name) == [], name  # text, muted text and accent all clear WCAG AA


def test_theme_css_follows_system_and_honours_the_toggle():
    from gauge.site.publisher import THEMES, theme_css
    css = theme_css("slate")
    assert THEMES["slate"]["light"]["bg"] in css and THEMES["slate"]["dark"]["bg"] in css
    assert "prefers-color-scheme:dark" in css and ":root:not([data-theme=light])" in css  # system dark, unless reader chose light
    assert ":root[data-theme=dark]" in css                                               # reader chose dark
    assert theme_css("no-such-theme") == theme_css("classic")                             # safe fallback


def test_built_pages_carry_theme_toggle_and_feed_links(tmp_path):
    from gauge.site.publisher import THEMES, accessibility_report
    build_site("Rin Writes", _arts(), tmp_path / "site", base_url="https://me", theme="paper")
    page = (tmp_path / "site" / "articles" / "op1.html").read_text(encoding="utf-8")
    assert THEMES["paper"]["light"]["bg"] in page
    assert 'id="theme-toggle"' in page and "aria-pressed" in page and " hidden>" in page  # no dead button without JS
    assert "gauge-theme" in page                                                         # the choice is remembered
    assert 'type="application/rss+xml"' in page and 'href="../feed.xml"' in page         # autodiscovery, correct depth
    assert 'type="application/atom+xml"' in page
    assert accessibility_report(page) == []
    with pytest.raises(ValueError):
        build_site("S", _arts(), tmp_path / "s2", theme="neon")


# ------------------------------------------------------------------ RSS and Atom
def test_feeds_are_well_formed_and_dated(tmp_path):
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    site = tmp_path / "site"
    build_site("Rin Writes", _arts(), site, base_url="https://me")
    rss = ET.fromstring((site / "feed.xml").read_text(encoding="utf-8"))
    items = rss.findall("./channel/item")
    assert len(items) == 3
    first = items[0]
    assert first.findtext("link") == "https://me/articles/chapter-001.html"   # newest first
    assert first.find("guid").get("isPermaLink") == "true"
    assert parsedate_to_datetime(first.findtext("pubDate")).year == 2026       # RFC 822 date
    assert first.findtext("{http://purl.org/dc/elements/1.1/}creator") == "Jane Doe"
    self_link = rss.find("./channel/{http://www.w3.org/2005/Atom}link")
    assert self_link.get("href") == "https://me/feed.xml" and self_link.get("rel") == "self"

    ns = {"a": "http://www.w3.org/2005/Atom"}
    atom = ET.fromstring((site / "atom.xml").read_text(encoding="utf-8"))
    entries = atom.findall("a:entry", ns)
    assert len(entries) == 3 and atom.findtext("a:id", namespaces=ns) == "https://me/"
    for e in entries:
        assert e.findtext("a:id", namespaces=ns).startswith("https://me/articles/")
        assert e.findtext("a:updated", namespaces=ns).endswith("+00:00")       # RFC 3339 with an offset
        assert e.findtext("a:author/a:name", namespaces=ns) == "Jane Doe"


def test_opinion_is_labelled_in_feeds_and_sections_split(tmp_path):
    import xml.etree.ElementTree as ET
    site = tmp_path / "site"
    written = build_site("Rin Writes", _arts(), site, base_url="https://me")
    names = {str(p.relative_to(site)) for p in written}
    assert {"feed.xml", "atom.xml", "feeds/serial.xml", "feeds/opinion.xml"} <= names
    op = ET.fromstring((site / "feeds" / "opinion.xml").read_text(encoding="utf-8"))
    items = op.findall("./channel/item")
    assert len(items) == 1 and items[0].findtext("category") == "Opinion"
    assert items[0].findtext("description").startswith("[Opinion]")            # labelled wherever it travels
    assert "&" in items[0].findtext("description")                             # escaped in the file, intact when parsed
    serial = ET.fromstring((site / "feeds" / "serial.xml").read_text(encoding="utf-8"))
    assert [i.findtext("category") for i in serial.findall("./channel/item")] == ["Chapter"]


def test_feeds_without_base_url_or_dates_stay_valid(tmp_path):
    import xml.etree.ElementTree as ET
    site = tmp_path / "site"
    arts = [Article("a", "Undated \x07 note", "Body.", "Jane Doe", summary="No date.")]  # a stray control char
    written = build_site("Rin Writes", arts, site)
    assert not (site / "feeds").exists()                                       # no serial/opinion -> no section feeds
    rss = ET.fromstring((site / "feed.xml").read_text(encoding="utf-8"))
    item = rss.find("./channel/item")
    assert item.find("pubDate") is None and item.find("guid").get("isPermaLink") == "false"
    assert item.findtext("title") == "Undated  note"                           # control char dropped, still well-formed
    atom = ET.fromstring((site / "atom.xml").read_text(encoding="utf-8"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    assert atom.findtext("a:entry/a:id", namespaces=ns) == "urn:gauge:rin-writes:a"
    assert any(p.name == "atom.xml" for p in written)
