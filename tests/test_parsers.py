from gauge.ingest import royalroad, scribblehub
from tests.conftest import FIXTURES


def test_royalroad_fiction_page():
    html = (FIXTURES / "rr_fiction.html").read_text(encoding="utf-8")
    n = royalroad.parse_fiction_page(html, "https://www.royalroad.com/fiction/12345/the-test-novel")
    assert n.source_id == "12345"
    assert n.title == "The Test Novel"
    assert n.author == "Testy McAuthor"
    assert n.followers == 12345 and n.favorites == 2345 and n.views == 1234567 and n.pages == 890
    assert n.rating == 4.5 and n.rating_count == 678
    assert "LitRPG" in n.tags and n.status == "ongoing"
    assert n.chapter_count == 2
    assert "rewind" in n.blurb.lower()
    refs = royalroad.parse_chapter_list(html, n.url)
    assert [r.source_chapter_id for r in refs] == ["111", "222"]
    assert refs[0].published_at is not None and refs[0].published_at < refs[1].published_at


def test_royalroad_chapter_and_listing():
    html = (FIXTURES / "rr_chapter.html").read_text(encoding="utf-8")
    text = royalroad.parse_chapter_page(html)
    assert text.startswith("\"Don't open it,\"")
    assert "advert" not in text.lower()
    assert "\n\n" in text
    listing = (FIXTURES / "rr_listing.html").read_text(encoding="utf-8")
    urls = royalroad.parse_listing(listing)
    assert urls == ["https://www.royalroad.com/fiction/1/a", "https://www.royalroad.com/fiction/2/b"]


def test_scribblehub_series_toc_chapter():
    html = (FIXTURES / "sh_series.html").read_text(encoding="utf-8")
    n = scribblehub.parse_series_page(html, "https://www.scribblehub.com/series/999/some-series/")
    assert n.source_id == "999" and n.title == "Some Series" and n.author == "Writer"
    assert n.views == 1500000 and n.favorites == 12000 and n.followers == 3400 and n.chapter_count == 120
    assert n.rating == 4.7
    assert n.genres == ["Fantasy", "Romance"] and "Cultivation" in n.tags
    refs = scribblehub.parse_toc(html)
    assert [r.index for r in refs] == [1, 2] and refs[0].title == "Chapter 1" and refs[0].source_chapter_id == "10"
    ch = (FIXTURES / "sh_chapter.html").read_text(encoding="utf-8")
    text = scribblehub.parse_chapter_page(ch)
    assert "Author note" not in text and text.startswith("The sect gates")


def test_source_matching(cfg):
    from gauge.ingest import build_sources, source_for_url

    srcs = build_sources(cfg)
    assert source_for_url(srcs, "https://www.royalroad.com/fiction/1/x").name == "royalroad"
    assert source_for_url(srcs, "https://www.scribblehub.com/series/1/x/").name == "scribblehub"
    assert source_for_url(srcs, "https://example.com") is None
