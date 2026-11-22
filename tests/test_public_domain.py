"""Fetch public-domain / openly-licensed books into the training library, licence-tagged."""
import json

from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.db import Database
from gauge.ingest.public_domain import fetch_gutenberg, fetch_wikisource, store_books
from gauge.learn.corpus import compile_corpus, training_permission

GUTENDEX = {"results": [{"id": 2701, "title": "Moby Dick", "authors": [{"name": "Melville, Herman", "death_year": 1891}],
                         "subjects": [], "formats": {"text/plain; charset=utf-8": "https://gutenberg.org/2701.txt"},
                         "download_count": 9000}]}
BOOK = ("*** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***\n" + ("Call me Ishmael. " * 300)
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***")
WIKI = json.dumps({"title": "On Liberty", "source": "=='''Chapter 1'''==\nThe [[subject|liberty]] of all. " + ("Free thought. " * 200)})


class FakeFetcher:
    def get(self, url):
        if "gutendex" in url:
            return json.dumps(GUTENDEX)
        if url.endswith(".txt"):
            return BOOK
        if "wikisource" in url:
            return WIKI
        return None


def test_fetch_gutenberg_full_text():
    books = fetch_gutenberg(FakeFetcher(), "moby dick")
    assert len(books) == 1
    b = books[0]
    assert b.source == "gutenberg" and b.title == "Moby Dick" and "Melville" in b.author
    assert "START OF THE PROJECT GUTENBERG" not in b.text and len(b.text.split()) > 200  # boilerplate stripped


def test_fetch_wikisource_strips_markup():
    books = fetch_wikisource(FakeFetcher(), "On Liberty")
    assert books and books[0].source == "wikisource"
    assert "'''" not in books[0].text and "[[" not in books[0].text and "liberty of all" in books[0].text.lower()


def test_permissions_cover_the_new_sources():
    for src in ("standardebooks", "standard_ebooks", "wikisource", "publicdomain", "cc0", "cc-by", "cc-by-sa"):
        assert training_permission(src)[1] is True  # all trainable
    assert training_permission("royalroad")[1] is False  # still excluded


def test_stored_books_enter_the_corpus(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    books = fetch_gutenberg(FakeFetcher(), "x") + fetch_wikisource(FakeFetcher(), "On Liberty")
    assert store_books(db, books) == 2
    rep = compile_corpus(db, cfg.projects_dir)
    assert {d.source for d in rep.docs} == {"gutenberg", "wikisource"} and rep.excluded == []


def test_import_source_tag_marks_permitted(tmp_path):
    # A Standard Ebooks / CC file imported with --source-tag is trainable; the corpus includes it.
    from gauge.ingest.local import import_novel
    book = tmp_path / "liberty.txt"
    book.write_text("A public-domain essay. " * 200, encoding="utf-8")
    imp = import_novel(book, meta_override={"source": "standardebooks", "title": "Liberty"})
    assert imp.novel.source == "standardebooks"
    assert training_permission(imp.novel.source)[1] is True


def test_cli_corpus_fetch_offline_is_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("GAUGE_OFFLINE", "1")
    data = str(tmp_path / "data")
    GaugeConfig(data_dir=data).save()
    r = CliRunner(env={"COLUMNS": "200"}).invoke(main, ["--data-dir", data, "corpus", "fetch", "--source", "gutenberg", "--query", "x"])
    assert r.exit_code == 0 and "offline" in r.output  # does not attempt the network in offline mode
