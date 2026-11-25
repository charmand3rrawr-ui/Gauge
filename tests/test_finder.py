"""The book finder: Gauge searches Project Gutenberg by itself for novels its library does not have."""
import json

import pytest

from gauge.gui import StudioController
from gauge.ingest import finder as fd
from gauge.ingest.finder import BookFinder
from gauge.learn.corpus import compile_corpus


def _body(title="A Book"):
    return "\n\n".join(f"CHAPTER {n}\n\n" + (f"In part {n} of {title} the ship ran before the wind and the captain kept his own "
                                               "counsel about the island. ") * 200 for n in ("I", "II", "III", "IV", "V", "VI"))


BODY = _body()
HEADER = "The Project Gutenberg eBook of {t}\n\nThis ebook is for the use of anyone anywhere.\n\n*** START OF THE PROJECT GUTENBERG EBOOK {t} ***\n\n"
FOOTER = "\n\n*** END OF THE PROJECT GUTENBERG EBOOK ***\n\nlicence text that must not be kept"


def _book(i, title, subjects=("Sea stories", "Adventure stories"), copyright_=False, downloads=1000, author="Stevenson, Robert Louis",
          url=None):
    return {"id": i, "title": title, "authors": [{"name": author}], "subjects": list(subjects), "bookshelves": ["Adventure"],
            "languages": ["en"], "copyright": copyright_, "media_type": "Text", "download_count": downloads,
            "formats": {"text/plain; charset=utf-8": url or f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8", "text/html": "x"}}


class _Web:
    """A stand-in for the internet: answers by address, remembers every address asked for."""

    def __init__(self, pages=None, texts=None, down=False):
        self.pages, self.texts, self.asked, self.down = pages or {}, texts or {}, [], down

    def get(self, url, **kw):
        self.asked.append(url)
        if url.startswith(fd.CATALOGUE):
            from urllib.parse import parse_qs, urlparse
            if self.down:
                return None
            q = parse_qs(urlparse(url).query)
            n = int(q["page"][0])
            page = self.pages.get((q["topic"][0], n))
            if page is None:                     # like the real service: no results on page one, "no such page" after
                return json.dumps({"results": []}) if n == 1 else None
            return json.dumps({"results": page})
        return self.texts.get(url)


def _text(title, body=None):
    return HEADER.format(t=title.upper()) + (body or _body(title)) + FOOTER


def _finder(cfg, db, pages, texts=None, down=False):
    web = _Web(pages, texts, down)
    return BookFinder(cfg, db, fetcher=web), web


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    monkeypatch.delenv("GAUGE_OFFLINE", raising=False)          # these tests use a stand-in internet, never the real one


# ------------------------------------------------------------------ where it may go
def test_it_can_only_talk_to_the_catalogue_and_project_gutenberg():
    assert fd.allowed("https://gutendex.com/books?topic=x") and fd.allowed("https://www.gutenberg.org/ebooks/1.txt.utf-8")
    for bad in ("http://www.gutenberg.org/ebooks/1.txt", "https://gutenberg.org.evil.example/x", "https://freewebnovel.com/book",
                "https://www.royalroad.com/fiction/1", "file:///etc/passwd", "https://evil.example/?u=https://gutendex.com"):
        assert not fd.allowed(bad), bad
    assert fd.ALLOWED_HOSTS == {"gutendex.com", "www.gutenberg.org", "gutenberg.org"}


def test_a_book_address_elsewhere_or_still_in_copyright_never_gets_through(cfg, db):
    listing = [_book(1, "Elsewhere", url="https://evil.example/1.txt"), _book(2, "Protected", copyright_=True),
               _book(3, "Unknown Status", copyright_=None), _book(4, "Kidnapped")]
    assert [b["id"] for b in fd.parse_catalogue({"results": listing})] == [4]
    finder, web = _finder(cfg, db, {})
    assert finder._get("https://evil.example/book.txt") is None and web.asked == []      # refused before any request


# ------------------------------------------------------------------ what it looks for
def test_it_searches_for_what_you_write(cfg, db):
    assert fd.topics_for(["Sci-fi", "Dark Fantasy"]) == ["science fiction", "fantasy", "fairy tales"]
    ctl = StudioController(cfg, db)
    ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds", genres=["Mystery"])
    finder, _ = _finder(cfg, db, {})
    got = finder.interests()
    assert got[:2] == ["detective", "mystery"] and "adventure" in got                    # your genres first, then a spread
    cfg.library.finder_topics = ["Sea Stories"]
    assert finder.interests()[0] == "sea stories"                                        # what you name comes before everything


def test_only_prose_fiction_counts_as_a_novel():
    assert fd.is_novel({"subjects": ["Sea stories", "Pirates -- Fiction"], "bookshelves": []})
    assert not fd.is_novel({"subjects": ["Sea poetry"], "bookshelves": ["Poetry"]})
    assert not fd.is_novel({"subjects": ["English drama"], "bookshelves": []})
    assert not fd.is_novel({"subjects": ["Science fiction -- Periodicals"], "bookshelves": []})
    assert not fd.is_novel({"subjects": ["Cookery"], "bookshelves": []})


# ------------------------------------------------------------------ a run
def test_a_run_finds_adds_and_remembers(cfg, db):
    cfg.library.finder_topics = ["sea stories"]
    pages = {("sea stories", 1): [_book(120, "Treasure Island", downloads=9000), _book(421, "Kidnapped", downloads=4000),
                                  _book(7, "Sea Poems", subjects=("Sea poetry",), downloads=99999),
                                  _book(8, "A Pamphlet", downloads=50)]}
    texts = {f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8": _text(t) for i, t in ((120, "Treasure Island"), (421, "Kidnapped"))}
    texts["https://www.gutenberg.org/ebooks/8.txt.utf-8"] = _text("A Pamphlet", "Too short to be a novel. " * 50)
    finder, web = _finder(cfg, db, pages, texts)
    seen = []
    res = finder.run(books=2, progress=seen.append)
    assert [a["title"] for a in res["added"]] == ["Treasure Island", "Kidnapped"]        # most read first
    assert "Found and added 2 novels from Project Gutenberg: Treasure Island by Stevenson, Robert Louis" in res["message"]
    assert "out of copyright" in res["message"] and seen[0]["reading"] == "Treasure Island"
    novel = next(n for n in db.novels() if n.key == "gutenberg:120")
    assert novel.source == "gutenberg" and novel.url.endswith("/ebooks/120") and novel.genres == ["sea stories"]
    chapters = db.chapters(novel.key)
    assert len(chapters) == 6 and chapters[0].title.startswith("CHAPTER I") and "licence text" not in chapters[-1].text
    assert "START OF THE PROJECT GUTENBERG" not in chapters[0].text
    # The licence lets it be trained on, and the journal says what was added and why.
    rep = compile_corpus(db, cfg.projects_dir)
    assert len(rep.docs) == 12 and not rep.excluded and rep.docs[0].permission == "public_domain"
    log = finder.journal()
    assert [r["id"] for r in log] == [120, 421] and "sea stories novel you did not have (9,000 downloads)" in log[0]["why"]
    assert log[0]["licence"].startswith("public domain in the US") and log[0]["chapters"] == 6 and log[0]["words"] > 20000
    assert all(fd.allowed(u) for u in web.asked) and len(web.asked) == 3                 # one search, two books: nothing else

    # The next run does not fetch what it has, or what it passed over, again.
    pages[("sea stories", 1)].append(_book(900, "The Sea-Wolf", author="London, Jack", downloads=3000))
    texts["https://www.gutenberg.org/ebooks/900.txt.utf-8"] = _text("The Sea-Wolf")
    web.asked.clear()
    again = finder.run(books=5, topics=["sea stories"])
    assert [a["title"] for a in again["added"]] == ["The Sea-Wolf"]
    assert "https://www.gutenberg.org/ebooks/120.txt.utf-8" not in web.asked
    assert "https://www.gutenberg.org/ebooks/8.txt.utf-8" in web.asked                    # the pamphlet was looked at once...
    web.asked.clear()
    finder.run(books=5, topics=["sea stories"])
    assert "https://www.gutenberg.org/ebooks/8.txt.utf-8" not in web.asked                # ...and not again


def test_a_book_you_already_have_under_another_source_is_not_added_twice(cfg, db, tmp_path):
    ctl = StudioController(cfg, db)
    mine = tmp_path / "Treasure Island.txt"
    mine.write_text(BODY, encoding="utf-8")
    ctl.add_material([str(mine)])
    finder, web = _finder(cfg, db, {("adventure", 1): [_book(120, "Treasure Island; or, the mutiny of the Hispaniola")]},
                          {"https://www.gutenberg.org/ebooks/120.txt.utf-8": _text("Treasure Island")})
    res = finder.run(books=3, topics=["adventure"])
    assert res["added"] == [] and len(web.asked) == 1                                    # searched, did not download


def test_a_book_that_says_it_is_copyrighted_is_dropped(cfg, db):
    marked = "The Project Gutenberg eBook\n\n** This is a COPYRIGHTED Project Gutenberg eBook, Details Below **\n\n" + BODY
    assert fd.says_copyrighted(marked) and not fd.says_copyrighted(_text("Kidnapped"))
    finder, _ = _finder(cfg, db, {("adventure", 1): [_book(5, "Modern Book")]},
                        {"https://www.gutenberg.org/ebooks/5.txt.utf-8": marked})
    assert finder.run(books=1, topics=["adventure"])["added"] == [] and not list(db.novels())


def test_limits_offline_dry_run_and_plan(cfg, db, monkeypatch):
    from gauge import dryrun

    pages = {("adventure", 1): [_book(i, f"Book {i}") for i in range(1, 6)]}
    texts = {f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8": _text(f"Book {i}") for i in range(1, 6)}
    finder, web = _finder(cfg, db, pages, texts)
    plan = finder.run(topics=["adventure"], plan_only=True)
    assert "Would search Project Gutenberg for up to 3 novels on: adventure" in plan["message"] and web.asked == []
    dryrun.set_dry_run(True)
    try:
        assert "Nothing was downloaded" in finder.run(topics=["adventure"])["message"] and web.asked == []
        assert any("find novels online" in h["action"] for h in dryrun.held())
    finally:
        dryrun.set_dry_run(False)
        dryrun.reset()
    monkeypatch.setenv("GAUGE_OFFLINE", "1")
    assert "offline mode" in finder.run(topics=["adventure"])["message"] and web.asked == []
    monkeypatch.delenv("GAUGE_OFFLINE")

    assert len(finder.run(topics=["adventure"])["added"]) == 3                           # the default per run
    cfg.library.finder_max_mb = 0.0001                                                   # the ceiling on what it may add
    assert "added its limit" in finder.run(topics=["adventure"])["message"]
    cfg.library.finder_max_mb = 300
    cfg.library.finder_max_book_mb = 0.01                                                # one enormous book is passed over
    assert finder.run(topics=["adventure"])["added"] == []
    stopped = BookFinder(cfg, db, fetcher=web).run(topics=["adventure"], should_stop=lambda: True)
    assert stopped["added"] == []


def test_nothing_new_and_a_silent_catalogue_are_said_plainly(cfg, db):
    finder, _ = _finder(cfg, db, {}, down=True)
    assert "found nothing new" in finder.run(topics=["adventure"])["message"]
    finder, _ = _finder(cfg, db, {("adventure", 1): [_book(7, "Sea Poems", subjects=("Sea poetry",))]})
    assert "Looked at 1 book online and none was a novel worth adding" in finder.run(topics=["adventure"])["message"]


def test_it_reads_further_into_the_catalogue_each_time(cfg, db):
    pages = {("adventure", 1): [_book(1, "One")], ("adventure", 2): [_book(2, "Two")]}
    texts = {f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8": _text(t) for i, t in ((1, "One"), (2, "Two"))}
    finder, _ = _finder(cfg, db, pages, texts)
    assert [a["title"] for a in finder.run(books=1, topics=["adventure"])["added"]] == ["One"]
    assert [a["title"] for a in finder.run(books=1, topics=["adventure"])["added"]] == []     # page one is used up...
    assert [a["title"] for a in finder.run(books=1, topics=["adventure"])["added"]] == ["Two"]  # ...so it turns the page


# ------------------------------------------------------------------ when the catalogue service is down
CSV = ('Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n'
       '120,Text,1994-02-01,Treasure Island,en,"Stevenson, Robert Louis, 1850-1894","Sea stories; Pirates -- Fiction",PR,Adventure\n'
       '2000,Text,1999-12-01,Don Quijote,es,"Cervantes Saavedra, Miguel de, 1547-1616",Spain -- Fiction,PQ,\n'
       '9000,Sound,2003-01-01,Treasure Island (audio),en,"Stevenson, Robert Louis, 1850-1894",Sea stories,PR,\n'
       '421,Text,1996-02-01,"Kidnapped\r\nBeing Memoirs of David Balfour",en,"Stevenson, Robert Louis, 1850-1894",Sea stories; Adventure stories,PR,\n')


def test_project_gutenbergs_own_catalogue_file_is_used_when_the_service_does_not_answer(cfg, db):
    books = fd.parse_offline_catalogue(CSV, ["en"])
    assert [b["id"] for b in books] == [120, 421]                                        # text, in English, lowest number first
    assert books[0]["authors"] == ["Stevenson, Robert Louis"] and books[0]["subjects"] == ["Sea stories", "Pirates -- Fiction"]
    assert books[1]["title"] == "Kidnapped Being Memoirs of David Balfour"
    assert books[0]["text_url"] == "https://www.gutenberg.org/ebooks/120.txt.utf-8"

    texts = {fd.OFFLINE_CATALOGUE: CSV, "https://www.gutenberg.org/ebooks/120.txt.utf-8": _text("Treasure Island"),
             "https://www.gutenberg.org/ebooks/421.txt.utf-8": _text("Kidnapped")}
    finder, web = _finder(cfg, db, {}, texts, down=True)                                 # the service answers nothing
    res = finder.run(books=2, topics=["sea stories"])
    assert [a["title"] for a in res["added"]] == ["Treasure Island", "Kidnapped Being Memoirs of David Balfour"]
    assert web.asked.count(fd.OFFLINE_CATALOGUE) == 1 and (cfg.models_dir / fd.OFFLINE_FILE).exists()
    assert sum(1 for u in web.asked if u.startswith(fd.CATALOGUE)) == 1                  # it stopped asking the silent service
    later, web2 = _finder(cfg, db, {}, texts, down=True)                                 # the file is kept: not fetched again
    later.run(books=1, topics=["sea stories"])
    assert fd.OFFLINE_CATALOGUE not in web2.asked


# ------------------------------------------------------------------ by itself
def test_by_itself_means_switched_on_and_due(cfg, db):
    pages = {("adventure", 1): [_book(1, "One"), _book(2, "Two")]}
    texts = {f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8": _text(t) for i, t in ((1, "One"), (2, "Two"))}
    cfg.library.finder_topics, cfg.library.finder_books = ["adventure"], 1
    finder, web = _finder(cfg, db, pages, texts)
    assert finder.run_if_due() is None and web.asked == []                               # off by default: nothing happens
    cfg.library.finder = True
    first = finder.run_if_due()
    assert first and len(first["added"]) == 1
    assert finder.run_if_due() is None                                                   # not again until a day has passed
    cfg.library.finder_hours = 0
    assert len(finder.run_if_due()["added"]) == 1

    def broken(*a, **k):
        raise RuntimeError("no network")

    finder.search = broken                                                               # it never stops the work it was called from
    assert finder.run_if_due() is None


def test_the_studio_and_the_command_line(cfg, db, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from gauge.cli import main

    pages = {("adventure", 1): [_book(1, "One"), _book(2, "Two")]}
    texts = {f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8": _text(t) for i, t in ((1, "One"), (2, "Two"))}
    web = _Web(pages, texts)
    monkeypatch.setattr("gauge.ingest.http.Fetcher", lambda cfg_: web)
    cfg.library.finder_topics = ["adventure"]
    cfg.save()
    ctl = StudioController(cfg, db)
    assert not ctl.finder_by_itself()["on"] and "only looks for novels when you press" in ctl.finder_by_itself()["message"]
    res = ctl.find_novels(books=1)
    assert len(res["added"]) == 1 and "Learn from the library now" in res["message"]
    assert [r["title"] for r in ctl.library()] == ["One"] and "gutenberg" in ctl.library_line(ctl.library()[0])
    on = ctl.finder_by_itself(True)
    assert on["on"] and "by itself: up to 3 about every 24 hours" in on["message"]

    runner = CliRunner(env={"COLUMNS": "200"})

    def run(*args):
        r = runner.invoke(main, ["--data-dir", cfg.data_dir, "corpus", "find", *args])
        assert r.exit_code == 0, r.output
        return r.output

    assert "Would search Project Gutenberg" in run("--plan")
    out = run("--books", "1")
    assert "Found and added 1 novel from Project Gutenberg: Two" in out and "reading Two" in out
    log = run("--log")
    assert "One" in log and "Two" in log and "adventure" in log
    assert "only look when you ask" in run("--by-itself", "off")


def test_the_teach_window_has_find_novels_online(cfg, db, tmp_path, monkeypatch):
    import sys

    from gauge import gui
    from tests.test_teach import _fake_tkinter, _wait_for_jobs, _Widget

    web = _Web({("adventure", 1): [_book(1, "Treasure Island")]}, {"https://www.gutenberg.org/ebooks/1.txt.utf-8": _text("Treasure Island")})
    monkeypatch.setattr("gauge.ingest.http.Fetcher", lambda cfg_: web)
    cfg.library.finder_topics = ["adventure"]
    mods, shown = _fake_tkinter(tmp_path)
    made: list = []

    class _Check(_Widget):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    mods["tkinter.ttk"].Checkbutton = _Check
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    _Widget.buttons["Teach Gauge"]()
    box = _Widget.listboxes[-1]
    _Widget.buttons["Find novels online"]()                       # the yes/no stand-in answers yes
    _wait_for_jobs()
    asked = [s for s in shown if s[0] == "ask"][-1][1][1]
    assert "Project Gutenberg" in asked and "cannot visit other sites" in asked and "Nothing about you" in asked
    assert "Treasure Island" in box.get()
    auto = next(c for c in made if "Keep finding novels by itself" in str(c.kw.get("text")))
    auto.kw["variable"].set(True)
    auto.kw["command"]()
    assert cfg.library.finder is True
    assert not [s for s in shown if s[0] == "error"], shown
