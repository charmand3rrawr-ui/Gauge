"""The Scout: open-source research connectors (parsed offline), caching, filing, agent integration, CLI."""
import json
import os

from gauge.learn import Trainer
from gauge.llm import FakeLLM, HashingEmbedder
from gauge.research import Scout, Snippet
from gauge.research.scout import Dossier, Finding, build_sources
from gauge.research.sources import (
    ArxivSource,
    ClimateSource,
    FakeSource,
    GutenbergSource,
    MediaWikiSource,
    SearxSource,
    best_passages,
    lead_paragraphs,
    parse_arxiv,
    parse_core_search,
    parse_geocode,
    parse_gutendex,
    parse_mediawiki_extracts,
    parse_mediawiki_search,
    parse_searx,
    strip_gutenberg_boilerplate,
    summarize_climate,
)
from gauge.write import WritePipeline
from gauge.write.agents.scout import ScoutAgent, extract_topics
from gauge.write.project import ChapterPlan, Project


class _Fetcher:
    """Scripted HTTP: maps URL substrings to bodies, records what was requested."""

    def __init__(self, routes):
        self.routes = routes
        self.urls = []

    def get(self, url, **kw):
        self.urls.append(url)
        for key, body in self.routes.items():
            if key in url:
                return body
        return None


# ------------------------------------------------------------------ parsers
def test_mediawiki_core_api_parsing_and_source():
    search = {"pages": [{"title": "Courier", "key": "Courier", "excerpt": "A <span class=\"searchmatch\">courier</span> delivers messages.",
                         "description": "person who delivers messages"},
                        {"title": "Guild", "key": "Guild", "excerpt": "An association", "description": "association of artisans"}]}
    assert parse_core_search(search)[0]["excerpt"] == "A courier delivers messages."
    html = ("<html><body><table class='infobox'><tr><td>junk junk junk junk junk junk junk junk junk junk junk junk junk junk junk junk</td></tr></table>"
            "<p>Short.</p><p>A courier is a person or company that delivers messages, packages and letters, historically on foot "
            "or horseback, and was often paid per delivery by a guild or a lord.<sup>[1]</sup></p>"
            "<p>In medieval Europe couriers were organised into guilds which fined members for late deliveries and protected "
            "sealed correspondence under penalty of law.</p><p>Third paragraph that should not be included in a two paragraph lead.</p></body></html>")
    lead = lead_paragraphs(html, n=2)
    assert lead.startswith("A courier is a person") and "[1]" not in lead and "junk" not in lead and "Third paragraph" not in lead
    f = _Fetcher({"search/page": json.dumps(search), "page/Courier/html": html})
    src = MediaWikiSource(f, "wikipedia", "en")
    out = src.search("courier guild", k=2)
    assert [s.title for s in out] == ["Courier", "Guild"]
    assert out[0].text.startswith("A courier is a person") and out[0].url == "https://en.wikipedia.org/wiki/Courier"
    assert out[1].text == "association of artisans An association"  # no lead fetched -> description + excerpt
    assert all(u.startswith("https://api.wikimedia.org/core/v1/wikipedia/en/") for u in f.urls)
    assert MediaWikiSource(_Fetcher({}), "wikivoyage", "en").search("x") == []
    # old Action API parsers still work
    assert parse_mediawiki_search({"query": {"search": [{"title": "A"}]}}) == ["A"]
    assert parse_mediawiki_extracts({"query": {"pages": {"1": {"title": "A", "extract": "<b>x</b> y"}}}})["A"] == "x y"


def test_gutenberg_parsing_passages_and_source():
    payload = {"results": [{"id": 345, "title": "Dracula", "authors": [{"name": "Stoker, Bram", "death_year": 1912}],
                            "subjects": ["Horror tales", "Vampires -- Fiction"], "download_count": 9000,
                            "formats": {"text/plain; charset=utf-8": "https://gutenberg.org/files/345.txt", "image/jpeg": "x.jpg"}}]}
    books = parse_gutendex(payload)
    assert books[0]["text_url"].endswith("345.txt") and books[0]["authors"] == ["Stoker, Bram"]
    raw = ("junk header\n*** START OF THE PROJECT GUTENBERG EBOOK DRACULA ***\n\nChapter 1\n\n" + "The courier arrived at the castle gate with his sealed bag and a letter for the Count. " * 6
           + "\n\nUnrelated paragraph about the weather and the mountains and nothing else at all for a while. " * 4
           + "\n\n*** END OF THE PROJECT GUTENBERG EBOOK DRACULA ***\nlicense junk")
    text = strip_gutenberg_boilerplate(raw)
    assert text.startswith("Chapter 1") and "license junk" not in text and "junk header" not in text
    passages = best_passages(text, "courier sealed bag", k=1)
    assert passages and "sealed bag" in passages[0]
    f = _Fetcher({"gutendex.com": json.dumps(payload), "345.txt": raw})
    src = GutenbergSource(f)
    out = src.search("courier castle", k=1)
    assert out and out[0].source == "gutenberg" and out[0].title == "Dracula" and "public domain" in out[0].license
    assert out[0].extra["authors"] == ["Stoker, Bram"]


def test_arxiv_climate_searx_parsing():
    xml = '''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1</id>
    <title>  Ten-second   closed timelike curves </title><summary>We show that short rewinds cost information.</summary></entry></feed>'''
    assert parse_arxiv(xml)[0]["title"] == "Ten-second closed timelike curves"
    assert ArxivSource(_Fetcher({"export.arxiv.org": xml})).search("time rewind", k=1)[0].source == "arxiv"
    geo = {"results": [{"name": "Lisbon", "country": "Portugal", "latitude": 38.7, "longitude": -9.1, "elevation": 45, "timezone": "Europe/Lisbon"}]}
    assert parse_geocode(geo)["name"] == "Lisbon" and parse_geocode({"results": []}) is None
    daily = {"daily": {"temperature_2m_max": [18, 17, 19], "temperature_2m_min": [11, 10, 12], "precipitation_sum": [0, 5.2, 0.3]}}
    s = summarize_climate(daily, "Lisbon, Portugal in 2024-11")
    assert "highs 18 C" in s and "1 of 3 days with rain" in s
    f = _Fetcher({"geocoding-api": json.dumps(geo), "archive-api": json.dumps(daily)})
    out = ClimateSource(f).search("Lisbon in November")
    assert out and out[0].title == "Climate: Lisbon" and "2024-11" in out[0].text and "month=11" not in f.urls[1]
    assert "start_date=2024-11-01" in f.urls[1]
    sx = {"results": [{"title": "Guild fines", "url": "https://x.org/a", "content": "Late couriers were fined."}]}
    assert parse_searx(sx)[0]["title"] == "Guild fines"
    assert SearxSource(_Fetcher({"/search?": json.dumps(sx)}), "http://localhost:8080").search("x")[0].url == "https://x.org/a"


# ------------------------------------------------------------------ scout: cache, filing, gutenberg import
def test_scout_caches_files_and_imports(cfg, db, tmp_path):
    fake = FakeSource("wikipedia")
    scout = Scout(cfg, db, sources={"wikipedia": fake})
    f1 = scout.search("medieval courier guild")
    assert f1.snippets and not f1.cached
    f2 = scout.search("medieval courier guild")
    assert f2.cached and fake.queries == ["medieval courier guild"]  # second hit served from SQLite
    assert db.research_cached("nope") is None
    d = scout.research([{"query": "medieval courier guild", "source": "wikipedia", "why": "pay"}, {"query": "night market", "source": "bogus"}],
                       project="P", chapter=3)
    assert len(d.findings) == 2 and d.findings[1].source == "wikipedia"  # unknown source falls back
    hist = db.research_history("P")
    assert len(hist) == 2 and hist[-1]["cached"] == 1
    proj = Project.create(tmp_path / "pp", "P", "idea", ["Fantasy"], 300)
    path = scout.file(proj.dir, d, "chapter 003 The Market")
    assert path.name == "chapter-003-the-market.md" and "never copy sentences" in path.read_text(encoding="utf-8")
    bib = (proj.dir / "research" / "sources.md").read_text(encoding="utf-8")
    assert "[wikipedia] About medieval courier guild" in bib
    scout.file(proj.dir, d, "chapter 003 The Market")  # idempotent bibliography
    assert (proj.dir / "research" / "sources.md").read_text(encoding="utf-8").count("About medieval courier guild") == 1
    assert scout.file(proj.dir, Dossier([Finding("q", "wikipedia", [])]), "empty") is None
    # gutenberg import into the corpus
    payload = {"results": [{"id": 7, "title": "Old Book", "authors": [{"name": "Anon"}], "subjects": ["Adventure stories -- Fiction"],
                            "download_count": 42, "formats": {"text/plain": "https://g.org/7.txt"}}]}
    body = "*** START OF THE PROJECT GUTENBERG EBOOK OLD BOOK ***\n\n" + "\n\n".join(f"Chapter {i}\n\n" + ("The road was long and the courier was tired. " * 60) for i in range(1, 5)) + "\n\n*** END OF THE PROJECT GUTENBERG EBOOK"
    g = GutenbergSource(_Fetcher({"gutendex.com": json.dumps(payload), "7.txt": body}))
    scout2 = Scout(cfg, db, sources={"gutenberg": g})
    keys = scout2.import_gutenberg(None, "old book")
    assert keys == ["gutenberg:7"] and db.chapter_count("gutenberg:7") == 4
    n = db.get_novel("gutenberg:7")
    assert n.genres == ["Adventure stories"] and n.views == 42 and "public domain" in n.tags


def test_build_sources_respects_config_and_offline(cfg, monkeypatch):
    monkeypatch.delenv("GAUGE_OFFLINE", raising=False)
    cfg.research.sources = ["wikipedia", "climate", "searx"]
    srcs = build_sources(cfg)
    assert set(srcs) == {"wikipedia", "climate"}  # searx needs a URL
    assert srcs["wikipedia"].base == "https://api.wikimedia.org/core/v1/wikipedia/en"
    assert srcs["wikipedia"].fetcher.cfg.respect_robots_txt is True
    cfg.research.searx_url = "http://localhost:8080"
    assert "searx" in build_sources(cfg)
    monkeypatch.setenv("GAUGE_OFFLINE", "1")
    assert build_sources(cfg) == {}


# ------------------------------------------------------------------ agent + pipeline
def test_scout_agent_briefs_files_and_fact_checks(cfg, db, tmp_path):
    fake = FakeSource("wikipedia")
    climate = FakeSource("climate", {"lisbon": [Snippet("climate", "Climate: Lisbon", "https://open-meteo.com/", "highs 18 C, 9 of 28 days with rain", "open data")]})
    scout = Scout(cfg, db, sources={"wikipedia": fake, "climate": climate, "wikivoyage": FakeSource("wikivoyage")})
    proj = Project.create(tmp_path / "pp", "P", "idea", ["Fantasy"], 300)
    agent = ScoutAgent(FakeLLM(), proj.dir, {"max_queries": 3, "fact_check": True}, scout=scout)
    from gauge.write.agents import SceneContext

    ctx = SceneContext(project=proj, number=2, plan=ChapterPlan(number=2, title="The Market", goal="Rin crosses the night market in the rain"),
                       cast_text="- Rin", world_text="Lowmarket, rain", genres=["Fantasy"])
    brief = agent.brief(ctx)
    assert "real-world detail found" in brief and "[climate: Climate: Lisbon]" in brief and "[wikipedia: About medieval courier guild]" in brief
    assert "9 of 28 days with rain" in brief
    files = list((proj.dir / "research" / "scout").glob("*.md"))
    assert len(files) == 1 and files[0].name.startswith("chapter-002")
    assert agent.state.researched[-1]["hits"] == 3
    rev = agent.review("Couriers in medieval guilds were fined for late deliveries.", ctx)
    assert rev.score == 1.0 and rev.data["claims"] == 1 and agent.state.claims_checked == 1
    contradicting = FakeLLM(canned=[json.dumps({"claims": [{"claim": "The moon is made of cheese", "query": "moon composition"}]}),
                                    json.dumps({"verdict": "contradicted", "note": "the moon is rock"})])
    agent2 = ScoutAgent(contradicting, proj.dir, {"fact_check": True}, scout=scout)
    rev2 = agent2.review("The moon is made of cheese.", ctx)
    assert rev2.score == 0.7 and "looks wrong" in rev2.issues[0]
    assert ScoutAgent(FakeLLM(), proj.dir, {}, scout=None).brief(ctx) == ""
    assert "claims checked" in agent.report()
    topics = extract_topics("# Ten Seconds - Series Bible\n\n**World:** Lowmarket, a rain-soaked megacity run by the Courier Guild.", "")
    assert "Lowmarket" in topics and "Courier Guild" in topics and "Series" not in topics


def test_pipeline_scout_in_the_room_and_dossier(cfg, db, chapter_text, tmp_path):
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    main, fast = FakeLLM(), FakeLLM()
    pipe = WritePipeline(cfg, db, llm=main, trainer=t, fast_llm=fast)
    wiki = FakeSource("wikipedia")
    pipe.use_research_sources({"wikipedia": wiki, "climate": FakeSource("climate"), "wikivoyage": FakeSource("wikivoyage")})
    p = pipe.new_project("Ten Seconds", "a courier who can rewind ten seconds", ["Sci-fi"], words=300)
    dossier = p.dir / "research" / "scout" / "project-dossier.md"
    assert dossier.exists() and wiki.queries  # proper nouns from the bible were looked up at creation
    assert "scout" in pipe.council(p).names()
    res = pipe.write_next(p)
    draft_prompt = next(c["prompt"] for c in main.calls_list if c["purpose"] == "draft")
    assert "### Scout" in draft_prompt and "real-world detail found" in draft_prompt
    assert any(r.agent == "scout" for r in res.agent_reviews)
    # the Librarian now serves what the Scout filed
    lib = pipe.council(p).get("librarian")
    assert lib is not None and any("project-dossier" in src or "chapter-001" in src for src, _ in lib.chunks)
    assert db.research_history("Ten Seconds")
    # research disabled -> no scout in the room, no dossier
    cfg.research.enabled = False
    pipe2 = WritePipeline(cfg, db, llm=FakeLLM(), trainer=t, fast_llm=FakeLLM())
    p2 = pipe2.new_project("Quiet", "a quiet story", ["Drama"], words=200)
    assert "scout" not in pipe2.council(p2).names() and not (p2.dir / "research" / "scout").exists()


def test_cli_scout(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "200"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    assert os.environ.get("GAUGE_OFFLINE") == "1"
    r = runner.invoke(main, ["--data-dir", data, "scout", "ask", "medieval courier"])
    assert r.exit_code == 0 and "no research sources" in r.output
    r = runner.invoke(main, ["--data-dir", data, "scout", "log"])
    assert r.exit_code == 0 and "medieval courier" in r.output  # even a fruitless ad-hoc lookup is logged
    # with a fake source wired in through monkeypatching the builder, the commands work end to end
    import gauge.research.scout as sc

    monkeypatch.setattr(sc, "build_sources", lambda cfg, fetcher=None: {"wikipedia": FakeSource("wikipedia")})
    r = runner.invoke(main, ["--data-dir", data, "scout", "ask", "medieval courier guild"])
    assert r.exit_code == 0 and "About medieval courier guild" in r.output
    r = runner.invoke(main, ["--data-dir", data, "new", "Ten Seconds", "--idea", "a courier", "--genre", "Sci-fi", "--words", "200"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--data-dir", data, "scout", "ask", "night market", "--project", "Ten Seconds"])
    assert r.exit_code == 0 and "filed" in r.output
    r = runner.invoke(main, ["--data-dir", data, "scout", "dossier", "Ten Seconds", "--topic", "sealed bags"])
    assert r.exit_code == 0 and "dossier" in r.output
    r = runner.invoke(main, ["--data-dir", data, "scout", "log", "--project", "Ten Seconds"])
    assert r.exit_code == 0 and "sealed bags" in r.output
    r = runner.invoke(main, ["--data-dir", data, "scout", "gutenberg", "dracula"])
    assert r.exit_code != 0 and "disabled" in r.output
    r = runner.invoke(main, ["--data-dir", data, "research", "Ten Seconds", "courier"])
    assert r.exit_code == 0 and "sealed bags" in r.output or "couriers" in r.output
