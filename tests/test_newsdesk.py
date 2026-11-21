"""Batch N: the multilingual newsdesk — trends, translation, original sourced commentary, review-gated schedule."""
import json
from datetime import date, datetime, timedelta

from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.llm import FakeLLM
from gauge.market.policy import REFUSED, check_action
from gauge.newsdesk.commentary import originality, write_commentary
from gauge.newsdesk.desk import publish_item, queue_for, run_cycle
from gauge.newsdesk.queue import PublishQueue
from gauge.newsdesk.translate import detect_script, translate
from gauge.newsdesk.trends import SourceItem, Topic, discover, rss_items, wikimedia_top

RSS = ("<rss><channel>"
       "<item><title>Bank of Japan raises rates</title><description>The central bank moved for the first time in years."
       "</description><link>https://news.example/boj</link></item>"
       "<item><title>Solar eclipse crosses three continents</title><description>Millions watched the sky darken."
       "</description><link>https://news.example/eclipse</link></item>"
       "</channel></rss>")
ATOM = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>日本銀行が利上げ</title>'
        '<summary>中央銀行が数年ぶりに動いた。</summary><link href="https://news.example/ja/boj"/></entry></feed>')


class FakeFetcher:
    def __init__(self):
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        if "pageviews/top/en" in url:
            return json.dumps({"items": [{"articles": [{"article": "Main_Page", "views": 9_000_000},
                                                       {"article": "Bank_of_Japan", "views": 500_000},
                                                       {"article": "Solar_eclipse", "views": 300_000}]}]})
        if "pageviews/top/ja" in url:
            return json.dumps({"items": [{"articles": [{"article": "Bank_of_Japan", "views": 400_000}]}]})
        if url.endswith("/rss"):
            return RSS
        if url.endswith("/atom"):
            return ATOM
        return None


def test_policy_refuses_rewriting_and_unreviewed_news():
    for bad in ("rewrite_republish", "auto_publish_unreviewed_news"):
        assert check_action(bad)[0] is False and bad in REFUSED
    for good in ("trend_discovery", "translate", "original_commentary"):
        assert check_action(good)[0] is True


def test_wikimedia_top_filters_non_topics():
    titles = [t.title for t in wikimedia_top(FakeFetcher(), "en", date(2026, 11, 20))]
    assert titles == ["Bank of Japan", "Solar eclipse"]  # Main_Page filtered out


def test_rss_and_atom_parse_headlines_only():
    f = FakeFetcher()
    rss = rss_items(f, "https://news.example/rss")
    assert [i.title for i in rss] == ["Bank of Japan raises rates", "Solar eclipse crosses three continents"]
    atom = rss_items(f, "https://news.example/atom", lang="ja")
    assert atom[0].title == "日本銀行が利上げ" and atom[0].url == "https://news.example/ja/boj"
    # Only feed URLs are fetched — never the linked article bodies.
    assert not any(u.startswith("https://news.example/boj") for u in f.urls)


def test_discover_ranks_multilingual_topics_first():
    topics = discover(FakeFetcher(), ["en", "ja"], ["https://news.example/rss"], date(2026, 11, 20))
    assert topics[0].title == "Bank of Japan" and set(topics[0].langs) == {"en", "ja"}


def test_translation_and_script_detection():
    assert detect_script("日本銀行") == "cjk" and detect_script("Привет") == "cyrillic" and detect_script("hello") == "latin"
    out = translate(FakeLLM(), "日本銀行が利上げ", "English", "ja")
    assert out and "日本銀行" in out  # the fake echoes; a real model returns English
    assert translate(FakeLLM(), "", "English") == ""


def test_commentary_is_original_sourced_and_labelled():
    src = [SourceItem("Bank of Japan raises rates", "The central bank moved for the first time in years.",
                      "https://news.example/boj")]
    c = write_commentary(FakeLLM(), "Bank of Japan", src, "Jane Doe")
    assert "Opinion:" in c.body and "By Jane Doe" in c.body
    assert "Sources:" in c.body and "https://news.example/boj" in c.body  # credits and links its source
    assert c.originality > 0.9 and not c.originality_flags


def test_originality_catches_copying():
    src = [SourceItem("Bank of Japan raises rates",
                      "The central bank moved for the first time in years, surprising markets that had priced in no change.",
                      "https://news.example/boj")]
    copied = ("The central bank moved for the first time in years, surprising markets that had priced in no change. "
              "That is the news.")
    score, worst = originality(copied, src)
    assert score < 0.75 and worst == "Bank of Japan raises rates"
    assert originality("A completely different argument about fiscal policy and compassion.", src)[0] == 1.0


def test_commentary_refuses_impersonating_byline():
    try:
        write_commentary(FakeLLM(), "anything", [], "Tyler Durden")
        raise AssertionError("should refuse")
    except PermissionError:
        pass


class _C:  # minimal commentary stand-in for queue tests
    def __init__(self, topic, clean=True, claims=None):
        self.topic, self.title, self.body, self.byline = topic, f"On {topic}", "body", "Jane Doe"
        self.claims_to_verify = claims or []
        self.balance_flags, self.headline_flags, self.originality_flags = [], [], []
        self.sources, self.originality, self.clean = [], 1.0, clean and not claims


def test_queue_review_gating_and_interval(tmp_path):
    q = PublishQueue(tmp_path / "q.json")
    risky = q.add(_C("rates", claims=["a statistic: 3%"]))
    clean = q.add(_C("eclipse"))
    now = datetime(2026, 11, 21, 12, 0)
    # Without auto-publish nothing goes out until a human approves.
    assert q.tick(now, 30, auto_publish=False, max_per_day=24) is None
    # With auto-publish only the CLEAN piece qualifies; the one with a factual claim still waits.
    assert q.tick(now, 30, auto_publish=True, max_per_day=24).id == clean.id
    q.mark_published(clean.id, now)
    # Interval: not due again for 30 minutes, even with an approved item.
    q.approve(risky.id)
    assert q.tick(now + timedelta(minutes=10), 30, False, 24) is None
    assert q.tick(now + timedelta(minutes=31), 30, False, 24).id == risky.id
    assert q.has_topic("rates") and not q.has_topic("unknown")


def test_queue_daily_cap(tmp_path):
    q = PublishQueue(tmp_path / "q.json")
    now = datetime(2026, 11, 21, 12, 0)
    for i in range(3):
        it = q.add(_C(f"t{i}"))
        q.mark_published(it.id, now - timedelta(hours=i))
    q.approve(q.add(_C("next")).id)
    assert q.tick(now + timedelta(hours=2), 30, False, max_per_day=3) is None  # cap reached in the last 24h


def test_run_cycle_end_to_end(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.llm.provider = "fake"
    cfg.ensure_dirs()
    from gauge.write.project import Project
    Project.create(cfg.projects_dir, "Desk", "a commentary site", ["Opinion"], 300)
    cfg.newsdesk.byline = "Jane Doe"
    cfg.newsdesk.project = "Desk"
    cfg.newsdesk.feeds = ["https://news.example/rss"]
    cfg.newsdesk.auto_publish = True
    topics = [Topic("Bank of Japan", "en", 1.0, "wikipedia", "https://en.wikipedia.org/wiki/Bank_of_Japan", ["en", "ja"])]
    rep = run_cycle(cfg, FakeLLM(), fetcher=FakeFetcher(), topics=topics, now=datetime(2026, 11, 21, 12, 0))
    assert rep.drafted and rep.published  # a clean original piece was drafted and published into the project
    arts = list((cfg.projects_dir / "desk" / "articles").glob("*.json"))
    assert arts and json.loads(arts[0].read_text())["kind"] == "opinion"
    # Second cycle: the topic is already queued, and the interval has not elapsed.
    rep2 = run_cycle(cfg, FakeLLM(), fetcher=FakeFetcher(), topics=topics, now=datetime(2026, 11, 21, 12, 5))
    assert "Bank of Japan" in rep2.skipped and rep2.published is None


def test_run_cycle_requires_byline(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.llm.provider = "fake"
    cfg.ensure_dirs()
    rep = run_cycle(cfg, FakeLLM(), fetcher=FakeFetcher(), topics=[Topic("X", "en", 1.0, "rss")])
    assert not rep.drafted and any("byline" in n for n in rep.notes)


def test_publish_item_writes_labelled_article(tmp_path):
    q = PublishQueue(tmp_path / "q.json")
    it = q.add(_C("eclipse"))
    p = publish_item(it, tmp_path / "articles")
    d = json.loads(p.read_text())
    assert d["kind"] == "opinion" and d["byline"] == "Jane Doe"


def test_cli_newsdesk_queue_flow(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.llm.provider = "fake"
    cfg.newsdesk.byline = "Jane Doe"
    cfg.save()
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", data, "newsdesk", "draft", "Bank of Japan"])
    assert r.exit_code == 0 and "queued as" in r.output, r.output
    item_id = r.output.split("queued as ")[1].split(",")[0]
    r = runner.invoke(main, ["--data-dir", data, "newsdesk", "queue"])
    assert item_id in r.output and "pending_review" in r.output
    r = runner.invoke(main, ["--data-dir", data, "newsdesk", "show", item_id])
    assert r.exit_code == 0 and "Opinion:" in r.output
    r = runner.invoke(main, ["--data-dir", data, "newsdesk", "approve", item_id])
    assert "approved" in r.output
    assert queue_for(GaugeConfig.load(data)).get(item_id).status == "approved"
    r = runner.invoke(main, ["--data-dir", data, "newsdesk", "draft", "x", "--byline", "ZeroHedge"])
    assert r.exit_code == 1 and "refused" in r.output
