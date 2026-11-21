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
