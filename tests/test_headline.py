"""Batch L: eye-catching but honest headlines (in-context, no clickbait) and opinion voice registers."""

from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.llm import FakeLLM
from gauge.site.headline import in_context, rank_headlines, score_headline, suggest_headlines
from gauge.site.opinion import VOICES, draft_opinion

BODY = ("Serial fiction is reshaping publishing. Readers chose momentum over polish. "
        "The weekly cadence rebuilt how stories find their readers, and the trade should notice.")


def test_in_context_detects_overreach():
    good = in_context("Serial fiction is reshaping publishing", BODY)
    assert good.in_context and good.supported >= 0.6
    bad = in_context("The Fed slashed interest rates to zero overnight", BODY)
    assert not bad.in_context and {"fed", "slashed", "rates"} & set(bad.unsupported)


def test_score_excludes_clickbait_rewards_supported():
    clickbait = score_headline("You won't believe what serials did next", BODY)
    assert clickbait.score == 0.0  # disqualified
    honest = score_headline("The weekly cadence reshaping how stories find readers", BODY)
    assert honest.score > 0.0 and honest.in_context


def test_rank_and_suggest():
    cands = ["Serial fiction is reshaping publishing", "You won't believe this SHOCKING trend",
             "Why readers chose momentum over polish"]
    ranked = rank_headlines(cands, BODY)
    assert ranked[0].score > 0 and all("believe" not in h.text.lower() or h.score == 0 for h in ranked)
    suggested = suggest_headlines(FakeLLM(), BODY, n=4)
    assert suggested and all(h.score > 0 for h in suggested)
    assert all("you won't believe" not in h.text.lower() for h in suggested)  # clickbait filtered out


def test_opinion_voices_available_and_honest():
    assert set(VOICES) == {"plain", "punchy", "contrarian"}
    d = draft_opinion(FakeLLM(), "Serials are reshaping publishing", "Jane Doe", voice="contrarian")
    # Contrarian voice still produces a labelled, bylined, non-clickbait piece under the author's own name.
    assert "Opinion:" in d.body and "By Jane Doe" in d.body and d.headline_flags == []


def test_cli_site_headline(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.llm.provider = "fake"
    cfg.save()
    f = tmp_path / "article.txt"
    f.write_text(BODY, encoding="utf-8")
    r = CliRunner(env={"COLUMNS": "200"}).invoke(main, ["--data-dir", data, "site", "headline", str(f)])
    assert r.exit_code == 0 and "reshaping" in r.output and "believe" not in r.output.lower()
