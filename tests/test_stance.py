"""Batch M: the declared editorial stance, the balance check, and transparent publication."""

from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.llm import FakeLLM
from gauge.site.opinion import draft_opinion
from gauge.site.stance import (
    EditorialStance,
    balance_check,
    default_stance,
    editorial_page_body,
    load_stance,
    save_stance,
    stance_guidance,
)


def test_default_stance_is_centrist_and_valued():
    s = default_stance()
    joined = " ".join(s.principles).lower()
    assert "fiscal" in joined and "compassion" in joined and "evidence" in joined
    g = stance_guidance(s, comedic=True).lower()
    assert "lean" in g and "steelman" in g and "comedic" in g
    # Honest framing: the stance is a point of view, not neutrality.
    assert "not a claim of neutrality" in editorial_page_body(s)


def test_balance_check_flags_one_sided():
    one_sided = "Cut taxes now. " * 40
    assert balance_check(one_sided), "a long one-sided piece should be flagged"
    balanced = ("Cut taxes now. " * 20 +
                "To be fair, critics argue this starves services; that is the strongest case against, and here is the answer. " * 4)
    assert balance_check(balanced) == []
    assert balance_check("short piece") == []  # too short to require balance


def test_stance_shapes_opinion_and_computes_balance():
    s = default_stance()
    d = draft_opinion(FakeLLM(), "Deficits matter but so do people", "Jane Doe", voice="punchy", stance=s, comedic=True)
    assert "Opinion:" in d.body and "By Jane Doe" in d.body
    assert hasattr(d, "balance_flags")  # the balance check ran


def test_stance_roundtrip(tmp_path):
    custom = EditorialStance(name="My desk", principles=["spend wisely", "be kind"], lean_rule="call each one on merit")
    p = save_stance(tmp_path / "stance.json", custom)
    loaded = load_stance(p)
    assert loaded.name == "My desk" and loaded.principles == ["spend wisely", "be kind"]
    assert load_stance(tmp_path / "missing.json").name == default_stance().name  # falls back to default


def test_cli_stance_and_editorial_page(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.llm.provider = "fake"
    cfg.save()
    from gauge.write.project import Project
    Project.create(cfg.projects_dir, "Platform", "a courier who can rewind", ["Sci-fi"], 300)
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", data, "site", "stance"])
    assert r.exit_code == 0 and "Evidence-first centrist" in r.output and "not a claim of neutrality" in r.output
    r = runner.invoke(main, ["--data-dir", data, "site", "build", "Platform", "--byline", "Jane Doe"])
    assert r.exit_code == 0
    page = cfg.projects_dir / "platform" / "site" / "articles" / "editorial-stance.html"
    assert page.exists() and "Where we stand" in page.read_text(encoding="utf-8")
