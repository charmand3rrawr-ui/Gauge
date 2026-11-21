"""Batch A: readability, retention curve, suitability contract, A/B variants."""
from pathlib import Path

from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.learn.retention import serial_retention, shape
from gauge.llm import FakeLLM
from gauge.write.agents import ReadabilityCoach, SuitabilityChecker
from gauge.write.agents.base import SceneContext
from gauge.write.agents.readability import measure, score_and_issues
from gauge.write.agents.suitability import heat_level
from gauge.write.variants import VariantLog, best_hook, best_title, rank_hooks, rank_titles

SMOOTH = ("\"Run,\" she said.\n\nI ran. The door slammed. Somewhere a dog barked once and went quiet.\n\n"
          "The alley was narrow and wet. I could hear the footsteps behind me, patient, unhurried, and "
          "that was the part that frightened me most of all, because patient meant certain.\n\n"
          "I turned left. Then right. A dead end.\n\n\"Okay,\" I whispered. \"Okay. One more time.\"\n\n"
          "The footsteps stopped.\n\n") * 3
WALL = ("The committee convened at the appointed hour notwithstanding considerable administrative "
        "difficulties that had accumulated over the preceding fortnight regarding the apportionment of "
        "responsibilities among the various subcommittees whose overlapping jurisdictions had occasioned "
        "no small measure of consternation. " * 12) * 3


class _P:
    def __init__(self, d):
        self.dir = Path(d)


def _ctx(tmp_path, number=1):
    return SceneContext(project=_P(tmp_path), number=number, plan=type("Pl", (), {"title": "T"})())


def test_readability_measure_and_bands(tmp_path):
    m = measure(SMOOTH)
    assert 0.0 <= m["burstiness"] and m["flesch"] > 50
    wall = measure(WALL)
    assert wall["wall_paragraph_ratio"] > 0 and wall["syllables_per_word"] > m["syllables_per_word"]
    score_smooth, _ = score_and_issues(m)
    score_wall, issues_wall = score_and_issues(wall)
    assert score_smooth > score_wall and issues_wall  # the wall of text earns complaints


def test_readability_agent(tmp_path):
    ag = ReadabilityCoach(FakeLLM(), tmp_path)
    rev = ag.review(WALL, _ctx(tmp_path))
    assert rev is not None and rev.agent == "readability" and rev.score < 1.0 and rev.issues
    ag.remember(WALL, _ctx(tmp_path))
    assert ag.state.history and "walls" in ag.report()
    assert ag.review("too short", _ctx(tmp_path)) is None


def test_suitability_contract(tmp_path):
    opts = {"age_band": "teen", "heat_max": 0, "content_lines": ["forbidden phrase"], "word_target": 0, "block": True}
    ag = SuitabilityChecker(FakeLLM(), tmp_path, opts)
    clean = ag.review(SMOOTH, _ctx(tmp_path))
    assert clean.score == 1.0 and not clean.issues
    dirty = ag.review(SMOOTH + "\n\nThey shared a long kiss. This contains a forbidden phrase.", _ctx(tmp_path))
    assert dirty.score == 0.0  # crossed line + over-heat, and block=True
    assert any("content line" in i for i in dirty.issues)
    assert heat_level("they kissed under the rain") >= 1


def test_retention_shape_and_serial():
    hooky = "\"Don't,\" she said. " + "The knife flashed and someone screamed and the lights died. " * 8 + "Then a voice said her name."
    flat = "It was a calm day. " * 40
    assert shape(hooky).score > shape(flat).score
    sr = serial_retention([10, 5, -2, 8, 3])
    assert sr["chapters"] == 5.0 and 0 < sr["growth_fraction"] <= 1.0


def test_variants_ranking_and_log(tmp_path):
    llm = FakeLLM()
    chosen, ranked = best_hook(llm, "She runs from the footsteps", n=3)
    assert chosen and len(ranked) >= 1 and ranked[0].score >= ranked[-1].score
    title, tranked = best_title(llm, "She runs", "opening text", n=4)
    assert title and tranked
    # title ranking prefers short punchy titles
    assert rank_titles(["Ten Seconds", "A Very Long And Overly Descriptive Chapter Title Indeed"])[0].text == "Ten Seconds"
    assert rank_hooks(["", "She ran."])[0].text == "She ran."
    log = VariantLog(tmp_path)
    log.record("title", 1, title, tranked[1:], tranked[0].score)
    assert log.rows() and log.rows()[0]["kind"] == "title"
    assert log.hit_rate() is None  # no outcomes recorded yet


def test_critic_retention_weight_is_inert_by_default(tmp_path):
    from gauge.write.critic import Critic
    from gauge.write.project import ChapterPlan
    plan = ChapterPlan(number=1, title="T", goal="g", hook="h", turn="t", cliffhanger="c", beats=["a"])
    c0 = Critic(FakeLLM(), None, retention_weight=0.0)
    cr0 = c0.critique(SMOOTH, plan)
    assert cr0.retention == 0.0
    c1 = Critic(FakeLLM(), None, retention_weight=0.3)
    cr1 = c1.critique(SMOOTH, plan)
    assert cr1.retention > 0.0


def test_cli_score_shows_curves(tmp_path):
    data = str(tmp_path / "data")
    GaugeConfig(data_dir=data).save()
    f = tmp_path / "ch.txt"
    f.write_text(SMOOTH, encoding="utf-8")
    r = CliRunner(env={"COLUMNS": "200"}).invoke(main, ["--data-dir", data, "score", str(f)])
    assert r.exit_code == 0 and "retention curve" in r.output and "readability" in r.output, r.output
