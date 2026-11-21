"""Batch E: reader-feedback import, the evaluation harness, and the Arc Keeper agent."""
from pathlib import Path

from click.testing import CliRunner

from gauge.cli import main
from gauge.learn.evaluation import evaluate_project, pearson, realized_retention
from gauge.llm import FakeLLM
from gauge.write.agents.arc_keeper import ArcKeeper, beat_overlap
from gauge.write.agents.base import SceneContext


def test_realized_retention_and_pearson():
    fb = {1: {"views": 100}, 2: {"views": 80}, 3: {"views": 120}, 4: {"views": 60}}
    ret = realized_retention(fb)
    assert ret[2] == 0.8 and round(ret[3], 2) == 1.5 and ret[4] == 0.5
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0
    assert pearson([1, 2], [1, 2]) is None  # too few points


class _Proj:
    def __init__(self, scores):
        self.scores = scores

    def read_meta(self, n):
        return {"score": self.scores[n]} if n in self.scores else {}


def test_evaluate_project_correlation():
    # Retention tracks score: high-score chapters retain better.
    fb = {1: {"views": 100}, 2: {"views": 120}, 3: {"views": 90}, 4: {"views": 140}, 5: {"views": 70}}
    scores = {2: 0.9, 3: 0.5, 4: 0.95, 5: 0.3}
    ev = evaluate_project(_Proj(scores), fb)
    assert ev.chapters_with_feedback == 4
    assert ev.score_vs_retention is not None and ev.score_vs_retention > 0.3
    assert "tracks real retention" in ev.verdict()
    assert ev.worst_chapters and ev.worst_chapters[0][0] == 5  # chapter 5 had the worst retention


def test_evaluate_handles_no_data():
    ev = evaluate_project(_Proj({}), {})
    assert ev.chapters_with_feedback == 0 and ev.score_vs_retention is None
    assert "not enough paired data" in ev.verdict()


def _ctx(tmp_path, arc, number=1):
    bible = type("B", (), {"arc": arc})()
    state = type("S", (), {"bible": bible})()
    project = type("P", (), {"dir": Path(tmp_path), "state": state})()
    return SceneContext(project=project, number=number, plan=type("Pl", (), {"title": "T"})())


def test_arc_keeper_tracks_progress_and_flags_drift(tmp_path):
    arc = ["Rin discovers she can rewind ten seconds",
           "the Broker notices her power and moves against her",
           "Jun vanishes and every rewind costs a memory of her"]
    ag = ArcKeeper(FakeLLM(), tmp_path)
    ctx = _ctx(tmp_path, arc, 1)
    # A chapter that delivers beat 0 advances the arc.
    rev = ag.review("Rin discovers she can rewind ten seconds in the rain, erasing a memory each time", ctx)
    assert rev is not None and rev.data["beat_overlap"] > 0.3
    ag.remember("Rin discovers she can rewind ten seconds, erasing a memory", ctx)
    assert ag.state.reached == 1
    # Several chapters that ignore the arc should raise a stall complaint.
    drift = "They ate noodles and talked about the weather and nothing else happened at all today"
    for n in range(2, 8):
        ag.review(drift, _ctx(tmp_path, arc, n))
        ag.remember(drift, _ctx(tmp_path, arc, n))
    final = ag.review(drift, _ctx(tmp_path, arc, 8))
    assert final.score < 1.0 and any("stall" in i for i in final.issues)
    assert "due is:" in ag.plan_notes(ctx.project)
    # No arc declared -> the agent is silent.
    assert ag.review("anything", _ctx(tmp_path, [], 1)) is None


def test_beat_overlap():
    assert beat_overlap("the courier discovers her rewind power", "courier discovers rewind power") > 0.5
    assert beat_overlap("unrelated text about cooking dinner", "courier discovers rewind power") == 0.0


def test_cli_feedback_import_and_evaluate(tmp_path):
    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "200"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    # Create a project so feedback has somewhere to land.
    from gauge.config import GaugeConfig
    from gauge.write.project import Project
    cfg = GaugeConfig.load(data)
    Project.create(cfg.projects_dir, "Ten Seconds", "a courier who can rewind", ["Sci-fi"], 300)
    csv = tmp_path / "fb.csv"
    csv.write_text("chapter,views,comments,likes\n1,100,5,10\n2,90,4,8\n3,120,9,15\n", encoding="utf-8")
    r = runner.invoke(main, ["--data-dir", data, "feedback-import", "Ten Seconds", str(csv)])
    assert r.exit_code == 0 and "imported 3" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "evaluate", "Ten Seconds"])
    assert r.exit_code == 0 and "retention data" in r.output, r.output
