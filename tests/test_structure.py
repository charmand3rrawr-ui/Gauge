"""Roadmap section D: structure checks (try-fail, conflict, cliff variety, beats) and the agent."""
from pathlib import Path

from gauge.llm import FakeLLM
from gauge.write.agents import StructureEditor
from gauge.write.agents.base import SceneContext
from gauge.write.structure import (
    cliff_type,
    conflict_level,
    expected_beat,
    report,
    tension_shape,
    try_fail,
)

EASY = ("She won easily and saved them all. She found the key and escaped and triumphed over the guards. ") * 20
TENSE = ("The blade fell and she screamed and ran as the fire roared and the men chased her through the smoke. ") * 20


def test_individual_structure_checks():
    tf = try_fail(EASY)
    assert tf["too_easy"] == 1.0 and tf["setback"] < 0.5
    assert conflict_level(TENSE) > conflict_level(EASY)
    assert cliff_type("And then the lights went out. Was she alone?") == "question"
    assert cliff_type("The signature was her own mother's. The truth was finally out.") == "revelation"
    assert expected_beat(1, 20) == "setup" and expected_beat(10, 20) == "midpoint turn" and expected_beat(20, 20) == "resolution"
    assert expected_beat(1, 1) == "standalone"
    assert tension_shape(TENSE) in {"rising", "falling", "flat", "spiky"}


def test_report_flags():
    r = report(EASY, number=10, total=20)
    joined = " ".join(r.issues)
    assert "little conflict" in joined and "no visible cost" in joined
    # Same cliff type three running is flagged.
    r2 = report(TENSE + " Who is there?", number=10, total=20, recent_cliffs=["question", "question"])
    assert any("same cliffhanger type" in i for i in r2.issues)


def test_structure_editor_agent(tmp_path):
    class _P:
        dir = Path(tmp_path)
        class state:  # noqa: N801
            outline = []
    ctx = SceneContext(project=_P(), number=10, plan=type("Pl", (), {"title": "T"})())
    ag = StructureEditor(FakeLLM(), tmp_path)
    rev = ag.review(EASY, ctx)
    assert rev is not None and rev.agent == "structure" and rev.issues
    ag.remember(EASY, ctx)
    assert ag.state.recent_cliffs  # the ending type was recorded
    assert ag.review("too short to judge the structure of", ctx) is None


def test_cliff_variety_across_chapters(tmp_path):
    class _P:
        dir = Path(tmp_path)
        class state:  # noqa: N801
            outline = []
    ag = StructureEditor(FakeLLM(), tmp_path)
    q = TENSE + " Who was there?"
    for n in range(1, 3):
        ctx = SceneContext(project=_P(), number=n, plan=type("Pl", (), {"title": "T"})())
        ag.review(q, ctx)
        ag.remember(q, ctx)
    rev3 = ag.review(q, SceneContext(project=_P(), number=3, plan=type("Pl", (), {"title": "T"})()))
    assert any("same cliffhanger type" in i for i in rev3.issues)


def test_structure_in_full_council(cfg):
    from gauge.write.agents import build_council
    from gauge.write.project import Project
    p = Project.create(cfg.projects_dir, "P", "idea", ["Sci-fi"], 300)
    assert "structure" in build_council(cfg, FakeLLM(), p).names()
