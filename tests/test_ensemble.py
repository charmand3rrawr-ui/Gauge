"""Roadmap section B: ensemble checks and the Cast Tracker agent."""
from pathlib import Path

from gauge.llm import FakeLLM
from gauge.write.agents import CastTracker
from gauge.write.agents.base import SceneContext
from gauge.write.characters import Character, PersonalityMatrix, Roster, Voice
from gauge.write.ensemble import (
    appeared,
    missing_wants_needs,
    report,
    static_relationships,
    unused_voice,
    want_need_same,
)


def _roster(tmp_path):
    r = Roster(Path(tmp_path))
    rin = Character(name="Rin", role="protagonist",
                    matrix=PersonalityMatrix(wants="clear the debt", needs="let someone help"),
                    voice=Voice(catchphrases=["One more time."]))
    jun = Character(name="Jun", role="supporting", matrix=PersonalityMatrix())  # no want/need
    broker = Character(name="The Broker", role="antagonist",
                       matrix=PersonalityMatrix(wants="own her", needs="own her"))  # want == need
    r.add(rin)
    r.add(jun)
    r.add(broker)
    return r


TEXT = "Rin ran through the rain. Rin counted the exits. Rin thought of Jun again. " * 4


def test_individual_ensemble_checks(tmp_path):
    r = _roster(tmp_path)
    assert set(appeared(TEXT, r)) >= {"Rin", "Jun"}
    assert missing_wants_needs(r) == ["Jun"]
    assert "The Broker" in want_need_same(r)
    assert unused_voice(TEXT, r) == ["Rin"]  # Rin appears but her catchphrase is absent
    # The Broker/Jun etc have no relationship history -> flagged static.
    assert static_relationships(r)


def test_report_bundles_issues(tmp_path):
    r = _roster(tmp_path)
    rep = report(TEXT, r, appearance_gap={"The Broker": 7}, gap_limit=5)
    joined = " ".join(rep.issues)
    assert "absent" in joined and "no want or need" in joined and "want and need are identical" in joined


def test_cast_tracker_agent(tmp_path):
    _roster(tmp_path)  # writes character sheets under tmp_path/characters via Roster? no — need to save
    r = _roster(tmp_path)
    r.save()

    class _P:
        dir = Path(tmp_path)
    ctx1 = SceneContext(project=_P(), number=1, plan=type("Pl", (), {"title": "T"})())
    ag = CastTracker(FakeLLM(), tmp_path)
    rev = ag.review(TEXT, ctx1)
    assert rev is not None and rev.agent == "cast_tracker"
    ag.remember(TEXT, ctx1)
    assert ag.state.last_seen.get("Rin") == 1
    # Several chapters later, Rin keeps appearing but the Broker never does -> absence flagged.
    for n in range(2, 9):
        ag.remember(TEXT, SceneContext(project=_P(), number=n, plan=ctx1.plan))
    rev8 = ag.review(TEXT, SceneContext(project=_P(), number=9, plan=ctx1.plan))
    assert any("absent" in i for i in rev8.issues)


def test_cast_tracker_in_full_council(cfg):
    from gauge.write.agents import build_council
    from gauge.write.project import Project
    p = Project.create(cfg.projects_dir, "P", "idea", ["Sci-fi"], 300)
    assert "cast_tracker" in build_council(cfg, FakeLLM(), p).names()
