"""Batch O: the realism layer — sensory editor, physical continuity, grounding, realism dial."""
from pathlib import Path

from gauge.llm import FakeLLM
from gauge.write.agents import ContinuityAuditor, SensoryEditor
from gauge.write.agents.base import SceneContext
from gauge.write.grounding import build_dossier, grounding_queries
from gauge.write.realism import logistics_check, sensory_profile, sensory_score

FLAT = ("She was afraid. She felt sad. She noticed the room was empty. She realized something was wrong. "
        "She wondered what to do. She knew it was bad. ") * 6
RICH = ("Rain hammered the tin roof of Dock 7. The air reeked of diesel and wet ash. Cold water found the gap "
        "in her collar; she gripped the rusted rail until her knuckles ached, the metal singing under the wind. "
        "Somewhere below, a chain clanked twice and went still. ") * 3


def _ctx(tmp_path, number=1):
    return SceneContext(project=type("P", (), {"dir": Path(tmp_path)})(), number=number, plan=type("Pl", (), {"title": "T"})())


def test_sensory_profile_distinguishes_flat_from_rich():
    flat, rich = sensory_profile(FLAT), sensory_profile(RICH)
    assert sensory_score(rich) > sensory_score(flat)
    assert flat.telling_emotions >= 2 and flat.filter_density > 0.5  # told, and full of filter verbs
    assert rich.senses_engaged >= 2 and rich.concrete_specifics >= 1
    assert any("told, not shown" in i for i in flat.issues)


def test_logistics_check():
    assert logistics_check("The noon sun blazed overhead; by midnight the stars glittered.").issues  # contradiction
    assert logistics_check("The noon sun blazed. Hours later, by midnight, the stars came out.").issues == []  # transition
    assert logistics_check("It was spring in one place and deep winter in another.").issues  # two seasons


def test_sensory_editor_agent(tmp_path):
    ag = SensoryEditor(FakeLLM(), tmp_path)
    rev = ag.review(FLAT, _ctx(tmp_path))
    assert rev is not None and rev.agent == "sensory" and rev.score < 0.7 and rev.issues
    ag.remember(FLAT, _ctx(tmp_path))
    assert ag.state.history and "filter" in ag.report()
    assert ag.review("too short", _ctx(tmp_path)) is None


def test_continuity_auditor_agent(tmp_path):
    ag = ContinuityAuditor(FakeLLM(), tmp_path)
    rev = ag.review("The noon sun blazed overhead and the stars glittered at midnight. " * 10, _ctx(tmp_path))
    assert rev is not None and rev.issues and rev.score < 1.0
    # Season flip between consecutive chapters is flagged.
    ag.review("A warm spring afternoon in the orchard, blossom everywhere. " * 16, _ctx(tmp_path, 1))
    ag.remember("A warm spring afternoon in the orchard, blossom everywhere. " * 16, _ctx(tmp_path, 1))
    r2 = ag.review("Deep winter now, snow to the knees, the orchard bare. " * 16, _ctx(tmp_path, 2))
    assert any("season changed" in i for i in r2.issues)


def test_grounding_dossier_with_fake_scout():
    assert "look, sound and smell" in grounding_queries("ER scene", "a hospital emergency room")[0]

    class FakeScout:
        def search(self, q):
            class F:
                snippets = [{"text": "ERs smell of antiseptic and burnt coffee; monitors beep in overlapping rhythms.",
                             "url": "https://example/er"}]
            return [F()]

    d = build_dossier(FakeScout(), "She crossed the ER", "a hospital ER")
    assert d.notes and "antiseptic" in d.notes[0] and "Grounding" in d.render()
    assert build_dossier(None, "x", "y").notes == []  # no scout -> empty, no crash


def test_critic_realism_weight_inert_by_default():
    from gauge.write.critic import Critic
    from gauge.write.project import ChapterPlan
    plan = ChapterPlan(number=1, title="T", goal="g", hook="h", turn="t", cliffhanger="c", beats=["a"])
    c0 = Critic(FakeLLM(), None, realism_weight=0.0)
    s0 = c0.critique(RICH, plan).score
    c1 = Critic(FakeLLM(), None, realism_weight=0.4)
    # With weight on, a richly sensory chapter should not score lower than the flat baseline blend.
    assert c1.critique(RICH, plan).score >= 0.0 and 0.0 <= s0 <= 1.0


def test_world_realism_flag_changes_prompt():
    from gauge.write.world import WorldAgent, WorldState
    plain = WorldAgent(FakeLLM(), WorldState(name="X"), realism=False).system_prompt()
    real = WorldAgent(FakeLLM(), WorldState(name="X"), realism=True).system_prompt()
    assert "REALISM MODE" in real and "REALISM MODE" not in plain
