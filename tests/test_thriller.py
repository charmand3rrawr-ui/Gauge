"""Crime/thriller craft: measured suspense, fair-play clues, expert register, antagonist signature."""
from gauge.config import GaugeConfig
from gauge.llm.client import FakeLLM
from gauge.write.agents import build_council
from gauge.write.agents.base import SceneContext
from gauge.write.agents.thriller_craft import ThrillerCraft
from gauge.write.project import ChapterPlan, Project
from gauge.write.thriller import (
    clues_present,
    fair_play,
    register_profile,
    signature_terms,
    suspense_profile,
    suspense_score,
    wants_thriller,
)

TENSE = ("The house was far too quiet. Something was wrong. She felt watched climbing the dark stair, "
         "the silence pressing close around her. A body lay in the study, the throat opened with care. "
         "Only minutes left before he came back. ")
CLIFF = "Then the lights went out?"


# ------------------------------------------------------------------ pure functions
def test_wants_thriller_gate():
    assert wants_thriller(["Crime", "Thriller"]) and wants_thriller([], "a noir psychological tone")
    assert not wants_thriller(["Romance", "Comedy"])


def test_suspense_rewards_tension_and_a_hook():
    hooked = suspense_profile(TENSE * 2 + "\n\n" + CLIFF)
    assert hooked.ticking_clock and hooked.ends_on_hook and suspense_score(hooked) > 0.7
    flat = suspense_profile("They sat and talked about the weather and the garden for a long, pleasant afternoon. " * 12)
    assert flat.issues and suspense_score(flat) < suspense_score(hooked)


def test_register_rewards_specifics_and_flags_handwaving():
    good = register_profile("The medical examiner noted rigor mortis and ran a tox screen; the blood spatter told the rest.",
                            ["forensic"])
    assert good.hits["forensic"] >= 3 and good.issues == []
    vague = register_profile("The detective examined the body, did some forensics, and just figured it out.", ["forensic"])
    assert vague.vague >= 1 and any("hand-waved" in i for i in vague.issues)


def test_fair_play_tracks_clues_and_herrings():
    fp = fair_play(["the muddy boot", "the torn ticket"], ["the jealous brother"], {"the muddy boot"}, set())
    assert fp.missing_clues == ["the torn ticket"] and fp.unresolved_herrings == ["the jealous brother"]
    assert clues_present("A muddy boot print by the door.", ["the muddy boot", "the torn ticket"]) == set()
    assert clues_present("She found the muddy boot.", ["the muddy boot"]) == {"the muddy boot"}


def test_signature_terms_from_text():
    sig = signature_terms("His signature was the staging, a trophy taken, the MO always precise.")
    assert {"signature", "staging", "trophy", "mo"} <= sig
    assert signature_terms("He left a calling card.", {"calling card"}) == {"calling card"}


# ------------------------------------------------------------------ the agent
def _ctx(project, number):
    plan = ChapterPlan(number=number, title=f"Ch{number}", goal="g", hook="h", turn="t", cliffhanger="c", beats=["b"])
    return SceneContext(project=project, number=number, plan=plan, genres=["Crime", "Thriller"])


def _proj(tmp_path):
    return Project.create(tmp_path / "proj", "The Collector", "a killer who stages victims", ["Crime", "Thriller"], 300)


def test_agent_scores_suspense_and_tracks_clues(tmp_path):
    p = _proj(tmp_path)
    ag = ThrillerCraft(FakeLLM(), p.dir, {"clues": ["the muddy boot", "the torn ticket"], "reveal_chapter": 3})
    ch1 = TENSE * 3 + "\n\nShe pocketed the muddy boot print cast. " + CLIFF
    r = ag.review(ch1, _ctx(p, 1))
    assert r is not None and "the muddy boot" in r.data["clues_shown"]
    ag.remember(ch1, _ctx(p, 1))
    assert "the muddy boot" in ag.state.seen_clues


def test_agent_flags_unfair_reveal(tmp_path):
    p = _proj(tmp_path)
    ag = ThrillerCraft(FakeLLM(), p.dir, {"clues": ["the muddy boot", "the torn ticket"],
                                          "red_herrings": ["the jealous brother"], "reveal_chapter": 2})
    # Chapter 1 shows only one of the two key clues.
    ch1 = TENSE * 3 + "\n\nThe muddy boot print was unmistakable. " + CLIFF
    ag.review(ch1, _ctx(p, 1))
    ag.remember(ch1, _ctx(p, 1))
    # Chapter 2 is the reveal, and the torn ticket + the red herring were never handled.
    reveal = TENSE * 3 + "\n\nThe killer was unmasked at last in the ruined hall. " + CLIFF
    r = ag.review(reveal, _ctx(p, 2))
    assert any("torn ticket" in i and "never shown" in i for i in r.issues)
    assert any("jealous brother" in i for i in r.issues)


def test_agent_brief_nudges_unplanted_clues_near_reveal(tmp_path):
    p = _proj(tmp_path)
    ag = ThrillerCraft(FakeLLM(), p.dir, {"clues": ["the torn ticket"], "reveal_chapter": 3})
    brief = ag.brief(_ctx(p, 2))  # within 2 chapters of the reveal
    assert "fair play" in brief and "torn ticket" in brief


def test_agent_attaches_only_for_dark_genres(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "d"))
    dark = _proj(tmp_path)
    assert "thriller_craft" in build_council(cfg, FakeLLM(), dark).names()
    cosy = Project.create(tmp_path / "cosy", "Sunny Days", "a gentle romance", ["Romance"], 300)
    assert "thriller_craft" not in build_council(cfg, FakeLLM(), cosy).names()


def test_agent_skips_short_text(tmp_path):
    p = _proj(tmp_path)
    ag = ThrillerCraft(FakeLLM(), p.dir, {})
    assert ag.review("A short line.", _ctx(p, 1)) is None
