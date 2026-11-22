"""Measured world-and-continuity checks (roadmap section C) and the World Auditor agent."""
from gauge.config import GaugeConfig
from gauge.llm.client import FakeLLM
from gauge.write.agents.base import SceneContext
from gauge.write.agents.world_auditor import WorldAuditor
from gauge.write.project import ChapterPlan, Project
from gauge.write.world_state import (
    check_rules,
    parse_rules,
    read_injuries,
    read_props,
    read_timeline,
    read_travel,
    read_weather,
)


# ------------------------------------------------------------------ pure functions
def test_timeline_sums_relative_markers():
    t = read_timeline("The next morning she woke. Three days later, the siege began.")
    assert t.days_elapsed == 4.0 and "Three days later" in t.markers


def test_timeline_respects_flashback_cue():
    t = read_timeline("Years earlier, as a child, she had been happy. " * 3 + "the next day a week later that evening an hour later")
    assert t.has_flashback_cue and t.issues == []  # a flashback excuses the many jumps


def test_weather_flags_season_contradiction():
    assert "summer" in read_weather("Snow fell across the summer fields.").issues[0]
    assert read_weather("The blizzard raged on.", established_season="summer").issues
    assert read_weather("Rain fell softly.", established_season="spring").issues == []


def test_injuries_and_vigorous_action_detected():
    assert read_injuries("Mara broke her leg when the wall fell.").injured == {"Mara": "broke her leg"}
    v = read_injuries("Mara sprinted up the stair and fought the guards.")
    assert "Mara" in v.vigorous and not v.recovery_cue
    assert read_injuries("Weeks later, her wound finally healed.").recovery_cue


def test_props_destroyed_and_used():
    assert "sword" in read_props("The sword shattered against the stone.").destroyed
    used = read_props("She drew the sword and raised the lantern.").used
    assert "sword" in used and "lantern" in used
    assert "sword and" not in used  # trailing conjunction trimmed


def test_travel_flags_instant_jump_between_known_places():
    tr = read_travel("She left Dawnmere. Moments later she stood in Blackwater.",
                     known_places={"Dawnmere", "Blackwater"})
    assert tr.issues and "Dawnmere" in tr.issues[0]
    # Without known places it stays quiet (too little to go on).
    assert read_travel("She left Dawnmere. Moments later she arrived.").issues == []


def test_world_rules_parse_and_check():
    rules = parse_rules("never: magic indoors\nalways: a price when magic\n# a comment\n\n")
    assert [r.kind for r in rules] == ["never", "always"]
    v = check_rules("He cast magic indoors with ease.", rules)
    assert len(v.violations) == 2  # the 'never' is broken and the 'always' price is missing
    assert check_rules("A quiet day with no sorcery at all.", rules).violations == []


# ------------------------------------------------------------------ the agent
def _ctx(project, number):
    plan = ChapterPlan(number=number, title=f"Ch{number}", goal="g", hook="h", turn="t", cliffhanger="c", beats=["b"])
    return SceneContext(project=project, number=number, plan=plan)


def _mk_project(tmp_path):
    return Project.create(tmp_path / "proj", "Saga", "an idea", ["Fantasy"], 300)


def test_world_auditor_carries_injury_across_chapters(tmp_path):
    p = _mk_project(tmp_path)
    wa = WorldAuditor(FakeLLM(), p.dir, {})
    hurt = "Kael was stabbed in the side during the ambush and collapsed against the cold wall of the keep. " * 8
    r1 = wa.review(hurt, _ctx(p, 1))
    assert r1 is not None
    wa.remember(hurt, _ctx(p, 1))
    assert "Kael" in wa.state.injured
    # Next chapter: Kael is in hard action with no recovery shown -> flagged.
    active = "Kael sprinted across the courtyard, vaulted the rail and fought three guards at once without pause. " * 8
    r2 = wa.review(active, _ctx(p, 2))
    assert any("Kael" in i and "hurt" in i for i in r2.issues)


def test_world_auditor_flags_destroyed_prop_reuse(tmp_path):
    p = _mk_project(tmp_path)
    wa = WorldAuditor(FakeLLM(), p.dir, {})
    broke = "In the duel the blade shattered against the iron gate and fell in pieces to the ground below. " * 8
    wa.review(broke, _ctx(p, 1))
    wa.remember(broke, _ctx(p, 1))
    assert "blade" in wa.state.destroyed
    reuse = "She drew the blade once more and swung it in a bright arc toward the startled sentry at the door. " * 8
    r = wa.review(reuse, _ctx(p, 2))
    assert any("blade" in i and "destroyed" in i for i in r.issues)


def test_world_auditor_applies_declared_world_rules(tmp_path):
    p = _mk_project(tmp_path)
    wa = WorldAuditor(FakeLLM(), p.dir, {"world_rules": "never: magic indoors"})
    text = "He cast magic indoors, and the chandeliers blazed with sudden impossible light above the hall. " * 8
    r = wa.review(text, _ctx(p, 1))
    assert any("world rule" in i for i in r.issues)


def test_world_auditor_short_text_is_skipped(tmp_path):
    p = _mk_project(tmp_path)
    wa = WorldAuditor(FakeLLM(), p.dir, {})
    assert wa.review("Too short.", _ctx(p, 1)) is None


def test_world_auditor_clean_chapter_scores_well(tmp_path):
    p = _mk_project(tmp_path)
    wa = WorldAuditor(FakeLLM(), p.dir, {})
    clean = ("The market was loud with bargaining and the smell of bread, and she moved through it slowly, "
             "listening to the vendors call their prices across the square. ") * 4
    r = wa.review(clean, _ctx(p, 1))
    assert r is not None and r.score >= 0.9 and r.issues == []


def test_world_auditor_is_in_the_lean_profile(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "d"))
    cfg.agents.profile = "lean"
    p = _mk_project(tmp_path)
    from gauge.write.agents import build_council
    assert "world_auditor" in build_council(cfg, FakeLLM(), p).names()
