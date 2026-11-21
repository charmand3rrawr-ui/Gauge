"""Roadmap section A: line-level prose-craft checks and the Craft Editor agent."""
from pathlib import Path

from gauge.llm import FakeLLM
from gauge.write.agents import CraftEditor
from gauge.write.agents.base import SceneContext
from gauge.write.prose import (
    anachronisms,
    cliche_hits,
    on_the_nose_lines,
    pov_heads,
    profile,
    score,
    sentence_opener_variety,
    talking_heads_runs,
    tense_mix,
    weak_verb_adverbs,
)

BAD = ("At the end of the day, her blood ran cold. She ran quickly. He walked slowly. She said softly. "
       "The knight pulled out his phone and sent an email. "
       "“I feel so sad,” she said. “I am so angry,” he said. "
       "She felt afraid. She felt afraid. She felt afraid. ") * 3
GOOD = ("Rain needled the window. She crossed the room, pulled the latch, and the cold hit her like a slap. "
        "Below, a gull argued with the wind. Counting the boats, she lost the number twice. "
        "Smoke curled from the chimney across the harbour; somewhere a chain dragged over stone.") * 3


def test_individual_checks():
    assert "at the end of the day" in cliche_hits(BAD) and cliche_hits(GOOD) == []
    assert len(weak_verb_adverbs("She ran quickly and said softly.")) == 2
    assert on_the_nose_lines('“I feel so sad,” she said.') == 1
    assert anachronisms("The knight checked his phone.", "medieval") == ["phone"]
    assert anachronisms("The knight checked his phone.", "") == []
    variety, overused = sentence_opener_variety("She ran. She hid. She waited. She ran. She wept.")
    assert variety < 0.6 and "she" in overused
    assert tense_mix("walked walked walked walked walked talks talks talks talks talks " * 3) > 0.6
    assert talking_heads_runs("“a”\n\n“b”\n\n“c”\n\n“d”\n\n“e”") == 1
    assert pov_heads("Rin felt cold. Jun thought it over.", ["Rin", "Jun"]) == 2


def test_profile_and_score_separate_good_from_bad():
    bad, good = profile(BAD, era="medieval"), profile(GOOD)
    assert score(bad) < score(good)
    assert bad.cliches and bad.weak_pairs and bad.on_the_nose and bad.anachronisms
    assert good.issues == [] and score(good) > 0.8


def test_craft_editor_agent(tmp_path):
    class _Proj:
        dir = Path(tmp_path)
        class state:  # noqa: N801
            class bible:  # noqa: N801
                era = "medieval"
    ctx = SceneContext(project=_Proj(), number=1, plan=type("Pl", (), {"title": "T"})())
    ag = CraftEditor(FakeLLM(), tmp_path, {"era": "medieval"})
    rev = ag.review(BAD, ctx)
    assert rev is not None and rev.agent == "craft_editor" and rev.score < 0.6 and rev.issues
    ag.remember(BAD, ctx)
    assert ag.state.history and "clich" in ag.report()
    assert ag.review("too short to judge", ctx) is None


def test_craft_editor_in_full_council(cfg):
    from gauge.write.agents import build_council
    from gauge.write.project import Project
    p = Project.create(cfg.projects_dir, "P", "idea", ["Sci-fi"], 300)
    names = build_council(cfg, FakeLLM(), p).names()
    assert "craft_editor" in names  # enabled by default, joins the full room
