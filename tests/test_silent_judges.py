"""A judge that failed or returned nothing does not vote.

The rubric critic, the in-character and world reviews, and the council's agents all ask a model for a
verdict. When that model fails, answers with something unusable, or leaves the score out, the chapter
is scored on what was actually measured and judged; nobody's silence is filled in with a default."""
import json

import pytest

from gauge.llm import FakeLLM
from gauge.llm.client import LLMError, given_number
from gauge.write import WritePipeline
from gauge.write.agents import (
    ALL_AGENTS,
    Archivist,
    Comedian,
    LineEditor,
    ReaderPanel,
    SceneContext,
    Schemer,
    SensitivityReader,
    Showrunner,
    ThemeKeeper,
    build_council,
)
from gauge.write.agents.archivist import Fact
from gauge.write.agents.scout import ScoutAgent
from gauge.write.characters import Character, CharacterAgent
from gauge.write.critic import JUDGE, Critic
from gauge.write.project import ChapterPlan, Project
from gauge.write.world import WorldAgent, WorldState

FAKE_JUDGED = 0.25 * 0.7 + 0.2 * 0.6 + 0.15 * 0.6 + 0.3 * 0.8 + 0.1 * 0.7  # the offline FakeLLM's rubric answer


def _always_fails() -> FakeLLM:
    return FakeLLM(fail_on=lambda prompt: True)


def _says(*answers) -> FakeLLM:
    """A model that answers with exactly these (JSON-encoded unless already text), in order."""
    return FakeLLM(canned=[a if isinstance(a, str) else json.dumps(a) for a in answers])


def _ctx(tmp_path, number=2):
    p = Project.create(tmp_path / "proj", "P", "idea", ["Thriller"], 300)
    plan = ChapterPlan(number=number, title="Chapter 2", goal="Rin confronts The Broker", hook="a knock",
                       turn="the Broker smiles", cliffhanger="Jun is gone")
    return p, SceneContext(project=p, number=number, plan=plan, prev_tail="She ran.", summary="Rin rewound time.",
                           cast_text="- Rin: wants out", genres=["Thriller"])


# ------------------------------------------------------------------ the helper
def test_given_number_only_accepts_what_was_actually_given():
    assert given_number(7, 0, 10) == 7.0 and given_number("7.5", 0, 10) == 7.5 and given_number(0, 0, 10) == 0.0
    for nothing in (None, "", "n/a", "8/10", [], {}, True, False, float("nan"), float("inf"), -1, 11, 85):
        assert given_number(nothing, 0, 10) is None, nothing


# ------------------------------------------------------------------ the rubric critic
@pytest.mark.parametrize("hooky", [True, False])
def test_a_judge_that_always_fails_gets_no_vote(chapter_text, hooky):
    """The acceptance case: the chapter's score is exactly what the measured part alone gives, and the
    result says the judge did not answer."""
    text, plan = chapter_text(hooky), ChapterPlan(number=1)
    deaf = _always_fails()
    cr = Critic(deaf, None).critique(text, plan)
    assert [c["purpose"] for c in deaf.calls_list] == ["critic"]  # it was asked, once, and it failed
    assert cr.judged is None and not cr.judge_answered and cr.rubric == {} and cr.rubric_coverage == 0.0
    assert cr.silent == [JUDGE]
    assert cr.score == round(cr.measured, 4)  # the measured part alone: no credit from the judge
    assert cr.notes == "" and Critic(deaf, None).problems(cr) == []  # and no words put in its mouth for the reviser

    # The same chapter with a judge that works is scored exactly as before: half measured, half judged.
    ok = Critic(FakeLLM(), None).critique(text, plan)
    assert ok.measured == cr.measured and ok.measured_issues == cr.measured_issues
    assert ok.judge_answered and ok.silent == [] and ok.rubric_coverage == 1.0
    assert ok.judged == pytest.approx(FAKE_JUDGED) and ok.rubric == {"hook": 7.0, "momentum": 6.0, "payoff": 6.0, "cliffhanger": 8.0, "clarity": 7.0}
    assert ok.score == round(0.5 * ok.measured + 0.5 * ok.judged, 4)
    # What a failed judge used to be worth: a 0.6 vote for half the score. It is gone.
    assert cr.score != round(0.5 * cr.measured + 0.5 * 0.6, 4)


@pytest.mark.parametrize("answers", [
    ["not json at all", "still not json"],  # the first answer and the repair retry are both unparseable
    [{}],
    [{"notes": "Looks fine to me."}],  # an opinion without a single score
    [{"hook": None, "momentum": "n/a", "payoff": "good", "cliffhanger": [], "clarity": True}],
    [{"hook": 85, "momentum": 70, "payoff": 90, "cliffhanger": 95, "clarity": 80}],  # not the 0-10 scale it was asked for
    [{"hook": -3, "momentum": float("nan"), "payoff": 11}],
])
def test_a_judge_that_answers_with_nothing_usable_gets_no_vote(chapter_text, answers):
    cr = Critic(_says(*answers), None).critique(chapter_text(True), ChapterPlan(number=1))
    assert cr.judged is None and cr.silent == [JUDGE] and cr.rubric == {}
    assert cr.score == round(cr.measured, 4)


def test_a_partly_answered_rubric_votes_only_with_what_it_answered(chapter_text):
    text, plan = chapter_text(False), ChapterPlan(number=1)
    cr = Critic(_says({"cliffhanger": 9, "clarity": "8", "hook": None, "notes": "The ending lands."}), None).critique(text, plan)
    assert cr.rubric == {"cliffhanger": 9.0, "clarity": 8.0}  # the three missing dimensions are not filled with 5
    assert cr.rubric_coverage == pytest.approx(0.4) and cr.judge_answered and cr.silent == []
    assert cr.judged == pytest.approx((0.3 * 0.9 + 0.1 * 0.8) / 0.4) and cr.notes == "The ending lands."
    # Its vote carries 40% of the judge's half, so the score sits between measured-only and a full vote.
    assert cr.score == round((0.5 * cr.measured + 0.2 * cr.judged) / 0.7, 4)
    assert cr.measured < cr.score < 0.5 * cr.measured + 0.5 * cr.judged


def test_measured_extras_still_count_when_the_judge_is_silent(chapter_text):
    text, plan = chapter_text(True), ChapterPlan(number=1)
    cr = Critic(_always_fails(), None, retention_weight=0.3).critique(text, plan)
    assert cr.silent == [JUDGE] and cr.score == round(0.7 * cr.measured + 0.3 * cr.retention, 4)


def test_unreadable_text_is_still_zero_and_no_judge_is_asked():
    llm = FakeLLM()
    cr = Critic(llm, None).critique("zxq vbn " * 400, ChapterPlan(number=1))
    assert cr.score == 0.0 and not cr.readable and cr.judged is None and llm.calls_list == []


# ------------------------------------------------------------------ character and world reviews
def test_character_review_without_a_verdict_does_not_vote():
    rin = Character(name="Rin")
    failed = CharacterAgent(_always_fails(), rin).review("x", 2)
    assert not failed.answered and failed.issues == []
    # Complaints without a score are still passed to the reviser, but carry no vote.
    scoreless = CharacterAgent(_says({"issues": [{"quote": "Okay", "problem": "too calm", "fix": "let her swear"}]}), rin).review("x", 2)
    assert not scoreless.answered and scoreless.issues[0].quote == "Okay"
    assert not CharacterAgent(_says({"consistency": 85}), rin).review("x", 2).answered  # off the 0-10 scale
    ok = CharacterAgent(FakeLLM(), rin).review("x", 2)
    assert ok.answered and ok.consistency == 0.8
    assert WritePipeline._consistency([failed, scoreless]) is None  # nobody answered: no share of the score
    assert WritePipeline._consistency([failed, ok, scoreless]) == 0.8  # a silent 10/10 no longer dilutes a real verdict


def test_world_review_without_a_verdict_does_not_vote():
    w = WorldState(name="X", current_location="here", time="noon", weather="clear")
    assert not WorldAgent(_always_fails(), w).review("t", 1).answered
    scoreless = WorldAgent(_says({"violations": [{"quote": "the sun", "rule": "it is night", "fix": "neon"}]}), w).review("t", 1)
    assert not scoreless.answered and scoreless.violations[0]["rule"] == "it is night"
    ok = WorldAgent(FakeLLM(), w).review("t", 1)
    assert ok.answered and ok.consistency == 0.9


# ------------------------------------------------------------------ the council's agents
def test_agents_whose_model_fails_return_no_review(tmp_path):
    p, ctx = _ctx(tmp_path)
    deaf = _always_fails()

    for cls in (ReaderPanel, Showrunner, Comedian, SensitivityReader):
        agent = cls(deaf, p.dir)
        assert agent.review("She ran. The door opened.", ctx) is None and agent.silent, cls.name

    schemer = Schemer(FakeLLM(), p.dir)
    schemer.setup(p)
    assert schemer.review("x", ctx).score == 1.0 and not schemer.silent
    schemer.llm = deaf
    assert schemer.review("x", ctx) is None and schemer.silent

    # Nothing is remembered about a chapter nobody read: no "zero laughs", no "clean".
    comedian = Comedian(deaf, p.dir)
    comedian.review("x", ctx)
    comedian.remember("x", ctx)
    assert comedian.state.laughs == []
    reader = SensitivityReader(deaf, p.dir, {"block_on_flags": True})
    reader.review("x", ctx)
    reader.remember("x", ctx)
    assert ctx.number not in reader.state.flags

    # A fact check that extracted no facts compared nothing with canon; it used to score 1.0.
    archivist = Archivist(deaf, p.dir)
    archivist.state.facts.append(Fact(subject="Rin", attribute="eye colour", value="grey", chapter=1))
    assert archivist.review("she ran and ran.", ctx) is None and archivist.silent

    class NoHits:
        def search(self, query, source, k=2):
            return type("Finding", (), {"snippets": []})()

    assert ScoutAgent(deaf, p.dir, {"fact_check": True}, scout=NoHits()).review("x", ctx) is None
    unverified = ScoutAgent(_says({"claims": [{"claim": "The moon is cheese", "query": "moon"}]}, "no verdict", "none"),
                            p.dir, {"fact_check": True}, scout=NoHits())
    assert unverified.review("x", ctx) is None and unverified.silent and unverified.state.claims_checked == 0


def test_agents_with_a_measured_part_fall_back_to_it(tmp_path, chapter_text):
    p, ctx = _ctx(tmp_path)
    text = chapter_text(True)
    heard, deaf = LineEditor(FakeLLM(), p.dir), LineEditor(_always_fails(), p.dir)
    a, b = heard.review(text, ctx), deaf.review(text, ctx)
    assert b.score == a.score and b.data == a.data  # the fake editor has no voice complaint, so both are the measurement
    assert deaf.silent and not heard.silent and not any(i.startswith("line:") for i in b.issues)

    theme = ThemeKeeper(FakeLLM(), p.dir)
    theme.setup(p)
    assert theme.review("nothing thematic", ctx).score == 1.0  # the fake model claims a motif is present
    theme.llm = _always_fails()
    rev = theme.review("nothing thematic", ctx)
    assert rev.score == 0.7 and theme.silent and rev.data["motifs"] == []  # only what is in the text counts


def test_an_answer_that_is_not_the_answer_is_no_answer(tmp_path):
    p, ctx = _ctx(tmp_path)
    for cls in (ReaderPanel, Showrunner, Comedian, SensitivityReader):
        agent = cls(_says({"ok": True}), p.dir)
        assert agent.review("x", ctx) is None and agent.silent, cls.name
    # One persona gave a probability, one only a comment: the comment is kept, but only the number votes.
    panel = ReaderPanel(_says({"readers": [{"persona": "binge", "click_next": 0.9},
                                           {"persona": "skimmer", "comment": "meh", "click_next": "maybe"}]}), p.dir)
    rev = panel.review("x", ctx)
    assert rev.score == 0.9 and rev.data["per_persona"] == {"binge": 0.9} and "skimmer: meh" in rev.notes
    assert ReaderPanel(_says({"readers": [{"persona": "skimmer", "comment": "meh"}]}), p.dir).review("x", ctx) is None


def test_council_leaves_silent_agents_out_of_the_score(cfg, tmp_path, chapter_text):
    p = Project.create(tmp_path / "pp", "Comic Tale", "idea", ["Comedy"], 300)
    ctx = SceneContext(project=p, number=1, plan=ChapterPlan(number=1, title="One", goal="Rin runs"), genres=["Comedy"])
    text = chapter_text(True)
    council = build_council(cfg, _always_fails(), p)
    reviews = council.review(text, ctx)
    asked = {"reader_panel", "showrunner", "comedian", "sensitivity"}  # verdicts that come only from the model
    assert asked <= set(council.silent) and "line_editor" in council.silent
    voted = {r.agent for r in reviews}
    assert not voted & asked and "line_editor" in voted and "pacing" in voted
    pairs = [(r.score, ALL_AGENTS[r.agent].weight) for r in reviews if ALL_AGENTS[r.agent].weight > 0]
    assert council.score(reviews) == pytest.approx(sum(s * w for s, w in pairs) / sum(w for _, w in pairs))

    # With a model that answers, the same agents vote and nobody is listed as silent.
    working = build_council(cfg, FakeLLM(), p)
    assert asked <= {r.agent for r in working.review(text, ctx)} and working.silent == []


# ------------------------------------------------------------------ end to end
class DeafJudges(FakeLLM):
    """Writes and plans like the FakeLLM, but every request for a verdict fails."""

    def generate(self, prompt, *args, purpose="generate", **kwargs):
        if purpose == "critic" or purpose.endswith(":review") or purpose in ("scout:claims", "scout:verify"):
            raise LLMError(f"no verdict for {purpose}")
        return super().generate(prompt, *args, purpose=purpose, **kwargs)


def test_pipeline_scores_a_chapter_on_measured_parts_when_no_judge_answers(cfg, db):
    pipe = WritePipeline(cfg, db, llm=DeafJudges())
    p = pipe.new_project("Ten Seconds", "a courier who can rewind time", ["Sci-fi"], words=300)
    res = pipe.write_next(p)
    crit = res.critique
    # The offline draft meets one of six measured targets. With every judge failing it used to clear the
    # bar first time at 0.72, on a 0.6 rubric default and 10/10 from reviewers who never answered.
    assert crit.measured < 0.5 and res.score < cfg.write.min_acceptable_score
    assert res.revisions == cfg.write.max_revisions  # so it is sent back for revision instead of waved through
    assert crit.judged is None and not crit.judge_answered
    assert {JUDGE, "world", "character:Rin", "agent:reader_panel", "agent:showrunner"} <= set(crit.silent)
    assert all(not r.answered for r in res.character_reviews) and not res.world_review.answered

    # The score is the measured critic score blended with the agents that measure; nothing else voted.
    assert not {r.agent for r in res.agent_reviews} & {"reader_panel", "showrunner", "schemer", "comedian"}
    pairs = [(r.score, ALL_AGENTS[r.agent].weight) for r in res.agent_reviews if ALL_AGENTS[r.agent].weight > 0]
    measured_room = sum(s * w for s, w in pairs) / sum(w for _, w in pairs)
    assert res.novelty is None  # first chapter: nothing to be novel against
    assert res.score == pytest.approx((1 - cfg.agents.weight) * round(crit.measured, 4) + cfg.agents.weight * measured_room)

    front = res.path.read_text(encoding="utf-8").split("---")[1]
    assert "\njudged: null\n" in front and "\njudge_answered: false\n" in front and "\nrubric: {}\n" in front
    assert '\nsilent_judges: ["judge", "character:Rin"' in front
    assert "\ncharacter_consistency: null\n" in front and "\nworld_consistency: null\n" in front


def test_pipeline_with_working_judges_is_unchanged(cfg, db):
    cfg.write.max_revisions = 0
    pipe = WritePipeline(cfg, db, llm=FakeLLM())
    p = pipe.new_project("Ten Seconds", "a courier who can rewind time", ["Sci-fi"], words=300)
    res = pipe.write_next(p)
    crit = res.critique
    assert crit.silent == [] and crit.judged == pytest.approx(FAKE_JUDGED) and crit.rubric_coverage == 1.0
    # The blend, step by step, as it has always been: rubric, then characters, world and the room.
    score = round(0.5 * crit.measured + 0.5 * crit.judged, 4)
    w = cfg.write.character_weight
    score = (1 - w) * score + w * sum(r.consistency for r in res.character_reviews) / len(res.character_reviews)
    w = cfg.write.world_weight
    score = (1 - w) * score + w * res.world_review.consistency
    pairs = [(r.score, ALL_AGENTS[r.agent].weight) for r in res.agent_reviews if ALL_AGENTS[r.agent].weight > 0]
    w = cfg.agents.weight
    score = (1 - w) * score + w * sum(s * x for s, x in pairs) / sum(x for _, x in pairs)
    assert res.score == pytest.approx(score)
    front = res.path.read_text(encoding="utf-8").split("---")[1]
    assert "\njudge_answered: true\n" in front and "\nsilent_judges: []\n" in front and "\ncharacter_consistency: 0.8\n" in front
