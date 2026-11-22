"""Subplot tracking, foreshadow->payoff linking (roadmap 34/37) and the arc-shape planner (35)."""
from gauge.llm.client import FakeLLM
from gauge.write.agents.base import SceneContext
from gauge.write.agents.pacing import PacingDirector
from gauge.write.agents.thread_keeper import ThreadKeeper
from gauge.write.project import ChapterPlan, Project
from gauge.write.threads import (
    dormant_subplots,
    foreshadow_events,
    parse_foreshadow,
    parse_subplots,
    subplot_hits,
)

FILLER = ("The river moved slow under the bridge and the town went about its quiet evening business, "
          "lamps coming on one by one along the far bank as the light failed. ")


# ------------------------------------------------------------------ pure functions
def test_subplot_parse_and_hits():
    sps = parse_subplots(["Rin's debt: debt, broker, money owed", "Jun's illness: fever, cough", "plainthread"])
    assert sps[0].label == "Rin's debt" and "broker" in sps[0].keywords
    assert sps[2].label == "plainthread" and sps[2].keywords == ["plainthread"]
    assert subplot_hits("The broker wanted the money owed by dusk.", sps) == {"Rin's debt"}


def test_dormant_subplots_counts_the_gap():
    last = {"Rin's debt": 2, "Jun's illness": 7}
    # At chapter 8 with gap 5: debt (8-2=6) is dormant, illness (8-7=1) is fresh.
    assert dormant_subplots(last, ["Rin's debt", "Jun's illness"], 8, 5) == ["Rin's debt"]
    # A never-seen subplot (last_seen 0) is dormant once the gap has passed.
    assert "New thread" in dormant_subplots({}, ["New thread"], 6, 5)


def test_foreshadow_parse_and_events():
    fs = parse_foreshadow(["a scar on his wrist -> the scar matched the killer", "locked drawer => inside the drawer", "junk"])
    assert len(fs) == 2 and fs[0].plant == "a scar on his wrist"
    plants, payoffs = foreshadow_events("She saw a scar on his wrist and said nothing.", fs)
    assert plants == {"a scar on his wrist"} and payoffs == set()


# ------------------------------------------------------------------ the agent
def _ctx(project, number):
    plan = ChapterPlan(number=number, title=f"Ch{number}", goal="g", hook="h", turn="t", cliffhanger="c", beats=["b"])
    return SceneContext(project=project, number=number, plan=plan)


def _proj(tmp_path):
    return Project.create(tmp_path / "proj", "Long Serial", "an idea", ["Drama"], 300)


def test_thread_keeper_silent_with_nothing_declared(tmp_path):
    p = _proj(tmp_path)
    tk = ThreadKeeper(FakeLLM(), p.dir, {})
    assert tk.review(FILLER * 6, _ctx(p, 1)) is None


def test_thread_keeper_flags_dropped_subplot(tmp_path):
    p = _proj(tmp_path)
    tk = ThreadKeeper(FakeLLM(), p.dir, {"subplots": ["Rin's debt: debt, broker"], "subplot_gap": 3})
    seen = FILLER * 5 + "The broker leaned in and named the debt she still owed. "
    tk.review(seen, _ctx(p, 1))
    tk.remember(seen, _ctx(p, 1))
    assert tk.state.subplot_last_seen["Rin's debt"] == 1
    # By chapter 4 (gap 3) with the subplot absent, it is flagged as dropped.
    r = tk.review(FILLER * 6, _ctx(p, 4))
    assert any("Rin's debt" in i and "gone dark" in i for i in r.issues)


def test_thread_keeper_flags_unearned_payoff(tmp_path):
    p = _proj(tmp_path)
    tk = ThreadKeeper(FakeLLM(), p.dir, {"foreshadow": ["a scar on his wrist -> the scar matched the killer"]})
    # Payoff with no prior plant -> unearned.
    reveal = FILLER * 5 + "In the lamplight the scar matched the killer exactly, and she finally understood. "
    r = tk.review(reveal, _ctx(p, 2))
    assert any("no earlier setup" in i for i in r.issues)


def test_thread_keeper_payoff_after_plant_is_clean(tmp_path):
    p = _proj(tmp_path)
    tk = ThreadKeeper(FakeLLM(), p.dir, {"foreshadow": ["a scar on his wrist -> the scar matched the killer"]})
    plant = FILLER * 5 + "She noticed a scar on his wrist, pale and old, and filed it away. "
    tk.review(plant, _ctx(p, 1))
    tk.remember(plant, _ctx(p, 1))
    payoff = FILLER * 5 + "At last the scar matched the killer, and the room went cold. "
    r = tk.review(payoff, _ctx(p, 5))
    assert not any("no earlier setup" in i for i in r.issues)
    tk.remember(payoff, _ctx(p, 5))
    assert "a scar on his wrist" in tk.state.paid


def test_thread_keeper_brief_nudges_pending_threads(tmp_path):
    p = _proj(tmp_path)
    tk = ThreadKeeper(FakeLLM(), p.dir, {"foreshadow": ["a scar on his wrist -> the scar matched the killer"]})
    plant = FILLER * 5 + "A scar on his wrist caught the light. "
    tk.review(plant, _ctx(p, 1))
    tk.remember(plant, _ctx(p, 1))
    assert "waiting to pay off" in tk.brief(_ctx(p, 2))


# ------------------------------------------------------------------ arc-shape planner (35)
def test_arc_schedule_breathes_and_places_a_reveal(tmp_path):
    pd = PacingDirector(FakeLLM(), (tmp_path / "proj"), {})
    for length, start in [(6, 1), (9, 3), (12, 1), (2, 1), (1, 1)]:
        sch = pd.arc_schedule(length, start=start)
        assert len(sch) == length
        assert all(sch[i] != sch[i + 1] for i in range(len(sch) - 1))  # never two of a kind back to back
    assert pd.arc_schedule(6, start=1)[-1] == "action"   # ends on a high
    assert "reveal" in pd.arc_schedule(9, start=1)       # a reveal is scheduled across a full arc
