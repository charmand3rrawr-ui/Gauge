"""Name-spelling drift (roadmap 28) and explicit clock times (roadmap 27)."""
from collections import Counter

from gauge.llm.client import FakeLLM
from gauge.write.agents.archivist import Archivist
from gauge.write.agents.base import SceneContext
from gauge.write.agents.continuity_auditor import ContinuityAuditor
from gauge.write.names import _is_slip, proper_names, spelling_drift
from gauge.write.project import ChapterPlan, Project
from gauge.write.realism import clock_check


def _ctx(project, number):
    plan = ChapterPlan(number=number, title=f"Ch{number}", goal="g", hook="h", turn="t", cliffhanger="c", beats=["b"])
    return SceneContext(project=project, number=number, plan=plan)


def _proj(tmp_path):
    return Project.create(tmp_path / "proj", "Serial", "an idea", ["Drama"], 300)


# ------------------------------------------------------------------ names: pure functions
def test_proper_names_need_a_mid_sentence_capital_unless_known():
    text = "Then the rain came. She found Katherine by the gate. Katherine laughed. Rin ran."
    names = proper_names(text)
    assert names["Katherine"] == 2 and "Then" not in names and "She" not in names
    assert "Rin" not in names                                  # only ever opens a sentence: proves nothing
    assert proper_names(text, known={"Rin"})["Rin"] == 1       # but a known name counts wherever it stands


def test_slip_rules_separate_typos_from_different_names():
    slips = [("Katharine", "Katherine"), ("Lowmarkt", "Lowmarket"), ("Rinn", "Rin"), ("Maar", "Mara"),
             ("Tomas", "Thomas"), ("Aldric", "Aldrik"), ("Catherine", "Katherine")]
    different = [("Mara", "Maya"), ("Jon", "Jan"), ("Sara", "Mara"), ("Rins", "Rin"), ("Brokers", "Broker"),
                 ("Tom", "Tim"), ("Rin", "Rin")]
    for v, c in slips:
        assert _is_slip(v, c), (v, c)
    for v, c in different:
        assert not _is_slip(v, c), (v, c)


def test_spelling_drift_respects_canon_and_cast():
    canon = {"Katherine": 12, "Lowmarket": 9, "Mara": 6, "Maya": 5, "Jun": 20, "Aldrik": 2}
    now = Counter({"Katharine": 1, "Lowmarkt": 2, "Katherine": 3, "Maya": 1, "Aldric": 1})
    drift = spelling_drift(now, canon, protected={"Mara", "Maya"})
    assert [(v, c) for v, c, *_ in drift] == [("Katharine", "Katherine"), ("Lowmarkt", "Lowmarket")]
    # 'Aldrik' has only been used twice, so it is not established enough to be drifted from.
    assert "Aldric" not in [v for v, *_ in drift]
    # A cast name is never reported as a slip, even when it sits next to another cast name.
    assert spelling_drift(Counter({"Maya": 4}), {"Mara": 30}, protected={"Mara", "Maya"}) == []
    # Both spellings used several times in one chapter are two people sharing a scene, not a slip...
    assert spelling_drift(Counter({"Julie": 4, "Julia": 5}), {"Julia": 40}) == []
    # ...but one stray variant among the usual spelling is, and so is a chapter that only uses the variant.
    assert spelling_drift(Counter({"Julie": 1, "Julia": 5}), {"Julia": 40})[0][:2] == ("Julie", "Julia")
    assert spelling_drift(Counter({"Katharine": 6}), {"Katherine": 40})[0][:2] == ("Katharine", "Katherine")
    # Without the cast list, a rare name next to a heavily used one is reported.
    assert spelling_drift(Counter({"Maar": 1}), {"Mara": 30})[0][:2] == ("Maar", "Mara")


# ------------------------------------------------------------------ names: the Archivist
def test_archivist_flags_name_drift_without_a_model_call(tmp_path):
    p = _proj(tmp_path)
    llm = FakeLLM()
    ar = Archivist(llm, p.dir, {})
    ar.state.names = {"Katherine": 14, "Lowmarket": 8}
    text = "She crossed Lowmarket in the rain and found Katharine waiting under the awning, soaked through."
    r = ar.review(text, _ctx(p, 9))
    assert r is not None and r.data["name_drift"] == ["Katharine"]
    assert "Katharine" in r.issues[0] and "Katherine" in r.issues[0] and r.score < 1.0
    assert llm.calls_list == []                                 # no facts yet -> the name check alone, measured
    clean = "She crossed Lowmarket in the rain and found Katherine waiting under the awning."
    assert ar.review(clean, _ctx(p, 9)) is None                 # nothing to say: unchanged behaviour


def test_archivist_builds_the_name_index_and_a_slip_stays_rare(tmp_path):
    p = _proj(tmp_path)
    ar = Archivist(FakeLLM(), p.dir, {})
    for n in range(1, 5):
        ar.remember("The broker watched Katherine cross Lowmarket, and Katherine did not look back.", _ctx(p, n))
    assert ar.state.names["Katherine"] == 8 and ar.state.names["Lowmarket"] == 4
    ar.remember("That night Katharine crossed Lowmarket alone.", _ctx(p, 5))   # the slip got through once
    assert ar.state.names["Katharine"] == 1
    r = ar.review("By morning Katharine was gone from Lowmarket for good.", _ctx(p, 6))
    assert "Katharine" in r.data["name_drift"]                  # still flagged next time: not established
    assert "Katherine" in ar.name_index() and "names indexed" in ar.report()


# ------------------------------------------------------------------ clock: pure function
def test_clock_runs_forward_or_is_explained():
    assert clock_check("At 9 a.m. she left. By 3 PM the market was closing. At 23:10 the last tram went.").issues == []
    assert clock_check("At 11 p.m. he gave up. The next morning, at 7 a.m., he tried the door again.").issues == []
    assert clock_check("At 11:50 p.m. the bell rang. At 12:10 a.m. she was still counting.").issues == []   # midnight passing
    assert clock_check("It was 6 p.m. now. Earlier, at 9 a.m., she had been certain.").issues == []         # a look back


def test_clock_flags_backwards_and_impossible_light():
    back = clock_check("At 4 p.m. the meeting began. The room filled slowly. At 2 p.m. she was still waiting outside.")
    assert len(back.issues) == 1 and "runs backwards" in back.issues[0] and "16:00" in back.issues[0]
    assert "runs backwards" in clock_check("By 18:00 the shop shut. She stood there. At 15:30 the shop was busy.").issues[0]
    sun = clock_check("At 2 a.m. the sunlight lay warm across the kitchen table.")
    assert sun.issues and "sunlight" in sun.issues[0]
    assert clock_check("At 2 p.m. the sunlight lay warm across the kitchen table.").issues == []
    assert clock_check("By noon the moonlight silvered the roofs.").issues
    assert clock_check("At 2 a.m. the street was empty. By morning, sunlight filled the room.").issues == []  # other sentence


def test_clock_ignores_dialogue_and_things_that_are_not_times():
    c = clock_check('It was 5 p.m. "Meet me at 9 a.m.," he said. She nodded.')
    assert [p for _, p in c.times] == ["5 p.m."] and c.issues == []
    assert clock_check("The odds were 3:1 and the score 14:30 meant nothing; she thought of the 3 of them.").times == []
    assert [t for t, _ in clock_check("At 11:50 p.m. the bell rang.").times] == [23 * 60 + 50]   # matched once, not twice


# ------------------------------------------------------------------ clock: the Continuity Auditor
def test_continuity_auditor_reports_the_clock(tmp_path):
    p = _proj(tmp_path)
    ca = ContinuityAuditor(FakeLLM(), p.dir, {})
    pad = "The room was quiet and the papers lay where she had left them on the long table by the window. " * 6
    bad = "At 4 p.m. the meeting began. " + pad + "At 2 p.m. she was still waiting outside the door."
    r = ca.review(bad, _ctx(p, 1))
    assert any("runs backwards" in i for i in r.issues) and r.data["clock"] == "16:00 to 14:00"
    ca.remember(bad, _ctx(p, 1))
    assert ca.state.history[-1]["clock"] == ["16:00", "14:00"] and "clock 16:00 to 14:00" in ca.report()
    good = "At 9 a.m. she arrived. " + pad + "By 5 p.m. the last of them had gone home."
    r2 = ca.review(good, _ctx(p, 2))
    assert not any("clock" in i for i in r2.issues) and r2.data["clock"] == "09:00 to 17:00"
