"""Write several, keep the best: judging drafts without a model."""
import pytest

from gauge.gui import StudioController
from gauge.llm import FakeLLM
from gauge.write import selection as sel
from gauge.write.project import Project

GOOD = ("Rin counted the exits before she sat down. The bag was still sealed, and Jun had not come back. "
        "She waited by the window until the street went quiet, then she took the stairs two at a time. ") * 3
INVENTS = GOOD.replace("Jun had not come back", "Marcus and Yelena and Tobias had not come back")
LOOPS = "She ran and she ran and she ran and she ran. " * 20


def test_invented_names_repetition_and_length_all_count():
    known = {"Rin", "Jun"}
    good, invents, loops = (sel.judge(t, known, target_words=110) for t in (GOOD, INVENTS, LOOPS))
    assert good.invented == [] and invents.invented == ["Marcus", "Tobias", "Yelena"]
    assert good.score > invents.score > -1 and good.score > loops.score
    assert loops.repeated > 0.8 and good.repeated < 0.75 and sel.repeated_share("too short") == 0.0
    assert sel.judge(GOOD, known, target_words=110).score > sel.judge(GOOD, known, target_words=1000).score   # far from the length asked
    assert sel.judge("no.", known).score == -1.0
    assert "invented names: Marcus, Tobias, Yelena" in invents.line() and "words" in good.line()
    # A name the story already has in a longer form is not an invention.
    assert sel.judge("Later, Rinna spoke and nobody answered her at all that night.", {"Rin"}).invented == []
    # The measured pull, when there is one, is part of the score.
    assert sel.judge(GOOD, known, 110, measure=lambda t: 0.9).score > sel.judge(GOOD, known, 110, measure=lambda t: 0.1).score
    assert sel.judge(GOOD, known, 110, measure=lambda t: 1 / 0).measured is None   # a failing measure is ignored


def test_the_best_draft_is_picked_and_every_verdict_is_kept():
    best, verdicts = sel.pick([LOOPS, INVENTS, GOOD], {"Rin", "Jun"}, 110)
    assert best == 2 and len(verdicts) == 3 and verdicts[2].score == max(v.score for v in verdicts)
    assert sel.pick([], set()) == (0, [])


def test_the_canon_is_the_bible_the_cast_and_what_was_written(cfg, db):
    ctl = StudioController(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    ctl.write_next(name)
    pipe = ctl.pipeline()
    p = pipe.open_project(name)
    known = sel.known_names(p, pipe.roster(p), ["Old Marrow kept the gate. Marrow did not sleep."])
    assert {"Rin", "Jun", "Marrow"} <= known and "Ten" in known
    assert sel.known_names(object()) == set()                                    # nothing to read: nothing known


class _Uneven(FakeLLM):
    """A model whose attempts differ: only one of them stays inside the story."""

    def __init__(self):
        super().__init__()
        self.temps: list[float] = []

    def generate(self, prompt, **kw):
        if kw.get("purpose") in ("draft", "draft:continue"):
            self.temps.append(kw.get("temperature"))
            i = len(self.temps) % 4
            return [LOOPS, INVENTS, GOOD, INVENTS][i]
        return super().generate(prompt, **kw)


def test_the_pipeline_writes_several_and_keeps_the_best(cfg, db):
    cfg.write.best_of = 4
    cfg.write.reach_length = False
    ctl = StudioController(cfg, db)
    llm = _Uneven()
    pipe = ctl.pipeline()
    pipe.llm = pipe.fast_llm = llm
    for part in ("planner", "drafter"):
        getattr(pipe, part).llm = llm
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    ctl.write_next(name, 110)
    p = Project(cfg.projects_dir, name)
    text, meta = p.read_chapter(1), p.read_meta(1)
    assert "Marcus" not in text and "she ran and she ran" not in text and "Rin counted the exits" in text
    assert meta["best_of"] == 4 and meta["choices"][0]["of"] == 4 and meta["choices"][0]["invented"] == []
    assert meta["choices"][0]["best"] > meta["choices"][0]["worst"]
    assert len(llm.temps) == 4 and len(set(llm.temps)) == 4                       # four attempts, each a little different

    cfg.write.best_of = 1                                                        # once is once
    llm.temps.clear()
    ctl._pipe = None
    pipe = ctl.pipeline()
    pipe.llm = pipe.fast_llm = llm
    for part in ("planner", "drafter"):
        getattr(pipe, part).llm = llm
    ctl.write_next(name, 110)
    assert len(llm.temps) == 1 and Project(cfg.projects_dir, name).read_meta(2)["best_of"] == 1


def test_gaugelm_writes_four_attempts_and_the_old_number_comes_back(cfg, db):
    from gauge.config import GaugeConfig

    cfg.write.best_of = 1
    ctl = StudioController(cfg, db)
    ctl.use_gaugelm(True)
    assert GaugeConfig.load(cfg.data_dir).write.best_of == 4
    ctl.use_gaugelm(False)
    assert GaugeConfig.load(cfg.data_dir).write.best_of == 1


def test_a_failed_attempt_does_not_lose_the_others(cfg, db):
    from gauge.write.drafter import Drafter

    class _Flaky(FakeLLM):
        n = 0

        def generate(self, prompt, **kw):
            _Flaky.n += 1
            if _Flaky.n == 1:
                raise RuntimeError("hiccup")
            return GOOD

    d = Drafter(_Flaky(), None, "")
    d.best_of, d.known_names = 3, {"Rin", "Jun"}
    assert d._write("write", 100, "draft", 110).startswith("Rin counted") and d.choices[0]["of"] == 2

    class _Dead(FakeLLM):
        def generate(self, prompt, **kw):
            raise RuntimeError("down")

    d2 = Drafter(_Dead(), None, "")
    d2.best_of = 2
    with pytest.raises(RuntimeError):
        d2._write("write", 100, "draft", 110)
