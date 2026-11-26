"""A whole novel of a set length: planned in volumes, a minimum length per chapter, written to the end."""
import sys

import pytest

from gauge.gui import StudioController
from gauge.llm import FakeLLM
from gauge.write import saga
from gauge.write.project import Project


class _Watch(FakeLLM):
    def __init__(self):
        super().__init__()
        self.seen: list[tuple[str, str]] = []
        self.fail_on_chapter = 0

    def generate(self, prompt, **kw):
        self.seen.append((kw.get("purpose", ""), prompt))
        if self.fail_on_chapter and kw.get("purpose") == "draft" and f"Write chapter {self.fail_on_chapter} " in prompt:
            raise RuntimeError("the engine stopped answering")
        return super().generate(prompt, **kw)


def _studio(cfg, db):
    ctl = StudioController(cfg, db)
    llm = _Watch()
    pipe = ctl.pipeline()
    pipe.llm = pipe.fast_llm = llm
    for part in ("planner", "drafter"):
        getattr(pipe, part).llm = llm
    return ctl, llm, pipe


# ------------------------------------------------------------------ the plan
def test_volumes_cover_every_chapter_exactly_once():
    for chapters in (1, 49, 50, 51, 500, 503, 10_000):
        vols = saga.plain_plan(chapters)
        assert vols[0]["first"] == 1 and vols[-1]["last"] == chapters
        assert all(a["last"] + 1 == b["first"] for a, b in zip(vols, vols[1:]))
        assert len(vols) == saga.volume_count(chapters)
    ten = saga.plain_plan(500)
    assert len(ten) == 10 and all(v["last"] - v["first"] == 49 for v in ten)
    assert "promise" in ten[0]["about"] and "ending" in ten[-1]["about"] and "cost counted" in ten[-1]["ends_with"]


def test_the_model_plans_the_volumes_and_a_failure_falls_back(cfg, db):
    ctl, llm, pipe = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    p = pipe.open_project(name)
    vols = saga.plan_novel(p, pipe.fast_llm, 500)
    assert len(vols) == 10 and vols[0]["title"] == "The Debt 1" and vols[-1]["ends_with"] == "Jun found"
    assert p.state.planned_chapters == 500 and Project(cfg.projects_dir, name).state.volumes[3]["first"] == 151
    asked = next(prompt for kind, prompt in llm.seen if kind == "saga:plan")
    assert "500 chapters, divided into 10 volumes" in asked and "The last volume must resolve the promise" in asked
    md = saga.plan_markdown(p)
    assert "500 chapters in 10 volumes" in md and "## Volume 10: The Rewind 10 (chapters 451-500)" in md

    class _Down(FakeLLM):
        def generate(self, prompt, **kw):
            raise RuntimeError("down")

    plain = saga.plan_novel(p, _Down(), 120)                       # no model: a structural plan, never a failure
    assert len(plain) == 3 and plain[0]["title"] == "Volume 1" and p.state.planned_chapters == 120


def test_the_outliner_is_told_where_it_is_and_when_to_end(cfg, db):
    ctl, _llm, pipe = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    p = pipe.open_project(name)
    assert saga.where(p, 1) == ""                                  # no length planned: nothing is said
    saga.plan_novel(p, pipe.fast_llm, 500)
    early = saga.where(p, 1)
    assert "THE WHOLE NOVEL is 500 chapters" in early and "500 chapters remain" in early
    assert 'VOLUME 1 of 10 (chapters 1-50), "The Debt 1"' in early and "50 chapters of this volume remain" in early
    assert "LAST CHAPTERS" not in early and "final stretch" not in early
    closing = saga.where(p, 146)                                   # the volume is about to end
    assert "The volume is ending" in closing and '"The Broker 4"' in closing
    assert "final stretch: close threads rather than open them" in saga.where(p, 480)
    last = saga.where(p, 496)
    assert "THESE ARE THE LAST CHAPTERS OF THE NOVEL" in last and "an ending, not a cliffhanger" in last
    assert saga.where(p, 501) == ""


# ------------------------------------------------------------------ the run
def test_a_short_novel_is_written_to_its_end_and_no_further(cfg, db):
    ctl, llm, pipe = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    p = pipe.open_project(name)
    seen = []
    res = pipe.write_novel(p, 12, min_words=400, progress=seen.append)
    assert res["finished"] and res["wrote"] == 12 and res["written"] == 12 and res["below_minimum"] == 0
    assert res["shortest"] >= 400 and "The novel is finished: 12 chapters" in res["message"]
    done = Project(cfg.projects_dir, name)
    assert all(len(done.read_chapter(n).split()) >= 400 for n in range(1, 13)) and done.read_chapter(13) is None
    assert all(done.read_meta(n)["min_words"] == 400 and done.read_meta(n)["short"] is False for n in (1, 12))
    assert [r["chapter"] for r in seen] == list(range(1, 13)) and seen[-1]["left"] == 0 and seen[0]["of"] == 12
    outlines = [prompt for kind, prompt in llm.seen if kind == "outline"]
    assert all("THE WHOLE NOVEL is 12 chapters" in o for o in outlines[1:])     # all but the one made before the plan
    assert "VOLUME 1 of 1 (chapters 1-12)" in outlines[1] and "This stretch starts at chapter 1;" in outlines[1]
    assert "THESE ARE THE LAST CHAPTERS OF THE NOVEL" in outlines[-1]
    again = pipe.write_novel(p)                                    # already finished: nothing more is written
    assert again["wrote"] == 0 and again["finished"]
    with pytest.raises(ValueError):
        pipe.write_novel(Project.create(cfg.projects_dir, "No Length", "idea", ["x"], 300))


def test_it_stops_between_chapters_and_carries_on_later(cfg, db):
    ctl, _llm, pipe = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    p = pipe.open_project(name)
    first = pipe.write_novel(p, 8, min_words=300, should_stop=lambda: p.state.written >= 3)
    assert not first["finished"] and first["written"] == 3 and first["reason"] == "stopped"
    assert "Stopped at chapter 3 of 8" in first["message"] and "5 to go; start again to carry on" in first["message"]
    # A STOP file beside the story does the same from another window.
    (p.dir / saga.STOP_FILE).write_text("stop")
    cleared = pipe.write_novel(p, should_stop=lambda: p.state.written >= 5)   # a request left from last time is cleared
    assert cleared["written"] == 5 and not (p.dir / saga.STOP_FILE).exists()
    ctl2, _llm2, pipe2 = _studio(cfg, db)                          # a new program, later
    p2 = pipe2.open_project(name)
    rest = pipe2.write_novel(p2)
    assert rest["finished"] and rest["wrote"] == 3 and p2.state.written == 8
    assert saga.status(p2)["line"].startswith("8 of 8 chapters written") and "no chapter under 300 words" in saga.status(p2)["line"]


def test_a_chapter_that_fails_is_tried_again_and_repeated_failure_stops_the_run(cfg, db):
    ctl, llm, pipe = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    p = pipe.open_project(name)
    llm.fail_on_chapter = 2
    res = pipe.write_novel(p, 4, min_words=200, max_failures=2, pause_after_failure=0)
    assert not res["finished"] and res["written"] == 1 and "stopped after 2 failures in a row" in res["reason"]
    assert "the engine stopped answering" in res["reason"]
    llm.fail_on_chapter = 0                                        # the engine is back
    assert pipe.write_novel(p, pause_after_failure=0)["finished"] and p.state.written == 4


def test_a_floor_is_reached_even_by_a_model_that_writes_little(cfg, db):
    from gauge.write.drafter import Drafter

    class _Terse(FakeLLM):
        calls = 0

        def generate(self, prompt, **kw):
            _Terse.calls += 1
            return " ".join(f"w{_Terse.calls}x{i}" for i in range(40))    # forty words at a time, whatever is asked

    ctl, _llm, pipe = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    p = pipe.open_project(name)
    plan = pipe.planner.ensure_plan(p, 1)
    text, used = Drafter(_Terse(), None, "").reach_length("Start. " * 10, 1000, p, plan, floor=1000)
    assert len(text.split()) >= 1000 and used >= 24                 # far more stretches than the usual limit of four

    class _Dry(FakeLLM):
        def generate(self, prompt, **kw):
            return "nothing more."

    stuck, tries = Drafter(_Dry(), None, "").reach_length("Start. " * 10, 1000, p, plan, floor=1000)
    assert len(stuck.split()) < 1000 and tries == 2                 # a model that has dried up does not hold the run for ever


# ------------------------------------------------------------------ the command line and the Studio
def test_the_novel_command(cfg, db):
    from click.testing import CliRunner

    from gauge.cli import main

    cfg.save()
    runner = CliRunner(env={"COLUMNS": "200"})

    def run(*args, code=0):
        r = runner.invoke(main, ["--data-dir", cfg.data_dir, *args])
        assert r.exit_code == code, r.output
        return r.output

    run("new", "Ten Seconds", "--idea", "a courier", "--genre", "Sci-fi", "--words", "200")
    assert "open-ended" in run("novel", "Ten Seconds", "--status")
    assert "--chapters 500" in run("novel", "Ten Seconds", "--plan", code=1)
    plan = run("novel", "Ten Seconds", "--chapters", "100", "--plan")
    assert "100 chapters in 2 volumes" in plan and "## Volume 2" in plan and "(chapters 51-100)" in plan
    out = run("novel", "Ten Seconds", "--chapters", "3", "--min-words", "300")
    assert "Chapter 3 of 3" in out and "The novel is finished: 3 chapters" in out and "hours left" in out
    assert "3 of 3 chapters written" in run("novel", "Ten Seconds", "--status")
    assert "Asked it to stop" in run("novel", "Ten Seconds", "--stop")
    assert "no project named" in run("novel", "Nope", code=1)


def test_the_studio_writes_a_whole_novel_and_can_be_stopped(cfg, db, tmp_path, monkeypatch):
    from gauge import gui
    from tests.test_teach import _Entry, _fake_tkinter, _wait_for_jobs, _Widget

    ctl = StudioController(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    assert "open-ended" in ctl.novel_status(name)["line"] and ctl.novel_status(None)["line"] == "Start or open a story first."
    with pytest.raises(ValueError):
        ctl.write_novel(name, "lots")
    seen = []
    res = ctl.write_novel(name, "4", "300 words", progress=seen.append)
    assert res["finished"] and len(seen) == 4 and ctl.novel_status(name)["line"].startswith("4 of 4 chapters written")
    assert "Nothing is left half-written" in ctl.stop_novel()["message"]

    mods, shown = _fake_tkinter(tmp_path)
    for mod_name, mod in mods.items():
        monkeypatch.setitem(sys.modules, mod_name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    mods["tkinter.simpledialog"].askstring = lambda *a, **k: "Ten Seconds"
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    assert "Write the whole novel..." in _Widget.buttons and "Write the whole novel..." in _Widget.menu
    _Widget.buttons["Open another story"]()
    _Widget.buttons["Write the whole novel..."]()
    assert {"Start (or carry on)", "Stop after this chapter"} <= set(_Widget.buttons)
    chapters_box = next(e.kw["textvariable"] for e in _Entry.made if e.kw["textvariable"].get() == "4")
    chapters_box.set("6")
    _Widget.buttons["Start (or carry on)"]()
    _wait_for_jobs()
    assert Project(cfg.projects_dir, "Ten Seconds").state.written == 6
    _Widget.buttons["Stop after this chapter"]()
    assert not [s for s in shown if s[0] == "error"], shown
