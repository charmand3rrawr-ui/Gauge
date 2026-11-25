"""GaugeLM's school: several free teachers, a tutor that corrects its own attempts, graded lessons."""
import json
import random

from gauge.gui import StudioController
from gauge.llm.client import OllamaLLM
from gauge.models import ChapterRecord, NovelRecord
from gauge.native import instruction_rows, school

TEXT = ("Rin counted the exits before she sat down and rubbed the scar on her palm. The bag was still sealed and the "
        "debt was still hers, and nobody in the room knew she had already run once tonight. ") * 30


class _Model:
    """A stand-in model: remembers what it was asked, answers a fixed thing."""

    def __init__(self, name="m", reply="Something happened on the night train, and it was not over yet. " * 4, up=True):
        self.name, self.reply, self.up, self.asked = name, reply, up, []

    def available(self):
        return self.up

    def generate(self, prompt, **kw):
        self.asked.append((kw.get("purpose"), prompt))
        if callable(self.reply):
            return self.reply(prompt)
        return self.reply


def _teacher(name="alpha", **kw):
    return school.Teacher(name, _Model(name, **kw), "Apache-2.0", True)


def _rows(cfg, which):
    f = cfg.models_dir / "datasets" / school.FILES[which]
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


# ------------------------------------------------------------------ who teaches
def test_licences_are_labelled_and_unknown_models_say_so():
    assert school.licence_for("qwen2.5:7b") == ("Apache-2.0", True)
    assert school.licence_for("qwen2.5:3b")[1] is False                    # that one size has its own terms
    assert school.licence_for("phi4:latest") == ("MIT", True) and school.licence_for("mistral-nemo")[1]
    assert school.licence_for("llama3.1:8b") == ("Llama Community License (own terms)", False)
    assert school.licence_for("someone/qwen2.5-coder:1.5b")[0] == "Apache-2.0"
    assert "unknown" in school.licence_for("brand-new-model")[0]


def test_every_chat_model_installed_teaches_the_writer_first(cfg, monkeypatch):
    cfg.llm.provider, cfg.llm.model = "ollama", "qwen2.5:14b"
    installed = ["llama3.2:3b", "nomic-embed-text:latest", "qwen2.5:14b", "phi4:latest", "bge-m3:latest"]
    monkeypatch.setattr(OllamaLLM, "models", lambda self: installed)
    monkeypatch.setattr(OllamaLLM, "available", lambda self: True)
    pool = school.teachers(cfg)
    assert [t.name for t in pool] == ["qwen2.5:14b", "llama3.2:3b", "phi4:latest"]      # no embedding models
    assert [t.open_licence for t in pool] == [True, False, True] and "on this computer" in pool[0].line()
    cfg.native.teachers = ["phi4", "not-installed"]                                     # the author chooses
    assert [t.name for t in school.teachers(cfg)] == ["phi4"]
    cfg.native.teachers = []
    monkeypatch.setattr(OllamaLLM, "models", lambda self: [])                           # the engine is off
    monkeypatch.setattr(OllamaLLM, "available", lambda self: False)
    assert school.teachers(cfg) == []


def test_gaugelm_is_still_taught_when_it_is_the_writer(cfg, db, monkeypatch):
    cfg.llm.provider, cfg.llm.model = "ollama", "qwen2.5:14b"
    ctl = StudioController(cfg, db)
    ctl.use_gaugelm(True)                                                              # remembers what the writer was
    assert cfg.llm.provider == "native"
    monkeypatch.setattr(OllamaLLM, "models", lambda self: ["qwen2.5:14b", "mistral:7b"])
    monkeypatch.setattr(OllamaLLM, "available", lambda self: True)
    assert [t.name for t in school.teachers(cfg)] == ["qwen2.5:14b", "mistral:7b"]
    line = ctl.school_line()
    assert "Taught by qwen2.5:14b and mistral:7b, taking turns" in line


def test_teachers_elsewhere_are_never_used_unless_listed_and_vouched_for(cfg, monkeypatch):
    from gauge.llm.client import OpenAICompatLLM

    cfg.llm.provider, cfg.llm.model = "ollama", "qwen2.5:14b"
    monkeypatch.setattr(OllamaLLM, "models", lambda self: ["qwen2.5:14b"])
    monkeypatch.setattr(OllamaLLM, "available", lambda self: True)
    monkeypatch.setattr(OpenAICompatLLM, "available", lambda self: True)
    monkeypatch.setenv("FREE_KEY", "k-123")
    cfg.native.outside_teachers = [{"name": "free-host", "base_url": "https://api.example.org/v1", "model": "open-model",
                                    "api_key_env": "FREE_KEY"}]
    assert [t.name for t in school.teachers(cfg)] == ["qwen2.5:14b"]                    # listed, but not vouched for
    cfg.native.outside_teachers[0]["terms_ok"] = True
    pool = school.teachers(cfg)
    assert [t.name for t in pool] == ["qwen2.5:14b", "free-host"]
    assert pool[1].where == "api.example.org" and not pool[1].open_licence and "provider's terms" in pool[1].licence


# ------------------------------------------------------------------ a session
def test_a_session_collects_answers_corrections_and_practice_scenes(cfg):
    teacher = _teacher(reply=lambda p: "mark: 3\nThe seal was whole again, and she ran." if "student writer" in p
                       else "A short scene about something that happened, told in plain words, ending on a hook. " * 3)
    student = _Model("gaugelm", reply="seal the the bag")
    seen = []
    res = school.session(cfg, [TEXT], want=9, pool=[teacher], student=student, step=100, progress=seen.append)
    assert (res["answers"], res["tutored"], res["lessons"], res["rows"]) == (3, 3, 3, 9)
    assert res["teacher"] == "alpha" and res["marks"] == [3.0, 3.0, 3.0] and seen[-1]["lessons"] == 3
    assert "alpha taught it: 3 worked answers, 3 corrections of its own attempts, 3 practice scenes." in res["message"]
    assert "marked its attempts 3.0 out of 10" in res["message"]
    tutor = _rows(cfg, "tutor")[0]
    assert tutor["response"] == "The seal was whole again, and she ran." and tutor["student"] == "seal the the bag"
    assert tutor["mark"] == 3.0 and tutor["teacher"] == "alpha" and tutor["licence"] == "Apache-2.0"
    assert "Rin counted the exits" in tutor["prompt"]                                   # set on the author's own text
    lesson = _rows(cfg, "lessons")[0]
    assert lesson["level"] == 1 and lesson["prompt"].startswith("Write a short scene about a courier on a night train")
    assert all(r["teacher"] == "alpha" for r in _rows(cfg, "answers"))
    assert len(student.asked) == 3 and all(kind == "school:attempt" for kind, _ in student.asked)
    # The trainer reads all three.
    assert len(instruction_rows(cfg.models_dir / "datasets")) == 9


def test_teachers_take_turns_and_one_teaches_a_whole_session(cfg):
    a, b = _teacher("alpha"), _teacher("beta")
    who = [school.session(cfg, [TEXT], want=4, pool=[a, b])["teacher"] for _ in range(4)]
    assert who == ["alpha", "beta", "alpha", "beta"]
    assert len(a.llm.asked) == len(b.llm.asked) == 8                                    # no swapping inside a session
    assert {r["teacher"] for r in _rows(cfg, "lessons")} == {"alpha", "beta"}
    # Without a trained GaugeLM there is nothing to tutor: answers and practice scenes only.
    assert not _rows(cfg, "tutor") and len(_rows(cfg, "answers")) == 8


def test_the_tutor_drills_what_gaugelm_is_worst_at(cfg):
    state = {"marks": {}, "sessions": 0, "lessons": 0}
    rng = random.Random(0)
    first = [school.weakest_first(state, rng)[0] for _ in range(1)]
    assert first == [school.TASKS[0][0]]                                                # untried kinds come first
    for kind, *_ in school.TASKS:
        school._note_mark(state, kind, 9.0, 10)
    school._note_mark(state, "dialogue", 1.0, 20)
    school._note_mark(state, "dialogue", 1.0, 30)
    picks = [school.weakest_first(state, rng)[0] for _ in range(400)]
    assert picks.count("dialogue") > 3 * max(picks.count(k) for k, *_ in school.TASKS if k != "dialogue")
    assert state["marks"]["dialogue"]["first"] == 9.0 and state["marks"]["dialogue"]["now"] < 5 and state["marks"]["dialogue"]["n"] == 3


def test_the_report_card_shows_marks_at_first_and_now(cfg):
    assert "has not been tutored yet" in school.report_card(cfg.models_dir / "datasets")["line"]
    marks = iter([2, 2, 2, 8, 8, 8])
    teacher = _teacher(reply=lambda p: f"mark: {next(marks)}\nBetter answer here." if "student writer" in p else "A scene. " * 20)
    for _ in range(2):
        school.session(cfg, [TEXT], want=9, pool=[teacher], student=_Model("gaugelm", reply="try"))
    card = school.report_card(cfg.models_dir / "datasets")
    assert card["sessions"] == 2 and card["counts"] == {"answers": 6, "tutor": 6, "lessons": 6}
    assert "at first" in card["line"] and "Weakest:" in card["line"] and len(card["rows"]) >= 3
    assert all(r["first"] in (2.0, 8.0) and r["practised"] >= 1 for r in card["rows"])


def test_a_tutors_reply_is_read_however_it_writes_the_mark():
    assert school.parse_marked("mark: 7\nThe better answer.") == (7.0, "The better answer.")
    assert school.parse_marked("**Mark: 4/10**\n\nBetter.\nSecond line.") == (4.0, "Better.\nSecond line.")
    assert school.parse_marked("Score: 15\nText")[0] == 10.0 and school.parse_marked("score 0\nText")[0] == 1.0
    assert school.parse_marked("Just a better answer with no mark.") == (None, "Just a better answer with no mark.")


def test_lessons_do_not_repeat_and_get_harder():
    seeds = {tuple(school.lesson_seed(i).values()) for i in range(2000)}
    assert len(seeds) == 2000
    assert [school.level_for(s) for s in (0, 1499, 1500, 5999, 6000)] == [1, 1, 2, 2, 3]
    assert school.LEVELS[1][0] < school.LEVELS[2][0] < school.LEVELS[3][0]


def test_a_pause_or_a_stumbling_teacher_costs_rows_not_the_session(cfg):
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] % 2:
            raise RuntimeError("the engine hiccuped")
        return "A scene that is long enough to keep, with a hook at the very end of it. " * 2

    res = school.session(cfg, [TEXT], want=6, pool=[_teacher(reply=flaky)])
    assert 0 < res["rows"] < 6 and res["teacher"] == "alpha"
    stopped = school.session(cfg, [TEXT], want=6, pool=[_teacher()], should_stop=lambda: True)
    assert stopped["rows"] == 0 and "gave nothing usable" in stopped["message"]
    assert school.session(cfg, [TEXT], want=6, pool=[])["message"].startswith("No teacher model is running")
    # With no text of the author's, there is nothing to set tasks on, but practice scenes can still be written.
    only = school.session(cfg, [], want=4, pool=[_teacher()])
    assert only["lessons"] == 4 and only["answers"] == 0


def test_a_teacher_is_only_shown_text_that_may_be_trained_on(cfg, db):
    from gauge.native.distill import sample_excerpts

    for source, title, word in (("royalroad", "Someone Else's Serial", "theirs"), ("local", "My Own Book", "mine")):
        n = NovelRecord(source=source, source_id=title, title=title)
        db.upsert_novel(n)
        db.upsert_chapter(ChapterRecord(novel_key=n.key, index=1, text=(word + " ") * 400))
    got = sample_excerpts(db, 12, random.Random(0))
    assert got and all("mine" in ex and "theirs" not in ex for ex in got)


# ------------------------------------------------------------------ the Studio and the command line
def test_training_from_the_studio_runs_a_school_session_first(cfg, db, tmp_path):
    from tests.test_teach import _novel_folder, _small_gaugelm

    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    assert ctl.teachers() == ["fake  -  on this computer, test model"]
    first = ctl.train_gaugelm(0.01)                               # not trained yet: answers and practice scenes
    assert first["lessons"] >= 4 and "fake taught it" in first["message"] and not _rows(cfg, "tutor")
    second = ctl.train_gaugelm(0.01)                              # now it exists, the tutor sets it tasks too
    assert second["lessons"] >= 4 and "correction" in second["message"] and _rows(cfg, "tutor")
    assert _rows(cfg, "tutor")[0]["mark"] == 4.0 and "The tutor's marks out of 10: 4.0 now" in ctl.report_card()["line"]
    assert "Taught by fake at the start of each training session" in ctl.school_line() and "marks out of 10" in ctl.school_line()
    cfg.native.tutor = False
    before = len(_rows(cfg, "tutor"))
    ctl.train_gaugelm(0.01)
    assert len(_rows(cfg, "tutor")) == before                     # switched off: no more attempts are marked


def test_school_commands(cfg, db, tmp_path):
    from click.testing import CliRunner

    from gauge.cli import main
    from tests.test_teach import _novel_folder, _small_gaugelm

    _small_gaugelm(cfg)
    cfg.save()
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    ctl.train_gaugelm(0.01, teach=False)
    runner = CliRunner(env={"COLUMNS": "200"})

    def run(*args):
        r = runner.invoke(main, ["--data-dir", cfg.data_dir, "native", *args])
        assert r.exit_code == 0, r.output
        return r.output

    assert "fake" in run("teachers") and "1 teacher(s)" in run("teachers")
    out = run("teach", "--rows", "9")
    assert "fake taught it" in out and "correction" in out and "native train --resume" in out
    report = run("report")
    assert "marks out of 10" in report and "at first" in report and "practice scenes" in report
    assert "correction" not in run("teach", "--rows", "4", "--no-tutor")
    run("teachers", "--use", "qwen2.5:7b, phi4")
    from gauge.config import GaugeConfig
    assert GaugeConfig.load(cfg.data_dir).native.teachers == ["qwen2.5:7b", "phi4"]
    run("teachers", "--use", "all")
    assert GaugeConfig.load(cfg.data_dir).native.teachers == []
