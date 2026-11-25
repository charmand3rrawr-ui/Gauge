"""Chapters of the length asked for, and measuring what GaugeLM understands."""
import json

import numpy as np
import pytest

from gauge.gui import StudioController
from gauge.llm import FakeLLM
from gauge.native import GaugeLM, ModelConfig, NativeTrainer, TrainConfig, school, understand
from gauge.write.project import Project


# ------------------------------------------------------------------ chapters as long as asked
class _Short(FakeLLM):
    """A model that always stops after about sixty words, however many it is asked for."""

    def __init__(self):
        super().__init__()
        self.kinds: list[str] = []

    def generate(self, prompt, **kw):
        kind = kw.get("purpose", "")
        self.kinds.append(kind)
        if kind in ("draft", "draft:continue", "revise"):
            n = self.kinds.count("draft:continue")
            return " ".join(f"stretch{n}word{i}" for i in range(60))
        return super().generate(prompt, **kw)


def _studio(cfg, db, llm):
    ctl = StudioController(cfg, db)
    pipe = ctl.pipeline()
    pipe.llm = pipe.fast_llm = llm
    for part in ("planner", "drafter"):
        getattr(pipe, part).llm = llm
    return ctl


def test_a_chapter_is_carried_on_until_it_is_as_long_as_asked(cfg, db):
    llm = _Short()
    ctl = _studio(cfg, db, llm)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    info = ctl.write_next(name, 300)
    assert info["words_asked"] == 300 and 270 <= info["words"] <= 400, info       # within a tenth of what was asked
    assert llm.kinds.count("draft") == 1 and llm.kinds.count("draft:continue") >= 4
    p = Project(cfg.projects_dir, name)
    text = p.read_chapter(1)
    assert "stretch0word0" in text and "stretch3word0" in text                     # the stretches are joined, in order
    assert p.read_meta(1)["words_asked"] == 300 and p.state.target_chapter_words == 300   # the story remembers
    assert ctl.chapter_words(name) == 300

    cfg.write.reach_length = False                                                 # switched off: it stops where the model stops
    llm.kinds.clear()
    short = ctl.write_next(name)
    assert short["words"] < 150 and "draft:continue" not in llm.kinds


def test_the_continuation_says_where_it_is_and_whether_to_finish(cfg, db):
    from gauge.write.drafter import Drafter

    seen: list[str] = []

    class _Watch(_Short):
        def generate(self, prompt, **kw):
            if kw.get("purpose") == "draft:continue":
                seen.append(prompt)
            return super().generate(prompt, **kw)

    ctl = _studio(cfg, db, _Watch())
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    pipe = ctl.pipeline()
    p = pipe.open_project(name)
    plan = pipe.planner.ensure_plan(p, 1)
    d = Drafter(pipe.llm, None, "")
    text, used = d.reach_length("The bag hummed. " * 10, 200, p, plan)
    assert used >= 2 and len(text.split()) >= 180
    assert "It has 30 words so far and" in seen[0] and "ENDS WITH" in seen[0] and "The bag hummed." in seen[0]
    assert "This stretch finishes the chapter" in seen[0]                          # 170 more fits in one stretch
    assert d.reach_length("Long enough already. " * 100, 200, p, plan) == ("Long enough already. " * 100, 0)

    class _Silent(_Short):
        def generate(self, prompt, **kw):
            return "" if kw.get("purpose") == "draft:continue" else super().generate(prompt, **kw)

    stuck, tries = Drafter(_Silent(), None, "").reach_length("Short.", 500, p, plan)
    assert stuck == "Short." and tries == 1                                        # a model with nothing to add ends it


def test_the_words_box_reads_what_people_type():
    read = StudioController.read_words
    assert (read("1500"), read(" 1,500 words "), read("2k"), read("2.5K"), read(800)) == (1500, 1500, 2000, 2500, 800)
    for bad, why in (("lots", "as a number"), ("", "as a number"), ("50", "from 100 to 20,000"), ("99999", "from 100 to 20,000")):
        with pytest.raises(ValueError) as e:
            read(bad)
        assert why in str(e.value), bad


def test_words_from_the_command_line_and_the_window(cfg, db, tmp_path, monkeypatch):
    import sys

    from click.testing import CliRunner

    from gauge import gui
    from gauge.cli import main
    from tests.test_teach import _Entry, _fake_tkinter, _wait_for_jobs, _Widget

    cfg.save()
    runner = CliRunner(env={"COLUMNS": "200"})
    runner.invoke(main, ["--data-dir", cfg.data_dir, "new", "Ten Seconds", "--idea", "a courier", "--genre", "Sci-fi", "--words", "200"])
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "write", "Ten Seconds", "--words", "350"])
    assert r.exit_code == 0 and "will be about 350 words" in r.output and " words  score=" in r.output, r.output
    assert Project(cfg.projects_dir, "Ten Seconds").state.target_chapter_words == 350
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "write", "Ten Seconds", "--words", "5"])
    assert r.exit_code != 0                                                        # out of range is refused

    mods, shown = _fake_tkinter(tmp_path)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    box = next(e.kw["textvariable"] for e in _Entry.made if e.kw.get("width") == 8)    # the words box
    assert box.get() == str(cfg.write.target_chapter_words)
    mods["tkinter.simpledialog"].askstring = lambda *a, **k: "Ten Seconds"
    _Widget.buttons["Open another story"]()
    assert box.get() == "350"                                                      # the story's own number is shown
    box.set("abc")
    _Widget.buttons["WRITE THE NEXT CHAPTER"]()
    assert shown[-1][0] == "info" and "as a number" in shown[-1][1][1]
    box.set("250")
    _Widget.buttons["WRITE THE NEXT CHAPTER"]()
    _wait_for_jobs()
    assert Project(cfg.projects_dir, "Ten Seconds").state.target_chapter_words == 250
    assert not [s for s in shown if s[0] == "error"], shown


# ------------------------------------------------------------------ what GaugeLM understands
WORDS = "the courier ran through the rain and the seal broke again while the footsteps waited behind her".split()


def _trained(tmp_path, steps=60):
    rng = np.random.default_rng(3)
    text = " ".join(rng.choice(WORDS, size=5000)) + " " + ("the lantern guttered and the ferryman muttered about the lantern again. " * 40)
    tr = NativeTrainer(tmp_path / "native")
    tr.train([("prose", text)], ModelConfig(vocab_size=320, n_layer=1, n_head=2, n_embd=32, n_kv_head=2, block_size=32),
             TrainConfig(steps=steps, batch_size=4, accum_steps=1, lr=2e-3, warmup=5, eval_every=30, checkpoint_every=30,
                         device="cpu", curriculum=False))
    model, tok = tr.load()
    return model, tok, text


def test_context_use_compares_seeing_the_passage_with_seeing_four_tokens(tmp_path):
    model, tok, text = _trained(tmp_path)
    c = understand.context_use(model, tok, text, samples=48)
    assert c["ok"] and c["tokens_of_context"] == 31 and c["samples"] == 48
    assert c["choices_with_context"] == pytest.approx(np.exp(c["loss_with_context"]))
    assert 0 <= c["first_guess_right"] <= c["in_top_five"] <= 1
    assert c["gain"] == pytest.approx(1 - c["loss_with_context"] / c["loss_without"])
    assert not understand.context_use(model, tok, "too short")["ok"]
    # A model that really cannot see back scores the same either way: the measure is zero when context is no use.
    blind = GaugeLM(ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=2, n_embd=32, n_kv_head=2, block_size=32), seed=0)
    for k in blind.p:
        if k.endswith(".wo"):
            blind.p[k][:] = 0.0                                                    # attention writes nothing
    assert abs(understand.context_use(blind, tok, text, samples=48)["gain"]) < 1e-6


def test_hard_words_are_the_ordinary_words_it_is_most_surprised_by(tmp_path):
    model, tok, text = _trained(tmp_path)
    sample = ("The Courier ran. " + "the lantern guttered and the ferryman muttered about the lantern again. " * 3
              + "a zyxwvut appeared and the zyxwvut was strange. the rain and the rain.")
    hard = understand.hard_words(model, tok, sample, n=5)
    words = [h["word"] for h in hard]
    assert words[0] == "zyxwvut" and hard[0]["seen"] == 2 and hard[0]["pieces"] > 1     # never seen in training
    assert "zyxwvut" in hard[0]["sentence"] and hard[0]["surprise"] > hard[-1]["surprise"]
    assert "Courier" not in words and all(w.islower() and len(w) >= 4 for w in words)   # names and short words are left out
    assert "zyxwvut" not in [h["word"] for h in understand.hard_words(model, tok, sample, n=5, skip={"zyxwvut"})]
    assert understand.hard_words(model, tok, "hi", n=5) == []


def test_alike_words_come_from_the_embeddings(tmp_path):
    model, tok, _ = _trained(tmp_path)
    res = understand.neighbours(model, tok, "courier", 4)
    assert res["ok"] and len(res["alike"]) == 4 and all(-1 <= a["closeness"] <= 1 for a in res["alike"])
    assert "courier" not in [a["word"] for a in res["alike"]]
    split = understand.neighbours(model, tok, "zyxwvut")
    assert not split["ok"] and "pieces, not one word" in split["reason"] and "grow --vocab" in split["reason"]


def test_the_report_reads_as_sentences(tmp_path):
    model, tok, text = _trained(tmp_path)
    rep = understand.report(model, tok, text, words=3)
    assert "instead of only the last 4 tokens" in rep["line"] and "first guess for the next piece is right" in rep["line"]
    assert len(rep["hard_words"]) <= 3 and ("words it is most lost on" in rep["line"]) == bool(rep["hard_words"])


def test_the_school_teaches_the_words_gaugelm_is_lost_on(cfg, db, tmp_path):
    from gauge.llm import make_llm
    from tests.test_teach import _novel_folder, _small_gaugelm

    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    ctl.train_gaugelm(0.01, teach=False)
    student = make_llm(cfg.llm.model_copy(update={"provider": "native", "model": "gaugelm", "native_dir": str(cfg.native_dir)}), "main")
    from gauge.learn.corpus import permitted_texts

    texts = [t for _, t in permitted_texts(db, cfg.projects_dir)]
    res = school.session(cfg, texts, want=8, student=student)
    assert res["words"] == 2 and "words it did not know, explained" in res["message"], res
    rows = [json.loads(line) for line in (cfg.models_dir / "datasets" / "words.jsonl").read_text().splitlines()]
    assert rows[0]["task"] == "word" and rows[0]["word"] in rows[0]["prompt"] and rows[0]["word"] in rows[0]["response"]
    assert rows[0]["surprise"] > 0 and rows[0]["teacher"] == "fake"
    again = school.session(cfg, texts, want=8, student=student)                    # the same words are not taught twice
    taught = [json.loads(line)["word"] for line in (cfg.models_dir / "datasets" / "words.jsonl").read_text().splitlines()]
    assert len(taught) == len(set(taught)) and again["words"] >= 0
    assert school.report_card(cfg.models_dir / "datasets")["counts"]["words"] == len(taught)
    # The trainer reads them.
    from gauge.native import instruction_rows
    assert any('mean in this passage' in kind for kind, _ in instruction_rows(cfg.models_dir / "datasets"))


def test_understanding_from_the_studio_and_the_command_line(cfg, db, tmp_path):
    from click.testing import CliRunner

    from gauge.cli import main
    from tests.test_teach import _novel_folder, _small_gaugelm

    _small_gaugelm(cfg)
    cfg.save()
    ctl = StudioController(cfg, db)
    assert "not been trained yet" in ctl.understanding()["message"]
    ctl.add_material([str(_novel_folder(tmp_path))])
    ctl.train_gaugelm(0.01, teach=False)
    got = ctl.understanding()
    assert got["ok"] and "instead of only the last 4 tokens" in got["message"]
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "native", "understand", "--words", "3"])
    assert r.exit_code == 0 and "context makes it" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "native", "alike", "zzzzqqqq"])
    assert r.exit_code == 0 and "pieces, not one word" in r.output
