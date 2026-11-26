"""The starter training rows that ship with Gauge."""
import json

from gauge.gui import StudioController
from gauge.native import instruction_rows, school, seed_lessons
from gauge.native.distill import TASKS


def test_every_row_is_well_formed_and_labelled():
    rows = seed_lessons.rows()
    info = seed_lessons.summary()
    assert len(rows) == info["rows"] == 70 and 4000 < info["words"] < 7000
    assert info["by_task"]["lesson"] == 36 and info["by_task"]["word"] == 10
    assert all(info["by_task"][kind] == 3 for kind, _t, _w in TASKS)              # each of Gauge's eight tasks, three times
    for r in rows:
        assert r["prompt"].strip() and r["response"].strip() and r["teacher"] == seed_lessons.SOURCE and r["licence"]
    assert len({r["prompt"] for r in rows}) == 70 and len({r["response"] for r in rows}) == 70   # nothing repeated
    assert "Claude" in seed_lessons.SOURCE and "Anthropic" in seed_lessons.SOURCE   # where it came from is on every row


def test_prompts_are_exactly_the_ones_the_school_uses():
    rows = seed_lessons.rows()
    lessons = [r for r in rows if r["task"] == "lesson"]
    for (i, _scene), r in zip(sorted(seed_lessons.SCENES.items()), lessons):
        seed = school.lesson_seed(i)
        assert r["prompt"] == school.LESSON_TASK.format(**seed)
        # The scene is about what its prompt says: the place or the event's key word turns up in it.
        subject = {w.strip(",.").lower()[:4] for w in (seed["where"] + " " + seed["what"]).split() if len(w) > 3}
        assert subject & {w.strip(",.\"").lower()[:4] for w in r["response"].split()}, (i, seed)
    templates = {kind: (t, w) for kind, t, w in TASKS}
    for key, kind, answer in seed_lessons.ANSWERS:
        t, w = templates[kind]
        row = next(r for r in rows if r["response"] == answer)
        assert row["prompt"] == t.format(excerpt=seed_lessons.EXCERPTS[key], n=w)
    for word, sentence, answer in seed_lessons.WORDS:
        row = next(r for r in rows if r.get("word") == word)
        assert row["prompt"] == school.WORD_TASK.format(word=word, sentence=sentence)
        assert word in sentence and answer.startswith(f'Here "{word}" means') and answer.count("\n") == 3


def test_answers_have_the_shape_each_task_asks_for():
    by = {}
    for _key, kind, answer in seed_lessons.ANSWERS:
        by.setdefault(kind, []).append(answer)
    assert all(1 <= len(a.split()) <= 5 for a in by["title"])
    for a in by["score"]:
        lines = a.splitlines()
        assert [ln.split(":")[0] for ln in lines] == ["hook", "momentum", "cliffhanger", "Advice"]
        assert all(1 <= int(ln.split(":")[1]) <= 10 for ln in lines[:3])
    assert all([ln.split(":")[0] for ln in a.splitlines()] == ["Title", "Hook", "Turn", "Cliffhanger"] for a in by["outline"])
    assert all([ln.split(":")[0] for ln in a.splitlines()] == ["Wants", "Fears", "Would refuse"] for a in by["brief"])
    assert all(len(a.splitlines()) == 6 and all(ln.startswith('"') for ln in a.splitlines()) for a in by["dialogue"])
    assert all(100 <= len(a.split()) <= 240 for a in by["continue"]) and all(90 <= len(a.split()) <= 150 for a in by["opening"])
    assert all(40 <= len(s.split()) <= 110 for s in seed_lessons.SCENES.values())


def test_it_is_off_until_switched_on_and_leaves_no_trace_when_off(cfg, db, tmp_path):
    from tests.test_teach import _novel_folder, _small_gaugelm

    datasets = cfg.models_dir / "datasets"
    assert cfg.native.seed_data is False and seed_lessons.sync(datasets, False) == 0 and not (datasets / seed_lessons.FILE).exists()
    assert seed_lessons.sync(datasets, True) == 70
    rows = [json.loads(line) for line in (datasets / seed_lessons.FILE).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 70 and len(instruction_rows(datasets)) == 70
    assert seed_lessons.sync(datasets, False) == 0 and not (datasets / seed_lessons.FILE).exists()
    assert instruction_rows(datasets) == []

    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    off = ctl.train_gaugelm(0.01, teach=False)
    assert not (datasets / seed_lessons.FILE).exists()
    ctl.set_switch("native.seed_data", True)
    on = ctl.train_gaugelm(0.01, teach=False)
    assert (datasets / seed_lessons.FILE).exists() and on["examples"] >= off["examples"] + 60
    ctl.set_switch("native.seed_data", False)
    ctl.train_gaugelm(0.01, teach=False)
    assert not (datasets / seed_lessons.FILE).exists()                           # switched off: taken away again


def test_the_seed_command(cfg):
    from click.testing import CliRunner

    from gauge.cli import main
    from gauge.config import GaugeConfig

    cfg.save()
    runner = CliRunner(env={"COLUMNS": "200"})

    def run(*args):
        r = runner.invoke(main, ["--data-dir", cfg.data_dir, "native", "seed", *args])
        assert r.exit_code == 0, r.output
        return r.output

    out = run()
    assert "70 starter rows" in out and "Not in use" in out and "Anthropic's terms" in out and "your decision" in out
    out = run("--use", "on", "--show", "1")
    assert "In use" in out and "a mapmaker in a workshop full of clocks" in out and "green door" in out
    assert GaugeConfig.load(cfg.data_dir).native.seed_data is True
    assert "Not in use" in run("--use", "off") and GaugeConfig.load(cfg.data_dir).native.seed_data is False
