"""Chapters saved before the coherence floor existed: an unreadable one is never training text, and the
author can find it. Nothing is deleted for them."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.learn.corpus import compile_corpus, export_corpus, permitted_texts
from gauge.learn.finetune import build_sft_from_project, unreadable_rows
from gauge.models import ChapterRecord, NovelRecord
from gauge.native import gather_texts, instruction_rows
from gauge.write.project import Project
from gauge.write.prose import COHERENCE_FLOOR, check_chapters
from tests.test_coherence import NOISE

# NOISE is what GaugeLM (tiny preset, ~67 steps) really wrote: accepted at 0.65 before the floor existed.
MARK = "areed`. Gauge:"  # a fragment only the noise contains
LOOP = "Call me Ishmael. " * 50  # under the floor too, but ingested text: not ours to judge


@pytest.fixture
def story(cfg, chapter_text):
    """Three saved chapters; the second is the word salad, written straight to disk as the old pipeline did."""
    p = Project.create(cfg.projects_dir, "Ten Seconds", "a courier who can rewind ten seconds", ["Sci-fi"], 300)
    p.write_chapter(1, "The Seal", chapter_text(True), {"title": "The Seal", "score": 0.71})
    p.write_chapter(2, "Static", NOISE, {"title": "Static", "score": 0.65})
    p.write_chapter(3, "The Tea", chapter_text(False), {"title": "The Tea", "score": 0.68})
    p.save()
    return p


def _files(root):
    return {str(f.relative_to(root)): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


def _files_under(files, sub):
    prefix = f"{sub}/"
    return {k[len(prefix):]: v for k, v in files.items() if k.startswith(prefix)}


def test_check_chapters_is_a_pure_measure(chapter_text):
    chapters = [(1, chapter_text(True)), (2, NOISE), (3, chapter_text(False))]
    checks = check_chapters(chapters)
    assert [(c.number, c.readable) for c in checks] == [(1, True), (2, False), (3, True)]
    bad = checks[1]
    assert bad.words == len(NOISE.split()) and bad.coherence.score < COHERENCE_FLOOR and bad.coherence.issues
    assert checks[0].coherence.score == 1.0 and checks[0].coherence.issues == []
    assert all(c.readable for c in check_chapters(chapters, floor=0.0))  # 0 = measure, never flag
    assert check_chapters([]) == []


def test_project_splits_saved_chapters_into_training_text_and_unreadable(story, chapter_text):
    assert story.chapter_numbers() == [1, 2, 3]
    assert story.chapter_prose(1) == chapter_text(True).strip() and story.chapter_prose(9) is None
    readable, unreadable = story.training_chapters()
    assert [n for n, _ in readable] == [1, 3] and [c.number for c in unreadable] == [2]
    assert not any(MARK in t for _, t in readable)
    everything, none = story.training_chapters(0.0)
    assert [n for n, _ in everything] == [1, 2, 3] and none == []


def test_gather_texts_skips_an_unreadable_saved_chapter_and_keeps_the_readable_ones(cfg, db, story, chapter_text):
    before = _files(cfg.root)
    docs = gather_texts(db, cfg.projects_dir, None)
    texts = [t for _, t in docs]
    assert len(docs) == 2 and chapter_text(True).strip() in texts and chapter_text(False).strip() in texts
    assert not any(MARK in t for t in texts)
    assert _files(cfg.root) == before  # skipped, not deleted or rewritten
    # the floor is the configured one: switched off, every saved chapter is training text again
    assert any(MARK in t for _, t in gather_texts(db, cfg.projects_dir, None, coherence_floor=0.0))


def test_gather_texts_does_not_judge_ingested_chapters(cfg, db, story):
    n = NovelRecord(source="gutenberg", source_id="1", title="Moby Dick")
    db.upsert_novel(n)
    db.upsert_chapter(ChapterRecord(novel_key=n.key, index=0, text=LOOP))
    assert ("prose", LOOP) in gather_texts(db, cfg.projects_dir, None)
    assert LOOP in [d.text for d in compile_corpus(db, cfg.projects_dir).docs]


def test_dataset_rows_made_from_an_unreadable_chapter_are_not_instruction_text(cfg, db, story, chapter_text):
    ds = cfg.models_dir / "datasets"
    ds.mkdir(parents=True)
    pair = {"prompt": "Chapter 2 of Ten Seconds: the seal breaks", "chosen": NOISE, "rejected": "an earlier draft",
            "project": "Ten Seconds", "chapter": 2}
    good = {"prompt": "Chapter 1 of Ten Seconds: the run", "chosen": chapter_text(True), "rejected": "an earlier draft",
            "project": "Ten Seconds", "chapter": 1}
    (ds / "self_preference.jsonl").write_text(json.dumps(good) + "\n" + json.dumps(pair) + "\n", encoding="utf-8")
    (ds / "author_edits.jsonl").write_text(json.dumps({"prompt": "p", "chosen": "kept", "rejected": "old"}) + "\n", encoding="utf-8")
    (ds / "sft.jsonl").write_text(json.dumps({"instruction": "write", "input": "x", "output": "the output"}) + "\n", encoding="utf-8")
    rows = instruction_rows(ds)
    answers = [a for _, a in rows]
    # the chapter-text row is held to the floor; short answers that are not chapters are left alone
    assert answers == ["the output", chapter_text(True), "kept"]
    assert NOISE in [a for _, a in instruction_rows(ds, 0.0)]
    assert not any(MARK in t for _, t in gather_texts(db, cfg.projects_dir, ds))
    found = unreadable_rows(ds, "Ten Seconds")
    assert [(r.file, r.line, r.chapter) for r in found] == [("self_preference.jsonl", 2, 2)] and found[0].coherence < COHERENCE_FLOOR
    assert unreadable_rows(ds, "Another Story") == [] and unreadable_rows(ds, "Ten Seconds", 0.0) == []


def test_corpus_leaves_an_unreadable_saved_chapter_out_and_says_so(cfg, db, story, chapter_text, tmp_path):
    rep = compile_corpus(db, cfg.projects_dir)
    assert [d.doc_id for d in rep.docs] == ["project:Ten Seconds#1", "project:Ten Seconds#3"]
    assert [d.doc_id for d in rep.unreadable] == ["project:Ten Seconds#2"]
    left_out = rep.unreadable[0]
    assert MARK in left_out.text and not left_out.permitted and "coherence" in left_out.note and rep.excluded == []
    texts = [t for _, t in permitted_texts(db, cfg.projects_dir)]
    assert texts == [chapter_text(True).strip(), chapter_text(False).strip()]
    manifest = export_corpus(rep, tmp_path / "out")
    assert manifest["documents"] == 2 and manifest["unreadable"] == {"documents": 1, "ids": ["project:Ten Seconds#2"]}
    assert MARK not in (tmp_path / "out" / "corpus.txt").read_text(encoding="utf-8")
    assert MARK not in (tmp_path / "out" / "corpus.jsonl").read_text(encoding="utf-8")
    off = compile_corpus(db, cfg.projects_dir, coherence_floor=0.0)
    assert len(off.docs) == 3 and off.unreadable == []


def test_corpus_scans_chapter_files_when_the_project_state_was_not_saved(cfg, db, chapter_text):
    p = Project.create(cfg.projects_dir, "Unsaved", "idea", ["Fantasy"], 300)
    p.write_chapter(1, "One", NOISE, {})
    p.write_chapter(2, "Two", chapter_text(True), {})  # no p.save(): project.json still says written = 0
    rep = compile_corpus(db, cfg.projects_dir)
    assert [d.doc_id for d in rep.docs] == ["project:Unsaved#2"] and [d.doc_id for d in rep.unreadable] == ["project:Unsaved#1"]


def test_build_sft_writes_no_row_for_an_unreadable_saved_chapter(cfg, story):
    ds = cfg.models_dir / "datasets"
    assert build_sft_from_project(story, ds) == 2
    rows = [json.loads(line) for line in (ds / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["chapter"] for r in rows] == [1, 3] and not any(MARK in r["completion"] for r in rows)
    assert build_sft_from_project(story, ds) == 0  # still idempotent; the unreadable one is not retried into the file
    assert build_sft_from_project(story, ds, coherence_floor=0.0) == 1  # the author's call, with the floor off
    assert unreadable_rows(ds, "Ten Seconds") and unreadable_rows(ds, "Ten Seconds")[0].file == "sft.jsonl"


# ------------------------------------------------------------------ the author finds them
def _cli_story(tmp_path, chapter_text, noise: bool = True):
    data = str(tmp_path / "d")
    runner = CliRunner(env={"COLUMNS": "200"})
    assert runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"]).exit_code == 0
    cfg = GaugeConfig.load(data)
    p = Project.create(cfg.projects_dir, "Ten Seconds", "a courier", ["Sci-fi"], 300)
    p.write_chapter(1, "The Seal", chapter_text(True), {"score": 0.71})
    p.write_chapter(2, "Static", NOISE if noise else chapter_text(False), {"score": 0.65})
    p.write_chapter(3, "The Tea", chapter_text(False), {"score": 0.68})
    p.save()
    return runner, data, cfg, p


def test_cli_chapters_check_lists_the_unreadable_chapter_and_deletes_nothing(tmp_path, chapter_text):
    runner, data, cfg, p = _cli_story(tmp_path, chapter_text)
    ds = cfg.models_dir / "datasets"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "self_preference.jsonl").write_text(json.dumps({"prompt": "Chapter 2", "chosen": NOISE, "rejected": "x",
                                                          "project": "Ten Seconds", "chapter": 2}) + "\n", encoding="utf-8")
    before = _files(cfg.root)
    r = runner.invoke(main, ["--data-dir", data, "chapters", "check", "Ten Seconds"])
    assert r.exit_code == 1, r.output
    assert "ch 2" in r.output and "coherence 0.00" in r.output and f"{len(NOISE.split())} words" in r.output
    assert str(p.chapter_path(2)) in r.output and "stray markup" in r.output
    assert "ch 1 " not in r.output and "ch 3 " not in r.output and str(p.chapter_path(1)) not in r.output
    assert "dataset row made from ch 2" in r.output and f"{ds / 'self_preference.jsonl'} line 1" in r.output
    assert "1 of 3 saved chapter(s) and 1 dataset row(s)" in r.output and "Nothing was changed" in r.output
    assert _files(cfg.root) == before and "Traceback" not in r.output

    r = runner.invoke(main, ["--data-dir", data, "chapters", "check", "Ten Seconds", "--all"])
    assert r.exit_code == 1 and "ch 1 " in r.output and "ch 3 " in r.output and r.output.count("coherence 1.00") == 2

    r = runner.invoke(main, ["--data-dir", data, "doctor"])
    assert r.exit_code == 0 and "Chapters" in r.output and "Ten Seconds" in r.output and "(ch 2)" in r.output
    assert "gauge chapters check" in r.output

    r = runner.invoke(main, ["--data-dir", data, "corpus", "stats"])
    assert r.exit_code == 0 and "left out" in r.output and "Ten Seconds#2" in r.output and "2 docs" in r.output

    r = runner.invoke(main, ["--data-dir", data, "build-sft", "Ten Seconds"])
    assert r.exit_code == 0 and "wrote 2" in r.output
    assert _files(cfg.projects_dir) == _files_under(before, cfg.projects_dir.relative_to(cfg.root))  # still nothing touched


def test_cli_chapters_check_is_quiet_about_a_readable_story(tmp_path, chapter_text):
    runner, data, cfg, p = _cli_story(tmp_path, chapter_text, noise=False)
    r = runner.invoke(main, ["--data-dir", data, "chapters", "check", "Ten Seconds"])
    assert r.exit_code == 0 and "all 3 saved chapter(s) read as prose" in r.output and "unreadable" not in r.output
    r = runner.invoke(main, ["--data-dir", data, "doctor"])
    assert r.exit_code == 0 and "Chapters" not in r.output
    r = runner.invoke(main, ["--data-dir", data, "chapters", "check", "No Such Story"])
    assert r.exit_code == 1 and "no project named" in r.output


def test_cli_follows_the_configured_floor(tmp_path, chapter_text):
    runner, data, cfg, p = _cli_story(tmp_path, chapter_text)
    assert runner.invoke(main, ["--data-dir", data, "settings", "set", "write.coherence_floor", "0"]).exit_code == 0
    r = runner.invoke(main, ["--data-dir", data, "chapters", "check", "Ten Seconds"])
    assert r.exit_code == 0 and "the coherence floor is off" in r.output and "--floor 0.5" in r.output
    r = runner.invoke(main, ["--data-dir", data, "chapters", "check", "Ten Seconds", "--floor", "0.5"])
    assert r.exit_code == 1 and "ch 2" in r.output  # measured on request even with the floor off
    r = runner.invoke(main, ["--data-dir", data, "doctor"])
    assert r.exit_code == 0 and "Chapters" not in r.output
    r = runner.invoke(main, ["--data-dir", data, "build-sft", "Ten Seconds"])
    assert r.exit_code == 0 and "wrote 3" in r.output  # floor off: the author keeps everything
    r = runner.invoke(main, ["--data-dir", data, "corpus", "stats"])
    assert r.exit_code == 0 and "left out" not in r.output and "3 docs" in r.output


def test_daemon_training_text_uses_the_configured_floor(cfg, db, story, monkeypatch):
    from gauge.daemon import Daemon

    trained_on = []
    monkeypatch.setattr("gauge.native.NativeTrainer.exists", lambda self: True)
    monkeypatch.setattr("gauge.native.NativeTrainer.continue_training",
                        lambda self, docs, *a, **k: trained_on.append([t for _, t in docs]) or {"docs": len(docs)})
    cfg.native.daemon_steps = 1
    assert Daemon(cfg, db).once(write=False)["native"] == {"docs": 2}
    assert not any(MARK in t for t in trained_on[-1])
    cfg.write.coherence_floor = 0.0  # the author's switch, honoured by the daemon too
    assert Daemon(cfg, db).once(write=False)["native"] == {"docs": 3}
    assert any(MARK in t for t in trained_on[-1])


def test_studio_training_text_uses_the_configured_floor(cfg, db, story, monkeypatch):
    from gauge.gui import StudioController

    trained_on = []

    def train(self, docs, *a, **k):
        trained_on.append([t for _, t in docs])
        return {"step": 1, "loss": 1.0, "seconds": 1.0}

    monkeypatch.setattr("gauge.native.NativeTrainer.train", train)
    ds = cfg.models_dir / "datasets"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "self_preference.jsonl").write_text(json.dumps({"prompt": "Chapter 2", "chosen": NOISE, "rejected": "x",
                                                          "project": "Ten Seconds", "chapter": 2}) + "\n", encoding="utf-8")
    ctl = StudioController(cfg, db)
    assert ctl.train_gaugelm(minutes=0.01, teach=False)["documents"] == 2
    assert not any(MARK in t for t in trained_on[-1])
    cfg.write.coherence_floor = 0.0
    assert ctl.train_gaugelm(minutes=0.01, teach=False)["documents"] == 4  # three chapters and the pair
    assert any(MARK in t for t in trained_on[-1])
