import json

from gauge.learn import Trainer
from gauge.llm import FakeLLM, HashingEmbedder
from gauge.models import ChapterRecord, NovelRecord
from gauge.write import WritePipeline
from gauge.write.drafter import clean_prose


def _seed_corpus(db, chapter_text, n_novels=12, chapters_each=3):
    for i in range(n_novels):
        hooky = i % 2 == 0
        followers = 20000 if hooky else 300
        n = NovelRecord(source="local", source_id=f"n{i}", title=f"Novel {i}", followers=followers,
                        rating=4.5 if hooky else 3.2, tags=["LitRPG"] if hooky else ["Slice of Life"],
                        blurb="What if?" if hooky else "A quiet tale.")
        db.upsert_novel(n)
        for c in range(1, chapters_each + 1):
            db.upsert_chapter(ChapterRecord(novel_key=n.key, index=c, source_chapter_id=str(c),
                                            title=f"Chapter {c}: T", text=chapter_text(hooky)))


def test_training_cycle_learns_hook_and_cliff(cfg, db, chapter_text):
    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    out = t.cycle()
    assert out["chapters_featurized"] == 36
    assert isinstance(out["model"], dict) and out["model"]["n"] == 12
    m = t.load_model()
    imps = dict(m.importances(top=100))
    # Hooky novels are the popular ones, so hook/cliff features should carry positive weight.
    assert imps["mean_hook"] > 0 and imps["mean_cliff"] > 0
    assert out["exemplars"] > 0
    lib = t.load_exemplars()
    found = lib.search("a door opens and a name is written", kind="ending", k=2)
    assert len(found) == 2 and all(e.kind == "ending" for e in found)
    guide = t.load_guide()
    assert guide.startswith("# What the data says")
    assert (cfg.models_dir / "Modelfile").exists()
    assert (cfg.models_dir / "datasets" / "preference.jsonl").exists()
    assert out["preference_pairs"] > 0
    # analyze again should be incremental
    assert t.analyze()["chapters_featurized"] == 0


def test_write_pipeline_end_to_end(cfg, db, chapter_text):
    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    llm = FakeLLM()
    pipe = WritePipeline(cfg, db, llm=llm, trainer=t)
    p = pipe.new_project("Ten Seconds", "a courier who can rewind time", ["Sci-fi", "Thriller"], words=400)
    assert p.state.bible.premise
    assert len(p.state.outline) == 3  # FakeLLM returns 3 chapters per outline call
    res = pipe.write_next(p)
    assert res.number == 1 and res.path.exists()
    body = res.path.read_text(encoding="utf-8")
    assert body.startswith("---") and "# Chapter 1:" in body
    assert 0.0 <= res.score <= 1.0
    # the FakeLLM summary call returns a premise-ish JSON without 'summary'; fallback must still run
    assert p.state.running_summary
    res2 = pipe.write_next(p)
    assert res2.number == 2
    st = pipe.status(p)
    assert st["written"] == 2 and st["avg_score"] is not None
    # chapter 4 requires extending the outline (ensure_plan path)
    pipe.write_next(p)
    r4 = pipe.write_next(p)
    assert r4.number == 4 and p.plan_for(4) is not None
    # project reloads from disk
    from gauge.write.project import Project

    again = Project(cfg.projects_dir, "Ten Seconds")
    assert again.state.written == 4
    assert json.loads(again.state_path.read_text())["name"] == "Ten Seconds"


def test_clean_prose_strips_headings_and_notes():
    raw = "# Chapter 3: The Door\n\nProse starts here.\n\nMore prose.\n\n---\nAuthor's note: thanks!"
    assert clean_prose(raw) == "Prose starts here.\n\nMore prose."


def test_critic_without_model(cfg, db, chapter_text):
    from gauge.write.critic import Critic
    from gauge.write.project import ChapterPlan

    c = Critic(FakeLLM(), None)
    good = c.critique(chapter_text(True), ChapterPlan(number=1))
    bad = c.critique(chapter_text(False), ChapterPlan(number=1))
    assert good.measured > bad.measured
    assert bad.measured_issues
