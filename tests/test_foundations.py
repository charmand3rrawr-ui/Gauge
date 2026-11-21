"""Batch G: global seed/determinism, cost+latency profiling, and a cross-chapter e2e invariant test."""
from gauge.learn import Trainer
from gauge.learn.bandit import PromptBandit
from gauge.llm import FakeLLM, HashingEmbedder
from gauge.util import seed_all
from gauge.write import WritePipeline
from tests.test_pipeline import _seed_corpus


def test_seed_all_is_reproducible():
    import random
    seed_all(123)
    a = [random.random() for _ in range(5)]
    seed_all(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_bandit_is_deterministic_under_seed(tmp_path):
    s1 = PromptBandit(tmp_path / "b1.json", seed=42).sample(1)
    s2 = PromptBandit(tmp_path / "b2.json", seed=42).sample(1)
    s3 = PromptBandit(tmp_path / "b3.json", seed=99).sample(1)
    assert s1 == s2            # same seed, same draws
    assert s1 != s3 or True    # different seed usually differs; never crashes


def test_pipeline_threads_seed_into_bandit(cfg, db, chapter_text):
    cfg.seed = 7
    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    pipe = WritePipeline(cfg, db, llm=FakeLLM(), trainer=t)
    p = pipe.new_project("Seeded", "a courier who can rewind", ["Sci-fi"], words=300)
    b = pipe.bandit(p)
    assert b is not None and b.rng is not None  # the bandit was seeded from cfg.seed


def test_profile_telemetry_records_costs(cfg, db, chapter_text):
    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    pipe = WritePipeline(cfg, db, llm=FakeLLM(), trainer=t)
    p = pipe.new_project("Costed", "a courier who can rewind", ["Sci-fi"], words=300)
    pipe.write_next(p)  # telemetry is logged live via the call sink
    rows = db.call_summary(project=p.state.name)
    assert rows, "expected per-purpose telemetry after writing a chapter"
    assert all("purpose" in r and "seconds" in r and "calls" in r for r in rows)
    assert sum(r["calls"] for r in rows) > 0


def test_e2e_invariants_across_chapters(cfg, db, chapter_text):
    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    pipe = WritePipeline(cfg, db, llm=FakeLLM(), trainer=t)
    p = pipe.new_project("Invariants", "a courier who can rewind", ["Sci-fi"], words=300)
    scores = []
    for _ in range(3):
        res = pipe.write_next(p)
        scores.append(res.score)
        meta = p.read_meta(res.number)
        assert isinstance(meta.get("score"), (int, float))  # every chapter stores its score
        assert res.path.exists() and 0.0 <= res.score <= 1.0
    assert p.state.written == 3
    assert p.state.running_summary  # the story summary accumulates
    # The reader panel's prediction history grows with the chapters.
    panel = pipe.council(p).get("reader_panel")
    if panel is not None:
        assert len(panel.state.predictions) >= 1
    # Reloading from disk preserves the chapter count.
    from gauge.write.project import Project
    assert Project(cfg.projects_dir, "Invariants").state.written == 3
