"""Compile a training library the AI may lawfully learn from, with licence/provenance tracking."""
import json

from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.db import Database
from gauge.learn.corpus import compile_corpus, export_corpus, permitted_texts, training_permission
from gauge.models import ChapterRecord, NovelRecord

# A project chapter has to read as prose: a looped filler line is under the coherence floor and is left out.
MINE = ("My own words here, set down the night the ferry stopped running. Nobody on the island said why. "
        "The harbour master locked his office and walked home without his coat, and by morning the timetable "
        "had been taken off the wall. I asked at the bakery. The woman there gave me bread and no answer.\n\n") * 3


def _seed(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    db.upsert_novel(NovelRecord(source="gutenberg", source_id="1", title="Moby Dick", author="Melville"))
    db.upsert_chapter(ChapterRecord(novel_key="gutenberg:1", index=0, source_chapter_id="0", title="Loomings",
                                    text="Call me Ishmael. " * 50))
    db.upsert_novel(NovelRecord(source="royalroad", source_id="9", title="Scraped Serial", author="X"))
    db.upsert_chapter(ChapterRecord(novel_key="royalroad:9", index=0, source_chapter_id="0", title="Ch1",
                                    text="A scraped chapter. " * 50))
    from gauge.write.project import Project
    p = Project.create(cfg.projects_dir, "Mine", "idea", ["Sci-fi"], 300)
    p.write_chapter(1, "T", MINE, {"title": "T"})
    return cfg, db


def test_training_permission_by_source():
    assert training_permission("gutenberg")[1] is True and training_permission("gutenberg")[0] == "public_domain"
    assert training_permission("local")[1] is True and training_permission("project")[1] is True
    assert training_permission("royalroad")[1] is False and training_permission("wattpad")[1] is False  # copyrighted


def test_compile_excludes_third_party_by_default(tmp_path):
    cfg, db = _seed(tmp_path)
    rep = compile_corpus(db, cfg.projects_dir)
    srcs = {d.source for d in rep.docs}
    assert "gutenberg" in srcs and "project" in srcs and "royalroad" not in srcs
    assert {d.source for d in rep.excluded} == {"royalroad"}
    assert rep.est_tokens > 0 and all(d.permitted for d in rep.docs)


def test_compile_can_opt_in_to_third_party(tmp_path):
    cfg, db = _seed(tmp_path)
    rep = compile_corpus(db, cfg.projects_dir, include_third_party=True)
    assert "royalroad" in {d.source for d in rep.docs} and rep.excluded == []


def test_manifest_and_export(tmp_path):
    cfg, db = _seed(tmp_path)
    rep = compile_corpus(db, cfg.projects_dir)
    man = export_corpus(rep, tmp_path / "corpus")
    assert man["documents"] == len(rep.docs) and man["est_tokens"] > 0
    assert "gutenberg" in man["by_source"] and man["excluded"]["sources"] == ["royalroad"]
    assert (tmp_path / "corpus" / "corpus.txt").exists()
    jl = (tmp_path / "corpus" / "corpus.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in jl]
    assert all("permission" in r and "source" in r for r in rows)  # provenance recorded per document


def test_permitted_texts_shape(tmp_path):
    cfg, db = _seed(tmp_path)
    docs = permitted_texts(db, cfg.projects_dir)
    assert docs and all(isinstance(t, tuple) and t[0] == "prose" for t in docs)
    # The scraped royalroad text is not present.
    assert not any("scraped chapter" in text for _, text in docs)


def test_cli_corpus(tmp_path):
    cfg, _ = _seed(tmp_path)
    data = str(cfg.root)
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", data, "corpus", "stats"])
    assert r.exit_code == 0 and "Training library" in r.output and "gutenberg" in r.output and "excluded" in r.output
    r = runner.invoke(main, ["--data-dir", data, "corpus", "build", "--out", str(tmp_path / "out")])
    assert r.exit_code == 0 and "compiled" in r.output and (tmp_path / "out" / "MANIFEST.json").exists()
