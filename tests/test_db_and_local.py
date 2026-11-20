import json

from gauge.ingest.local import import_path, split_single_file
from gauge.models import ChapterRecord, NovelRecord


def test_db_roundtrip(db):
    n = NovelRecord(source="local", source_id="s", title="S", followers=10, tags=["a"], genres=["b"])
    db.upsert_novel(n)
    db.upsert_chapter(ChapterRecord(novel_key=n.key, index=1, source_chapter_id="c1", text="word " * 300))
    db.upsert_chapter(ChapterRecord(novel_key=n.key, index=1, source_chapter_id="c1", text="again " * 300))
    assert db.chapter_count(n.key) == 1
    assert db.known_chapter_ids(n.key) == {"c1"}
    got = db.get_novel(n.key)
    assert got.title == "S" and got.tags == ["a"] and got.genres == ["b"]
    n.followers = 20
    db.upsert_novel(n)
    assert db.get_novel(n.key).followers == 20
    assert [h["followers"] for h in db.metric_history(n.key)] == [10, 20]
    db.save_features(n.key, "novel", {"x": 1.0})
    assert db.features("novel")[n.key] == {"x": 1.0}
    db.follow("http://example/1", "x")
    db.follow("http://example/1", "x")
    assert db.follows() == [("http://example/1", "x")]


def test_split_single_file():
    text = "Chapter 1: Start\n\nIt begins.\n\nChapter 2 - Middle\n\nIt continues here.\n\n# Chapter 3\n\nEnd."
    parts = split_single_file(text)
    assert [t for t, _ in parts] == ["Chapter 1: Start", "Chapter 2 - Middle", "Chapter 3"]
    assert parts[1][1] == "It continues here."


def test_import_folder_with_sidecar(tmp_path):
    d = tmp_path / "MyNovel"
    d.mkdir()
    for i in (1, 2, 10):
        (d / f"ch{i}.txt").write_text(f"Chapter body {i}. " * 50, encoding="utf-8")
    (d / "novel.json").write_text(json.dumps({"title": "My Novel", "followers": 500, "rating": 4.2,
                                              "genres": ["Fantasy"], "chapters": {"1": {"views": 99}}}))
    (imp,) = import_path(d)
    novel, chapters = imp.novel, imp.chapters
    assert novel.title == "My Novel" and novel.followers == 500 and novel.source == "local"
    assert [c.title for c in chapters] == ["ch1", "ch2", "ch10"]  # natural sort
    assert chapters[0].views == 99 and chapters[1].views is None
