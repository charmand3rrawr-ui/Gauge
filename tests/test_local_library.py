"""Learning from local folders: libraries, EPUB/DOCX, metadata inference, incremental re-scans."""
import csv
import json
import zipfile
from pathlib import Path

from gauge.ingest import IngestRunner
from gauge.ingest.local import (
    import_path,
    infer_from_name,
    is_library,
    normalize_meta,
    read_docx,
    read_epub,
    split_single_file,
)
from gauge.learn import Trainer
from gauge.llm import HashingEmbedder


def _prose(n_words: int, seed: str = "x") -> str:
    base = ("\"Run,\" she said. The blade was already in the door before I understood what she meant. "
            "Someone screamed below us. We went right.\n\n")
    out, w = [], 0
    while w < n_words:
        out.append(base)
        w += len(base.split())
    return "".join(out) + f"\n\nThe door opened on its own, {seed}."


def make_epub(path: Path, title: str, author: str, chapters, subjects=("Fantasy",)) -> Path:
    manifest = "".join(f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>' for i in range(len(chapters)))
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
    subj = "".join(f"<dc:subject>{s}</dc:subject>" for s in subjects)
    opf = f'''<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
<metadata><dc:title>{title}</dc:title><dc:creator>{author}</dc:creator>{subj}<dc:description>A test blurb.</dc:description></metadata>
<manifest>{manifest}</manifest><spine>{spine}</spine></package>'''
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf", opf)
        for i, (h, body) in enumerate(chapters):
            paras = "".join(f"<p>{p}</p>" for p in body.split("\n\n"))
            z.writestr(f"OEBPS/c{i}.xhtml", f"<html><body><h1>{h}</h1>{paras}</body></html>")
    return path


def make_docx(path: Path, title: str, paragraphs) -> Path:
    """paragraphs: list of (style_or_None, text)."""
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = ""
    for style, text in paragraphs:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body += f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"
    doc = f'<?xml version="1.0"?><w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>'
    core = f'''<?xml version="1.0"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title><dc:creator>Doc Author</dc:creator></cp:coreProperties>'''
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", doc)
        z.writestr("docProps/core.xml", core)
    return path


# ------------------------------------------------------------------ splitting & metadata
def test_split_handles_words_roman_markdown_and_pseudo():
    text = "Chapter One\n\n" + _prose(200, "a") + "\n\nChapter Two\n\n" + _prose(200, "b") + "\n\nCHAPTER III\n\n" + _prose(200, "c")
    parts = split_single_file(text)
    assert [t for t, _ in parts] == ["Chapter One", "Chapter Two", "CHAPTER III"]
    md = "## The Door\n\n" + _prose(150) + "\n\n## The Hall\n\n" + _prose(150)
    assert [t for t, _ in split_single_file(md)] == ["The Door", "The Hall"]
    long_flat = _prose(7000)
    parts = split_single_file(long_flat, pseudo_words=2500)
    assert 2 <= len(parts) <= 4 and parts[0][0] == "Part 1"
    assert split_single_file("Chapter 1\n\nshort")[0][1] == "short"


def test_front_matter_becomes_prologue_when_long():
    text = _prose(500, "lead") + "\n\nChapter 1\n\n" + _prose(200) + "\n\nChapter 2\n\n" + _prose(200)
    parts = split_single_file(text)
    assert parts[0][0] == "Prologue" and len(parts) == 3


def test_infer_from_name_and_normalize_meta():
    assert infer_from_name("Jane Doe - The Long Night (v2)") == {"title": "The Long Night", "author": "Jane Doe"}
    assert infer_from_name("The Long Night by Jane Doe") == {"title": "The Long Night", "author": "Jane Doe"}
    assert infer_from_name("my_quiet_serial") == {"title": "My Quiet Serial"}
    m = normalize_meta({"Followers": "12,000", "genre": "Fantasy; Romance", "stars": "4.5", "Description": "x", "junk": None})
    assert m == {"followers": 12000, "genres": ["Fantasy", "Romance"], "user_rating": 4.5, "blurb": "x"}


# ------------------------------------------------------------------ formats
def test_epub_without_ebooklib(tmp_path):
    ep = make_epub(tmp_path / "Book.epub", "Ash and Salt", "R. Writer",
                   [("Chapter 1: Gates", _prose(300, "a")), ("Chapter 2: Walls", _prose(300, "b")), ("Copyright", "All rights reserved.")],
                   subjects=("Fantasy", "Cultivation"))
    meta, docs = read_epub(ep)
    assert meta["title"] == "Ash and Salt" and meta["author"] == "R. Writer" and meta["genres"] == ["Fantasy", "Cultivation"]
    assert len(docs) == 3
    (imp,) = import_path(ep)
    assert imp.novel.title == "Ash and Salt" and imp.novel.author == "R. Writer" and imp.novel.blurb == "A test blurb."
    assert [c.title for c in imp.chapters] == ["Chapter 1: Gates", "Chapter 2: Walls"]  # copyright page dropped
    assert "Chapter 1: Gates" not in imp.chapters[0].text


def test_docx_single_book(tmp_path):
    d = make_docx(tmp_path / "draft.docx", "Draft Title", [
        ("Heading1", "Chapter 1"), (None, _prose(200, "a")), ("Heading1", "Chapter 2"), (None, _prose(200, "b"))])
    meta, text = read_docx(d)
    assert meta["title"] == "Draft Title" and text.startswith("# Chapter 1")
    (imp,) = import_path(d)
    assert imp.novel.title == "Draft Title" and imp.novel.author == "Doc Author"
    assert len(imp.chapters) == 2 and imp.chapters[1].title == "Chapter 2"


def test_cp1252_and_html_chapter_files(tmp_path):
    d = tmp_path / "Novel"
    d.mkdir()
    (d / "01.txt").write_bytes(("Chapter 1\n\n" + _prose(200, "café")).encode("cp1252"))
    (d / "02.html").write_text(f"<html><body><h2>The Hall</h2><p>{_prose(200)}</p><script>x()</script></body></html>", encoding="utf-8")
    (d / "README.md").write_text("notes about the novel", encoding="utf-8")
    (d / "notes.txt").write_text("too short", encoding="utf-8")
    (imp,) = import_path(d)
    assert len(imp.chapters) == 2 and "café" in imp.chapters[0].text
    assert imp.chapters[1].title == "The Hall" and "x()" not in imp.chapters[1].text
    assert not imp.warnings  # README/notes are skipped by name, not warned about


# ------------------------------------------------------------------ libraries
def _make_library(root: Path) -> Path:
    lib = root / "Library"
    lib.mkdir()
    a = lib / "Ann Author - First Serial"
    a.mkdir()
    for i in range(1, 4):
        (a / f"chapter_{i}.txt").write_text(_prose(250, f"a{i}"), encoding="utf-8")
    b = lib / "second_serial"
    b.mkdir()
    (b / "novel.json").write_text(json.dumps({"title": "Second Serial", "followers": 3000, "tags": ["LitRPG"]}))
    for i in range(1, 3):
        (b / f"c{i}.md").write_text(f"# Chapter {i}\n\n" + _prose(250, f"b{i}"), encoding="utf-8")
    make_epub(lib / "Third.epub", "Third Book", "E. Pub", [("One", _prose(250, "c1")), ("Two", _prose(250, "c2"))])
    nested = lib / "Completed"
    nested.mkdir()
    c = nested / "fourth"
    c.mkdir()
    (c / "1.txt").write_text(_prose(250, "d1"), encoding="utf-8")
    (c / "2.txt").write_text(_prose(250, "d2"), encoding="utf-8")
    (lib / "Whole Book by Some One.txt").write_text(
        "\n\n".join(f"Chapter {i}\n\n" + _prose(200, f"w{i}") for i in range(1, 4)), encoding="utf-8")
    (lib / "stray_note.txt").write_text("just a short note, not a novel", encoding="utf-8")
    with (lib / "library.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "followers", "rating", "stars", "genres"])
        wr.writerow(["Ann Author - First Serial", "15000", "4.6", "5", "Fantasy|Action"])
        wr.writerow(["Third.epub", "", "", "2", ""])
    return lib


def test_library_detection_and_import(tmp_path):
    lib = _make_library(tmp_path)
    assert is_library(lib) and not is_library(lib / "second_serial") and not is_library(lib / "Ann Author - First Serial")
    assert is_library(lib / "Completed")
    imps = import_path(lib)
    by_title = {i.novel.title: i for i in imps}
    assert set(by_title) == {"First Serial", "Second Serial", "Third Book", "Fourth", "Whole Book"}
    assert by_title["Whole Book"].novel.author == "Some One" and len(by_title["Whole Book"].chapters) == 3
    first = by_title["First Serial"].novel
    assert first.author == "Ann Author" and first.followers == 15000 and first.rating == 4.6
    assert first.user_rating == 5 and first.genres == ["Fantasy", "Action"]
    assert by_title["Second Serial"].novel.followers == 3000 and len(by_title["Second Serial"].chapters) == 2
    assert by_title["Third Book"].novel.user_rating == 2 and by_title["Third Book"].novel.author == "E. Pub"
    assert len(by_title["Fourth"].chapters) == 2
    # forcing single-novel mode on a library folder fails loudly instead of silently mixing books
    try:
        import_path(lib, library=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_incremental_rescan_and_watch(cfg, db, tmp_path):
    lib = _make_library(tmp_path)
    r = IngestRunner(cfg, db, sources=[])
    res = r.import_local(lib)
    assert {x["title"] for x in res} == {"First Serial", "Second Serial", "Third Book", "Fourth", "Whole Book"}
    assert all(x["new"] == x["chapters"] and x["changed"] == 0 for x in res)
    assert {x["title"]: x["labelled"] for x in res}["Fourth"] is False
    # second pass: nothing changes
    res2 = {x["title"]: x for x in r.import_local(lib)}
    assert all(x["new"] == 0 and x["changed"] == 0 and x["removed"] == 0 for x in res2.values())
    # edit one chapter, add one, remove one elsewhere
    a = lib / "Ann Author - First Serial"
    (a / "chapter_2.txt").write_text(_prose(260, "edited"), encoding="utf-8")
    (a / "chapter_4.txt").write_text(_prose(260, "a4"), encoding="utf-8")
    (lib / "Completed" / "fourth" / "2.txt").unlink()
    # pre-compute features so we can check edited ones are invalidated
    Trainer(cfg, db, embedder=HashingEmbedder(32)).analyze()
    key_first = res2["First Serial"]["key"]
    assert db.feature_for(f"{key_first}#2", "chapter") is not None
    res3 = {x["title"]: x for x in r.import_local(lib)}
    assert res3["First Serial"]["new"] == 1 and res3["First Serial"]["changed"] == 1
    assert res3["Fourth"]["removed"] == 1 and db.chapter_count(res3["Fourth"]["key"]) == 1
    assert db.feature_for(f"{key_first}#2", "chapter") is None
    assert db.feature_for(f"{key_first}#1", "chapter") is not None
    # watch -> run() picks the library up as a local follow
    r.watch(str(lib))
    assert db.follows()[0][1] == "local"
    totals = r.run()
    assert totals["novels"] == 5 and totals["errors"] == 0


def test_unlabelled_novels_still_feed_exemplars_and_user_rating_trains(cfg, db, tmp_path):
    lib = _make_library(tmp_path)
    r = IngestRunner(cfg, db, sources=[])
    r.import_local(lib)
    # add more rated novels so a user_rating model can fit (>= 5 labelled)
    for i in range(4):
        d = tmp_path / f"extra{i}"
        d.mkdir()
        for c in range(2):
            (d / f"{c}.txt").write_text(_prose(250, f"e{i}{c}"), encoding="utf-8")
        r.import_local(d, meta_override={"user_rating": 1 + i})
    cfg.learn.target = "user_rating"
    t = Trainer(cfg, db, embedder=HashingEmbedder(32))
    out = t.cycle()
    assert isinstance(out["model"], dict) and out["model"]["n"] == 6  # First, Third + 4 extras
    lib_items = t.load_exemplars().items
    titles = {e.novel_title for e in lib_items}
    assert "Fourth" in titles and "Second Serial" in titles  # unlabelled for user_rating, still present


def test_cli_import_library_rate_novels(tmp_path):
    from click.testing import CliRunner

    from gauge.cli import main

    lib = _make_library(tmp_path)
    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "220"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    res = runner.invoke(main, ["--data-dir", data, "import", str(lib), "--watch"])
    assert res.exit_code == 0, res.output
    assert "Second Serial" in res.output and "Watching" in res.output
    res = runner.invoke(main, ["--data-dir", data, "novels", "--source", "local"])
    assert res.exit_code == 0 and "local:fourth" in res.output
    res = runner.invoke(main, ["--data-dir", data, "rate", "local:fourth", "--stars", "4"])
    assert res.exit_code == 0 and "Rated" in res.output
    res = runner.invoke(main, ["--data-dir", data, "rate", "local:nope", "--stars", "4"])
    assert "unknown" in res.output
    single = tmp_path / "Solo Author - Solo Book.txt"
    single.write_text("Chapter 1\n\n" + _prose(300) + "\n\nChapter 2\n\n" + _prose(300), encoding="utf-8")
    res = runner.invoke(main, ["--data-dir", data, "import", str(single), "--stars", "3", "--genre", "Horror"])
    assert res.exit_code == 0 and "Solo Book" in res.output
    res = runner.invoke(main, ["--data-dir", data, "ingest"])  # re-scans the watched library
    assert res.exit_code == 0 and "'novels': 5" in res.output
