"""Programming knowledge, the deterministic toolkit, bot creation, and guarded self-modification."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gauge.code import (
    BotRegistry,
    Engineer,
    KnowledgeBase,
    SelfProgrammer,
    run_tests,
    toolkit,
)
from gauge.code.knowledge import CARDS
from gauge.llm import FakeLLM, LLMError, make_llm
from gauge.llm.embeddings import HashingEmbedder


# ------------------------------------------------------------------ knowledge
def test_knowledge_cards_search_learn_and_self_index(tmp_path):
    kb = KnowledgeBase(HashingEmbedder(256), extra_dir=tmp_path / "kb")
    assert {"python-core", "powershell", "batch", "bash", "javascript", "sql-sqlite", "regex", "html-css", "testing-quality",
            "windows-automation", "gauge-internals"} <= set(CARDS)
    assert all(len(v) > 400 for v in CARDS.values())
    top = [c.key for c in kb.search("register a scheduled task in windows powershell", 2)]
    assert "powershell" in top or "windows-automation" in top
    assert kb.search("sqlite upsert on conflict", 1)[0].key == "sql-sqlite"
    notes = tmp_path / "notes.md"
    notes.write_text("# Royal Road HTML\n\nOnly p, em, strong, hr survive paste.\n\n# Epub Notes\n\nmimetype first, stored.\n", encoding="utf-8")
    cards = kb.learn_file(notes)
    assert [c.key for c in cards] == ["royal-road-html", "epub-notes"] and (tmp_path / "kb" / "royal-road-html.md").exists()
    assert kb.search("pasting into royal road", 1)[0].key == "royal-road-html"
    again = KnowledgeBase(HashingEmbedder(256), extra_dir=tmp_path / "kb")
    assert any(c.key == "epub-notes" for c in again.cards)  # learned cards persist
    n = kb.index_self(Path(__file__).resolve().parents[1] / "gauge")
    assert n > 30 and any(c.key == "self:gauge/code/selfmod.py" for c in kb.cards)
    hit = kb.search("self modification proposals apply rollback guardrails", 6)
    assert any(c.key.startswith("self:gauge/code/selfmod") for c in hit)
    assert "###" in kb.render(kb.search("regex", 1)) and len(kb.render(kb.cards, max_chars=500)) <= 520


# ------------------------------------------------------------------ toolkit
def test_toolkit_text_files_data(tmp_path):
    md = "---\ntitle: x\n---\n\n# Chapter 1: Gates\n\nShe *ran*. **Fast.**\n\n> A quote\nspanning lines\n\n***\n\n\"Hi,\" he said.\n"
    html = toolkit.markdown_to_html(md)
    assert html.startswith("<h1>Chapter 1: Gates</h1>") and "<em>ran</em>" in html and "<strong>Fast.</strong>" in html
    assert "<hr>" in html and "<blockquote><p>A quote spanning lines</p></blockquote>" in html and "&quot;Hi,&quot;" in html
    page = toolkit.html_page("T & T", html)
    assert page.startswith("<!doctype html>") and "T &amp; T" in page
    st = toolkit.text_stats(toolkit.strip_front_matter(md))
    assert st["words"] >= 8 and st["dialogue_lines"] == 1 and st["paragraphs"] >= 4
    assert toolkit.word_count("one two-three four's") == 3
    p = toolkit.write_text_atomic(tmp_path / "a" / "b.txt", "héllo")
    assert p.read_text(encoding="utf-8") == "héllo" and not (tmp_path / "a" / "b.txt.tmp").exists()
    (tmp_path / "a" / "c.txt").write_bytes("café".encode("cp1252"))
    assert toolkit.read_text(tmp_path / "a" / "c.txt") == "café"
    for i in (10, 2, 1):
        (tmp_path / "a" / f"ch{i}.md").write_text("x", encoding="utf-8")
    assert [f.name for f in toolkit.list_files(tmp_path / "a", ("*.md",))] == ["ch1.md", "ch2.md", "ch10.md"]
    assert len(toolkit.file_hash(p)) == 64
    toolkit.write_json(tmp_path / "j.json", {"a": 1, "when": Path("x")})
    assert toolkit.read_json(tmp_path / "j.json")["a"] == 1 and toolkit.read_json(tmp_path / "nope", 5) == 5
    toolkit.append_jsonl(tmp_path / "l.jsonl", {"n": 1})
    toolkit.append_jsonl(tmp_path / "l.jsonl", {"n": 2})
    assert len((tmp_path / "l.jsonl").read_text().splitlines()) == 2
    toolkit.write_csv(tmp_path / "t.csv", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    assert toolkit.read_csv(tmp_path / "t.csv") == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]
    b1 = toolkit.backup(p, tmp_path / "bk", keep=1)
    import time

    time.sleep(1.1)
    b2 = toolkit.backup(p, tmp_path / "bk", keep=1)
    assert b2.exists() and not b1.exists()  # pruned to the newest
    seen = []
    toolkit.watch_folder(tmp_path / "a", lambda files: seen.append([f.name for f in files]), interval_s=0, patterns=("*.md",), iterations=2)
    assert seen and set(seen[0]) == {"ch1.md", "ch2.md", "ch10.md"} and len(seen) == 1  # second poll: nothing changed
    assert toolkit.now_iso().endswith("+00:00")
    code, out, err = toolkit.run([sys.executable, "-c", "print('ok')"])
    assert code == 0 and out.strip() == "ok"


# ------------------------------------------------------------------ engineer + bots
def test_engineer_builds_a_working_bot_and_registry_runs_it(cfg, tmp_path):
    kb = KnowledgeBase(HashingEmbedder(128))
    eng = Engineer(FakeLLM(), kb, cfg.bots_dir, max_fix_rounds=1)
    cls = eng.classify("count words in every chapter file and write a csv daily")
    assert cls["needs_ai"] is False and cls["language"] == "python" and cls["schedule"] == "daily"
    assert eng.classify("summarize each chapter in one line")["needs_ai"] is True
    res = eng.build("wordcount", "count words in every chapter file and write a csv daily")
    assert res["tests_passed"] and res["needs_ai"] is False and set(res["files"]) == {"README.md", "main.py", "test_main.py"}
    reg = BotRegistry(cfg.bots_dir)
    assert reg.exists("wordcount") and reg.manifest("wordcount")["schedule"] == "daily"
    folder = tmp_path / "chapters"
    folder.mkdir()
    (folder / "c1.md").write_text("Alpha beta gamma.", encoding="utf-8")
    out = reg.run("wordcount", ["--folder", str(folder), "--out", str(tmp_path / "wc.csv")])
    assert out["exit"] == 0 and "1 files" in out["stdout"] and (tmp_path / "wc.csv").exists()
    assert reg.manifest("wordcount")["runs"] == 1 and (reg.path("wordcount") / "run.log").exists()
    bad = reg.run("wordcount", ["--folder", str(tmp_path / "missing")])
    assert bad["exit"] == 2 and "not a folder" in bad["stderr"]
    ok, report = run_tests(reg.path("wordcount"))
    assert ok, report
    # schedule: due immediately after creation, not again right after a run
    assert "wordcount" not in reg.due()  # it ran a moment ago
    man = reg.manifest("wordcount")
    man["last_run"] = "2000-01-01T00:00:00"
    (reg.path("wordcount") / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    assert reg.due() == ["wordcount"]
    assert reg.list()[0]["name"] == "wordcount" and reg.remove("wordcount") and not reg.exists("wordcount") and not reg.remove("wordcount")


def test_deterministic_bots_cannot_reach_the_model(monkeypatch):
    monkeypatch.setenv("GAUGE_NO_LLM", "1")
    with pytest.raises(LLMError):
        make_llm(__import__("gauge.config", fromlist=["LLMConfig"]).LLMConfig(provider="fake"))
    monkeypatch.delenv("GAUGE_NO_LLM")
    assert make_llm(__import__("gauge.config", fromlist=["LLMConfig"]).LLMConfig(provider="fake")).available()


# ------------------------------------------------------------------ self-modification on a scratch repo
def _mini_repo(root: Path) -> Path:
    (root / "gauge").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "gauge" / "__init__.py").write_text("", encoding="utf-8")
    (root / "gauge" / "mathy.py").write_text('"""Tiny module.\n\nAdds numbers."""\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n', encoding="utf-8")
    (root / "tests" / "test_mathy.py").write_text("from gauge.mathy import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('[tool.ruff]\nline-length = 100\n[tool.ruff.lint]\nselect = ["E", "F"]\n', encoding="utf-8")
    (root / "installer").mkdir()
    (root / "installer" / "bootstrap.ps1").write_text("# protected", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"], cwd=root, check=True)
    return root


def test_self_programmer_propose_test_apply_rollback(tmp_path):
    root = _mini_repo(tmp_path / "repo")
    kb = KnowledgeBase(HashingEmbedder(128))
    kb.index_self(root / "gauge")
    sp = SelfProgrammer(FakeLLM(), kb, tmp_path / "self", root=root, test_timeout=120)
    # guardrails
    assert sp.check_path("installer/bootstrap.ps1") and sp.check_path("../x.py") and sp.check_path("gauge/code/selfmod.py")
    assert sp.check_path("pyproject.toml") and sp.check_path("gauge/a.txt") and sp.check_path("gauge/mathy.py") is None
    assert "remove tests" in sp.check_content("tests/test_mathy.py", "def test_other():\n    pass\n", "def test_add():\n    pass\n")
    assert "skip" in sp.check_content("tests/test_mathy.py", "import pytest\n\ndef test_add():\n    pytest.skip('x')\n", "def test_add():\n    pass\n")
    assert "size" in sp.check_content("gauge/mathy.py", "x" * 1000, "short file content here")
    assert "compile" in sp.check_content("gauge/mathy.py", "def (:\n", "def add(a, b):\n    return a + b\n")
    assert sp.check_content("gauge/mathy.py", "def add(a, b):\n    return a + b  # ok\n", "def add(a, b):\n    return a + b\n") is None
    # propose (fake appends a harmless comment to the shown file)
    assert sp.relevant_files("adds numbers") == ["gauge/mathy.py"]
    prop = sp.propose("Document the add function", files=["gauge/mathy.py"])
    assert list(prop.files) == ["gauge/mathy.py"] and "touched by a self-proposal" in prop.files["gauge/mathy.py"]
    assert prop.diff.startswith("--- a/gauge/mathy.py") and (sp.dir / "proposals" / prop.id / "change.diff").exists()
    assert sp.load(prop.id).files == prop.files and sp.list()[0]["id"] == prop.id
    # cannot apply before testing
    with pytest.raises(PermissionError):
        sp.apply(prop)
    ok, report = sp.test(prop)
    assert ok, report
    assert "passed" in report
    original = (root / "gauge" / "mathy.py").read_text(encoding="utf-8")
    assert "touched" not in original  # testing never touched the real tree
    res = sp.apply(prop)
    assert res["applied"] and res["files"] == ["gauge/mathy.py"] and res["committed"]
    assert "touched by a self-proposal" in (root / "gauge" / "mathy.py").read_text(encoding="utf-8")
    log = subprocess.run(["git", "log", "--oneline", "-1"], cwd=root, capture_output=True, text=True).stdout
    assert "self: Document the add function" in log
    assert sp.apply(sp.load(prop.id)) == {"applied": False, "reason": "already applied"}
    rb = sp.rollback(prop.id)
    assert rb["restored"] == ["gauge/mathy.py"] and (root / "gauge" / "mathy.py").read_text(encoding="utf-8") == original
    # a proposal that breaks the tests is refused by test()
    bad = sp.propose("break it", files=["gauge/mathy.py"])
    bad.files["gauge/mathy.py"] = original.replace("return a + b", "return a - b")
    sp.save(bad)
    ok2, report2 = sp.test(bad)
    assert not ok2 and "failed" in report2
    with pytest.raises(PermissionError):
        sp.apply(bad)
    # refused files are reported, not written
    refusing = FakeLLM(canned=[json.dumps({"files": {"installer/bootstrap.ps1": "evil", "gauge/mathy.py": original + "\n# ok\n"}, "rationale": "r"})])
    prop3 = SelfProgrammer(refusing, kb, tmp_path / "self", root=root).propose("x", files=["gauge/mathy.py"])
    assert list(prop3.files) == ["gauge/mathy.py"] and any("protected" in r for r in prop3.meta["refused"])


def test_self_heal_from_journal(tmp_path, db):
    root = _mini_repo(tmp_path / "repo")
    kb = KnowledgeBase(HashingEmbedder(128))
    kb.index_self(root / "gauge")
    for _ in range(3):
        db.add_journal("write", "ValueError: adds numbers wrongly", "Traceback ... gauge/mathy.py line 5")
    db.add_journal("ingest", "once only", "")
    rec = db.recurring_errors(3)
    assert len(rec) == 1 and rec[0]["n"] == 3
    sp = SelfProgrammer(FakeLLM(), kb, tmp_path / "self", root=root, test_timeout=120)
    out = sp.heal(rec, auto_apply=False)
    assert len(out) == 1 and out[0]["tested"] and not out[0]["applied"] and out[0]["files"] == ["gauge/mathy.py"]
    assert sp.list()[0]["source"] == "journal"
    out2 = sp.heal(rec, auto_apply=True)
    assert out2[0]["applied"] and "touched" in (root / "gauge" / "mathy.py").read_text(encoding="utf-8")
    assert db.resolve_journal("ValueError: adds numbers wrongly") == 3 and db.journal() and db.journal()[0]["message"] == "once only"


def test_cli_bot_code_self(tmp_path, chapter_text):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "200"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    r = runner.invoke(main, ["--data-dir", data, "code", "cards"])
    assert r.exit_code == 0 and "python-core" in r.output
    r = runner.invoke(main, ["--data-dir", data, "code", "search", "sqlite upsert", "-k", "1"])
    assert r.exit_code == 0 and "ON CONFLICT" in r.output
    notes = tmp_path / "mynotes.md"
    notes.write_text("# My Site Rules\n\nNo more than 3 chapters a day.\n", encoding="utf-8")
    r = runner.invoke(main, ["--data-dir", data, "code", "learn", str(notes)])
    assert r.exit_code == 0 and "my-site-rules" in r.output
    r = runner.invoke(main, ["--data-dir", data, "bot", "new", "wc", "--task", "count words in chapter files and write a csv"])
    assert r.exit_code == 0 and '"tests_passed": true' in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "bot", "list"])
    assert r.exit_code == 0 and "wc" in r.output and "ok" in r.output
    folder = tmp_path / "ch"
    folder.mkdir()
    (folder / "a.md").write_text(chapter_text(True), encoding="utf-8")
    r = runner.invoke(main, ["--data-dir", data, "bot", "run", "wc", "--", "--folder", str(folder), "--out", str(tmp_path / "o.csv")])
    assert r.exit_code == 0 and (tmp_path / "o.csv").exists(), r.output
    r = runner.invoke(main, ["--data-dir", data, "bot", "schedule", "wc", "hourly"])
    assert r.exit_code == 0 and "hourly" in r.output
    r = runner.invoke(main, ["--data-dir", data, "bot", "test", "wc"])
    assert r.exit_code == 0
    r = runner.invoke(main, ["--data-dir", data, "self", "list"])
    assert r.exit_code == 0 and "no proposals" in r.output
    r = runner.invoke(main, ["--data-dir", data, "self", "journal"])
    assert r.exit_code == 0 and "journal is empty" in r.output
    r = runner.invoke(main, ["--data-dir", data, "bot", "remove", "wc"])
    assert r.exit_code == 0


def test_daemon_runs_due_bots(cfg, db, chapter_text):
    from gauge.daemon import Daemon

    eng = Engineer(FakeLLM(), KnowledgeBase(HashingEmbedder(64)), cfg.bots_dir)
    eng.build("wc", "count words daily", schedule="on-cycle")
    cfg.learn.absorb_per_cycle = 0
    cfg.native.daemon_steps = 0
    cfg.research.enabled = False
    cfg.compute.auto_device = False
    out = Daemon(cfg, db).once(write=False)
    assert out.get("bots") == {"wc": 0}
    assert BotRegistry(cfg.bots_dir).manifest("wc")["runs"] == 1
    assert "GAUGE_NO_LLM" not in os.environ  # the runner's env never leaks into the parent
