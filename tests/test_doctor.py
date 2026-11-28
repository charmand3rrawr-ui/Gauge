"""Gauge's doctor: the Check for problems button and `gauge doctor`."""
import json
import sys
from types import SimpleNamespace

from click.testing import CliRunner

from gauge import doctor
from gauge.cli import main
from gauge.gui import StudioController


def _find(res, name):
    return [f for f in res["findings"] if f["name"] == name]


def test_a_fresh_install_is_checked_and_each_problem_says_what_to_do(cfg, db):
    res = doctor.run(cfg, db, sensed=[])
    names = {f["name"] for f in res["findings"]}
    assert {"Python", "Gauge", "Data folder", "Disk space", "Library", "Writer", "Capacity limit", "Stories"} <= names
    assert _find(res, "Writer")[0]["status"] == "info" and "test model" in _find(res, "Writer")[0]["says"]
    lib = _find(res, "Library")[0]
    assert lib["status"] == "warn" and "Teach Gauge" in lib["fix"] and lib["action"] == "teach"
    assert all(f["fix"] for f in res["findings"] if f["status"] in ("problem", "warn"))     # every fault says what to do
    assert res["ok"] and "teach" in res["actions"] and res["summary"].startswith("Gauge works.")
    statuses = [f["status"] for f in res["findings"]]
    assert statuses == sorted(statuses, key=doctor.ORDER.get)                               # the worst first
    assert "contains no keys" in res["text"] and "What to do:" in res["text"]


def test_the_report_never_contains_a_key(cfg, db, monkeypatch, tmp_path):
    from gauge.llm import claude, helpers

    monkeypatch.setenv("GAUGE_CLAUDE_KEY_FILE", str(tmp_path / "home" / "claude-key.txt"))
    monkeypatch.setenv("GAUGE_HELPER_KEYS_FILE", str(tmp_path / "home" / "helper-keys.json"))
    claude.save_key("sk-ant-secret-claude-key")
    helpers.save_key("groq", "gsk-secret-helper-key")
    cfg.helpers.providers = {"groq": {"model": "llama-3.3-70b-versatile", "terms_ok": True, "teach": True, "check": True}}
    cfg.claude.teach = True
    monkeypatch.setattr(claude, "installed", lambda: True)
    res = doctor.run(cfg, db, sensed=[])
    assert "secret" not in res["text"] and "secret" not in json.dumps(res["findings"])
    assert any(f["name"] == "Free helpers" and "Groq" in f["says"] for f in res["findings"])


def test_a_broken_check_does_not_hide_the_others(cfg, db, monkeypatch):
    def broken(cfg, db):
        raise RuntimeError("boom")

    monkeypatch.setattr(doctor, "CHECKS", [("Broken", broken), *doctor.CHECKS])
    res = doctor.run(cfg, db, sensed=[])
    assert _find(res, "Broken")[0]["says"].startswith("Could not check this (RuntimeError: boom)")
    assert _find(res, "Python") and _find(res, "Stories")


def test_unreadable_settings_and_stories_are_problems(cfg, db):
    cfg.config_path.write_text('{"llm": {"provider": "fake",', encoding="utf-8")
    bad = cfg.projects_dir / "broken-story"
    bad.mkdir(parents=True)
    (bad / "project.json").write_text("not json", encoding="utf-8")
    res = doctor.run(cfg, db, sensed=[])
    assert _find(res, "Settings")[0]["status"] == "problem" and "Notepad" in _find(res, "Settings")[0]["fix"]
    stories = _find(res, "Stories")
    assert stories[0]["status"] == "problem" and "broken-story" in stories[0]["says"] and "chapters are safe" in stories[0]["fix"]
    assert not res["ok"] and res["summary"].startswith("2 problems")


def test_the_writer_is_checked_for_each_kind_of_model(cfg, db, monkeypatch):
    from gauge.llm.client import OllamaLLM

    cfg.llm.provider = "native"
    w = _find(doctor.run(cfg, db, sensed=[]), "Writer")[0]
    assert w["status"] == "problem" and "not been trained" in w["says"] and w["action"] == "engine"

    cfg.llm.provider, cfg.llm.model = "ollama", "qwen2.5:14b"
    monkeypatch.setattr(OllamaLLM, "available", lambda self: False)
    w = _find(doctor.run(cfg, db, sensed=[]), "Writer")[0]
    assert w["status"] == "problem" and "not answering" in w["says"] and "Ollama app" in w["fix"]

    monkeypatch.setattr(OllamaLLM, "available", lambda self: True)
    monkeypatch.setattr(OllamaLLM, "models", lambda self: ["llama3.1:8b", "qwen2.5:7b"])
    cfg.llm.fast_model = "llama3.2:3b"
    res = doctor.run(cfg, db, sensed=[])
    w = _find(res, "Writer")[0]
    assert w["status"] == "problem" and "ollama pull qwen2.5:14b" in w["fix"]
    assert "ollama pull llama3.2:3b" in _find(res, "Fast model")[0]["fix"]
    assert "ollama pull nomic-embed-text" in _find(res, "Embeddings")[0]["fix"]

    monkeypatch.setattr(OllamaLLM, "models", lambda self: ["qwen2.5:14b", "llama3.2:3b:latest", "nomic-embed-text:latest"])
    res = doctor.run(cfg, db, sensed=[])
    assert _find(res, "Writer")[0]["status"] == "ok" and not _find(res, "Embeddings") and not _find(res, "Fast model")

    from gauge.llm import claude

    cfg.llm.provider = "claude"
    monkeypatch.setattr(claude, "installed", lambda: False)
    w = _find(doctor.run(cfg, db, sensed=[]), "Writer")[0]
    assert w["status"] == "problem" and w["action"] == "claude"


def test_a_graphics_card_gaugelm_cannot_use_yet_is_pointed_out(cfg, db, monkeypatch):
    import gauge.compute.detect as detect

    card = SimpleNamespace(kind="gpu", vendor="nvidia", name="NVIDIA GeForce RTX 4070")
    monkeypatch.setattr(detect, "backend_status", lambda: {"cupy": False, "cupy_devices": 0})
    f = _find(doctor.run(cfg, db, sensed=[card]), "Graphics card")[0]
    assert f["status"] == "warn" and "RTX 4070" in f["says"] and f["action"] == "devices"
    monkeypatch.setattr(detect, "backend_status", lambda: {"cupy": True, "cupy_devices": 1})
    assert _find(doctor.run(cfg, db, sensed=[card]), "Graphics card")[0]["status"] == "ok"
    assert _find(doctor.run(cfg, db, sensed=[]), "Graphics card")[0]["status"] == "info"


def test_modes_that_change_what_gauge_does_are_mentioned(cfg, db, monkeypatch):
    from gauge import dryrun

    monkeypatch.setenv(dryrun.ENV, "1")
    cfg.security.locked, cfg.security.locked_reason = True, "too many failed logins"
    res = doctor.run(cfg, db, sensed=[])
    assert _find(res, "Dry run")[0]["status"] == "warn" and "too many failed logins" in _find(res, "Lockdown")[0]["says"]


def test_gauge_doctor_prints_the_same_checks(tmp_path):
    data = str(tmp_path / "d")
    run = CliRunner(env={"COLUMNS": "200"})
    assert run.invoke(main, ["--data-dir", data, "init", "--provider", "fake"]).exit_code == 0
    out = run.invoke(main, ["--data-dir", data, "doctor"])
    assert out.exit_code == 0 and "fake llm" in out.output                     # the old lines are still there
    assert "Library: The library is empty" in out.output and "What to do: Press Teach Gauge" in out.output
    assert "Gauge works." in out.output


def test_the_studio_controller_runs_the_doctor_and_remembers_the_devices(cfg, db, monkeypatch):
    import gauge.compute as compute

    calls = []
    monkeypatch.setattr(compute, "scan", lambda: calls.append(1) or [])
    ctl = StudioController(cfg, db)
    seen = []
    res = ctl.doctor(seen.append)
    assert res["findings"] and seen[0] == "Python" and "Stories" in seen
    ctl.doctor()
    assert calls == [1]                                    # looked for once, not on every check


def test_the_window_has_a_check_for_problems_button(cfg, db, tmp_path, monkeypatch):
    """The real window code, driven with the stand-in toolkit: it proves the handlers are wired, not how it looks."""
    import gauge.compute as compute
    from gauge import gui
    from tests.test_teach import _fake_tkinter, _wait_for_jobs, _Widget

    monkeypatch.setattr(compute, "scan", lambda: [])
    mods, shown = _fake_tkinter(tmp_path)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    assert "Check for problems" in _Widget.buttons and "Check Gauge for problems..." in _Widget.menu
    copied = []
    monkeypatch.setattr(_Widget, "clipboard_append", lambda self, text: copied.append(text), raising=False)
    _Widget.buttons["Check for problems"]()
    _wait_for_jobs()
    assert {"Check again", "Copy the report", "Open Teach Gauge"} <= set(_Widget.buttons)    # a fix button for what it found
    _Widget.buttons["Copy the report"]()
    assert copied and copied[0].startswith("Gauge check,") and "Library: The library is empty" in copied[0]
    _Widget.buttons["Check again"]()
    _wait_for_jobs()
    _Widget.buttons["Open Teach Gauge"]()                  # opens the window that fixes it
    assert "Add files..." in _Widget.buttons
