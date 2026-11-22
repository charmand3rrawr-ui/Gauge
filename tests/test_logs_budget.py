"""Structured logging and the daemon's error budget (roadmap 96)."""
import json
import logging
from datetime import timedelta

import pytest
from click.testing import CliRunner

from gauge import dryrun
from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.daemon import Daemon
from gauge.ops.budget import ErrorBudget, budget_for
from gauge.ops.logs import event, read_log, redact, remove_structured_log, setup_structured_log
from gauge.util import utcnow


@pytest.fixture
def jlog(tmp_path):
    path = tmp_path / "logs" / "gauge.jsonl"
    setup_structured_log(path)
    yield path
    remove_structured_log()


# ------------------------------------------------------------------ structured log
def test_records_are_json_lines_with_fields_and_errors(jlog):
    log = logging.getLogger("gauge.test")
    event(log, "cycle_done", duration_s=12.5, failed_steps=["ingest"])
    try:
        raise ValueError("boom")
    except ValueError:
        event(log, "cycle_failed", logging.ERROR, "cycle failed", exc_info=True)
    lines = [json.loads(x) for x in jlog.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    first, second = lines
    assert first["event"] == "cycle_done" and first["duration_s"] == 12.5 and first["failed_steps"] == ["ingest"]
    assert first["level"] == "info" and first["ts"].endswith("+00:00") and first["logger"] == "gauge.test"
    assert second["level"] == "error" and second["error"]["type"] == "ValueError" and "boom" in second["error"]["traceback"]


def test_secrets_are_masked_in_fields_and_messages(jlog):
    log = logging.getLogger("gauge.test")
    event(log, "worker_call", msg="calling worker with token=abc123XYZ and password: hunter2",
          api_key="sk-live-999", nested={"Authorization": "Bearer zzz", "host": "10.0.0.2"})
    rec = json.loads(jlog.read_text(encoding="utf-8").splitlines()[0])
    raw = json.dumps(rec)
    for secret in ("abc123XYZ", "hunter2", "sk-live-999", "Bearer zzz"):
        assert secret not in raw
    assert rec["nested"]["host"] == "10.0.0.2"                      # ordinary fields survive
    assert redact({"note": "plain", "n": 3}) == {"note": "plain", "n": 3}


def test_read_log_filters_and_tolerates_junk(jlog):
    log = logging.getLogger("gauge.test")
    log.info("plain info")
    log.warning("a warning")
    event(log, "cycle_done", duration_s=1.0)
    with jlog.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
    assert [r["msg"] for r in read_log(jlog, min_level="warning")] == ["a warning"]
    assert [r["event"] for r in read_log(jlog, event_name="cycle_done")] == ["cycle_done"]
    assert len(read_log(jlog, tail=2)) == 2
    assert read_log(jlog.with_name("missing.jsonl")) == []


def test_log_rotates_instead_of_growing_forever(tmp_path):
    path = tmp_path / "logs" / "gauge.jsonl"
    handler = setup_structured_log(path, max_mb=1, backups=2)
    handler.maxBytes = 2000                                          # shrink the limit so the test is quick
    log = logging.getLogger("gauge.test")
    for i in range(200):
        log.info("record %d %s", i, "x" * 40)
    remove_structured_log()
    assert path.exists() and path.with_name("gauge.jsonl.1").exists()
    assert path.stat().st_size <= 2600
    assert not path.with_name("gauge.jsonl.3").exists()             # only `backups` old files are kept


def test_setup_replaces_rather_than_stacks_handlers(tmp_path):
    before = len(logging.getLogger().handlers)
    setup_structured_log(tmp_path / "a.jsonl")
    setup_structured_log(tmp_path / "b.jsonl")
    assert len(logging.getLogger().handlers) == before + 1
    remove_structured_log()
    assert len(logging.getLogger().handlers) == before


# ------------------------------------------------------------------ error budget
def _fail(db, n, kind="ingest"):
    for i in range(n):
        db.add_journal(kind, f"ConnectionError: attempt {i}", "traceback")


def test_budget_states_and_backoff(db):
    b = ErrorBudget(db, allowed=10, window_hours=24, max_backoff=4)
    assert b.status().state == "healthy" and b.status().backoff == 1.0
    _fail(db, 5)
    s = b.status()
    assert s.state == "strained" and s.backoff == 2.0 and s.remaining == 5
    _fail(db, 5, kind="newsdesk")
    s = b.status()
    assert s.exhausted and s.backoff == 4.0 and s.remaining == 0 and s.by_kind == {"ingest": 5, "newsdesk": 5}


def test_budget_refills_with_time_and_when_errors_are_resolved(db):
    b = ErrorBudget(db, allowed=3, window_hours=24)
    _fail(db, 3)
    assert b.status().exhausted
    assert b.status(now=utcnow() + timedelta(hours=25)).state == "healthy"   # errors aged out of the window
    for i in range(3):
        db.resolve_journal(f"ConnectionError: attempt {i}")
    assert b.status().state == "healthy"                                       # resolved errors stop counting


def test_budget_off_never_degrades(db):
    _fail(db, 50)
    s = ErrorBudget(db, allowed=0).status()
    assert not s.enabled and not s.exhausted and s.backoff == 1.0 and s.state == "off"


def test_daemon_holds_automation_when_budget_is_exhausted(cfg, db, monkeypatch):
    monkeypatch.delenv(dryrun.ENV, raising=False)
    cfg.daemon.error_budget = 3
    cfg.daemon.active_project = "Anything"
    cfg.newsdesk.enabled = True
    out = Daemon(cfg, db).once(write=False)
    assert out["error_budget"]["state"] == "healthy" and "newsdesk" in out   # runs normally while healthy
    _fail(db, 3)
    d = Daemon(cfg, db)
    out = d.once(write=True)
    assert out["error_budget"]["state"] == "exhausted"
    assert out["write"] == "held: error budget exhausted"            # no chapter written while unhealthy
    assert "newsdesk" not in out                                     # and no publishing
    assert "train" in out                                            # learning continues
    assert d._backoff() == float(cfg.daemon.max_backoff)             # and the loop slows down


def test_daemon_emits_a_structured_cycle_record(cfg, db, jlog, monkeypatch):
    monkeypatch.delenv(dryrun.ENV, raising=False)
    Daemon(cfg, db).once(write=False)
    recs = read_log(jlog, event_name="cycle_done")
    assert recs and recs[-1]["budget_state"] == "healthy" and recs[-1]["degraded"] is False
    assert isinstance(recs[-1]["duration_s"], (int, float)) and "train" in recs[-1]["steps"]


# ------------------------------------------------------------------ CLI
def test_cli_logs_and_budget(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.daemon.error_budget = 2
    cfg.save()
    from gauge.db import Database
    db = Database(cfg.db_path)
    _fail(db, 2)
    db.close()
    run = CliRunner(env={"COLUMNS": "200"})
    r = run.invoke(main, ["--data-dir", data, "budget"])
    assert r.exit_code == 0 and "exhausted" in r.output and "2 of 2" in r.output, r.output
    before = len(logging.getLogger().handlers)
    r = run.invoke(main, ["--data-dir", data, "logs", "--tail", "5"])
    assert r.exit_code == 0
    assert len(logging.getLogger().handlers) == before               # the file handler is detached after each command
    r = run.invoke(main, ["--data-dir", data, "audit"])
    assert "Error budget: 2 unresolved errors per 24h" in r.output


def test_budget_for_reads_config(cfg, db):
    cfg.daemon.error_budget, cfg.daemon.error_window_hours, cfg.daemon.max_backoff = 7, 12, 3
    b = budget_for(cfg, db)
    assert (b.allowed, b.window_hours, b.max_backoff) == (7, 12, 3)
