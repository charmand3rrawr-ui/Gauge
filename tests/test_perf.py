"""Roadmap section G (subset): the LLM response cache (61) and the telemetry dashboard (69)."""

from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.db import Database
from gauge.llm import FakeLLM, make_llm
from gauge.llm.cache import CachingLLM
from gauge.ops import dashboard_html


class _Counting(FakeLLM):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.underlying = 0

    def generate(self, *a, **k):
        self.underlying += 1
        return super().generate(*a, **k)


def test_cache_reuses_deterministic_calls(tmp_path):
    inner = _Counting()
    c = CachingLLM(inner, tmp_path / "cache")
    a = c.generate("[agent:x:y] brief", temperature=0.2, json_mode=True)
    b = c.generate("[agent:x:y] brief", temperature=0.2, json_mode=True)
    assert a == b and inner.underlying == 1  # second request served from cache
    assert c.cache_stats()["hits"] == 1 and c.cache_stats()["entries"] == 1


def test_cache_skips_creative_prose(tmp_path):
    inner = _Counting()
    c = CachingLLM(inner, tmp_path / "cache")
    c.generate("write a chapter", temperature=0.9)
    c.generate("write a chapter", temperature=0.9)
    assert inner.underlying == 2  # high-temperature prose is never cached


def test_cache_proxies_wrapped_methods(tmp_path):
    c = CachingLLM(FakeLLM(), tmp_path / "cache")
    assert c.available() is True and "fake" in c.describe() and c.context_window() > 0


def test_make_llm_wraps_when_cache_enabled(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.llm.provider = "fake"
    cfg = GaugeConfig.load(str(tmp_path / "data")) if (tmp_path / "data" / "config.json").exists() else cfg
    cfg.llm.provider = "fake"
    cfg.llm.cache = True
    cfg.llm.cache_dir = str(tmp_path / "cache")
    llm = make_llm(cfg.llm)
    assert isinstance(llm, CachingLLM)
    cfg.llm.cache = False
    assert not isinstance(make_llm(cfg.llm), CachingLLM)


def test_dashboard_html(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    db.log_call("draft", "m", 100, 200, 1500, True, 0, "P", 1)
    db.log_call("critic", "m", 50, 40, 300, True, 0, "P", 1)
    from gauge.write.project import Project
    p = Project.create(cfg.projects_dir, "P", "idea", ["Sci-fi"], 300)
    p.write_chapter(1, "T", "word " * 80, {"title": "T", "score": 0.72})
    out = dashboard_html(db, cfg.projects_dir)
    assert "Cost by purpose" in out and "draft" in out and "ch1" in out and "0.72" in out


def test_cli_dashboard(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.ensure_dirs()
    cfg.save()
    out = tmp_path / "dash.html"
    r = CliRunner(env={"COLUMNS": "200"}).invoke(main, ["--data-dir", data, "dashboard", "--out", str(out)])
    assert r.exit_code == 0 and out.exists() and "Gauge telemetry" in out.read_text()
