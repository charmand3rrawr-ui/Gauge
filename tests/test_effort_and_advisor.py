"""More thinking per chapter, and the strongest model the computer can run."""
import types

import pytest

from gauge.gui import StudioController
from gauge.llm import advisor
from gauge.write import effort


# ------------------------------------------------------------------ the effort dial
def test_each_level_asks_for_more_than_the_one_before():
    costs = [effort.LEVELS[name]["cost"] for name in effort.ORDER]
    assert costs == sorted(costs) and effort.ORDER == ["quick", "normal", "deep", "maximum"]
    for a, b in zip(effort.ORDER, effort.ORDER[1:]):
        lo, hi = effort.LEVELS[a], effort.LEVELS[b]
        assert hi["best_of"] >= lo["best_of"] and hi["max_revisions"] > lo["max_revisions"]
        assert hi["min_acceptable_score"] > lo["min_acceptable_score"] and hi["muse_directions"] >= lo["muse_directions"]


def test_the_dial_sets_the_settings_and_normal_is_the_default(cfg):
    assert effort.current(cfg) == "normal"                        # the defaults are exactly the normal level
    deep = effort.apply(cfg, "Deep")
    w = cfg.write
    assert (w.best_of, w.max_revisions, w.muse_directions, w.ab_titles, w.ab_hooks) == (2, 3, 5, True, True)
    assert deep["level"] == "deep" and "2.5 times as long" in deep["message"] and effort.current(cfg) == "deep"
    quick = effort.apply(cfg, "quick")
    assert (w.best_of, w.max_revisions, w.muse_enabled, w.improv_enabled) == (1, 0, False, False) and quick["cost"] == 0.4
    w.max_revisions = 4                                           # changed by hand since
    assert effort.current(cfg) == "custom"
    with pytest.raises(ValueError):
        effort.apply(cfg, "heroic")
    # With GaugeLM writing, attempts never drop below four above quick, and the room stays lean.
    cfg.llm.provider, cfg.agents.profile = "native", "lean"
    assert effort.apply(cfg, "normal")["attempts"] == 4 and effort.apply(cfg, "maximum")["attempts"] == 6
    assert cfg.agents.profile == "lean" and effort.apply(cfg, "quick")["attempts"] == 2
    cfg.llm.provider = "ollama"
    effort.apply(cfg, "maximum")
    assert cfg.agents.profile == "full" and cfg.write.best_of == 3


def test_the_dial_is_a_switch_and_a_command(cfg, db):
    from click.testing import CliRunner

    from gauge.cli import main
    from gauge.config import GaugeConfig

    ctl = StudioController(cfg, db)
    assert ctl.effort()["level"] == "normal"
    ctl.set_switch("write.effort", "deep")                         # one switch moves all of them
    saved = GaugeConfig.load(cfg.data_dir)
    assert saved.write.effort == "deep" and saved.write.max_revisions == 3 and saved.write.best_of == 2
    assert ctl.effort("quick")["level"] == "quick" and GaugeConfig.load(cfg.data_dir).write.max_revisions == 0
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "effort", "maximum"])
    assert r.exit_code == 0 and "Thinking effort is maximum" in r.output
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "effort"])
    assert "Thinking effort is maximum" in r.output and "quick" in r.output and "(about 5x the time)" in r.output


def test_more_effort_really_makes_more_attempts(cfg, db):
    from gauge.llm import FakeLLM

    class _Count(FakeLLM):
        def __init__(self):
            super().__init__()
            self.kinds: list[str] = []

        def generate(self, prompt, **kw):
            self.kinds.append(kw.get("purpose", ""))
            return super().generate(prompt, **kw)

    counts = {}
    for level in ("quick", "deep"):
        ctl = StudioController(cfg, db)
        ctl.effort(level)
        cfg.write.reach_length = False
        llm = _Count()
        pipe = ctl.pipeline()
        pipe.llm = pipe.fast_llm = llm
        for part in ("planner", "drafter"):
            getattr(pipe, part).llm = llm
        name = ctl.start_story("A courier who can rewind ten seconds", name=f"Story {level}")
        ctl.write_next(name, 200)
        counts[level] = llm.kinds.count("draft")
    assert counts == {"quick": 1, "deep": 2}


# ------------------------------------------------------------------ the strongest model that fits
def test_the_strongest_model_that_fits_is_recommended():
    big = advisor.advise("llama3.1:8b", vram_gb=24, ram_gb=64)
    assert big["best"] == "qwen2.5:32b" and big["better"] and big["needs_download"]
    assert "stronger writer than llama3.1:8b" in big["message"] and "on the graphics card" in big["message"] and "20 GB" in big["message"]
    assert advisor.advise("qwen2.5:32b", 24, 64)["better"] is False
    assert "already the strongest" in advisor.advise("qwen2.5:32b", 24, 64)["message"]
    mid = advisor.advise("llama3.2:3b", vram_gb=12, ram_gb=32)
    assert mid["best"] == "qwen2.5:14b" and [f["model"] for f in mid["fits"]][-1] == "qwen2.5:14b"
    huge = advisor.advise("qwen2.5:14b", vram_gb=48, ram_gb=128)
    assert huge["best"] == "llama3.3:70b"
    # Something already downloaded wins a tie; nothing bigger than mid-sized is suggested without a card.
    assert advisor.advise("llama3.2:3b", 8, 16, installed=["llama3.1:8b"])["best"] == "llama3.1:8b"
    assert advisor.advise("llama3.2:3b", 8, 16)["best"] == "qwen2.5:7b"                # otherwise the openly licensed one
    slow = advisor.advise("llama3.2:3b", vram_gb=0, ram_gb=64)
    assert slow["best"] == "qwen2.5:14b" and "in ordinary memory, which is slow" in slow["message"]
    assert advisor.advise("llama3.2:3b", 0, 8)["best"] == "llama3.2:3b"
    assert advisor.advise("x", 0, 0)["best"] is None
    assert "does not know how" in advisor.advise("some-new-model", 8, 16)["message"]
    assert (advisor.rank_of("qwen2.5:14b"), advisor.rank_of("gemma2:27b"), advisor.rank_of("mystery")) == (3, 4, 0)


def test_fetching_a_model_reports_plainly(monkeypatch):
    ran = []

    def run(ok):
        def go(cmd, **kw):
            ran.append(cmd)
            return types.SimpleNamespace(returncode=0 if ok else 1, stdout="", stderr="Error: no space left on device")
        return go

    assert advisor.fetch("qwen2.5:32b", run=run(True))["ok"] and ran[-1][-2:] == ["pull", "qwen2.5:32b"]
    bad = advisor.fetch("qwen2.5:32b", run=run(False))
    assert not bad["ok"] and "no space left on device" in bad["message"] and "free disk space" in bad["message"]
    monkeypatch.setattr(advisor.shutil, "which", lambda name: None)
    assert "Ollama program was not found" in advisor.fetch("qwen2.5:32b")["message"]
    from gauge import dryrun
    dryrun.set_dry_run(True)
    try:
        assert advisor.fetch("qwen2.5:32b", run=run(True))["message"].startswith("Dry run")
    finally:
        dryrun.set_dry_run(False)
        dryrun.reset()


def test_the_studio_finds_and_switches_to_a_stronger_model(cfg, db, monkeypatch):
    from gauge.config import GaugeConfig
    from gauge.llm.client import OllamaLLM

    cfg.llm.provider, cfg.llm.model = "ollama", "llama3.1:8b"
    ctl = StudioController(cfg, db)
    monkeypatch.setattr(OllamaLLM, "models", lambda self: ["llama3.1:8b"])
    monkeypatch.setattr(advisor, "hardware", lambda: {"vram_gb": 24.0, "ram_gb": 64.0})
    adv = ctl.stronger_model()
    assert adv["better"] and adv["best"] == "qwen2.5:32b"
    monkeypatch.setattr(advisor, "fetch", lambda model, **k: {"ok": False, "message": "Could not download."})
    assert not ctl.get_stronger_model()["changed"] and cfg.llm.model == "llama3.1:8b"   # a failed download changes nothing
    monkeypatch.setattr(advisor, "fetch", lambda model, **k: {"ok": True, "message": "ok"})
    got = ctl.get_stronger_model()
    assert got["changed"] and GaugeConfig.load(cfg.data_dir).llm.model == "qwen2.5:32b"
    assert not ctl.get_stronger_model()["changed"]                                      # already there
    cfg.llm.provider = "native"
    assert "GaugeLM is writing" in ctl.stronger_model()["message"]
