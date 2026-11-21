"""The switches: one catalog behind the Studio panel and `gauge settings`; rebel craft mode."""
import json

import pytest

from gauge import settings as st
from gauge.config import GaugeConfig
from gauge.gui import StudioController
from gauge.learn import Trainer
from gauge.llm import FakeLLM, HashingEmbedder
from gauge.write import WritePipeline


def test_catalog_matches_config_and_coerces(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.ensure_dirs()
    keys = [s.key for s in st.SWITCHES]
    assert len(keys) == len(set(keys)) and "write.rebel" in keys and "code.auto_apply" in keys
    for sw in st.SWITCHES:
        st.get(cfg, sw.key)  # every key resolves
        if sw.kind == "choice":
            assert st.get(cfg, sw.key) in sw.choices or sw.kind == "text"
    assert st.groups() == ["Writing", "The room", "Learning", "Model", "Security", "Automation"]
    # booleans accept words
    assert st.set_value(cfg, "write.rebel", "on", save=False) is True and st.get(cfg, "write.rebel") is True
    assert st.set_value(cfg, "write.rebel", "0", save=False) is False
    with pytest.raises(ValueError):
        st.set_value(cfg, "write.rebel", "maybe", save=False)
    # numbers are ranged and rounded
    assert st.set_value(cfg, "write.rebel_level", "0.777", save=False) == 0.777
    with pytest.raises(ValueError):
        st.set_value(cfg, "write.rebel_level", "1.5", save=False)
    assert st.set_value(cfg, "write.max_revisions", "2.6", save=False) == 3
    with pytest.raises(ValueError):
        st.set_value(cfg, "write.max_revisions", "x", save=False)
    # choices and text
    assert st.set_value(cfg, "agents.profile", "LEAN", save=False) == "lean"
    with pytest.raises(ValueError):
        st.set_value(cfg, "agents.profile", "huge", save=False)
    assert st.set_value(cfg, "llm.fast_model", " llama3.2:3b ", save=False) == "llama3.2:3b"
    assert st.set_value(cfg, "native.device", "cuda", save=False) == "cuda" and cfg.llm.native_device == "cuda"
    with pytest.raises(KeyError):
        st.set_value(cfg, "nope.key", 1, save=False)
    # persistence and reset
    st.set_value(cfg, "code.auto_apply", True)
    assert json.loads(cfg.config_path.read_text())["code"]["auto_apply"] is True
    assert GaugeConfig.load(cfg.data_dir).code.auto_apply is True
    assert st.reset(cfg, "code.auto_apply") is False and GaugeConfig.load(cfg.data_dir).code.auto_apply is False
    rows = st.state(cfg)
    row = next(r for r in rows if r["key"] == "code.auto_apply")
    assert row["warning"] and row["kind"] == "bool" and row["value"] is False
    assert next(r for r in rows if r["key"] == "research.respect_robots_txt")["warning"]


def test_studio_controller_exposes_switches(cfg, db):
    ctl = StudioController(cfg, db)
    rows = ctl.switches()
    assert any(r["key"] == "write.rebel" for r in rows)
    assert ctl.set_switch("write.rebel", True) is True and GaugeConfig.load(cfg.data_dir).write.rebel is True
    with pytest.raises(ValueError):
        ctl.set_switch("write.rebel_level", 7)


def test_rebel_mode_changes_the_prompts(cfg, db, chapter_text):
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    cfg.write.rebel = True
    cfg.write.rebel_level = 0.8
    main, fast = FakeLLM(), FakeLLM()
    pipe = WritePipeline(cfg, db, llm=main, trainer=t, fast_llm=fast)
    assert pipe.guide_compact.startswith("REBEL MODE (80%)")
    p = pipe.new_project("Rebel", "a courier", ["Sci-fi"], words=200)
    muse_prompt = next(c["prompt"] for c in fast.calls_list if c["purpose"] == "muse")
    assert "deliberately break a convention" in muse_prompt and "At least 2 of the directions" in muse_prompt  # round(3 * 0.8)
    pipe.write_next(p)
    critic_prompt = next(c["prompt"] for c in fast.calls_list if c["purpose"] == "critic")
    assert critic_prompt.startswith("Rebel mode is on")
    assert "REBEL MODE" in next(c["system"] for c in main.calls_list if c["purpose"] == "draft")
    # off by default: none of that appears
    cfg.write.rebel = False
    pipe2 = WritePipeline(cfg, db, llm=FakeLLM(), trainer=t, fast_llm=FakeLLM())
    assert "REBEL" not in pipe2.guide_compact and pipe2.muse.rebel_level == 0.0 and pipe2.critic.rebel is False


def test_cli_settings(tmp_path):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "220"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    r = runner.invoke(main, ["--data-dir", data, "settings", "list"])
    assert r.exit_code == 0 and "write.rebel" in r.output and "Rebel craft mode" in r.output
    r = runner.invoke(main, ["--data-dir", data, "settings", "list", "--group", "Automation"])
    assert r.exit_code == 0 and "code.auto_apply" in r.output and "write.rebel" not in r.output
    r = runner.invoke(main, ["--data-dir", data, "settings", "on", "write.rebel"])
    assert r.exit_code == 0 and "write.rebel = True" in r.output
    assert json.loads((tmp_path / "data" / "config.json").read_text())["write"]["rebel"] is True
    r = runner.invoke(main, ["--data-dir", data, "settings", "set", "write.rebel_level", "0.9"])
    assert r.exit_code == 0 and "0.9" in r.output
    r = runner.invoke(main, ["--data-dir", data, "settings", "set", "write.rebel_level", "9"])
    assert r.exit_code != 0 and "outside" in r.output
    r = runner.invoke(main, ["--data-dir", data, "settings", "on", "code.auto_apply"])
    assert r.exit_code == 0 and "Keep off unless you trust it" in r.output  # warning shown when switching on
    r = runner.invoke(main, ["--data-dir", data, "settings", "off", "code.auto_apply"])
    assert r.exit_code == 0 and "= False" in r.output
    r = runner.invoke(main, ["--data-dir", data, "settings", "show", "research.respect_robots_txt"])
    assert r.exit_code == 0 and "robots.txt" in r.output and "Your call" in r.output
    r = runner.invoke(main, ["--data-dir", data, "settings", "reset", "write.rebel"])
    assert r.exit_code == 0 and "reset to False" in r.output
    r = runner.invoke(main, ["--data-dir", data, "settings", "set", "bogus.key", "1"])
    assert r.exit_code != 0 and "unknown switch" in r.output
