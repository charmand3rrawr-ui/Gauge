"""Growing GaugeLM: a bigger model that gives the same answers, so nothing it learned is lost."""
import json
import sys

import numpy as np
import pytest

from gauge.gui import StudioController
from gauge.native import AdamW, GaugeLM, ModelConfig, NativeTrainer, TrainConfig
from gauge.native import grow as g

# The four named sizes, scaled down (heads of 8 instead of 64) so the real code runs in a moment.
SHAPES = [dict(n_layer=4, n_head=4, n_embd=32, n_kv_head=2, block_size=32),
          dict(n_layer=8, n_head=8, n_embd=64, n_kv_head=4, block_size=64),
          dict(n_layer=12, n_head=12, n_embd=96, n_kv_head=4, block_size=128),
          dict(n_layer=24, n_head=16, n_embd=128, n_kv_head=8, block_size=256)]


def _trained(cfg: ModelConfig, steps: int = 12) -> tuple[GaugeLM, np.ndarray]:
    m = GaugeLM(cfg, seed=1)
    rng = np.random.default_rng(0)
    data = rng.integers(0, cfg.vocab_size, size=3000)
    data[1::2] = (data[0::2] * 7 + 3) % cfg.vocab_size
    opt = AdamW(m.p, lr=3e-3)
    T = cfg.block_size
    for _ in range(steps):
        ix = rng.integers(0, len(data) - T - 1, size=4)
        x, y = np.stack([data[i:i + T] for i in ix]), np.stack([data[i + 1:i + T + 1] for i in ix])
        _, _, caches = m.forward(x, y)
        opt.step(m.p, m.backward(caches))
    return m, data


def test_a_grown_model_gives_the_same_answers_through_every_size():
    m, data = _trained(ModelConfig(vocab_size=120, **SHAPES[0]))
    x, y = np.stack([data[i:i + 32] for i in (0, 40)]), np.stack([data[i + 1:i + 33] for i in (0, 40)])
    loss0 = m.forward(x, y)[1]
    for shape in SHAPES[1:]:
        cfg2 = ModelConfig(vocab_size=120, **shape)
        m2 = g.grow(m, cfg2)
        assert m2.n_params() == g.n_params(cfg2) > m.n_params()
        assert g.difference(m, m2) < 1e-4                        # the same numbers out, to rounding
        _, loss, caches = m2.forward(x, y)
        assert abs(loss - loss0) < 1e-4
        # The new parts are switched off but not dead: gradients reach them on the first step.
        grads = m2.backward(caches)
        new_layer = next(i for i in range(cfg2.n_layer) if i not in g.layer_positions(m.cfg.n_layer, cfg2.n_layer))
        assert np.abs(grads[f"l{new_layer}.wo"]).max() > 0 and np.abs(grads[f"l{new_layer}.w2"]).max() > 0
        assert np.abs(grads["wte"][:, m.cfg.n_embd:]).max() > 0   # the new dimensions
        assert np.abs(grads["l0.w2"][m.cfg.hidden:]).max() > 0    # the new feed-forward units
        m = m2
    # A longer context alone changes no weights at all.
    longer = g.grow(m, ModelConfig(vocab_size=120, **{**SHAPES[-1], "block_size": 512}))
    assert all(np.array_equal(longer.p[k], m.p[k]) for k in m.p) and g.difference(m, longer) == 0.0
    # ...and the grown model really can read the longer input.
    assert longer.forward(np.zeros((1, 300), dtype=np.int64))[0].shape == (1, 300, 120)


def test_old_heads_keep_reading_their_own_keys_when_the_grouping_changes():
    # 4 heads sharing 2 key/value heads -> 8 sharing 4: everything stays where it was.
    assert g.head_layout(4, 2, 8, 4) == ({0: 0, 1: 1, 2: 2, 3: 3}, [0, 1, None, None])
    # 8 sharing 4 (pairs) -> 12 sharing 4 (threes): each pair moves to the front of its three.
    q, kv = g.head_layout(8, 4, 12, 4)
    assert kv == [0, 1, 2, 3] and [q[h] // 3 for h in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    # 12 sharing 4 (threes) -> 16 sharing 8 (pairs): a three no longer fits one pair, so its keys are copied.
    q, kv = g.head_layout(12, 4, 16, 8)
    assert kv == [0, 0, 1, 1, 2, 2, 3, 3] and len(set(q.values())) == 12
    assert all(kv[q[h] // 2] == h // 3 for h in range(12))      # every old head still reads its own keys
    assert g.head_layout(12, 4, 16, 4)[1] == [0, 1, 2, 3]       # looser grouping (fours) always fits
    with pytest.raises(g.GrowError) as e:
        g.head_layout(6, 3, 8, 2)                                # fewer key/value heads than it was trained with
    assert "use at least 3 key/value heads" in str(e.value)


def test_impossible_growths_are_refused_with_the_reason():
    old = ModelConfig(vocab_size=120, **SHAPES[1])
    for change, why in ((dict(n_layer=4), "can only grow"), (dict(n_embd=32, n_head=4), "can only grow"),
                        (dict(vocab_size=500), "vocabulary"), (dict(n_embd=128, n_head=8), "attention head must stay 8"),
                        (dict(arch="gpt"), "Llama-style"), ({}, "already is")):
        with pytest.raises(g.GrowError) as e:
            g.check(old, ModelConfig(**{**old.__dict__, **change}))
        assert why in str(e.value), (change, str(e.value))
    with pytest.raises(g.GrowError):
        g.target_config(old, "small", layers=10)                 # a name or numbers, not both
    with pytest.raises(g.GrowError):
        g.target_config(old, width=100)                          # not a multiple of the head size


def test_the_next_size_is_the_next_named_size_or_double():
    from gauge.native import preset_config

    tiny = preset_config("tiny")
    nxt = g.target_config(tiny)
    assert g.size_name(tiny) == "tiny" and g.size_name(nxt) == "small" and nxt.vocab_size == tiny.vocab_size
    assert g.size_name(g.target_config(nxt)) == "medium" and g.size_name(g.target_config(tiny, "large")) == "large"
    assert "small: 8 layers, 512 wide, 8 heads, reads 512 tokens at once" in g.describe(nxt)
    custom = ModelConfig(vocab_size=120, **SHAPES[0])           # no named size has heads of 8: it doubles
    d = g.target_config(custom)
    assert (d.n_layer, d.n_embd, d.n_head, d.kv_heads, d.block_size) == (8, 64, 8, 4, 64) and g.size_name(custom) == ""
    only_deeper = g.target_config(custom, layers=6)
    assert (only_deeper.n_layer, only_deeper.n_embd, only_deeper.n_head) == (6, 32, 4)
    wider = g.target_config(custom, width=48)
    assert (wider.n_head, wider.kv_heads) == (6, 3)


# ------------------------------------------------------------------ on disk, with the real trainer
def _docs() -> list[tuple[str, str]]:
    words = "the courier ran through the rain and the seal broke again while the footsteps waited behind her".split()
    rng = np.random.default_rng(3)
    return [("prose", " ".join(rng.choice(words, size=6000)))]


def _tcfg(steps: int) -> TrainConfig:
    return TrainConfig(steps=steps, batch_size=4, accum_steps=1, lr=2e-3, warmup=5, eval_every=10, checkpoint_every=10,
                       device="cpu", curriculum=False)


def test_growing_the_saved_model_keeps_the_step_and_training_carries_on(tmp_path):
    d = tmp_path / "native"
    tr = NativeTrainer(d)
    small = ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=32, n_kv_head=2, block_size=32)
    first = tr.train(_docs(), small, _tcfg(30))
    old, tok = tr.load()
    plan = g.can_grow(d)
    assert plan["ok"] and plan["times"] > 2 and "1 layers, 32 wide" in plan["now"] and "2 layers, 64 wide" in plan["next"]

    res = g.grow_checkpoint(d)
    assert res["step"] == 30 and res["params_after"] > res["params_before"] and res["difference"] < 1e-4
    assert "kept everything it had learned" in res["message"] and "start training again" in res["message"]
    new, tok2 = tr.load()
    assert (new.cfg.n_layer, new.cfg.n_embd, new.cfg.vocab_size) == (2, 64, old.cfg.vocab_size)
    ids = np.array([tok.encode("the courier ran through the rain")])
    assert np.abs(old.forward(ids)[0] - new.forward(ids)[0]).max() < 1e-4 and tok2.vocab_size == tok.vocab_size
    st = tr.status()
    assert st["step"] == 30 and st["n_params"] == res["params_after"]
    backup = d / g.BACKUP_DIR / res["backup"]
    assert (backup / "weights.npz").exists() and json.loads((backup / "model.json").read_text())["n_params"] == res["params_before"]
    assert g.growth_history(d)[0]["step"] == 30

    # Training resumes on the grown model, from the same step, and the loss does not jump back up.
    more = tr.train(_docs(), small, _tcfg(30), resume=True)
    assert more["step"] == 60 and more["n_params"] == res["params_after"]
    assert more["loss"] < first["loss"] + 0.05

    undone = g.undo_grow(d)
    assert undone["step"] == 30 and "back to how it was at step 30" in undone["message"]
    assert tr.status()["step"] == 30 and tr.status()["n_params"] == res["params_before"]
    assert not backup.exists() and g.growth_history(d) == []
    with pytest.raises(g.GrowError):
        g.undo_grow(d)


def test_a_failed_check_changes_nothing_on_disk(tmp_path, monkeypatch):
    d = tmp_path / "native"
    NativeTrainer(d).train(_docs(), ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=32, n_kv_head=2, block_size=32), _tcfg(10))
    before = (d / "weights.npz").read_bytes()
    monkeypatch.setattr(g, "difference", lambda *a, **k: 0.5)
    with pytest.raises(g.GrowError) as e:
        g.grow_checkpoint(d)
    assert "nothing was changed" in str(e.value)
    assert (d / "weights.npz").read_bytes() == before and not (d / g.BACKUP_DIR).exists() and not (d / g.HISTORY).exists()
    assert not g.can_grow(tmp_path / "empty")["ok"]
    with pytest.raises(g.GrowError):
        g.grow_checkpoint(tmp_path / "empty")


# ------------------------------------------------------------------ the command line and the Studio
def _small_gaugelm(cfg):
    n = cfg.native
    n.size, n.n_layer, n.n_head, n.n_kv_head, n.n_embd, n.block_size, n.vocab_size = "", 1, 2, 2, 32, 32, 400
    n.steps, n.batch_size, n.accum_steps, n.device = 12, 4, 1, "cpu"


def test_the_studio_grows_gaugelm_and_can_undo_it(cfg, db, tmp_path):
    from gauge.config import GaugeConfig
    from tests.test_teach import _novel_folder

    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    assert "not been trained yet" in ctl.gaugelm_size()["line"]
    assert not ctl.grow_gaugelm()["grown"]
    ctl.add_material([str(_novel_folder(tmp_path))])
    ctl.use_gaugelm(True)
    step = ctl.train_gaugelm(minutes=0.01, teach=False)["step"]
    line = ctl.gaugelm_size()["line"]
    assert "It can grow to 2 layers, 64 wide" in line and "times the weights" in line

    ctl._training["running"] = True                              # never while it is training
    assert "Pause training" in ctl.grow_gaugelm()["message"] and "Pause training" in ctl.undo_grow_gaugelm()["message"]
    ctl._training["running"] = False

    res = ctl.grow_gaugelm()
    assert res["grown"] and "Undo the last growth" in res["message"]
    saved = GaugeConfig.load(cfg.data_dir)                       # the settings follow the model
    assert (saved.native.n_layer, saved.native.n_embd, saved.native.block_size, saved.llm.num_ctx) == (2, 64, 64, 64)
    assert ctl.engine()["ready"]                                 # it can still write straight away
    after = ctl.train_gaugelm(minutes=0.01, teach=False)         # and training carries on with the bigger model
    assert after["trained"] and after["step"] > step and after["resumed"]
    assert NativeTrainer(cfg.native_dir).status()["config"]["n_embd"] == 64

    back = ctl.undo_grow_gaugelm()
    assert back["undone"] and GaugeConfig.load(cfg.data_dir).native.n_embd == 32
    assert "nothing to undo" in ctl.undo_grow_gaugelm()["message"]


def test_grow_from_the_command_line(tmp_path):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "200"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    r = runner.invoke(main, ["--data-dir", data, "native", "grow"])
    assert r.exit_code == 1 and "not been trained yet" in r.output
    NativeTrainer(tmp_path / "data" / "models" / "native").train(
        _docs(), ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=32, n_kv_head=2, block_size=32), _tcfg(10))
    r = runner.invoke(main, ["--data-dir", data, "native", "grow", "--plan", "--layers", "3"])
    assert r.exit_code == 0 and "after: 3 layers, 32 wide" in r.output and "Nothing was changed" in r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "grow", "--to", "small"])
    assert r.exit_code == 1 and "attention head must stay 16" in r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "grow", "--layers", "2", "--width", "64"])
    assert r.exit_code == 0 and "after: 2 layers, 64 wide, 4 heads" in r.output and "kept everything it had learned" in r.output
    assert "native train --resume" in r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "grow", "--undo"])
    assert r.exit_code == 0 and "back to how it was at step 10" in r.output


def test_the_window_has_grow_and_undo(cfg, db, tmp_path, monkeypatch):
    from gauge import gui
    from tests.test_teach import _fake_tkinter, _novel_folder, _wait_for_jobs, _Widget

    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    ctl.train_gaugelm(minutes=0.01, teach=False)
    mods, shown = _fake_tkinter(tmp_path)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    _Widget.buttons["Change the AI"]()
    assert {"Grow GaugeLM", "Undo the last growth"} <= set(_Widget.buttons)
    _Widget.buttons["Grow GaugeLM"]()                           # the yes/no stand-in answers yes
    _wait_for_jobs()
    asked = [s for s in shown if s[0] == "ask"]
    assert asked and "keeps everything it has learned" in asked[-1][1][1] and "2 layers, 64 wide" in asked[-1][1][1]
    assert NativeTrainer(cfg.native_dir).status()["config"]["n_embd"] == 64
    _Widget.buttons["Undo the last growth"]()
    _wait_for_jobs()
    assert NativeTrainer(cfg.native_dir).status()["config"]["n_embd"] == 32
    assert not [s for s in shown if s[0] == "error"], shown
