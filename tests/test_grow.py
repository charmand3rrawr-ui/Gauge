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
                        (dict(vocab_size=500), "vocabulary grows separately"),
                        (dict(n_embd=96, n_head=8), "whole multiple"),             # heads of 12 cannot hold heads of 8
                        (dict(arch="gpt"), "cannot change the kind of model"), ({}, "already is")):
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
    assert res["grown"] and "Undo the last size change" in res["message"]
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
    r = runner.invoke(main, ["--data-dir", data, "native", "grow", "--width", "40"])
    assert r.exit_code == 1 and "multiple of 16" in r.output
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
    assert {"Grow GaugeLM", "Grow the vocabulary", "Shrink GaugeLM", "Undo the last size change"} <= set(_Widget.buttons)
    _Widget.buttons["Grow GaugeLM"]()                           # the yes/no stand-in answers yes
    _wait_for_jobs()
    asked = [s for s in shown if s[0] == "ask"]
    assert asked and "keeps everything it has learned" in asked[-1][1][1] and "2 layers, 64 wide" in asked[-1][1][1]
    assert NativeTrainer(cfg.native_dir).status()["config"]["n_embd"] == 64
    _Widget.buttons["Undo the last size change"]()
    _wait_for_jobs()
    assert NativeTrainer(cfg.native_dir).status()["config"]["n_embd"] == 32
    before = NativeTrainer(cfg.native_dir).status()["config"]["vocab_size"]
    _Widget.buttons["Grow the vocabulary"]()
    _wait_for_jobs()
    assert "It keeps every token it knows" in [s for s in shown if s[0] == "ask"][-1][1][1]
    assert NativeTrainer(cfg.native_dir).status()["config"]["vocab_size"] > before
    _Widget.buttons["Grow GaugeLM"]()
    _wait_for_jobs()
    called: list = []
    monkeypatch.setattr(StudioController, "shrink_gaugelm",
                        lambda self, *a, **k: (called.append(1), {"shrunk": False, "message": "not this time"})[1])
    _Widget.buttons["Shrink GaugeLM"]()                         # the button is wired; the teaching itself is tested below
    _wait_for_jobs()
    assert called
    assert "A smaller model holds less" in [s for s in shown if s[0] == "ask"][-1][1][1]
    assert not [s for s in shown if s[0] == "error"], shown


# ------------------------------------------------------------------ bigger heads, and the older kind of model
def test_heads_can_grow_to_a_whole_multiple_of_their_size():
    m, data = _trained(ModelConfig(vocab_size=90, n_layer=2, n_head=4, n_embd=32, n_kv_head=2, block_size=32))
    x = np.stack([data[i:i + 32] % 90 for i in (0, 40)])
    for kw in (dict(n_embd=64, n_head=4, n_kv_head=2),               # same heads, twice the size
               dict(n_embd=96, n_head=4, n_kv_head=4),               # three times the size, ungrouped
               dict(n_embd=128, n_head=8, n_kv_head=4, n_layer=3)):  # bigger and more heads, and deeper
        cfg = ModelConfig(**{**m.cfg.__dict__, **kw})
        big = g.grow(m, cfg)
        assert cfg.head_dim % m.cfg.head_dim == 0 and cfg.head_dim > m.cfg.head_dim
        assert g.difference(m, big) < 1e-4 and big.n_params() == g.n_params(cfg)
        _, _, caches = big.forward(x, x)
        assert np.abs(big.backward(caches)["l0.wq"]).max() > 0
        back, was_cut = g.cut_down(big, m.cfg)                      # and cutting it back gives the model it was
        assert was_cut and all(np.allclose(back.p[k], m.p[k], atol=1e-5) for k in m.p)


def test_an_older_gpt_style_model_grows_deeper_and_longer_but_not_wider():
    old = GaugeLM(ModelConfig(vocab_size=90, n_layer=2, n_head=4, n_embd=32, block_size=16, arch="gpt"), seed=2)
    cfg = g.target_config(old.cfg)
    assert (cfg.n_layer, cfg.block_size, cfg.n_embd, cfg.arch) == (4, 32, 32, "gpt")
    new = g.grow(old, cfg)
    assert g.difference(old, new) < 1e-5 and new.n_params() == g.n_params(cfg) and old.n_params() == g.n_params(old.cfg)
    assert new.forward(np.zeros((1, 30), dtype=np.int64))[0].shape == (1, 30, 90)       # reads past the old limit
    with pytest.raises(g.GrowError) as e:
        g.target_config(old.cfg, width=64)
    assert "deeper and read more at once, but not wider" in str(e.value)
    with pytest.raises(g.GrowError):
        g.smaller_config(old.cfg)


# ------------------------------------------------------------------ the vocabulary
def test_a_tokenizer_can_be_extended_and_keeps_every_token():
    from gauge.native import BPETokenizer

    text = _docs()[0][1] + " the lighthouse keeper counted the ships " * 200
    small = BPETokenizer.train(text, 300)
    more = small.extended(text, 340)
    assert more.vocab_size > small.vocab_size and more.merges[:len(small.merges)] == small.merges
    assert more.merges == BPETokenizer.train(text, more.vocab_size).merges      # the same as learning it all at once
    line = "the lighthouse keeper counted the ships"
    assert more.decode(more.encode(line)) == line and len(more.encode(line)) < len(small.encode(line))
    assert more.special["<bos>"] == small.special["<bos>"] + (more.vocab_size - small.vocab_size)
    assert small.extended("xyz", 400).vocab_size == small.vocab_size            # nothing repeats: nothing to add


def test_growing_the_vocabulary_keeps_every_old_token_and_training_recovers(tmp_path):
    from gauge.native import BPETokenizer

    d = tmp_path / "native"
    tr = NativeTrainer(d)
    docs = [("prose", _docs()[0][1] + " the lighthouse keeper counted the ships as the tide went out " * 300)]
    tr.train(docs, ModelConfig(vocab_size=285, n_layer=1, n_head=2, n_embd=32, n_kv_head=2, block_size=32), _tcfg(60))
    old, tok = tr.load()
    texts = [t for _, t in docs]
    with pytest.raises(g.GrowError):
        g.grow_vocab_checkpoint(d, texts, tok.vocab_size)                       # it can only grow
    with pytest.raises(g.GrowError):
        g.grow_vocab_checkpoint(d, [""], 400)
    res = g.grow_vocab_checkpoint(d, texts, tok.vocab_size + 25)
    new, tok2 = tr.load()
    assert res["kind"] == "vocabulary" and res["added"] == tok2.vocab_size - tok.vocab_size > 0
    assert new.cfg.vocab_size == tok2.vocab_size and res["difference"] < 1e-5 and 0 < res["fewer_tokens"] < 1
    assert "kept every token it knew" in res["message"] and "Start training again" in res["message"]
    n_old = len(tok.vocab)
    assert np.array_equal(new.p["wte"][:n_old], old.p["wte"][:n_old])           # old tokens: the same rows
    assert np.array_equal(new.p["wte"][-5:], old.p["wte"][-5:])                 # the special tokens moved to the end
    a, b = tok2.merges[n_old - 256]
    assert np.allclose(new.p["wte"][n_old], (old.p["wte"][a] + old.p["wte"][b]) / 2)   # a new token starts as its two parts
    assert all(np.array_equal(new.p[k], old.p[k]) for k in old.p if k != "wte")
    # Text cut the old way scores every old token exactly as it did.
    ids = np.array([tok.encode("the courier ran through the rain")])
    assert np.array_equal(old.forward(ids)[0][..., :n_old], new.forward(ids)[0][..., :n_old])
    # The new tokens are untrained; a short session teaches the model to use them.
    sample = docs[0][1][:3000]
    assert res["loss_per_char_before"] > 0 and res["loss_per_char_after"] > 0
    more = tr.train(docs, new.cfg, _tcfg(60), resume=True)
    m3, t3 = tr.load()
    assert more["step"] == 120 and t3.vocab_size == tok2.vocab_size
    assert g.loss_per_char(m3, t3, sample) < g.loss_per_char(new, tok2, sample)
    # Undo puts back the old tokenizer as well as the old weights.
    back = g.undo_grow(d)
    assert back["kind"] == "vocabulary" and "before its vocabulary grew" in back["message"]
    assert BPETokenizer.load(d / "tokenizer.json").vocab_size == tok.vocab_size
    assert tr.status()["config"]["vocab_size"] == tok.vocab_size and tr.status()["step"] == 60


# ------------------------------------------------------------------ shrinking
def test_teacher_targets_have_the_right_gradient():
    from gauge.native.model import softmax

    m = GaugeLM(ModelConfig(vocab_size=30, n_layer=1, n_head=2, n_embd=16, n_kv_head=1, block_size=8), seed=0)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 30, (2, 8)), rng.integers(0, 30, (2, 8))
    soft = softmax(np, rng.standard_normal((2, 8, 30)))
    plain = m.forward(x, y)[1]
    _, mixed, caches = m.forward(x, y, soft=soft, soft_weight=0.6)
    grads = m.backward(caches)
    assert mixed != plain
    for name in ("wte", "l0.wq", "l0.w2", "lnf_g"):
        i = tuple(int(rng.integers(0, n)) for n in m.p[name].shape)
        keep = float(m.p[name][i])
        m.p[name][i] = keep + 1e-3
        up = m.forward(x, y, soft=soft, soft_weight=0.6)[1]
        m.p[name][i] = keep - 1e-3
        down = m.forward(x, y, soft=soft, soft_weight=0.6)[1]
        m.p[name][i] = keep
        assert abs((up - down) / 2e-3 - float(grads[name][i])) < 1e-3, name


def test_shrinking_cuts_the_model_down_teaches_it_and_can_be_undone(tmp_path):
    d = tmp_path / "native"
    tr = NativeTrainer(d)
    small = ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=32, n_kv_head=2, block_size=32)
    tr.train(_docs(), small, _tcfg(40))
    g.grow_checkpoint(d)
    tr.train(_docs(), small, _tcfg(40), resume=True)
    big, tok = tr.load()
    assert (big.cfg.n_layer, big.cfg.n_embd) == (2, 64)
    # With nothing said, it goes back to the shape it had before it grew.
    assert g.smaller_config(big.cfg, history=g.growth_history(d)).n_embd == 32
    seen: list = []
    res = g.shrink_checkpoint(d, _docs(), steps=30, batch_size=4, lr=2e-3, progress=seen.append)
    now, _ = tr.load()
    assert res["kind"] == "shrink" and res["cut_from_big"] and res["taught_steps"] == 30 and seen[-1]["step"] == 30
    assert (now.cfg.n_layer, now.cfg.n_embd, now.cfg.vocab_size) == (1, 32, big.cfg.vocab_size)
    assert res["params_after"] < res["params_before"] and "bigger model is kept" in res["message"]
    # Being taught made it better than the bare cut, and it stays close to the big model.
    assert res["loss_per_char_after"] < res["loss_per_char_cut"]
    assert res["loss_per_char_after"] < res["loss_per_char_before"] * 1.25
    assert tr.train(_docs(), small, _tcfg(10), resume=True)["n_params"] == res["params_after"]   # and trains on
    back = g.undo_grow(d)                                           # the big model again
    assert back["kind"] == "shrink" and tr.status()["config"]["n_embd"] == 64
    assert g.undo_grow(d)["kind"] == "shape" and tr.status()["config"]["n_embd"] == 32      # and back before the growth

    stop = g.shrink_checkpoint(d, _docs(), layers=1, width=16, heads=1, kv_heads=1, steps=1000, should_stop=lambda: True)
    assert stop["taught_steps"] == 1 and tr.status()["config"]["n_embd"] == 16               # a pause ends the teaching
    with pytest.raises(g.GrowError):
        g.shrink_checkpoint(d, _docs(), layers=4, width=64)                                    # not smaller
    with pytest.raises(g.GrowError):
        g.shrink_checkpoint(tmp_path / "empty", _docs())


def test_a_shape_that_cannot_be_cut_starts_fresh_and_is_still_taught():
    big = GaugeLM(ModelConfig(vocab_size=90, n_layer=2, n_head=4, n_embd=48, n_kv_head=4, block_size=32), seed=1)
    odd = ModelConfig(vocab_size=90, n_layer=1, n_head=2, n_embd=40, n_kv_head=2, block_size=32)   # heads of 20 from heads of 12
    model, was_cut = g.cut_down(big, odd)
    assert not was_cut and model.n_params() == g.n_params(odd)


def test_vocabulary_and_shrink_from_the_command_line_and_the_studio(cfg, db, tmp_path):
    from click.testing import CliRunner

    from gauge.cli import main
    from gauge.config import GaugeConfig
    from tests.test_teach import _novel_folder

    _small_gaugelm(cfg)
    cfg.native.vocab_size = 300
    cfg.save()
    ctl = StudioController(cfg, db)
    assert not ctl.grow_vocabulary()["grown"] and not ctl.shrink_gaugelm()["shrunk"]       # nothing trained yet
    ctl.add_material([str(_novel_folder(tmp_path))])
    ctl.train_gaugelm(minutes=0.01, teach=False)
    v0 = NativeTrainer(cfg.native_dir).status()["config"]["vocab_size"]
    assert f"Its vocabulary is {v0} tokens and can grow to 4096" in ctl.gaugelm_size()["line"]

    res = ctl.grow_vocabulary(v0 + 20)
    assert res["grown"] and res["vocab"] > v0 and "Undo the last size change" in res["message"]
    assert GaugeConfig.load(cfg.data_dir).native.vocab_size == res["vocab"]
    assert ctl.train_gaugelm(minutes=0.01, teach=False)["trained"]                         # trains on with the new tokens
    assert ctl.grow_gaugelm()["grown"]
    ctl._training["running"] = True
    assert "Pause training" in ctl.shrink_gaugelm()["message"] and "Pause training" in ctl.grow_vocabulary()["message"]
    ctl._training["running"] = False
    shrunk = ctl.shrink_gaugelm(minutes=0.02)
    assert shrunk["shrunk"] and shrunk["taught_steps"] >= 1 and not ctl.training()["running"]
    assert GaugeConfig.load(cfg.data_dir).native.n_embd == 32 and ctl.engine()["ready"]
    assert ctl.undo_grow_gaugelm()["undone"] and GaugeConfig.load(cfg.data_dir).native.n_embd == 64

    runner = CliRunner(env={"COLUMNS": "200"})
    data = cfg.data_dir
    v1 = NativeTrainer(cfg.native_dir).status()["config"]["vocab_size"]
    r = runner.invoke(main, ["--data-dir", data, "native", "grow", "--vocab", str(v1 + 10)])
    assert r.exit_code == 0 and "vocabulary grew from" in r.output and "native grow --undo" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "grow", "--plan"])
    assert r.exit_code == 0 and f"vocabulary {NativeTrainer(cfg.native_dir).status()['config']['vocab_size']} tokens" in r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "shrink", "--steps", "5"])
    assert r.exit_code == 0 and "GaugeLM shrank from" in r.output and "taught 5/5" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "shrink", "--layers", "9", "--steps", "5"])
    assert r.exit_code == 1 and "not smaller" in r.output
