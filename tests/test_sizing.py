"""Sizes up to 20 trillion weights: what each takes, and the check that it fits."""
import pytest

from gauge.native import PRESETS, ModelConfig, NativeTrainer, TrainConfig, sizing
from gauge.native import grow as g


def test_the_ladder_reaches_twenty_trillion_weights():
    rows = sizing.ladder()
    assert [r["size"] for r in rows] == ["tiny", "small", "medium", "large", "2b", "20b", "160b", "1t", "5t", "20t"]
    weights = [r["weights"] for r in rows]
    assert weights == sorted(weights)
    top = rows[-1]
    assert abs(top["weights"] - 20e12) / 20e12 < 0.01                        # twenty trillion, to within one per cent
    assert sizing.count(top["weights"]) == "20.0 trillion"
    assert sizing.memory(top["hold_bytes"]) == "80.2 TB" and sizing.memory(top["train_bytes"]) == "320.6 TB"
    assert "roughly 13 times all the good text thought to exist" in top["line"]
    assert "times all the good text" not in rows[0]["line"]                  # a small model is easy to feed
    # The count used for planning is the count of the model that is actually built.
    from gauge.native import GaugeLM
    tiny = ModelConfig(**PRESETS["tiny"])
    assert GaugeLM(tiny).n_params() == sizing.weights(tiny)


def test_every_size_can_be_grown_into_the_next_all_the_way_up():
    cfg = ModelConfig(**PRESETS["tiny"])
    reached = ["tiny"]
    for _ in range(12):
        try:
            cfg = g.target_config(cfg)
        except g.GrowError:
            break
        name = g.size_name(cfg)
        if not name:
            break
        reached.append(name)
    assert reached == list(PRESETS)                                          # tiny ... large ... 20t, one step at a time
    for a, b in zip(list(PRESETS), list(PRESETS)[1:]):
        small, big = ModelConfig(**PRESETS[a]), ModelConfig(**{**PRESETS[b], "vocab_size": PRESETS[a]["vocab_size"]})
        g.check(small, big)                                                  # head size, heads and grouping all allow it
        assert g.n_params(big) <= g.STEP_LIMIT * g.n_params(small) * 1.2
    assert g.target_config(ModelConfig(**PRESETS["tiny"]), "20t").n_layer == 424
    assert "20.0 trillion weights" in g.describe(ModelConfig(**PRESETS["20t"]))


def test_numbers_are_said_in_words():
    assert [sizing.count(n) for n in (950, 3.9e6, 25.2e9, 20e12, 4e15)] == ["950", "3.9 million", "25.2 billion", "20.0 trillion",
                                                                            "4.0 quadrillion"]
    assert [sizing.memory(n) for n in (5e5, 15.7e6, 9.6e9, 80.2e12, 3e15)] == ["500 KB", "15.7 MB", "9.6 GB", "80.2 TB", "3.0 PB"]


def test_a_size_that_does_not_fit_is_refused_with_the_numbers(monkeypatch, tmp_path):
    big = ModelConfig(**PRESETS["20t"])
    monkeypatch.setattr(sizing, "available_bytes", lambda device="cpu": 32e9)
    monkeypatch.setattr(sizing, "CHECK", True)
    with pytest.raises(sizing.SizeError) as e:
        sizing.ensure_fits(big)
    said = str(e.value)
    assert "20.0 trillion weights" in said and "320.6 TB" in said and "this computer has 32.0 GB" in said and "native.memory_check" in said
    sizing.ensure_fits(ModelConfig(**PRESETS["tiny"]))                       # a size that fits passes silently
    with pytest.raises(sizing.SizeError):
        sizing.ensure_fits(big, training=False)                              # too big even just to run
    # Training refuses before anything is built or any tokenizer is trained.
    docs = [("prose", "the courier ran through the rain " * 400)]
    with pytest.raises(sizing.SizeError):
        NativeTrainer(tmp_path / "n").train(docs, big, TrainConfig(steps=1, device="cpu"))
    assert not (tmp_path / "n" / "tokenizer.json").exists()
    # A machine Gauge cannot measure, or the check switched off: the run is allowed to proceed.
    monkeypatch.setattr(sizing, "available_bytes", lambda device="cpu": 0.0)
    sizing.ensure_fits(big)
    monkeypatch.setattr(sizing, "available_bytes", lambda device="cpu": 32e9)
    monkeypatch.setattr(sizing, "CHECK", False)
    sizing.ensure_fits(big)


def test_growing_past_what_fits_is_refused_and_changes_nothing(monkeypatch, tmp_path):
    import numpy as np

    d = tmp_path / "native"
    words = "the courier ran through the rain and the seal broke again".split()
    docs = [("prose", " ".join(np.random.default_rng(3).choice(words, size=4000)))]
    NativeTrainer(d).train(docs, ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=32, n_kv_head=2, block_size=32),
                           TrainConfig(steps=5, batch_size=4, accum_steps=1, warmup=1, eval_every=5, checkpoint_every=5, device="cpu",
                                       curriculum=False))
    before = (d / "weights.npz").read_bytes()
    monkeypatch.setattr(sizing, "CHECK", True)
    monkeypatch.setattr(sizing, "available_bytes", lambda device="cpu": 1e5)  # a machine with almost no memory
    with pytest.raises(g.GrowError) as e:
        g.grow_checkpoint(d)
    assert "needs about" in str(e.value) and (d / "weights.npz").read_bytes() == before and not (d / g.BACKUP_DIR).exists()


def test_size_commands(cfg, monkeypatch):
    from click.testing import CliRunner

    from gauge.cli import main
    from gauge.config import GaugeConfig

    cfg.save()
    runner = CliRunner(env={"COLUMNS": "220"})
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "native", "presets"])
    assert r.exit_code == 0 and "20t" in r.output and "20.0 trillion" in r.output and "80.2 TB" in r.output
    monkeypatch.setattr(sizing, "available_bytes", lambda device="cpu": 32e9)
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "native", "needs", "20t"])
    assert r.exit_code == 0 and "424 layers, 65,536 wide, 1024 heads" in r.output and "To train it: about 320.6 TB" in r.output
    assert "times too little to train it" in r.output and "10,0" in r.output      # about ten thousand times too little
    assert runner.invoke(main, ["--data-dir", cfg.data_dir, "native", "needs", "enormous"]).exit_code == 1
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "settings", "set", "native.memory_check", "false"])
    assert r.exit_code == 0 and GaugeConfig.load(cfg.data_dir).native.memory_check is False and sizing.CHECK is False
    sizing.CHECK = True
