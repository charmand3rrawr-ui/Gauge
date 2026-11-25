"""The device switch: choose one or several sensed devices; several share every training step."""
import numpy as np
import pytest

from gauge.compute import devices as dv
from gauge.compute.detect import Device, parse_nvidia_smi
from gauge.gui import StudioController
from gauge.native import GaugeLM, ModelConfig, NativeTrainer, TrainConfig
from gauge.native.parallel import Replicas, split_device


def _machine(ready=True, cards=2):
    cpu = Device("cpu", "cpu", "Ryzen 7 (16 threads)", None, "numpy", True, source="os")
    out = [cpu]
    names = [("RTX 3060", 12288), ("RTX 4090", 24564), ("RTX 3060", 12288)]
    for i in range(cards):
        name, mem = names[i]
        out.append(Device("gpu", "nvidia", name, mem, "cupy", ready, source="nvidia-smi", index=i,
                          install_hint="" if ready else "pip install cupy-cuda12x"))
    return out


@pytest.fixture(autouse=True)
def _fresh_environment(monkeypatch):
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(dv._applied, "visible", None)


# ------------------------------------------------------------------ what is sensed, and what a choice means
def test_two_cards_of_the_same_model_are_two_devices(monkeypatch):
    from gauge.compute import detect

    monkeypatch.setattr(detect.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(detect, "_run", lambda cmd, timeout=8.0: "RTX 3060, 12288, 551.2\nRTX 3060, 12288, 551.2\n")
    cards = [d for d in detect.scan() if d.kind == "gpu"]
    assert [d.key for d in cards] == ["gpu0", "gpu1"] and detect.scan()[0].key == "cpu"
    assert [d.index for d in parse_nvidia_smi("A, 1, x\nB, 2, x")] == [0, 0]            # numbered by scan, not the parser


def test_automatic_picks_the_biggest_ready_card_or_the_processor(cfg):
    r = dv.resolve(cfg, _machine())
    assert r["automatic"] and r["use"] == ["gpu1"] and r["native_device"] == "cuda" and r["train_devices"] == []
    assert r["engine_gpu"] and r["visible"] == ""                                       # nothing is hidden from the program
    assert "Graphics card 2: RTX 4090, 24 GB (chosen automatically)" in dv.describe(cfg, _machine())
    none_ready = dv.resolve(cfg, _machine(ready=False))
    assert none_ready["use"] == ["cpu"] and none_ready["native_device"] == "cpu"
    assert "cannot use it until its driver library is installed" in none_ready["notes"][0]
    assert dv.resolve(cfg, _machine(cards=0))["use"] == ["cpu"]


def test_choosing_one_or_several_devices(cfg):
    m = _machine(cards=3)
    one = dv.choose(cfg, ["gpu0"], m)
    assert one["use"] == ["gpu0"] and one["visible"] == "0" and one["train_devices"] == [] and cfg.native.device == "cuda"
    both = dv.choose(cfg, ["gpu0", "gpu1"], m)
    assert both["gpus"] == ["gpu1", "gpu0"]                                             # the model lives on the largest card
    assert both["visible"] == "1,0" and both["train_devices"] == ["cuda:0", "cuda:1"]
    assert "Graphics card 1: RTX 3060, 12 GB and Graphics card 2: RTX 4090, 24 GB" in both["message"]
    mixed = dv.choose(cfg, ["cpu", "gpu2"], m)
    assert mixed["train_devices"] == ["cuda:0", "cpu"] and mixed["visible"] == "2" and mixed["engine_gpu"]
    only_cpu = dv.choose(cfg, ["cpu"], m)
    assert only_cpu["native_device"] == "cpu" and cfg.native.device == "cpu" and not only_cpu["engine_gpu"]
    assert not cfg.llm.use_gpu and "downloaded model is kept on the processor" in only_cpu["message"]
    auto = dv.choose(cfg, ["auto"], m)
    assert auto["automatic"] and cfg.compute.devices == [] and cfg.native.device == "auto" and cfg.llm.use_gpu
    with pytest.raises(ValueError) as e:
        dv.choose(cfg, ["gpu7"], m)
    assert "no such device: gpu7" in str(e.value) and "cpu, gpu0, gpu1, gpu2" in str(e.value)


def test_a_chosen_card_that_is_not_ready_falls_back_and_says_why(cfg):
    m = _machine(ready=False, cards=1)
    r = dv.choose(cfg, ["gpu0"], m)
    assert r["use"] == ["cpu"] and r["native_device"] == "cpu"
    assert any("was chosen but is not ready" in n for n in r["notes"]) and any("processor is used" in n for n in r["notes"])
    rows = dv.sensed(cfg, m)
    card = next(x for x in rows if x["key"] == "gpu0")
    assert not card["ready"] and card["can_set_up"] and "Set up the graphics card" in card["note"]
    cfg.compute.devices = ["gpu5"]                                                      # unplugged since it was chosen
    assert "is not on this computer now" in dv.resolve(cfg, m)["notes"][0]


def test_the_choice_is_saved_and_reaches_the_engine(cfg):
    from gauge.config import GaugeConfig
    from gauge.llm.client import OllamaLLM

    dv.choose(cfg, ["cpu"], _machine())
    saved = GaugeConfig.load(cfg.data_dir)
    assert saved.compute.devices == ["cpu"] and saved.llm.use_gpu is False and saved.native.device == "cpu"
    saved.llm.provider = "ollama"
    assert OllamaLLM(saved.llm, "x")._payload("hi", "", None, None, False, None)["options"]["num_gpu"] == 0
    dv.choose(cfg, ["gpu1"], _machine())
    again = GaugeConfig.load(cfg.data_dir)
    assert again.llm.use_gpu and "num_gpu" not in OllamaLLM(again.llm, "x")._payload("hi", "", None, None, False, None)["options"]


def test_only_the_chosen_cards_are_visible_to_the_program(cfg, monkeypatch):
    import os
    import sys

    assert dv.apply_environment(cfg) == "nothing" and "CUDA_VISIBLE_DEVICES" not in os.environ
    cfg.compute.devices = ["gpu1", "gpu0", "cpu"]
    assert dv.apply_environment(cfg) == "set"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,0" and os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert dv.apply_environment(cfg) == "unchanged"
    monkeypatch.setitem(sys.modules, "cupy", object())                                  # the graphics library has started
    cfg.compute.devices = ["gpu0"]
    assert dv.apply_environment(cfg) == "restart" and os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    cfg.compute.devices = []
    dv.apply_environment(cfg)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert dv.train_devices(cfg) == []                                                  # one device or automatic: an ordinary run


# ------------------------------------------------------------------ setting a card up
def test_setting_up_the_graphics_card_runs_one_command_and_reports_plainly(monkeypatch):
    import sys
    import types

    cmd = dv.setup_command()
    assert cmd[:4] == [sys.executable, "-m", "pip", "install"] and cmd[-1] == "cupy-cuda12x[ctk]"
    with pytest.raises(ValueError):
        dv.setup_command("amd")
    ran = []

    def pip(ok):
        def run(c, **kw):
            ran.append(c)
            return types.SimpleNamespace(returncode=0 if ok else 1, stdout="", stderr="ERROR: No matching distribution found")
        return run

    status = {"cupy": False, "cupy_devices": 0}
    monkeypatch.setattr(dv, "backend_status", lambda: dict(status))
    failed = dv.setup(run=pip(False))
    assert not failed["ok"] and "No matching distribution" in failed["message"] and "NVIDIA driver" in failed["message"]
    later = dv.setup(run=pip(True))                                                     # installed, not yet visible
    assert later["installed"] and not later["ok"] and "Close and reopen Gauge" in later["message"]
    status.update(cupy=True, cupy_devices=1)
    assert dv.setup(run=pip(True))["message"] == "The graphics card is already set up for GaugeLM."
    assert len(ran) == 2                                                                # nothing is downloaded twice

    from gauge import dryrun
    status.update(cupy=False, cupy_devices=0)
    dryrun.set_dry_run(True)
    try:
        held = dv.setup(run=pip(True))
    finally:
        dryrun.set_dry_run(False)
        dryrun.reset()
    assert held["message"].startswith("Dry run: would run") and len(ran) == 2


# ------------------------------------------------------------------ several devices share a step
def _model():
    return GaugeLM(ModelConfig(vocab_size=60, n_layer=2, n_head=2, n_embd=32, n_kv_head=1, block_size=16), seed=1)


def _batches(n, rng):
    return [(rng.integers(0, 60, (4, 16)), rng.integers(0, 60, (4, 16))) for _ in range(n)]


def test_devices_sharing_a_step_equal_one_big_batch():
    rng = np.random.default_rng(0)
    bs = _batches(3, rng)
    ref = _model()
    _, loss_ref, caches = ref.forward(np.concatenate([b[0] for b in bs]), np.concatenate([b[1] for b in bs]))
    g_ref = ref.backward(caches)
    team = Replicas(_model(), ["cpu", "cpu", "cpu"])
    grads, loss, tokens = team.step(bs)
    assert len(team) == 3 and tokens == 192 and abs(loss - loss_ref) < 1e-6
    assert max(float(np.abs(grads[k] - g_ref[k]).max()) for k in grads) < 1e-6
    # After the first device steps, the others are brought back in line with it.
    team.primary.model.p["wte"] += 1.0
    team.sync()
    assert all(np.array_equal(r.model.p["wte"], team.primary.model.p["wte"]) for r in team.others)
    assert team.describe() == "Training on cpu and cpu and cpu."
    with pytest.raises(ValueError):
        team.step(bs[:2])
    assert split_device("cuda:1") == ("cuda", 1) and split_device("cpu") == ("cpu", 0)


def test_a_device_that_fails_or_cannot_keep_up_is_rested_and_training_carries_on(monkeypatch):
    rng = np.random.default_rng(0)
    team = Replicas(_model(), ["cpu", "cpu", "cpu"])
    slow, broken = team.others

    def breaks(*a, **k):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(broken.model, "forward", breaks)
    real = slow.model.forward

    def crawls(*a, **k):
        import time
        time.sleep(0.4)
        return real(*a, **k)

    monkeypatch.setattr(slow.model, "forward", crawls)
    grads, loss, tokens = team.step(_batches(3, rng))
    assert tokens == 128 and np.isfinite(loss)                     # the broken one's batch is not counted
    assert len(team) == 1 and len(team.rested) == 2 and "rested" in team.describe()
    grads, loss, tokens = team.step(_batches(1, rng))               # carries on alone
    assert tokens == 64
    # A card that cannot even be started is left out from the beginning.
    lone = Replicas(_model(), ["cpu", "cuda:3"])
    assert len(lone) == 1 and lone.rested == ["cuda:3"]


def test_the_trainer_shares_steps_across_chosen_devices(tmp_path):
    words = "the courier ran through the rain and the seal broke again while the footsteps waited behind her".split()
    docs = [("prose", " ".join(np.random.default_rng(3).choice(words, size=6000)))]
    mcfg = ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=32, n_kv_head=2, block_size=32)
    tc = TrainConfig(steps=12, batch_size=4, accum_steps=1, lr=2e-3, warmup=3, eval_every=6, checkpoint_every=6,
                     device="cpu", curriculum=False, devices=["cpu", "cpu"])
    res = NativeTrainer(tmp_path / "two").train(docs, mcfg, tc)
    assert res["devices"] == ["cpu", "cpu"] and res["rested"] == [] and res["step"] == 12
    assert res["tokens_seen"] == 12 * 2 * 4 * 32                   # twice the text per step
    single = NativeTrainer(tmp_path / "one").train(docs, mcfg, TrainConfig(**{**tc.__dict__, "devices": []}))
    assert single["devices"] == [] and single["tokens_seen"] == 12 * 4 * 32
    assert np.isfinite(res["loss"]) and res["loss"] < 6.0


# ------------------------------------------------------------------ the Studio and the command line
def test_the_studio_switches_devices_on_and_off(cfg, db, monkeypatch):
    ctl = StudioController(cfg, db)
    ctl._sensed = _machine(cards=2)
    info = ctl.devices()
    assert info["automatic"] and [r["key"] for r in info["rows"] if r["on"]] == ["gpu1"]
    assert [r["label"] for r in info["rows"]] == ["Processor: Ryzen 7 (16 threads)", "Graphics card 1: RTX 3060, 12 GB",
                                                  "Graphics card 2: RTX 4090, 24 GB"]
    two = ctl.set_device("gpu0", True)                              # a second one on: both are used
    assert two["changed"] and not two["automatic"] and [r["key"] for r in two["rows"] if r["on"]] == ["gpu0", "gpu1"]
    assert cfg.compute.devices == ["gpu0", "gpu1"]
    one = ctl.set_device("gpu1", False)
    assert [r["key"] for r in one["rows"] if r["on"]] == ["gpu0"]
    last = ctl.set_device("gpu0", False)                            # the last one cannot be switched off
    assert not last["changed"] and "At least one device has to stay on" in last["message"] and cfg.compute.devices == ["gpu0"]
    ctl._training["running"] = True
    assert "Pause training" in ctl.set_device("cpu", True)["message"] and cfg.compute.devices == ["gpu0"]
    ctl._training["running"] = False
    auto = ctl.set_devices_automatic()
    assert auto["automatic"] and cfg.compute.devices == []

    from gauge.compute import devices as mod
    monkeypatch.setattr(mod, "setup", lambda *a, **k: {"ok": True, "installed": True, "message": "The graphics card is set up."})
    monkeypatch.setattr("gauge.compute.scan", lambda: _machine(cards=1))
    done = ctl.setup_gpu()
    assert done["ok"] and done["message"] == "The graphics card is set up." and len(done["rows"]) == 2


def test_device_commands(cfg, monkeypatch):
    from click.testing import CliRunner

    from gauge.cli import main
    from gauge.compute import devices as mod

    cfg.save()
    monkeypatch.setattr(mod, "scan", lambda: _machine(cards=2))
    runner = CliRunner(env={"COLUMNS": "220"})

    def run(*args, code=0):
        r = runner.invoke(main, ["--data-dir", cfg.data_dir, "compute", *args])
        assert r.exit_code == code, r.output
        return r.output

    out = run("devices")
    assert "gpu0" in out and "RTX 4090" in out and "chosen automatically" in out and "Automatic:" in out
    out = run("use", "gpu0", "gpu1")
    assert "Graphics card 1: RTX 3060, 12 GB and Graphics card 2: RTX 4090, 24 GB" in out
    assert "ON" in run("devices")
    assert "kept on the processor" in run("use", "cpu")
    assert "no such device: gpu9" in run("use", "gpu9", code=1)
    assert "chosen automatically" in run("use", "auto")
    assert run("setup-gpu", "--show").strip().endswith("cupy-cuda12x[ctk]")


def test_the_switches_window_has_a_switch_for_each_device(cfg, db, tmp_path, monkeypatch):
    import sys

    from gauge import gui
    from tests.test_teach import _fake_tkinter, _wait_for_jobs, _Widget

    monkeypatch.setattr("gauge.compute.scan", lambda: _machine(ready=False, cards=1))
    mods, shown = _fake_tkinter(tmp_path)
    made: list = []

    class _Check(_Widget):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    mods["tkinter.ttk"].Checkbutton = _Check
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    assert "Devices (processor and graphics cards)..." in _Widget.menu
    _Widget.buttons["Switches"]()
    _wait_for_jobs()
    labels = [c.kw.get("text") for c in made]
    assert "Automatic (let Gauge pick)" in labels and "Processor: Ryzen 7 (16 threads)" in labels
    assert "Graphics card 1: RTX 3060, 12 GB" in labels and "Set up the graphics card" in _Widget.buttons
    cpu = next(c for c in made if c.kw.get("text") == "Processor: Ryzen 7 (16 threads)")
    cpu.kw["variable"].set(True)
    cpu.kw["command"]()                                             # flipping the switch saves the choice
    assert cfg.compute.devices == ["cpu"]
    assert not [s for s in shown if s[0] == "error"], shown
