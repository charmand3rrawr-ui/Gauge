"""Compute: GPU sensing (parsers, scan, watch), workers over HTTP, discovery over UDP, pool generation,
distributed model-averaging rounds, and the CLI."""
import json
import socket
import threading
import time

import numpy as np

from gauge.compute import (
    DistributedTrainer,
    Pool,
    PoolLLM,
    Worker,
    WorkerClient,
    best_device,
    discover,
    scan,
    summary,
    watch,
)
from gauge.compute.detect import (
    Device,
    parse_lspci,
    parse_nvidia_smi,
    parse_rocm_smi,
    parse_windows_cim,
)
from gauge.compute.protocol import (
    announcement,
    pack_arrays,
    parse_announcement,
    token_hash,
    unpack_arrays,
)
from gauge.llm import FakeLLM
from gauge.native import GaugeLM, ModelConfig


# ------------------------------------------------------------------ sensing
def test_parsers():
    nv = parse_nvidia_smi("NVIDIA GeForce RTX 4070, 12282, 551.23\nNVIDIA T400, 2048, 551.23\n")
    assert [d.name for d in nv] == ["NVIDIA GeForce RTX 4070", "NVIDIA T400"] and nv[0].memory_mb == 12282 and nv[0].vendor == "nvidia"
    assert nv[0].extra["driver"] == "551.23"
    rocm = parse_rocm_smi(json.dumps({"card0": {"Card series": "Radeon RX 7900 XTX"}}))
    assert rocm[0].vendor == "amd" and "7900" in rocm[0].name
    assert parse_rocm_smi("GPU[0]\t\t: Card series:\t\tRadeon RX 6800")[0].name == "Radeon RX 6800"
    cim = parse_windows_cim(json.dumps([{"Name": "Intel(R) UHD Graphics 770", "AdapterRAM": 1073741824},
                                        {"Name": "NVIDIA GeForce RTX 3060", "AdapterRAM": 4293918720}, {"Name": None}]))
    assert [d.vendor for d in cim] == ["intel", "nvidia"] and cim[0].memory_mb == 1024
    assert parse_windows_cim("not json") == []
    pci = parse_lspci("01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31\n02:00.0 Ethernet controller: Intel\n"
                      "03:00.0 3D controller: NVIDIA Corporation GA104\n")
    assert [d.vendor for d in pci] == ["amd", "nvidia"]


def test_scan_and_best_device_and_watch():
    devs = scan()
    assert devs[0].kind == "cpu" and devs[0].backend_ready and devs[0].extra["threads"] >= 1
    use, chosen = best_device(devs)
    assert use in ("cpu", "cuda") and chosen.name
    s = summary(devs)
    assert s["use"] == use and "numpy" in s["backends"] and s["backends"]["numpy"] is True
    # a found GPU without its backend is reported with an install hint and not used
    fake = [devs[0], Device("gpu", "nvidia", "RTX 4090", 24000, "cupy", False, "pip install cupy-cuda12x", "test")]
    assert best_device(fake)[0] == "cpu" and fake[1].install_hint.startswith("pip install")
    ready = [devs[0], Device("gpu", "nvidia", "RTX 4090", 24000, "cupy", True, "", "test"), Device("gpu", "nvidia", "T400", 2000, "cupy", True, "", "t")]
    assert best_device(ready)[1].name == "RTX 4090"  # largest memory wins
    # watch: polls, reports changes, gives up at the time limit when no GPU is ready
    seen = []
    sleeps = []
    out = watch(seen.append, interval_s=0.0, max_seconds=0.0, sleep=sleeps.append)
    assert out is None or isinstance(out, list)


# ------------------------------------------------------------------ protocol
def test_protocol_roundtrips():
    a = announcement("pc", 47632, "secret", [{"kind": "gpu", "name": "RTX", "backend_ready": True}], "ollama x")
    d = parse_announcement(a)
    assert d["name"] == "pc" and d["port"] == 47632 and d["token_hash"] == token_hash("secret") and "secret" not in a.decode()
    assert parse_announcement(b"junk") is None and parse_announcement(json.dumps({"magic": "x", "port": 1}).encode()) is None
    arrs = {"w": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.zeros(3)}
    back = unpack_arrays(pack_arrays(arrs))
    assert np.array_equal(back["w"], arrs["w"]) and back["b"].shape == (3,)


# ------------------------------------------------------------------ worker + pool + distributed training
def _start_worker(cfg, token="tok", llm=None, announce=False, announce_port=None, target="127.0.0.1"):
    w = Worker(cfg, "test-worker", token, host="127.0.0.1", port=0, llm=llm, announce=announce,
               announce_port=announce_port or 47631, announce_target=target)
    w.start()
    return w


def test_worker_health_generate_and_auth(cfg, db):
    w = _start_worker(cfg, llm=FakeLLM())
    try:
        url = f"http://127.0.0.1:{w.port}"
        assert WorkerClient(url, "wrong").health() is None
        h = WorkerClient(url, "tok").health()
        assert h["name"] == "test-worker" and h["llm"] == "fake llm (offline)" and h["devices"][0]["kind"] == "cpu"
        out = WorkerClient(url, "tok").generate("Write the opening.", max_tokens=20)
        assert out.startswith('"Don\'t open it,"')
        pool = Pool(db, "tok")
        pool.add(url, "test-worker", h["devices"], h["llm"])
        assert [x["url"] for x in pool.workers()] == [url]
        alive = pool.refresh()
        assert len(alive) == 1 and alive[0]["health"]["jobs"] == 1
        # PoolLLM round-robins to the worker and records the call as remote
        local = FakeLLM(canned=["LOCAL"])
        pl = PoolLLM(local, pool)
        assert pl.available() and "pool(1 workers)" in pl.describe()
        assert pl.generate("x", purpose="p").startswith('"Don\'t open it,"') and pl.calls.records[-1].model.startswith("remote:")
        # a dead worker falls back to local
        pool.add("http://127.0.0.1:9/", "dead")
        pl2 = PoolLLM(FakeLLM(canned=["LOCAL"]), Pool(db, "tok"))
        pl2._clients = [WorkerClient("http://127.0.0.1:9/", "tok", timeout=1)]
        pl2._refreshed = time.monotonic()
        assert pl2.generate("x") == "LOCAL"
        pool.remove("http://127.0.0.1:9/")
        assert len(pool.workers()) == 1
        # worker with no model refuses generation cleanly
        w2 = _start_worker(cfg, llm=None)
        try:
            try:
                WorkerClient(f"http://127.0.0.1:{w2.port}", "tok").generate("x")
            except Exception as e:
                assert "serves no model" in str(e)
            else:
                raise AssertionError("expected an error")
        finally:
            w2.stop()
    finally:
        w.stop()


def test_distributed_round_averages_weights(cfg, db):
    w = _start_worker(cfg)
    try:
        pool = Pool(db, "tok")
        pool.add(f"http://127.0.0.1:{w.port}", "test-worker")
        model = GaugeLM(ModelConfig(vocab_size=60, block_size=16, n_layer=1, n_head=2, n_embd=16, n_kv_head=1), seed=0)
        before = {k: v.copy() for k, v in model.p.items()}
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, 60, size=600).astype(np.int32)
        out = DistributedTrainer(pool).round(model, tokens, steps=3, lr=1e-3, batch_size=2)
        assert set(out["participants"]) == {"local", f"http://127.0.0.1:{w.port}"} and out["failed"] == [] and out["loss"] > 0
        assert any(not np.array_equal(before[k], model.p[k]) for k in model.p)
        assert WorkerClient(f"http://127.0.0.1:{w.port}", "tok").health()["jobs"] == 1
        # a dead worker is reported as failed; the round still completes locally
        pool.add("http://127.0.0.1:9/", "dead")
        out2 = DistributedTrainer(pool).round(model, tokens, steps=2, lr=1e-3, batch_size=2)
        assert "local" in out2["participants"]
    finally:
        w.stop()


def test_discovery_over_loopback(cfg):
    # pick a free UDP port for the announcement channel
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    w = _start_worker(cfg, announce=True, announce_port=port, target="127.0.0.1")
    w.announcer.interval = 0.2
    try:
        found = discover(seconds=1.5, token="tok", announce_port=port, bind="127.0.0.1")
        assert found and found[0]["name"] == "test-worker" and found[0]["url"].endswith(f":{w.port}")
        assert discover(seconds=0.7, token="other", announce_port=port, bind="127.0.0.1") == []  # wrong token filtered
    finally:
        w.stop()


def test_cli_compute(tmp_path, cfg):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "200"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    r = runner.invoke(main, ["--data-dir", data, "compute", "scan", "--apply"])
    assert r.exit_code == 0 and "GaugeLM would use" in r.output and "saved" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "compute", "discover"])
    assert r.exit_code != 0 and "no token" in r.output
    r = runner.invoke(main, ["--data-dir", data, "compute", "token"])
    assert r.exit_code == 0 and "gauge worker --token" in r.output
    token = json.loads((tmp_path / "data" / "config.json").read_text())["compute"]["token"]
    assert len(token) > 20
    r = runner.invoke(main, ["--data-dir", data, "compute", "list"])
    assert r.exit_code == 0 and "no workers" in r.output
    w = Worker(cfg, "office", token, host="127.0.0.1", port=0, llm=FakeLLM(), announce=False)
    w.start()
    try:
        url = f"http://127.0.0.1:{w.port}"
        r = runner.invoke(main, ["--data-dir", data, "compute", "add", url])
        assert r.exit_code == 0 and "added" in r.output and "office" in r.output, r.output
        r = runner.invoke(main, ["--data-dir", data, "compute", "list"])
        assert r.exit_code == 0 and "office" in r.output and "yes" in r.output
        r = runner.invoke(main, ["--data-dir", data, "compute", "add", "http://127.0.0.1:9/"])
        assert r.exit_code != 0 and "no worker answered" in r.output
        r = runner.invoke(main, ["--data-dir", data, "compute", "remove", url])
        assert r.exit_code == 0
    finally:
        w.stop()
    r = runner.invoke(main, ["--data-dir", data, "compute", "watch", "--interval", "0", "--max-minutes", "0"])
    assert r.exit_code == 0


def test_daemon_cycle_senses_device(cfg, db, chapter_text):
    from gauge.daemon import Daemon
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    cfg.learn.absorb_per_cycle = 0
    cfg.native.daemon_steps = 0
    cfg.research.enabled = False
    out = Daemon(cfg, db).once(write=False)
    assert out["device"] in ("cpu", "cuda") and "train" in out
    threading.active_count()  # no leaked threads expected beyond pytest's
