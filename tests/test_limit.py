"""The capacity limit: Gauge uses at most a set share of each device (default 20%)."""
import os
import threading

import pytest
from click.testing import CliRunner

from gauge import limit
from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.limit import Governor, ProcessGovernor


@pytest.fixture(autouse=True)
def _unlimited_afterwards():
    yield
    limit.reset()


class _Clock:
    """A clock and a sleep that share one notion of time, so pacing can be tested without waiting."""

    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def work(self, seconds):
        self.now += seconds


# ------------------------------------------------------------------ the default and the share of cores
def test_the_default_is_twenty_percent_and_the_environment_overrides(monkeypatch):
    assert GaugeConfig().compute.capacity_percent == 20
    monkeypatch.delenv(limit.ENV, raising=False)
    cfg = GaugeConfig()
    assert limit.configured_percent(cfg) == 20
    cfg.compute.capacity_percent = 55
    assert limit.configured_percent(cfg) == 55
    monkeypatch.setenv(limit.ENV, "100")
    assert limit.configured_percent(cfg) == 100                # one run can lift it without touching the config
    assert limit.clamp_percent(0) == 1 and limit.clamp_percent(250) == 100 and limit.clamp_percent("junk") == 20


@pytest.mark.parametrize("percent,total,want", [(20, 4, 1), (20, 8, 1), (20, 16, 3), (20, 32, 6), (50, 8, 4),
                                                (100, 8, 8), (20, 1, 1), (1, 64, 1)])
def test_cores_never_exceed_the_share_and_never_drop_to_zero(percent, total, want):
    got = limit.cores_for(percent, total)
    assert got == want
    assert got == total or got / total <= percent / 100 or got == 1


# ------------------------------------------------------------------ pacing
def test_governor_rests_so_a_busy_device_averages_its_share():
    clock = _Clock()
    gov = Governor(20, load=1.0, sleep=clock.sleep, clock=clock)
    busy = 0.0
    for _ in range(10):                                         # ten requests of 3 seconds each on a graphics card
        with gov.work():
            clock.work(3.0)
            busy += 3.0
    with gov.work():                                            # the rest owed by the tenth is paid before an eleventh
        pass
    assert busy / clock.now == pytest.approx(0.20, abs=0.001)   # busy exactly a fifth of the time
    assert clock.slept and all(s == pytest.approx(12.0) for s in clock.slept)


def test_a_single_call_is_never_delayed_and_no_limit_means_no_rest():
    clock = _Clock()
    gov = Governor(20, load=1.0, sleep=clock.sleep, clock=clock)
    with gov.work():
        clock.work(5.0)
    assert clock.slept == []                                    # the rest is owed by the next call, not this one
    assert gov.ready_in() == pytest.approx(20.0)
    free = Governor(100, sleep=clock.sleep, clock=clock)
    for _ in range(5):
        with free.work():
            clock.work(2.0)
    assert clock.slept == [] and free.pause_for(100) == 0.0


def test_work_that_already_fits_the_share_is_not_slowed():
    clock = _Clock()
    pinned = Governor(20, load=0.125, sleep=clock.sleep, clock=clock)   # 1 of 8 cores is already under a fifth
    with pinned.work():
        clock.work(10.0)
    assert pinned.ready_in() == 0.0
    quarter = Governor(20, load=0.25, sleep=clock.sleep, clock=clock)   # 1 of 4 cores is a quarter: rest a little
    assert quarter.pause_for(8.0) == pytest.approx(2.0)                 # 8 busy in 10 at a quarter = a fifth overall


def test_limited_work_runs_one_piece_at_a_time():
    gov = Governor(50, load=0.1)                                # under its share: no rests, so only the lock is tested
    running, peak = 0, 0
    lock = threading.Lock()

    def job():
        nonlocal running, peak
        with gov.work():
            with lock:
                running += 1
                peak = max(peak, running)
            threading.Event().wait(0.02)
            with lock:
                running -= 1

    threads = [threading.Thread(target=job) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_process_governor_measures_real_processor_time_and_repays_unpaced_work():
    clock = _Clock()
    cpu = {"used": 0.0}
    gov = ProcessGovernor(20, load=0.25, cores=4, sleep=clock.sleep, clock=clock, cpu_clock=lambda: cpu["used"])
    # 10 seconds of set-up on one core that nothing paced (25% of a 4-core machine: over the limit).
    clock.work(10.0)
    cpu["used"] += 10.0
    gov.pace(0.0)                                               # the first paced step repays it
    assert cpu["used"] / (clock.now * 4) == pytest.approx(0.20)
    for _ in range(20):                                         # then steady steps of one core-second each
        clock.work(1.0)
        cpu["used"] += 1.0
        gov.pace(1.0)
    assert cpu["used"] / (clock.now * 4) == pytest.approx(0.20)
    assert gov.pace(2.0, load=1.0) == pytest.approx(8.0)        # graphics-card work is paced by time instead


def test_process_governor_does_not_save_up_idle_time_for_a_burst():
    clock = _Clock()
    cpu = {"used": 0.0}
    gov = ProcessGovernor(20, load=0.25, cores=4, sleep=clock.sleep, clock=clock, cpu_clock=lambda: cpu["used"])
    clock.work(3600.0)                                          # an hour of doing nothing
    for _ in range(30):
        clock.work(1.0)
        cpu["used"] += 1.0
        gov.pace(1.0)
    start = 3600.0
    assert cpu["used"] / ((clock.now - start) * 4) <= 0.21      # the hour of idleness bought no extra speed


# ------------------------------------------------------------------ where it is applied
def test_configure_builds_governors_for_the_engine_this_process_and_each_linked_device(monkeypatch):
    monkeypatch.setattr(limit, "has_gpu", lambda: True)
    assert limit.configure(20) == 20
    assert limit.engine().percent == 20 and limit.engine().load == 1.0       # a graphics card runs flat out while it works
    assert isinstance(limit.compute(), ProcessGovernor) and limit.compute().percent == 20
    a, b = limit.device("http://worker-a:47632"), limit.device("http://worker-b:47632")
    assert a is not b and a is limit.device("http://worker-a:47632") and a.percent == 20
    assert limit.gpu_memory_fraction() == pytest.approx(0.2)
    assert limit.engine_threads() == limit.cores_for(20, os.cpu_count() or 1)
    limit.configure(100)
    assert limit.engine_threads() is None and limit.gpu_memory_fraction() is None and not limit.engine().limited


def test_the_ai_engine_is_told_how_many_threads_it_may_use():
    from gauge.llm.client import OllamaLLM
    cfg = GaugeConfig().llm
    llm = OllamaLLM(cfg, cfg.model)
    assert "num_thread" not in llm._payload("p", "", None, None, False, None)["options"]   # no limit: Ollama decides
    limit.configure(20)
    assert llm._payload("p", "", None, None, False, None)["options"]["num_thread"] == limit.engine_threads()


def test_models_are_governed_under_a_limit_and_untouched_without_one(monkeypatch):
    from gauge.llm.client import OllamaLLM, make_llm
    from gauge.llm.governed import GovernedLLM
    monkeypatch.delenv("GAUGE_NO_LLM", raising=False)
    cfg = GaugeConfig().llm
    assert isinstance(make_llm(cfg), OllamaLLM)                 # 100% (the tests' setting): the plain client
    limit.configure(20)
    governed = make_llm(cfg)
    assert isinstance(governed, GovernedLLM) and governed.model == cfg.model      # everything else still delegates
    cfg.provider = "fake"
    assert not isinstance(make_llm(cfg), GovernedLLM)           # the offline stand-in does no real work to limit


def test_governed_model_spaces_its_calls():
    from gauge.llm.governed import GovernedLLM
    clock = _Clock()

    class Slow:
        model = "m"

        def generate(self, prompt, **_):
            clock.work(4.0)
            return "ok:" + prompt

    llm = GovernedLLM(Slow(), Governor(20, load=1.0, sleep=clock.sleep, clock=clock))
    assert [llm.generate(str(i)) for i in range(3)] == ["ok:0", "ok:1", "ok:2"]
    assert clock.slept == [pytest.approx(16.0), pytest.approx(16.0)] and llm.model == "m"


def test_the_pool_prefers_the_linked_device_that_has_rested_longest(db, monkeypatch):
    from gauge.compute.pool import PoolLLM
    limit.configure(20)
    sent: list[str] = []

    class Client:
        def __init__(self, url):
            self.url = url

        def generate(self, prompt, **_):
            sent.append(self.url)
            return "text"

    class Local:
        model = "local"
        calls = None

        def generate(self, *a, **k):
            return "local"

    pool = PoolLLM(Local(), pool=None)
    a, b = Client("http://a"), Client("http://b")
    monkeypatch.setattr(pool, "_workers", lambda: [a, b])
    limit.device("http://a")._ready_at = limit.device("http://a")._clock() + 500      # a is still resting
    pool.generate("one")
    assert sent == ["http://b"]                                  # so the work goes to b, which is ready


def test_a_linked_worker_reports_its_own_limit(cfg):
    from gauge.compute.worker import WorkerState
    limit.configure(35)
    assert WorkerState(cfg, "w", "tok").info()["capacity_percent"] == 35


def test_process_limits_pin_cap_threads_and_lower_priority(monkeypatch):
    pinned, lowered = [], []
    monkeypatch.setattr(limit, "_pin_to_cores", lambda n: pinned.append(n) or True)
    monkeypatch.setattr(limit, "_lower_priority", lambda: lowered.append(1) or True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(var, "99")
    info = limit.apply_process_limits(20)
    want = limit.cores_for(20, os.cpu_count() or 1)
    assert info["cores_allowed"] == want and info["affinity"] and info["priority"] and info["threads"]
    assert pinned == [want] and lowered == [1]
    assert os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == str(want)
    off = limit.apply_process_limits(100)
    assert not off["affinity"] and pinned == [want]             # 100%: nothing is pinned


def test_a_limit_that_cannot_be_applied_never_stops_gauge(monkeypatch):
    def boom(*_):
        raise OSError("not permitted")
    monkeypatch.setattr(limit, "_pin_to_cores", boom)
    monkeypatch.setattr(limit, "_lower_priority", boom)
    info = limit.apply_process_limits(20)
    assert info["affinity"] is False and info["priority"] is False and info["threads"] is True


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="core pinning is tested where the OS exposes it")
def test_cores_are_really_pinned_and_restored(monkeypatch):
    monkeypatch.setattr(limit, "_lower_priority", lambda: True)   # do not renice the test process
    before = os.sched_getaffinity(0)
    try:
        info = limit.apply_process_limits(20)
        assert info["affinity"] and len(os.sched_getaffinity(0)) == limit.cores_for(20, len(before)) or len(before) == 1
    finally:
        limit.apply_process_limits(100)
    assert os.sched_getaffinity(0) == before


# ------------------------------------------------------------------ reporting
def test_plan_says_what_the_limit_means_including_memory_it_cannot_shrink(monkeypatch):
    monkeypatch.delenv(limit.ENV, raising=False)
    monkeypatch.setattr(limit, "total_memory_gb", lambda: 16.0)
    monkeypatch.setattr(limit, "has_gpu", lambda: True)
    cfg = GaugeConfig()
    cfg.llm.model = "llama3.1:8b"
    pl = limit.plan(cfg, [{"name": "desk", "url": "http://desk", "capacity_percent": 20}])
    assert pl["percent"] == 20 and pl["limited"] and pl["rest_per_second_of_work"] == 4.0
    assert pl["memory_share_gb"] == 3.2 and pl["model_needs_gb"] == 5.5 and pl["model_fits_share"] is False
    assert pl["linked"] == [{"name": "desk", "url": "http://desk", "own_limit": 20, "paced_at": 20}]
    cfg.llm.model = "llama3.2:3b"
    assert limit.plan(cfg)["model_fits_share"] is True
    assert "20%" in limit.describe(cfg) and "graphics card" in limit.describe(cfg)
    cfg.compute.capacity_percent = 100
    assert limit.plan(cfg)["limited"] is False and "No capacity limit" in limit.describe(cfg)


def test_cli_shows_and_sets_the_limit(tmp_path, monkeypatch):
    monkeypatch.delenv(limit.ENV, raising=False)
    monkeypatch.setattr(limit, "_pin_to_cores", lambda n: True)
    monkeypatch.setattr(limit, "_lower_priority", lambda: True)
    data = str(tmp_path / "data")
    run = CliRunner(env={"COLUMNS": "200"})
    r = run.invoke(main, ["--data-dir", data, "compute", "limit"])
    assert r.exit_code == 0 and "Capacity limit: 20% of each device" in r.output and "cores" in r.output, r.output
    r = run.invoke(main, ["--data-dir", data, "compute", "limit", "100"])
    assert r.exit_code == 0 and "off" in r.output
    assert GaugeConfig.load(data).compute.capacity_percent == 100
    assert run.invoke(main, ["--data-dir", data, "compute", "limit", "0"]).exit_code != 0     # out of range


def test_the_limit_is_a_switch_that_takes_effect_at_once(cfg, monkeypatch):
    from gauge import settings as st
    monkeypatch.delenv(limit.ENV, raising=False)
    monkeypatch.setattr(limit, "_pin_to_cores", lambda n: True)
    monkeypatch.setattr(limit, "_lower_priority", lambda: True)
    assert st.BY_KEY["compute.capacity_percent"].kind == "int"
    st.set_value(cfg, "compute.capacity_percent", 40, save=False)
    assert limit.percent() == 40 and limit.engine().percent == 40


def test_windows_calls_declare_their_handle_types():
    # Not runnable off Windows, so pin the thing that matters: every kernel32 call goes through the one
    # helper that declares the handle and mask types. An undeclared handle is truncated on 64-bit Windows
    # and the call fails without a sound.
    import inspect
    src = inspect.getsource(limit)
    helper = inspect.getsource(limit._kernel32)
    for name in ("GetCurrentProcess", "GetProcessAffinityMask", "SetProcessAffinityMask", "SetPriorityClass"):
        assert f"k32.{name}.restype" in helper and f"k32.{name}.argtypes" in helper, name
    outside = src.replace(helper, "")
    assert 'WinDLL("kernel32", use_last_error=True)' not in outside   # nobody opens kernel32 around the helper
    assert outside.count("_kernel32()") == 3                          # pin, restore, priority
