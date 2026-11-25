"""The capacity limit: Gauge uses at most a set share of each device it runs on.

``compute.capacity_percent`` (default 20) is the share of a device Gauge may use. It applies to this
computer and to every linked worker, each of which enforces it on itself. 100 turns the limit off.

How it is enforced, and how firmly:

* Processor, this process and its children (firm). Gauge pins itself to that share of the processor's
  cores (CPU affinity), caps its math libraries to the same number of threads, and lowers its priority.
  It cannot use a core it is not pinned to. A machine with fewer than five cores cannot give exactly 20%
  with one core, so there the remainder is made up by resting between pieces of work.
* The AI engine, Ollama (firm for processor threads, averaged for the graphics card). Ollama is a
  separate program, so Gauge cannot pin it. Each request tells it to use only the allowed number of
  processor threads. Nothing can make a graphics card run a request at a fifth of its speed, so Gauge
  spaces its requests out: after a request that took t seconds it waits long enough that the card was
  busy for at most the allowed share of the time. Calls are made one at a time.
* GaugeLM on a graphics card (firm for memory): its memory pool is capped at the share.
* Linked devices. Each worker applies all of the above to itself from its own setting, and the
  coordinator also spaces out what it sends to each worker, so an older worker is not overused either.

What the limit cannot do: shrink a model. A loaded model occupies its full size in memory whatever the
limit says. ``plan()`` reports that share of memory and whether the chosen model fits inside it, so the
choice of model stays an informed one.

This module imports nothing heavy on purpose: it has to set the thread caps before numpy is loaded.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Optional

log = logging.getLogger(__name__)

ENV = "GAUGE_CAPACITY_PERCENT"
DEFAULT_PERCENT = 20
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
# Approximate memory a loaded model needs, in GB, for the "does it fit in the share" report.
MODEL_GB = {"llama3.2:3b": 2.5, "llama3.1:8b": 5.5, "qwen2.5:7b": 5.5, "qwen2.5:14b": 10.0, "gemma2:9b": 6.5, "gaugelm": 0.2}


def clamp_percent(value: Any) -> int:
    try:
        return max(1, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return DEFAULT_PERCENT


def configured_percent(cfg: Any = None) -> int:
    """The limit in force: the environment wins (so one run can override), then the config, then 20."""
    env = os.environ.get(ENV, "").strip()
    if env:
        return clamp_percent(env)
    try:
        return clamp_percent(cfg.compute.capacity_percent)
    except AttributeError:
        return DEFAULT_PERCENT


def cores_for(percent: int, total: int) -> int:
    """How many of `total` cores the share allows: never more than the share, never fewer than one."""
    total = max(1, int(total))
    if percent >= 100:
        return total
    return max(1, (total * percent) // 100)


class Governor:
    """Keeps the average use of one device at or under the limit by resting after each piece of work.

    `load` is how much of the device a piece of work occupies while it runs (1.0 for a graphics card or an
    unpinned program; the pinned share of cores for work inside this process). If the work already fits
    inside the limit there is nothing to rest for. Below 100% work is done one piece at a time."""

    def __init__(self, percent: int = 100, load: float = 1.0, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, max_pause: float = 1800.0):
        self.percent = clamp_percent(percent)
        self.load = max(0.0, min(1.0, float(load)))
        self._sleep, self._clock, self._max_pause = sleep, clock, max_pause
        self._lock = threading.Lock()
        self._ready_at = 0.0
        self.rested = 0.0          # seconds spent resting so far (for reports and tests)

    @property
    def limited(self) -> bool:
        return self.percent < 100

    def pause_for(self, seconds_worked: float, load: Optional[float] = None) -> float:
        """How long to rest after `seconds_worked` so the average stays at the limit."""
        share = self.percent / 100.0
        u = self.load if load is None else max(0.0, min(1.0, float(load)))
        if not self.limited or u <= share or seconds_worked <= 0:
            return 0.0
        return min(self._max_pause, seconds_worked * (u / share - 1.0))

    def ready_in(self) -> float:
        """Seconds until this device may be given more work."""
        return max(0.0, self._ready_at - self._clock())

    @contextmanager
    def work(self, load: Optional[float] = None) -> Iterator[None]:
        """Run one piece of work on the device. Waits out the rest owed from the previous piece first, so a
        single call is never delayed and a steady stream averages out at the limit."""
        if not self.limited:
            yield
            return
        with self._lock:
            wait = self.ready_in()
            if wait > 0:
                self.rested += wait
                self._sleep(wait)
            t0 = self._clock()
            try:
                yield
            finally:
                now = self._clock()
                self._ready_at = now + self.pause_for(now - t0, load)

    def pace(self, seconds_worked: float, load: Optional[float] = None) -> float:
        """For a loop that cannot be split into calls (training steps): rest now for the work just done."""
        rest = self.pause_for(seconds_worked, load)
        if rest > 0:
            self.rested += rest
            self._sleep(rest)
        return rest


class ProcessGovernor(Governor):
    """The governor for work inside this process. Instead of assuming how busy a step was, it measures:
    processor time this process has really used against the time that has passed and the cores the machine
    has. Work it could not pace as it happened (loading, building a tokenizer) is repaid at the next
    step, so the average over a busy stretch lands on the limit, not just the paced part of it."""

    IDLE_RESET = 120.0   # after this long without work, start a fresh account: no saving up for a burst

    def __init__(self, percent: int = 100, load: float = 1.0, cores: Optional[int] = None,
                 cpu_clock: Callable[[], float] = time.process_time, **kw: Any):
        super().__init__(percent, load, **kw)
        self._cores = max(1, cores or os.cpu_count() or 1)
        self._cpu_clock = cpu_clock
        self._cpu0 = cpu_clock()
        self._t0 = self._clock()
        self._last = self._t0

    def pace(self, seconds_worked: float, load: Optional[float] = None) -> float:
        if load is not None:                       # work on a graphics card: processor time says nothing about it
            return super().pace(seconds_worked, load)
        if not self.limited:
            return 0.0
        now = self._clock()
        if now - self._last > self.IDLE_RESET:
            self._cpu0, self._t0 = self._cpu_clock() - seconds_worked * self.load * self._cores, now - seconds_worked
        used = self._cpu_clock() - self._cpu0       # processor-seconds really spent
        owed = used / (self.percent / 100.0 * self._cores) - (now - self._t0)
        rest = min(self._max_pause, max(0.0, owed))
        if rest > 0:
            self.rested += rest
            self._sleep(rest)
        self._last = self._clock()
        return rest


# ---------------------------------------------------------------------------- this process
_state: dict[str, Any] = {"percent": 100, "engine": Governor(100), "compute": ProcessGovernor(100), "devices": {}, "applied": None,
                          "full_affinity": None}
_state_lock = threading.Lock()


def has_gpu() -> bool:
    """Is there a graphics card the AI engine is likely to be using? (NVIDIA, or Apple Silicon.)"""
    if shutil.which("nvidia-smi"):
        return True
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def configure(percent: Any) -> int:
    """Set the limit for this process. Returns the limit in force."""
    p = clamp_percent(percent)
    total = os.cpu_count() or 1
    cpu_share = cores_for(p, total) / total
    with _state_lock:
        _state["percent"] = p
        _state["engine"] = Governor(p, load=1.0 if has_gpu() else cpu_share)   # the AI engine: card, or pinned threads
        _state["compute"] = ProcessGovernor(p, load=cpu_share, cores=total)    # work inside this process
        _state["devices"] = {}
    return p


def reset() -> None:
    configure(100)


def percent() -> int:
    return int(_state["percent"])


def engine() -> Governor:
    """The governor for requests to the AI engine on this computer."""
    return _state["engine"]


def compute() -> ProcessGovernor:
    """The governor for heavy loops inside this process (GaugeLM training)."""
    return _state["compute"]


def device(key: str) -> Governor:
    """The governor for one linked device (a worker), created on first use."""
    with _state_lock:
        gov = _state["devices"].get(key)
        if gov is None or gov.percent != _state["percent"]:
            gov = _state["devices"][key] = Governor(_state["percent"], load=1.0)
        return gov


def engine_threads() -> Optional[int]:
    """Processor threads the AI engine may use per request, or None when there is no limit."""
    if percent() >= 100:
        return None
    return cores_for(percent(), os.cpu_count() or 1)


def gpu_memory_fraction() -> Optional[float]:
    """Share of graphics memory GaugeLM's pool may take, or None when there is no limit."""
    return None if percent() >= 100 else percent() / 100.0


def apply_process_limits(percent_: Any = None) -> dict[str, Any]:
    """Pin this process (and the programs it starts) to its share of the cores, cap the math threads and
    lower the priority. Safe to call more than once. Returns what was actually applied."""
    p = percent() if percent_ is None else clamp_percent(percent_)
    total = os.cpu_count() or 1
    allowed = cores_for(p, total)
    info: dict[str, Any] = {"percent": p, "cores_total": total, "cores_allowed": allowed, "affinity": False,
                            "priority": False, "threads": False}
    if p >= 100:
        _restore_affinity()
        _state["applied"] = info
        return info
    for var in _THREAD_VARS:           # read by numpy's math library when it is first loaded
        os.environ[var] = str(allowed)
    info["threads"] = True
    try:
        info["affinity"] = _pin_to_cores(allowed)
    except Exception as e:             # a limit that cannot be applied must never stop Gauge from starting
        log.warning("could not pin to %d core(s): %s", allowed, e)
    try:
        info["priority"] = _lower_priority()
    except Exception as e:
        log.warning("could not lower the priority: %s", e)
    _state["applied"] = info
    return info


def _kernel32() -> Any:
    """Windows' kernel32 with the handle and mask types declared. Without them a 64-bit process hands the
    system a truncated handle and the call quietly fails."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetCurrentProcess.argtypes = []
    k32.GetProcessAffinityMask.restype = wintypes.BOOL
    k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    return k32


def _pin_to_cores(allowed: int) -> bool:
    if hasattr(os, "sched_setaffinity"):                                   # Linux
        if _state["full_affinity"] is None:
            _state["full_affinity"] = sorted(os.sched_getaffinity(0))
        full = _state["full_affinity"]
        os.sched_setaffinity(0, set(full[-max(1, min(allowed, len(full))):]))   # the last cores: core 0 is the busiest
        return True
    if platform.system() == "Windows":
        import ctypes

        k32 = _kernel32()
        handle = k32.GetCurrentProcess()
        if _state["full_affinity"] is None:
            proc, system = ctypes.c_size_t(0), ctypes.c_size_t(0)
            if not k32.GetProcessAffinityMask(handle, ctypes.byref(proc), ctypes.byref(system)):
                return False
            _state["full_affinity"] = [i for i in range(ctypes.sizeof(ctypes.c_size_t) * 8) if proc.value >> i & 1]
        full = _state["full_affinity"]
        mask = 0
        for i in full[-max(1, min(allowed, len(full))):]:
            mask |= 1 << i
        return bool(k32.SetProcessAffinityMask(handle, mask))
    return False                                                            # macOS has no affinity; threads and priority still apply


def _restore_affinity() -> None:
    full = _state.get("full_affinity")
    if not full:
        return
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(full))
        elif platform.system() == "Windows":
            mask = 0
            for i in full:
                mask |= 1 << i
            k32 = _kernel32()
            k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask)
    except Exception as e:
        log.warning("could not restore the full set of cores: %s", e)


_priority_lowered = False


def _lower_priority() -> bool:
    global _priority_lowered
    if _priority_lowered:
        return True
    if platform.system() == "Windows":
        k32 = _kernel32()
        below_normal = 0x00004000
        ok = bool(k32.SetPriorityClass(k32.GetCurrentProcess(), below_normal))
    else:
        os.nice(10)
        ok = True
    _priority_lowered = ok
    return ok


# ---------------------------------------------------------------------------- reporting
def total_memory_gb() -> float:
    """This computer's memory in GB, best effort (0 when it cannot be read)."""
    try:
        if platform.system() == "Windows":
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _Mem()
            m.dwLength = ctypes.sizeof(_Mem)
            ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 1024 ** 3, 1)
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="ascii", errors="replace") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 1024 ** 2, 1)
        if platform.system() == "Darwin":
            import subprocess

            return round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout) / 1024 ** 3, 1)
    except Exception:
        pass
    return 0.0


def plan(cfg: Any = None, workers: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """What the limit means on this computer and each linked device, in numbers a person can check."""
    p = configured_percent(cfg) if cfg is not None else percent()
    total = os.cpu_count() or 1
    allowed = cores_for(p, total)
    gpu = has_gpu()
    mem = total_memory_gb()
    model = ""
    try:
        model = str(cfg.llm.model)
    except AttributeError:
        pass
    need = MODEL_GB.get(model)
    share_gb = round(mem * p / 100, 1) if mem else 0.0
    rest_factor = 0.0 if p >= 100 else round((1.0 if gpu else allowed / total) / (p / 100) - 1.0, 2)
    out: dict[str, Any] = {
        "percent": p, "limited": p < 100,
        "cores_total": total, "cores_allowed": allowed, "core_share_percent": round(100 * allowed / total, 1),
        "engine_threads": allowed if p < 100 else None,
        "gpu": gpu, "rest_per_second_of_work": max(0.0, rest_factor),
        "memory_gb": mem, "memory_share_gb": share_gb, "model": model, "model_needs_gb": need,
        "model_fits_share": (need <= share_gb) if (need is not None and mem) else None,
        "applied": _state.get("applied"),
        "linked": [{"name": w.get("name") or w.get("url", ""), "url": w.get("url", ""),
                    "own_limit": w.get("capacity_percent"), "paced_at": p} for w in (workers or [])],
    }
    return out


def describe(cfg: Any = None) -> str:
    """One sentence for a status line."""
    pl = plan(cfg)
    if not pl["limited"]:
        return "No capacity limit: Gauge may use the whole device."
    return (f"Capacity limit {pl['percent']}%: {pl['cores_allowed']} of {pl['cores_total']} processor cores"
            + (", graphics card busy at most that share of the time" if pl["gpu"] else "") + ".")
