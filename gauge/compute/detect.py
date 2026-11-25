"""Sense the compute this machine has: CPU, and any GPU from NVIDIA, AMD or Intel.

Detection uses the vendor tools and OS inventories that exist on the machine (nvidia-smi,
rocm-smi, Windows CIM, lspci) and then checks which numpy-compatible Python backend could
drive each device (CuPy for CUDA and ROCm, dpnp for Intel). Hardware without a backend is
reported with the exact install command, and `watch()` notices when a device appears later,
for example an external GPU plugged in while the daemon is running.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Device:
    kind: str  # "cpu" | "gpu"
    vendor: str  # "nvidia" | "amd" | "intel" | "apple" | "cpu" | "unknown"
    name: str
    memory_mb: Optional[int] = None
    backend: str = ""  # "cupy" | "cupy-rocm" | "dpnp" | "numpy" | ""
    backend_ready: bool = False
    install_hint: str = ""
    source: str = ""  # which probe found it
    extra: dict[str, Any] = field(default_factory=dict)
    index: int = 0  # among the graphics cards, in the order the vendor tool lists them (bus order)

    @property
    def key(self) -> str:
        """The name the device switch uses: 'cpu', 'gpu0', 'gpu1', ..."""
        return "cpu" if self.kind == "cpu" else f"gpu{self.index}"

    @property
    def device_string(self) -> str:
        """What `native.device` should be to use this."""
        if self.kind == "cpu":
            return "cpu"
        return "cuda" if self.backend_ready and self.backend.startswith("cupy") else "cpu"


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# ---- parsers (pure, tested) -----------------------------------------------------------
def parse_nvidia_smi(out: str) -> list[Device]:
    devs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            mem = re.search(r"(\d+)", parts[1])
            devs.append(Device("gpu", "nvidia", parts[0], int(mem.group(1)) if mem else None, source="nvidia-smi",
                               extra={"driver": parts[2] if len(parts) > 2 else ""}))
    return devs


def parse_rocm_smi(out: str) -> list[Device]:
    devs = []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        for line in out.splitlines():
            m = re.search(r"Card series:\s*(.+)$", line)
            if m:
                devs.append(Device("gpu", "amd", m.group(1).strip(), source="rocm-smi"))
        return devs
    for _card, info in data.items():
        if isinstance(info, dict):
            name = info.get("Card series") or info.get("Card SKU") or info.get("Device Name") or "AMD GPU"
            devs.append(Device("gpu", "amd", str(name), source="rocm-smi"))
    return devs


def parse_windows_cim(out: str) -> list[Device]:
    """Output of: Get-CimInstance Win32_VideoController | Select Name,AdapterRAM | ConvertTo-Json"""
    devs = []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return devs
    rows = data if isinstance(data, list) else [data]
    for r in rows:
        if not isinstance(r, dict) or not r.get("Name"):
            continue
        name = str(r["Name"])
        low = name.lower()
        vendor = "nvidia" if "nvidia" in low or "geforce" in low or "quadro" in low or "rtx" in low else \
            "amd" if "amd" in low or "radeon" in low else "intel" if "intel" in low else "unknown"
        ram = r.get("AdapterRAM")
        mem = int(ram) // (1024 * 1024) if isinstance(ram, (int, float)) and ram > 0 else None
        devs.append(Device("gpu", vendor, name, mem, source="windows-cim"))
    return devs


def parse_lspci(out: str) -> list[Device]:
    devs = []
    for line in out.splitlines():
        if not re.search(r"VGA|3D controller|Display controller", line, re.I):
            continue
        name = line.split(":", 2)[-1].strip()
        low = name.lower()
        vendor = "nvidia" if "nvidia" in low else "amd" if "amd" in low or "ati" in low or "radeon" in low else \
            "intel" if "intel" in low else "unknown"
        devs.append(Device("gpu", vendor, name, source="lspci"))
    return devs


# ---- backends -------------------------------------------------------------------------
def backend_status() -> dict[str, Any]:
    st: dict[str, Any] = {"numpy": True, "cupy": False, "cupy_devices": 0, "cupy_rocm": False, "dpnp": False}
    try:
        import cupy as cp  # type: ignore

        st["cupy"] = True
        try:
            st["cupy_devices"] = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            st["cupy_devices"] = 0
        try:
            st["cupy_rocm"] = bool(getattr(cp.cuda.runtime, "is_hip", False))
        except Exception:
            pass
    except Exception:
        pass
    try:
        import dpnp  # type: ignore  # noqa: F401

        st["dpnp"] = True
    except Exception:
        pass
    return st


HINTS = {
    "nvidia": "pip install cupy-cuda12x   (NVIDIA driver 525+; CUDA 11 cards: cupy-cuda11x)",
    "amd": "pip install cupy-rocm-6-0   (Linux/WSL2 with ROCm; Windows AMD has no numpy-compatible backend yet)",
    "intel": "Intel GPUs: no numpy-compatible backend GaugeLM can use yet; the CPU path is used",
    "apple": "Apple GPUs: no numpy-compatible backend yet; the CPU path is used",
}


def _cpu_device() -> Device:
    name = platform.processor() or platform.machine() or "cpu"
    threads = os.cpu_count() or 1
    blas = ""
    try:
        cfg = np.show_config(mode="dicts")  # numpy >= 1.25
        blas = str(cfg.get("Build Dependencies", {}).get("blas", {}).get("name", ""))
    except Exception:
        pass
    return Device("cpu", "cpu", f"{name} ({threads} threads)", None, "numpy", True, source="os",
                  extra={"threads": threads, "blas": blas})


def scan() -> list[Device]:
    """Everything this machine could compute with, backends checked, hints attached."""
    devs: list[Device] = [_cpu_device()]
    gpus: list[Device] = []
    if shutil.which("nvidia-smi"):
        gpus += parse_nvidia_smi(_run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]))
    if shutil.which("rocm-smi"):
        gpus += parse_rocm_smi(_run(["rocm-smi", "--showproductname", "--json"]))
    if platform.system() == "Windows" and not gpus:
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if ps:
            gpus += parse_windows_cim(_run([ps, "-NoProfile", "-Command",
                                            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json"]))
    if platform.system() == "Linux" and not gpus and shutil.which("lspci"):
        gpus += parse_lspci(_run(["lspci"]))
    if platform.system() == "Darwin" and not gpus:
        gpus.append(Device("gpu", "apple", "Apple GPU (Metal)", source="platform"))
    st = backend_status()
    # Two cards of the same model are two devices: only the same card reported by two probes is a repeat.
    per_source: dict[str, list[Device]] = {}
    for g in gpus:
        per_source.setdefault(g.source, []).append(g)
    first = next(iter(per_source.values()), [])
    known = {g.name for g in first}
    gpus = first + [g for src, rest in list(per_source.items())[1:] for g in rest if g.name not in known]
    for i, g in enumerate(gpus):
        g.index = i
        if g.vendor == "nvidia":
            g.backend = "cupy"
            g.backend_ready = st["cupy"] and st["cupy_devices"] > 0 and not st["cupy_rocm"]
        elif g.vendor == "amd":
            g.backend = "cupy-rocm"
            g.backend_ready = st["cupy"] and st["cupy_devices"] > 0 and st["cupy_rocm"]
        elif g.vendor == "intel":
            g.backend = "dpnp" if st["dpnp"] else ""
            g.backend_ready = False  # the model does not run on dpnp yet; reported, not used
        if not g.backend_ready:
            g.install_hint = HINTS.get(g.vendor, "")
        devs.append(g)
    return devs


def best_device(devices: Optional[list[Device]] = None) -> tuple[str, Device]:
    """The device string GaugeLM should use right now, and the Device it refers to."""
    devices = devices if devices is not None else scan()
    ready = [d for d in devices if d.kind == "gpu" and d.backend_ready]
    if ready:
        ready.sort(key=lambda d: -(d.memory_mb or 0))
        return ready[0].device_string, ready[0]
    return "cpu", devices[0]


def summary(devices: list[Device]) -> dict[str, Any]:
    dev, chosen = best_device(devices)
    return {"use": dev, "chosen": chosen.name, "devices": [asdict(d) for d in devices], "backends": backend_status()}


def watch(callback: Callable[[list[Device]], None], interval_s: float = 30.0, max_seconds: Optional[float] = None,
          sleep: Callable[[float], None] = time.sleep) -> Optional[list[Device]]:
    """Poll until a GPU with a ready backend appears (e.g. an eGPU plugged in); returns the scan."""
    t0 = time.monotonic()
    last_names: set[str] = set()
    while True:
        devs = scan()
        names = {d.name for d in devs if d.kind == "gpu"}
        if names - last_names:
            callback(devs)
            last_names = names
        if any(d.kind == "gpu" and d.backend_ready for d in devs):
            return devs
        if max_seconds is not None and time.monotonic() - t0 >= max_seconds:
            return None
        sleep(interval_s)
