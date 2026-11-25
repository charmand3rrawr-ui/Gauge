"""The device switch: which of the devices Gauge senses it may use, one or several.

`compute.devices` in the settings holds the choice: a list of keys such as ``["gpu0"]``,
``["gpu0", "gpu1"]`` or ``["cpu"]``. An empty list means automatic: the best graphics card that is
ready, otherwise the processor.

What the choice changes:

* **GaugeLM** (Gauge's own model) trains and writes on the chosen devices. Several graphics cards share
  every training step (`gauge.native.parallel`). The processor chosen alongside a graphics card is
  timed and rested if it cannot keep up.
* **The downloaded model** runs in the AI engine (Ollama), which manages graphics cards itself. Gauge
  can tell it to stay on the processor: that happens when no graphics card is chosen. Which card it
  uses when one is allowed is the engine's decision.

A graphics card only shows as ready when the library that drives it is installed (CuPy for NVIDIA).
`setup_command()` gives the command and `setup()` runs it.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any, Optional

from .detect import Device, backend_status, scan

log = logging.getLogger(__name__)

AUTO_LINE = "Automatic: the best graphics card that is ready, otherwise the processor."


def _label(d: Device) -> str:
    if d.kind == "cpu":
        return f"Processor: {d.name}"
    mem = f", {round(d.memory_mb / 1024)} GB" if d.memory_mb else ""
    return f"Graphics card {d.index + 1}: {d.name}{mem}"


def chosen_keys(cfg: Any) -> list[str]:
    return [str(k).strip().lower() for k in (getattr(cfg.compute, "devices", None) or []) if str(k).strip()]


def sensed(cfg: Any, devices: Optional[list[Device]] = None) -> list[dict[str, Any]]:
    """Every device this computer has, for the switch: key, label, whether it can be used and whether it is chosen."""
    devices = scan() if devices is None else devices
    chosen = chosen_keys(cfg)
    auto = resolve(cfg, devices)["use"] if not chosen else []
    rows = []
    for d in devices:
        ready = d.kind == "cpu" or d.backend_ready
        note = ""
        if not ready:
            note = ("Needs its driver library before GaugeLM can use it: press 'Set up the graphics card'."
                    if d.vendor == "nvidia" else d.install_hint or "GaugeLM cannot use this card.")
        rows.append({"key": d.key, "label": _label(d), "kind": d.kind, "vendor": d.vendor, "ready": ready,
                     "on": d.key in chosen if chosen else d.key in auto, "note": note,
                     "can_set_up": d.kind == "gpu" and not ready and d.vendor == "nvidia"})
    return rows


def resolve(cfg: Any, devices: Optional[list[Device]] = None) -> dict[str, Any]:
    """What the choice means right now: the devices that will really be used, what GaugeLM is told, and
    anything the author should know (a chosen card that is not ready, for example)."""
    devices = scan() if devices is None else devices
    by_key = {d.key: d for d in devices}
    chosen = chosen_keys(cfg)
    notes: list[str] = []
    ready_gpus = sorted((d for d in devices if d.kind == "gpu" and d.backend_ready), key=lambda d: -(d.memory_mb or 0))
    if not chosen:
        use = [ready_gpus[0].key] if ready_gpus else ["cpu"]
        waiting = [d for d in devices if d.kind == "gpu" and not d.backend_ready and d.vendor == "nvidia"]
        if not ready_gpus and waiting:
            notes.append(f"{_label(waiting[0])} is here but GaugeLM cannot use it until its driver library is installed.")
    else:
        use = []
        for k in chosen:
            d = by_key.get(k)
            if d is None:
                notes.append(f"{k} was chosen but is not on this computer now.")
            elif d.kind == "gpu" and not d.backend_ready:
                notes.append(f"{_label(d)} was chosen but is not ready: {d.install_hint or 'no library can drive it yet'}")
            else:
                use.append(k)
        if not use:
            use = ["cpu"]
            notes.append("None of the chosen devices can be used, so the processor is used.")
    gpus = [k for k in use if k != "cpu"]
    gpus.sort(key=lambda k: -(by_key[k].memory_mb or 0))                 # the model lives on the largest card
    # Chosen cards are the only ones this program is allowed to see, so they are numbered from 0 in this order.
    train = [f"cuda:{i}" for i in range(len(gpus))] + (["cpu"] if "cpu" in use and gpus else [])
    return {"chosen": chosen, "automatic": not chosen, "use": use, "gpus": gpus,
            "native_device": "cuda" if gpus else "cpu",
            "train_devices": train if len(train) > 1 else [],
            "visible": ",".join(str(by_key[k].index) for k in gpus) if chosen and gpus else "",
            "engine_gpu": bool(gpus) or not chosen,                      # may the downloaded model use a graphics card
            "labels": [_label(by_key[k]) for k in use], "notes": notes}


def train_devices(cfg: Any) -> list[str]:
    """The devices a training run should share, or [] for the ordinary single-device run. Looks at the
    computer only when more than one device was chosen."""
    if len(chosen_keys(cfg)) < 2:
        return []
    try:
        return list(resolve(cfg)["train_devices"])
    except Exception as e:
        log.warning("could not work out the training devices: %s", e)
        return []


def describe(cfg: Any, devices: Optional[list[Device]] = None) -> str:
    """One or two sentences for a status line."""
    r = resolve(cfg, devices)
    names = r["labels"]
    line = ("GaugeLM uses " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1])
            + (" (chosen automatically)." if r["automatic"] else "."))
    if not r["engine_gpu"]:
        line += " The downloaded model is kept on the processor."
    return " ".join([line] + r["notes"])


def choose(cfg: Any, keys: Optional[list[str]], devices: Optional[list[Device]] = None) -> dict[str, Any]:
    """Save the choice. None, [] or ['auto'] means automatic. Unknown keys are refused."""
    devices = scan() if devices is None else devices
    wanted = [str(k).strip().lower() for k in (keys or []) if str(k).strip()]
    if wanted in ([], ["auto"], ["automatic"]):
        wanted = []
    have = {d.key for d in devices}
    unknown = [k for k in wanted if k not in have]
    if unknown:
        raise ValueError(f"no such device: {', '.join(unknown)}. This computer has: {', '.join(sorted(have))}.")
    cfg.compute.devices = list(dict.fromkeys(wanted))
    r = resolve(cfg, devices)
    # The rest of Gauge reads these two, so they follow the choice.
    cfg.native.device = "auto" if r["automatic"] else r["native_device"]
    cfg.llm.native_device = cfg.native.device
    cfg.llm.use_gpu = r["engine_gpu"]
    cfg.save()
    restart = apply_environment(cfg) == "restart"
    msg = describe(cfg, devices)
    if restart:
        msg += " Close and reopen Gauge for the change of graphics card to take effect."
    return {**r, "restart": restart, "message": msg}


_applied: dict[str, Optional[str]] = {"visible": None}


def apply_environment(cfg: Any) -> str:
    """Limit this program to the chosen graphics cards. Must happen before the graphics library starts,
    so it is called when Gauge starts; a change made later needs a restart. Returns 'set', 'unchanged',
    'nothing' or 'restart'."""
    keys = [k for k in chosen_keys(cfg) if k.startswith("gpu") and k[3:].isdigit()]
    visible = ",".join(k[3:] for k in keys)
    if not keys and _applied["visible"] is None:
        return "nothing"
    if _applied["visible"] == visible:
        return "unchanged"
    started = "cupy" in sys.modules
    if keys:
        # Number the cards the way the vendor's tool lists them, not "fastest first", so the keys match.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        os.environ["HIP_VISIBLE_DEVICES"] = visible
    else:
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            os.environ.pop(name, None)
    _applied["visible"] = visible
    return "restart" if started else "set"


# ---------------------------------------------------------------------------------- setting a card up
def setup_command(vendor: str = "nvidia") -> list[str]:
    """The command that installs the library GaugeLM needs to drive a graphics card, into this Python.
    For NVIDIA that is CuPy with the CUDA components it needs, so only the card's normal driver is required."""
    if vendor != "nvidia":
        raise ValueError("only NVIDIA cards can be set up automatically")
    return [sys.executable, "-m", "pip", "install", "--upgrade", "cupy-cuda12x[ctk]"]


def setup(vendor: str = "nvidia", run: Any = None, timeout: float = 3600.0) -> dict[str, Any]:
    """Install the graphics library. It is a large download (several hundred megabytes). Returns whether
    a graphics card is usable afterwards."""
    before = backend_status()
    if before["cupy"] and before["cupy_devices"] > 0:
        return {"ok": True, "installed": False, "message": "The graphics card is already set up for GaugeLM."}
    from .. import dryrun

    cmd = setup_command(vendor)
    if dryrun.is_dry_run():
        dryrun.hold("install graphics library", " ".join(cmd))
        return {"ok": False, "installed": False, "message": "Dry run: would run " + " ".join(cmd)}
    try:
        res = (run or subprocess.run)(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "installed": False, "message": f"Could not install the graphics library: {e}"}
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-1:] or ["no details"]
        return {"ok": False, "installed": False,
                "message": "Could not install the graphics library (" + tail[0][:200] + "). "
                           "Check the internet connection and that the NVIDIA driver is up to date, then try again."}
    import importlib

    importlib.invalidate_caches()
    after = backend_status()
    if after["cupy"] and after["cupy_devices"] > 0:
        return {"ok": True, "installed": True,
                "message": "The graphics card is set up. GaugeLM will train and write on it from now on."}
    if after["cupy"]:
        return {"ok": False, "installed": True,
                "message": "The graphics library is installed but found no card it can use. Update the NVIDIA driver, "
                           "restart the computer, and open Gauge again."}
    return {"ok": False, "installed": True,
            "message": "The graphics library was installed. Close and reopen Gauge to start using the graphics card."}
