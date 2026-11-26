"""Which model should write: the strongest one this computer can actually run.

The largest single step in how well Gauge writes is the model doing the writing. The installer picks a
safe one for the hardware it finds, and stops at a mid-sized model. A computer with a large graphics
card can run something considerably stronger, and this says so, and fetches it when asked.

A model "fits" when its weights fit in the graphics card's memory with some room for the text it is
reading. On a computer without a suitable card it must fit in ordinary memory and will be slow, so
nothing larger than a mid-sized model is suggested there.

Sizes are approximate (the standard 4-bit builds that Ollama serves by default).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any, Optional

log = logging.getLogger(__name__)

# name, memory it needs in GB (weights plus working room), rank (higher writes better), licence note
LADDER: list[tuple[str, float, int, str]] = [
    ("llama3.2:3b", 3.5, 1, "Llama Community License"),
    ("qwen2.5:7b", 6.5, 2, "Apache-2.0"),
    ("llama3.1:8b", 6.5, 2, "Llama Community License"),
    ("qwen2.5:14b", 11.5, 3, "Apache-2.0"),
    ("qwen2.5:32b", 23.0, 4, "Apache-2.0"),
    ("llama3.3:70b", 46.0, 5, "Llama Community License"),
]
SLOW_LIMIT = 3          # the best rank worth running without a graphics card


def rank_of(model: str) -> int:
    for name, _need, rank, _lic in LADDER:
        if model == name or model.split(":")[0] == name.split(":")[0] and model.split(":")[-1] == name.split(":")[-1]:
            return rank
    low = model.lower()
    for size, rank in (("70b", 5), ("72b", 5), ("32b", 4), ("27b", 4), ("14b", 3), ("12b", 3), ("9b", 2), ("8b", 2), ("7b", 2), ("3b", 1)):
        if size in low:
            return rank
    return 0


def hardware() -> dict[str, float]:
    """Graphics memory (the largest card) and ordinary memory, in GB."""
    from .. import limit

    vram = 0.0
    try:
        from ..compute import scan

        vram = max((float(d.memory_mb or 0) / 1024 for d in scan() if d.kind == "gpu" and d.vendor in ("nvidia", "amd")), default=0.0)
    except Exception as e:
        log.warning("could not look at the graphics card: %s", e)
    return {"vram_gb": round(vram, 1), "ram_gb": float(limit.total_memory_gb() or 0)}


def advise(current: str, vram_gb: float, ram_gb: float, installed: Optional[list[str]] = None) -> dict[str, Any]:
    """The strongest model that fits, how it compares with the one in use, and what to do about it."""
    installed = installed or []
    fits = []
    for name, need, rank, lic in LADDER:
        on_card = vram_gb > 0 and need <= vram_gb
        in_memory = ram_gb > 0 and need <= ram_gb * 0.6 and rank <= SLOW_LIMIT
        if on_card or in_memory:
            fits.append({"model": name, "needs_gb": need, "rank": rank, "licence": lic, "on_card": on_card,
                         "installed": name in installed or f"{name}:latest" in installed})
    if not fits:
        return {"current": current, "best": None, "better": False, "fits": [],
                "message": "Gauge could not work out a model for this computer (it could not read how much memory there is)."}
    top = max(f["rank"] for f in fits)
    choices = [f for f in fits if f["rank"] == top]
    best = next((f for f in choices if f["installed"]), None) or next((f for f in choices if f["licence"] == "Apache-2.0"), choices[0])
    now = rank_of(current)
    unknown = bool(current) and now == 0              # a model Gauge cannot place: say so, do not claim to beat it
    better = best["rank"] > now and not unknown
    where = "on the graphics card" if best["on_card"] else "in ordinary memory, which is slow"
    if better:
        msg = (f"This computer can run {best['model']} ({where}), a stronger writer than {current or 'the one in use'}. "
               + ("It is already downloaded." if best["installed"] else f"It is a download of about {best['needs_gb'] - 3:.0f} GB."))
    elif unknown:
        msg = f"Gauge does not know how {current} compares. The strongest model it knows this computer can run is {best['model']}."
    else:
        msg = f"{current} is already the strongest kind of model this computer can run well."
    return {"current": current, "current_rank": now, "best": best["model"], "better": better, "fits": fits,
            "needs_download": better and not best["installed"], "message": msg}


def fetch(model: str, run: Any = None, timeout: float = 6 * 3600) -> dict[str, Any]:
    """Download a model with the local engine. A large download; returns whether it worked."""
    from .. import dryrun

    exe = shutil.which("ollama")
    if exe is None and run is None:
        return {"ok": False, "message": "The Ollama program was not found. Open the Ollama app, then try again."}
    cmd = [exe or "ollama", "pull", model]
    if dryrun.is_dry_run():
        dryrun.hold("download a model", " ".join(cmd))
        return {"ok": False, "message": "Dry run: would run " + " ".join(cmd)}
    try:
        res = (run or subprocess.run)(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "message": f"Could not download {model}: {e}"}
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-1:] or ["no details"]
        return {"ok": False, "message": f"Could not download {model} ({tail[0][:160]}). Check the internet connection and free disk space."}
    return {"ok": True, "message": f"{model} is downloaded."}
