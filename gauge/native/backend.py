"""Array backend: numpy on the CPU, CuPy on an NVIDIA GPU when installed (``pip install cupy-cuda12x``).

CuPy is a drop-in numpy on CUDA, which is a library like numpy, not model code: every line of the
model stays ours, it just runs 10-100x faster. ``native.device`` selects: ``auto`` (GPU if present),
``cuda`` or ``cpu``."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def get_xp(device: str = "auto"):
    if device in ("auto", "cuda"):
        try:
            import cupy as cp  # type: ignore

            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp
            if device == "cuda":
                log.warning("cuda requested but no CUDA device found; using numpy")
        except Exception:
            if device == "cuda":
                log.warning("cuda requested but cupy is not installed (pip install cupy-cuda12x); using numpy")
    return np


def is_gpu(xp) -> bool:
    return xp is not np


def to_numpy(a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    try:
        return a.get()  # cupy
    except AttributeError:
        return np.asarray(a)


def from_numpy(xp, a: np.ndarray):
    return a if xp is np else xp.asarray(a)


def device_name(xp) -> str:
    if xp is np:
        return "cpu (numpy)"
    try:
        props = xp.cuda.runtime.getDeviceProperties(0)
        return f"cuda ({props['name'].decode() if isinstance(props['name'], bytes) else props['name']})"
    except Exception:
        return "cuda"
