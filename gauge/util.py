"""Tiny shared helpers."""
from __future__ import annotations

from datetime import datetime, timezone


def seed_all(seed: int) -> None:
    """Seed the process-wide RNGs so a run is reproducible. Call once at the start of a run; pass the
    same seed to any component that keeps its own RNG (e.g. the prompt bandit)."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite-friendly, no deprecation warnings)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
