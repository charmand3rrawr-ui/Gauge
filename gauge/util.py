"""Tiny shared helpers."""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite-friendly, no deprecation warnings)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
