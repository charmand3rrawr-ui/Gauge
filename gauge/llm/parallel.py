"""Run independent LLM calls side by side. Ollama serves concurrent requests when
``OLLAMA_NUM_PARALLEL`` > 1 (the Windows installer sets it); other servers batch too."""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")
R = TypeVar("R")


def run_parallel(fn: Callable[[T], R], items: Sequence[T], workers: int = 4,
                 label: str = "task") -> list[Optional[R]]:
    """Apply ``fn`` to every item, preserving order. A failing item yields ``None`` and a log
    line instead of taking the others down with it."""
    items = list(items)
    if not items:
        return []
    workers = max(1, min(workers, len(items)))

    def safe(item: T) -> Optional[R]:
        try:
            return fn(item)
        except Exception as e:  # noqa: BLE001 - isolation is the point
            log.warning("%s failed for %r: %s", label, getattr(item, "name", item), e)
            return None

    if workers == 1:
        return [safe(i) for i in items]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gauge-llm") as ex:
        return list(ex.map(safe, items))
