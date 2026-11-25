"""A proxy that keeps a language model inside the capacity limit (see gauge.limit)."""
from __future__ import annotations

from typing import Any, Optional


class GovernedLLM:
    """Delegates everything to the wrapped model, but makes each generation one governed piece of work:
    calls happen one at a time and are spaced so the device stays at or under its share."""

    def __init__(self, inner: Any, governor: Any, load: Optional[float] = None):
        self._inner = inner
        self._governor = governor
        self._load = load

    def __getattr__(self, name: str) -> Any:  # available(), describe(), calls, model, context_window(), ...
        return getattr(self._inner, name)

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False,
                 schema: Optional[dict[str, Any]] = None, purpose: str = "generate") -> str:
        with self._governor.work(self._load):
            return self._inner.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens,
                                        json_mode=json_mode, schema=schema, purpose=purpose)
