"""A content-addressed response cache for the LLM (roadmap item 61).

Wraps any LLM. The same structured request (a brief, a review, a memory update, a translation) made twice
returns the stored answer instead of paying for it again. It caches only the deterministic, repeatable
calls — low temperature, or JSON/schema bookkeeping — and never caches creative prose, so chapter variety
is untouched. Off by default; turn on with ``llm.cache``. Each answer lives in its own file keyed by a
hash of the full request, so concurrent agents never contend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class CachingLLM:
    """A transparent caching proxy. Delegates everything it does not override to the wrapped model."""

    def __init__(self, inner: Any, cache_dir: Path, max_temperature: float = 0.4):
        self._inner = inner
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_t = max_temperature
        self.hits = 0
        self.misses = 0

    def __getattr__(self, name: str) -> Any:  # proxy available(), describe(), calls, model, context_window(), ...
        return getattr(self._inner, name)

    def _cacheable(self, temperature: Optional[float], json_mode: bool, schema: Optional[dict]) -> bool:
        if json_mode or schema is not None:
            return True
        return temperature is not None and temperature <= self._max_t

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False,
                 schema: Optional[dict[str, Any]] = None, purpose: str = "generate") -> str:
        if not self._cacheable(temperature, json_mode, schema):
            return self._inner.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens,
                                        json_mode=json_mode, schema=schema, purpose=purpose)
        key = _key({"p": prompt, "s": system, "t": temperature, "m": max_tokens, "j": json_mode,
                    "sc": schema, "pu": purpose, "model": getattr(self._inner, "model", "")})
        path = self._dir / f"{key}.txt"
        if path.exists():
            try:
                self.hits += 1
                return path.read_text(encoding="utf-8")
            except OSError:
                pass
        out = self._inner.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens,
                                   json_mode=json_mode, schema=schema, purpose=purpose)
        self.misses += 1
        try:
            path.write_text(out, encoding="utf-8")
        except OSError:
            pass
        return out

    def cache_stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(list(self._dir.glob("*.txt")))}
