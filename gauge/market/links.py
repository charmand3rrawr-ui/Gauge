"""UTM-tagged links and a privacy-respecting click log (roadmap item 71).

Tag the links you share so you can tell which channel actually sends readers, without tracking the
readers themselves: the click log stores only aggregate counts by campaign and source — no IP, no
identifier, nothing about a person. You own the links and the log; it is analytics for you, not
surveillance of them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse


def utm_link(url: str, source: str, medium: str = "social", campaign: str = "") -> str:
    """Append UTM parameters to a URL, preserving any existing query string."""
    parts = urlparse(url)
    params = [("utm_source", source), ("utm_medium", medium)]
    if campaign:
        params.append(("utm_campaign", campaign))
    extra = urlencode([(k, v) for k, v in params if v])
    query = f"{parts.query}&{extra}" if parts.query else extra
    return urlunparse(parts._replace(query=query))


def tagged_links(url: str, sources: list[str], medium: str = "social", campaign: str = "") -> dict[str, str]:
    return {s: utm_link(url, s, medium, campaign) for s in sources}


@dataclass
class ClickLog:
    """Aggregate-only click counts. No personal data is ever recorded."""
    path: Path

    def _load(self) -> dict:
        p = Path(self.path)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def record(self, campaign: str, source: str) -> None:
        data = self._load()
        key = f"{campaign or '-'}|{source or '-'}"
        data[key] = int(data.get(key, 0)) + 1
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def totals(self) -> dict[str, int]:
        return dict(sorted(self._load().items(), key=lambda kv: -kv[1]))

    def by_source(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, n in self._load().items():
            src = key.split("|", 1)[-1]
            out[src] = out.get(src, 0) + int(n)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
