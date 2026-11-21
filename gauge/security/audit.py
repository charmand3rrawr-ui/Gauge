"""A tamper-evident audit log: every security-relevant event is appended as a JSON line whose hash
includes the previous line's hash. Editing or deleting a past line breaks the chain, and
`verify_chain()` says where."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = ""
        with self.path.open("rb") as f:
            for line in f:
                if line.strip():
                    last = line.decode("utf-8", errors="replace")
        try:
            return json.loads(last)["hash"] if last else "0" * 64
        except (json.JSONDecodeError, KeyError):
            return "0" * 64

    def event(self, kind: str, detail: str, severity: int = 1, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        with self._lock:
            prev = self._last_hash()
            rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "severity": int(severity), "detail": str(detail)[:500],
                   "data": data or {}, "prev": prev}
            rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

    def read(self, limit: int = 200, min_severity: int = 1) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("severity", 1) >= min_severity:
                rows.append(r)
        return rows[-limit:]

    def verify_chain(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"ok": True, "entries": 0}
        prev = "0" * 64
        n = 0
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                return {"ok": False, "entries": n, "broken_at": i, "reason": "unreadable line"}
            h = r.get("hash")
            body = {k: v for k, v in r.items() if k != "hash"}
            if r.get("prev") != prev or hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest() != h:
                return {"ok": False, "entries": n, "broken_at": i, "reason": "hash chain broken (edited or deleted entry)"}
            prev = h
            n += 1
        return {"ok": True, "entries": n}

    def counts(self, since_s: float = 86400.0) -> dict[str, int]:
        cutoff = time.time() - since_s
        out: dict[str, int] = {}
        for r in self.read(limit=5000):
            try:
                ts = time.mktime(time.strptime(r["at"], "%Y-%m-%dT%H:%M:%S"))
            except (KeyError, ValueError):
                continue
            if ts >= cutoff:
                out[r["kind"]] = out.get(r["kind"], 0) + 1
        return out
