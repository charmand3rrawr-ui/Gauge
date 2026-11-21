"""Tamper evidence for Gauge's own code and models.

`seal()` writes a manifest of SHA-256 hashes for the source tree, installers and model files, and
signs it with an HMAC key kept in the data folder (owner-only permissions where the OS supports
it). `verify()` recomputes and reports changed, added and removed files and whether the signature
still holds. The daemon verifies at the start of each cycle and refuses self-modification and
bot runs while the tree is in an unknown state.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Optional

SEALED_GLOBS = ("gauge/**/*.py", "scripts/*", "installer/*", "install.ps1", "gauge.bat", "Gauge Studio.bat", "Install Gauge.bat", "pyproject.toml",
                "requirements.txt")


def _key(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: Path, extra_dirs: Optional[list[Path]] = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in SEALED_GLOBS:
        for p in root.glob(pattern):
            if p.is_file() and "__pycache__" not in p.parts:
                out[p.relative_to(root).as_posix()] = file_hash(p)
    for d in extra_dirs or []:
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix in (".npz", ".json") and p.name != "train_state.json":
                    out[f"models:{p.relative_to(d).as_posix()}"] = file_hash(p)
    return out


def seal(root: Path, security_dir: Path, extra_dirs: Optional[list[Path]] = None) -> dict[str, Any]:
    files = _files(root, extra_dirs)
    payload = json.dumps({"files": files, "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, sort_keys=True)
    sig = hmac.new(_key(security_dir / "integrity.key"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    security_dir.mkdir(parents=True, exist_ok=True)
    (security_dir / "manifest.json").write_text(json.dumps({"payload": payload, "signature": sig}), encoding="utf-8")
    return {"files": len(files), "signature": sig[:16]}


def verify(root: Path, security_dir: Path, extra_dirs: Optional[list[Path]] = None) -> dict[str, Any]:
    mp = security_dir / "manifest.json"
    if not mp.exists():
        return {"status": "unsealed", "changed": [], "added": [], "removed": [], "signature_ok": None}
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
        payload, sig = m["payload"], m["signature"]
    except (json.JSONDecodeError, KeyError):
        return {"status": "corrupt", "changed": [], "added": [], "removed": [], "signature_ok": False}
    expected = hmac.new(_key(security_dir / "integrity.key"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    sig_ok = hmac.compare_digest(sig, expected)
    sealed = json.loads(payload)["files"]
    current = _files(root, extra_dirs)
    changed = sorted(k for k in sealed if k in current and sealed[k] != current[k])
    added = sorted(k for k in current if k not in sealed)
    removed = sorted(k for k in sealed if k not in current)
    status = "ok" if sig_ok and not changed and not removed else ("tampered" if not sig_ok or changed or removed else "ok")
    if status == "ok" and added:
        status = "ok+new"
    return {"status": status, "changed": changed, "added": added, "removed": removed, "signature_ok": sig_ok,
            "sealed_at": json.loads(payload).get("sealed_at")}
