"""Back up and restore the data directory, with per-file integrity.

A backup is an ordinary ``.tar.gz`` of the data directory plus a ``MANIFEST.json`` recording the
SHA-256 of every file. ``verify_backup`` recomputes those hashes so a corrupted or tampered archive is
caught before you trust it, and ``restore_backup`` verifies before it writes. Model weights can be large
and are reproducible by retraining, so they are excluded by default; pass ``include_models`` to keep them.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iter_files(root: Path, include_models: bool) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel_parts = set(p.relative_to(root).parts)
        if rel_parts & _SKIP_DIRS:
            continue
        if not include_models and "models" in p.relative_to(root).parts and p.suffix in (".npz", ".bin", ".pt", ".safetensors"):
            continue
        out.append(p)
    return out


@dataclass
class BackupResult:
    path: Path
    files: int
    bytes: int


def make_backup(data_dir: Path, out_path: Path, include_models: bool = False) -> BackupResult:
    """Write a verified tar.gz of the data directory to out_path."""
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = _iter_files(data_dir, include_models)
    manifest = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "root": data_dir.name, "include_models": include_models, "files": {}}
    total = 0
    with tarfile.open(out_path, "w:gz") as tar:
        for p in files:
            rel = str(p.relative_to(data_dir))
            raw = p.read_bytes()
            manifest["files"][rel] = {"sha256": _sha256(raw), "bytes": len(raw)}
            total += len(raw)
            info = tarfile.TarInfo(name=f"data/{rel}")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        mraw = json.dumps(manifest, indent=2).encode("utf-8")
        minfo = tarfile.TarInfo(name="MANIFEST.json")
        minfo.size = len(mraw)
        tar.addfile(minfo, io.BytesIO(mraw))
    return BackupResult(out_path, len(files), total)


def _read_manifest(tar: tarfile.TarFile) -> dict:
    m = tar.extractfile("MANIFEST.json")
    if m is None:
        raise ValueError("backup has no MANIFEST.json")
    return json.loads(m.read().decode("utf-8"))


@dataclass
class VerifyResult:
    ok: bool
    files: int
    mismatched: list[str]
    missing: list[str]
    created: str = ""


def verify_backup(path: Path) -> VerifyResult:
    """Recompute every file's hash against the manifest. ok is True only when all match."""
    with tarfile.open(path, "r:gz") as tar:
        manifest = _read_manifest(tar)
        mismatched, missing = [], []
        for rel, meta in manifest["files"].items():
            try:
                member = tar.extractfile(f"data/{rel}")
            except KeyError:
                member = None
            if member is None:
                missing.append(rel)
                continue
            if _sha256(member.read()) != meta["sha256"]:
                mismatched.append(rel)
    return VerifyResult(not (mismatched or missing), len(manifest["files"]), mismatched, missing,
                        manifest.get("created", ""))


def restore_backup(path: Path, dest_dir: Path, overwrite: bool = False) -> dict:
    """Verify, then extract the data files into dest_dir. Refuses a corrupt archive and, unless overwrite,
    a non-empty destination. Path traversal in member names is rejected."""
    dest = Path(dest_dir)
    v = verify_backup(path)
    if not v.ok:
        raise ValueError(f"backup failed integrity check: {len(v.mismatched)} mismatched, {len(v.missing)} missing")
    if dest.exists() and any(dest.iterdir()) and not overwrite:
        raise ValueError(f"{dest} is not empty; pass overwrite=True to restore into it")
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(path, "r:gz") as tar:
        manifest = _read_manifest(tar)
        for rel in manifest["files"]:
            if ".." in Path(rel).parts or Path(rel).is_absolute():
                raise ValueError(f"unsafe path in backup: {rel}")
            try:
                member = tar.extractfile(f"data/{rel}")
            except KeyError:
                member = None
            if member is None:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(member.read())
            written += 1
    return {"restored": written, "into": str(dest), "created": v.created}
