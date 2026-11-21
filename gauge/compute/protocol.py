"""Wire format shared by workers and the coordinator: token check, announcements, numpy payloads."""
from __future__ import annotations

import hashlib
import hmac
import io
import json
from typing import Any

import numpy as np

ANNOUNCE_PORT = 47631
ANNOUNCE_MAGIC = "GAUGE-WORKER-1"


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def check_token(provided: str, expected: str) -> bool:
    return bool(expected) and hmac.compare_digest(provided or "", expected)


def announcement(name: str, port: int, token: str, devices: list[dict[str, Any]], llm: str) -> bytes:
    return json.dumps({"magic": ANNOUNCE_MAGIC, "name": name, "port": port, "token_hash": token_hash(token),
                       "devices": [{"kind": d.get("kind"), "name": d.get("name"), "backend_ready": d.get("backend_ready")} for d in devices][:6],
                       "llm": llm}).encode("utf-8")


def parse_announcement(data: bytes) -> dict[str, Any] | None:
    try:
        d = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict) or d.get("magic") != ANNOUNCE_MAGIC or not isinstance(d.get("port"), int):
        return None
    return d


def pack_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def unpack_arrays(data: bytes, max_bytes: int = 512 * 1024 * 1024) -> dict[str, np.ndarray]:
    if len(data) > max_bytes:
        raise ValueError(f"payload too large ({len(data)} bytes)")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:  # never unpickle from the network
        return {k: z[k] for k in z.files}
