"""The compute pool: workers you registered or that announced themselves, and how to use them.

* ``discover()`` listens for worker announcements on the LAN for a few seconds. It sends nothing
  and probes nothing: only machines running `gauge worker` are ever found.
* ``WorkerClient`` talks to one worker (health, generate, train) with the shared token.
* ``PoolLLM`` is an LLM-protocol object that spreads generation over the workers that serve a
  model, falling back to the local model when none answers.
* ``DistributedTrainer`` runs model-averaging rounds: every worker (and this machine) trains
  from the same weights on its own shard for K steps; the weights are averaged. Simple, robust to
  slow or lost workers, and a 4M-parameter model is a 16 MB payload on a LAN.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import requests

from ..db import Database
from ..llm.client import CallLog, CallRecord, LLMError
from ..llm.tokens import estimate_tokens
from .protocol import ANNOUNCE_PORT, pack_arrays, parse_announcement, token_hash, unpack_arrays

log = logging.getLogger(__name__)


def discover(seconds: float = 3.0, token: str = "", announce_port: int = ANNOUNCE_PORT, bind: str = "") -> list[dict[str, Any]]:
    """Collect announcements. If a token is given, only workers with the matching hash are kept."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((bind, announce_port))
    except OSError as e:
        log.warning("cannot listen for workers on udp/%d: %s", announce_port, e)
        return []
    sock.settimeout(0.5)
    found: dict[str, dict[str, Any]] = {}
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            try:
                data, (host, _) = sock.recvfrom(65535)
            except TimeoutError:
                continue
            a = parse_announcement(data)
            if not a:
                continue
            if token and a.get("token_hash") != token_hash(token):
                continue
            url = f"http://{host}:{a['port']}"
            found[url] = {"url": url, "name": a.get("name", host), "devices": a.get("devices", []), "llm": a.get("llm", "")}
    finally:
        sock.close()
    return list(found.values())


def _session() -> requests.Session:
    """Workers live on your LAN: never route them through an HTTP proxy from the environment."""
    s = requests.Session()
    s.trust_env = False
    return s


class WorkerClient:
    def __init__(self, url: str, token: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.http = _session()

    def _h(self) -> dict[str, str]:
        return {"X-Gauge-Token": self.token}

    def health(self) -> Optional[dict[str, Any]]:
        try:
            r = self.http.get(f"{self.url}/health", headers=self._h(), timeout=5)
            return r.json() if r.ok else None
        except (requests.RequestException, ValueError):
            return None

    def generate(self, prompt: str, **kw: Any) -> str:
        r = self.http.post(f"{self.url}/generate", headers=self._h(), json={"prompt": prompt, **kw}, timeout=self.timeout)
        if not r.ok:
            raise LLMError(f"worker {self.url}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()["text"]

    def train(self, weights: dict[str, np.ndarray], model_meta: dict[str, Any], tokens: np.ndarray, steps: int, lr: float,
              batch_size: int) -> tuple[dict[str, np.ndarray], float]:
        arrays = dict(weights)
        arrays["__meta__"] = np.frombuffer(json.dumps({"model": model_meta, "steps": steps, "lr": lr, "batch_size": batch_size}).encode("utf-8"), dtype=np.uint8)
        arrays["__tokens__"] = tokens.astype(np.int32)
        r = self.http.post(f"{self.url}/train", headers=self._h(), data=pack_arrays(arrays), timeout=self.timeout)
        if not r.ok:
            raise LLMError(f"worker {self.url}: HTTP {r.status_code} {r.text[:200]}")
        out = unpack_arrays(r.content)
        loss = float(out.pop("__loss__")[0])
        return out, loss


class Pool:
    """Registry of workers in the database plus liveness checks."""

    def __init__(self, db: Database, token: str):
        self.db = db
        self.token = token

    def add(self, url: str, name: str = "", devices: Optional[list] = None, llm: str = "") -> None:
        self.db.upsert_worker(url, name or url, json.dumps(devices or []), llm)

    def remove(self, url: str) -> None:
        self.db.remove_worker(url)

    def workers(self) -> list[dict[str, Any]]:
        return self.db.workers()

    def refresh(self) -> list[dict[str, Any]]:
        """Ping every worker; mark alive/dead; return the alive ones with their health info."""
        alive = []
        for w in self.workers():
            h = WorkerClient(w["url"], self.token).health()
            self.db.touch_worker(w["url"], h is not None, json.dumps(h or {}))
            if h is not None:
                alive.append({**w, "health": h})
        return alive

    def discover_and_register(self, seconds: float = 3.0, announce_port: int = ANNOUNCE_PORT) -> list[dict[str, Any]]:
        found = discover(seconds, self.token, announce_port)
        for f in found:
            self.add(f["url"], f["name"], f["devices"], f["llm"])
        return found

    def clients(self, with_llm: bool = False) -> list[WorkerClient]:
        out = []
        for w in self.refresh():
            if with_llm and not (w["health"].get("llm")):
                continue
            out.append(WorkerClient(w["url"], self.token))
        return out


class PoolLLM:
    """Spread generation over workers that serve a model; fall back to the local LLM."""

    def __init__(self, local, pool: Pool, calls: Optional[CallLog] = None):
        self.local = local
        self.pool = pool
        self.calls = calls or getattr(local, "calls", CallLog())
        self.model = getattr(local, "model", "pool")
        self._clients: list[WorkerClient] = []
        self._i = 0
        self._lock = threading.Lock()
        self._refreshed = 0.0

    def _workers(self) -> list[WorkerClient]:
        if time.monotonic() - self._refreshed > 60:
            try:
                self._clients = self.pool.clients(with_llm=True)
            except Exception as e:
                log.warning("pool refresh failed: %s", e)
            self._refreshed = time.monotonic()
        return self._clients

    def available(self) -> bool:
        return self.local.available() or bool(self._workers())

    def describe(self) -> str:
        return f"pool({len(self._workers())} workers) + {self.local.describe()}"

    def context_window(self) -> int:
        return self.local.context_window()

    def generate(self, prompt: str, system: str = "", temperature=None, max_tokens=None, json_mode: bool = False, schema=None,
                 purpose: str = "generate") -> str:
        from .. import limit

        workers = self._workers()
        if workers:
            with self._lock:
                # Take turns, but under the capacity limit prefer the linked device that has rested longest.
                order = workers[self._i % len(workers):] + workers[:self._i % len(workers)]
                client = min(order, key=lambda c: limit.device(c.url).ready_in())
                self._i += 1
            t0 = time.monotonic()
            try:
                with limit.device(client.url).work():  # each linked device is held to its share
                    out = client.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens,
                                          json_mode=json_mode, schema=schema, purpose=purpose)
                self.calls.add(CallRecord(purpose, f"remote:{client.url}", estimate_tokens(system) + estimate_tokens(prompt), estimate_tokens(out),
                                          int((time.monotonic() - t0) * 1000), True))
                return out
            except Exception as e:
                log.warning("worker %s failed (%s); falling back to local", client.url, e)
                self._clients = [c for c in self._clients if c is not client]
        return self.local.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                                   schema=schema, purpose=purpose)


class DistributedTrainer:
    def __init__(self, pool: Pool, local_device: str = "cpu"):
        self.pool = pool
        self.local_device = local_device

    def round(self, model, tokens: np.ndarray, steps: int, lr: float, batch_size: int = 4, include_local: bool = True) -> dict[str, Any]:
        """One model-averaging round across all live workers (+ this machine). Updates model in place."""
        from ..native.backend import from_numpy, to_numpy
        from .worker import train_on_shard

        clients = self.pool.clients()
        n_parts = len(clients) + (1 if include_local else 0)
        if n_parts == 0:
            raise LLMError("no workers alive and local training disabled")
        shards = np.array_split(tokens, n_parts)
        weights = {k: to_numpy(v) for k, v in model.p.items()}
        meta = {"config": model.cfg.__dict__}
        results: list[tuple[str, Optional[dict[str, np.ndarray]], float]] = []

        def remote(i: int, client: WorkerClient):
            from .. import limit

            try:
                with limit.device(client.url).work():  # each linked device is held to its share
                    w, loss = client.train(weights, meta, shards[i], steps, lr, batch_size)
                return (client.url, w, loss)
            except Exception as e:
                log.warning("worker %s failed in round: %s", client.url, e)
                return (client.url, None, float("nan"))

        with ThreadPoolExecutor(max_workers=max(1, len(clients) + 1)) as ex:
            futs = [ex.submit(remote, i, c) for i, c in enumerate(clients)]
            if include_local:
                lw, lloss = train_on_shard(weights, meta, shards[-1], steps, lr, batch_size, device=self.local_device)
                results.append(("local", lw, lloss))
            results.extend(f.result() for f in futs)
        good = [(u, w, ls) for u, w, ls in results if w is not None]
        if not good:
            raise LLMError("every participant failed this round")
        avg = {k: np.mean([w[k] for _, w, _ in good], axis=0).astype(np.float32) for k in weights}
        for k in model.p:
            model.p[k] = from_numpy(model.xp, avg[k])
        return {"participants": [u for u, _, _ in good], "failed": [u for u, w, _ in results if w is None],
                "loss": round(float(np.nanmean([ls for _, _, ls in good])), 4), "steps_each": steps, "tokens": int(len(tokens))}
