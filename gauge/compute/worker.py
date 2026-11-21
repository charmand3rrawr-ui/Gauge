"""`gauge worker`: lend this machine's compute to your other Gauge installation.

A small HTTP server (standard library) that answers three things, all behind a shared token:
GET /health (name, devices, which model it can serve), POST /generate (run a prompt on this
machine's model: GaugeLM or a local Ollama), POST /train (take weights, a token shard and a step
count, train locally, return the updated weights and loss). It also announces itself on the LAN
by UDP broadcast so the coordinator can find it without scanning anything.

Only machines where you start this process are ever used. The token is required on every
request; announcements carry only its hash.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import socket
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import GaugeConfig
from .detect import scan, summary
from .protocol import ANNOUNCE_PORT, announcement, check_token, pack_arrays, unpack_arrays

log = logging.getLogger(__name__)


def is_private(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


class Guard:
    """Per-peer rate limiting and failed-auth lockout for the worker."""

    def __init__(self, per_minute: int = 120, lockout_failures: int = 10, lockout_s: float = 600.0, allow_public: bool = False, audit=None):
        self.per_minute = per_minute
        self.lockout_failures = lockout_failures
        self.lockout_s = lockout_s
        self.allow_public = allow_public
        self.audit = audit
        self.hits: dict[str, deque] = defaultdict(deque)
        self.failures: dict[str, int] = defaultdict(int)
        self.locked_until: dict[str, float] = {}
        self.lock = threading.Lock()

    def admit(self, host: str) -> Optional[str]:
        """None if the request may proceed, else a reason to refuse."""
        now = time.monotonic()
        with self.lock:
            if not self.allow_public and not is_private(host):
                self._note("worker_rejected_peer", f"non-private peer {host}", 2)
                return "peers outside private networks are not accepted"
            if self.locked_until.get(host, 0) > now:
                return "too many failed tokens; try later"
            q = self.hits[host]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                self._note("rate_limit", f"{host} over {self.per_minute}/min", 2)
                return "rate limit"
            q.append(now)
        return None

    def auth_failed(self, host: str) -> None:
        with self.lock:
            self.failures[host] += 1
            self._note("auth_failure", f"bad token from {host}", 2)
            if self.failures[host] >= self.lockout_failures:
                self.locked_until[host] = time.monotonic() + self.lockout_s
                self.failures[host] = 0
                self._note("auth_lockout", f"{host} locked out for {int(self.lockout_s)}s", 3)

    def auth_ok(self, host: str) -> None:
        with self.lock:
            self.failures[host] = 0

    def _note(self, kind: str, detail: str, sev: int) -> None:
        if self.audit is not None:
            try:
                self.audit.event(kind, detail, sev)
            except Exception:
                pass


class WorkerState:
    def __init__(self, cfg: GaugeConfig, name: str, token: str, llm=None, model_dir: Optional[Path] = None):
        self.cfg = cfg
        self.name = name
        self.token = token
        sec = getattr(cfg, "security", None)
        audit = None
        try:
            from ..security.audit import AuditLog

            audit = AuditLog(cfg.security_dir / "audit.jsonl")
        except Exception:
            audit = None
        self.guard = Guard(per_minute=getattr(sec, "rate_limit_per_minute", 120), lockout_failures=getattr(sec, "auth_lockout_failures", 10),
                           allow_public=getattr(sec, "allow_public_workers", False), audit=audit)
        self.max_body = getattr(sec, "max_request_bytes", 256 * 1024 * 1024)
        self.llm = llm  # any LLM protocol object; None = no generation service
        self.model_dir = model_dir  # for /train: where a model+tokenizer can be loaded (optional)
        self.devices = scan()
        self.busy = 0
        self.lock = threading.Lock()
        self.started = time.time()
        self.jobs = 0

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "devices": summary(self.devices)["devices"], "use": summary(self.devices)["use"],
                "llm": self.llm.describe() if self.llm else "", "busy": self.busy, "jobs": self.jobs,
                "uptime_s": round(time.time() - self.started)}


def train_on_shard(weights: dict[str, np.ndarray], model_meta: dict[str, Any], tokens: np.ndarray, steps: int, lr: float,
                   batch_size: int, device: str = "cpu") -> tuple[dict[str, np.ndarray], float]:
    """Local training from given weights on a token shard; returns (new weights, mean loss)."""
    from ..native.backend import from_numpy, to_numpy
    from ..native.model import AdamW, GaugeLM, ModelConfig
    from ..native.train import batches

    cfg = ModelConfig(**{k: v for k, v in model_meta["config"].items() if k in ModelConfig.__dataclass_fields__})
    model = GaugeLM(cfg, device=device)
    for k in model.p:
        model.p[k] = from_numpy(model.xp, weights[k])
    opt = AdamW(model.p, lr=lr, xp=model.xp)
    rng = np.random.default_rng(int(time.time()) % 100000)
    gen = batches(tokens, cfg.block_size, min(batch_size, max(1, (len(tokens) - cfg.block_size - 1))), rng)
    losses = []
    for _ in range(steps):
        x, y = next(gen)
        _, loss, caches = model.forward(x, y)
        opt.step(model.p, model.backward(caches), lr=lr)
        losses.append(loss)
    return {k: to_numpy(v) for k, v in model.p.items()}, float(np.mean(losses)) if losses else float("nan")


def make_handler(state: WorkerState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GaugeWorker/1"

        def log_message(self, fmt, *args):  # quiet
            log.debug("worker: " + fmt, *args)

        def _auth(self) -> bool:
            host = self.client_address[0]
            refused = state.guard.admit(host)
            if refused:
                self._json(429 if "rate" in refused or "failed" in refused else 403, {"error": refused})
                return False
            if check_token(self.headers.get("X-Gauge-Token", ""), state.token):
                state.guard.auth_ok(host)
                return True
            state.guard.auth_failed(host)
            self._json(401, {"error": "bad token"})
            return False

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length", "0"))
            if n > state.max_body:
                raise ValueError(f"request body too large ({n} bytes)")
            return self.rfile.read(n) if n else b""

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                if not self._auth():
                    return
                self._json(200, state.info())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self._auth():
                return
            if self.path == "/generate":
                if state.llm is None:
                    self._json(503, {"error": "this worker serves no model"})
                    return
                try:
                    req = json.loads(self._body().decode("utf-8"))
                    with state.lock:
                        state.busy += 1
                    try:
                        out = state.llm.generate(req.get("prompt", ""), system=req.get("system", ""), temperature=req.get("temperature"),
                                                 max_tokens=req.get("max_tokens"), json_mode=bool(req.get("json_mode")),
                                                 schema=req.get("schema"), purpose=str(req.get("purpose", "remote")))
                    finally:
                        with state.lock:
                            state.busy -= 1
                            state.jobs += 1
                    self._json(200, {"text": out})
                except Exception as e:
                    self._json(500, {"error": str(e)})
            elif self.path == "/train":
                try:
                    raw = self._body()
                    arrays = unpack_arrays(raw)
                    meta = json.loads(arrays.pop("__meta__").tobytes().decode("utf-8"))
                    tokens = arrays.pop("__tokens__")
                    with state.lock:
                        state.busy += 1
                    try:
                        new_w, loss = train_on_shard(arrays, meta["model"], tokens, int(meta["steps"]), float(meta["lr"]),
                                                     int(meta.get("batch_size", 4)), device=summary(state.devices)["use"])
                    finally:
                        with state.lock:
                            state.busy -= 1
                            state.jobs += 1
                    new_w["__loss__"] = np.asarray([loss], dtype=np.float64)
                    body = pack_arrays(new_w)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as e:
                    log.exception("train failed")
                    self._json(500, {"error": str(e)})
            else:
                self._json(404, {"error": "not found"})

    return Handler


class Announcer(threading.Thread):
    def __init__(self, state: WorkerState, port: int, interval: float = 5.0, announce_port: int = ANNOUNCE_PORT,
                 target: str = "<broadcast>"):
        super().__init__(daemon=True)
        self.state, self.port, self.interval, self.announce_port, self.target = state, port, interval, announce_port, target
        self.stop = threading.Event()

    def run(self) -> None:
        payload = announcement(self.state.name, self.port, self.state.token, summary(self.state.devices)["devices"],
                               self.state.llm.describe() if self.state.llm else "")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self.stop.is_set():
                try:
                    sock.sendto(payload, (self.target, self.announce_port))
                except OSError as e:
                    log.debug("announce failed: %s", e)
                self.stop.wait(self.interval)
        finally:
            sock.close()


class Worker:
    def __init__(self, cfg: GaugeConfig, name: str, token: str, host: str = "0.0.0.0", port: int = 47632, llm=None,
                 announce: bool = True, announce_port: int = ANNOUNCE_PORT, announce_target: str = "<broadcast>"):
        self.state = WorkerState(cfg, name, token, llm, cfg.native_dir)
        self.server = ThreadingHTTPServer((host, port), make_handler(self.state))
        self.port = self.server.server_address[1]
        self.announcer = Announcer(self.state, self.port, announce_port=announce_port, target=announce_target) if announce else None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        if self.announcer:
            self.announcer.start()

    def stop(self) -> None:
        if self.announcer:
            self.announcer.stop.set()
        self.server.shutdown()
        self.server.server_close()

    def serve(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
