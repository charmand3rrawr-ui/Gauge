"""The control server: a small, hardened HTTP API over Gauge, plus a read-only dashboard.

Endpoints (all under /api, all token-authenticated):
  GET  /api/health                         liveness + version
  GET  /api/projects                       list projects
  GET  /api/projects/<name>                status of one project
  GET  /api/projects/<name>/chapter/<n>    a chapter's text and meta
  GET  /api/newsdesk/queue                 the newsdesk review queue
  POST /api/projects/<name>/write          write the next chapter (mutating)
  POST /api/cycle                          run one daemon learning cycle (mutating)

GET / serves a dashboard that calls the API with a token you paste in. Security: loopback by default;
binding to a non-loopback host requires a token and accepts non-private peers (still token-gated), and
`serve` prints a warning. The worker's Guard (rate limit, auth lockout) applies to every request.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import unquote

from ..compute.protocol import check_token
from ..compute.worker import Guard, is_private
from ..config import GaugeConfig

log = logging.getLogger(__name__)
VERSION = "1"


class ServerState:
    def __init__(self, cfg: GaugeConfig, token: str, allow_public: bool = False):
        self.cfg = cfg
        self.token = token
        audit = None
        try:
            from ..security.audit import AuditLog
            audit = AuditLog(cfg.security_dir / "audit.jsonl")
        except Exception:
            audit = None
        sec = cfg.security
        self.guard = Guard(per_minute=sec.rate_limit_per_minute, lockout_failures=sec.auth_lockout_failures,
                           allow_public=allow_public, audit=audit)
        self.max_body = 8 * 1024 * 1024  # control API bodies are small
        self._pipe = None

    def pipeline(self):
        if self._pipe is None:
            from ..db import Database
            from ..write import WritePipeline
            self._pipe = WritePipeline(self.cfg, Database(self.cfg.db_path))
        return self._pipe

    # ---- operations the API exposes --------------------------------------------------------------
    def projects(self) -> list[dict[str, Any]]:
        from ..write.project import Project
        out = []
        for name in Project.list(self.cfg.projects_dir):
            try:
                p = Project(self.cfg.projects_dir, name)
                out.append({"name": name, "written": p.state.written, "genres": p.state.genres})
            except Exception:
                out.append({"name": name})
        return out

    def status(self, name: str) -> Optional[dict[str, Any]]:
        from ..write.project import Project
        if not Project.exists(self.cfg.projects_dir, name):
            return None
        return self.pipeline().status(self.pipeline().open_project(name))

    def chapter(self, name: str, n: int) -> Optional[dict[str, Any]]:
        from ..write.project import Project
        if not Project.exists(self.cfg.projects_dir, name):
            return None
        p = Project(self.cfg.projects_dir, name)
        text = p.read_chapter(n)
        if text is None:
            return None
        return {"chapter": n, "text": text, "meta": p.read_meta(n)}

    def newsdesk_queue(self) -> list[dict[str, Any]]:
        from ..newsdesk.desk import queue_for
        return [{"id": i.id, "title": i.title, "status": i.status, "clean": i.clean} for i in queue_for(self.cfg).items()]

    def write_next(self, name: str) -> dict[str, Any]:
        from ..write.project import Project
        if not Project.exists(self.cfg.projects_dir, name):
            return {"error": "no such project"}
        pipe = self.pipeline()
        if not pipe.llm.available():
            return {"error": f"LLM not reachable ({pipe.llm.describe()})"}
        res = pipe.write_next(pipe.open_project(name))
        return {"chapter": res.number, "title": res.title, "score": round(res.score, 3), "path": str(res.path)}

    def run_cycle(self) -> dict[str, Any]:
        from ..daemon import Daemon
        return Daemon(self.cfg, self.pipeline().db).once()


def make_handler(state: ServerState):
    dashboard = _dashboard_html()

    class Handler(BaseHTTPRequestHandler):
        server_version = "GaugeServer/" + VERSION

        def log_message(self, fmt, *args):
            log.debug("server: " + fmt, *args)

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
            self._json(401, {"error": "token required"})
            return False

        def _json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> dict:
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n > state.max_body:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path == "/index.html":
                self._html(dashboard)
                return
            if not self.path.startswith("/api/"):
                self._json(404, {"error": "not found"})
                return
            if not self._auth():
                return
            parts = [unquote(p) for p in self.path.split("?")[0].strip("/").split("/") if p]
            try:
                if parts == ["api", "health"]:
                    self._json(200, {"ok": True, "version": VERSION, "data_dir": str(state.cfg.root)})
                elif parts == ["api", "projects"]:
                    self._json(200, {"projects": state.projects()})
                elif len(parts) == 3 and parts[:2] == ["api", "projects"]:
                    st = state.status(parts[2])
                    self._json(200 if st else 404, st or {"error": "no such project"})
                elif len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "chapter":
                    ch = state.chapter(parts[2], int(parts[4]))
                    self._json(200 if ch else 404, ch or {"error": "no such chapter"})
                elif parts == ["api", "newsdesk", "queue"]:
                    self._json(200, {"queue": state.newsdesk_queue()})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as e:
                self._json(500, {"error": str(e)})

        def do_POST(self):  # noqa: N802
            if not self._auth():
                return
            parts = [unquote(p) for p in self.path.split("?")[0].strip("/").split("/") if p]
            try:
                if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "write":
                    self._json(200, state.write_next(parts[2]))
                elif parts == ["api", "cycle"]:
                    self._json(200, state.run_cycle())
                else:
                    self._json(404, {"error": "not found"})
            except Exception as e:
                self._json(500, {"error": str(e)})

    return Handler


def build_server(cfg: GaugeConfig, host: Optional[str] = None, port: Optional[int] = None,
                 token: Optional[str] = None) -> tuple[ThreadingHTTPServer, ServerState, str]:
    """Create (but do not start) the server. Returns (httpd, state, token). Generates a token if none."""
    host = host or cfg.server.host
    port = cfg.server.port if port is None else port
    tok = token or cfg.server.token or os.environ.get("GAUGE_SERVER_TOKEN", "") or secrets.token_urlsafe(24)
    allow_public = not is_private(host) and host not in ("127.0.0.1", "localhost", "::1")
    state = ServerState(cfg, tok, allow_public=allow_public)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    return httpd, state, tok


def serve(cfg: GaugeConfig, host: Optional[str] = None, port: Optional[int] = None, token: Optional[str] = None) -> None:
    host = host or cfg.server.host
    exposed = host not in ("127.0.0.1", "localhost", "::1")
    if exposed and not (token or cfg.server.token or os.environ.get("GAUGE_SERVER_TOKEN")):
        raise ValueError("refusing to bind to a non-loopback host without a token; set server.token or pass --token")
    httpd, state, tok = build_server(cfg, host, port, token)
    where = f"http://{host}:{httpd.server_address[1]}"
    print(f"Gauge server on {where}  (dashboard at {where}/ )")
    print(f"API token: {tok}")
    if exposed:
        print("WARNING: bound to a non-loopback address. The control API is reachable from the network. "
              "It is token-gated and rate-limited, but put it behind a VPN or reverse proxy with TLS; "
              "do not expose it to the open internet.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.shutdown()


def _dashboard_html() -> str:
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Gauge</title>
<style>:root{--fg:#1a1a1a;--bg:#fff;--muted:#666;--rule:#e5e5e5;--accent:#8a1b1b}
@media(prefers-color-scheme:dark){:root{--fg:#e8e8e8;--bg:#141414;--muted:#9a9a9a;--rule:#2a2a2a;--accent:#e07a7a}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:16px}h1{font-size:1.5rem}input,button{font:inherit;padding:8px;border:1px solid var(--rule);border-radius:6px;background:var(--bg);color:var(--fg)}
button{cursor:pointer;background:var(--accent);color:#fff;border:0}.card{border:1px solid var(--rule);border-radius:8px;padding:12px;margin:10px 0}
.muted{color:var(--muted);font-size:.85rem}a{color:var(--accent)}</style></head><body><div class="wrap">
<h1>Gauge</h1><p class="muted">Headless control dashboard. Paste your API token; it stays in this browser.</p>
<p><input id="tok" type="password" placeholder="API token" size="40"> <button onclick="save()">Connect</button></p>
<div id="out"></div>
<script>
const t=()=>localStorage.getItem('gauge_token')||'';
function save(){localStorage.setItem('gauge_token',document.getElementById('tok').value);load();}
async function api(p,m){const r=await fetch(p,{method:m||'GET',headers:{'X-Gauge-Token':t()}});return r.json();}
async function load(){
 const out=document.getElementById('out');out.innerHTML='loading...';
 try{const h=await api('/api/health');if(h.error){out.innerHTML='<div class=card>'+h.error+'</div>';return;}
 const d=await api('/api/projects');let html='<div class=card>Connected. data: '+h.data_dir+'</div>';
 for(const p of d.projects){html+='<div class=card><b>'+p.name+'</b> <span class=muted>'+(p.written||0)+' chapters</span> '
 +'<button onclick="writeNext(\\''+p.name+'\\')">Write next</button></div>';}
 const q=await api('/api/newsdesk/queue');if(q.queue&&q.queue.length){html+='<div class=card><b>Newsdesk queue</b><br>';
 for(const i of q.queue){html+=i.status+' — '+i.title+'<br>';}html+='</div>';}
 out.innerHTML=html;}catch(e){out.innerHTML='<div class=card>error: '+e+'</div>';}}
async function writeNext(n){const r=await api('/api/projects/'+encodeURIComponent(n)+'/write','POST');
 alert(r.error?r.error:('Chapter '+r.chapter+': '+r.title+' (score '+r.score+')'));load();}
if(t()){document.getElementById('tok').value=t();load();}
</script></div></body></html>"""
