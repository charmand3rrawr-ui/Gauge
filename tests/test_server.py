"""Batch P: the headless control server — auth, read endpoints, a safe action, and the dashboard."""
import threading

import pytest
import requests

from gauge.config import GaugeConfig
from gauge.server.app import build_server, serve


@pytest.fixture
def server(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.llm.provider = "fake"
    cfg.learn.min_chapters_for_training = 5
    cfg.ensure_dirs()
    from gauge.write.project import Project
    Project.create(cfg.projects_dir, "Ten Seconds", "a courier who can rewind", ["Sci-fi"], 300)
    httpd, state, tok = build_server(cfg, "127.0.0.1", 0, "secret-token")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, cfg
    httpd.shutdown()


H = {"X-Gauge-Token": "secret-token"}


def test_auth_is_required(server):
    base, _ = server
    assert requests.get(base + "/api/health").status_code == 401
    assert requests.get(base + "/api/projects", headers={"X-Gauge-Token": "wrong"}).status_code == 401
    assert requests.get(base + "/api/health", headers=H).json()["ok"] is True


def test_read_endpoints(server):
    base, _ = server
    projs = requests.get(base + "/api/projects", headers=H).json()["projects"]
    assert any(p["name"] == "Ten Seconds" for p in projs)
    st = requests.get(base + "/api/projects/Ten Seconds", headers=H).json()
    assert "written" in st
    assert requests.get(base + "/api/projects/Nope", headers=H).status_code == 404
    assert requests.get(base + "/api/newsdesk/queue", headers=H).json()["queue"] == []


def test_dashboard_served_without_api_token(server):
    base, _ = server
    html = requests.get(base + "/").text  # the HTML shell is public; the API it calls is not
    assert "Gauge" in html and "X-Gauge-Token" in html


def test_write_action_over_api(server):
    base, _ = server
    r = requests.post(base + "/api/projects/Ten Seconds/write", headers=H, timeout=60).json()
    assert r.get("chapter") == 1 and "title" in r and 0.0 <= r["score"] <= 1.0
    ch = requests.get(base + "/api/projects/Ten Seconds/chapter/1", headers=H).json()
    assert ch["chapter"] == 1 and ch["text"] and "score" in ch["meta"]


def test_serve_refuses_public_bind_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("GAUGE_SERVER_TOKEN", raising=False)
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.ensure_dirs()
    with pytest.raises(ValueError, match="without a token"):
        serve(cfg, host="0.0.0.0")  # non-loopback + no token must be refused
