"""Gauge's Claude calls, made with Anthropic's real library against a stand-in server on this computer.

Skipped unless the library is installed (`pip install anthropic`). No key and no internet are needed:
the library is pointed at a local server that speaks enough of the API (model lookup, and a streamed
message) to prove the library accepts the way Gauge calls it and that Gauge reads its answers
correctly. It does not prove what the real service would answer."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("anthropic")

from gauge.llm import claude, make_llm  # noqa: E402
from gauge.llm.client import LLMError  # noqa: E402

TEXT = "The lamp went out. Mara held her breath and listened."


class _Api(BaseHTTPRequestHandler):
    seen: list = []
    refuse = False

    def log_message(self, *_):
        pass

    def _authorised(self) -> bool:
        return self.headers.get("x-api-key") == "good-key" or self.headers.get("authorization") == "Bearer good-key"

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._authorised():
            self._json(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
            return
        model = self.path.split("/")[-1].split("?")[0]
        self._json(200, {"id": model, "type": "model", "display_name": model, "created_at": "2026-01-01T00:00:00Z"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Api.seen.append({"body": body, "beta": self.headers.get("anthropic-beta", "")})
        if not self._authorised():
            self._json(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
            return
        events: list[dict] = [{"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 120, "output_tokens": 1}}}]
        delta: dict = {"stop_reason": "end_turn", "stop_sequence": None}
        if _Api.refuse:
            delta = {"stop_reason": "refusal", "stop_sequence": None,
                     "stop_details": {"type": "refusal", "category": "frontier_llm", "explanation": "declined"}}
        else:
            events += [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": TEXT[:20]}},
                {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": TEXT[20:]}},
                {"type": "content_block_stop", "index": 1}]
        events += [{"type": "message_delta", "delta": delta, "usage": {"output_tokens": 0 if _Api.refuse else 60}},
                   {"type": "message_stop"}]
        data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def api(monkeypatch, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Api.seen, _Api.refuse = [], False
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("GAUGE_CLAUDE_KEY_FILE", str(tmp_path / "home" / "claude-key.txt"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    yield _Api
    server.shutdown()


def _cfg(cfg):
    cfg.llm.claude, cfg.llm.claude_dir = cfg.claude, str(cfg.root)
    cfg.llm.provider, cfg.llm.model, cfg.llm.retries = "claude", "claude-opus-5-5", 0
    return cfg


def test_the_real_library_accepts_gauges_request_and_gauge_reads_its_answer(cfg, api):
    cfg = _cfg(cfg)
    claude.save_key("good-key")
    llm = make_llm(cfg.llm)
    assert llm.check()["ok"] and llm.available()
    out = llm.generate("Write one line.", system="Plain words.", temperature=0.9, max_tokens=200, purpose="school:lesson")
    assert out == TEXT                                    # the thinking block is skipped, the text pieces are joined
    sent = api.seen[-1]
    assert sent["beta"] == claude.FALLBACK_BETA and sent["body"]["fallbacks"] == "default"
    assert sent["body"]["output_config"] == {"effort": "medium"} and sent["body"]["stream"] is True
    assert sent["body"]["max_tokens"] == 16000 and sent["body"]["system"] == "Plain words."
    assert "temperature" not in sent["body"] and "thinking" not in sent["body"]
    today = claude.Ledger(cfg.root).today()
    assert (today["input"], today["output"]) == (120, 60) and today["usd"] == pytest.approx((120 * 4 + 60 * 20) / 1e6)


def test_the_real_library_reports_a_refusal_and_a_bad_key(cfg, api):
    cfg = _cfg(cfg)
    claude.save_key("good-key")
    api.refuse = True
    with pytest.raises(claude.ClaudeRefused) as e:
        make_llm(cfg.llm).generate("anything")
    assert e.value.category == "frontier_llm"
    api.refuse = False
    claude.ClaudeLLM(cfg.llm, "claude-haiku-5-5").generate("x")
    assert "fallbacks" not in api.seen[-1]["body"] and api.seen[-1]["beta"] == ""
    claude.save_key("wrong-key")
    llm = make_llm(cfg.llm)
    assert "did not accept the key" in llm.check()["message"]
    with pytest.raises(LLMError, match="did not accept the key"):
        llm.generate("x")
