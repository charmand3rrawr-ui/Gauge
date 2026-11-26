"""Free AI helpers: hosted models with a free allowance that teach GaugeLM and check its work.

No real provider is contacted. A stand-in server on this computer speaks the same OpenAI-style protocol
(model list, key check, chat), and Gauge's real client code talks to it over HTTP."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner

from gauge.cli import main
from gauge.gui import HELPERS_NOTE, StudioController
from gauge.llm import helpers
from gauge.llm.client import LLMError
from gauge.native import checkwork, instruction_rows, school

TEXT = ("Rin counted the exits before she sat down and rubbed the scar on her palm. The bag was still sealed and the "
        "debt was still hers, and nobody in the room knew she had already run once tonight. ") * 30
SCENE = "The lamp went out. Mara held her breath and listened to the stairs, and counted to nine before she moved. " * 3
CHAPTER = "\n\n".join(
    f"Paragraph {i}. Rin ran down the wet street with the bag under her arm and did not look back at the men. "
    "The rain was loud on the roofs and her breath was loud in her ears. She counted the doors as she passed them, "
    "because counting was better than thinking about the broken seal." for i in range(1, 13))


# ------------------------------------------------------------------ a stand-in provider
class _Provider(BaseHTTPRequestHandler):
    chats: list = []
    lookups: list = []
    mode = "ok"                                           # ok | busy (429) | pay (402) | think
    models = [{"id": "text-embedding-3"}, {"id": "llama-3.1-8b-instant"}, {"id": "llama-3.3-70b-versatile"},
              {"id": "whisper-large-v3"}, {"id": "llama-guard-4"}]

    def log_message(self, *_):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _good_key(self):
        return self.headers.get("authorization") == "Bearer good-key"

    def do_GET(self):
        _Provider.lookups.append(self.path)
        if not self._good_key():
            self._send(401, {"error": {"message": "Invalid API Key"}})
        elif self.path.endswith("/key"):
            self._send(200, {"data": {"is_free_tier": True}})
        else:
            self._send(200, {"data": _Provider.models})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Provider.chats.append({"body": body, "auth": self.headers.get("authorization")})
        if not self._good_key():
            self._send(401, {"error": {"message": "Invalid API Key"}})
            return
        if _Provider.mode == "busy":
            self._send(429, {"error": {"message": "Rate limit reached"}})
            return
        if _Provider.mode == "pay":
            self._send(402, {"error": {"message": "Insufficient credits"}})
            return
        prompt = body["messages"][-1]["content"]
        if "Check it." in prompt:
            passage = prompt.split("story:\n\n", 1)[-1].split("\n\nCheck it.", 1)[0]
            words = passage.split()
            text = "Here is my check:\n```json\n" + json.dumps({
                "marks": {"clarity": 6, "grammar": 14, "sense": 4, "interest": "5"},
                "problems": [{"quote": " ".join(words[2:6]), "problem": "It is not clear who is running.", "fix": "Name her."},
                             {"quote": "the dragon roared", "problem": "Nothing leads up to this.", "fix": "Cut it."}],
                "corrected": "Rin ran. " + " ".join(words[3:])}) + "\n```"
        else:
            text = SCENE
        if _Provider.mode == "think":
            text = "<think>\nThe user wants a scene. Let me plan.\n</think>\n\n" + text
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}}],
                         "usage": {"prompt_tokens": 50, "completion_tokens": 40}})


@pytest.fixture
def web(monkeypatch, tmp_path):
    """The stand-in provider, with Groq, OpenRouter and Gemini all pointed at it, and Gauge allowed online."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever).start()      # not a daemon: the window tests wait on daemon threads
    _Provider.chats, _Provider.lookups, _Provider.mode = [], [], "ok"
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    for pid in ("groq", "openrouter", "gemini", "mistral"):
        monkeypatch.setitem(helpers.PROVIDERS, pid, {**helpers.PROVIDERS[pid], "base_url": url})
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy", "GAUGE_OFFLINE",
                 "GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("GAUGE_HELPER_KEYS_FILE", str(tmp_path / "home" / "helper-keys.json"))
    monkeypatch.setattr(helpers, "PACING", 0.0)
    helpers.HelperLLM._last.clear()
    try:
        yield _Provider, url
    finally:
        server.shutdown()
        server.server_close()


def _on(cfg, pid="groq"):
    assert helpers.connect(cfg, pid, "good-key")["ok"]
    assert helpers.allow(cfg, pid, True)["ok"]
    return cfg


def _rows(cfg, which):
    f = cfg.models_dir / "datasets" / school.FILES[which]
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


class _Local:
    def __init__(self, reply=SCENE):
        self.reply, self.asked = reply, []

    def available(self):
        return True

    def generate(self, prompt, **kw):
        self.asked.append(kw.get("purpose"))
        return self.reply(prompt) if callable(self.reply) else self.reply


# ------------------------------------------------------------------ the catalogue
def test_the_catalogue_says_what_each_free_plan_is_and_what_it_keeps():
    assert {"groq", "openrouter", "gemini", "mistral"} <= set(helpers.PROVIDERS)
    for pid, info in helpers.PROVIDERS.items():
        assert info["base_url"].startswith("https://") and info["keys"].startswith("https://") and info["terms"].startswith("https://")
        assert info["free"] and info["keeps"] and 0 < info["daily"] <= 300 and info["pace"] >= 1, pid
    assert helpers.PROVIDERS["openrouter"]["free_only"] and helpers.PROVIDERS["openrouter"]["daily"] < 50
    assert helpers.PROVIDERS["gemini"]["teach"] is False and "compete" in helpers.PROVIDERS["gemini"]["keeps"]
    assert "may train" in helpers.PROVIDERS["mistral"]["keeps"]
    assert "free but not private" in HELPERS_NOTE and "sent to" in HELPERS_NOTE


def test_only_free_conversation_models_are_offered_best_guess_first():
    listing = {"data": [{"id": "openai/gpt-x", "pricing": {"prompt": "0.000002", "completion": "0.00001"}},
                        {"id": "google/gemma-4-9b-it:free", "pricing": {"prompt": "0", "completion": "0"}},
                        {"id": "meta/llama-4-8b:free"}, {"id": "meta/llama-4-70b:free"}, {"id": "some/embed-model:free"},
                        {"id": "quiet/free-in-price-only", "pricing": {"prompt": "0", "completion": "0"}}]}
    got = helpers.chat_models("openrouter", listing)
    assert got[:2] == ["meta/llama-4-70b:free", "meta/llama-4-8b:free"]          # the preferred family, the bigger one first
    assert "openai/gpt-x" not in got and "some/embed-model:free" not in got        # never one that costs; never a non-chat one
    assert "quiet/free-in-price-only" in got
    gem = {"data": [{"id": "models/embedding-001"}, {"id": "models/gemini-3-flash-preview"}, {"id": "models/gemini-2.5-flash"},
                    {"id": "models/gemini-2.5-pro"}, {"id": "models/gemini-2.5-flash-preview-tts"}, {"id": "models/imagen-4"}]}
    assert helpers.chat_models("gemini", gem) == ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-pro"]
    assert helpers.chat_models("groq", ["whisper-large-v3", "llama-3.3-70b-versatile", "qwen-3-32b"])[0] == "llama-3.3-70b-versatile"


def test_a_key_is_only_sent_over_an_encrypted_connection_or_at_home():
    assert helpers.safe_address("https://api.example.com/v1/") == "https://api.example.com/v1"
    assert helpers.safe_address("http://localhost:1234/v1") and helpers.safe_address("http://192.168.1.20:8000/v1")
    with pytest.raises(ValueError, match="not encrypted"):
        helpers.safe_address("http://api.example.com/v1")
    with pytest.raises(ValueError):
        helpers.safe_address("api.example.com")


# ------------------------------------------------------------------ connecting
def test_connecting_tries_the_key_for_free_and_sends_nothing_else(cfg, web):
    api, _ = web
    res = helpers.connect(cfg, "groq", "wrong-key")
    assert not res["ok"] and "did not accept the key" in res["message"] and helpers.key_for("groq") == ""
    res = helpers.connect(cfg, "groq", "  good-key\n")
    assert res["ok"] and res["model"] == "llama-3.3-70b-versatile" and res["models"] == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    assert "Nothing is sent to it until you switch it on" in res["message"]
    assert api.chats == []                                # asking for the model list is all that happened
    saved = cfg.config_path.read_text()
    assert "good-key" not in saved and helpers.keys_file().exists() and cfg.root not in helpers.keys_file().parents
    row = next(r for r in helpers.status(cfg) if r["id"] == "groq")
    assert row["connected"] and not row["on"] and not row["teach"] and "switched off" in row["line"]
    assert school.helper_teachers(cfg) == [] and helpers.working(cfg, "check") == []   # connected is not switched on
    assert not helpers.connect(cfg, "openrouter", "wrong-key")["ok"]                   # its model list is public: the key is tried apart
    assert any(p.endswith("/key") for p in api.lookups)
    assert "No key was given" in helpers.connect(cfg, "mistral", "")["message"]
    assert "not a helper Gauge knows" in helpers.connect(cfg, "nosuch", "good-key")["message"]


def test_the_question_must_be_answered_yes_before_anything_is_sent(cfg, web):
    helpers.connect(cfg, "groq", "good-key")
    ask = helpers.ask_text(cfg, "groq")
    for must in ("What is sent", "What it keeps", "terms are yours to read", "train another AI model", "free only while",
                 "300 requests a day", "llama-3.3-70b-versatile"):
        assert must in ask
    assert "left switched off" in helpers.allow(cfg, "groq", False)["message"] and helpers.working(cfg, "teach") == []
    assert "teaches GaugeLM and checks its work" in helpers.allow(cfg, "groq", True)["message"]
    assert [pid for pid, _ in helpers.working(cfg, "teach")] == ["groq"]
    # Gemini's terms bar developing competing models, so it starts as a checker only.
    _on(cfg, "gemini")
    assert [pid for pid, _ in helpers.working(cfg, "teach")] == ["groq"]
    assert [pid for pid, _ in helpers.working(cfg, "check")] == ["groq", "gemini"]
    assert "it checks its work" in helpers.allow(cfg, "gemini", True)["message"]


def test_a_service_can_be_added_by_its_address_and_removed(cfg, web):
    _, url = web
    assert "not encrypted" in helpers.connect(cfg, "myhost", "good-key", base_url="http://example.com/v1")["message"]
    res = helpers.connect(cfg, "myhost", "good-key", base_url=url, model="llama-3.1-8b-instant")
    assert res["ok"] and res["model"] == "llama-3.1-8b-instant" and "myhost" in helpers.known(cfg)
    assert "knows nothing about what it keeps" in helpers.ask_text(cfg, "myhost")
    out = helpers.disconnect(cfg, "myhost")
    assert "disconnected" in out["message"] and "myhost" not in helpers.known(cfg) and helpers.key_for("myhost", cfg) == ""


# ------------------------------------------------------------------ asking a helper
def test_a_request_is_plain_about_what_it_is_for_and_only_the_answer_is_kept(cfg, web):
    api, _ = web
    _on(cfg)
    api.mode = "think"
    llm = helpers.HelperLLM(cfg, "groq", "llama-3.3-70b-versatile")
    out = llm.generate("Write a short scene.", temperature=0.9, max_tokens=120, purpose="school:lesson")
    assert out == SCENE.strip()                           # the model's thinking out loud is dropped
    sent = api.chats[-1]
    assert sent["auth"] == "Bearer good-key" and sent["body"]["model"] == "llama-3.3-70b-versatile"
    system = sent["body"]["messages"][0]
    assert system["role"] == "system" and "GaugeLM" in system["content"] and "saved as a worked example" in system["content"]
    assert sent["body"]["max_tokens"] >= 512 and "response_format" not in sent["body"]
    assert helpers.Usage(cfg.root).today("groq") == 1


def test_it_stays_inside_the_days_allowance_and_rests_when_told_to(cfg, web):
    api, _ = web
    _on(cfg)
    helpers.set_role(cfg, "groq", daily=2)
    llm = helpers.HelperLLM(cfg, "groq", "llama-3.3-70b-versatile")
    llm.generate("one")
    llm.generate("two")
    with pytest.raises(helpers.HelperBusy, match="2 requests Gauge allows it in a day"):
        llm.generate("three")
    assert len(api.chats) == 2 and not llm.available()    # the third was never sent
    assert school.helper_teachers(cfg) == []              # and it is not offered as a teacher again today
    helpers.set_role(cfg, "groq", daily=50)
    api.mode = "busy"
    llm = helpers.HelperLLM(cfg, "groq", "llama-3.3-70b-versatile")
    llm.cfg.retry_backoff_seconds = 0.0
    with pytest.raises(helpers.HelperBusy, match="too many requests"):
        llm.generate("four")
    sent = len(api.chats)
    api.mode = "ok"
    with pytest.raises(helpers.HelperBusy, match="asked for a pause"):
        llm.generate("five")                              # still resting: left alone, nothing sent
    assert len(api.chats) == sent and "resting" in next(r for r in helpers.status(cfg) if r["id"] == "groq")["line"]


def test_a_request_for_payment_stops_it_and_nothing_is_paid(cfg, web, monkeypatch):
    api, _ = web
    assert "no model Gauge can use for free" in helpers.connect(cfg, "openrouter", "good-key")["message"]   # none of these is free
    monkeypatch.setattr(api, "models", [{"id": "meta/llama-4-70b:free"}, {"id": "openai/gpt-x", "pricing": {"prompt": "0.01"}}])
    _on(cfg, "openrouter")
    assert helpers.entry(cfg, "openrouter")["model"] == "meta/llama-4-70b:free"
    api.mode = "pay"
    llm = helpers.HelperLLM(cfg, "openrouter", "meta/llama-4-70b:free")
    with pytest.raises(helpers.HelperBusy, match="asked for payment"):
        llm.generate("x")
    assert helpers.Usage(cfg.root).resting("openrouter") > 23 * 3600


def test_dry_runs_and_offline_mode_send_nothing(cfg, web, monkeypatch):
    from gauge import dryrun

    api, _ = web
    _on(cfg)
    llm = helpers.HelperLLM(cfg, "groq", "llama-3.3-70b-versatile")
    monkeypatch.setenv(dryrun.ENV, "1")
    dryrun.reset()
    with pytest.raises(LLMError, match="dry run"):
        llm.generate("x")
    assert dryrun.held()[-1]["action"] == "request to a free AI helper"
    monkeypatch.delenv(dryrun.ENV)
    dryrun.reset()
    monkeypatch.setenv("GAUGE_OFFLINE", "1")
    with pytest.raises(LLMError, match="stay offline"):
        llm.generate("x")
    assert api.chats == [] and helpers.working(cfg, "teach") == [] and not helpers.connect(cfg, "mistral", "good-key")["ok"]


# ------------------------------------------------------------------ teaching
def test_a_helper_teaches_in_turn_and_its_lessons_are_labelled(cfg, web):
    _on(cfg)
    pool = school.helper_teachers(cfg)
    assert [t.name for t in pool] == ["groq:llama-3.3-70b-versatile"] and pool[0].where == "Groq" and not pool[0].open_licence
    res = school.session(cfg, [TEXT], want=4, pool=pool, student=None)
    assert res["rows"] == 4 and res["teacher"] == "groq:llama-3.3-70b-versatile"
    rows = _rows(cfg, "lessons") + _rows(cfg, "answers")
    assert all(r["teacher"].startswith("groq:") and "Groq's terms" in r["licence"] for r in rows)
    assert any("groq:" in t.name for t in school.teachers(cfg))        # and it is in the school's ordinary rotation
    ctl = StudioController(cfg, None)
    assert ctl.helper_forget("groq")["removed"] == 4 and _rows(cfg, "lessons") == []


def test_when_a_helper_has_had_enough_the_next_teacher_carries_on(cfg, web):
    _on(cfg)
    helpers.set_role(cfg, "groq", daily=2)
    local = _Local()
    pool = school.helper_teachers(cfg) + [school.Teacher("qwen2.5:7b", local, "Apache-2.0", True)]
    res = school.session(cfg, [TEXT], want=6, pool=pool, student=None)
    by = [r["teacher"] for r in _rows(cfg, "lessons") + _rows(cfg, "answers")]
    assert sum(t.startswith("groq:") for t in by) == 2 and "qwen2.5:7b" in by and local.asked
    assert "then qwen2.5:7b" in res["message"] and "rests until tomorrow" in res["message"]
    helpers.set_role(cfg, "groq", teach=False)            # switched to checking only: no longer a teacher
    assert school.helper_teachers(cfg) == []


# ------------------------------------------------------------------ checking GaugeLM's work
def test_a_chapter_is_cut_into_passages_from_start_middle_and_end():
    parts = checkwork.pieces(CHAPTER, 3)
    assert len(parts) == 3 and parts[0].startswith("Paragraph 1.") and "Paragraph 12." in parts[-1]
    assert all(100 <= len(p.split()) <= 300 for p in parts)
    assert checkwork.pieces("Too short to check.", 3) == [] and len(checkwork.pieces(CHAPTER, 1)) == 1


def test_a_checkers_reply_is_kept_only_as_far_as_it_can_be_trusted():
    passage = "Rin ran down the wet street with the bag under her arm. She did not look back at the men behind her at all."
    reply = json.dumps({"marks": {"clarity": 14, "grammar": 0, "sense": "6", "interest": "high"},
                        "problems": [{"quote": "ran down the wet", "problem": "Flat.", "fix": "Show it."},
                                     {"quote": "a sentence that is not there", "problem": "Made up."},
                                     {"quote": "bag", "problem": ""}, "not an object"],
                        "corrected": "Rin ran down the wet street, the bag tight under her arm, and never once looked back at the men."})
    got = checkwork.read_reply("Sure! " + reply, passage)
    assert got["marks"] == {"clarity": 10.0, "grammar": 1.0, "sense": 6.0}        # clipped to 1..10; the unreadable one dropped
    assert [p["quote"] for p in got["problems"]] == ["ran down the wet"]          # only a fault that can be found in the text
    assert got["corrected"].startswith("Rin ran down")
    assert checkwork.read_reply(json.dumps({"marks": {"clarity": 5}, "corrected": "Something else entirely."}), passage)["corrected"] == ""
    assert checkwork.read_reply("I think it is fine.", passage) is None
    assert checkwork.read_reply(json.dumps({"problems": []}), passage) is None    # no marks: not a check


def test_a_hosted_helper_checks_a_chapter_and_its_corrections_become_lessons(cfg, web):
    _on(cfg)
    res = checkwork.check_text(cfg, CHAPTER, source="story chapter 1", step=120, panel=1)
    assert res["ok"] and res["checked_by"] == ["groq:llama-3.3-70b-versatile"] and res["pieces"] == 3
    assert res["marks"] == {"clarity": 6.0, "grammar": 10.0, "sense": 4.0, "interest": 5.0} and res["overall"] == 6.2
    # The same words were faulted in all three passages: reported once, and found in the text.
    assert [p["quote"] for p in res["problems"]] == ["Rin ran down the"] and res["problems"][0]["problem"].startswith("It is not clear")
    assert "the dragon roared" not in json.dumps(res["problems"])                 # the invented fault is not reported
    assert res["rows"] == 3 and "kept as lessons" in res["message"] and "6.2 out of 10" in res["message"]
    rows = _rows(cfg, "checks")
    assert len(rows) == 3 and rows[0]["task"] == "correct" and rows[0]["response"].startswith("Rin ran.")
    assert rows[0]["prompt"].startswith("Rewrite this passage so that it is clear and correct")
    assert any(kind.startswith("chat:Rewrite this passage") for kind, _ in instruction_rows(cfg.models_dir / "datasets"))
    assert "Checked 1 time" in checkwork.trend(cfg.models_dir / "datasets")["line"]
    assert "It is not clear who is running." in checkwork.report(res)


def test_a_checker_that_may_not_teach_only_marks(cfg, web):
    _on(cfg, "gemini")                                    # checks, but its answers are not kept as lessons
    res = checkwork.check_text(cfg, CHAPTER, panel=1)
    assert res["ok"] and res["rows"] == 0 and _rows(cfg, "checks") == [] and res["problems"]
    res = checkwork.check_text(cfg, CHAPTER)              # a panel of two: Gemini and the model on this computer
    assert res["checked_by"] == ["gemini:llama-3.3-70b-versatile", "fake"]
    assert {r["teacher"] for r in _rows(cfg, "checks")} == {"fake"}               # only the local one's corrections are kept


def test_two_checkers_are_averaged_and_no_checker_is_said_plainly(cfg, monkeypatch):
    def reply(clarity):
        return lambda prompt: json.dumps({"marks": {"clarity": clarity, "grammar": 6, "sense": 6, "interest": 6}, "problems": []})

    a = school.Teacher("alpha", _Local(reply(4)), "Apache-2.0", True)
    b = school.Teacher("beta", _Local(reply(8)), "MIT", True)
    c = school.Teacher("gamma", _Local(reply(1)), "MIT", True)
    res = checkwork.check_text(cfg, CHAPTER, pool=[a, b, c], n_pieces=2)
    assert res["checked_by"] == ["alpha", "beta"] and res["marks"]["clarity"] == 6.0 and not c.llm.asked     # a panel of two
    assert "No fault was pointed at" in checkwork.report(res)
    res = checkwork.check_text(cfg, CHAPTER, pool=[school.Teacher("mute", _Local("no json here"), "MIT", True)])
    assert not res["ok"] and "nothing usable" in res["message"]
    assert "Nothing could check it" in checkwork.check_text(cfg, CHAPTER, pool=[])["message"]
    assert "too little text" in checkwork.check_text(cfg, "Short.", pool=[a])["message"]


def test_what_gaugelm_writes_is_checked_straight_away(cfg, db, web, monkeypatch):
    ctl = StudioController(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds")
    pipe = ctl.pipeline()                                 # the stand-in writes; then GaugeLM is made the writer of record
    _on(cfg)
    cfg.llm.provider = "native"
    monkeypatch.setattr(ctl, "pipeline", lambda: pipe)
    monkeypatch.setattr(ctl, "_require_engine", lambda: None)
    monkeypatch.setattr("gauge.native.school.teachers", lambda cfg, check=True: [])       # no model on this computer
    info = ctl.write_next(name)
    assert "Checked by groq:llama-3.3-70b-versatile" in info["check"] and "out of 10" in info["check"]
    assert (cfg.projects_dir / pipe.open_project(name).dir.name / "checks" / "chapter_0001.json").exists()
    cfg.helpers.check_after_chapter = False
    assert ctl.write_next(name)["check"] == ""
    cfg.llm.provider = "fake"
    res = ctl.check_chapter(name, 1)                      # and any chapter can be checked when asked
    assert res["ok"] and res["report"] and "Chapter 1." in res["message"]
    assert "Open or start a story" in ctl.check_chapter(None)["message"]


# ------------------------------------------------------------------ the command line and the window
def test_helper_commands(cfg, db, web):
    cfg.save()
    run = CliRunner(env={"COLUMNS": "220"})
    args = ["--data-dir", str(cfg.root), "helpers"]
    out = run.invoke(main, [*args, "list"])
    assert out.exit_code == 0 and "Groq: not connected" in out.output and "has not been checked yet" in out.output
    out = run.invoke(main, [*args, "connect", "groq"], input="wrong-key\n")
    assert out.exit_code == 1 and "did not accept the key" in out.output and "wrong-key" not in out.output
    out = run.invoke(main, [*args, "connect", "groq"], input="good-key\nn\n")
    assert "Groq is connected" in out.output and "terms are yours to read" in out.output and "left switched off" in out.output
    assert "good-key" not in out.output
    out = run.invoke(main, [*args, "connect", "groq", "--yes"], input="\n")              # the saved key is used again
    assert "is switched on" in out.output
    out = run.invoke(main, [*args, "models", "groq"])
    assert "llama-3.3-70b-versatile" in out.output and "whisper" not in out.output
    out = run.invoke(main, [*args, "set", "groq", "--model", "llama-3.1-8b-instant", "--daily", "20", "--teach", "off"])
    assert "llama-3.1-8b-instant" in out.output and "20 requests a day" in out.output and "no longer teaches" in out.output
    out = run.invoke(main, [*args, "try", "groq"])
    assert "Groq answered: The lamp went out." in out.output
    out = run.invoke(main, [*args, "list"])
    assert "Groq: checks, with llama-3.1-8b-instant; 1 of 20 requests today" in out.output
    out = run.invoke(main, [*args, "disconnect", "groq", "--forget"])
    assert "disconnected" in out.output and helpers.key_for("groq") == ""


def test_the_window_code_runs_the_helpers_flow(cfg, db, web, tmp_path, monkeypatch):
    """The real window code, driven with the stand-in toolkit: it proves the handlers are wired, not how it looks."""
    from gauge import gui
    from tests.test_teach import _Entry, _fake_tkinter, _wait_for_jobs, _Widget

    mods, shown = _fake_tkinter(tmp_path)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    assert {"Free AI helpers...", "Check this chapter with another AI"} <= set(_Widget.menu)
    _Widget.buttons["Change the AI"]()
    _Widget.buttons["Free helpers..."]()
    assert {"Connect helper", "Disconnect helper", "Use this model", "Teaching on / off", "Checking on / off", "Try it",
            "Check the chapter I am reading", "Remove its lessons"} <= set(_Widget.buttons)
    box = _Widget.listboxes[-1]
    assert any("Groq: not connected" in line for line in box.items)
    key_var = [e for e in _Entry.made if e.kw.get("show") == "*"][-1].kw["textvariable"]
    key_var.set("good-key")
    _Widget.buttons["Connect helper"]()                   # Groq is the first in the list; the stand-in answers yes to its question
    _wait_for_jobs()
    assert key_var.get() == "" and helpers.key_for("groq") == "good-key"
    assert any(kind == "ask" and "terms are yours to read" in args[1] for kind, args in shown)
    assert any("Groq: teaches and checks" in line for line in box.items)
    _Widget.buttons["Try it"]()
    _wait_for_jobs()
    assert helpers.Usage(cfg.root).today("groq") == 1
    _Widget.buttons["Teaching on / off"]()
    assert any("Groq: checks, with" in line for line in box.items)
    _Widget.buttons["Checking on / off"]()
    assert helpers.working(cfg, "check") == []
    _Widget.buttons["Check the chapter I am reading"]()   # no story is open: it says so
    assert any(kind == "info" and "Open or start a story" in args[1] for kind, args in shown)
    _Widget.buttons["Disconnect helper"]()
    assert helpers.key_for("groq") == ""
