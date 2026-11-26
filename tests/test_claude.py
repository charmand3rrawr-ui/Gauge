"""Claude as a paid teacher and writer: the key, the spending limit, refusals, and its place in the school.

Anthropic's library is not installed where the tests run and no key is available, so these drive
Gauge's code against a stand-in for the library. `test_claude_sdk.py` drives the real library against
a local stand-in server whenever the library is installed."""
import json
import sys
import types
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from gauge.cli import main
from gauge.gui import CLAUDE_NOTE, CLAUDE_TEACH_ASK, CLAUDE_WRITE_ASK, StudioController
from gauge.llm import claude, make_llm
from gauge.llm.client import LLMError
from gauge.native import school

TEXT = ("Rin counted the exits before she sat down and rubbed the scar on her palm. The bag was still sealed and the "
        "debt was still hers, and nobody in the room knew she had already run once tonight. ") * 30
SCENE = "The lamp went out. Mara held her breath and listened to the stairs, and counted to nine before she moved. " * 3


# ------------------------------------------------------------------ a stand-in for Anthropic's library
class _Stream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def _message(text=SCENE, stop="end_turn", category="", model="claude-opus-5-5", tokens=(500, 800), iterations=None):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    details = SimpleNamespace(category=category or None, explanation="declined") if stop == "refusal" else None
    usage = SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1], cache_creation_input_tokens=0,
                            cache_read_input_tokens=0, iterations=iterations)
    return SimpleNamespace(content=content, stop_reason=stop, stop_details=details, model=model, usage=usage)


class _Fake:
    """What the stand-in library was asked, and how it answers."""

    def __init__(self):
        self.requests: list[dict] = []
        self.lookups = 0
        self.reply = lambda params: _message(model=params["model"])


def _fake_anthropic(fake: _Fake):
    mod = types.ModuleType("anthropic")

    class APIError(Exception):
        def __init__(self, message="error", status_code=500):
            super().__init__(message)
            self.message, self.status_code = message, status_code

    class APIConnectionError(APIError):
        pass

    class APIStatusError(APIError):
        pass

    for name in ("BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError", "RateLimitError"):
        setattr(mod, name, type(name, (APIStatusError,), {}))
    mod.APIError, mod.APIConnectionError, mod.APIStatusError = APIError, APIConnectionError, APIStatusError

    class _Messages:
        def __init__(self, owner, beta):
            self.owner, self.beta = owner, beta

        def stream(self, **params):
            fake.requests.append({**params, "_beta_namespace": self.beta, "_key": self.owner.api_key})
            if self.owner.api_key != "good-key":
                raise mod.AuthenticationError("invalid x-api-key", 401)
            return _Stream(fake.reply(params))

    class _Models:
        def __init__(self, owner):
            self.owner = owner

        def retrieve(self, model):
            fake.lookups += 1
            if self.owner.api_key != "good-key":
                raise mod.AuthenticationError("invalid x-api-key", 401)
            if not model.startswith("claude-"):
                raise mod.NotFoundError("no such model", 404)
            return SimpleNamespace(id=model)

    class Anthropic:
        def __init__(self, api_key=None, timeout=None, max_retries=None):
            self.api_key = api_key
            self.messages = _Messages(self, False)
            self.beta = SimpleNamespace(messages=_Messages(self, True))
            self.models = _Models(self)

        def with_options(self, **_):
            return self

    mod.Anthropic = Anthropic
    return mod


@pytest.fixture
def anthropic(monkeypatch, tmp_path):
    """The stand-in library, installed, with a working key saved where Studio would save it."""
    fake = _Fake()
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(fake))
    monkeypatch.setenv("GAUGE_CLAUDE_KEY_FILE", str(tmp_path / "home" / "claude-key.txt"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    claude.save_key("good-key")
    return fake


@pytest.fixture
def no_sdk(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "anthropic", None)          # importing it now fails, as on a fresh install
    monkeypatch.setenv("GAUGE_CLAUDE_KEY_FILE", str(tmp_path / "home" / "claude-key.txt"))


def _claude_cfg(cfg):
    cfg.llm.claude, cfg.llm.claude_dir = cfg.claude, str(cfg.root)   # what GaugeConfig.load does
    return cfg


def _teaching(cfg):
    _claude_cfg(cfg)
    cfg.claude.teach = cfg.claude.terms_ok = True
    return cfg


def _rows(cfg, which):
    f = cfg.models_dir / "datasets" / school.FILES[which]
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


class _Local:
    """A free local teacher."""

    def __init__(self):
        self.asked = []

    def available(self):
        return True

    def generate(self, prompt, **kw):
        self.asked.append(kw.get("purpose"))
        return "Something happened on the night train, and it was not over yet. " * 4


# ------------------------------------------------------------------ the call itself
def test_a_request_is_shaped_the_way_the_current_models_require(cfg, anthropic):
    cfg = _claude_cfg(cfg)
    cfg.llm.provider, cfg.llm.model = "claude", "claude-opus-5-5"
    llm = make_llm(cfg.llm)
    out = llm.generate("Write a short scene.", system="Plain words.", temperature=0.9, max_tokens=200, purpose="draft")
    assert out == SCENE
    req = anthropic.requests[-1]
    assert req["model"] == "claude-opus-5-5" and req["output_config"] == {"effort": "medium"}
    assert "temperature" not in req and "top_p" not in req and "thinking" not in req     # rejected by these models
    assert req["max_tokens"] >= 16000                    # room for its thinking, whatever the caller's small figure
    assert req["messages"] == [{"role": "user", "content": "Write a short scene."}]      # no prefilled answer
    assert req["system"] == "Plain words."
    # A declined request is re-run by Anthropic on the model it recommends: asked for by default.
    assert req["_beta_namespace"] and req["fallbacks"] == "default" and req["betas"] == [claude.FALLBACK_BETA]
    assert llm.calls.records[-1].ok and llm.calls.records[-1].output_tokens == 800


def test_haiku_has_no_server_side_fallback_so_none_is_asked_for(cfg, anthropic):
    llm = claude.ClaudeLLM(_claude_cfg(cfg).llm, "claude-haiku-5-5")
    llm.generate("x")
    req = anthropic.requests[-1]
    assert not req["_beta_namespace"] and "fallbacks" not in req and "betas" not in req
    cfg.claude.fallbacks = False                         # and it can be switched off for the others
    claude.ClaudeLLM(cfg.llm, "claude-opus-5-5").generate("x")
    assert "fallbacks" not in anthropic.requests[-1]


def test_json_is_asked_for_in_words_not_by_prefilling_the_answer(cfg, anthropic):
    anthropic.reply = lambda p: _message('{"ok": true}')
    llm = claude.ClaudeLLM(_claude_cfg(cfg).llm, "claude-opus-5-5")
    assert llm.generate("Score it.", json_mode=True, schema={"type": "object"}) == '{"ok": true}'
    req = anthropic.requests[-1]
    assert "one JSON object" in req["system"] and '"type": "object"' in req["system"]
    assert len(req["messages"]) == 1 and req["messages"][0]["role"] == "user"


def test_a_refusal_is_reported_and_never_read_as_an_answer(cfg, anthropic):
    anthropic.reply = lambda p: _message("partial text that must be thrown away", stop="refusal", category="frontier_llm")
    llm = claude.ClaudeLLM(_claude_cfg(cfg).llm, "claude-opus-5-5")
    with pytest.raises(claude.ClaudeRefused) as e:
        llm.generate("anything", purpose="school:tutor")
    assert e.value.category == "frontier_llm" and "declined" in str(e.value)
    assert len(anthropic.requests) == 1                  # asked once: no rewording, no second try
    today = claude.Ledger(cfg.root).today()
    assert today["refused"] == 1 and today["last_refusal"] == "frontier_llm"
    assert not llm.calls.records[-1].ok
    anthropic.reply = lambda p: _message("", stop="refusal")     # declined before any output, with no reason given
    with pytest.raises(claude.ClaudeRefused) as e:
        llm.generate("anything")
    assert e.value.category == ""


def test_errors_become_plain_sentences(cfg, anthropic):
    claude.save_key("wrong-key")
    llm = claude.ClaudeLLM(_claude_cfg(cfg).llm, "claude-opus-5-5")
    with pytest.raises(LLMError, match="did not accept the key"):
        llm.generate("x")
    assert llm.available() is False and "did not accept the key" in llm.check()["message"]
    claude.save_key("good-key")
    assert claude.ClaudeLLM(cfg.llm, "claude-opus-5-5").check()["ok"]
    mod = sys.modules["anthropic"]

    def limited(_params):
        raise mod.RateLimitError("slow down", 429)

    anthropic.reply = limited
    with pytest.raises(LLMError, match="limiting how fast"):
        claude.ClaudeLLM(cfg.llm, "claude-opus-5-5").generate("x")


def test_without_the_library_it_says_how_to_connect(cfg, no_sdk):
    assert claude.installed() is False
    with pytest.raises(LLMError, match="not set up on this computer"):
        claude.ClaudeLLM(_claude_cfg(cfg).llm, "claude-opus-5-5").generate("x")
    st = claude.status(cfg)
    assert st["installed"] is False and "not connected" in st["line"]
    cfg.claude.teach = cfg.claude.terms_ok = False
    assert school.claude_teacher(_teaching(cfg)) is None

    def failing(cmd, **_):
        assert cmd[1:4] == ["-m", "pip", "install"] and any(part.startswith("anthropic") for part in cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="ERROR: no internet")

    res = claude.setup(run=failing)
    assert res["ok"] is False and "no internet" in res["message"]


# ------------------------------------------------------------------ money
def test_every_call_is_costed_and_the_daily_limit_stops_the_next_one(cfg, anthropic):
    cfg = _claude_cfg(cfg)
    cfg.claude.daily_usd = 0.03
    llm = claude.ClaudeLLM(cfg.llm, "claude-opus-5-5")
    llm.generate("one", purpose="school:lesson")          # 500 in, 800 out at $4 / $20 a million = 1.8 cents
    today = claude.Ledger(cfg.root).today()
    assert today["calls"] == 1 and today["usd"] == pytest.approx(0.018)
    assert today["by"] == {"teaching": pytest.approx(0.018)}
    llm.generate("two", purpose="draft")                  # 3.6 cents: now over the 3-cent limit
    with pytest.raises(claude.ClaudeLimit, match="today's spending limit"):
        llm.generate("three")
    assert len(anthropic.requests) == 2                   # the third was never sent
    cfg.claude.daily_usd = 0                              # 0 = no limit
    llm.generate("four")
    assert claude.Ledger(cfg.root).total()["calls"] == 3


def test_a_dry_run_sends_nothing_to_claude(cfg, anthropic, monkeypatch):
    from gauge import dryrun

    monkeypatch.setenv(dryrun.ENV, "1")
    dryrun.reset()
    with pytest.raises(LLMError, match="dry run"):
        claude.ClaudeLLM(_claude_cfg(cfg).llm, "claude-opus-5-5").generate("x", purpose="draft")
    assert anthropic.requests == [] and dryrun.held()[-1]["action"] == "paid Claude request"
    dryrun.reset()


def test_prices_err_on_the_careful_side(cfg):
    assert claude.price("claude-opus-5-5") == (4.0, 20.0) and claude.price("claude-sonnet-5-5") == (2.0, 10.0)
    assert claude.price("claude-haiku-5-5") == (0.10, 0.50) and claude.price("claude-haiku-5-5", 150_000) == (0.50, 2.50)
    assert claude.price("claude-something-new") == (8.0, 40.0)        # unknown: priced as the dearest known
    # A declined request re-run on another model: every attempt is counted, at the dearer model's price.
    attempts = [SimpleNamespace(type="message", input_tokens=500, output_tokens=10),
                SimpleNamespace(type="fallback_message", input_tokens=500, output_tokens=800)]
    tin, tout, usd = claude.message_cost(_message(model="claude-opus-5", iterations=attempts), "claude-fable-5-2")
    assert (tin, tout) == (1000, 810) and usd == pytest.approx((1000 * 8 + 810 * 40) / 1e6)
    assert claude.money(2) == "$2.00" and claude.money(0.018) == "1.8 cents"


# ------------------------------------------------------------------ the key
def test_the_key_is_kept_out_of_gauges_data_and_never_in_the_settings(cfg, anthropic, tmp_path):
    f = claude.key_file()
    assert f.read_text() == "good-key" and cfg.root not in f.parents
    cfg = _claude_cfg(cfg)
    cfg.save()
    saved = cfg.config_path.read_text()
    assert "good-key" not in saved and "api_key" not in json.loads(saved)["claude"]
    assert "claude" not in json.loads(saved)["llm"]       # the settings are stored once, not copied under the writer
    assert claude.key_source().startswith("the key saved in")
    assert claude.forget_key() and claude.saved_key() == "" and claude.key_source() == ""


def test_connecting_keeps_a_working_key_and_drops_one_that_does_not_work(cfg, db, anthropic):
    ctl = StudioController(_claude_cfg(cfg), db)
    claude.forget_key()
    res = ctl.claude_connect("")
    assert res["ok"] is False and "No key was given" in res["message"]
    res = ctl.claude_connect("  wrong-key \n")
    assert res["ok"] is False and claude.saved_key() == ""           # a refused key is not kept
    res = ctl.claude_connect(" good-key\n")
    assert res["ok"] and claude.saved_key() == "good-key" and "Nothing is sent to Claude until" in res["message"]
    res = ctl.claude_connect("another-wrong-key")
    assert res["ok"] is False and claude.saved_key() == "good-key"   # the working key is put back
    assert anthropic.requests == []                                   # connecting never spends anything
    out = ctl.claude_disconnect()
    assert claude.saved_key() == "" and "disconnected" in out["message"] and cfg.claude.teach is False


# ------------------------------------------------------------------ the school
def test_claude_teaches_only_after_the_author_switches_it_on(cfg, db, anthropic):
    ctl = StudioController(_claude_cfg(cfg), db)
    assert school.claude_teacher(cfg) is None and all(not t.lead for t in school.teachers(cfg))
    assert ctl.claude_teach(True)["message"] == "Claude's teaching was left off."        # no yes to the question: off
    assert cfg.claude.teach is False
    cfg.claude.teach = True                                # the switch alone, without that yes, is not enough either
    assert school.claude_teacher(cfg) is None
    res = ctl.claude_teach(True, terms_ok=True)
    assert cfg.claude.teach and cfg.claude.terms_ok and "leads GaugeLM's school" in res["message"]
    pool = school.teachers(cfg)
    assert pool[0].lead and pool[0].name == "claude-opus-5-5" and pool[0].where == "Anthropic's computers"
    assert pool[0].licence == claude.LICENCE and not pool[0].open_licence
    assert "Taught by Claude" in ctl.school_line() and "$2.00 a day" in ctl.school_line()
    assert ctl.claude_teach(False)["teaching"] is False and school.claude_teacher(cfg) is None


def test_a_lesson_from_claude_is_labelled_and_claude_is_told_what_it_is_for(cfg, db, anthropic):
    cfg = _teaching(cfg)
    res = school.session(cfg, [TEXT], want=4, pool=school.teachers(cfg), student=None)
    assert res["rows"] == 4 and res["teacher"] == "claude-opus-5-5"
    for which in ("lessons", "answers"):
        rows = _rows(cfg, which)
        assert rows and all(r["teacher"] == "claude-opus-5-5" and r["licence"] == claude.LICENCE for r in rows)
    for req in anthropic.requests:
        assert "GaugeLM" in req["system"] and "trained on" in req["system"]       # said plainly, every time
        assert req["output_config"] == {"effort": "low"}                           # short lessons, kept cheap
    assert claude.Ledger(cfg.root).today()["by"].keys() == {"teaching"}
    assert school.rows_by(cfg.models_dir / "datasets", "claude") == 4


def test_when_the_limit_is_reached_the_free_teachers_carry_on(cfg, db, anthropic):
    cfg = _teaching(cfg)
    cfg.claude.daily_usd = 0.03                            # two lessons' worth
    local = _Local()
    pool = [school.claude_teacher(cfg), school.Teacher("qwen2.5:7b", local, "Apache-2.0", True)]
    res = school.session(cfg, [TEXT], want=6, pool=pool, student=None)
    assert len(anthropic.requests) == 2 and local.asked    # Claude stopped at the limit; the free model took over
    assert "spending limit" in res["message"] and "then qwen2.5:7b" in res["message"]
    by = {r["teacher"] for which in ("lessons", "answers") for r in _rows(cfg, which)}
    assert by == {"claude-opus-5-5", "qwen2.5:7b"}
    assert school.claude_teacher(cfg) is None              # and it is not in the next session's pool today
    res = school.session(cfg, [TEXT], want=2, pool=[school.Teacher("qwen2.5:7b", local, "Apache-2.0", True)], student=None)
    assert res["teacher"] == "qwen2.5:7b"


def test_a_refusal_about_training_models_ends_claudes_teaching_for_the_session(cfg, db, anthropic):
    cfg = _teaching(cfg)
    anthropic.reply = lambda p: _message("", stop="refusal", category="frontier_llm")
    res = school.session(cfg, [TEXT], want=6, pool=[school.claude_teacher(cfg)], student=None)
    assert res["rows"] == 0 and len(anthropic.requests) == 1          # asked once, then left alone
    assert "declined to teach" in res["message"] and "will not reword it" in res["message"]
    res = school.session(cfg, [TEXT], want=6, pool=school.teachers(cfg), student=None)   # with a free teacher there too
    assert len(anthropic.requests) == 2 and res["rows"] == 5 and "then fake" in res["message"]
    assert all(r["teacher"] == "fake" for which in ("lessons", "answers") for r in _rows(cfg, which))
    # Other refusals (a dark passage, say) cost a row each; three end its turn.
    anthropic.requests.clear()
    anthropic.reply = lambda p: _message("", stop="refusal", category="cyber")
    res = school.session(cfg, [TEXT], want=8, pool=[school.claude_teacher(cfg)], student=None)
    assert len(anthropic.requests) == 3 and "declined three requests" in res["message"]


def test_claudes_lessons_can_all_be_removed(cfg, db, anthropic):
    cfg = _teaching(cfg)
    ctl = StudioController(cfg, db)
    local = school.Teacher("qwen2.5:7b", _Local(), "Apache-2.0", True)
    school.session(cfg, [TEXT], want=4, pool=[local], student=None)
    got = ctl.claude_lesson(4)
    assert got["rows"] == 4 and "That cost about" in got["message"] and got["lessons"] == 4
    res = ctl.claude_forget()
    assert res["removed"] == 4 and res["lessons"] == 0
    left = [r["teacher"] for which in ("lessons", "answers") for r in _rows(cfg, which)]
    assert left and set(left) == {"qwen2.5:7b"}            # the free teacher's rows are untouched
    assert "no lessons from Claude" in ctl.claude_forget()["message"]
    cfg.claude.teach = False
    assert "switched off" in ctl.claude_lesson(4)["message"]


def test_training_from_the_studio_is_led_by_claude(cfg, db, anthropic):
    ctl = StudioController(_teaching(cfg), db)
    res = ctl._train_session([("chapter", TEXT)], minutes=0.01, progress=None, teach=True)
    assert res["lessons"] > 0 and "claude-opus-5-5 taught it" in res["message"]
    assert anthropic.requests and all(r["_key"] == "good-key" for r in anthropic.requests)


# ------------------------------------------------------------------ Claude as the writer
def test_writing_with_claude_and_going_back(cfg, db, anthropic):
    cfg = _claude_cfg(cfg)
    cfg.llm.provider, cfg.llm.model, cfg.llm.num_ctx = "ollama", "qwen2.5:14b", 8192
    ctl = StudioController(cfg, db)
    res = ctl.use_claude(True)
    assert cfg.llm.provider == "claude" and cfg.llm.model == "claude-opus-5-5" and cfg.llm.num_ctx == 32768
    assert res["is_claude"] and "paid" in res["label"] and "Every request is paid for" in res["message"]
    assert "stronger than any model this computer can run" in ctl.stronger_model()["message"]
    ctl.claude_set(model="claude-sonnet-5-5   (Claude Sonnet 5.5: half the price)")       # as the window's list gives it
    assert cfg.claude.model == cfg.llm.model == "claude-sonnet-5-5"
    ctl.use_gaugelm(True)                                  # Claude -> GaugeLM -> back: the downloaded model is remembered
    assert cfg.llm.provider == "native"
    ctl.use_claude(False)
    assert (cfg.llm.provider, cfg.llm.model, cfg.llm.num_ctx) == ("ollama", "qwen2.5:14b", 8192)
    ctl.use_claude(True)
    ctl.claude_disconnect()                                # disconnecting also stops it writing
    assert cfg.llm.provider == "ollama"


def test_claude_is_not_slowed_by_the_capacity_limit(cfg, anthropic, monkeypatch):
    monkeypatch.setenv("GAUGE_CAPACITY_PERCENT", "20")     # the limit is about this computer; Claude runs elsewhere
    cfg = _claude_cfg(cfg)
    cfg.llm.provider, cfg.llm.model = "claude", "claude-opus-5-5"
    assert isinstance(make_llm(cfg.llm), claude.ClaudeLLM)


def test_settings_are_checked(cfg, db, anthropic):
    ctl = StudioController(_claude_cfg(cfg), db)
    assert "not a Claude model" in ctl.claude_set(model="gpt-x")["message"]
    assert "not an amount" in ctl.claude_set(daily_usd="lots")["message"]
    assert ctl.claude_set(daily_usd="$5")["limit"] == 5.0
    assert "No daily limit" in ctl.claude_set(daily_usd="0")["message"]
    assert "Effort is one of" in ctl.claude_set(effort="extreme")["message"]
    assert ctl.claude_set(effort="high", teach_effort="medium")["changed"] and cfg.claude.effort == "high"


def test_the_questions_put_to_the_author_say_what_matters():
    for text in (CLAUDE_NOTE, CLAUDE_TEACH_ASK, CLAUDE_WRITE_ASK):
        assert "Anthropic" in text
    assert "costs money" in CLAUDE_TEACH_ASK and "sent to Anthropic" in CLAUDE_TEACH_ASK
    assert "training competing AI models" in CLAUDE_TEACH_ASK and "for you to judge" in CLAUDE_TEACH_ASK
    assert "does not try to get round" in CLAUDE_TEACH_ASK
    assert "never shows it again" in CLAUDE_NOTE and "paid for" in CLAUDE_WRITE_ASK


# ------------------------------------------------------------------ the command line and the window
def test_claude_commands(cfg, db, anthropic, monkeypatch):
    cfg.save()
    run = CliRunner(env={"COLUMNS": "200"})
    args = ["--data-dir", str(cfg.root), "claude"]
    out = run.invoke(main, [*args, "status", "--check"])
    assert out.exit_code == 0 and "Claude is connected" in out.output and "claude-opus-5-5 *" in out.output
    out = run.invoke(main, [*args, "teach", "on"], input="n\n")
    assert "for you to judge" in out.output and "left off" in out.output
    out = run.invoke(main, [*args, "teach", "on"], input="y\n")
    assert "leads GaugeLM's school" in out.output
    out = run.invoke(main, [*args, "set", "--limit", "1", "--model", "claude-haiku-5-5"])
    assert "Claude Haiku 5.5" in out.output and "$1.00" in out.output
    out = run.invoke(main, [*args, "lesson", "--rows", "2"])
    assert "claude-haiku-5-5 taught it" in out.output and "That cost about" in out.output
    out = run.invoke(main, [*args, "forget", "--yes"])
    assert "Removed 2 lessons" in out.output
    out = run.invoke(main, [*args, "connect"], input="wrong-key\n")
    assert out.exit_code == 1 and "did not accept the key" in out.output and "wrong-key" not in out.output
    assert claude.saved_key() == "good-key"
    out = run.invoke(main, [*args, "disconnect"])
    assert "disconnected" in out.output and claude.saved_key() == ""


def test_the_window_code_runs_the_claude_flow(cfg, db, anthropic, tmp_path, monkeypatch):
    """The real window code, driven with the stand-in toolkit: it proves the handlers are wired, not how it looks."""
    from gauge import gui
    from tests.test_teach import _Entry, _fake_tkinter, _wait_for_jobs, _Widget

    mods, shown = _fake_tkinter(tmp_path)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    _claude_cfg(cfg)
    claude.forget_key()
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    _Widget.buttons["Change the AI"]()
    _Widget.buttons["Claude..."]()
    assert {"Connect", "Disconnect", "Save", "Let Claude teach GaugeLM", "Stop Claude teaching", "Have a lesson now",
            "Write with Claude", "Write with the downloaded model", "Remove Claude's lessons"} <= set(_Widget.buttons)
    key_box = next(e for e in _Entry.made if e.kw.get("show") == "*")            # the key is typed into a masked box
    key_var = key_box.kw["textvariable"]
    key_var.set("good-key")
    _Widget.buttons["Connect"]()
    _wait_for_jobs()
    assert key_var.get() == "" and claude.saved_key() == "good-key"               # and is not left on the screen
    _Widget.buttons["Let Claude teach GaugeLM"]()                                 # the stand-in answers yes
    _wait_for_jobs()
    assert cfg.claude.teach and cfg.claude.terms_ok
    assert any(kind == "ask" and "for you to judge" in args[1] for kind, args in shown)
    _Widget.buttons["Have a lesson now"]()
    _wait_for_jobs()
    assert school.rows_by(cfg.models_dir / "datasets", "claude") > 0
    _Widget.buttons["Write with Claude"]()
    _wait_for_jobs()
    assert cfg.llm.provider == "claude"
    _Widget.buttons["Write with the downloaded model"]()
    assert cfg.llm.provider != "claude"
    _Widget.buttons["Stop Claude teaching"]()
    assert cfg.claude.teach is False
    _Widget.buttons["Remove Claude's lessons"]()
    assert school.rows_by(cfg.models_dir / "datasets", "claude") == 0
    _Widget.buttons["Disconnect"]()
    assert claude.saved_key() == ""
