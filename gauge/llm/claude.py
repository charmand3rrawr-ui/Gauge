"""Claude, Anthropic's model, as a writer and a teacher for Gauge (``llm.provider = "claude"``).

Everything else in Gauge runs on this computer and costs nothing. Claude is different in three ways,
and this module is built around them:

* **It costs money.** Every call is paid for from your Anthropic account. Each call's tokens are
  written to a ledger (``claude_usage.json``) with an estimated cost, and Gauge stops asking once
  today's estimate reaches ``claude.daily_usd``. The estimate uses the list prices below; your invoice
  is the truth.
* **Your text leaves this computer.** Whatever is in a prompt (excerpts of your library, your chapters)
  is sent to Anthropic.
* **It needs a key.** The key is never put in ``config.json`` and never asked for in a chat. It is read
  from the file Studio saves in your user folder, or from wherever Anthropic's own library finds one
  (the ``ANTHROPIC_API_KEY`` environment variable, or an ``ant auth login`` profile).

The calls go through Anthropic's official Python library (``pip install anthropic``), which Gauge
installs only when you connect Claude.

Claude can decline a request. That arrives as an ordinary answer marked ``refusal``; Gauge raises
``ClaudeRefused`` and does not rephrase the request to get round it. With ``claude.fallbacks`` on
(the default) Anthropic first re-runs a declined request on the model it recommends for that case.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..config import ClaudeConfig, LLMConfig
from .client import CallLog, CallRecord, LLMError
from .tokens import estimate_tokens

log = logging.getLogger(__name__)

LEDGER = "claude_usage.json"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
TERMS_URL = "https://www.anthropic.com/legal"
KEYS_URL = "https://console.anthropic.com/settings/keys"
LICENCE = "Anthropic's terms (own terms)"

# List prices in US dollars per million tokens (input, output), and what each model is for.
MODELS: dict[str, dict[str, Any]] = {
    "claude-opus-5-5": {"name": "Claude Opus 5.5", "price": (4.0, 20.0), "says": "the usual choice: very strong, mid-priced"},
    "claude-sonnet-5-5": {"name": "Claude Sonnet 5.5", "price": (2.0, 10.0), "says": "half the price of Opus, a little less strong"},
    "claude-haiku-5-5": {"name": "Claude Haiku 5.5", "price": (0.10, 0.50), "says": "by far the cheapest; fine for short lessons"},
    "claude-fable-5-2": {"name": "Claude Fable 5.2", "price": (8.0, 40.0), "says": "the strongest, at twice the price of Opus"},
}
DEFAULT_MODEL = "claude-opus-5-5"
NO_SERVER_FALLBACK = ("claude-haiku",)       # models for which Anthropic offers no server-side fallback

TEACHER_SYSTEM = (
    "You are helping to teach GaugeLM, a very small language model that its owner trains on their own computer to "
    "write serial fiction. Where a request speaks of 'a student writer', the student is GaugeLM. Your answers are saved "
    "as worked examples that GaugeLM is trained on, so write exactly the text that is asked for and nothing else: no "
    "preamble, no notes, no closing remark. The student is small, so prefer plain words and clear sentences.")


class ClaudeRefused(LLMError):
    """Claude declined the request. `category` is Anthropic's reason when it gives one."""

    def __init__(self, message: str, category: str = ""):
        super().__init__(message)
        self.category = category


class ClaudeLimit(LLMError):
    """Today's spending limit has been reached, so Claude is not asked again until tomorrow."""


# ---------------------------------------------------------------------------------- the library
def sdk() -> Any:
    """Anthropic's Python library, or a sentence saying how to get it."""
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("Claude is not set up on this computer yet. In Gauge Studio press Change the AI, then Claude, "
                       "then Connect; or run: gauge claude connect") from e
    return anthropic


def installed() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def setup_command() -> list[str]:
    return [sys.executable, "-m", "pip", "install", "--upgrade", "anthropic>=1.0"]


def setup(run: Any = None, timeout: float = 900.0) -> dict[str, Any]:
    """Install Anthropic's library into this Python. A small download."""
    if installed():
        return {"ok": True, "installed": False, "message": "Anthropic's library is already installed."}
    from .. import dryrun

    cmd = setup_command()
    if dryrun.is_dry_run():
        dryrun.hold("install Anthropic's library", " ".join(cmd))
        return {"ok": False, "installed": False, "message": "Dry run: would run " + " ".join(cmd)}
    try:
        res = (run or subprocess.run)(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "installed": False, "message": f"Could not install Anthropic's library: {e}"}
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-1:] or ["no details"]
        return {"ok": False, "installed": False,
                "message": "Could not install Anthropic's library (" + tail[0][:200] + "). Check the internet connection and try again."}
    import importlib

    importlib.invalidate_caches()
    ok = installed()
    return {"ok": ok, "installed": ok, "message": "Anthropic's library is installed." if ok else
            "The install finished but the library cannot be loaded. Close Gauge, open it again and press Connect once more."}


# ---------------------------------------------------------------------------------- the key
def key_file() -> Path:
    """Where Studio keeps the key: in your user folder, outside Gauge's data, so it is never part of a
    backup, an export or an update. It is a plain text file that only your user account should read."""
    override = os.environ.get("GAUGE_CLAUDE_KEY_FILE")
    return Path(override) if override else Path.home() / ".gauge" / "claude-key.txt"


def saved_key() -> str:
    try:
        return key_file().read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def save_key(key: str) -> Path:
    key = "".join(key.split())
    if not key:
        raise ValueError("no key was given")
    f = key_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(key, encoding="utf-8")
    try:
        f.chmod(0o600)                     # on Windows the user folder is already private to the account
    except OSError:
        pass
    return f


def forget_key() -> bool:
    try:
        key_file().unlink()
        return True
    except OSError:
        return False


def key_source() -> str:
    """Where a key would come from, in words, or '' when Gauge can see none. A profile made with
    Anthropic's own `ant auth login` is not visible from here; `available()` finds out by asking."""
    if saved_key():
        return f"the key saved in {key_file()}"
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if os.environ.get(name):
            return f"the {name} environment variable"
    return ""


def client(timeout: float = 600.0, retries: int = 2) -> Any:
    """A connection made with the saved key, or else with whatever Anthropic's library finds itself."""
    anthropic = sdk()
    key = saved_key()
    if key:
        return anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=retries)
    return anthropic.Anthropic(timeout=timeout, max_retries=retries)


# ---------------------------------------------------------------------------------- what it costs
def price(model: str, prompt_tokens: int = 0) -> tuple[float, float]:
    """Dollars per million tokens (input, output). A model Gauge does not know is priced as the dearest
    one it does, so the spending limit errs on the careful side."""
    low = model.lower()
    for name, info in MODELS.items():
        if low.startswith(name):
            if name == "claude-haiku-5-5" and prompt_tokens > 100_000:
                return 0.50, 2.50
            return info["price"]
    if low.startswith("claude-mythos"):
        return MODELS["claude-fable-5-2"]["price"]
    return max(info["price"] for info in MODELS.values())


def cost(model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0) -> float:
    pin, pout = price(model, input_tokens + cache_write + cache_read)
    return (input_tokens * pin + cache_write * pin * 1.25 + cache_read * pin * 0.1 + output_tokens * pout) / 1_000_000


def _n(obj: Any, name: str) -> int:
    v = getattr(obj, name, None) if not isinstance(obj, dict) else obj.get(name)
    return int(v or 0)


def message_cost(message: Any, asked_model: str) -> tuple[int, int, float]:
    """(input tokens, output tokens, estimated dollars) for one answer. When a declined request was
    re-run on another model, each attempt is counted at the dearer of the two models' prices."""
    usage = getattr(message, "usage", None)
    served = str(getattr(message, "model", "") or asked_model)
    dear = max((asked_model, served), key=lambda m: price(m)[1])
    attempts = [a for a in (getattr(usage, "iterations", None) or []) if _n(a, "input_tokens") or _n(a, "output_tokens")]
    parts = attempts if len(attempts) > 1 else [usage]
    tin = tout = 0
    usd = 0.0
    for part in parts:
        i, o = _n(part, "input_tokens"), _n(part, "output_tokens")
        cw, cr = _n(part, "cache_creation_input_tokens"), _n(part, "cache_read_input_tokens")
        tin += i + cw + cr
        tout += o
        usd += cost(dear if len(parts) > 1 else served, i, o, cw, cr)
    return tin, tout, usd


class Ledger:
    """What Claude has been asked and what it cost, a line per day."""

    _lock = threading.Lock()

    def __init__(self, directory: str | Path):
        self.path = Path(directory or ".") / LEDGER

    def _read(self) -> dict[str, Any]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        d.setdefault("days", {})
        return d

    @staticmethod
    def day() -> str:
        return time.strftime("%Y-%m-%d")

    def add(self, model: str, input_tokens: int, output_tokens: int, usd: float, purpose: str = "", refused: str = "") -> dict[str, Any]:
        with self._lock:
            d = self._read()
            rec = d["days"].setdefault(self.day(), {"calls": 0, "input": 0, "output": 0, "usd": 0.0, "refused": 0, "by": {}})
            rec["calls"] += 1
            rec["input"] += int(input_tokens)
            rec["output"] += int(output_tokens)
            rec["usd"] = round(float(rec["usd"]) + float(usd), 6)
            if refused:
                rec["refused"] = int(rec.get("refused", 0)) + 1
                rec["last_refusal"] = refused
            kind = "teaching" if purpose.startswith(("school", "distill")) else "writing"
            rec["by"][kind] = round(float(rec["by"].get(kind, 0.0)) + float(usd), 6)
            rec["model"] = model
            d["days"] = dict(sorted(d["days"].items())[-90:])
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(d, indent=1), encoding="utf-8")
            except OSError as e:
                log.warning("could not write the record of Claude spending: %s", e)
            return rec

    def today(self) -> dict[str, Any]:
        rec = self._read()["days"].get(self.day()) or {}
        return {"calls": int(rec.get("calls", 0)), "input": int(rec.get("input", 0)), "output": int(rec.get("output", 0)),
                "usd": float(rec.get("usd", 0.0)), "refused": int(rec.get("refused", 0)),
                "last_refusal": str(rec.get("last_refusal", "")), "by": dict(rec.get("by", {}))}

    def total(self) -> dict[str, Any]:
        days = self._read()["days"]
        return {"days": len(days), "calls": sum(int(r.get("calls", 0)) for r in days.values()),
                "usd": round(sum(float(r.get("usd", 0.0)) for r in days.values()), 4)}


def lesson_estimate(model: str) -> float:
    """Roughly what one short lesson costs: about 600 tokens in and 900 out, thinking included. A guess
    from the size of the prompts, not a measurement; the ledger records what each one really used."""
    return cost(model, 600, 900)


def money(usd: float) -> str:
    return f"${usd:.2f}" if usd >= 0.995 or usd == 0 else f"{usd * 100:.1f} cents" if usd >= 0.001 else "under a tenth of a cent"


# ---------------------------------------------------------------------------------- the model
class ClaudeLLM:
    """Claude behind the same four methods as every other model Gauge uses."""

    def __init__(self, cfg: LLMConfig, model: Optional[str] = None, calls: Optional[CallLog] = None,
                 effort: Optional[str] = None, system: str = "", connection: Any = None):
        self.cfg = cfg
        self.settings: ClaudeConfig = cfg.claude or ClaudeConfig()
        model = model or cfg.model
        self.model = model if str(model).startswith("claude") else self.settings.model
        self.calls = calls or CallLog()
        self.effort = effort if effort in EFFORTS else (self.settings.effort if self.settings.effort in EFFORTS else "medium")
        self.system = system
        self.ledger = Ledger(cfg.claude_dir or ".")
        self._client = connection

    # -- plumbing
    def _connection(self) -> Any:
        if self._client is None:
            self._client = client(timeout=float(self.cfg.timeout_seconds), retries=int(self.cfg.retries))
        return self._client

    def describe(self) -> str:
        return f"claude {self.model} (Anthropic, over the internet; effort {self.effort})"

    def context_window(self) -> int:
        return self.cfg.num_ctx

    def available(self) -> bool:
        """Does the key work and does Anthropic know this model? Asking costs nothing."""
        try:
            self._quick().models.retrieve(self.model)
            return True
        except Exception as e:
            log.debug("claude is not available: %s", e)
            return False

    def _quick(self) -> Any:
        """The same connection, but giving up after a few seconds: for questions a window is waiting on."""
        return self._connection().with_options(timeout=8.0, max_retries=0)

    def check(self) -> dict[str, Any]:
        """The same question as `available`, answered in a sentence."""
        try:
            anthropic = sdk()
        except LLMError as e:
            return {"ok": False, "message": str(e)}
        try:
            self._quick().models.retrieve(self.model)
        except anthropic.AuthenticationError:
            return {"ok": False, "message": "Anthropic did not accept the key. Check it was copied whole, or make a new one at " + KEYS_URL + "."}
        except anthropic.PermissionDeniedError:
            return {"ok": False, "message": f"The key works but may not use {self.model}. Choose another model."}
        except anthropic.NotFoundError:
            return {"ok": False, "message": f"Anthropic does not know a model called {self.model}. Choose another."}
        except anthropic.APIConnectionError:
            return {"ok": False, "message": "Could not reach Anthropic. Check the internet connection."}
        except Exception as e:
            return {"ok": False, "message": f"Claude is not answering: {str(e)[:200]}"}
        return {"ok": True, "message": f"Connected to Claude ({self.model})."}

    def _within_limit(self) -> None:
        limit = float(self.settings.daily_usd or 0)
        if limit > 0:
            spent = self.ledger.today()["usd"]
            if spent >= limit:
                raise ClaudeLimit(f"Claude has reached today's spending limit ({money(spent)} of {money(limit)}), so it is not "
                                  "asked again until tomorrow. Raise the limit in Change the AI, Claude, if you want more today.")

    @staticmethod
    def _json_rule(json_mode: bool, schema: Optional[dict[str, Any]]) -> str:
        if schema is not None:
            return ("Reply with one JSON object and nothing else: no words before or after it and no code fence. "
                    "It must fit this JSON schema:\n" + json.dumps(schema))
        if json_mode:
            return "Reply with one JSON object and nothing else: no words before or after it and no code fence."
        return ""

    def _record(self, purpose: str, t0: float, ok: bool, prompt_tokens: int, output_tokens: int) -> None:
        dur = int((time.monotonic() - t0) * 1000)
        self.calls.add(CallRecord(purpose, self.model, prompt_tokens, output_tokens, dur, ok, 0,
                                  round(output_tokens / max(0.001, dur / 1000.0), 1)))

    # -- the call
    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False,
                 schema: Optional[dict[str, Any]] = None, purpose: str = "generate") -> str:
        """One question, one answer. `temperature` is accepted for compatibility and not sent: the current
        Claude models do not take it. `max_tokens` is only ever raised, never lowered below
        `claude.max_tokens`, because the figure has to cover Claude's thinking as well as its answer; how
        long the answer is comes from what the prompt asks for."""
        anthropic = sdk()
        from .. import dryrun

        if dryrun.is_dry_run():                        # a paid request to an outside service: held in a dry run
            dryrun.hold("paid Claude request", purpose)
            raise LLMError(f"{purpose}: this is a dry run, so Claude was not asked.")
        self._within_limit()
        t0 = time.monotonic()
        whole_system = "\n\n".join(x for x in (self.system, system, self._json_rule(json_mode, schema)) if x)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(int(self.settings.max_tokens or 16000), 4 * int(max_tokens or 0)),
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.effort},
        }
        if whole_system:
            params["system"] = whole_system
        fallbacks = bool(self.settings.fallbacks) and not self.model.lower().startswith(NO_SERVER_FALLBACK)
        est_in = estimate_tokens(whole_system) + estimate_tokens(prompt)
        try:
            conn = self._connection()
            if fallbacks:
                with conn.beta.messages.stream(**params, betas=[FALLBACK_BETA], fallbacks="default") as stream:
                    message = stream.get_final_message()
            else:
                with conn.messages.stream(**params) as stream:
                    message = stream.get_final_message()
        except anthropic.BadRequestError as e:
            self._record(purpose, t0, False, est_in, 0)
            raise LLMError(f"{purpose}: Anthropic rejected the request: {_said(e)}") from e
        except anthropic.AuthenticationError as e:
            self._record(purpose, t0, False, est_in, 0)
            raise LLMError("Anthropic did not accept the key. Connect Claude again with a working key.") from e
        except anthropic.PermissionDeniedError as e:
            self._record(purpose, t0, False, est_in, 0)
            raise LLMError(f"The key may not use {self.model}: {_said(e)}") from e
        except anthropic.NotFoundError as e:
            self._record(purpose, t0, False, est_in, 0)
            raise LLMError(f"Anthropic does not know a model called {self.model}.") from e
        except anthropic.RateLimitError as e:
            self._record(purpose, t0, False, est_in, 0)
            raise LLMError(f"{purpose}: Anthropic is limiting how fast this key may ask. Wait a minute and try again.") from e
        except anthropic.APIStatusError as e:
            self._record(purpose, t0, False, est_in, 0)
            raise LLMError(f"{purpose}: Anthropic answered with an error ({e.status_code}): {_said(e)}") from e
        except anthropic.APIConnectionError as e:
            self._record(purpose, t0, False, est_in, 0)
            raise LLMError(f"{purpose}: could not reach Anthropic. Check the internet connection.") from e
        tin, tout, usd = message_cost(message, self.model)
        if message.stop_reason == "refusal":           # check before reading any content: a declined answer may be partial
            details = getattr(message, "stop_details", None)
            category = str(getattr(details, "category", "") or "")
            self.ledger.add(self.model, tin, tout, usd, purpose, refused=category or "declined")
            self._record(purpose, t0, False, tin or est_in, tout)
            raise ClaudeRefused(f"{purpose}: Claude declined this request" + (f" ({category})" if category else "") + ".", category)
        text = "".join(block.text for block in message.content if block.type == "text")
        self.ledger.add(self.model, tin, tout, usd, purpose)
        self._record(purpose, t0, True, tin or est_in, tout or estimate_tokens(text))
        if message.stop_reason == "max_tokens":
            log.warning("%s: Claude's answer was cut off at the length limit (claude.max_tokens)", purpose)
        return text


def _said(e: Any) -> str:
    return str(getattr(e, "message", "") or e)[:300]


# ---------------------------------------------------------------------------------- the settings, in words
def status(cfg: Any, check: bool = False) -> dict[str, Any]:
    """Where things stand: is the library there, is there a key, what has been spent today, is it teaching."""
    cl: ClaudeConfig = cfg.claude
    ledger = Ledger(cfg.root)
    today = ledger.today()
    source = key_source() if installed() else ""
    limit = float(cl.daily_usd or 0)
    out: dict[str, Any] = {
        "installed": installed(), "key": bool(source), "key_source": source, "model": cl.model,
        "name": MODELS.get(cl.model, {}).get("name", cl.model), "teaching": bool(cl.teach and cl.terms_ok),
        "writing": cfg.llm.provider == "claude", "today": today, "limit": limit, "total": ledger.total(),
        "left": max(0.0, limit - today["usd"]) if limit > 0 else None, "connected": None,
    }
    if check and out["installed"]:
        res = ClaudeLLM(cfg.llm, cl.model).check()
        out["connected"], out["check"] = res["ok"], res["message"]
    if not out["installed"]:
        line = "Claude is not connected. Press Connect to set it up (you need an Anthropic API key)."
    elif out["connected"] is False:
        line = out["check"]
    elif not source and out["connected"] is None:
        line = "Claude's library is installed but Gauge sees no key. Paste a key and press Connect."
    else:
        line = f"Claude is {'connected' if out['connected'] else 'set up'} ({out['name']})."
        line += (" It leads GaugeLM's school." if out["teaching"] else " It is not teaching GaugeLM (switch that on below).")
        if out["writing"]:
            line += " It is also Gauge's writer."
    spent = f"Spent today: about {money(today['usd'])} over {today['calls']} request{'' if today['calls'] == 1 else 's'}"
    spent += f", of a limit of {money(limit)} a day." if limit > 0 else ", with no daily limit set."
    if today["refused"]:
        spent += f" Claude declined {today['refused']} request{'' if today['refused'] == 1 else 's'} today."
    out["line"], out["spent_line"] = line, spent
    return out
