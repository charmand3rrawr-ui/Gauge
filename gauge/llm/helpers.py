"""Free AI helpers: hosted models with a free allowance that can teach GaugeLM and check its work.

Several companies let anyone with a free account send a limited number of requests a day to large
models. Gauge can use them in two ways:

* **to teach** - a helper joins GaugeLM's school (`native/school.py`) beside the free models on this
  computer: it writes practice scenes, answers Gauge's tasks and corrects GaugeLM's own attempts;
* **to check** - a helper reads what GaugeLM wrote, marks it, and points at the faults
  (`native/checkwork.py`).

They are free, not private, and not unlimited, and this module is built around that:

* **Nothing is used until you connect it.** You get the key yourself from the provider and answer one
  question about what is sent and under whose terms. The key is kept in a file in your user folder
  (or read from an environment variable), never in Gauge's settings.
* **It stays free.** Gauge sends each helper only so many requests a day (well inside the allowance),
  spaces them out, and leaves a helper alone for a while when it says "too many requests". On
  OpenRouter only the models offered at no charge are ever used. Gauge cannot see your account: if you
  attach a payment method at a provider, that provider's rules on charging apply.
* **Your text leaves this computer.** Passages from your library and what GaugeLM wrote are sent to
  the provider. Some free plans keep what you send and use it to improve their own products; the
  catalogue below says which, as read in November 2026.
* **Models come and go.** Gauge asks each provider which models it has rather than trusting a list
  written into the code.

Every helper speaks the same OpenAI-style protocol, so the calls go through the client Gauge already
has for LM Studio and llama.cpp. Any other service with that protocol can be added by its address.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from ..config import LLMConfig
from .client import CallLog, LLMError, OpenAICompatLLM

log = logging.getLogger(__name__)

STATE = "helpers_state.json"
PACING = 1.0                 # multiplies the pause between requests to one helper (tests set it to 0)
READ_ON = "as read in November 2026; their own pages are the truth"

# What Gauge knows about each provider. `daily` is how many requests Gauge will send it in a day, chosen to sit
# well inside the free allowance; `pace` is the least number of seconds between two requests.
PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {
        "name": "Groq", "base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
        "keys": "https://console.groq.com/keys", "terms": "https://console.groq.com/docs/legal/services-agreement",
        "free": "A free plan with no payment card: open models such as Llama, run very fast, with a daily allowance.",
        "keeps": "Groq says it does not keep what you send or train on it.",
        "daily": 300, "pace": 2.5, "prefer": ["llama-3.3-70b", "70b", "llama", "qwen", "gemma"]},
    "openrouter": {
        "name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
        "keys": "https://openrouter.ai/settings/keys", "terms": "https://openrouter.ai/terms",
        "free": "One key for many companies' models. Gauge uses only the ones offered at no charge (their names end "
                "in ':free'): about 50 requests a day on an account that has never bought credits.",
        "keeps": "Each free model is run by a different company with its own rule on keeping what you send; some keep it.",
        "daily": 40, "pace": 4.0, "free_only": True, "key_check": "/key",     # its model list is public, so the key is tried here
        "prefer": ["llama", "gemma", "mistral", "qwen", "nemotron"]},
    "gemini": {
        "name": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY", "keys": "https://aistudio.google.com/apikey", "terms": "https://ai.google.dev/gemini-api/terms",
        "free": "A free allowance on Google's Gemini models, set per model.",
        "keeps": "On the free allowance Google uses what you send, and the answers, to improve its products, and people may "
                 "read it. Its terms do not allow using the service to develop models that compete with it: Gauge switches "
                 "its teaching off to begin with, so it only checks.",
        "daily": 100, "pace": 6.0, "prefer": ["flash"], "teach": False},
    "mistral": {
        "name": "Mistral", "base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY",
        "keys": "https://console.mistral.ai/api-keys", "terms": "https://mistral.ai/terms",
        "free": "A free 'Experiment' plan on Mistral's own models (it asks for a phone number).",
        "keeps": "On the free plan Mistral may train its models on what you send unless you switch that off in its privacy settings.",
        "daily": 200, "pace": 1.5, "prefer": ["mistral-small-latest", "mistral-small", "open-mistral-nemo", "mistral-medium", "mistral"]},
}
NOT_CHAT = ("embed", "whisper", "tts", "guard", "moderation", "rerank", "imagen", "image", "veo", "audio", "speech",
            "transcri", "ocr", "aqa", "live", "robotics", "computer-use", "lyria", "bge-", "clip", "pii")

HELPER_SYSTEM = (
    "You are helping with GaugeLM, a very small language model that its owner trains on their own computer to write "
    "serial fiction. Where a request speaks of 'a student writer', the student is GaugeLM. Your answer may be saved as a "
    "worked example for it, so write exactly what is asked for and nothing else: no preamble, no notes, no closing remark. "
    "Prefer plain words and clear sentences.")


class HelperBusy(LLMError):
    """The helper said 'too many requests', or reached the number Gauge allows it in a day. It is left alone for a while."""


# ---------------------------------------------------------------------------------- the catalogue
def provider(pid: str, cfg: Any = None) -> dict[str, Any]:
    """What is known about a helper: a catalogue entry, or one the author added by its address."""
    if pid in PROVIDERS:
        return PROVIDERS[pid]
    entry = ((cfg.helpers.providers if cfg is not None else {}) or {}).get(pid) or {}
    if not entry.get("base_url"):
        raise ValueError(f"'{pid}' is not a helper Gauge knows. Known: {', '.join(PROVIDERS)}. To add another, give its address.")
    return {"name": pid, "base_url": str(entry["base_url"]), "key_env": "", "keys": "", "terms": "",
            "free": "A service you added by its address.", "keeps": "Gauge knows nothing about what it keeps; read its terms.",
            "daily": 100, "pace": 2.0, "prefer": []}


def known(cfg: Any) -> list[str]:
    """Every helper there is: the catalogue, then the ones the author added."""
    return list(PROVIDERS) + [p for p in (cfg.helpers.providers or {}) if p not in PROVIDERS]


def safe_address(url: str) -> str:
    """A key is only ever sent over an encrypted connection, or to this computer or the home network."""
    u = urlparse(url.strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("The address should start with https://")
    if u.scheme == "http":
        host = u.hostname
        try:
            local = ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == "localhost" or host.endswith(".local")
        if not local:
            raise ValueError("That address is not encrypted (http). Use the https:// address, so the key cannot be read on the way.")
    return url.strip().rstrip("/")


# ---------------------------------------------------------------------------------- the keys
def keys_file() -> Path:
    override = os.environ.get("GAUGE_HELPER_KEYS_FILE")
    return Path(override) if override else Path.home() / ".gauge" / "helper-keys.json"


def _keys() -> dict[str, str]:
    try:
        d = json.loads(keys_file().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in d.items() if v} if isinstance(d, dict) else {}


def _write_keys(d: dict[str, str]) -> None:
    f = keys_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=1), encoding="utf-8")
    try:
        f.chmod(0o600)
    except OSError:
        pass


def save_key(pid: str, key: str) -> None:
    key = "".join(key.split())
    if not key:
        raise ValueError("no key was given")
    d = _keys()
    d[pid] = key
    _write_keys(d)


def forget_key(pid: str) -> bool:
    d = _keys()
    had = d.pop(pid, None) is not None
    if had:
        _write_keys(d)
    return had


def key_for(pid: str, cfg: Any = None) -> str:
    """The saved key, or else the provider's usual environment variable."""
    saved = _keys().get(pid, "")
    if saved:
        return saved
    try:
        env = provider(pid, cfg).get("key_env") or ""
    except ValueError:
        env = ""
    return os.environ.get(env, "") if env else ""


# ---------------------------------------------------------------------------------- which models
def _version(mid: str) -> tuple[float, float]:
    """(newest version number, biggest size in billions) found in a model's name, for choosing among likes."""
    low = mid.lower()
    sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b\b", low)]
    nums = [float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w]*b\b)", low.replace("-", " ").replace("/", " "))]
    return (max([n for n in nums if n < 100] or [0.0]), max(sizes or [0.0]))


def chat_models(pid: str, listing: Any, cfg: Any = None) -> list[str]:
    """The models of a provider that can hold a conversation and, where it matters, cost nothing. Best guess first."""
    info = provider(pid, cfg)
    rows = listing.get("data", listing.get("models", [])) if isinstance(listing, dict) else listing
    ids: list[str] = []
    for row in rows or []:
        mid = str(row.get("id") or row.get("name") or "") if isinstance(row, dict) else str(row)
        mid = mid.removeprefix("models/")
        low = mid.lower()
        if not mid or any(x in low for x in NOT_CHAT):
            continue
        if info.get("free_only"):
            price = (row.get("pricing") or {}) if isinstance(row, dict) else {}
            free = low.endswith(":free") or (price and str(price.get("prompt")) in ("0", "0.0") and str(price.get("completion")) in ("0", "0.0"))
            if not free:
                continue
        ids.append(mid)
    ids = list(dict.fromkeys(ids))

    def rank(mid: str) -> tuple[int, int, float, float]:
        low = mid.lower()
        place = next((i for i, pat in enumerate(info.get("prefer") or []) if pat in low), 99)
        unfinished = int(any(x in low for x in ("preview", "exp", "beta")))
        ver, size = _version(mid)
        return (place, unfinished, -size, -ver)

    return sorted(ids, key=rank)


def list_models(pid: str, key: str, cfg: Any = None, timeout: float = 20.0) -> dict[str, Any]:
    """Ask the provider which models it has. This is also how a key is tried: it costs nothing."""
    info = provider(pid, cfg)
    base, headers = info["base_url"].rstrip("/"), {"Authorization": f"Bearer {key}"}
    rejected = {"ok": False, "models": [], "rejected": True,
                "message": f"{info['name']} did not accept the key. Check it was copied whole"
                           + (f", or make a new one at {info['keys']}." if info.get("keys") else ".")}
    try:
        if info.get("key_check") and requests.get(base + info["key_check"], headers=headers, timeout=timeout).status_code in (400, 401, 403):
            return rejected
        r = requests.get(base + "/models", headers=headers, timeout=timeout)
    except requests.RequestException as e:
        return {"ok": False, "models": [], "message": f"Could not reach {info['name']}: {str(e)[:120]}. Check the internet connection."}
    if r.status_code in (400, 401, 403):
        return rejected
    if not r.ok:
        return {"ok": False, "models": [], "message": f"{info['name']} answered with an error ({r.status_code})."}
    try:
        models = chat_models(pid, r.json(), cfg)
    except ValueError:
        models = []
    if not models:
        return {"ok": False, "models": [], "message": f"{info['name']} accepted the key but listed no model Gauge can use for free."}
    return {"ok": True, "models": models, "message": f"{info['name']} accepted the key and offers {len(models)} usable model(s)."}


# ---------------------------------------------------------------------------------- how much each has been asked
class Usage:
    """Requests sent to each helper today, and until when a helper that said 'too many' is left alone."""

    _lock = threading.Lock()

    def __init__(self, directory: str | Path):
        self.path = Path(directory or ".") / STATE

    def _read(self) -> dict[str, Any]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        return d if isinstance(d, dict) else {}

    def _write(self, d: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(d, indent=1), encoding="utf-8")
        except OSError as e:
            log.warning("could not write the record of helper requests: %s", e)

    @staticmethod
    def day() -> str:
        return time.strftime("%Y-%m-%d")

    def today(self, pid: str) -> int:
        rec = self._read().get(pid) or {}
        return int(rec.get("count", 0)) if rec.get("day") == self.day() else 0

    def resting(self, pid: str) -> float:
        """Seconds until this helper may be asked again; 0 when it may be asked now."""
        return max(0.0, float((self._read().get(pid) or {}).get("rest_until", 0)) - time.time())

    def count(self, pid: str) -> int:
        with self._lock:
            d = self._read()
            rec = d.get(pid) or {}
            if rec.get("day") != self.day():
                rec = {"day": self.day(), "count": 0, "rest_until": rec.get("rest_until", 0)}
            rec["count"] = int(rec.get("count", 0)) + 1
            d[pid] = rec
            self._write(d)
            return rec["count"]

    def rest(self, pid: str, minutes: float, why: str = "") -> None:
        with self._lock:
            d = self._read()
            rec = d.get(pid) or {"day": self.day(), "count": 0}
            rec["rest_until"] = time.time() + max(0.0, minutes) * 60
            rec["why"] = why[:200]
            d[pid] = rec
            self._write(d)


def daily_allowance(pid: str, cfg: Any) -> int:
    """How many requests Gauge may send this helper in a day."""
    entry = (cfg.helpers.providers or {}).get(pid) or {}
    if int(entry.get("daily") or 0) > 0:
        return int(entry["daily"])
    if int(cfg.helpers.daily_requests or 0) > 0:
        return int(cfg.helpers.daily_requests)
    return int(provider(pid, cfg)["daily"])


# ---------------------------------------------------------------------------------- the model
class HelperLLM(OpenAICompatLLM):
    """One helper's model: the ordinary OpenAI-style client, kept inside the day's allowance and paced."""

    _last: dict[str, float] = {}
    _pace_lock = threading.Lock()

    def __init__(self, cfg: Any, pid: str, model: str, calls: Optional[CallLog] = None, system: str = HELPER_SYSTEM):
        info = provider(pid, cfg)
        base: LLMConfig = cfg.llm.model_copy(update={
            "provider": "openai", "base_url": info["base_url"], "model": model, "fast_model": "", "api_key": key_for(pid, cfg),
            "retries": 1, "timeout_seconds": min(int(cfg.llm.timeout_seconds), 120), "structured_output": False})
        super().__init__(base, model, calls)
        self.pid, self.info, self.system = pid, info, system
        self.usage = Usage(cfg.root)
        self.allowance = daily_allowance(pid, cfg)
        self.rest_minutes = float(cfg.helpers.rest_minutes)

    def describe(self) -> str:
        return f"{self.info['name']} {self.model} (a free helper, over the internet)"

    def available(self) -> bool:
        """Usable right now: a key, not resting, and requests left today. Nothing is sent to find out."""
        return bool(self.cfg.api_key) and not os.environ.get("GAUGE_OFFLINE") and self.usage.resting(self.pid) <= 0 \
            and self.usage.today(self.pid) < self.allowance

    def _wait_my_turn(self) -> None:
        gap = float(self.info.get("pace", 2.0)) * PACING
        with self._pace_lock:
            wait = self._last.get(self.pid, 0.0) + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[self.pid] = time.monotonic()

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False,
                 schema: Optional[dict[str, Any]] = None, purpose: str = "generate") -> str:
        from .. import dryrun

        name = self.info["name"]
        if dryrun.is_dry_run():                      # text leaving this computer for an outside service: held in a dry run
            dryrun.hold("request to a free AI helper", f"{name}: {purpose}")
            raise LLMError(f"{purpose}: this is a dry run, so {name} was not asked.")
        if os.environ.get("GAUGE_OFFLINE"):
            raise LLMError(f"{purpose}: Gauge is set to stay offline, so {name} was not asked.")
        if not self.cfg.api_key:
            raise LLMError(f"{name} is not connected: there is no key for it.")
        left = self.usage.resting(self.pid)
        if left > 0:
            raise HelperBusy(f"{name} asked for a pause; Gauge will try it again in about {max(1, round(left / 60))} minute(s).")
        if self.usage.today(self.pid) >= self.allowance:
            raise HelperBusy(f"{name} has had the {self.allowance} requests Gauge allows it in a day, so it rests until tomorrow.")
        self._wait_my_turn()
        self.usage.count(self.pid)                   # counted when sent: a refused request still uses the allowance
        whole_system = "\n\n".join(x for x in (self.system, system) if x)
        if json_mode or schema is not None:          # asked for in words: free models differ in what else they accept
            whole_system += "\n\nReply with one JSON object and nothing else: no words before or after it and no code fence."
        try:
            out = super().generate(prompt, system=whole_system, temperature=temperature,
                                   max_tokens=max(512, 2 * int(max_tokens or self.cfg.max_tokens)), purpose=purpose)
        except LLMError as e:
            said = str(e)
            if "HTTP 429" in said:
                self.usage.rest(self.pid, self.rest_minutes, said)
                raise HelperBusy(f"{name} said it has had too many requests. Gauge will leave it alone for "
                                 f"{self.rest_minutes:g} minutes.") from e
            if "HTTP 401" in said or "HTTP 403" in said:
                raise LLMError(f"{name} no longer accepts the key. Connect it again with a working key.") from e
            if "HTTP 402" in said:
                self.usage.rest(self.pid, 24 * 60, said)
                raise HelperBusy(f"{name} asked for payment, so Gauge has stopped using it for a day. Nothing was paid.") from e
            raise
        # Some free models think out loud first; only the answer is wanted.
        return re.sub(r"(?s)<think>.*?</think>", "", out or "").strip()


# ---------------------------------------------------------------------------------- connecting and the settings
def entry(cfg: Any, pid: str) -> dict[str, Any]:
    return dict((cfg.helpers.providers or {}).get(pid) or {})


def connect(cfg: Any, pid: str, key: str = "", base_url: str = "", model: str = "") -> dict[str, Any]:
    """Try a key (asking the provider for its models, which is free), keep it if it works, and choose a model.
    Nothing is sent to the helper's models, and it neither teaches nor checks, until `allow` is called."""
    pid = pid.strip().lower()
    if os.environ.get("GAUGE_OFFLINE"):
        return {"ok": False, "message": "Gauge is set to stay offline (GAUGE_OFFLINE), so no helper can be connected."}
    providers = dict(cfg.helpers.providers or {})
    mine = dict(providers.get(pid) or {})
    if pid not in PROVIDERS:
        if base_url:
            try:
                mine["base_url"] = safe_address(base_url)
            except ValueError as e:
                return {"ok": False, "message": str(e)}
        if not mine.get("base_url"):
            return {"ok": False, "message": f"'{pid}' is not a helper Gauge knows ({', '.join(PROVIDERS)}). To add another, give its address."}
        providers[pid] = mine
        cfg.helpers.providers = providers
    info = provider(pid, cfg)
    key = "".join(str(key or "").split()) or key_for(pid, cfg)
    if not key:
        where = f" Make one at {info['keys']}." if info.get("keys") else ""
        return {"ok": False, "message": f"No key was given for {info['name']}.{where}"}
    got = list_models(pid, key, cfg)
    if not got["ok"]:
        return got
    save_key(pid, key)
    chosen = model if model in got["models"] else (mine.get("model") if mine.get("model") in got["models"] else got["models"][0])
    mine.update({"model": chosen})
    mine.setdefault("teach", bool(info.get("teach", True)))
    mine.setdefault("check", True)
    mine.setdefault("terms_ok", False)
    providers[pid] = mine
    cfg.helpers.providers = providers
    cfg.save()
    msg = f"{info['name']} is connected. Gauge chose the model {chosen}; {len(got['models'])} are on offer."
    msg += " Nothing is sent to it until you switch it on." if not mine["terms_ok"] else ""
    return {"ok": True, "models": got["models"], "model": chosen, "message": msg}


def allow(cfg: Any, pid: str, yes: bool) -> dict[str, Any]:
    """The author's own yes to what is sent, to what the provider may keep, and to its terms on using answers
    to train a model. Without it a connected helper is never asked anything."""
    mine = entry(cfg, pid)
    if not mine.get("model"):
        return {"ok": False, "message": "Connect it first."}
    mine["terms_ok"] = bool(yes)
    cfg.helpers.providers = {**(cfg.helpers.providers or {}), pid: mine}
    cfg.save()
    name = provider(pid, cfg)["name"]
    if not yes:
        return {"ok": True, "message": f"{name} was left switched off."}
    roles = " and ".join(r for r, on in (("teaches GaugeLM", mine.get("teach")), ("checks its work", mine.get("check"))) if on)
    return {"ok": True, "message": f"{name} is switched on: it {roles or 'has no job yet (switch teaching or checking on)'}."}


def set_role(cfg: Any, pid: str, teach: Optional[bool] = None, check: Optional[bool] = None, model: Optional[str] = None,
             daily: Optional[int] = None) -> dict[str, Any]:
    mine = entry(cfg, pid)
    if not mine.get("model"):
        return {"ok": False, "message": "Connect it first."}
    said = []
    name = provider(pid, cfg)["name"]
    if teach is not None:
        mine["teach"] = bool(teach)
        said.append(f"{name} {'teaches GaugeLM' if teach else 'no longer teaches GaugeLM'}.")
    if check is not None:
        mine["check"] = bool(check)
        said.append(f"{name} {'checks its work' if check else 'no longer checks its work'}.")
    if model:
        mine["model"] = model.strip()
        said.append(f"{name} now uses the model {mine['model']}.")
    if daily is not None:
        mine["daily"] = max(0, int(daily))
        said.append(f"{name} may be sent {daily_allowance_of(mine, pid, cfg)} requests a day.")
    cfg.helpers.providers = {**(cfg.helpers.providers or {}), pid: mine}
    cfg.save()
    return {"ok": True, "message": " ".join(said) or "Nothing was changed."}


def daily_allowance_of(mine: dict[str, Any], pid: str, cfg: Any) -> int:
    return int(mine.get("daily") or 0) or int(cfg.helpers.daily_requests or 0) or int(provider(pid, cfg)["daily"])


def disconnect(cfg: Any, pid: str) -> dict[str, Any]:
    had = forget_key(pid)
    providers = dict(cfg.helpers.providers or {})
    name = provider(pid, cfg)["name"] if (pid in PROVIDERS or pid in providers) else pid
    providers.pop(pid, None)
    cfg.helpers.providers = providers
    cfg.save()
    return {"ok": True, "message": f"{name} is disconnected" + (": its saved key was deleted." if had else ".")
            + " What it already taught GaugeLM is kept until you remove it."}


def ask_text(cfg: Any, pid: str) -> str:
    """The question put to the author before a helper is ever asked anything."""
    info = provider(pid, cfg)
    mine = entry(cfg, pid)
    lines = [f"Switch {info['name']} on?", "",
             f"Gauge will send it up to {daily_allowance_of(mine, pid, cfg)} requests a day, using the model {mine.get('model', '?')}.", "",
             "1. What is sent. Passages from your library, and what GaugeLM wrote, go to " + info["name"] + " over the internet.",
             "",
             "2. What it keeps (" + READ_ON + "). " + info["keeps"], "",
             "3. Its terms are yours to read. Some providers, and some models' licences, do not allow their answers to be "
             "used to train another AI model. Gauge cannot decide that for you" + (f": {info['terms']}" if info.get("terms") else "."),
             "",
             "4. It is free only while your account there is on its free plan. Gauge counts its own requests and stops at the "
             "number above, but it cannot see your account: if you add a payment method there, that provider's charges apply.",
             "", "Every lesson is labelled with who wrote it, so it can be removed later."]
    return "\n".join(lines)


def status(cfg: Any) -> list[dict[str, Any]]:
    """Every helper, connected or not, in words."""
    usage = Usage(cfg.root)
    out = []
    for pid in known(cfg):
        info, mine = provider(pid, cfg), entry(cfg, pid)
        connected = bool(mine.get("model") and key_for(pid, cfg))
        on = connected and bool(mine.get("terms_ok"))
        allowance = daily_allowance_of(mine, pid, cfg)
        used, rest = usage.today(pid), usage.resting(pid)
        if not connected:
            line = f"{info['name']}: not connected. {info['free']}" + (f" Key: {info['keys']}" if info.get("keys") else "")
        else:
            jobs = [j for j, flag in (("teaches", mine.get("teach")), ("checks", mine.get("check"))) if flag]
            what = (" and ".join(jobs) if jobs else "has no job") if on else "connected but switched off"
            line = f"{info['name']}: {what}, with {mine['model']}; {used} of {allowance} requests today"
            line += f"; resting for {max(1, round(rest / 60))} more minute(s)." if rest > 0 else "."
        out.append({"id": pid, "name": info["name"], "connected": connected, "on": on, "teach": bool(on and mine.get("teach")),
                    "check": bool(on and mine.get("check")), "model": mine.get("model", ""), "used": used, "allowance": allowance,
                    "resting": rest, "free": info["free"], "keeps": info["keeps"], "keys": info.get("keys", ""),
                    "terms": info.get("terms", ""), "line": line})
    return out


def working(cfg: Any, job: str, calls: Optional[CallLog] = None) -> list[tuple[str, HelperLLM]]:
    """The helpers that are switched on for a job ('teach' or 'check') and can be asked right now."""
    if os.environ.get("GAUGE_OFFLINE"):
        return []
    out = []
    for row in status(cfg):
        if row[job]:
            llm = HelperLLM(cfg, row["id"], row["model"], calls)
            if llm.available():
                out.append((row["id"], llm))
    return out
