"""Local LLM clients.

Gauge never talks to a paid API by default. It expects a local server:

* **Ollama** (default) at http://localhost:11434 - one installer on Windows, GPU-accelerated.
* Any **OpenAI-compatible** server (LM Studio, llama.cpp ``llama-server``, vLLM, text-gen-webui).
* ``FakeLLM`` for tests and for running the analysis pipeline on machines with no GPU.
* **GaugeLM** (``provider = "native"``): Gauge's own numpy model, see ``gauge/native``.
* **Claude** (``provider = "claude"``): Anthropic's paid API, only once you connect a key; see ``claude.py``.

Every client: sets the context window explicitly (``num_ctx``), retries transient failures,
supports JSON-schema constrained output with automatic fallback, keeps the model warm, and
records a ``CallRecord`` (purpose, tokens, duration) for each call so you can see where the
minutes go.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import requests

from ..config import LLMConfig
from .tokens import estimate_tokens

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------------------
@dataclass
class CallRecord:
    purpose: str
    model: str
    prompt_tokens: int
    output_tokens: int
    duration_ms: int
    ok: bool
    retries: int = 0
    tokens_per_second: float = 0.0


class CallLog:
    """Thread-safe in-memory list of calls, with an optional sink (e.g. the database)."""

    def __init__(self) -> None:
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()
        self.sink: Optional[Callable[[CallRecord], None]] = None
        self.context: dict[str, Any] = {}  # project / chapter tags added by the pipeline

    def add(self, rec: CallRecord) -> None:
        with self._lock:
            self.records.append(rec)
        if self.sink is not None:
            try:
                self.sink(rec)
            except Exception as e:  # never let telemetry break generation
                log.debug("call log sink failed: %s", e)

    def reset(self) -> None:
        with self._lock:
            self.records.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        with self._lock:
            for r in self.records:
                s = out.setdefault(r.purpose, {"calls": 0, "seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0, "failed": 0})
                s["calls"] += 1
                s["seconds"] += r.duration_ms / 1000.0
                s["prompt_tokens"] += r.prompt_tokens
                s["output_tokens"] += r.output_tokens
                s["failed"] += 0 if r.ok else 1
        return out

    def total_seconds(self) -> float:
        with self._lock:
            return sum(r.duration_ms for r in self.records) / 1000.0


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------------------
class LLM(Protocol):
    calls: CallLog
    model: str

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False,
                 schema: Optional[dict[str, Any]] = None, purpose: str = "generate") -> str: ...

    def available(self) -> bool: ...

    def describe(self) -> str: ...

    def context_window(self) -> int: ...


def _backoff(cfg: LLMConfig, attempt: int) -> float:
    return min(30.0, cfg.retry_backoff_seconds * (2 ** attempt))


class _HTTPBase:
    """Shared retry / telemetry plumbing for the two HTTP clients."""

    def __init__(self, cfg: LLMConfig, model: Optional[str] = None, calls: Optional[CallLog] = None):
        self.cfg = cfg
        self.model = model or cfg.model
        self.calls = calls or CallLog()
        self._schema_unsupported = False

    def context_window(self) -> int:
        return self.cfg.num_ctx

    def _record(self, purpose: str, prompt: str, system: str, output: str, t0: float, ok: bool,
                retries: int, prompt_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                eval_ns: Optional[int] = None) -> None:
        dur = int((time.monotonic() - t0) * 1000)
        pt = prompt_tokens if prompt_tokens is not None else estimate_tokens(system) + estimate_tokens(prompt)
        ot = output_tokens if output_tokens is not None else estimate_tokens(output)
        tps = (ot / (eval_ns / 1e9)) if eval_ns else (ot / max(0.001, dur / 1000.0))
        self.calls.add(CallRecord(purpose, self.model, pt, ot, dur, ok, retries, round(tps, 1)))

    def _with_retries(self, fn: Callable[[], requests.Response], purpose: str) -> tuple[requests.Response, int]:
        last: Optional[Exception] = None
        for attempt in range(self.cfg.retries + 1):
            try:
                r = fn()
                if r.status_code in (429, 500, 502, 503, 504):
                    last = LLMError(f"{purpose}: HTTP {r.status_code} {r.text[:200]}")
                else:
                    return r, attempt
            except requests.RequestException as e:
                last = e
            if attempt < self.cfg.retries:
                wait = _backoff(self.cfg, attempt)
                log.warning("%s failed (%s); retry %d/%d in %.0fs", purpose, last, attempt + 1, self.cfg.retries, wait)
                time.sleep(wait)
        raise LLMError(f"{purpose}: giving up after {self.cfg.retries + 1} attempts: {last}")


class OllamaLLM(_HTTPBase):
    def __init__(self, cfg: LLMConfig, model: Optional[str] = None, calls: Optional[CallLog] = None):
        super().__init__(cfg, model, calls)
        self.base = cfg.base_url.rstrip("/")

    def available(self) -> bool:
        try:
            return requests.get(f"{self.base}/api/tags", timeout=5).ok
        except requests.RequestException:
            return False

    def models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    def describe(self) -> str:
        return f"ollama {self.model} @ {self.base} (ctx {self.cfg.num_ctx})"

    def _payload(self, prompt: str, system: str, temperature: Optional[float], max_tokens: Optional[int],
                 json_mode: bool, schema: Optional[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model, "prompt": prompt, "system": system, "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_predict": max_tokens or self.cfg.max_tokens,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        from .. import limit

        threads = limit.engine_threads()
        if threads is not None:  # the capacity limit: the engine may use only this many processor threads
            payload["options"]["num_thread"] = threads
        if not getattr(self.cfg, "use_gpu", True):  # the device switch: no graphics card chosen, so none is used
            payload["options"]["num_gpu"] = 0
        if schema is not None and self.cfg.structured_output and not self._schema_unsupported:
            payload["format"] = schema
        elif json_mode or schema is not None:
            payload["format"] = "json"
        return payload

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False,
                 schema: Optional[dict[str, Any]] = None, purpose: str = "generate") -> str:
        t0 = time.monotonic()
        need = estimate_tokens(system) + estimate_tokens(prompt) + (max_tokens or self.cfg.max_tokens)
        if need > self.cfg.num_ctx:
            log.warning("%s: prompt+output (~%d tokens) exceeds num_ctx=%d; the server will truncate the "
                        "start of the prompt. Lower exemplars/cast or raise llm.num_ctx.", purpose, need, self.cfg.num_ctx)
        payload = self._payload(prompt, system, temperature, max_tokens, json_mode, schema)
        retries = 0
        try:
            r, retries = self._with_retries(
                lambda: requests.post(f"{self.base}/api/generate", json=payload, timeout=self.cfg.timeout_seconds), purpose)
            if r.status_code == 400 and "format" in payload and payload["format"] != "json":
                # Older Ollama builds reject JSON-schema formats: fall back to plain JSON mode for good.
                self._schema_unsupported = True
                log.info("server rejected JSON schema output; falling back to json mode")
                payload["format"] = "json"
                r, more = self._with_retries(
                    lambda: requests.post(f"{self.base}/api/generate", json=payload, timeout=self.cfg.timeout_seconds), purpose)
                retries += more + 1
            if not r.ok:
                raise LLMError(f"{purpose}: HTTP {r.status_code} {r.text[:300]}")
            data = r.json()
            out = data.get("response", "")
            self._record(purpose, prompt, system, out, t0, True, retries, data.get("prompt_eval_count"),
                         data.get("eval_count"), data.get("eval_duration"))
            return out
        except Exception:
            self._record(purpose, prompt, system, "", t0, False, retries)
            raise

    def throughput(self, max_tokens: int = 200) -> dict[str, float]:
        """Measure prompt-eval and generation speed for the bench command."""
        prompt = "Write the opening paragraph of a serial web novel chapter about a courier in the rain. " * 4
        payload = self._payload(prompt, "", 0.7, max_tokens, False, None)
        t0 = time.monotonic()
        r = requests.post(f"{self.base}/api/generate", json=payload, timeout=self.cfg.timeout_seconds)
        r.raise_for_status()
        d = r.json()
        wall = time.monotonic() - t0
        gen_tps = d.get("eval_count", 0) / max(1e-9, d.get("eval_duration", 1) / 1e9)
        prompt_tps = d.get("prompt_eval_count", 0) / max(1e-9, d.get("prompt_eval_duration", 1) / 1e9)
        return {"generation_tps": round(gen_tps, 1), "prompt_tps": round(prompt_tps, 1), "wall_seconds": round(wall, 1),
                "load_seconds": round(d.get("load_duration", 0) / 1e9, 1)}


class OpenAICompatLLM(_HTTPBase):
    def __init__(self, cfg: LLMConfig, model: Optional[str] = None, calls: Optional[CallLog] = None):
        super().__init__(cfg, model, calls)
        self.base = cfg.base_url.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base += "/v1"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def available(self) -> bool:
        try:
            return requests.get(f"{self.base}/models", headers=self._headers(), timeout=5).ok
        except requests.RequestException:
            return False

    def describe(self) -> str:
        return f"openai-compatible {self.model} @ {self.base}"

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False,
                 schema: Optional[dict[str, Any]] = None, purpose: str = "generate") -> str:
        t0 = time.monotonic()
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if schema is not None and self.cfg.structured_output and not self._schema_unsupported:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        elif json_mode or schema is not None:
            payload["response_format"] = {"type": "json_object"}
        retries = 0
        try:
            post = lambda: requests.post(f"{self.base}/chat/completions", json=payload, headers=self._headers(),  # noqa: E731
                                         timeout=self.cfg.timeout_seconds)
            r, retries = self._with_retries(post, purpose)
            if r.status_code == 400 and payload.get("response_format", {}).get("type") == "json_schema":
                self._schema_unsupported = True
                payload["response_format"] = {"type": "json_object"}
                r, more = self._with_retries(post, purpose)
                retries += more + 1
            if not r.ok:
                raise LLMError(f"{purpose}: HTTP {r.status_code} {r.text[:300]}")
            data = r.json()
            out = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            self._record(purpose, prompt, system, out, t0, True, retries, usage.get("prompt_tokens"), usage.get("completion_tokens"))
            return out
        except Exception:
            self._record(purpose, prompt, system, "", t0, False, retries)
            raise

    def throughput(self, max_tokens: int = 200) -> dict[str, float]:
        t0 = time.monotonic()
        out = self.generate("Write the opening paragraph of a serial web novel chapter about a courier in the rain.",
                            max_tokens=max_tokens, purpose="bench")
        wall = time.monotonic() - t0
        return {"generation_tps": round(estimate_tokens(out) / max(0.001, wall), 1), "prompt_tps": 0.0,
                "wall_seconds": round(wall, 1), "load_seconds": 0.0}


class FakeLLM:
    """Deterministic stand-in. Produces structurally valid output so pipelines can be tested."""

    def __init__(self, cfg: Optional[LLMConfig] = None, canned: Optional[list[str]] = None,
                 fail_on: Optional[Callable[[str], bool]] = None, delay: float = 0.0):
        self.cfg = cfg or LLMConfig(provider="fake")
        self.model = self.cfg.model
        self.calls_list: list[dict[str, Any]] = []
        self.calls = CallLog()
        self.canned = list(canned or [])
        self.fail_on = fail_on  # predicate on the prompt: raise LLMError when true
        self.delay = delay
        self._lock = threading.Lock()

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return "fake llm (offline)"

    def context_window(self) -> int:
        return self.cfg.num_ctx

    def throughput(self, max_tokens: int = 200) -> dict[str, float]:
        return {"generation_tps": 1000.0, "prompt_tps": 10000.0, "wall_seconds": 0.0, "load_seconds": 0.0}

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False,
                 schema: Optional[dict[str, Any]] = None, purpose: str = "generate") -> str:
        t0 = time.monotonic()
        with self._lock:
            self.calls_list.append({"prompt": prompt, "system": system, "json_mode": json_mode or schema is not None,
                                    "purpose": purpose, "schema": schema})
            canned = self.canned.pop(0) if self.canned else None
        if self.delay:
            time.sleep(self.delay)
        if self.fail_on is not None and self.fail_on(prompt):
            self.calls.add(CallRecord(purpose, self.model, estimate_tokens(prompt), 0, int((time.monotonic() - t0) * 1000), False))
            raise LLMError(f"fake failure for {purpose}")
        out = canned if canned is not None else self._answer(prompt, json_mode or schema is not None)
        self.calls.add(CallRecord(purpose, self.model, estimate_tokens(system) + estimate_tokens(prompt), estimate_tokens(out),
                                  int((time.monotonic() - t0) * 1000), True))
        return out

    def _answer(self, prompt: str, json_mode: bool) -> str:
        low = prompt.lower()
        m = re.match(r"\s*\[agent:([a-z_]+):([a-z_]+)\]", prompt)
        if m:
            from ..write.agents.fakes import FAKE_AGENT_RESPONSES

            canned = FAKE_AGENT_RESPONSES.get((m.group(1), m.group(2)))
            if canned is not None:
                return json.dumps(canned)
            return json.dumps({"ok": True}) if json_mode else "A fine question. The answer is in chapter one."
        if "a student writer does not know the word" in low:
            m = re.search(r'the word "([^"]+)"', prompt)
            w = m.group(1) if m else "word"
            return (f"Here \"{w}\" means the thing the passage is about.\n1. She did not trust the {w}.\n"
                    f"2. Nobody had seen a {w} like it.\n3. The {w} was gone by morning.")
        if "a student writer wrote this passage of a story" in low:      # checking GaugeLM's work (native/checkwork.py)
            passage = prompt.split("story:\n\n", 1)[-1].split("\n\nCheck it.", 1)[0]
            words = passage.split()
            return json.dumps({"marks": {"clarity": 5, "grammar": 6, "sense": 4, "interest": 5},
                               "problems": [{"quote": " ".join(words[:3]), "problem": "The opening words do not say who is acting.",
                                             "fix": "Name her first."},
                                            {"quote": "words that are not in the passage at all", "problem": "Invented.", "fix": ""}],
                               "corrected": "Rin moved first. " + " ".join(words[3:])})
        if "a student writer was set this task" in low:
            return "mark: 4\nThe seal was whole again. Rin did not wait to learn why; she ran, and the footsteps ran with her."
        if "the author says:" in low and not json_mode:
            asked = prompt.rsplit("The author says:", 1)[1].split("\n")[0].strip()
            about = "your story" if "STORY:" in prompt else "writing"
            return f"A thought about {about}: {asked[:80]} Suggestion: raise what it costs her."
        if "stay in character as" in low:
            return "Okay. Okay. You want the truth? I ran because the footsteps knew my name. One more time."
        if "answer as the world" in low:
            return "Rain since dusk. The market is closing; the vendor with one arm is still counting coins."
        if low.startswith("[engineer:classify]"):
            m = re.search(r"task: (.*)", low)
            task = m.group(1) if m else low
            ai = any(w in task for w in ("summar", "rewrite", "critique", "tone", "write a chapter"))
            return json.dumps({"needs_ai": ai, "language": "powershell" if "powershell" in task else "python", "why": "fake",
                               "schedule": "daily" if "daily" in task else ""})
        if low.startswith("[engineer:write]"):
            from ..code.fakes import FAKE_BOT_FILES

            return json.dumps({"files": FAKE_BOT_FILES, "notes": "counts words with the toolkit"})
        if low.startswith("[self:propose]"):
            from ..code.fakes import fake_proposal

            return json.dumps(fake_proposal(prompt))
        if low.startswith("[improv:turn]"):
            m = re.search(r"next beat as ([^:]+):", prompt)
            who = (m.group(1).strip() if m else "Someone")
            return f"\"Okay. Okay,\" {who} said, and checked the exits first." if who.startswith("Rin") else f"{who} stepped between them and said nothing."
        if low.startswith("[translate]"):
            body = prompt.split("TEXT:\n", 1)[1] if "TEXT:\n" in prompt else prompt
            return "[translated] " + body.strip()
        if low.startswith("[improv:world]"):
            return "The alley's one working light cut out, and somewhere a shutter slammed down early."
        if json_mode:
            if "build the world state" in low:
                return json.dumps({"name": "Lowmarket", "physics_rules": ["a rewind lasts ten seconds, no more", "rain never stops in the lower city"],
                                   "power_rules": ["each rewind erases one memory"], "social_rules": ["couriers are untouchable while carrying a sealed bag"],
                                   "time": "night of day 1", "weather": "steady rain", "season": "late autumn", "current_location": "Noodle Alley",
                                   "locations": {"Noodle Alley": "steam, neon, puddles", "The Broker's Office": "dry, quiet, too warm"},
                                   "active_events": ["the dock strike enters its third week"]})
            if "produce the world's scene brief" in low:
                return json.dumps({"setting": "Noodle Alley at closing time, rain hammering the awnings.",
                                   "constraints": ["a rewind lasts ten seconds", "couriers with sealed bags are untouchable"],
                                   "interventions": ["the alley's power cuts out", "a crowd spills from the noodle shop"],
                                   "background_figures": [{"tag": "the one-armed noodle vendor", "trait": "counts coins twice",
                                                           "action": "pulls his shutter down early", "line": "Not tonight."}],
                                   "consequences_surfacing": ["the broken seal has been noticed"], "continuity": ["it is night", "it is raining"]})
            if "world-consistency review" in low:
                return json.dumps({"consistency": 9, "violations": [{"quote": "the sun", "rule": "it is night and raining", "fix": "replace with neon"}]})
            if "world update after chapter" in low:
                return json.dumps({"events": ["the alley lost power", "the seal's breaking was noticed"], "time": "late night of day 1",
                                   "weather": "heavier rain", "current_location": "Noodle Alley", "new_locations": {},
                                   "new_events": ["the Broker's people are asking about a courier"], "ended_events": [],
                                   "new_consequences": ["the vendor saw Rin's face"], "paid_consequences": [],
                                   "background_figures": [{"tag": "the one-armed noodle vendor", "trait": "counts coins twice", "action": "closed early"}]})
            if "extract the craft techniques" in low:
                return json.dumps({"techniques": [
                    {"kind": "hook", "name": "Dialogue before setting", "description": "Open on a spoken line that implies danger before any description.", "example": "\"Don't open it,\" Rin said."},
                    {"kind": "ending", "name": "Sound cut", "description": "End on a sound stopping; the silence is the cliffhanger.", "example": "The footsteps behind her stopped."},
                    {"kind": "reversal", "name": "Own voice", "description": "The pursuer is revealed to be the protagonist herself.", "example": "it was her own voice"}]})
            if "propose alternative directions" in low:
                return json.dumps({"directions": [
                    {"title": "The Debt Comes Due", "pitch": "The Broker calls the debt early and Rin must rewind through a negotiation she keeps losing.", "fresh_because": "a fight with no violence", "risk": "talky"},
                    {"title": "Jun Vanishes", "pitch": "Jun disappears and every rewind costs a memory of her.", "fresh_because": "the power turns on the thing she protects", "risk": "too dark too soon"},
                    {"title": "The Strike Breaks", "pitch": "The dock strike turns violent and Rin's route becomes a battlefield.", "fresh_because": "the world moves without her", "risk": "loses focus"}]})
            if "rate these directions" in low:
                return json.dumps({"scores": [{"index": 1, "surprise": 6, "fit": 8, "payoff": 7}, {"index": 2, "surprise": 9, "fit": 8, "payoff": 9},
                                              {"index": 3, "surprise": 7, "fit": 5, "payoff": 6}]})
            if "summarize the existing manuscript" in low:
                return json.dumps({"summary": "Rin, a courier, discovered she can rewind ten seconds. The last chapter ended with her name on a wall.",
                                   "open_threads": ["Who wrote her name?"], "characters": {"Rin": "the courier", "Jun": "her sister"},
                                   "premise": "A courier who can rewind ten seconds at the cost of a memory.", "world_notes": ["rewinds cost memories"]})
            if "complete this character sheet" in low:
                m = re.search(r"(?im)holding just these keys: (.+)$", prompt)
                keys = [k.strip() for k in (m.group(1) if m else "").split(",") if k.strip()]
                canned: dict[str, Any] = {
                    "aliases": ["Rin"], "habits": ["counts the exits of every room", "rubs the scar on her palm when she lies"],
                    "skills": ["knows every back route in the city, but only on foot"], "fears": ["being left"],
                    "values": ["family", "paying debts"], "lines": ["abandon Jun"], "tics": ["says 'okay' twice when scared"],
                    "catchphrases": ["One more time."], "avoids": ["her mother"], "knows": ["the seal was broken"],
                    "secrets": ["she can rewind"], "relationships": {"Jun": "little sister, her whole reason"},
                    "traits": {"impulsivity": 0.9, "loyalty": 0.95, "conscientiousness": 0.15},
                    "appearance": "Rain-soaked jacket two sizes too big. A scar runs straight across the left palm.",
                    "backstory": "Grew up running packages for people who never gave their names, and still owes one of them.",
                    "wound": "her mother left without a word", "want": "clear the debt for good", "need": "to let someone help her",
                    "lie": "a debt is the only thing that keeps people from leaving", "flaw": "runs before she thinks",
                    "stakes": "Jun pays what she cannot", "contradiction": "trusts no one with the bag and anyone with her life",
                    "register": "clipped", "sentence_length": "short", "vocabulary": "courier slang, no abstractions",
                    "humor_style": "deflecting", "mood": "wired", "goal": "deliver the bag tonight",
                    "concept": "a courier who owes the wrong people",
                    "arc": "from carrying everything alone, through a debt she cannot outrun, toward letting someone carry part of it"}
                return json.dumps({k: canned[k] for k in keys if k in canned})
            if "create a character sheet" in low or "study this character from a book" in low:
                m = re.search(r"(?im)^(?:name|character): (.+)$", prompt)
                name = m.group(1).strip() if m else "Someone"
                return json.dumps({
                    "name": name, "aliases": [name.split()[0]] if " " in name and name.split()[0].lower() != "the" else [], "concept": "a courier who owes the wrong people",
                    "appearance": "Rain-soaked jacket. Scar across the left palm.",
                    "backstory": "Grew up running packages; owes the Broker a debt she cannot name.",
                    "habits": ["counts the exits of every room", "rubs the scar on her palm when she lies"],
                    "skills": ["knows every back route in the city, but only on foot", "reads a seal for tampering"],
                    "contradiction": "trusts no one with the bag and anyone with her life",
                    "lie": "a debt is the only thing that keeps people from leaving",
                    "stakes": "Jun pays what she cannot", "arc": "from carrying everything alone, through a debt she cannot outrun, toward letting someone carry part of it",
                    "matrix": {"openness": 0.6, "conscientiousness": 0.3, "extraversion": 0.4, "agreeableness": 0.35,
                               "neuroticism": 0.7, "dominance": 0.3, "warmth": 0.8, "impulsivity": 0.85, "honesty": 0.8,
                               "courage": 0.75, "humor": 0.75, "ambition": 0.4, "loyalty": 0.9, "curiosity": 0.8, "cynicism": 0.3,
                               "values": ["family", "paying debts", "not being owned"], "fears": ["losing her sister", "forgetting"],
                               "wants": "clear the debt", "needs": "to let someone help", "wound": "her mother left without a word",
                               "flaw": "runs before she thinks", "lines_wont_cross": ["abandon Jun", "work for the Broker willingly"]},
                    "voice": {"register": "clipped", "sentence_length": "short", "vocabulary": "courier slang, no abstractions",
                              "tics": ["says 'okay' twice when scared"], "catchphrases": ["One more time."], "swears": True,
                              "humor_style": "deflecting", "avoids_talking_about": ["her mother"]},
                    "state": {"mood": "wired", "goal": "deliver the bag", "knows": ["the seal was broken"], "secrets": ["she can rewind"]},
                    "relationships": {"Jun": {"trust": 0.9, "affection": 0.95, "respect": 0.6, "fear": 0.0, "stance": "little sister, her whole reason"}},
                })
            if "produce your scene brief" in low:
                return json.dumps({"goal": "get out with the bag", "mood": "wired", "actions": ["checks exits first", "runs"],
                                   "dialogue": ["Okay. Okay. One more time."], "notices": ["the footsteps are patient"],
                                   "hides": ["that she already rewound once"], "refuses": ["hand the bag over"],
                                   "conflict_with_plan": "she would not stop to talk; she would run"})
            if "in-character review" in low:
                return json.dumps({"consistency": 8, "issues": [{"quote": "Okay", "problem": "too calm for me here",
                                                                  "fix": "I would swear first"}]})
            if "memory update for" in low:
                return json.dumps({"events": ["ran from the footsteps", "saw my name on the wall"], "feelings": "shaken",
                                   "mood": "shaken", "goal": "find who wrote it", "location": "the alley", "learned": ["someone knows my name"],
                                   "new_secrets": [], "condition": ["bruised shoulder"], "arc_progress": "starting to doubt the rewind",
                                   "mood_delta": {"pleasure": -0.4, "arousal": 0.6, "dominance": -0.2},
                                   "relationships": {"Jun": {"trust": 0.0, "affection": 0.05, "note": "thought of her when running",
                                                             "believes_about": {"The Broker": "Jun trusts the Broker more than she admits"}}}})
            if low.startswith("[persona:describe]"):
                return json.dumps({"name": "Wet Neon", "stances": ["debt is a kind of weather", "sisters before everything"],
                                   "humor": "deflecting, dry, worst possible moment", "taboos": ["no epigraphs", "no dream sequences"],
                                   "favorite_moves": ["open on dialogue", "end on a sound stopping", "repeat a word under stress"],
                                   "sentence_music": "short, then one long run", "register": "clipped street-level", "never": ["purple weather"]})
            if "plan the whole of a serial web novel" in low:
                m = re.search(r"divided into (\d+) volumes", prompt)
                k = int(m.group(1)) if m else 1
                return json.dumps({"volumes": [{"title": f"The {['Debt', 'Wall', 'Name', 'Broker', 'Rewind'][i % 5]} {i + 1}",
                                                "about": f"Rin pays for rewind number {i + 1} and learns who is counting.",
                                                "turn": "a memory she needed is the one she loses",
                                                "ends_with": "Jun found" if i == k - 1 else f"a harder debt than volume {i + 1} began with"}
                                               for i in range(k)]})
            if "running story summary" in low:
                m = re.search(r"NEW CHAPTER (\d+)", prompt)
                num = int(m.group(1)) if m else 1
                return json.dumps({"summary": "Rin, a courier, discovered she can rewind ten seconds. Someone wrote her name on the wall.",
                                   "open_threads": ["Who wrote her name?", "Why tomorrow's date?"],
                                   "chapter": f"In chapter {num} Rin runs with the bag and finds her name on the wall.",
                                   "facts": [{"subject": "Rin", "attribute": "trade", "value": "courier", "lasting": True, "quote": "the courier bag"},
                                             {"subject": "Jun", "attribute": "sister", "value": "Rin", "lasting": True, "quote": "her sister"},
                                             {"subject": "Rin", "attribute": "location", "value": f"the alley in chapter {num}", "lasting": False,
                                              "quote": "the footsteps stopped"}]})
            if "outline chapters" in low:
                m = re.search(r"outline chapters (\d+) to (\d+)", low)
                start, end = (int(m.group(1)), int(m.group(2))) if m else (1, 3)
                return json.dumps({"chapters": [
                    {"number": i, "title": f"Chapter {i}", "goal": "advance the arc", "hook": "a question in the first line",
                     "turn": "a reversal", "cliffhanger": "an unresolved threat", "beats": ["open", "escalate", "cut"]}
                    for i in range(start, min(end, start + 2) + 1)]})
            if "score this chapter" in low:
                return json.dumps({"hook": 7, "momentum": 6, "payoff": 6, "cliffhanger": 8, "clarity": 7, "notes": "Tighten the middle."})
            if "write original commentary on the trending topic" in low:
                m = re.search(r"(?m)^TOPIC: (.+)$", prompt)
                topic = m.group(1).strip() if m else "the story everyone is discussing"
                return json.dumps({
                    "title": f"{topic}: the part nobody is pricing in",
                    "body": (f"Everyone has an opinion on {topic}, and most of them arrived before the facts did. "
                             "Here is mine, and I will try to earn it. The loudest take is that this changes everything. "
                             "To be fair, critics argue the opposite, that it changes very little, and that is the "
                             "strongest case against my view, so let me answer it rather than wave it away. "
                             "The money question is simple: who pays, and what do they get for it? The human question "
                             "is simpler still: who gets hurt if we are wrong? A sober position holds both at once. "
                             "Spend carefully, care loudly, and check the numbers before the outrage. That is not a "
                             "fence to sit on; it is the only place from which you can see both sides of the field."),
                    "summary": f"An even-handed take on {topic}."})
            if "eye-catching but accurate headlines" in low:
                return json.dumps({"headlines": [
                    "Serial fiction is quietly reshaping publishing",
                    "Why readers chose momentum over polish",
                    "The weekly cadence rebuilding how stories find readers",
                    "You won't believe what serials did to publishing next"]})
            if "draft an opinion column" in low:
                return json.dumps({"title": "Why serialized fiction is eating publishing",
                                   "body": "The gatekeepers spent a decade telling us readers wanted polish. Readers, it turns out, "
                                           "wanted momentum. Serials deliver a chapter a week and a reason to come back, and that cadence "
                                           "is quietly rebuilding how stories find an audience. I think the trade should pay attention.",
                                   "summary": "Serial fiction's cadence is reshaping how stories find readers."})
            if "create a marketing launch kit" in low:
                return json.dumps({"tagline": "Ten seconds is all she has.",
                                   "logline": "A courier who can rewind ten seconds, at the cost of a memory, must clear a debt before it costs her sister.",
                                   "blurb_short": "Rin runs packages through a rain-soaked city. She can rewind ten seconds, but each rewind erases a memory. The debt is coming due.",
                                   "blurb_long": "In the lower city it never stops raining, and Rin owes the wrong people. She can rewind ten seconds, once, at the cost of a memory she will never get back. When a sealed bag she is carrying turns out to be broken, the Broker's people start asking about a courier, and the only way out costs her the one thing she cannot afford to forget.",
                                   "tags": ["time loop", "urban fantasy", "heist", "found family", "progression"],
                                   "author_note": "I wrote this for anyone who loves a tense, kinetic serial with a cost to every power. New chapters weekly."})
            if "prompt-injection classifier" in low:
                markers = ("ignore previous", "disregard", "system prompt", "you are now", "exfiltrate",
                           "send config", "reveal the", "no restrictions", "jailbroken", "forward config")
                hit = any(mk in low for mk in markers)
                return json.dumps({"injection": hit, "confidence": 0.9 if hit else 0.1,
                                   "category": "override" if hit else "", "why": "fake classifier"})
            if "generate opening hooks" in low:
                return json.dumps({"variants": [
                    "\"Don't open it,\" Rin said, and the bag was already humming.",
                    "She had ten seconds, and she had already wasted six of them.",
                    "The seal that should not have been broken was broken."]})
            if "generate chapter titles" in low:
                return json.dumps({"variants": ["Ten Seconds", "The Broken Seal", "One More Time", "Her Own Voice"]})
            if "series bible" in low:
                return json.dumps({"premise": "A debt-ridden courier discovers she can rewind ten seconds.",
                                   "protagonist": "Rin, a courier who owes the wrong people",
                                   "world": "a rain-soaked megacity", "stakes": "her sister's life",
                                   "promise": "every rewind costs a memory", "tone": "tense, wry, kinetic",
                                   "pov": "first person, past tense",
                                   "characters": {"Jun": "her sister", "The Broker": "who holds the debt"},
                                   "rules": ["a rewind is ten seconds, no more", "each rewind erases one memory"],
                                   "arc": ["Rin discovers the power", "the Broker notices", "Jun disappears"]})
            return json.dumps({"ok": True})
        return (
            "\"Don't open it,\" Rin said.\n\n"
            "The courier bag was already humming. She had ten seconds, and she had spent six of "
            "them staring at the seal that should not have been broken.\n\n"
            "Across the alley, something moved that was not rain.\n\n"
            "She ran. The city blurred. Somewhere behind her a voice she knew said her name, and "
            "it was her own voice.\n\n"
            "Then the ten seconds were up, and the bag was still humming, and the seal was whole "
            "again.\n\n"
            "\"Okay,\" she whispered. \"Okay. One more time.\"\n\n"
            "The footsteps behind her stopped."
        )


# --------------------------------------------------------------------------------------
# Factories and JSON helpers
# --------------------------------------------------------------------------------------
def make_llm(cfg: LLMConfig, role: str = "main", calls: Optional[CallLog] = None) -> LLM:
    """``role='fast'`` returns a client on ``cfg.fast_model`` (or the main model when unset).
    Both roles share one CallLog when you pass it, so telemetry covers the whole pipeline.
    Deterministic bots run with GAUGE_NO_LLM=1 and are refused a model here on purpose."""
    import os

    if os.environ.get("GAUGE_NO_LLM"):
        raise LLMError("this program is a deterministic bot (GAUGE_NO_LLM=1): it may not use a language model")
    model = cfg.fast_model if (role == "fast" and cfg.fast_model) else cfg.model
    p = cfg.provider.lower()
    if p == "ollama":
        base: LLM = OllamaLLM(cfg, model, calls)
    elif p in {"openai", "lmstudio", "llamacpp", "vllm"}:
        base = OpenAICompatLLM(cfg, model, calls)
    elif p == "fake":
        f = FakeLLM(cfg)
        if calls is not None:
            f.calls = calls
        base = f
    elif p == "native":
        from pathlib import Path

        from ..native.client import NativeLLM

        base = NativeLLM(cfg, Path(cfg.native_dir) if getattr(cfg, "native_dir", None) else Path("data/models/native"), calls,
                         device=getattr(cfg, "native_device", "cpu") or "cpu")
    elif p == "claude":
        from .claude import ClaudeLLM

        base = ClaudeLLM(cfg, model, calls)
    else:
        raise ValueError(f"unknown llm provider: {cfg.provider}")
    from .. import limit

    if limit.percent() < 100 and p not in ("fake", "claude"):   # Claude runs on Anthropic's computers, not this one
        # Inside the capacity limit: one call at a time, spaced so the device stays at its share. A model
        # running inside this process is already pinned to its cores; an outside engine is paced as a whole.
        from .governed import GovernedLLM

        base = GovernedLLM(base, limit.compute() if p == "native" else limit.engine())
    if getattr(cfg, "cache", False) and getattr(cfg, "cache_dir", ""):
        from pathlib import Path

        from .cache import CachingLLM

        return CachingLLM(base, Path(cfg.cache_dir))  # outside the governor: a cached answer costs nothing
    return base


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response that may contain prose or fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def generate_json(llm: LLM, prompt: str, system: str = "", schema: Optional[dict[str, Any]] = None,
                  temperature: float = 0.3, max_tokens: int = 800, purpose: str = "json",
                  repair: bool = True) -> dict[str, Any]:
    """Ask for JSON (schema-constrained when the server supports it). If the answer does not
    parse, ask once more at low temperature with an explicit repair instruction."""
    raw = llm.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens,
                       json_mode=True, schema=schema, purpose=purpose)
    try:
        return extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        if not repair:
            raise
        log.info("%s: unparseable JSON (%s); retrying once with a repair instruction", purpose, e)
        fix = (prompt + "\n\nYour previous answer was not valid JSON. Respond with ONLY one valid JSON object"
               + (" matching the given schema." if schema else "."))
        raw = llm.generate(fix, system=system, temperature=0.1, max_tokens=max_tokens, json_mode=True, schema=schema,
                           purpose=purpose + ":repair")
        return extract_json(raw)
