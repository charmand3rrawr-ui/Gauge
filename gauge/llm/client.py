"""Local LLM clients.

Gauge never talks to a paid API by default. It expects a local server:

* **Ollama** (default) at http://localhost:11434 - one installer on Windows, GPU-accelerated.
* Any **OpenAI-compatible** server (LM Studio, llama.cpp ``llama-server``, vLLM, text-gen-webui).
* ``FakeLLM`` for tests and for running the analysis pipeline on machines with no GPU.

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
        if "stay in character as" in low:
            return "Okay. Okay. You want the truth? I ran because the footsteps knew my name. One more time."
        if "answer as the world" in low:
            return "Rain since dusk. The market is closing; the vendor with one arm is still counting coins."
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
            if "create a character sheet" in low:
                m = re.search(r"(?im)^name: (.+)$", prompt)
                name = m.group(1).strip() if m else "Someone"
                return json.dumps({
                    "name": name, "aliases": [name.split()[0]] if " " in name and name.split()[0].lower() != "the" else [], "concept": "a courier who owes the wrong people",
                    "appearance": "Rain-soaked jacket. Scar across the left palm.",
                    "backstory": "Grew up running packages; owes the Broker a debt she cannot name.",
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
                                   "relationships": {"Jun": {"trust": 0.0, "affection": 0.05, "note": "thought of her when running"}}})
            if "running story summary" in low:
                return json.dumps({"summary": "Rin rewound time and saw her own name on a wall.",
                                   "open_threads": ["Who wrote her name?", "Why tomorrow's date?"]})
            if "outline chapters" in low:
                m = re.search(r"outline chapters (\d+) to (\d+)", low)
                start, end = (int(m.group(1)), int(m.group(2))) if m else (1, 3)
                return json.dumps({"chapters": [
                    {"number": i, "title": f"Chapter {i}", "goal": "advance the arc", "hook": "a question in the first line",
                     "turn": "a reversal", "cliffhanger": "an unresolved threat", "beats": ["open", "escalate", "cut"]}
                    for i in range(start, min(end, start + 2) + 1)]})
            if "score this chapter" in low:
                return json.dumps({"hook": 7, "momentum": 6, "payoff": 6, "cliffhanger": 8, "clarity": 7, "notes": "Tighten the middle."})
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
    Both roles share one CallLog when you pass it, so telemetry covers the whole pipeline."""
    model = cfg.fast_model if (role == "fast" and cfg.fast_model) else cfg.model
    p = cfg.provider.lower()
    if p == "ollama":
        return OllamaLLM(cfg, model, calls)
    if p in {"openai", "lmstudio", "llamacpp", "vllm"}:
        return OpenAICompatLLM(cfg, model, calls)
    if p == "fake":
        f = FakeLLM(cfg)
        if calls is not None:
            f.calls = calls
        return f
    raise ValueError(f"unknown llm provider: {cfg.provider}")


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
