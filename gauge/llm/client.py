"""Local LLM clients.

Gauge never talks to a paid API by default. It expects a local server:

* **Ollama** (default) at http://localhost:11434 - one installer on Windows, GPU-accelerated.
* Any **OpenAI-compatible** server (LM Studio, llama.cpp ``llama-server``, vLLM, text-gen-webui).
* ``FakeLLM`` for tests and for running the analysis pipeline on machines with no GPU.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Protocol

import requests

from ..config import LLMConfig


class LLM(Protocol):
    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False) -> str: ...

    def available(self) -> bool: ...

    def describe(self) -> str: ...


class OllamaLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.base = cfg.base_url.rstrip("/")

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=5)
            return r.ok
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
        return f"ollama {self.cfg.model} @ {self.base}"

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_predict": max_tokens or self.cfg.max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{self.base}/api/generate", json=payload, timeout=self.cfg.timeout_seconds)
        r.raise_for_status()
        return r.json().get("response", "")


class OpenAICompatLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
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
        return f"openai-compatible {self.cfg.model} @ {self.base}"

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(f"{self.base}/chat/completions", json=payload, headers=self._headers(),
                          timeout=self.cfg.timeout_seconds)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class FakeLLM:
    """Deterministic stand-in. Produces structurally valid output so pipelines can be tested."""

    def __init__(self, cfg: Optional[LLMConfig] = None, canned: Optional[list[str]] = None):
        self.cfg = cfg or LLMConfig(provider="fake")
        self.calls: list[dict[str, Any]] = []
        self.canned = list(canned or [])

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return "fake llm (offline)"

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        if self.canned:
            return self.canned.pop(0)
        if json_mode:
            low = prompt.lower()
            if "running story summary" in low:
                return json.dumps({"summary": "Rin rewound time and saw her own name on a wall.",
                                   "open_threads": ["Who wrote her name?", "Why tomorrow's date?"]})
            if "outline chapters" in low:
                m = re.search(r"outline chapters (\d+) to (\d+)", low)
                start, end = (int(m.group(1)), int(m.group(2))) if m else (1, 3)
                return json.dumps({
                    "chapters": [
                        {"number": i, "title": f"Chapter {i}", "goal": "advance the arc",
                         "hook": "a question in the first line", "turn": "a reversal",
                         "cliffhanger": "an unresolved threat", "beats": ["open", "escalate", "cut"]}
                        for i in range(start, min(end, start + 2) + 1)
                    ]
                })
            if "score this chapter" in low:
                return json.dumps({"hook": 7, "momentum": 6, "payoff": 6, "cliffhanger": 8,
                                   "clarity": 7, "notes": "Tighten the middle."})
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


def make_llm(cfg: LLMConfig) -> LLM:
    p = cfg.provider.lower()
    if p == "ollama":
        return OllamaLLM(cfg)
    if p in {"openai", "lmstudio", "llamacpp", "vllm"}:
        return OpenAICompatLLM(cfg)
    if p == "fake":
        return FakeLLM(cfg)
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
