"""GaugeLM as an LLM provider for the pipeline (``llm.provider = "native"``).

A small model cannot be trusted to emit JSON, so structured requests are answered by
**schema-guided filling**: every field in the schema is generated on its own with a short budget,
typed by the schema (strings, lists split on separators, numbers parsed or defaulted), and the
object is assembled in code. Slower than one call, but it never produces unparseable output and
needs no outside library.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import LLMConfig
from ..llm.client import CallLog, CallRecord, LLMError
from ..llm.tokens import estimate_tokens
from .model import GaugeLM
from .tokenizer import BPETokenizer
from .train import NativeTrainer

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class NativeLLM:
    def __init__(self, cfg: LLMConfig, model_dir: Path, calls: Optional[CallLog] = None, model: Optional[GaugeLM] = None,
                 tokenizer: Optional[BPETokenizer] = None):
        self.cfg = cfg
        self.model_dir = model_dir
        self.model_name = "gaugelm"
        self.calls = calls or CallLog()
        self._lock = threading.Lock()
        self._model = model
        self._tok = tokenizer
        self.rng = np.random.default_rng()

    # ---- protocol ----------------------------------------------------------------
    @property
    def model(self) -> str:  # the pipeline reads .model as a name
        return self.model_name

    def _ensure(self) -> tuple[GaugeLM, BPETokenizer]:
        if self._model is None or self._tok is None:
            tr = NativeTrainer(self.model_dir)
            if not tr.exists():
                raise LLMError("no native model trained yet: run `gauge native train`")
            self._model, self._tok = tr.load()
        return self._model, self._tok

    def available(self) -> bool:
        try:
            self._ensure()
            return True
        except LLMError:
            return False

    def describe(self) -> str:
        st = NativeTrainer(self.model_dir).status()
        return f"native GaugeLM @ {self.model_dir} ({st.get('n_params') or 0:,} params, step {st.get('step', 0)})" if st.get("trained") else \
            f"native GaugeLM @ {self.model_dir} (not trained)"

    def context_window(self) -> int:
        try:
            return self._ensure()[0].cfg.block_size
        except LLMError:
            return 256

    def throughput(self, max_tokens: int = 100) -> dict[str, float]:
        t0 = time.monotonic()
        out = self.generate("Write the opening of a chapter.", max_tokens=max_tokens, purpose="bench")
        wall = time.monotonic() - t0
        return {"generation_tps": round(estimate_tokens(out) / max(0.001, wall), 1), "prompt_tps": 0.0, "wall_seconds": round(wall, 1), "load_seconds": 0.0}

    # ---- text --------------------------------------------------------------------
    def _complete(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        model, tok = self._ensure()
        full = (system + "\n\n" + prompt).strip() if system else prompt
        ids = tok.encode_chat(full)
        with self._lock:
            out = model.generate(ids, max_new_tokens=max_tokens, temperature=temperature, top_k=40, top_p=0.95,
                                 stop_tokens={tok.eos}, rng=self.rng, banned={tok.special["<prompt>"], tok.special["<response>"], tok.bos, tok.pad},
                                 min_tokens=min(24, max(1, max_tokens // 4)))
        return tok.decode(out[len(ids):]).strip()

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 json_mode: bool = False, schema: Optional[dict[str, Any]] = None, purpose: str = "generate") -> str:
        t0 = time.monotonic()
        temp = self.cfg.temperature if temperature is None else temperature
        budget = min(max_tokens or self.cfg.max_tokens, 1200)
        try:
            if schema is not None or json_mode:
                obj = self.fill_schema(schema or {"type": "object", "properties": {"text": {"type": "string"}}}, prompt, system, temp)
                out = json.dumps(obj, ensure_ascii=False)
            else:
                out = self._complete(prompt, system, budget, temp)
            self.calls.add(CallRecord(purpose, self.model_name, estimate_tokens(system) + estimate_tokens(prompt), estimate_tokens(out),
                                      int((time.monotonic() - t0) * 1000), True))
            return out
        except Exception:
            self.calls.add(CallRecord(purpose, self.model_name, estimate_tokens(prompt), 0, int((time.monotonic() - t0) * 1000), False))
            raise

    # ---- schema-guided structured output --------------------------------------------
    def fill_schema(self, schema: dict[str, Any], prompt: str, system: str = "", temperature: float = 0.7, depth: int = 0,
                    root: Optional[dict[str, Any]] = None) -> Any:
        root = root or schema
        schema = self._resolve(schema, root)
        t = schema.get("type")
        if t == "object" or "properties" in schema:
            obj: dict[str, Any] = {}
            for key, sub in list(schema.get("properties", {}).items())[:24]:
                obj[key] = self._fill_field(key, self._resolve(sub, root), prompt, system, temperature, depth, root)
            return obj
        return self._fill_field("value", schema, prompt, system, temperature, depth, root)

    def _fill_field(self, key: str, sub: dict[str, Any], prompt: str, system: str, temperature: float, depth: int,
                    root: dict[str, Any]) -> Any:
        t = sub.get("type")
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), "string")
        if (t == "object" or "properties" in sub) and depth < 2:
            if "properties" in sub:
                return self.fill_schema(sub, f"{prompt}\n\nNow the part called {key}.", system, temperature, depth + 1, root)
            return {}  # free-form dicts (e.g. relationships) are left for the author
        if t == "array":
            items = self._resolve(sub.get("items", {"type": "string"}), root)
            if (items.get("type") == "object" or "properties" in items) and depth < 2:
                count = 3 if key in {"chapters", "directions"} else 1
                return [self.fill_schema(items, f"{prompt}\n\nItem {i + 1} of {key}.", system, temperature, depth + 1, root)
                        for i in range(count)]
            text = self._complete(f"{prompt}\n\nList the {key.replace('_', ' ')}, separated by semicolons:", system, 60, temperature)
            parts = [p.strip(" -*\n\t\"'") for p in re.split(r"[;\n]", text) if p.strip(" -*\n\t\"'")]
            if items.get("type") in ("number", "integer"):
                return [self._num(p, items) for p in parts[:6]]
            return parts[:6]
        if t in ("number", "integer"):
            text = self._complete(f"{prompt}\n\nGive only a number for {key.replace('_', ' ')}:", system, 6, 0.3)
            return self._num(text, sub)
        if t == "boolean":
            text = self._complete(f"{prompt}\n\nAnswer yes or no for {key.replace('_', ' ')}:", system, 3, 0.3).lower()
            return "yes" in text or "true" in text
        # strings
        budget = 120 if key in {"premise", "backstory", "world", "pitch", "summary", "blurb", "appearance"} else 40
        text = self._complete(f"{prompt}\n\nThe {key.replace('_', ' ')}:", system, budget, temperature)
        return " ".join(text.split("\n")[0:3]).strip().strip('"')

    def _num(self, text: str, sub: dict[str, Any]) -> float | int:
        m = _NUM.search(text or "")
        default = sub.get("default", 5 if sub.get("type") == "integer" else 0.5)
        if not m:
            return default
        v = float(m.group(0))
        lo, hi = sub.get("minimum"), sub.get("maximum")
        if lo is not None:
            v = max(float(lo), v)
        if hi is not None:
            v = min(float(hi), v)
        return int(v) if sub.get("type") == "integer" else v

    @staticmethod
    def _resolve(sub: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in sub:
            name = sub["$ref"].split("/")[-1]
            return root.get("$defs", {}).get(name, {}) or {"type": "string"}
        if "anyOf" in sub:
            for opt in sub["anyOf"]:
                if opt.get("type") != "null":
                    return NativeLLM._resolve(opt, root)
        return sub
