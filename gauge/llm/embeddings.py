"""Embeddings for the exemplar library (retrieval of passages that *worked*).

Ollama's ``nomic-embed-text`` is used when available. Otherwise a hashing bag-of-ngrams embedder
keeps everything working offline; it is worse at semantics but fine for style retrieval.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np
import requests

from ..config import LLMConfig


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def describe(self) -> str: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def describe(self) -> str:
        return f"hashing-ngram({self.dim})"

    def _tokens(self, text: str) -> list[str]:
        ws = re.findall(r"[a-z']+", text.lower())
        grams = ws + [f"{a}_{b}" for a, b in zip(ws, ws[1:])]
        return grams

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
                sign = 1.0 if (h >> 63) else -1.0
                out[i, h % self.dim] += sign
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class OllamaEmbedder:
    def __init__(self, cfg: LLMConfig, dim: int = 768):
        self.cfg = cfg
        self.base = cfg.base_url.rstrip("/")
        self.dim = dim

    def describe(self) -> str:
        return f"ollama {self.cfg.embed_model}"

    def available(self) -> bool:
        try:
            r = requests.post(f"{self.base}/api/embed",
                              json={"model": self.cfg.embed_model, "input": "ping"}, timeout=30)
            if r.ok:
                self.dim = len(r.json()["embeddings"][0])
                return True
            return False
        except (requests.RequestException, KeyError, IndexError):
            return False

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        r = requests.post(f"{self.base}/api/embed",
                          json={"model": self.cfg.embed_model, "input": texts},
                          timeout=self.cfg.timeout_seconds)
        r.raise_for_status()
        arr = np.asarray(r.json()["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def make_embedder(cfg: LLMConfig, prefer_local_model: bool = True) -> Embedder:
    if prefer_local_model and cfg.provider == "ollama":
        e = OllamaEmbedder(cfg)
        if e.available():
            return e
    return HashingEmbedder()
