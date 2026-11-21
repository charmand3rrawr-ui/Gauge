"""Measuring GaugeLM: speed for a size on this machine, held-out perplexity, instruction shape."""
from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .backend import device_name
from .model import PRESETS, GaugeLM, ModelConfig
from .tokenizer import BPETokenizer
from .train import NativeTrainer, batches, preset_config


def bench(size: str = "tiny", device: str = "cpu", steps: int = 3, batch_size: int = 4, vocab_size: int = 2048) -> dict[str, Any]:
    """Tokens/second for training and generation at a preset size, and what a real run would take."""
    cfg = preset_config(size, vocab_size)
    model = GaugeLM(cfg, device=device)
    xp = model.xp
    rng = np.random.default_rng(0)
    data = rng.integers(0, cfg.vocab_size, size=cfg.block_size * (batch_size + 2) + 2).astype(np.int32)
    gen = batches(data, cfg.block_size, batch_size, rng)
    x, y = next(gen)
    model.forward(x, y)  # warm-up
    t0 = time.monotonic()
    for _ in range(steps):
        x, y = next(gen)
        _, _, caches = model.forward(x, y)
        model.backward(caches)
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    train_tps = steps * x.size / max(1e-6, time.monotonic() - t0)
    t0 = time.monotonic()
    out = model.generate([1, 2, 3], 24, temperature=0.9, rng=rng)
    gen_tps = (len(out) - 3) / max(1e-6, time.monotonic() - t0)
    per_step_tokens = batch_size * cfg.block_size
    steps_2k_hours = 2000 * per_step_tokens / train_tps / 3600
    return {"size": size, "device": device_name(xp), "n_params": model.n_params(), "block_size": cfg.block_size,
            "train_tokens_per_s": round(train_tps), "generate_tokens_per_s": round(gen_tps, 1),
            "hours_for_2000_steps_at_batch": {str(batch_size): round(steps_2k_hours, 2)},
            "note": "generation speed is per token; a 2,000-word chapter is roughly 2,700 tokens"}


def perplexity(model: GaugeLM, tok: BPETokenizer, texts: list[str], max_tokens: int = 20_000) -> dict[str, float]:
    ids: list[int] = []
    for t in texts:
        ids.extend(tok.encode(t, bos=True, eos=True))
        if len(ids) >= max_tokens:
            break
    data = np.asarray(ids[:max_tokens], dtype=np.int32)
    block = model.cfg.block_size
    if len(data) < block + 2:
        return {"tokens": len(data), "loss": float("nan"), "perplexity": float("nan")}
    losses = []
    for i in range(0, len(data) - block - 1, block):
        x = data[i:i + block][None, :]
        y = data[i + 1:i + block + 1][None, :]
        losses.append(model.forward(x, y)[1])
    loss = float(np.mean(losses))
    return {"tokens": int(len(data)), "loss": round(loss, 4), "perplexity": round(math.exp(min(20, loss)), 2)}


PROBES = [
    ("Write one line of dialogue in which a character refuses to open a door.", r"[\"“].+[\"”]"),
    ("Name the chapter in three words:", r"^\S+(\s+\S+){0,4}$"),
    ("Give a number from 1 to 10 for how tense this scene is:", r"\b(10|[1-9])\b"),
    ("Write the first sentence of a chapter that starts in the middle of a chase.", r"\w+.*[.!?]"),
]


def instruction_probe(llm) -> dict[str, Any]:
    """Does the model answer in the right *shape*? Cheap proxy for instruction following."""
    hits = 0
    details = []
    for prompt, pattern in PROBES:
        out = llm.generate(prompt, max_tokens=40, temperature=0.7, purpose="eval").strip()
        ok = bool(re.search(pattern, out, re.S | re.M))
        hits += int(ok)
        details.append({"prompt": prompt[:50], "answer": out[:80], "ok": ok})
    return {"score": round(hits / len(PROBES), 2), "details": details}


def evaluate_model(model_dir: Path, texts: list[str], llm=None, device: str = "cpu") -> dict[str, Any]:
    tr = NativeTrainer(model_dir)
    model, tok = tr.load(device=device)
    out: dict[str, Any] = {"n_params": model.n_params(), "arch": model.cfg.arch, "device": device_name(model.xp),
                           "held_out": perplexity(model, tok, texts)}
    if llm is not None:
        out["instructions"] = instruction_probe(llm)
    return out


def preset_table() -> list[dict[str, Any]]:
    rows = []
    for name, p in PRESETS.items():
        cfg = ModelConfig(**p)
        n = GaugeLM(ModelConfig(**{**p, "vocab_size": 256}), seed=0).n_params() if False else _count(cfg)
        rows.append({"size": name, **p, "params_millions": round(n / 1e6, 1)})
    return rows


def _count(cfg: ModelConfig) -> int:
    C, V, L, Hd, Hk, D = cfg.n_embd, cfg.vocab_size, cfg.n_layer, cfg.hidden, cfg.kv_heads, cfg.head_dim
    per_layer = C * C + 2 * C * Hk * D + C * C + 3 * C * Hd + 2 * C
    return V * C + C + L * per_layer
