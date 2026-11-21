"""GaugeLM: a GPT-style transformer in numpy, forward and backward written by hand.

No PyTorch, no TensorFlow, no outside model weights. Pre-norm blocks with causal multi-head
attention and a GELU MLP, tied input/output embeddings, learned positional embeddings, AdamW.
Small enough to train on a desktop CPU; honest about what that means (see docs/NATIVE.md).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    k = 0.7978845608028654
    u = k * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * k * (1.0 + 3 * 0.044715 * x * x)


def layernorm_fwd(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_bwd(dy: np.ndarray, cache):
    xhat, rstd, g = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    db = dy.reshape(-1, xhat.shape[-1]).sum(0)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


class GaugeLM:
    def __init__(self, cfg: ModelConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        C, V, T, L = cfg.n_embd, cfg.vocab_size, cfg.block_size, cfg.n_layer
        s = 0.02
        self.p: dict[str, np.ndarray] = {
            "wte": (rng.standard_normal((V, C)) * s).astype(np.float32),
            "wpe": (rng.standard_normal((T, C)) * s).astype(np.float32),
            "lnf_g": np.ones(C, np.float32), "lnf_b": np.zeros(C, np.float32),
        }
        for i in range(L):
            self.p[f"l{i}.ln1_g"] = np.ones(C, np.float32)
            self.p[f"l{i}.ln1_b"] = np.zeros(C, np.float32)
            self.p[f"l{i}.qkv"] = (rng.standard_normal((C, 3 * C)) * s).astype(np.float32)
            self.p[f"l{i}.qkv_b"] = np.zeros(3 * C, np.float32)
            self.p[f"l{i}.proj"] = (rng.standard_normal((C, C)) * s / math.sqrt(2 * L)).astype(np.float32)
            self.p[f"l{i}.proj_b"] = np.zeros(C, np.float32)
            self.p[f"l{i}.ln2_g"] = np.ones(C, np.float32)
            self.p[f"l{i}.ln2_b"] = np.zeros(C, np.float32)
            self.p[f"l{i}.fc"] = (rng.standard_normal((C, 4 * C)) * s).astype(np.float32)
            self.p[f"l{i}.fc_b"] = np.zeros(4 * C, np.float32)
            self.p[f"l{i}.fc2"] = (rng.standard_normal((4 * C, C)) * s / math.sqrt(2 * L)).astype(np.float32)
            self.p[f"l{i}.fc2_b"] = np.zeros(C, np.float32)
        self.mask = np.tril(np.ones((T, T), dtype=bool))

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    # ---- forward ---------------------------------------------------------------
    def _attn_fwd(self, x: np.ndarray, i: int, kv_cache: Optional[dict] = None):
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        qkv = x @ self.p[f"l{i}.qkv"] + self.p[f"l{i}.qkv_b"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        if kv_cache is not None:
            if f"k{i}" in kv_cache:
                k = np.concatenate([kv_cache[f"k{i}"], k], axis=2)
                v = np.concatenate([kv_cache[f"v{i}"], v], axis=2)
            kv_cache[f"k{i}"], kv_cache[f"v{i}"] = k, v
        S = k.shape[2]
        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(D)
        mask = self.mask[S - T:S, :S] if kv_cache is not None else self.mask[:T, :T]
        att = np.where(mask, att, -1e9)
        att = softmax(att)
        y = att @ v  # B,H,T,D
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = y @ self.p[f"l{i}.proj"] + self.p[f"l{i}.proj_b"]
        return out, (x, q, k, v, att, y)

    def _attn_bwd(self, dout: np.ndarray, cache, i: int, grads: dict) -> np.ndarray:
        x, q, k, v, att, y = cache
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        grads[f"l{i}.proj"] += y.reshape(-1, C).T @ dout.reshape(-1, C)
        grads[f"l{i}.proj_b"] += dout.reshape(-1, C).sum(0)
        dy = (dout @ self.p[f"l{i}.proj"].T).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        datt = dy @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dy
        dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
        dscores = dscores / math.sqrt(D)
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q
        dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)
        dqkv = np.concatenate([dq, dk, dv], axis=-1)
        grads[f"l{i}.qkv"] += x.reshape(-1, C).T @ dqkv.reshape(-1, 3 * C)
        grads[f"l{i}.qkv_b"] += dqkv.reshape(-1, 3 * C).sum(0)
        return dqkv @ self.p[f"l{i}.qkv"].T

    def forward(self, idx: np.ndarray, targets: Optional[np.ndarray] = None, kv_cache: Optional[dict] = None, pos_offset: int = 0):
        """idx: (B,T) ints. Returns (logits, loss, caches)."""
        B, T = idx.shape
        pos = np.arange(pos_offset, pos_offset + T)
        x = self.p["wte"][idx] + self.p["wpe"][pos][None, :, :]
        caches: list[Any] = [idx]
        for i in range(self.cfg.n_layer):
            h, ln1c = layernorm_fwd(x, self.p[f"l{i}.ln1_g"], self.p[f"l{i}.ln1_b"])
            a, ac = self._attn_fwd(h, i, kv_cache)
            x = x + a
            h2, ln2c = layernorm_fwd(x, self.p[f"l{i}.ln2_g"], self.p[f"l{i}.ln2_b"])
            pre = h2 @ self.p[f"l{i}.fc"] + self.p[f"l{i}.fc_b"]
            act = gelu(pre)
            m = act @ self.p[f"l{i}.fc2"] + self.p[f"l{i}.fc2_b"]
            x = x + m
            caches.append((ln1c, ac, ln2c, h2, pre, act))
        xf, lnfc = layernorm_fwd(x, self.p["lnf_g"], self.p["lnf_b"])
        caches.append(lnfc)
        logits = xf @ self.p["wte"].T
        loss = None
        if targets is not None:
            probs = softmax(logits.reshape(-1, self.cfg.vocab_size).astype(np.float64))
            flat_t = targets.reshape(-1)
            valid = flat_t >= 0
            n = max(1, int(valid.sum()))
            loss = float(-np.log(probs[np.arange(len(flat_t))[valid], flat_t[valid]] + 1e-12).sum() / n)
            caches.append((probs, flat_t, valid, n, xf))
        return logits, loss, caches

    # ---- backward --------------------------------------------------------------
    def backward(self, caches) -> dict[str, np.ndarray]:
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        probs, flat_t, valid, n, xf = caches[-1]
        lnfc = caches[-2]
        idx = caches[0]
        B, T = idx.shape
        C = self.cfg.n_embd
        dlogits = probs.copy()
        dlogits[np.arange(len(flat_t))[valid], flat_t[valid]] -= 1.0
        dlogits[~valid] = 0.0
        dlogits = (dlogits / n).astype(np.float32)
        grads["wte"] += dlogits.T @ xf.reshape(-1, C)
        dxf = (dlogits @ self.p["wte"]).reshape(B, T, C)
        dx, dg, db = layernorm_bwd(dxf, lnfc)
        grads["lnf_g"] += dg
        grads["lnf_b"] += db
        for i in reversed(range(self.cfg.n_layer)):
            ln1c, ac, ln2c, h2, pre, act = caches[1 + i]
            # MLP
            dm = dx
            grads[f"l{i}.fc2"] += act.reshape(-1, 4 * C).T @ dm.reshape(-1, C)
            grads[f"l{i}.fc2_b"] += dm.reshape(-1, C).sum(0)
            dact = dm @ self.p[f"l{i}.fc2"].T
            dpre = dact * gelu_grad(pre)
            grads[f"l{i}.fc"] += h2.reshape(-1, C).T @ dpre.reshape(-1, 4 * C)
            grads[f"l{i}.fc_b"] += dpre.reshape(-1, 4 * C).sum(0)
            dh2 = dpre @ self.p[f"l{i}.fc"].T
            dx2, dg, db = layernorm_bwd(dh2, ln2c)
            grads[f"l{i}.ln2_g"] += dg
            grads[f"l{i}.ln2_b"] += db
            dx = dx + dx2
            # attention
            da = dx
            dh = self._attn_bwd(da, ac, i, grads)
            dx1, dg, db = layernorm_bwd(dh, ln1c)
            grads[f"l{i}.ln1_g"] += dg
            grads[f"l{i}.ln1_b"] += db
            dx = dx + dx1
        np.add.at(grads["wte"], idx.reshape(-1), dx.reshape(-1, C))
        np.add.at(grads["wpe"], np.tile(np.arange(T), B), dx.reshape(-1, C))
        return grads

    # ---- generation ------------------------------------------------------------
    def generate(self, idx: list[int], max_new_tokens: int, temperature: float = 1.0, top_k: int = 40, top_p: float = 0.95,
                 repetition_penalty: float = 1.15, stop_tokens: Optional[set[int]] = None, rng: Optional[np.random.Generator] = None,
                 banned: Optional[set[int]] = None, min_tokens: int = 0) -> list[int]:
        rng = rng or np.random.default_rng()
        T = self.cfg.block_size
        out = list(idx)
        kv: dict = {}
        # Prime the cache with the prompt (truncated to the last block_size-1 tokens).
        ctx = out[-(T - 1):]
        logits, _, _ = self.forward(np.asarray([ctx]), kv_cache=kv, pos_offset=0)
        pos = len(ctx)
        for produced in range(max_new_tokens):
            lg = logits[0, -1].astype(np.float64)
            if stop_tokens and produced < min_tokens:
                for t in stop_tokens:
                    lg[t] = -1e9
            if repetition_penalty > 1.0:
                recent = set(out[-64:])
                for t in recent:
                    lg[t] = lg[t] / repetition_penalty if lg[t] > 0 else lg[t] * repetition_penalty
            if banned:
                for t in banned:
                    lg[t] = -1e9
            if temperature <= 0.01:
                nxt = int(lg.argmax())
            else:
                lg = lg / temperature
                if top_k and top_k < len(lg):
                    cut = np.sort(lg)[-top_k]
                    lg = np.where(lg < cut, -1e9, lg)
                p = softmax(lg)
                if 0 < top_p < 1.0:
                    order = np.argsort(-p)
                    cum = np.cumsum(p[order])
                    keep = cum - p[order] < top_p
                    mask = np.zeros_like(p, dtype=bool)
                    mask[order[keep]] = True
                    p = np.where(mask, p, 0.0)
                    p = p / p.sum()
                nxt = int(rng.choice(len(p), p=p))
            out.append(nxt)
            if stop_tokens and nxt in stop_tokens:
                break
            if pos >= T:
                # Context full: re-prime with the last block of tokens.
                kv = {}
                ctx = out[-(T - 1):]
                logits, _, _ = self.forward(np.asarray([ctx]), kv_cache=kv, pos_offset=0)
                pos = len(ctx)
            else:
                logits, _, _ = self.forward(np.asarray([[nxt]]), kv_cache=kv, pos_offset=pos)
                pos += 1
        return out

    # ---- persistence -----------------------------------------------------------
    def save(self, directory: Path, extra: Optional[dict[str, Any]] = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "weights.npz", **self.p)
        meta = {"config": asdict(self.cfg), "n_params": self.n_params(), **(extra or {})}
        (directory / "model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> GaugeLM:
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        m = cls(ModelConfig(**meta["config"]))
        with np.load(directory / "weights.npz") as z:
            for k in m.p:
                m.p[k] = z[k]
        return m


class AdamW:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 3e-4, betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: Optional[float] = None, clip: float = 1.0) -> float:
        self.t += 1
        lr = self.lr if lr is None else lr
        norm = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-6)) if clip else 1.0
        for k, p in params.items():
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            decay = self.wd if p.ndim >= 2 and not k.endswith(("_g", "_b")) else 0.0
            params[k] = (p - lr * (mhat / (np.sqrt(vhat) + self.eps) + decay * p)).astype(np.float32)
        return norm
