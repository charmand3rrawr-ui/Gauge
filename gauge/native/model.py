"""GaugeLM: a transformer language model with forward and backward written by hand.

Two architectures live here, selected by ``ModelConfig.arch``:

* ``"llama"`` (default): RMSNorm, rotary position embeddings, SwiGLU feed-forward, grouped-query
  attention, no biases, tied embeddings. The design that small modern models use, because it
  trains faster and generalises better per parameter than the 2019 GPT recipe.
* ``"gpt"``: LayerNorm, learned positions, GELU, biases. Kept so earlier checkpoints still load.

Everything runs on numpy, or on CuPy when a CUDA GPU is present (``backend.py``). No outside model
code and no downloaded weights anywhere.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .backend import from_numpy, get_xp, to_numpy


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    n_kv_head: int = 0  # 0 = same as n_head (no grouping); 2 or 4 = grouped-query attention
    arch: str = "llama"  # "llama" | "gpt"
    rope_base: float = 10000.0
    mlp_ratio: float = 2.6667  # SwiGLU hidden = ratio * n_embd (Llama uses 8/3)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def hidden(self) -> int:
        return int(self.mlp_ratio * self.n_embd) if self.arch == "llama" else 4 * self.n_embd


PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {"n_layer": 4, "n_head": 4, "n_embd": 256, "block_size": 256, "n_kv_head": 2, "vocab_size": 4096},
    "small": {"n_layer": 8, "n_head": 8, "n_embd": 512, "block_size": 512, "n_kv_head": 4, "vocab_size": 8192},
    "medium": {"n_layer": 12, "n_head": 12, "n_embd": 768, "block_size": 1024, "n_kv_head": 4, "vocab_size": 16384},
    "large": {"n_layer": 24, "n_head": 16, "n_embd": 1024, "block_size": 2048, "n_kv_head": 8, "vocab_size": 32000},
    # Beyond what any home computer can hold. The arithmetic is the same; the machine is the limit (see sizing.py).
    # Head size stays 64 and each step is at most ten times the last, so a model can be grown from one to the next.
    "2b": {"n_layer": 32, "n_head": 40, "n_embd": 2560, "block_size": 4096, "n_kv_head": 20, "vocab_size": 32000},
    "20b": {"n_layer": 48, "n_head": 96, "n_embd": 6144, "block_size": 8192, "n_kv_head": 48, "vocab_size": 32000},
    "160b": {"n_layer": 96, "n_head": 192, "n_embd": 12288, "block_size": 16384, "n_kv_head": 96, "vocab_size": 65536},
    "1t": {"n_layer": 192, "n_head": 384, "n_embd": 24576, "block_size": 32768, "n_kv_head": 192, "vocab_size": 65536},
    "5t": {"n_layer": 256, "n_head": 640, "n_embd": 40960, "block_size": 32768, "n_kv_head": 320, "vocab_size": 131072},
    "20t": {"n_layer": 424, "n_head": 1024, "n_embd": 65536, "block_size": 65536, "n_kv_head": 512, "vocab_size": 131072},
}


# --------------------------------------------------------------------------------------
# Primitives (each with a backward)
# --------------------------------------------------------------------------------------
def gelu(xp, x):
    return 0.5 * x * (1.0 + xp.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def gelu_grad(xp, x):
    k = 0.7978845608028654
    t = xp.tanh(k * (x + 0.044715 * x ** 3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * k * (1.0 + 3 * 0.044715 * x * x)


def silu(xp, x):
    return x / (1.0 + xp.exp(-x))


def silu_grad(xp, x):
    s = 1.0 / (1.0 + xp.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def layernorm_fwd(xp, x, g, b, eps: float = 1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / xp.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_bwd(dy, cache):
    xhat, rstd, g = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    db = dy.reshape(-1, xhat.shape[-1]).sum(0)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def rmsnorm_fwd(xp, x, g, eps: float = 1e-5):
    rms = xp.sqrt((x * x).mean(-1, keepdims=True) + eps)
    xhat = x / rms
    return xhat * g, (xhat, rms, g)


def rmsnorm_bwd(dy, cache):
    xhat, rms, g = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    dxhat = dy * g
    dx = (dxhat - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / rms
    return dx, dg


def rope_tables(xp, T: int, D: int, base: float, offset: int = 0):
    inv = base ** (-xp.arange(0, D, 2, dtype=xp.float32) / D)
    pos = xp.arange(offset, offset + T, dtype=xp.float32)[:, None]
    ang = pos * inv[None, :]  # T, D/2
    return xp.cos(ang)[None, None], xp.sin(ang)[None, None]  # 1,1,T,D/2


def rope_apply(xp, x, cos, sin, inverse: bool = False):
    """x: B,H,T,D with D even; rotate pairs (x[2i], x[2i+1]). inverse rotates back (the gradient)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    if inverse:
        y1, y2 = x1 * cos + x2 * sin, -x1 * sin + x2 * cos
    else:
        y1, y2 = x1 * cos - x2 * sin, x1 * sin + x2 * cos
    out = xp.empty_like(x)
    out[..., 0::2], out[..., 1::2] = y1, y2
    return out


def softmax(xp, x, axis: int = -1):
    m = x.max(axis=axis, keepdims=True)
    e = xp.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# --------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------
class GaugeLM:
    def __init__(self, cfg: ModelConfig, seed: int = 0, device: str = "cpu"):
        self.cfg = cfg
        self.xp = get_xp(device)
        xp = self.xp
        rng = np.random.default_rng(seed)
        C, V, T, L, Hk, D = cfg.n_embd, cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.kv_heads, cfg.head_dim
        s = 0.02
        p: dict[str, np.ndarray] = {"wte": (rng.standard_normal((V, C)) * s).astype(np.float32)}
        if cfg.arch == "gpt":
            p["wpe"] = (rng.standard_normal((T, C)) * s).astype(np.float32)
            p["lnf_g"], p["lnf_b"] = np.ones(C, np.float32), np.zeros(C, np.float32)
            for i in range(L):
                p[f"l{i}.ln1_g"], p[f"l{i}.ln1_b"] = np.ones(C, np.float32), np.zeros(C, np.float32)
                p[f"l{i}.qkv"] = (rng.standard_normal((C, 3 * C)) * s).astype(np.float32)
                p[f"l{i}.qkv_b"] = np.zeros(3 * C, np.float32)
                p[f"l{i}.proj"] = (rng.standard_normal((C, C)) * s / math.sqrt(2 * L)).astype(np.float32)
                p[f"l{i}.proj_b"] = np.zeros(C, np.float32)
                p[f"l{i}.ln2_g"], p[f"l{i}.ln2_b"] = np.ones(C, np.float32), np.zeros(C, np.float32)
                p[f"l{i}.fc"] = (rng.standard_normal((C, 4 * C)) * s).astype(np.float32)
                p[f"l{i}.fc_b"] = np.zeros(4 * C, np.float32)
                p[f"l{i}.fc2"] = (rng.standard_normal((4 * C, C)) * s / math.sqrt(2 * L)).astype(np.float32)
                p[f"l{i}.fc2_b"] = np.zeros(C, np.float32)
        else:
            Hd = cfg.hidden
            p["lnf_g"] = np.ones(C, np.float32)
            for i in range(L):
                p[f"l{i}.ln1_g"] = np.ones(C, np.float32)
                p[f"l{i}.wq"] = (rng.standard_normal((C, C)) * s).astype(np.float32)
                p[f"l{i}.wk"] = (rng.standard_normal((C, Hk * D)) * s).astype(np.float32)
                p[f"l{i}.wv"] = (rng.standard_normal((C, Hk * D)) * s).astype(np.float32)
                p[f"l{i}.wo"] = (rng.standard_normal((C, C)) * s / math.sqrt(2 * L)).astype(np.float32)
                p[f"l{i}.ln2_g"] = np.ones(C, np.float32)
                p[f"l{i}.w1"] = (rng.standard_normal((C, Hd)) * s).astype(np.float32)  # gate
                p[f"l{i}.w3"] = (rng.standard_normal((C, Hd)) * s).astype(np.float32)  # up
                p[f"l{i}.w2"] = (rng.standard_normal((Hd, C)) * s / math.sqrt(2 * L)).astype(np.float32)  # down
        self.p = {k: from_numpy(xp, v) for k, v in p.items()}
        self.mask = from_numpy(xp, np.tril(np.ones((T, T), dtype=bool)))

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    @property
    def is_llama(self) -> bool:
        return self.cfg.arch == "llama"

    def _norm_fwd(self, x, name: str):
        if self.is_llama:
            return rmsnorm_fwd(self.xp, x, self.p[f"{name}_g"])
        return layernorm_fwd(self.xp, x, self.p[f"{name}_g"], self.p[f"{name}_b"])

    def _norm_bwd(self, dy, cache, name: str, grads):
        if self.is_llama:
            dx, dg = rmsnorm_bwd(dy, cache)
            grads[f"{name}_g"] += dg
            return dx
        dx, dg, db = layernorm_bwd(dy, cache)
        grads[f"{name}_g"] += dg
        grads[f"{name}_b"] += db
        return dx

    # ---- attention ---------------------------------------------------------------
    def _attn_fwd(self, x, i: int, kv_cache: Optional[dict], pos_offset: int):
        xp = self.xp
        B, T, C = x.shape
        H, Hk, D = self.cfg.n_head, self.cfg.kv_heads, self.cfg.head_dim
        if self.is_llama:
            q = (x @ self.p[f"l{i}.wq"]).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k = (x @ self.p[f"l{i}.wk"]).reshape(B, T, Hk, D).transpose(0, 2, 1, 3)
            v = (x @ self.p[f"l{i}.wv"]).reshape(B, T, Hk, D).transpose(0, 2, 1, 3)
            cos, sin = rope_tables(xp, T, D, self.cfg.rope_base, pos_offset)
            q_r, k_r = rope_apply(xp, q, cos, sin), rope_apply(xp, k, cos, sin)
        else:
            qkv = x @ self.p[f"l{i}.qkv"] + self.p[f"l{i}.qkv_b"]
            q, k, v = xp.split(qkv, 3, axis=-1)
            q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, Hk, D).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, Hk, D).transpose(0, 2, 1, 3)
            q_r, k_r, cos, sin = q, k, None, None
        if kv_cache is not None:
            if f"k{i}" in kv_cache:
                k_r = xp.concatenate([kv_cache[f"k{i}"], k_r], axis=2)
                v = xp.concatenate([kv_cache[f"v{i}"], v], axis=2)
            kv_cache[f"k{i}"], kv_cache[f"v{i}"] = k_r, v
        S = k_r.shape[2]
        rep = H // Hk
        k_full = xp.repeat(k_r, rep, axis=1) if rep > 1 else k_r
        v_full = xp.repeat(v, rep, axis=1) if rep > 1 else v
        att = (q_r @ k_full.transpose(0, 1, 3, 2)) / math.sqrt(D)
        mask = self.mask[S - T:S, :S] if kv_cache is not None else self.mask[:T, :T]
        att = xp.where(mask, att, xp.float32(-1e9))
        att = softmax(xp, att)
        y = (att @ v_full).transpose(0, 2, 1, 3).reshape(B, T, C)
        out = y @ (self.p[f"l{i}.wo"] if self.is_llama else self.p[f"l{i}.proj"])
        if not self.is_llama:
            out = out + self.p[f"l{i}.proj_b"]
        return out, (x, q_r, k_full, v_full, att, y, cos, sin, q)

    def _attn_bwd(self, dout, cache, i: int, grads):
        xp = self.xp
        x, q_r, k_full, v_full, att, y, cos, sin, _q = cache
        B, T, C = x.shape
        H, Hk, D = self.cfg.n_head, self.cfg.kv_heads, self.cfg.head_dim
        rep = H // Hk
        wo = self.p[f"l{i}.wo"] if self.is_llama else self.p[f"l{i}.proj"]
        grads[f"l{i}.wo" if self.is_llama else f"l{i}.proj"] += y.reshape(-1, C).T @ dout.reshape(-1, C)
        if not self.is_llama:
            grads[f"l{i}.proj_b"] += dout.reshape(-1, C).sum(0)
        dy = (dout @ wo.T).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        datt = dy @ v_full.transpose(0, 1, 3, 2)
        dv_full = att.transpose(0, 1, 3, 2) @ dy
        dscores = att * (datt - (datt * att).sum(-1, keepdims=True)) / math.sqrt(D)
        dq_r = dscores @ k_full
        dk_full = dscores.transpose(0, 1, 3, 2) @ q_r
        if rep > 1:  # grouped heads: sum gradients over the heads that shared each kv head
            dk_r = dk_full.reshape(B, Hk, rep, T, D).sum(2)
            dv = dv_full.reshape(B, Hk, rep, T, D).sum(2)
        else:
            dk_r, dv = dk_full, dv_full
        if self.is_llama:
            dq = rope_apply(xp, dq_r, cos, sin, inverse=True)
            dk = rope_apply(xp, dk_r, cos, sin, inverse=True)
            dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
            dk = dk.transpose(0, 2, 1, 3).reshape(B, T, Hk * D)
            dv = dv.transpose(0, 2, 1, 3).reshape(B, T, Hk * D)
            xf = x.reshape(-1, C)
            grads[f"l{i}.wq"] += xf.T @ dq.reshape(-1, C)
            grads[f"l{i}.wk"] += xf.T @ dk.reshape(-1, Hk * D)
            grads[f"l{i}.wv"] += xf.T @ dv.reshape(-1, Hk * D)
            return dq @ self.p[f"l{i}.wq"].T + dk @ self.p[f"l{i}.wk"].T + dv @ self.p[f"l{i}.wv"].T
        dq = dq_r.transpose(0, 2, 1, 3).reshape(B, T, C)
        dk = dk_r.transpose(0, 2, 1, 3).reshape(B, T, C)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)
        dqkv = xp.concatenate([dq, dk, dv], axis=-1)
        grads[f"l{i}.qkv"] += x.reshape(-1, C).T @ dqkv.reshape(-1, 3 * C)
        grads[f"l{i}.qkv_b"] += dqkv.reshape(-1, 3 * C).sum(0)
        return dqkv @ self.p[f"l{i}.qkv"].T

    # ---- feed-forward ---------------------------------------------------------------
    def _mlp_fwd(self, h, i: int):
        xp = self.xp
        if self.is_llama:
            a = h @ self.p[f"l{i}.w1"]
            b = h @ self.p[f"l{i}.w3"]
            act = silu(xp, a) * b
            return act @ self.p[f"l{i}.w2"], (h, a, b, act)
        pre = h @ self.p[f"l{i}.fc"] + self.p[f"l{i}.fc_b"]
        act = gelu(xp, pre)
        return act @ self.p[f"l{i}.fc2"] + self.p[f"l{i}.fc2_b"], (h, pre, None, act)

    def _mlp_bwd(self, dm, cache, i: int, grads):
        xp = self.xp
        h, a, b, act = cache
        C = h.shape[-1]
        Hd = act.shape[-1]
        if self.is_llama:
            grads[f"l{i}.w2"] += act.reshape(-1, Hd).T @ dm.reshape(-1, C)
            dact = dm @ self.p[f"l{i}.w2"].T
            da = dact * b * silu_grad(xp, a)
            db = dact * silu(xp, a)
            hf = h.reshape(-1, C)
            grads[f"l{i}.w1"] += hf.T @ da.reshape(-1, Hd)
            grads[f"l{i}.w3"] += hf.T @ db.reshape(-1, Hd)
            return da @ self.p[f"l{i}.w1"].T + db @ self.p[f"l{i}.w3"].T
        grads[f"l{i}.fc2"] += act.reshape(-1, Hd).T @ dm.reshape(-1, C)
        grads[f"l{i}.fc2_b"] += dm.reshape(-1, C).sum(0)
        dpre = (dm @ self.p[f"l{i}.fc2"].T) * gelu_grad(xp, a)
        grads[f"l{i}.fc"] += h.reshape(-1, C).T @ dpre.reshape(-1, Hd)
        grads[f"l{i}.fc_b"] += dpre.reshape(-1, Hd).sum(0)
        return dpre @ self.p[f"l{i}.fc"].T

    # ---- forward / backward ----------------------------------------------------------
    def forward(self, idx, targets=None, kv_cache: Optional[dict] = None, pos_offset: int = 0,
                soft=None, soft_weight: float = 0.5):
        """idx: (B,T) ints. Returns (logits, loss, caches). Loss ignores targets < 0.

        `soft` (B,T,V probabilities from a teacher model) mixes a second target in: the loss becomes
        (1 - soft_weight) x the usual loss + soft_weight x the cross-entropy against the teacher."""
        xp = self.xp
        idx = from_numpy(xp, np.asarray(idx))
        B, T = idx.shape
        x = self.p["wte"][idx]
        if not self.is_llama:
            x = x + self.p["wpe"][pos_offset:pos_offset + T][None, :, :]
        caches: list[Any] = [idx]
        for i in range(self.cfg.n_layer):
            h, n1 = self._norm_fwd(x, f"l{i}.ln1")
            a, ac = self._attn_fwd(h, i, kv_cache, pos_offset)
            x = x + a
            h2, n2 = self._norm_fwd(x, f"l{i}.ln2")
            m, mc = self._mlp_fwd(h2, i)
            x = x + m
            caches.append((n1, ac, n2, mc))
        xf, nf = self._norm_fwd(x, "lnf")
        caches.append(nf)
        logits = xf @ self.p["wte"].T
        loss = None
        if targets is not None:
            targets = from_numpy(xp, np.asarray(targets))
            probs = softmax(xp, logits.reshape(-1, self.cfg.vocab_size).astype(xp.float64))
            flat_t = targets.reshape(-1)
            valid = flat_t >= 0
            n = max(1, int(valid.sum()))
            rows = xp.arange(flat_t.shape[0])[valid]
            loss = float(-xp.log(probs[rows, flat_t[valid]] + 1e-12).sum() / n)
            soft_flat = None
            if soft is not None:
                soft_flat = from_numpy(xp, np.asarray(soft)).reshape(-1, self.cfg.vocab_size).astype(xp.float64)
                taught = float(-(soft_flat[valid] * xp.log(probs[valid] + 1e-12)).sum() / n)
                loss = (1.0 - soft_weight) * loss + soft_weight * taught
            caches.append((probs, flat_t, valid, n, xf, soft_flat, soft_weight))
        return logits, loss, caches

    def backward(self, caches) -> dict[str, Any]:
        xp = self.xp
        grads = {k: xp.zeros_like(v) for k, v in self.p.items()}
        probs, flat_t, valid, n, xf, soft_flat, soft_weight = caches[-1]
        nf = caches[-2]
        idx = caches[0]
        B, T = idx.shape
        C = self.cfg.n_embd
        dlogits = probs.copy()
        rows = xp.arange(flat_t.shape[0])[valid]
        if soft_flat is None:
            dlogits[rows, flat_t[valid]] -= 1.0
        else:                                   # the target is a mix of the right token and the teacher's guess
            dlogits[rows, flat_t[valid]] -= 1.0 - soft_weight
            dlogits -= soft_weight * soft_flat
        dlogits[~valid] = 0.0
        dlogits = (dlogits / n).astype(xp.float32)
        grads["wte"] += dlogits.T @ xf.reshape(-1, C)
        dxf = (dlogits @ self.p["wte"]).reshape(B, T, C)
        dx = self._norm_bwd(dxf, nf, "lnf", grads)
        for i in reversed(range(self.cfg.n_layer)):
            n1, ac, n2, mc = caches[1 + i]
            dh2 = self._mlp_bwd(dx, mc, i, grads)
            dx = dx + self._norm_bwd(dh2, n2, f"l{i}.ln2", grads)
            dh = self._attn_bwd(dx, ac, i, grads)
            dx = dx + self._norm_bwd(dh, n1, f"l{i}.ln1", grads)
        if xp is np:
            np.add.at(grads["wte"], idx.reshape(-1), dx.reshape(-1, C))
            if not self.is_llama:
                np.add.at(grads["wpe"], np.tile(np.arange(T), B), dx.reshape(-1, C))
        else:
            xp.add.at(grads["wte"], idx.reshape(-1), dx.reshape(-1, C))
            if not self.is_llama:
                xp.add.at(grads["wpe"], xp.tile(xp.arange(T), B), dx.reshape(-1, C))
        return grads

    # ---- generation ------------------------------------------------------------------
    def generate(self, idx: list[int], max_new_tokens: int, temperature: float = 0.9, top_k: int = 40, top_p: float = 0.95,
                 repetition_penalty: float = 1.15, stop_tokens: Optional[set[int]] = None, rng: Optional[np.random.Generator] = None,
                 banned: Optional[set[int]] = None, min_tokens: int = 0, no_repeat_ngram: int = 3, min_p: float = 0.05) -> list[int]:
        rng = rng or np.random.default_rng()
        T = self.cfg.block_size
        out = list(idx)
        kv: dict = {}
        ctx = out[-(T - 1):]
        logits, _, _ = self.forward(np.asarray([ctx]), kv_cache=kv, pos_offset=0)
        pos = len(ctx)
        for produced in range(max_new_tokens):
            lg = to_numpy(logits[0, -1]).astype(np.float64)
            if stop_tokens and produced < min_tokens:
                for t in stop_tokens:
                    lg[t] = -1e9
            if repetition_penalty > 1.0:
                for t in set(out[-64:]):
                    lg[t] = lg[t] / repetition_penalty if lg[t] > 0 else lg[t] * repetition_penalty
            if no_repeat_ngram and len(out) >= no_repeat_ngram:
                # Block any token that would complete an n-gram already present in the output.
                prefix = tuple(out[-(no_repeat_ngram - 1):]) if no_repeat_ngram > 1 else ()
                for j in range(len(out) - no_repeat_ngram + 1):
                    if tuple(out[j:j + no_repeat_ngram - 1]) == prefix:
                        lg[out[j + no_repeat_ngram - 1]] = -1e9
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
                p = softmax(np, lg)
                if min_p > 0:
                    p = np.where(p < min_p * p.max(), 0.0, p)
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
                kv = {}
                ctx = out[-(T - 1):]
                logits, _, _ = self.forward(np.asarray([ctx]), kv_cache=kv, pos_offset=0)
                pos = len(ctx)
            else:
                logits, _, _ = self.forward(np.asarray([[nxt]]), kv_cache=kv, pos_offset=pos)
                pos += 1
        return out

    # ---- persistence -----------------------------------------------------------------
    def save(self, directory: Path, extra: Optional[dict[str, Any]] = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "weights.npz", **{k: to_numpy(v) for k, v in self.p.items()})
        meta = {"config": asdict(self.cfg), "n_params": self.n_params(), **(extra or {})}
        (directory / "model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> GaugeLM:
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        raw = dict(meta["config"])
        if "arch" not in raw:  # checkpoint from the first GPT-style version
            raw["arch"] = "gpt"
        cfg = ModelConfig(**{k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__})
        m = cls(cfg, device=device)
        with np.load(directory / "weights.npz") as z:
            for k in m.p:
                m.p[k] = from_numpy(m.xp, z[k])
        return m


class AdamW:
    def __init__(self, params: dict[str, Any], lr: float = 3e-4, betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1, xp=np):
        self.xp = xp
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = {k: xp.zeros_like(v) for k, v in params.items()}
        self.v = {k: xp.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, Any], grads: dict[str, Any], lr: Optional[float] = None, clip: float = 1.0) -> float:
        xp = self.xp
        self.t += 1
        lr = self.lr if lr is None else lr
        norm = math.sqrt(sum(float((g.astype(xp.float64) ** 2).sum()) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-6)) if clip else 1.0
        for k, p in params.items():
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            decay = self.wd if p.ndim >= 2 and not k.endswith(("_g", "_b")) else 0.0
            params[k] = (p - lr * (mhat / (xp.sqrt(vhat) + self.eps) + decay * p)).astype(xp.float32)
        return norm
