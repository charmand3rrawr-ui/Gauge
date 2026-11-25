"""Grow GaugeLM into a bigger model without losing what it learned.

A model's size is fixed when it is created, and training only changes the values of its weights. To
get more capacity you used to have to start again from nothing. This module instead builds a larger
model that computes *exactly the same function* as the trained one, with the extra capacity switched
off, so training can carry on from where it was and gradually put the new capacity to use.

How each dimension grows (the Llama-style architecture only):

* **Deeper.** A new layer adds its output to the stream that runs through the model. With its output
  weights at zero it adds nothing, so the model behaves as before. The new layers are spread evenly
  between the old ones.
* **Wider.** The old model lives in the first ``C`` of the new ``C'`` dimensions; the new ones start
  at zero and nothing writes to them yet. The norm layers divide by the root-mean-square over all
  dimensions, which the empty ones would shrink, so everything written into the stream is scaled up by
  ``sqrt(C'/C)`` and the norm gains are scaled down to match. The result is exact, not approximate.
* **More attention heads.** The head size stays the same (the rotary position frequencies depend on
  it), so a wider model has more heads. Old heads keep their weights and move to slots where they still
  share the same key/value head; where the grouping gets tighter a key/value head is copied. New heads
  write zero.
* **A bigger feed-forward layer.** New hidden units have zero output weights.
* **Longer context.** Positions are rotary, so nothing is stored per position: the limit just moves.

What does not grow is the vocabulary. The tokenizer was built with the first training run and changing
it would change what every token means.

The new parts are not dead: the weights that *read* into them are random, so gradients reach the zero
output weights from the first training step.

Before anything on disk is replaced the grown model is checked against the old one on the same input,
and the old checkpoint is kept so the growth can be undone.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .backend import to_numpy
from .model import PRESETS, GaugeLM, ModelConfig

log = logging.getLogger(__name__)

SIZES = ("tiny", "small", "medium", "large")
BACKUP_DIR = "before-grow"
HISTORY = "growth.json"
FILES = ("model.json", "weights.npz", "train_state.json")
TOLERANCE = 2e-3          # largest allowed difference between old and new answers (logits are of order 1-10)


class GrowError(ValueError):
    """The requested shape cannot be reached from this model, or the grown model failed its check."""


# ------------------------------------------------------------------------------------------- shapes
def size_name(cfg: ModelConfig) -> str:
    """The preset this shape matches (the vocabulary aside), or '' for a custom shape."""
    for name, p in PRESETS.items():
        if all(getattr(cfg, k) == v for k, v in p.items() if k != "vocab_size") and cfg.arch == "llama":
            return name
    return ""


def n_params(cfg: ModelConfig) -> int:
    C, V, L, Hk, D, Hd = cfg.n_embd, cfg.vocab_size, cfg.n_layer, cfg.kv_heads, cfg.head_dim, cfg.hidden
    per_layer = 2 * C + 2 * C * C + 2 * C * Hk * D + 3 * C * Hd
    return V * C + C + L * per_layer


def describe(cfg: ModelConfig) -> str:
    name = size_name(cfg)
    return (f"{name + ': ' if name else ''}{cfg.n_layer} layers, {cfg.n_embd} wide, {cfg.n_head} heads, "
            f"reads {cfg.block_size} tokens at once, {n_params(cfg) / 1e6:.1f} million weights")


def head_layout(H: int, Hk: int, H2: int, Hk2: int) -> tuple[dict[int, int], list[Optional[int]]]:
    """Where each old attention head goes in the new model.

    Query heads are grouped: with ``rep = H // Hk``, query head ``h`` reads key/value head ``h // rep``.
    Returns (old query head -> new query head, and for each new key/value head the old one it copies or
    None for a new one). Every old query head must still read a copy of its old key/value head."""
    if H % Hk or H2 % Hk2:
        raise GrowError("the number of heads must be a multiple of the number of key/value heads")
    rep, rep2 = H // Hk, H2 // Hk2
    need = -(-rep // rep2)                      # new key/value heads needed per old one
    if Hk * need > Hk2:
        raise GrowError(f"{H2} heads with {Hk2} key/value heads cannot hold the {H} trained heads; "
                        f"use at least {Hk * need} key/value heads")
    q_map: dict[int, int] = {}
    kv_src: list[Optional[int]] = [None] * Hk2
    for j in range(Hk):
        for r in range(rep):
            g = j * need + r // rep2
            kv_src[g] = j
            q_map[j * rep + r] = g * rep2 + r % rep2
    return q_map, kv_src


def check(old: ModelConfig, new: ModelConfig) -> None:
    """Raise GrowError unless `new` can be grown from `old`."""
    if old.arch != "llama" or new.arch != "llama":
        raise GrowError("only the Llama-style GaugeLM can be grown; this is an older GPT-style model. "
                        "Train a new one with `gauge native train`.")
    if new.vocab_size != old.vocab_size:
        raise GrowError("the vocabulary cannot change when a model is grown")
    if new.n_embd % new.n_head:
        raise GrowError(f"a width of {new.n_embd} cannot be split into {new.n_head} heads")
    if new.head_dim != old.head_dim:
        raise GrowError(f"the size of each attention head must stay {old.head_dim} (width divided by heads); "
                        f"{new.n_embd} wide with {new.n_head} heads gives {new.head_dim}")
    for label, a, b in (("layers", old.n_layer, new.n_layer), ("width", old.n_embd, new.n_embd),
                        ("context", old.block_size, new.block_size), ("feed-forward size", old.hidden, new.hidden)):
        if b < a:
            raise GrowError(f"a model can only grow: {label} would go from {a} to {b}")
    head_layout(old.n_head, old.kv_heads, new.n_head, new.kv_heads)
    if n_params(new) <= n_params(old) and new.block_size <= old.block_size:
        raise GrowError("that is the size the model already is")


def target_config(old: ModelConfig, to: Optional[str] = None, layers: Optional[int] = None, width: Optional[int] = None,
                  heads: Optional[int] = None, kv_heads: Optional[int] = None, context: Optional[int] = None) -> ModelConfig:
    """The shape to grow into: a named size, the next size up, or the given numbers. The vocabulary and
    the head size are always kept."""
    custom = any(v is not None for v in (layers, width, heads, kv_heads, context))
    if to and custom:
        raise GrowError("give a size name or the numbers, not both")
    if custom:
        C = width or old.n_embd
        if C % old.head_dim and heads is None:
            raise GrowError(f"the width must be a multiple of {old.head_dim}")
        H = heads or C // old.head_dim
        rep = old.n_head // old.kv_heads
        Hk = kv_heads or (H // rep if H % rep == 0 else H)
        new = replace(old, n_layer=layers or old.n_layer, n_embd=C, n_head=H, n_kv_head=Hk, block_size=context or old.block_size)
        check(old, new)
        return new
    if to:
        if to not in PRESETS:
            raise GrowError(f"unknown size {to!r}; sizes: {', '.join(SIZES)}")
        new = replace(old, **{k: v for k, v in PRESETS[to].items() if k != "vocab_size"})
        check(old, new)
        return new
    for name in SIZES:                           # the next size up that this model can reach
        cand = replace(old, **{k: v for k, v in PRESETS[name].items() if k != "vocab_size"})
        if n_params(cand) > n_params(old):
            try:
                check(old, cand)
                return cand
            except GrowError:
                continue
    # No named size fits (a custom shape, or already the largest): double it.
    new = replace(old, n_layer=old.n_layer * 2, n_embd=old.n_embd * 2, n_head=old.n_head * 2, n_kv_head=old.kv_heads * 2,
                  block_size=min(4096, old.block_size * 2))
    check(old, new)
    return new


def layer_positions(L: int, L2: int) -> list[int]:
    """Where the old layers sit in the deeper model; new layers fill the gaps, evenly."""
    return [i * L2 // L for i in range(L)]


# ------------------------------------------------------------------------------------------- the surgery
def grow(model: GaugeLM, new_cfg: ModelConfig, seed: int = 0) -> GaugeLM:
    """A model of shape `new_cfg` that gives the same answers as `model`. Built on the CPU."""
    old_cfg = model.cfg
    check(old_cfg, new_cfg)
    C, C2, D = old_cfg.n_embd, new_cfg.n_embd, old_cfg.head_dim
    H, Hk, Hd = old_cfg.n_head, old_cfg.kv_heads, old_cfg.hidden
    s = math.sqrt(C2 / C)                         # what is written into the stream is scaled up by this
    q_map, kv_src = head_layout(H, Hk, new_cfg.n_head, new_cfg.kv_heads)
    old = {k: to_numpy(v).astype(np.float32) for k, v in model.p.items()}
    out = GaugeLM(new_cfg, seed=seed, device="cpu")        # fresh random weights: the new parts keep these
    p = {k: np.array(to_numpy(v), dtype=np.float32) for k, v in out.p.items()}

    def gain(g: np.ndarray, scale: float) -> np.ndarray:
        g2 = np.full(C2, float(g.mean()) * scale, np.float32)
        g2[:C] = g * scale
        return g2

    # The embedding (also the output layer): old columns scaled up, new columns empty.
    p["wte"][:] = 0.0
    p["wte"][:, :C] = old["wte"] * s
    # The last norm undoes the scale twice: once for the stream, once for the scaled output layer.
    p["lnf_g"] = gain(old["lnf_g"], 1.0 / (s * s))

    placed = dict(zip(layer_positions(old_cfg.n_layer, new_cfg.n_layer), range(old_cfg.n_layer)))
    for i2 in range(new_cfg.n_layer):
        a = f"l{i2}."
        # Nothing writes into the stream yet from new parts: output weights start at zero everywhere.
        p[a + "wo"][:] = 0.0
        p[a + "w2"][:] = 0.0
        if i2 not in placed:                      # a new layer: reads at the scale a fresh model would see, writes nothing
            p[a + "ln1_g"][:] = 1.0 / s
            p[a + "ln2_g"][:] = 1.0 / s
            continue
        b = f"l{placed[i2]}."
        p[a + "ln1_g"] = gain(old[b + "ln1_g"], 1.0 / s)
        p[a + "ln2_g"] = gain(old[b + "ln2_g"], 1.0 / s)
        for h, h2 in q_map.items():               # query heads keep their weights, in their new slots
            p[a + "wq"][:C, h2 * D:(h2 + 1) * D] = old[b + "wq"][:, h * D:(h + 1) * D]
            p[a + "wo"][h2 * D:(h2 + 1) * D, :C] = old[b + "wo"][h * D:(h + 1) * D, :] * s
        for g, j in enumerate(kv_src):            # key/value heads, copied where the grouping got tighter
            if j is not None:
                p[a + "wk"][:C, g * D:(g + 1) * D] = old[b + "wk"][:, j * D:(j + 1) * D]
                p[a + "wv"][:C, g * D:(g + 1) * D] = old[b + "wv"][:, j * D:(j + 1) * D]
        p[a + "w1"][:C, :Hd] = old[b + "w1"]
        p[a + "w3"][:C, :Hd] = old[b + "w3"]
        p[a + "w2"][:Hd, :C] = old[b + "w2"] * s
    out.p = {k: v for k, v in p.items()}
    return out


def difference(old: GaugeLM, new: GaugeLM, ids: Optional[np.ndarray] = None, seed: int = 0) -> float:
    """The largest difference between the two models' answers on the same input."""
    if ids is None:
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, old.cfg.vocab_size, size=(2, min(old.cfg.block_size, 48)))
    a = to_numpy(old.forward(ids)[0]).astype(np.float64)
    b = to_numpy(new.forward(ids)[0]).astype(np.float64)
    return float(np.abs(a - b).max())


# ------------------------------------------------------------------------------------------- on disk
def _history(model_dir: Path) -> list[dict[str, Any]]:
    f = model_dir / HISTORY
    try:
        return list(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else []
    except (OSError, ValueError):
        return []


def growth_history(model_dir: Path) -> list[dict[str, Any]]:
    return _history(Path(model_dir))


def can_grow(model_dir: Path) -> dict[str, Any]:
    """What growing would do, without doing it: the shape now, the shape after, and why not if it cannot."""
    model_dir = Path(model_dir)
    meta_file = model_dir / "model.json"
    if not meta_file.exists():
        return {"ok": False, "reason": "GaugeLM has not been trained yet. Train it first; it can be grown afterwards."}
    raw = dict(json.loads(meta_file.read_text(encoding="utf-8"))["config"])
    raw.setdefault("arch", "gpt")
    cfg = ModelConfig(**{k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__})
    info: dict[str, Any] = {"now": describe(cfg), "size": size_name(cfg), "params": n_params(cfg), "config": asdict(cfg),
                            "undo": bool(_history(model_dir)) and (model_dir / BACKUP_DIR).exists()}
    try:
        new = target_config(cfg)
    except GrowError as e:
        return {**info, "ok": False, "reason": str(e)}
    return {**info, "ok": True, "next": describe(new), "next_size": size_name(new), "next_params": n_params(new),
            "times": round(n_params(new) / max(1, n_params(cfg)), 1), "next_config": asdict(new)}


def grow_checkpoint(model_dir: Path, to: Optional[str] = None, seed: int = 0, **shape: Any) -> dict[str, Any]:
    """Grow the saved model in `model_dir`. The old checkpoint is copied aside first, the grown model is
    checked against it, and only then are the files replaced. Training carries on from the same step."""
    model_dir = Path(model_dir)
    if not (model_dir / "model.json").exists():
        raise GrowError("GaugeLM has not been trained yet. Train it first; it can be grown afterwards.")
    old = GaugeLM.load(model_dir, device="cpu")
    new_cfg = target_config(old.cfg, to, **shape)
    new = grow(old, new_cfg, seed=seed)
    diff = difference(old, new)
    if not diff <= TOLERANCE:                     # also catches NaN
        raise GrowError(f"the grown model did not match the trained one (difference {diff:.4g}); nothing was changed")
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    step = int(meta.get("step", 0))
    backup = model_dir / BACKUP_DIR / f"step-{step}-{size_name(old.cfg) or old.cfg.n_embd}"
    backup.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, backup / name)
    extra = {k: v for k, v in meta.items() if k not in ("config", "n_params")}
    new.save(model_dir, extra)
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step, "from": asdict(old.cfg), "to": asdict(new_cfg),
              "params_before": old.n_params(), "params_after": new.n_params(), "difference": diff, "backup": backup.name}
    (model_dir / HISTORY).write_text(json.dumps(_history(model_dir) + [record], indent=1), encoding="utf-8")
    log.info("grew GaugeLM from %s to %s (difference %.3g)", describe(old.cfg), describe(new_cfg), diff)
    return {**record, "before": describe(old.cfg), "after": describe(new_cfg), "size": size_name(new_cfg),
            "times": round(new.n_params() / max(1, old.n_params()), 1), "backup_path": str(backup),
            "message": (f"GaugeLM grew from {old.n_params() / 1e6:.1f} to {new.n_params() / 1e6:.1f} million weights "
                        f"({describe(new_cfg)}). It kept everything it had learned: its answers are unchanged "
                        f"(checked, difference {diff:.1g}). The new capacity is empty until it trains, so start "
                        f"training again. A bigger model is slower to train and to write with.")}


def undo_grow(model_dir: Path) -> dict[str, Any]:
    """Put back the model as it was before the last growth. Training done since then is lost."""
    model_dir = Path(model_dir)
    hist = _history(model_dir)
    if not hist:
        raise GrowError("GaugeLM has not been grown, so there is nothing to undo")
    last = hist[-1]
    backup = model_dir / BACKUP_DIR / str(last["backup"])
    if not (backup / "model.json").exists() or not (backup / "weights.npz").exists():
        raise GrowError(f"the saved copy from before the growth is missing ({backup})")
    for name in FILES:
        if (backup / name).exists():
            shutil.copy2(backup / name, model_dir / name)
    shutil.rmtree(backup)
    (model_dir / HISTORY).write_text(json.dumps(hist[:-1], indent=1), encoding="utf-8")
    cfg = ModelConfig(**{k: v for k, v in last["from"].items() if k in ModelConfig.__dataclass_fields__})
    return {"step": int(last["step"]), "size": size_name(cfg), "config": asdict(cfg), "after": describe(cfg),
            "message": f"GaugeLM is back to how it was at step {last['step']} ({describe(cfg)}). "
                       "Training done since it grew was dropped."}


def apply_to_config(cfg: Any, shape: dict[str, Any]) -> None:
    """Keep Gauge's settings in step with the model on disk after it grew (or the growth was undone)."""
    mc = ModelConfig(**{k: v for k, v in shape.items() if k in ModelConfig.__dataclass_fields__})
    n = cfg.native
    n.size = size_name(mc)
    n.n_layer, n.n_head, n.n_embd, n.n_kv_head = mc.n_layer, mc.n_head, mc.n_embd, mc.kv_heads
    n.block_size, n.vocab_size = mc.block_size, mc.vocab_size
    if cfg.llm.provider == "native":
        cfg.llm.num_ctx = mc.block_size
