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

* **Bigger heads.** A head can grow to a whole multiple of its size: rotary pair i of the old head
  turns at the same rate as pair k*i of a head k times the size, so the old head sits inside the new one.

Two more changes of size live here that are *not* free, and say so:

* **The vocabulary** (`grow_vocab_checkpoint`). The tokenizer learns further merges, so every old token
  keeps its number and meaning, and the model gets a row for each new one. Text cut the old way scores
  the old tokens exactly as before, but the text is now cut differently and the new tokens are
  untrained, so the model dips until it has trained on them.
* **Shrinking** (`shrink_checkpoint`). The small model is cut out of the big one (growth backwards) and
  then taught by it. A smaller model holds less; the report says how much was lost.

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
FILES = ("model.json", "weights.npz", "train_state.json", "tokenizer.json")
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
    if cfg.arch != "llama":
        return V * C + cfg.block_size * C + 2 * C + L * (12 * C * C + 13 * C)
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
    if old.arch != new.arch:
        raise GrowError("growing cannot change the kind of model (Llama-style or the older GPT-style)")
    if new.vocab_size != old.vocab_size:
        raise GrowError("the vocabulary grows separately (`gauge native grow --vocab`), not with the shape")
    if old.arch != "llama":
        # LayerNorm subtracts the mean over every dimension, so empty new dimensions cannot be hidden from it.
        if (new.n_embd, new.n_head, new.kv_heads) != (old.n_embd, old.n_head, old.kv_heads):
            raise GrowError("an older GPT-style model can grow deeper and read more at once, but not wider. "
                            "Use --layers and --context, or train a new Llama-style model.")
        if new.n_layer < old.n_layer or new.block_size < old.block_size:
            raise GrowError("a model can only grow")
        if (new.n_layer, new.block_size) == (old.n_layer, old.block_size):
            raise GrowError("that is the size the model already is")
        return
    if new.n_embd % new.n_head:
        raise GrowError(f"a width of {new.n_embd} cannot be split into {new.n_head} heads")
    if new.head_dim % old.head_dim or new.head_dim % 2:
        raise GrowError(f"the size of each attention head (width divided by heads) is {old.head_dim} and can only "
                        f"grow to a whole multiple of that; {new.n_embd} wide with {new.n_head} heads gives {new.head_dim}")
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
        H = heads or (old.n_head if old.arch != "llama" else C // old.head_dim)
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
    if old.arch != "llama":                      # deeper and longer is all an older model can do
        new = replace(old, n_layer=old.n_layer * 2, block_size=min(4096, old.block_size * 2))
        check(old, new)
        return new
    for name in SIZES:                           # the next size up that this model can reach
        cand = replace(old, **{k: v for k, v in PRESETS[name].items() if k != "vocab_size"})
        if n_params(old) < n_params(cand) <= 8 * n_params(old):     # a step, not a leap (tiny to small is 6.4 times)
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
    if old_cfg.arch != "llama":
        return _grow_gpt(model, new_cfg, seed)
    C, C2, D, D2 = old_cfg.n_embd, new_cfg.n_embd, old_cfg.head_dim, new_cfg.head_dim
    k = D2 // D
    # A bigger head holds the old one: rotary pair i of a head of size D turns at the same rate as pair
    # k*i of a head of size k*D, so each old pair moves there. The score is divided by sqrt(head size),
    # so the queries are scaled up by sqrt(k) to keep it the same.
    dmap = np.array([2 * k * (d // 2) + d % 2 for d in range(D)])
    sq = math.sqrt(k)
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
            if k > 1:                             # the head's new dimensions ask nothing yet
                p[a + "wq"][:, h2 * D2:(h2 + 1) * D2] = 0.0
            p[a + "wq"][:C, h2 * D2 + dmap] = old[b + "wq"][:, h * D:(h + 1) * D] * sq
            p[a + "wo"][h2 * D2 + dmap, :C] = old[b + "wo"][h * D:(h + 1) * D, :] * s
        for g, j in enumerate(kv_src):            # key/value heads, copied where the grouping got tighter
            if j is not None:
                p[a + "wk"][:C, g * D2 + dmap] = old[b + "wk"][:, j * D:(j + 1) * D]
                p[a + "wv"][:C, g * D2 + dmap] = old[b + "wv"][:, j * D:(j + 1) * D]
        p[a + "w1"][:C, :Hd] = old[b + "w1"]
        p[a + "w3"][:C, :Hd] = old[b + "w3"]
        p[a + "w2"][:Hd, :C] = old[b + "w2"] * s
    out.p = {k: v for k, v in p.items()}
    return out


def _grow_gpt(model: GaugeLM, new_cfg: ModelConfig, seed: int = 0) -> GaugeLM:
    """The older GPT-style model: more layers (each new one writes nothing yet) and more positions."""
    old_cfg = model.cfg
    old = {k: to_numpy(v).astype(np.float32) for k, v in model.p.items()}
    out = GaugeLM(new_cfg, seed=seed, device="cpu")
    p = {k: np.array(to_numpy(v), dtype=np.float32) for k, v in out.p.items()}
    for name in ("wte", "lnf_g", "lnf_b"):
        p[name] = old[name].copy()
    p["wpe"][:old_cfg.block_size] = old["wpe"]            # positions past the old limit start random
    placed = dict(zip(layer_positions(old_cfg.n_layer, new_cfg.n_layer), range(old_cfg.n_layer)))
    for i2 in range(new_cfg.n_layer):
        a = f"l{i2}."
        if i2 in placed:
            for key in [key for key in old if key.startswith(f"l{placed[i2]}.")]:
                p[a + key.split(".", 1)[1]] = old[key].copy()
        else:
            for name in ("proj", "proj_b", "fc2", "fc2_b"):
                p[a + name][:] = 0.0
    out.p = dict(p)
    return out


def difference(old: GaugeLM, new: GaugeLM, ids: Optional[np.ndarray] = None, seed: int = 0) -> float:
    """The largest difference between the two models' answers on the same input."""
    if ids is None:
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, old.cfg.vocab_size, size=(2, min(old.cfg.block_size, 48)))
    a = to_numpy(old.forward(ids)[0]).astype(np.float64)
    b = to_numpy(new.forward(ids)[0]).astype(np.float64)
    return float(np.abs(a - b).max())


# ------------------------------------------------------------------------------------------- the vocabulary
VOCAB_STEPS = (4096, 8192, 16384, 32000)


def next_vocab(current: int) -> int:
    """The next vocabulary size up: the named sizes use 4096, 8192, 16384 and 32000."""
    return next((v for v in VOCAB_STEPS if v > current), current * 2)


def grow_vocab(model: GaugeLM, old_tok: Any, new_tok: Any) -> GaugeLM:
    """The same model with rows for the tokens `new_tok` adds to `old_tok`.

    A new token is two existing tokens joined (which may themselves be new), so its row starts as the
    average of theirs. The special tokens stay last, so their rows move up. Nothing else changes."""
    cfg = model.cfg
    n_old, n_new = len(old_tok.vocab), len(new_tok.vocab)
    if cfg.vocab_size != old_tok.vocab_size:
        raise GrowError("the model and its tokenizer do not match")
    if n_new <= n_old or new_tok.merges[:len(old_tok.merges)] != old_tok.merges:
        raise GrowError("the new vocabulary must contain every token of the old one and add to it")
    specials = cfg.vocab_size - n_old
    old = to_numpy(model.p["wte"]).astype(np.float32)
    wte = np.zeros((n_new + specials, cfg.n_embd), np.float32)
    wte[:n_old] = old[:n_old]
    for t in range(n_old, n_new):
        a, b = new_tok.merges[t - 256]
        wte[t] = (wte[a] + wte[b]) / 2.0
    wte[n_new:] = old[n_old:]
    out = GaugeLM(replace(cfg, vocab_size=n_new + specials), seed=0, device="cpu")
    out.p = {k: np.array(to_numpy(v), dtype=np.float32) for k, v in model.p.items()}
    out.p["wte"] = wte
    return out


def vocab_difference(old: GaugeLM, new: GaugeLM, n_old: int, seed: int = 0) -> float:
    """Text cut the old way must score every old token exactly as before: the largest difference."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, n_old, size=(2, min(old.cfg.block_size, 48)))
    a = to_numpy(old.forward(ids)[0]).astype(np.float64)
    b = to_numpy(new.forward(ids)[0]).astype(np.float64)
    specials = old.cfg.vocab_size - n_old
    keep = np.concatenate([np.arange(n_old), new.cfg.vocab_size - specials + np.arange(specials)])
    return float(np.abs(a - b[..., keep]).max())


def loss_per_char(model: GaugeLM, tok: Any, text: str, max_tokens: int = 6000) -> float:
    """How surprised the model is by `text`, per character. Unlike the loss per token, this can be
    compared between vocabularies, which cut the same text into different numbers of tokens."""
    ids = np.array(tok.encode(text), dtype=np.int64)
    if len(ids) > max_tokens:
        text = tok.decode([int(i) for i in ids[:max_tokens]])
        ids = ids[:max_tokens]
    T = model.cfg.block_size
    total, n = 0.0, 0
    for start in range(0, len(ids) - 1, T):
        x, y = ids[start:start + T], ids[start + 1:start + T + 1]
        x = x[:len(y)]
        if len(y) < 2:
            break
        total += float(model.forward(x[None, :], y[None, :])[1]) * len(y)
        n += len(y)
    return total / max(1, len(text)) if n else float("nan")


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
                            "vocab": cfg.vocab_size, "next_vocab": next_vocab(cfg.vocab_size),
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
    backup = _backup(model_dir, step, str(size_name(old.cfg) or old.cfg.n_embd))
    extra = {k: v for k, v in meta.items() if k not in ("config", "n_params")}
    new.save(model_dir, extra)
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "shape", "step": step, "from": asdict(old.cfg),
              "to": asdict(new_cfg), "params_before": old.n_params(), "params_after": new.n_params(), "difference": diff,
              "backup": backup.name}
    (model_dir / HISTORY).write_text(json.dumps(_history(model_dir) + [record], indent=1), encoding="utf-8")
    log.info("grew GaugeLM from %s to %s (difference %.3g)", describe(old.cfg), describe(new_cfg), diff)
    return {**record, "before": describe(old.cfg), "after": describe(new_cfg), "size": size_name(new_cfg),
            "times": round(new.n_params() / max(1, old.n_params()), 1), "backup_path": str(backup),
            "message": (f"GaugeLM grew from {old.n_params() / 1e6:.1f} to {new.n_params() / 1e6:.1f} million weights "
                        f"({describe(new_cfg)}). It kept everything it had learned: its answers are unchanged "
                        f"(checked, difference {diff:.1g}). The new capacity is empty until it trains, so start "
                        f"training again. A bigger model is slower to train and to write with.")}


def undo_grow(model_dir: Path) -> dict[str, Any]:
    """Put back the model as it was before the last change of size (a growth, a larger vocabulary or a
    shrink). Training done since then is lost. Can be repeated to step back through earlier changes."""
    model_dir = Path(model_dir)
    hist = _history(model_dir)
    if not hist:
        raise GrowError("GaugeLM's size has not been changed, so there is nothing to undo")
    last = hist[-1]
    backup = model_dir / BACKUP_DIR / str(last["backup"])
    if not (backup / "model.json").exists() or not (backup / "weights.npz").exists():
        raise GrowError(f"the saved copy from before the change is missing ({backup})")
    for name in FILES:
        if (backup / name).exists():
            shutil.copy2(backup / name, model_dir / name)
    shutil.rmtree(backup)
    (model_dir / HISTORY).write_text(json.dumps(hist[:-1], indent=1), encoding="utf-8")
    cfg = ModelConfig(**{k: v for k, v in last["from"].items() if k in ModelConfig.__dataclass_fields__})
    what = {"vocabulary": "its vocabulary grew", "shrink": "it shrank"}.get(last.get("kind", "shape"), "it grew")
    return {"step": int(last["step"]), "size": size_name(cfg), "config": asdict(cfg), "after": describe(cfg),
            "kind": last.get("kind", "shape"),
            "message": f"GaugeLM is back to how it was at step {last['step']}, before {what} ({describe(cfg)}, "
                       f"{cfg.vocab_size} tokens). Training done since then was dropped."}


def _backup(model_dir: Path, step: int, label: str) -> Path:
    backup = model_dir / BACKUP_DIR / f"step-{step}-{label}"
    n = 2
    while backup.exists():                       # the same step can be resized more than once
        backup = model_dir / BACKUP_DIR / f"step-{step}-{label}-{n}"
        n += 1
    backup.mkdir(parents=True)
    for name in FILES:
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, backup / name)
    return backup


def _sample(texts: list[str], chars: int = 12000, pieces: int = 8) -> str:
    """A sample of the library to measure on: pieces taken evenly from beginning to end, not just the end."""
    joined = "\n".join(t for t in texts if t)
    if len(joined) <= chars:
        return joined
    each = chars // pieces
    step = (len(joined) - each) // (pieces - 1)
    return "\n".join(joined[i * step:i * step + each] for i in range(pieces))


def grow_vocab_checkpoint(model_dir: Path, texts: list[str], vocab_size: Optional[int] = None) -> dict[str, Any]:
    """Give the saved model a larger vocabulary, learned from `texts` (the library).

    Every token it has is kept, with the same number and the same row; text cut the old way scores the
    old tokens exactly as before (checked). But the text is now cut into fewer, longer tokens, and the
    new ones start as the average of their parts, so the model is a little worse until it has trained on
    them. The report says by how much, per character so the two vocabularies can be compared."""
    from .tokenizer import BPETokenizer

    model_dir = Path(model_dir)
    if not (model_dir / "model.json").exists() or not (model_dir / "tokenizer.json").exists():
        raise GrowError("GaugeLM has not been trained yet. Train it first; it can be grown afterwards.")
    old = GaugeLM.load(model_dir, device="cpu")
    tok = BPETokenizer.load(model_dir / "tokenizer.json")
    target = int(vocab_size or next_vocab(tok.vocab_size))
    if target <= tok.vocab_size:
        raise GrowError(f"the vocabulary is already {tok.vocab_size} tokens; it can only grow")
    text = "\n".join(t for t in texts if t)
    if not text.strip():
        raise GrowError("there is no text to learn new tokens from. Add books or your own writing first.")
    tok2 = tok.extended(text, target)
    added = tok2.vocab_size - tok.vocab_size
    if added <= 0:
        raise GrowError("the library has no more repeated pieces to make tokens from. Add more text first.")
    new = grow_vocab(old, tok, tok2)
    diff = vocab_difference(old, new, len(tok.vocab))
    if not diff <= TOLERANCE:
        raise GrowError(f"the model with the larger vocabulary did not match the trained one (difference {diff:.4g}); "
                        "nothing was changed")
    sample = _sample(texts)
    before, after = loss_per_char(old, tok, sample), loss_per_char(new, tok2, sample)
    fewer = 1.0 - len(tok2.encode(sample)) / max(1, len(tok.encode(sample)))
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    step = int(meta.get("step", 0))
    backup = _backup(model_dir, step, f"vocab-{tok.vocab_size}")
    new.save(model_dir, {k: v for k, v in meta.items() if k not in ("config", "n_params")})
    tok2.save(model_dir / "tokenizer.json")
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "vocabulary", "step": step, "from": asdict(old.cfg),
              "to": asdict(new.cfg), "params_before": old.n_params(), "params_after": new.n_params(), "difference": diff,
              "backup": backup.name, "loss_per_char_before": before, "loss_per_char_after": after}
    (model_dir / HISTORY).write_text(json.dumps(_history(model_dir) + [record], indent=1), encoding="utf-8")
    worse = (after / before - 1.0) if before == before and before > 0 else float("nan")
    return {**record, "added": added, "vocab": tok2.vocab_size, "fewer_tokens": fewer, "backup_path": str(backup),
            "after": describe(new.cfg),
            "message": (f"GaugeLM's vocabulary grew from {tok.vocab_size} to {tok2.vocab_size} tokens. The same text now "
                        f"takes {fewer:.0%} fewer tokens, so it reads further back and writes faster. It kept every token "
                        "it knew, but the new ones are untrained"
                        + (f": it is about {worse:.0%} less sure of the text until it trains on them, which usually takes "
                           "a short session. " if worse > 0.005 else ", so it needs a session to learn to use them. ")
                        + "Start training again.")}


# ------------------------------------------------------------------------------------------- shrinking
def smaller_config(big: ModelConfig, to: Optional[str] = None, layers: Optional[int] = None, width: Optional[int] = None,
                   heads: Optional[int] = None, kv_heads: Optional[int] = None, context: Optional[int] = None,
                   history: Optional[list[dict[str, Any]]] = None) -> ModelConfig:
    """The shape to shrink into: a named size, the given numbers, or by default the shape it had before
    it last grew (otherwise the next named size down, otherwise half). The vocabulary is kept."""
    if big.arch != "llama":
        raise GrowError("only the Llama-style GaugeLM can be shrunk")
    fields = ModelConfig.__dataclass_fields__
    if any(v is not None for v in (layers, width, heads, kv_heads, context)):
        C = width or big.n_embd
        H = heads or max(1, C // big.head_dim)
        small = replace(big, n_layer=layers or big.n_layer, n_embd=C, n_head=H,
                        n_kv_head=kv_heads or max(1, min(big.kv_heads, H)), block_size=context or big.block_size)
    elif to:
        if to not in PRESETS:
            raise GrowError(f"unknown size {to!r}; sizes: {', '.join(SIZES)}")
        small = replace(big, **{k: v for k, v in PRESETS[to].items() if k != "vocab_size"})
    else:
        small = None
        for rec in reversed(history or []):
            if rec.get("kind", "shape") == "shape":
                small = replace(ModelConfig(**{k: v for k, v in rec["from"].items() if k in fields}), vocab_size=big.vocab_size)
                break
        if small is None:
            for name in reversed(SIZES):
                cand = replace(big, **{k: v for k, v in PRESETS[name].items() if k != "vocab_size"})
                if n_params(cand) < n_params(big):
                    small = cand
                    break
        if small is None:
            small = replace(big, n_layer=max(1, big.n_layer // 2), n_embd=big.n_embd // 2, n_head=max(1, big.n_head // 2),
                            n_kv_head=max(1, big.kv_heads // 2), block_size=max(16, big.block_size // 2))
    if small.n_embd % small.n_head or small.n_head % small.kv_heads or small.head_dim % 2:
        raise GrowError(f"{small.n_embd} wide cannot be split into {small.n_head} heads ({small.kv_heads} key/value heads)")
    if n_params(small) >= n_params(big):
        raise GrowError("that is not smaller than the model is now")
    return small


def cut_down(big: GaugeLM, small_cfg: ModelConfig, seed: int = 0) -> tuple[GaugeLM, bool]:
    """A smaller model cut out of a bigger one: growth run backwards. Returns (model, was_cut).

    If the big model was just grown from this shape, this gives back exactly the model it was. After
    more training it gives that model plus what the old parts learned since, without the parts that are
    cut away; teaching (`shrink_checkpoint`) then makes up some of the difference. When the big model
    could not have been grown from this shape, there is nothing to cut and the small model starts random."""
    out = GaugeLM(small_cfg, seed=seed, device="cpu")
    try:
        check(small_cfg, big.cfg)
    except GrowError:
        return out, False
    c, C, d, D = small_cfg.n_embd, big.cfg.n_embd, small_cfg.head_dim, big.cfg.head_dim
    k, hd = D // d, small_cfg.hidden
    s, sq = math.sqrt(C / c), math.sqrt(D // d)
    dmap = np.array([2 * k * (x // 2) + x % 2 for x in range(d)])
    q_map, kv_src = head_layout(small_cfg.n_head, small_cfg.kv_heads, big.cfg.n_head, big.cfg.kv_heads)
    src = {key: to_numpy(v).astype(np.float32) for key, v in big.p.items()}
    p: dict[str, np.ndarray] = {"wte": src["wte"][:, :c] / s, "lnf_g": src["lnf_g"][:c] * s * s}
    for i, i2 in enumerate(layer_positions(small_cfg.n_layer, big.cfg.n_layer)):
        a, b = f"l{i}.", f"l{i2}."
        p[a + "ln1_g"], p[a + "ln2_g"] = src[b + "ln1_g"][:c] * s, src[b + "ln2_g"][:c] * s
        p[a + "wq"] = np.concatenate([src[b + "wq"][:c, q_map[h] * D + dmap] / sq for h in range(small_cfg.n_head)], axis=1)
        p[a + "wo"] = np.concatenate([src[b + "wo"][q_map[h] * D + dmap, :c] / s for h in range(small_cfg.n_head)], axis=0)
        slots = [kv_src.index(j) for j in range(small_cfg.kv_heads)]
        p[a + "wk"] = np.concatenate([src[b + "wk"][:c, g * D + dmap] for g in slots], axis=1)
        p[a + "wv"] = np.concatenate([src[b + "wv"][:c, g * D + dmap] for g in slots], axis=1)
        p[a + "w1"], p[a + "w3"] = src[b + "w1"][:c, :hd], src[b + "w3"][:c, :hd]
        p[a + "w2"] = src[b + "w2"][:hd, :c] / s
    out.p = {key: np.ascontiguousarray(v, dtype=np.float32) for key, v in p.items()}
    return out, True


def shrink_checkpoint(model_dir: Path, docs: list[tuple[str, str]], to: Optional[str] = None, steps: int = 300,
                      minutes: Optional[float] = None, batch_size: int = 8, lr: float = 6e-4, soft_weight: float = 0.7,
                      should_stop: Any = None, progress: Any = None, seed: int = 0, **shape: Any) -> dict[str, Any]:
    """Replace the saved model with a smaller one that the big one has taught.

    The small model is cut out of the big one where it can be, then trained on the library with the big
    model's guesses as its targets as well as the real next token. This loses some of what the big model
    knew: a smaller model holds less. The report says how much. The big model is kept and can be put back."""
    from .model import AdamW, softmax
    from .tokenizer import BPETokenizer
    from .train import TrainConfig, batches, encode_corpus, lr_at

    model_dir = Path(model_dir)
    if not (model_dir / "model.json").exists() or not (model_dir / "tokenizer.json").exists():
        raise GrowError("GaugeLM has not been trained yet, so there is nothing to shrink.")
    big = GaugeLM.load(model_dir, device="cpu")
    tok = BPETokenizer.load(model_dir / "tokenizer.json")
    small_cfg = smaller_config(big.cfg, to, history=_history(model_dir), **shape)
    small, was_cut = cut_down(big, small_cfg, seed=seed)
    tcfg = TrainConfig(steps=steps, batch_size=batch_size, lr=lr, warmup=min(20, max(1, steps // 10)), min_lr=lr / 10)
    data = encode_corpus(tok, docs, tcfg.max_corpus_chars)
    T = small_cfg.block_size
    if len(data) < T * 4:
        raise GrowError("there is too little text in the library to teach the smaller model. Add more first.")
    sample = _sample([t for _, t in docs])
    big_loss, cut_loss = loss_per_char(big, tok, sample), loss_per_char(small, tok, sample)
    rng = np.random.default_rng(seed)
    gen = batches(data, T, batch_size, rng)
    opt = AdamW(small.p, lr=lr, weight_decay=tcfg.weight_decay)
    from .. import limit

    governor = limit.compute()
    t0 = time.monotonic()
    done, last = 0, None
    while done < steps:
        t_step = time.monotonic()
        x, y = next(gen)
        soft = softmax(np, to_numpy(big.forward(x)[0]).astype(np.float64))
        _, last, caches = small.forward(x, y, soft=soft, soft_weight=soft_weight)
        opt.step(small.p, small.backward(caches), lr=lr_at(done, tcfg), clip=tcfg.grad_clip)
        done += 1
        if progress and (done % 10 == 0 or done == steps):
            progress({"step": done, "of": steps, "loss": round(float(last), 4)})
        if minutes is not None and time.monotonic() - t0 > minutes * 60:
            break
        if should_stop is not None and should_stop():
            break
        governor.pace(time.monotonic() - t_step, None)
    small_loss = loss_per_char(small, tok, sample)
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    step = int(meta.get("step", 0))
    backup = _backup(model_dir, step, f"big-{size_name(big.cfg) or big.cfg.n_embd}")
    small.save(model_dir, {k: v for k, v in meta.items() if k not in ("config", "n_params")})
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "shrink", "step": step, "from": asdict(big.cfg),
              "to": asdict(small_cfg), "params_before": big.n_params(), "params_after": small.n_params(),
              "backup": backup.name, "loss_per_char_before": big_loss, "loss_per_char_after": small_loss,
              "loss_per_char_cut": cut_loss, "taught_steps": done, "cut_from_big": was_cut}
    (model_dir / HISTORY).write_text(json.dumps(_history(model_dir) + [record], indent=1), encoding="utf-8")
    worse = small_loss / big_loss - 1.0 if big_loss > 0 else float("nan")
    return {**record, "after": describe(small_cfg), "size": size_name(small_cfg), "backup_path": str(backup),
            "seconds": round(time.monotonic() - t0, 1),
            "message": (f"GaugeLM shrank from {big.n_params() / 1e6:.1f} to {small.n_params() / 1e6:.1f} million weights "
                        f"({describe(small_cfg)}). The bigger model taught it for {done} steps. "
                        + (f"It is about {worse:.0%} less sure of the text than the bigger model was: a smaller "
                           "model holds less. " if worse > 0.005
                           else "On a sample of the library it is as sure of the text as the bigger model was. ")
                        + "It is faster to train and to write with. The bigger model is kept and can be put back.")}


def apply_to_config(cfg: Any, shape: dict[str, Any]) -> None:
    """Keep Gauge's settings in step with the model on disk after it grew (or the growth was undone)."""
    mc = ModelConfig(**{k: v for k, v in shape.items() if k in ModelConfig.__dataclass_fields__})
    n = cfg.native
    n.size = size_name(mc)
    n.n_layer, n.n_head, n.n_embd, n.n_kv_head = mc.n_layer, mc.n_head, mc.n_embd, mc.kv_heads
    n.block_size, n.vocab_size = mc.block_size, mc.vocab_size
    if cfg.llm.provider == "native":
        cfg.llm.num_ctx = mc.block_size
