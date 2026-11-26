"""How big GaugeLM may be, and what a size would take.

Nothing in GaugeLM's code limits its size: the model is the same arithmetic at four million weights and
at twenty trillion. What limits it is the machine. This module says, for any shape, what the machine
would have to provide, and stops a run that cannot fit before it fails halfway.

The figures:

* **To hold it**: four bytes per weight.
* **To train it**: about four times that (the weights, their gradients, and the optimiser's two running
  averages), plus working room that grows with the batch.
* **Text worth training on**: about twenty tokens per weight, a widely used rule of thumb for getting
  the most from a model of a given size. Far less than that and a big model memorises instead of learning.
* **Work**: about six operations per weight per token of text.

The check compares "to train it" with the memory Gauge can see. On a machine whose memory Gauge cannot
measure, switch the check off (`native.memory_check`); the run then simply proceeds, and the machine
either has the memory or it does not.
"""
from __future__ import annotations

from typing import Any

from .model import PRESETS, ModelConfig

CHECK = True                     # set from the settings when Gauge starts (native.memory_check)
TOKENS_PER_WEIGHT = 20
ALL_TEXT_TOKENS = 3e13           # a rough upper estimate of good-quality text in existence, for scale


class SizeError(MemoryError):
    """The model would not fit in this machine's memory."""


def count(n: float) -> str:
    """A number of weights or tokens in words: 3.9 million, 25.2 billion, 20.0 trillion."""
    for size, name in ((1e15, "quadrillion"), (1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if n >= size:
            return f"{n / size:.1f} {name}"
    return f"{n:.0f}"


def memory(nbytes: float) -> str:
    for size, name in ((1e15, "PB"), (1e12, "TB"), (1e9, "GB"), (1e6, "MB")):
        if nbytes >= size:
            return f"{nbytes / size:.1f} {name}"
    return f"{nbytes / 1e3:.0f} KB"


def weights(cfg: ModelConfig) -> int:
    from .grow import n_params

    return n_params(cfg)


def needs(cfg: ModelConfig) -> dict[str, Any]:
    """What a model of this shape takes: memory to hold and to train, text worth training on, and work."""
    n = weights(cfg)
    tokens = n * TOKENS_PER_WEIGHT
    return {"weights": n, "hold_bytes": n * 4, "train_bytes": n * 16, "tokens": tokens, "operations": 6.0 * n * tokens,
            "times_all_text": tokens / ALL_TEXT_TOKENS,
            "line": (f"{count(n)} weights. To hold it: {memory(n * 4)} of memory. To train it: about {memory(n * 16)}. "
                     f"Text worth training it on: about {count(tokens)} tokens"
                     + (f", roughly {tokens / ALL_TEXT_TOKENS:.0f} times all the good text thought to exist." if tokens > ALL_TEXT_TOKENS
                        else "."))}


def available_bytes(device: str = "cpu") -> float:
    """The memory Gauge can see for a model on this device, or 0 when it cannot tell."""
    if device in ("cuda", "auto"):
        try:
            import cupy as cp  # type: ignore

            if cp.cuda.runtime.getDeviceCount() > 0:
                return float(cp.cuda.Device(0).mem_info[1])
        except Exception:
            pass
        if device == "cuda":
            return 0.0
    from .. import limit

    return float(limit.total_memory_gb() or 0) * 1e9


def ensure_fits(cfg: ModelConfig, device: str = "cpu", training: bool = True) -> None:
    """Refuse, with the numbers, a model this machine cannot hold. Does nothing when the check is off or
    the machine's memory cannot be read."""
    if not CHECK:
        return
    have = available_bytes(device)
    if have <= 0:
        return
    need = weights(cfg) * (16 if training else 4)
    if need > have:
        raise SizeError(f"This size has {count(weights(cfg))} weights and needs about {memory(need)} of memory to "
                        f"{'train' if training else 'run'}; this computer has {memory(have)}. Choose a smaller size, or, on a "
                        "machine whose memory Gauge cannot measure, switch off the memory check (native.memory_check).")


def ladder() -> list[dict[str, Any]]:
    """Every named size, smallest first, with what it takes."""
    out = []
    for name, p in PRESETS.items():
        cfg = ModelConfig(**p)
        out.append({"size": name, "layers": cfg.n_layer, "width": cfg.n_embd, "heads": cfg.n_head, "context": cfg.block_size,
                    "vocab": cfg.vocab_size, **needs(cfg)})
    return out
