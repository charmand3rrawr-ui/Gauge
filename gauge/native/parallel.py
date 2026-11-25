"""Training on several devices at once.

Each chosen device holds a copy of the model. On every step each copy works through its own batch at
the same time, the gradients are averaged on the first device, the optimiser steps there, and the new
weights are copied back out. That is the same as one step on a batch as many times larger, in about
the time of one batch: more text learned from per step, not a different result.

A device much slower than the fastest (a processor next to a graphics card) would make every step wait
for it, so it is timed on the first step and rested if it cannot keep up.

Device names: ``"cpu"`` for the processor, ``"cuda:0"``, ``"cuda:1"`` for graphics cards in the order
Gauge was allowed to see them. The first name is where the model itself lives.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any, Optional

import numpy as np

from .backend import from_numpy, to_numpy
from .model import GaugeLM

log = logging.getLogger(__name__)

TOO_SLOW = 4.0          # a device this many times slower than the fastest is rested


def split_device(spec: str) -> tuple[str, int]:
    """'cuda:1' -> ('cuda', 1); 'cpu' -> ('cpu', 0)."""
    kind, _, idx = spec.partition(":")
    return kind, int(idx) if idx.isdigit() else 0


def device_context(spec: str) -> Any:
    """Something to run a device's work inside: the graphics card it belongs to, or nothing for the processor."""
    kind, idx = split_device(spec)
    if kind != "cuda":
        return contextlib.nullcontext()
    import cupy as cp  # type: ignore

    return cp.cuda.Device(idx)


class Replica:
    def __init__(self, spec: str, model: GaugeLM):
        self.spec = spec
        self.model = model
        self.seconds = 0.0
        self.error: Optional[BaseException] = None
        self.loss = 0.0
        self.grads: dict[str, Any] = {}


class Replicas:
    """Copies of one model on several devices, kept identical, that share each training step."""

    def __init__(self, model: GaugeLM, specs: list[str]):
        if not specs:
            raise ValueError("no devices given")
        self.primary = Replica(specs[0], model)
        self.others: list[Replica] = []
        self.rested: list[str] = []
        self._timed = False
        for spec in specs[1:]:
            try:
                with device_context(spec):
                    copy = GaugeLM(model.cfg, seed=0, device="cuda" if split_device(spec)[0] == "cuda" else "cpu")
                    self._limit_memory(copy)
                self.others.append(Replica(spec, copy))
            except Exception as e:  # a card that cannot be used is left out, never fatal
                log.warning("cannot train on %s: %s", spec, e)
                self.rested.append(spec)
        self.sync()

    @staticmethod
    def _limit_memory(model: GaugeLM) -> None:
        """The capacity limit, on each graphics card separately (the pool's limit is per device)."""
        if model.xp is np:
            return
        from .. import limit

        share = limit.gpu_memory_fraction()
        if share is not None:
            try:
                model.xp.get_default_memory_pool().set_limit(fraction=share)
            except Exception as e:
                log.warning("could not cap graphics memory: %s", e)

    @property
    def active(self) -> list[str]:
        return [self.primary.spec] + [r.spec for r in self.others]

    def __len__(self) -> int:
        return 1 + len(self.others)

    def sync(self) -> None:
        """Copy the first device's weights to every other device."""
        if not self.others:
            return
        host = {k: to_numpy(v) for k, v in self.primary.model.p.items()}
        for r in self.others:
            with device_context(r.spec):
                r.model.p = {k: from_numpy(r.model.xp, v) for k, v in host.items()}

    def _work(self, r: Replica, x: np.ndarray, y: np.ndarray) -> None:
        t0 = time.monotonic()
        try:
            with device_context(r.spec):
                _, r.loss, caches = r.model.forward(x, y)
                grads = r.model.backward(caches)
                # Bring the gradients to the host inside the timing, so the measure is of the whole job.
                r.grads = grads if r is self.primary else {k: to_numpy(v) for k, v in grads.items()}
        except BaseException as e:  # reported by step(); a thread must not die silently
            r.error = e
        r.seconds = time.monotonic() - t0

    def step(self, batches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, Any], float, int]:
        """One shared step. `batches` holds one (x, y) per active device. Returns the averaged gradients on
        the first device, the mean loss, and the number of tokens read."""
        team = [self.primary] + self.others
        if len(batches) != len(team):
            raise ValueError(f"{len(team)} devices need {len(team)} batches, got {len(batches)}")
        threads = []
        for r, (x, y) in zip(team[1:], batches[1:]):
            r.error = None
            t = threading.Thread(target=self._work, args=(r, x, y), daemon=True)
            t.start()
            threads.append(t)
        self.primary.error = None
        self._work(self.primary, *batches[0])
        for t in threads:
            t.join()
        if self.primary.error is not None:
            raise self.primary.error
        good = [self.primary]
        for r in list(self.others):
            if r.error is not None:           # a device that fails is dropped and training carries on
                log.warning("%s stopped working (%s); carrying on without it", r.spec, r.error)
                self.others.remove(r)
                self.rested.append(r.spec)
            else:
                good.append(r)
        xp = self.primary.model.xp
        grads = self.primary.grads
        for r in good[1:]:
            for k, g in r.grads.items():
                grads[k] += from_numpy(xp, g)
        if len(good) > 1:
            for k in grads:
                grads[k] /= len(good)
        loss = float(np.mean([r.loss for r in good]))
        tokens = int(sum(batches[i][0].size for i, r in enumerate(team) if r in good))
        if not self._timed:
            self._rest_the_slow(good)
        return grads, loss, tokens

    def _rest_the_slow(self, team: list[Replica]) -> None:
        """After the first shared step: a device far slower than the fastest would hold every step back."""
        self._timed = True
        fastest = min(r.seconds for r in team)
        for r in list(self.others):
            if r in team and fastest > 0 and r.seconds > TOO_SLOW * fastest and r.seconds > 0.05:
                log.info("%s took %.2fs against %.2fs on the fastest device; resting it", r.spec, r.seconds, fastest)
                self.others.remove(r)
                self.rested.append(r.spec)

    def describe(self) -> str:
        line = "Training on " + " and ".join(self.active) if len(self) > 1 else f"Training on {self.primary.spec}"
        if self.rested:
            line += f" ({', '.join(self.rested)} rested: too slow to keep up, or not usable)"
        return line + "."
