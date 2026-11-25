"""Training GaugeLM on your corpus, and keeping it learning as new chapters arrive.

The corpus is everything Gauge has read: ingested chapters, your imported novels, public-domain
books, your own project chapters, plus the instruction rows the pipeline accumulates (sft,
self-preference "chosen" sides, author edits) in a prompt/response format so the model learns to
answer a prompt and not only to continue prose. Training is plain AdamW with warmup and cosine
decay, checkpointed every N steps; ``continue_training`` picks up from the last checkpoint with
only the text seen since, which is what the daemon calls each cycle.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..db import Database
from .backend import device_name
from .model import PRESETS, AdamW, GaugeLM, ModelConfig
from .tokenizer import BPETokenizer

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 8
    accum_steps: int = 1  # gradient accumulation: effective batch = batch_size * accum_steps
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    checkpoint_every: int = 200
    max_corpus_chars: int = 20_000_000
    chat_weight: int = 3  # instruction rows are repeated this many times (they are few and matter most)
    curriculum: bool = True  # prose first, instruction rows concentrated in the last third of training
    device: str = "cpu"  # cpu | cuda | auto
    # Train on several devices at once ("cuda:0", "cuda:1", "cpu"); the first is where the model lives. Empty = just `device`.
    devices: list[str] = field(default_factory=list)
    seed: int = 0


def preset_config(size: str, vocab_size: Optional[int] = None) -> ModelConfig:
    if size not in PRESETS:
        raise ValueError(f"unknown size {size!r}; choose from {', '.join(PRESETS)}")
    cfg = ModelConfig(**PRESETS[size])
    if vocab_size:
        cfg.vocab_size = vocab_size
    return cfg


# --------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------
def gather_texts(db: Database, projects_dir: Optional[Path] = None, datasets_dir: Optional[Path] = None,
                 since: Optional[str] = None) -> list[tuple[str, str]]:
    """(kind, text) documents. kind is 'prose' or 'chat:<prompt>' for instruction rows."""
    docs: list[tuple[str, str]] = []
    for novel in db.novels():
        for ch in db.chapters(novel.key):
            if since and ch.fetched_at.isoformat() < since:
                continue
            docs.append(("prose", ch.text))
    if projects_dir and projects_dir.exists():
        from ..write.project import Project

        for name in Project.list(projects_dir):
            p = Project(projects_dir, name)
            for n in range(1, p.state.written + 1):
                t = p.read_chapter(n)
                if t:
                    import re

                    docs.append(("prose", re.sub(r"^#[^\n]*\n+", "", t)))
    docs.extend(instruction_rows(datasets_dir))
    return dedup(docs)


def instruction_rows(datasets_dir: Optional[Path]) -> list[tuple[str, str]]:
    """The rows that teach behaviour rather than prose: curated instruction data, the preferred side of
    draft-versus-accepted pairs, the author's edits, the big model's answers to Gauge's own tasks
    (distillation), the character school's lessons, a teacher's corrections of GaugeLM's own attempts, and the
    graded practice scenes teachers wrote for it. Each is ('chat:<prompt>', answer). All of it was produced on this machine."""
    rows: list[tuple[str, str]] = []
    if not datasets_dir or not datasets_dir.exists():
        return rows
    for name in ("sft.jsonl", "self_preference.jsonl", "author_edits.jsonl", "distill.jsonl", "characters.jsonl",
                 "tutor.jsonl", "lessons.jsonl", "words.jsonl"):
        f = datasets_dir / name
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "output" in r:
                rows.append((f"chat:{r.get('instruction', '')}\n{r.get('input', '')}", r["output"]))
            elif "chosen" in r:
                rows.append((f"chat:{r.get('prompt', '')}", r["chosen"]))
            elif "response" in r:
                rows.append((f"chat:{r.get('prompt', '')}", r["response"]))
    return rows


def dedup(docs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop exact and near-exact duplicates (same first 400 chars), keeping first occurrence."""
    import hashlib

    seen: set[str] = set()
    out = []
    for kind, text in docs:
        key = hashlib.sha1((kind[:40] + text[:400]).encode("utf-8", errors="replace")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, text))
    return out


def encode_corpus(tok: BPETokenizer, docs: list[tuple[str, str]], max_chars: int, chat_weight: int = 1,
                  split: bool = False):
    """Token stream(s). With split=True returns (prose, chat) so training can schedule them."""
    prose: list[int] = []
    chat: list[int] = []
    used = 0
    for kind, text in docs:
        if used >= max_chars:
            break
        text = text[: max_chars - used]
        used += len(text)
        if kind.startswith("chat:"):
            row = tok.encode_chat(kind[5:], text)
            for _ in range(max(1, chat_weight)):
                chat.extend(row)
        else:
            prose.extend(tok.encode(text, bos=True, eos=True))
    if split:
        return np.asarray(prose, dtype=np.int32), np.asarray(chat, dtype=np.int32)
    return np.asarray(prose + chat, dtype=np.int32)


def batches(data: np.ndarray, block: int, batch: int, rng: np.random.Generator):
    n = len(data) - block - 1
    if n <= 0:
        raise ValueError(f"corpus too small ({len(data)} tokens) for block_size {block}; ingest or import more text")
    while True:
        ix = rng.integers(0, n, size=batch)
        x = np.stack([data[i:i + block] for i in ix])
        y = np.stack([data[i + 1:i + block + 1] for i in ix])
        yield x, y


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if step >= cfg.steps:
        return cfg.min_lr
    frac = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * frac))


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
class NativeTrainer:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.tok_path = model_dir / "tokenizer.json"
        self.state_path = model_dir / "train_state.json"

    # ---- status -----------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        st = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        meta = json.loads((self.dir / "model.json").read_text(encoding="utf-8")) if (self.dir / "model.json").exists() else {}
        return {"trained": bool(meta), "n_params": meta.get("n_params"), "config": meta.get("config"), **st}

    def exists(self) -> bool:
        return (self.dir / "model.json").exists() and self.tok_path.exists()

    def load(self, device: str = "cpu") -> tuple[GaugeLM, BPETokenizer]:
        return GaugeLM.load(self.dir, device=device), BPETokenizer.load(self.tok_path)

    # ---- tokenizer -----------------------------------------------------------------
    def build_tokenizer(self, docs: list[tuple[str, str]], vocab_size: int, max_chars: int = 3_000_000) -> BPETokenizer:
        text = "\n".join(t for _, t in docs)
        tok = BPETokenizer.train(text, vocab_size=vocab_size, max_chars=max_chars)
        tok.save(self.tok_path)
        return tok

    # ---- training ------------------------------------------------------------------
    def train(self, docs: list[tuple[str, str]], mcfg: ModelConfig, tcfg: TrainConfig, resume: bool = False,
              progress: Optional[Callable[[dict[str, Any]], None]] = None, time_budget_s: Optional[float] = None,
              should_stop: Optional[Callable[[], bool]] = None) -> dict[str, Any]:
        """`should_stop` is asked after every step; when it says yes the model is saved and training returns,
        so a pause never loses work and `resume=True` carries on from that step."""
        rng = np.random.default_rng(tcfg.seed)
        if resume and self.exists():
            model, tok = self.load(device=tcfg.device)
            st = self.status()
            step0 = int(st.get("step", 0))
        else:
            tok = self.build_tokenizer(docs, mcfg.vocab_size)
            mcfg.vocab_size = tok.vocab_size
            model = GaugeLM(mcfg, seed=tcfg.seed, device=tcfg.device)
            step0 = 0
        xp = model.xp
        prose, chat = encode_corpus(tok, docs, tcfg.max_corpus_chars, chat_weight=tcfg.chat_weight, split=True)
        data = np.concatenate([prose, chat]) if len(chat) else prose
        if len(data) < model.cfg.block_size * 4:
            raise ValueError(f"corpus too small: {len(data)} tokens. Ingest or import more text first.")
        n_val = max(model.cfg.block_size * 2, len(data) // 20)
        train_data, val_data = data[:-n_val], data[-n_val:]
        block = model.cfg.block_size
        gen_all = batches(train_data, block, tcfg.batch_size, rng)
        gen_prose = batches(prose, block, tcfg.batch_size, rng) if len(prose) > block + 1 else gen_all
        gen_chat = batches(chat, block, tcfg.batch_size, rng) if len(chat) > block + 1 else gen_all
        opt = AdamW(model.p, lr=tcfg.lr, weight_decay=tcfg.weight_decay, xp=xp)
        t0 = time.monotonic()
        losses: list[float] = []
        tokens_seen = int(self.status().get("tokens_seen", 0)) if resume else 0
        last_val = None
        step = step0
        total = step0 + tcfg.steps
        log.info("training on %s: %s params, %d train tokens (%d chat)", device_name(xp), f"{model.n_params():,}", len(train_data), len(chat))
        from .. import limit

        governor = limit.compute()
        on_gpu = xp is not np
        stopped = False
        team = None
        if len(tcfg.devices) > 1:                # several devices share every step (see parallel.py)
            from .parallel import Replicas

            team = Replicas(model, list(tcfg.devices))
            log.info("native train: %s", team.describe())
        while step < total:
            step_t0 = time.monotonic()
            # Curriculum: mostly prose early, instruction rows concentrated late (they teach behaviour).
            frac = (step - step0) / max(1, tcfg.steps)
            grads = None
            def draw(frac: float = frac):
                if tcfg.curriculum and len(chat) > block + 1 and rng.random() < (0.15 if frac < 0.66 else 0.6):
                    return next(gen_chat)
                if tcfg.curriculum and len(prose) > block + 1:
                    return next(gen_prose)
                return next(gen_all)

            for _ in range(max(1, tcfg.accum_steps)):
                if team is not None and len(team) > 1:      # one batch per device, at the same time
                    g, loss, seen = team.step([draw() for _ in range(len(team))])
                else:
                    x, y = draw()
                    _, loss, caches = model.forward(x, y)
                    g = model.backward(caches)
                    seen = x.size
                if grads is None:
                    grads = g
                else:
                    for k in grads:
                        grads[k] += g[k]
                losses.append(loss)
                tokens_seen += seen
            if tcfg.accum_steps > 1:
                for k in grads:
                    grads[k] /= tcfg.accum_steps
            norm = opt.step(model.p, grads, lr=lr_at(step - step0, tcfg), clip=tcfg.grad_clip)
            if team is not None:
                team.sync()
            step += 1
            if step % tcfg.eval_every == 0 or step == total:
                last_val = self.evaluate(model, val_data, rng, batches_n=4)
                rec = {"step": step, "loss": round(float(np.mean(losses[-tcfg.eval_every:])), 4), "val_loss": round(last_val, 4),
                       "grad_norm": round(norm, 3), "tokens_seen": tokens_seen, "tok_per_s": round(tokens_seen / max(1e-6, time.monotonic() - t0))}
                log.info("native train %s", rec)
                if progress:
                    progress(rec)
            if step % tcfg.checkpoint_every == 0 or step == total:
                self._save(model, step, tokens_seen, last_val, losses[-1])
            if time_budget_s is not None and time.monotonic() - t0 > time_budget_s:
                self._save(model, step, tokens_seen, last_val, losses[-1])
                break
            if should_stop is not None and should_stop():
                self._save(model, step, tokens_seen, last_val, losses[-1])
                stopped = True
                break
            # The capacity limit: rest after each step so this device averages its share (a graphics card
            # runs flat out while it works; the processor is already pinned to its cores).
            governor.pace(time.monotonic() - step_t0, 1.0 if on_gpu else None)
        return {"step": step, "loss": round(float(np.mean(losses[-50:])), 4) if losses else None,
                "val_loss": round(last_val, 4) if last_val is not None else None, "tokens_seen": tokens_seen,
                "seconds": round(time.monotonic() - t0, 1), "n_params": model.n_params(), "corpus_tokens": int(len(data)),
                "device": device_name(xp), "arch": model.cfg.arch, "stopped": stopped,
                "devices": team.active if team is not None else [], "rested": team.rested if team is not None else []}

    def evaluate(self, model: GaugeLM, data: np.ndarray, rng: np.random.Generator, batches_n: int = 4) -> float:
        if len(data) <= model.cfg.block_size + 1:
            return float("nan")
        gen = batches(data, model.cfg.block_size, 4, rng)
        vals = []
        for _ in range(batches_n):
            x, y = next(gen)
            vals.append(model.forward(x, y)[1])
        return float(np.mean(vals))

    def _save(self, model: GaugeLM, step: int, tokens_seen: int, val_loss: Optional[float], loss: float) -> None:
        model.save(self.dir, {"step": step})
        self.state_path.write_text(json.dumps({"step": step, "tokens_seen": tokens_seen, "val_loss": val_loss, "loss": loss,
                                               "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1), encoding="utf-8")

    def continue_training(self, docs: list[tuple[str, str]], steps: int, time_budget_s: Optional[float] = None,
                          progress: Optional[Callable[[dict[str, Any]], None]] = None, device: str = "cpu") -> dict[str, Any]:
        """Keep learning from new material without resetting weights (the daemon's path)."""
        if not self.exists():
            raise FileNotFoundError("no native model yet; run `gauge native train` first")
        model, _ = self.load(device=device)
        tcfg = TrainConfig(steps=steps, warmup=min(20, steps), lr=2e-4, min_lr=5e-5, eval_every=max(10, steps // 4),
                           checkpoint_every=max(10, steps // 2), device=device)
        return self.train(docs, model.cfg, tcfg, resume=True, progress=progress, time_budget_s=time_budget_s)


def config_dict(mcfg: ModelConfig, tcfg: TrainConfig) -> dict[str, Any]:
    return {"model": asdict(mcfg), "train": asdict(tcfg)}
