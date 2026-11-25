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
from dataclasses import asdict, dataclass
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
                 since: Optional[str] = None, coherence_floor: Optional[float] = None) -> list[tuple[str, str]]:
    """(kind, text) documents. kind is 'prose' or 'chat:<prompt>' for instruction rows. Your own chapters
    under ``coherence_floor`` (default: prose.COHERENCE_FLOOR; 0 = keep everything) are left out: a chapter
    saved before the floor existed may be the noise an under-trained model wrote, and training on it
    teaches the model its own noise."""
    from ..write.prose import COHERENCE_FLOOR

    floor = COHERENCE_FLOOR if coherence_floor is None else coherence_floor
    docs: list[tuple[str, str]] = []
    for novel in db.novels():
        for ch in db.chapters(novel.key):
            if since and ch.fetched_at.isoformat() < since:
                continue
            docs.append(("prose", ch.text))
    if projects_dir and projects_dir.exists():
        from ..write.project import Project

        for name in Project.list(projects_dir):
            readable, _ = Project(projects_dir, name).training_chapters(floor)
            docs.extend(("prose", t) for _, t in readable)
    docs.extend(instruction_rows(datasets_dir, floor))
    return dedup(docs)


def instruction_rows(datasets_dir: Optional[Path], coherence_floor: Optional[float] = None) -> list[tuple[str, str]]:
    """The rows that teach behaviour rather than prose: curated instruction data, the preferred side of
    draft-versus-accepted pairs, the author's edits, and the big model's answers to Gauge's own tasks
    (distillation). Each is ('chat:<prompt>', answer). All of it was produced on this machine.

    A preferred side that is the text of one of your chapters (the row names its project and chapter) is
    held to the same coherence floor as the chapter files, so an unreadable chapter accepted before the
    floor existed is not learned from this way either. Other answers are short or structured, and the
    measure does not apply to them."""
    from ..write.prose import COHERENCE_FLOOR, coherence

    floor = COHERENCE_FLOOR if coherence_floor is None else coherence_floor
    rows: list[tuple[str, str]] = []
    if not datasets_dir or not datasets_dir.exists():
        return rows
    skipped = 0
    for name in ("sft.jsonl", "self_preference.jsonl", "author_edits.jsonl", "distill.jsonl"):
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
                if floor > 0 and "project" in r and "chapter" in r and coherence(str(r["chosen"])).score < floor:
                    skipped += 1
                    continue
                rows.append((f"chat:{r.get('prompt', '')}", r["chosen"]))
            elif "response" in r:
                rows.append((f"chat:{r.get('prompt', '')}", r["response"]))
    if skipped:
        log.info("%d dataset row(s) of chapter text under the coherence floor (%.2f) left out of the training text; "
                 "see `gauge chapters check`", skipped, floor)
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
              progress: Optional[Callable[[dict[str, Any]], None]] = None, time_budget_s: Optional[float] = None) -> dict[str, Any]:
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
        while step < total:
            # Curriculum: mostly prose early, instruction rows concentrated late (they teach behaviour).
            frac = (step - step0) / max(1, tcfg.steps)
            grads = None
            for _ in range(max(1, tcfg.accum_steps)):
                if tcfg.curriculum and len(chat) > block + 1 and rng.random() < (0.15 if frac < 0.66 else 0.6):
                    x, y = next(gen_chat)
                elif tcfg.curriculum and len(prose) > block + 1:
                    x, y = next(gen_prose)
                else:
                    x, y = next(gen_all)
                _, loss, caches = model.forward(x, y)
                g = model.backward(caches)
                if grads is None:
                    grads = g
                else:
                    for k in grads:
                        grads[k] += g[k]
                losses.append(loss)
                tokens_seen += x.size
            if tcfg.accum_steps > 1:
                for k in grads:
                    grads[k] /= tcfg.accum_steps
            norm = opt.step(model.p, grads, lr=lr_at(step - step0, tcfg), clip=tcfg.grad_clip)
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
        return {"step": step, "loss": round(float(np.mean(losses[-50:])), 4) if losses else None,
                "val_loss": round(last_val, 4) if last_val is not None else None, "tokens_seen": tokens_seen,
                "seconds": round(time.monotonic() - t0, 1), "n_params": model.n_params(), "corpus_tokens": int(len(data)),
                "device": device_name(xp), "arch": model.cfg.arch}

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
