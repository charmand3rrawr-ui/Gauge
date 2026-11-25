"""What GaugeLM understands: how far it uses context, which words it is lost on, which words it thinks are alike.

A language model has no dictionary. It learns what a word means from the company the word keeps: the
words around it, over many passages. Two things inside the model carry that:

* each token has a row of numbers (its embedding). Words used in similar surroundings end up with
  similar rows, which is as close as the model gets to "meaning";
* attention lets each position look back over the passage, so the same word can be read differently
  in different surroundings. That is "context".

Neither can be given to it directly; they are learned. What can be done is to **measure** them, so it
is plain whether it is learning, and to **aim the teaching** at the words it understands least. This
module does the measuring:

* `context_use` - how much better it predicts the next piece of text when it can see the passage than
  when it can see only the last few tokens. If context means nothing to it, the two are the same;
* `hard_words` - the words in your library it is most surprised by, each with a sentence it appears in.
  The school (`school.py`) has a teacher explain exactly those, in that sentence;
* `neighbours` - the words it treats as most alike, from the embeddings.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

import numpy as np

from .backend import to_numpy
from .model import GaugeLM
from .tokenizer import PRETOKEN

SHORT = 4           # "almost no context": only this many tokens back


def _log_probs(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _ids(tok: Any, text: str, limit: int) -> np.ndarray:
    return np.array(tok.encode(text)[:limit], dtype=np.int64)


def context_use(model: GaugeLM, tok: Any, text: str, samples: int = 96, seed: int = 0) -> dict[str, Any]:
    """Predict the same next tokens twice: once seeing the passage (as far back as the model can), once
    seeing only the last few tokens. Returns both losses, the share by which context helps, how often
    the right token was its first guess or among its top five, and both as 'how many choices it is
    hesitating between' (the exponential of the loss), which is easier to picture than a loss."""
    long = model.cfg.block_size - 1
    ids = _ids(tok, text, 60_000)
    if len(ids) < long + 2 or long <= SHORT:
        return {"ok": False, "reason": "there is too little text to measure on"}
    rng = np.random.default_rng(seed)
    at = rng.integers(long, len(ids) - 1, size=min(samples, len(ids) - long - 1))
    full = np.stack([ids[t - long:t] for t in at])
    short = np.stack([ids[t - SHORT:t] for t in at])
    target = ids[at]
    lp_full = _log_probs(to_numpy(model.forward(full)[0])[:, -1, :])
    lp_short = _log_probs(to_numpy(model.forward(short)[0])[:, -1, :])
    rows = np.arange(len(at))
    loss_full, loss_short = float(-lp_full[rows, target].mean()), float(-lp_short[rows, target].mean())
    order = np.argsort(-lp_full, axis=-1)[:, :5]
    top1 = float((order[:, 0] == target).mean())
    top5 = float((order == target[:, None]).any(axis=1).mean())
    gain = 1.0 - loss_full / loss_short if loss_short > 0 else 0.0
    return {"ok": True, "tokens_of_context": long, "loss_with_context": loss_full, "loss_without": loss_short,
            "gain": gain, "choices_with_context": math.exp(loss_full), "choices_without": math.exp(loss_short),
            "first_guess_right": top1, "in_top_five": top5, "samples": int(len(at))}


def _sentence_around(text: str, start: int, end: int, limit: int = 260) -> str:
    a = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start), text.rfind("\n", 0, start)) + 1
    ends = [i for i in (text.find(".", end), text.find("!", end), text.find("?", end), text.find("\n", end)) if i != -1]
    b = (min(ends) + 1) if ends else len(text)
    return " ".join(text[a:b].split())[:limit]


def hard_words(model: GaugeLM, tok: Any, text: str, n: int = 10, skip: Optional[set[str]] = None,
               max_tokens: int = 6000, min_count: int = 2) -> list[dict[str, Any]]:
    """The words the model is most surprised by in `text`, most worth teaching first.

    Each word's surprise is the loss summed over the tokens that spell it, averaged over its
    appearances. Only ordinary words count: lower-case, four letters or more, seen at least
    `min_count` times (a word seen once teaches little). Names and numbers are left out. The ranking
    favours words that are both surprising and frequent, since those pay back most."""
    skip = skip or set()
    words: list[tuple[str, int, int, int, int]] = []       # (word, first token, token count, char start, char end)
    ids: list[int] = []
    for m in PRETOKEN.finditer(text):
        toks = tok._encode_word(m.group(0))
        if len(ids) + len(toks) > max_tokens:
            break
        words.append((m.group(0), len(ids), len(toks), m.start(), m.end()))
        ids.extend(toks)
    T = model.cfg.block_size
    if len(ids) < 8:
        return []
    arr = np.array(ids, dtype=np.int64)
    loss = np.zeros(len(arr), dtype=np.float64)
    # Windows that overlap by half, so every token is judged with a fair amount of context behind it.
    step = max(1, T // 2)
    for start in range(0, len(arr) - 1, step):
        x = arr[start:start + T]
        if len(x) < 2:
            break
        lp = _log_probs(to_numpy(model.forward(x[None, :-1])[0])[0])
        got = -lp[np.arange(len(x) - 1), x[1:]]
        lo = 0 if start == 0 else (len(x) - 1) - min(step, len(x) - 1)     # keep the later half, which had context
        loss[start + 1 + lo:start + len(x)] = got[lo:]
        if start + T >= len(arr):
            break
    seen: dict[str, dict[str, Any]] = {}
    for word, first, count, c0, c1 in words:
        bare = word.strip()
        if not re.fullmatch(r"[a-z]{4,}", bare) or bare in skip or first == 0:
            continue
        s = float(loss[first:first + count].sum())
        rec = seen.setdefault(bare, {"word": bare, "total": 0.0, "count": 0, "pieces": count, "where": (c0, c1)})
        rec["total"] += s
        rec["count"] += 1
    out = []
    for rec in seen.values():
        if rec["count"] < min_count:
            continue
        mean = rec["total"] / rec["count"]
        c0, c1 = rec["where"]
        out.append({"word": rec["word"], "surprise": round(mean, 3), "seen": rec["count"], "pieces": rec["pieces"],
                    "sentence": _sentence_around(text, c0, c1), "rank": mean * math.log(1.0 + rec["count"])})
    out.sort(key=lambda r: -r["rank"])
    return out[:n]


def neighbours(model: GaugeLM, tok: Any, word: str, k: int = 8) -> dict[str, Any]:
    """The whole-word tokens whose embeddings are closest to `word`'s: what the model treats as alike."""
    word = word.strip()
    pieces = tok.encode(" " + word)
    if len(pieces) != 1:
        return {"word": word, "ok": False, "pieces": [tok.decode([p]) for p in pieces],
                "reason": f"GaugeLM reads '{word}' as {len(pieces)} pieces, not one word. A larger vocabulary "
                          "(gauge native grow --vocab) gives common words a token of their own."}
    wte = to_numpy(model.p["wte"]).astype(np.float64)
    pool = [i for i, b in enumerate(tok.vocab) if re.fullmatch(rb" [a-z]{3,}", b) and i != pieces[0]]
    if not pool:
        return {"word": word, "ok": False, "pieces": [word], "reason": "the vocabulary has no other whole words yet"}
    v = wte[pieces[0]]
    m = wte[pool]
    sims = (m @ v) / (np.linalg.norm(m, axis=1) * np.linalg.norm(v) + 1e-12)
    best = np.argsort(-sims)[:k]
    return {"word": word, "ok": True, "pieces": [word],
            "alike": [{"word": tok.vocab[pool[i]].decode("utf-8", "replace").strip(), "closeness": round(float(sims[i]), 3)} for i in best]}


def report(model: GaugeLM, tok: Any, text: str, words: int = 8) -> dict[str, Any]:
    """How well it understands, in sentences a person can read, with the numbers behind them."""
    ctx = context_use(model, tok, text)
    hard = hard_words(model, tok, text, n=words)
    lines = []
    if ctx["ok"]:
        lines.append(f"Seeing the passage ({ctx['tokens_of_context']} tokens back) instead of only the last {SHORT} tokens, "
                     f"it hesitates between about {ctx['choices_with_context']:.0f} next pieces instead of "
                     f"{ctx['choices_without']:.0f}: context makes it {ctx['gain']:.0%} surer.")
        lines.append(f"Its first guess for the next piece is right {ctx['first_guess_right']:.0%} of the time, "
                     f"and the right one is among its top five {ctx['in_top_five']:.0%} of the time.")
        if ctx["gain"] < 0.05:
            lines.append("That is almost no use of context yet: it is mostly guessing from the last few tokens. "
                         "More training, more text and a longer reading length are what change this.")
    else:
        lines.append("Not enough text to measure how it uses context: " + ctx["reason"] + ".")
    if hard:
        lines.append("The words it is most lost on: " + ", ".join(h["word"] for h in hard) + ".")
    return {"context": ctx, "hard_words": hard, "lines": lines, "line": " ".join(lines)}
