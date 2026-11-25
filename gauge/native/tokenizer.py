"""Byte-level BPE tokenizer, pure Python. Trained on your own corpus, saved as JSON.

Starts from the 256 byte values, learns merges on the most frequent adjacent pairs inside
pre-tokenized words (a GPT-2 style split on words, numbers, punctuation and whitespace), and adds
a few special tokens the model uses to tell prompt from response.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

PRETOKEN = re.compile(r" ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+")
SPECIALS = ["<pad>", "<bos>", "<eos>", "<prompt>", "<response>"]


class BPETokenizer:
    def __init__(self, merges: Optional[list[tuple[int, int]]] = None):
        self.merges: list[tuple[int, int]] = list(merges or [])
        self._rebuild()

    # ---- vocabulary bookkeeping -------------------------------------------------
    def _rebuild(self) -> None:
        self.ranks: dict[tuple[int, int], int] = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.special: dict[str, int] = {}
        for i, s in enumerate(SPECIALS):
            self.special[s] = len(self.vocab) + i
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(SPECIALS)

    @property
    def pad(self) -> int:
        return self.special["<pad>"]

    @property
    def bos(self) -> int:
        return self.special["<bos>"]

    @property
    def eos(self) -> int:
        return self.special["<eos>"]

    # ---- training -----------------------------------------------------------------
    @staticmethod
    def _learn(seqs: dict[tuple[int, ...], int], merges: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
        """Add merges of the most frequent adjacent pair until there are `target` of them or no pair repeats."""
        next_id = 256 + len(merges)
        while len(merges) < target:
            pairs: Counter = Counter()
            for seq, n in seqs.items():
                for a, b in zip(seq, seq[1:]):
                    pairs[(a, b)] += n
            if not pairs:
                break
            (a, b), count = pairs.most_common(1)[0]
            if count < 2:
                break
            merges.append((a, b))
            new: dict[tuple[int, ...], int] = {}
            for seq, n in seqs.items():
                if a in seq:
                    out = []
                    i = 0
                    while i < len(seq):
                        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                            out.append(next_id)
                            i += 2
                        else:
                            out.append(seq[i])
                            i += 1
                    seq = tuple(out)
                new[seq] = new.get(seq, 0) + n
            seqs = new
            next_id += 1
        return merges

    @classmethod
    def train(cls, text: str, vocab_size: int = 4096, max_chars: int = 3_000_000) -> BPETokenizer:
        words = Counter(PRETOKEN.findall(text[:max_chars]))
        seqs: dict[tuple[int, ...], int] = {tuple(w.encode("utf-8")): n for w, n in words.items()}
        return cls(cls._learn(seqs, [], max(0, vocab_size - 256 - len(SPECIALS))))

    def extended(self, text: str, vocab_size: int, max_chars: int = 3_000_000) -> BPETokenizer:
        """A tokenizer with a larger vocabulary that keeps every token this one has. The new tokens are
        further merges learned on `text` as this tokenizer cuts it, so their ids come after the old merges
        (the special tokens stay last and so move up by the number of tokens added)."""
        words = Counter(PRETOKEN.findall(text[:max_chars]))
        seqs: dict[tuple[int, ...], int] = {}
        for w, n in words.items():
            seq = tuple(self._encode_word(w))
            seqs[seq] = seqs.get(seq, 0) + n
        return BPETokenizer(self._learn(seqs, list(self.merges), max(0, vocab_size - 256 - len(SPECIALS))))

    # ---- encode / decode -----------------------------------------------------------
    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        seq = list(word.encode("utf-8"))
        while len(seq) >= 2:
            best, best_rank = None, None
            for i in range(len(seq) - 1):
                r = self.ranks.get((seq[i], seq[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            seq = seq[:best] + [256 + best_rank] + seq[best + 2:]
        if len(self._cache) < 200_000:
            self._cache[word] = seq
        return seq

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids: list[int] = [self.bos] if bos else []
        for w in PRETOKEN.findall(text):
            ids.extend(self._encode_word(w))
        if eos:
            ids.append(self.eos)
        return ids

    def encode_chat(self, prompt: str, response: str = "") -> list[int]:
        ids = [self.bos, self.special["<prompt>"]] + self.encode(prompt) + [self.special["<response>"]]
        if response:
            ids += self.encode(response) + [self.eos]
        return ids

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            if i < len(self.vocab):
                out += self.vocab[i]
            # special tokens are dropped from text
        return out.decode("utf-8", errors="replace")

    # ---- persistence --------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"merges": self.merges, "specials": SPECIALS}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls([tuple(m) for m in d["merges"]])
