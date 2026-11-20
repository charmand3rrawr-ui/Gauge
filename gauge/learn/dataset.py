"""Export training data for optional LoRA fine-tuning of the local model.

Gauge does not fine-tune inside the process (that needs a GPU toolchain). It writes clean,
standard files you can hand to Unsloth / Axolotl / LLaMA-Factory, then load into Ollama:

* ``sft.jsonl``        - {"instruction", "input", "output"} rows: outline beat -> chapter prose
* ``preference.jsonl`` - {"prompt", "chosen", "rejected"} rows for DPO/ORPO: passages from
  high-engagement novels vs. matched passages from low-engagement novels
* ``Modelfile``        - an Ollama Modelfile that bakes the learned style guide into a system prompt

Only passages (<=180 words) are exported for preference pairs, and SFT rows are built from
your own ``local`` imports by default, to keep the dataset within fair-use style learning
rather than wholesale reproduction. See docs/ETHICS.md.
"""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from ..db import Database
from ..models import NovelRecord
from ..text import head, tail
from .engagement import TARGETS


def export_preference_pairs(db: Database, novels: Sequence[NovelRecord], target: str,
                            out_path: Path, pairs: int = 500, seed: int = 7) -> int:
    fn = TARGETS[target]
    scored = [(fn(n), n) for n in novels if fn(n) is not None]
    scored.sort(key=lambda t: t[0])
    if len(scored) < 6:
        return 0
    cut = max(1, len(scored) // 3)
    low = [n for _, n in scored[:cut]]
    high = [n for _, n in scored[-cut:]]
    rng = random.Random(seed)
    rows = []
    for _ in range(pairs * 3):
        if len(rows) >= pairs:
            break
        hn, ln = rng.choice(high), rng.choice(low)
        hc, lc = db.chapters(hn.key), db.chapters(ln.key)
        if not hc or not lc:
            continue
        kind = rng.choice(["opening", "ending"])
        h_ch, l_ch = rng.choice(hc), rng.choice(lc)
        pick = head if kind == "opening" else tail
        prompt = (f"Write the {kind} of a serial web novel chapter in the "
                  f"{', '.join((hn.genres + hn.tags)[:3]) or 'web fiction'} genre so that the reader "
                  f"{'cannot stop reading' if kind == 'opening' else 'must click to the next chapter'}.")
        rows.append({"prompt": prompt, "chosen": pick(h_ch.text, 180), "rejected": pick(l_ch.text, 180)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def export_sft(db: Database, novels: Sequence[NovelRecord], out_path: Path,
               sources: Sequence[str] = ("local",), max_rows: int = 2000) -> int:
    rows: list[dict[str, str]] = []
    for n in novels:
        if n.source not in sources:
            continue
        for ch in db.chapters(n.key):
            if len(rows) >= max_rows:
                break
            summary = head(ch.text, 60)
            rows.append({
                "instruction": "Write the next chapter of this serial web novel. Open on a hook, "
                               "escalate, and end unresolved.",
                "input": f"Novel: {n.title}\nGenres: {', '.join(n.genres + n.tags)}\n"
                         f"Chapter {ch.index} title: {ch.title}\nIt begins: {summary}...",
                "output": ch.text,
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def write_modelfile(base_model: str, style_guide_compact: str, out_path: Path,
                    temperature: float = 0.9) -> Path:
    system = (
        "You are a web novel author whose chapters readers binge. Follow these evidence-based "
        "rules learned from reader engagement data:\n" + style_guide_compact
    ).replace('"', "'")
    text = (
        f"FROM {base_model}\n"
        f"PARAMETER temperature {temperature}\n"
        "PARAMETER num_ctx 8192\n"
        f"SYSTEM \"\"\"{system}\"\"\"\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
