"""Fine-tuning data from Gauge's own work, and a runner for the optional LoRA step.

Two sources beat the corpus for teaching a model *your* story in *your* taste:

* **self-preference pairs**: the first draft versus the accepted revision, whenever revising
  improved the score (``datasets/self_preference.jsonl``);
* **author edits**: when you edit a chapter file by hand, the original Gauge wrote versus the
  text you kept (``datasets/author_edits.jsonl``). Detected by comparing the saved copy of the
  accepted chapter with the file on disk.

``gauge finetune`` validates the datasets and, when a CUDA GPU and Unsloth are present, runs
``scripts/finetune_lora.py``; otherwise it prints the exact command to run on a machine that has
them. Gauge never trains silently.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid a circular import with the write package
    from ..write.project import Project


def _strip_front(text: str) -> str:
    text = re.sub(r"^---.*?---\s*", "", text, flags=re.S)
    return re.sub(r"^#[^\n]*\n+", "", text).strip()


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_self_pair(datasets_dir: Path, project: str, chapter: int, prompt_summary: str, rejected: str, chosen: str,
                     score_before: float, score_after: float) -> Optional[Path]:
    if score_after <= score_before or not rejected or not chosen or rejected.strip() == chosen.strip():
        return None
    path = datasets_dir / "self_preference.jsonl"
    _append(path, {"prompt": prompt_summary, "chosen": chosen, "rejected": rejected, "project": project, "chapter": chapter,
                   "score_before": round(score_before, 3), "score_after": round(score_after, 3)})
    return path


def save_accepted_copy(project: Project, chapter: int, text: str) -> Path:
    d = project.dir / "drafts" / "accepted"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"chapter_{chapter:03d}.md"
    p.write_text(text, encoding="utf-8")
    return p


def build_sft_from_project(project: Project, datasets_dir: Path, coherence_floor: Optional[float] = None) -> int:
    """Instruction-tuning data from a serial's own accepted chapters: each row pairs a short
    instruction (derived from the chapter's stored plan) with the chapter text as the completion.
    This is how GaugeLM learns from what Gauge itself judged good. A chapter under ``coherence_floor``
    (default prose.COHERENCE_FLOOR; 0 = keep everything) was accepted before the floor existed and is
    not a lesson: no row is written for it. Returns rows written."""
    from ..write.prose import COHERENCE_FLOOR

    unreadable = {c.number for c in project.training_chapters(COHERENCE_FLOOR if coherence_floor is None else coherence_floor)[1]}
    path = datasets_dir / "sft.jsonl"
    existing = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                existing.add((r.get("project"), r.get("chapter")))
            except json.JSONDecodeError:
                continue
    name = project.state.name
    written = 0
    for n in range(1, project.state.written + 1):
        if (name, n) in existing or n in unreadable:
            continue
        text = project.read_chapter(n)
        if not text or len(text.split()) < 50:
            continue
        meta = project.read_meta(n) if hasattr(project, "read_meta") else {}
        title = meta.get("title") or f"Chapter {n}"
        direction = meta.get("direction") or ""
        prompt = f"Write chapter {n} of \"{name}\", titled \"{title}\"." + (f" Direction: {direction}" if direction else "")
        _append(path, {"prompt": prompt, "completion": _strip_front(text), "project": name, "chapter": n,
                       "score": meta.get("score")})
        written += 1
    return written


@dataclass
class DatasetRowCheck:
    file: str  # dataset file name, e.g. "self_preference.jsonl"
    line: int  # 1-based line of the row in that file
    chapter: int
    coherence: float


def unreadable_rows(datasets_dir: Path, project: str, coherence_floor: Optional[float] = None) -> list[DatasetRowCheck]:
    """Dataset rows made from ``project``'s chapters whose kept text (the SFT completion, or the chosen
    side of a preference pair) is under the coherence floor: the same unreadable chapter, copied into the
    fine-tuning data before the floor existed. Reads only; the author decides what to delete."""
    from ..write.prose import COHERENCE_FLOOR, coherence

    floor = COHERENCE_FLOOR if coherence_floor is None else coherence_floor
    out: list[DatasetRowCheck] = []
    for name in ("sft.jsonl", "self_preference.jsonl", "author_edits.jsonl"):
        f = datasets_dir / name
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = r.get("completion") or r.get("chosen") if isinstance(r, dict) else None
            if not text or r.get("project") != project or not isinstance(r.get("chapter"), int):
                continue
            score = coherence(str(text), floor).score
            if score < floor:
                out.append(DatasetRowCheck(name, i, r["chapter"], score))
    return out


def sync_author_edits(project: Project, datasets_dir: Path) -> list[dict[str, Any]]:
    """Compare each chapter on disk with the copy Gauge accepted; record edits as preference pairs."""
    out = []
    accepted_dir = project.dir / "drafts" / "accepted"
    if not accepted_dir.exists():
        return out
    seen_path = accepted_dir / ".synced.json"
    seen = json.loads(seen_path.read_text(encoding="utf-8")) if seen_path.exists() else {}
    for p in sorted(accepted_dir.glob("chapter_*.md")):
        n = int(p.stem.split("_")[1])
        original = p.read_text(encoding="utf-8").strip()
        current = project.read_chapter(n)
        if not current:
            continue
        current = _strip_front(current)
        h = hashlib.sha1(current.encode("utf-8")).hexdigest()[:16]
        if current == original or seen.get(str(n)) == h:
            continue
        changed_words = abs(len(current.split()) - len(original.split()))
        _append(datasets_dir / "author_edits.jsonl", {"prompt": f"Chapter {n} of {project.state.name}", "chosen": current,
                                                       "rejected": original, "project": project.state.name, "chapter": n})
        seen[str(n)] = h
        out.append({"chapter": n, "words_changed": changed_words})
    seen_path.write_text(json.dumps(seen), encoding="utf-8")
    return out


def dataset_summary(datasets_dir: Path) -> dict[str, int]:
    out = {}
    for name in ("sft.jsonl", "preference.jsonl", "self_preference.jsonl", "author_edits.jsonl"):
        p = datasets_dir / name
        out[name] = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
    return out


def cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def unsloth_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("unsloth") is not None
    except Exception:
        return False


def plan(datasets_dir: Path, base_model: str, out_dir: Path, min_pairs: int = 50) -> dict[str, Any]:
    s = dataset_summary(datasets_dir)
    pairs = s["self_preference.jsonl"] + s["author_edits.jsonl"] + s["preference.jsonl"]
    script = Path(__file__).resolve().parents[2] / "scripts" / "finetune_lora.py"
    cmd = [sys.executable, str(script), "--datasets", str(datasets_dir), "--base", base_model, "--out", str(out_dir)]
    return {"datasets": s, "preference_pairs": pairs, "enough_data": pairs >= min_pairs and s["sft.jsonl"] + pairs >= min_pairs,
            "cuda": cuda_available(), "unsloth": unsloth_available(), "ollama": shutil.which("ollama") is not None,
            "command": " ".join(cmd), "script": str(script)}


def run(datasets_dir: Path, base_model: str, out_dir: Path) -> int:
    p = plan(datasets_dir, base_model, out_dir)
    return subprocess.call(p["command"].split())
