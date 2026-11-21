"""Compile a training library the AI may lawfully learn from — with the licence of every item tracked.

Training a model on text you do not have the right to use is the problem this module exists to prevent.
It assembles a corpus only from sources that are safe to train on — public-domain works (Project
Gutenberg), files you imported yourself, and your own written chapters — and records the licence and
provenance of each document so the corpus is auditable. Third-party scraped fiction (Royal Road,
Scribble Hub, and the like) is copyrighted and is EXCLUDED by default; it is included only if you
explicitly opt in, and even then it is labelled so you know what is in your training set and why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# (permission, ok-to-train-by-default, note). "owned" = your own writing; "user_owned" = files you
# imported and are responsible for; "public_domain" = free to use; "third_party" = someone else's rights.
_PERMISSION = {
    "gutenberg": ("public_domain", True, "Project Gutenberg: public domain in the US; check your jurisdiction"),
    "gutendex": ("public_domain", True, "Project Gutenberg: public domain in the US; check your jurisdiction"),
    "local": ("user_owned", True, "imported by you: train only on files you wrote or hold the rights to"),
    "import": ("user_owned", True, "imported by you: train only on files you wrote or hold the rights to"),
    "project": ("owned", True, "your own Gauge-written chapters"),
}
_DEFAULT = ("third_party", False, "scraped from a third party and copyrighted; not yours to train on")


def training_permission(source: str) -> tuple[str, bool, str]:
    return _PERMISSION.get((source or "").lower(), _DEFAULT)


@dataclass
class CorpusDoc:
    doc_id: str
    kind: str            # "prose" | "chat:<prompt>"
    text: str
    source: str
    title: str = ""
    permission: str = ""
    permitted: bool = True
    note: str = ""


@dataclass
class CorpusReport:
    docs: list[CorpusDoc] = field(default_factory=list)
    excluded: list[CorpusDoc] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(d.text) for d in self.docs)

    @property
    def est_tokens(self) -> int:
        return self.chars // 4

    def by_source(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for d in self.docs:
            s = out.setdefault(d.source, {"docs": 0, "chars": 0, "permission": d.permission, "note": d.note})
            s["docs"] += 1
            s["chars"] += len(d.text)
        return out

    def manifest(self) -> dict[str, Any]:
        return {"documents": len(self.docs), "chars": self.chars, "est_tokens": self.est_tokens,
                "by_source": self.by_source(),
                "excluded": {"documents": len(self.excluded),
                             "sources": sorted({d.source for d in self.excluded})},
                "note": "Only sources safe to train on are included; third-party copyrighted text is excluded unless opted in."}


def _dedup(docs: list[CorpusDoc]) -> list[CorpusDoc]:
    import hashlib
    seen: set[str] = set()
    out = []
    for d in docs:
        key = hashlib.sha1((d.kind[:40] + d.text[:400]).encode("utf-8", "replace")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def compile_corpus(db: Any, projects_dir: Optional[Path] = None, include_third_party: bool = False) -> CorpusReport:
    """Assemble the training library, filtered by licence. Your own chapters are always included; third-party
    scraped fiction only when include_third_party is True (and always labelled)."""
    docs: list[CorpusDoc] = []
    excluded: list[CorpusDoc] = []
    for novel in db.novels():
        perm, ok, note = training_permission(novel.source)
        for i, ch in enumerate(db.chapters(novel.key)):
            d = CorpusDoc(f"{novel.key}#{i}", "prose", ch.text, novel.source, novel.title, perm, ok or include_third_party, note)
            (docs if (ok or include_third_party) else excluded).append(d)
    if projects_dir and Path(projects_dir).exists():
        import re

        from ..write.project import Project
        for name in Project.list(Path(projects_dir)):
            p = Project(Path(projects_dir), name)
            upto = p.state.written
            if upto <= 0:  # state may not be persisted; scan the chapter files directly
                n = 1
                while p.read_chapter(n) is not None and n < 10000:
                    upto = n
                    n += 1
            for n in range(1, upto + 1):
                t = p.read_chapter(n)
                if t:
                    perm, _, note = training_permission("project")
                    docs.append(CorpusDoc(f"project:{name}#{n}", "prose", re.sub(r"^#[^\n]*\n+", "", t),
                                          "project", name, perm, True, note))
    return CorpusReport(_dedup(docs), excluded)


def permitted_texts(db: Any, projects_dir: Optional[Path] = None, include_third_party: bool = False) -> list[tuple[str, str]]:
    """The corpus as (kind, text) tuples the native trainer consumes — licence-filtered."""
    return [(d.kind, d.text) for d in compile_corpus(db, projects_dir, include_third_party).docs]


def export_corpus(report: CorpusReport, out_dir: Path) -> dict[str, Any]:
    """Write corpus.jsonl (text + provenance), corpus.txt (plain, for the tokenizer/trainer), and MANIFEST.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for d in report.docs:
            fh.write(json.dumps({"id": d.doc_id, "kind": d.kind, "source": d.source, "title": d.title,
                                 "permission": d.permission, "text": d.text}, ensure_ascii=False) + "\n")
    (out / "corpus.txt").write_text("\n\n".join(d.text for d in report.docs), encoding="utf-8")
    manifest = report.manifest()
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
