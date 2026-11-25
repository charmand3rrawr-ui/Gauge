"""The character school: Gauge learns how to build a character from finished ones.

The blueprint (`gauge.write.blueprint`) says what a new character needs. This module is how that is
taught. It keeps a small library of complete character sheets the author approved, and uses them in
three ways:

* **In the prompt.** When Gauge creates a character, the checklist for that role and the closest
  examples go into the request, so the model sees the depth that is expected and not just the keys.
* **In training.** Every example becomes instruction rows in `datasets/characters.jsonl`, which the
  GaugeLM trainer reads: the breakdown itself, whole sheets from a one-line concept, single items, and
  "what is missing from this sheet" exercises.
* **As a bar.** A sheet with gaps would teach Gauge to leave gaps, so an example is only admitted when
  it passes the blueprint's own audit.

Examples come from three places: sheets the author fills in, characters promoted from the author's own
stories, and characters studied from books that are permitted for training (public domain, or the
author's own). Third-party copyrighted books are refused, the same as everywhere else in Gauge.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..write import blueprint as bp
from ..write.characters import Character, CharacterState, character_from_sheet

log = logging.getLogger(__name__)

SCHOOL_FILE = "character_school.json"
DATASET = "characters.jsonl"
MIN_SCORE = 0.8            # share of the blueprint an example must fill to be admitted
SHEET_EXTS = {".txt", ".md", ".json", ".sheet"}
CREATE_INSTRUCTION = "Create a character sheet for a serial web novel."


@dataclass
class Example:
    id: str
    name: str
    role: str
    source: str                       # "author", "project:<story>", "study:<book>"
    score: float
    sheet: dict[str, Any]
    genres: list[str] = field(default_factory=list)
    note: str = ""
    added: str = ""

    def character(self) -> Character:
        return character_from_sheet(self.name, self.sheet, self.role, str(self.sheet.get("concept", "")))

    def line(self) -> str:
        where = {"author": "written by you"}.get(self.source, self.source.replace("project:", "from your story ")
                                                 .replace("study:", "studied from "))
        return f"{self.name}  -  {self.role}, {where}, {self.score:.0%} complete"


def _clean(c: Character) -> Character:
    """A copy with only what belongs on a sheet: no memories, mood history or chapter numbers."""
    out = c.model_copy(deep=True)
    out.state = CharacterState(mood=c.state.mood, goal=c.state.goal, knows=list(c.state.knows)[:8],
                               secrets=list(c.state.secrets)[:4])
    out.created_chapter = 0
    return out


def looks_like_sheet(path: Path) -> bool:
    """True for a file written in Gauge's fill-in sheet format (it starts with the sheet header)."""
    try:
        if path.suffix.lower() not in SHEET_EXTS or not path.is_file():
            return False
        with path.open(encoding="utf-8-sig", errors="replace") as fh:
            head = fh.read(400)
    except OSError:
        return False
    return head.lstrip().lower().startswith(bp.SHEET_HEADER.lower())


class CharacterSchool:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.path = self.models_dir / SCHOOL_FILE
        self.dataset = self.models_dir / "datasets" / DATASET
        self._examples: Optional[list[Example]] = None

    # ---- storage -------------------------------------------------------------------
    def examples(self) -> list[Example]:
        if self._examples is None:
            self._examples = []
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
                    for d in raw.get("examples", []):
                        self._examples.append(Example(**{k: d[k] for k in Example.__dataclass_fields__ if k in d}))
                except (OSError, ValueError, TypeError, KeyError) as e:
                    log.warning("could not read %s: %s", self.path, e)
        return self._examples

    def _save(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "examples": [asdict(e) for e in self.examples()]}, indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self.export_training()

    # ---- adding --------------------------------------------------------------------
    def add(self, c: Character, source: str = "author", genres: Optional[list[str]] = None, note: str = "",
            force: bool = False, cast_size: Optional[int] = None) -> dict[str, Any]:
        """Admit a character as an example if it is complete enough to teach from. Adding a character with
        the same name from the same source replaces the earlier one."""
        c = _clean(c)
        # Relationships are judged on what the sheet holds: an example carries no cast with it.
        a = bp.audit(c, cast_size if cast_size is not None else (1 if c.role != "minor" else 0))
        if a.score < MIN_SCORE and not force:
            return {"added": False, "name": c.name, "score": a.score, "audit": a,
                    "message": f"{c.name} was not added: the sheet is {a.score:.0%} complete and an example needs "
                               f"{MIN_SCORE:.0%}. " + _gap_text(a)}
        eid = hashlib.sha1(f"{source}|{c.name.lower()}".encode()).hexdigest()[:10]
        ex = Example(eid, c.name, c.role, source, a.score, bp.sheet_json(c),
                     [g.strip().lower() for g in (genres or []) if g.strip()], note,
                     datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        rest = [e for e in self.examples() if e.id != eid]
        replaced = len(rest) != len(self.examples())
        self._examples = rest + [ex]
        self._save()
        msg = f"{'Updated' if replaced else 'Added'} {c.name} ({c.role}, {a.score:.0%} complete)."
        if a.gaps():
            msg += " Still worth filling in: " + ", ".join(bp.ITEMS[k].label for k in a.gaps()[:5]) + "."
        return {"added": True, "name": c.name, "score": a.score, "audit": a, "id": eid, "replaced": replaced, "message": msg}

    def add_text(self, text: str, **kw: Any) -> dict[str, Any]:
        try:
            c = bp.character_from_text(text)
        except ValueError as e:
            why = "it has not been filled in yet (it has no name)" if "no name" in str(e) else f"it could not be read ({e})"
            return {"added": False, "name": "", "score": 0.0, "message": f"was not added: {why}"}
        except Exception as e:
            return {"added": False, "name": "", "score": 0.0, "message": f"was not added: it could not be read ({e})"}
        return self.add(c, **kw)

    def add_path(self, path: Path, **kw: Any) -> list[dict[str, Any]]:
        """A sheet file, or a folder of them. One unreadable file never stops the rest."""
        path = Path(path)
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SHEET_EXTS and p.is_file()) if path.is_dir() else [path]
        out = []
        for f in files:
            if not f.exists():
                out.append({"added": False, "name": f.name, "score": 0.0, "message": f"{f.name}: not found"})
                continue
            try:
                text = f.read_text(encoding="utf-8-sig", errors="replace")
            except OSError as e:
                out.append({"added": False, "name": f.name, "score": 0.0, "message": f"{f.name}: {e}"})
                continue
            res = self.add_text(text, **kw)
            res["file"] = f.name
            if not res.get("name"):
                res["message"] = f"{f.name} {res['message']}"
            out.append(res)
        return out

    def promote(self, roster: Any, project: str, names: Optional[list[str]] = None,
                genres: Optional[list[str]] = None, force: bool = False) -> list[dict[str, Any]]:
        """Learn from the characters of one of the author's own stories."""
        wanted = {n.lower() for n in (names or [])}
        out = []
        cast = list(roster.ordered())
        for c in cast:
            if wanted and c.name.lower() not in wanted:
                continue
            if not wanted and c.role == "minor":
                continue
            out.append(self.add(c, source=f"project:{project}", genres=genres, force=force, cast_size=len(cast) - 1))
        return out

    def study(self, db: Any, llm: Any, novel: str, name: str, role: str = "supporting",
              genres: Optional[list[str]] = None, force: bool = False) -> dict[str, Any]:
        """Read a character out of a book in the library and keep them as an example. Only for books that
        are permitted for training: public domain, openly licensed, or imported by the author."""
        from ..llm import generate_json
        from ..write import prompts
        from ..write.schemas import SheetOut, schema_of
        from .corpus import training_permission

        found = _find_novel(db, novel)
        if found is None:
            return {"added": False, "name": name, "score": 0.0, "message": f"There is no book matching {novel!r} in the library."}
        perm, ok, why = training_permission(found.source)
        if not ok:
            return {"added": False, "name": name, "score": 0.0, "permission": perm,
                    "message": f"{found.title} was not studied: {why}."}
        passages = _passages(db, found.key, name)
        if not passages:
            return {"added": False, "name": name, "score": 0.0,
                    "message": f"{name} does not appear in {found.title}. Check the spelling of the name."}
        prompt = prompts.CHARACTER_STUDY_PROMPT.format(name=name, role=role, title=found.title, passages="\n\n".join(passages),
                                                       checklist=bp.checklist(role))
        try:
            d = generate_json(llm, prompt, schema=schema_of(SheetOut), temperature=0.3, max_tokens=1600,
                              purpose="character:study")
        except Exception as e:
            return {"added": False, "name": name, "score": 0.0, "message": f"The story engine could not read {name}: {e}"}
        d["name"] = name
        c = character_from_sheet(name, d, role, str(d.get("concept", "")))
        res = self.add(c, source=f"study:{found.title}", genres=genres, note=f"{perm}; {len(passages)} passages", force=force)
        res["permission"] = perm
        return res

    def remove(self, name_or_id: str) -> bool:
        key = name_or_id.strip().lower()
        rest = [e for e in self.examples() if e.id != key and e.name.lower() != key]
        if len(rest) == len(self.examples()):
            return False
        self._examples = rest
        self._save()
        return True

    # ---- using ---------------------------------------------------------------------
    def pick(self, role: str = "supporting", genres: Optional[list[str]] = None, k: int = 2,
             exclude: str = "") -> list[Example]:
        """The examples closest to what is being built: same role first, then shared genres, then the
        most complete; the author's own sheets win a tie."""
        want = {g.strip().lower() for g in (genres or [])}
        pool = [e for e in self.examples() if e.name.lower() != exclude.strip().lower()]

        def rank(e: Example) -> tuple:
            return (-(2 if e.role == role else 0) - len(want & set(e.genres)), -(1 if e.source == "author" else 0),
                    -e.score, e.id)

        return sorted(pool, key=rank)[:max(0, k)]

    def guidance(self, role: str = "supporting", genres: Optional[list[str]] = None, k: int = 2, exclude: str = "",
                 checklist: bool = True) -> str:
        """The block that goes into the creation prompt: what this role needs, and finished examples."""
        parts = []
        if checklist:
            parts.append(f"WHAT A FINISHED {role.upper()} NEEDS, and the test each part must pass:\n" + bp.checklist(role))
        chosen = self.pick(role, genres, k, exclude)
        if chosen:
            parts.append("EXAMPLES the author approved. Match their depth and how specific they are. Do not reuse "
                         "their names, details or wording:")
            for i, e in enumerate(chosen, 1):
                parts.append(f"Example {i} ({e.role}):\n" + json.dumps(e.sheet, ensure_ascii=False))
        return ("\n\n".join(parts) + "\n") if parts else ""

    def stats(self) -> dict[str, Any]:
        ex = self.examples()
        by_role: dict[str, int] = {}
        for e in ex:
            by_role[e.role] = by_role.get(e.role, 0) + 1
        return {"examples": len(ex), "by_role": by_role, "rows": _count_lines(self.dataset),
                "mean_score": round(sum(e.score for e in ex) / len(ex), 3) if ex else 0.0}

    # ---- training rows -------------------------------------------------------------
    def training_rows(self) -> list[dict[str, str]]:
        """Instruction rows that teach character building. The breakdown itself is always taught; every
        example adds a whole-sheet row, a row per item, and exercises in spotting what a sheet lacks."""
        rows: list[dict[str, str]] = []

        def row(instruction: str, inp: str, output: str) -> None:
            if output.strip():
                rows.append({"instruction": instruction, "input": inp, "output": output.strip()})

        for title, why, items in bp.SECTIONS:
            row("What does a newly built character need?", f"Section: {title}",
                why + "\n" + "\n".join(f"- {it.label}: {it.what} Test: {it.test}" for it in items))
            for it in items:
                row("Explain one part of a character sheet, and the test a good answer passes.", it.label,
                    f"{it.what} Test: {it.test}")
        for role in bp.EVERYONE:
            row("List what a finished character of this role needs.", f"Role: {role}", bp.checklist(role))

        for e in self.examples():
            c = e.character()
            who = f"Name: {c.name}\nRole: {c.role}\nConcept from the author: {c.concept}"
            row(CREATE_INSTRUCTION, who, json.dumps(e.sheet, ensure_ascii=False))
            row("Write this character out as a filled-in character sheet.", who, bp.sheet_text(c))
            filled = []
            for it in bp.ITEMS.values():
                if it.kind in ("axes", "flag", "relationships") or it.key in ("name", "role", "concept"):
                    continue
                v = bp.value_of(c, it.key)
                text = "; ".join(str(x) for x in v) if isinstance(v, list) else str(v or "")
                if text.strip():
                    filled.append(it)
                    row("Build one part of a character.", f"{who}\nPart: {it.label}. {it.what}", text)
            for o, r in c.relationships.items():
                if r.stance.strip():
                    row("Build one part of a character.", f"{who}\nPart: how they stand with {o}.", r.stance)
            # Exercises: take a few items away, and the answer names exactly those.
            order = sorted(filled, key=lambda it: hashlib.sha1(f"{e.id}{it.key}".encode()).hexdigest())
            for start in (0, 3):
                gone = order[start:start + 3]
                if len(gone) < 2:
                    continue
                partial = c.model_copy(deep=True)
                for it in gone:
                    _blank(partial, it)
                missing = sorted(gone, key=lambda it: list(bp.ITEMS).index(it.key))
                row("Check this character sheet against the blueprint. What is missing, and what would fix it?",
                    bp.sheet_text(partial),
                    "\n".join(f"Missing: {it.label}. {it.what} Test: {it.test}" for it in missing))
            a = bp.audit(c, 1 if c.relationships else 0)
            row("Check this character sheet against the blueprint. What is missing, and what would fix it?", bp.sheet_text(c),
                "Nothing is missing. " + a.summary() if not a.gaps() else
                "\n".join(f"{'Missing' if f.status == 'missing' else 'Too thin'}: {f.label}. {bp.ITEMS[f.key].test}"
                          for f in a.findings if f.status != "ok"))
        return rows

    def export_training(self) -> dict[str, Any]:
        """Write the rows where the GaugeLM trainer looks for them."""
        rows = self.training_rows()
        try:
            self.dataset.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.dataset.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(self.dataset)
        except OSError as e:
            log.warning("could not write %s: %s", self.dataset, e)
            return {"rows": 0, "examples": len(self.examples()), "path": str(self.dataset), "error": str(e)}
        return {"rows": len(rows), "examples": len(self.examples()), "path": str(self.dataset)}


# ---------------------------------------------------------------------------------- helpers
def _gap_text(a: bp.Audit) -> str:
    parts = []
    if a.missing:
        parts.append("Missing: " + ", ".join(f.label for f in a.missing[:10]) + ".")
    if a.thin:
        parts.append("Too thin: " + ", ".join(f"{f.label}" + (f" ({f.note})" if f.note else "") for f in a.thin[:6]) + ".")
    return " ".join(parts)


def _blank(c: Character, it: bp.Item) -> None:
    empty: Any = [] if it.kind == "list" else ""
    targets = {"aliases": (c, "aliases"), "habits": (c, "habits"), "skills": (c, "skills"), "fears": (c.matrix, "fears"),
               "values": (c.matrix, "values"), "lines": (c.matrix, "lines_wont_cross"), "tics": (c.voice, "tics"),
               "catchphrases": (c.voice, "catchphrases"), "avoids": (c.voice, "avoids_talking_about"),
               "knows": (c.state, "knows"), "secrets": (c.state, "secrets"), "appearance": (c, "appearance"),
               "backstory": (c, "backstory"), "wound": (c.matrix, "wound"), "want": (c.matrix, "wants"),
               "need": (c.matrix, "needs"), "lie": (c, "lie"), "flaw": (c.matrix, "flaw"), "stakes": (c, "stakes"),
               "contradiction": (c, "contradiction"), "register": (c.voice, "tone"),
               "sentence_length": (c.voice, "sentence_length"), "vocabulary": (c.voice, "vocabulary"),
               "humor_style": (c.voice, "humor_style"), "mood": (c.state, "mood"), "goal": (c.state, "goal"), "arc": (c, "arc")}
    if it.key in targets:
        obj, attr = targets[it.key]
        setattr(obj, attr, empty)


def _count_lines(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _find_novel(db: Any, wanted: str) -> Any:
    w = wanted.strip().lower()
    novels = list(db.novels())
    for n in novels:
        if n.key.lower() == w or n.title.lower() == w:
            return n
    close = [n for n in novels if w and w in n.title.lower()]
    return close[0] if len(close) == 1 else None


def _passages(db: Any, key: str, name: str, budget: int = 7000, each: int = 700) -> list[str]:
    """Paragraphs where the character appears, spread across the whole book, within a size budget."""
    first = name.split()[0] if name.split() else name
    pat = re.compile(rf"(?<![A-Za-z])(?:{re.escape(name)}|{re.escape(first)})(?![A-Za-z])")
    hits: list[str] = []
    for ch in db.chapters(key):
        for para in re.split(r"\n\s*\n", ch.text or ""):
            para = " ".join(para.split())
            if len(para) >= 80 and pat.search(para):
                hits.append(para[:each])
    if not hits:
        return []
    want = max(1, budget // each)
    if len(hits) <= want:
        return hits
    step = len(hits) / want
    return [hits[int(i * step)] for i in range(want)]
