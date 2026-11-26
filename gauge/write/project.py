"""A writing project: premise, bible, outline and the chapters written so far.

Everything is plain Markdown/JSON on disk under ``data/projects/<name>/`` so you can edit the
bible or an outline by hand between cycles and the daemon will pick up your changes.
"""
from __future__ import annotations

import builtins
import json
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChapterPlan(BaseModel):
    number: int
    title: str = ""
    goal: str = ""
    hook: str = ""
    turn: str = ""
    cliffhanger: str = ""
    beats: list[str] = Field(default_factory=list)


class Bible(BaseModel):
    premise: str = ""
    protagonist: str = ""
    world: str = ""
    stakes: str = ""
    promise: str = ""  # the central promise to the reader (what they will get every chapter)
    tone: str = ""
    pov: str = "first person, past tense"
    characters: dict[str, str] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)  # world/magic rules the writer must not break
    arc: list[str] = Field(default_factory=list)  # 5-8 sentence long-form arc


class ProjectState(BaseModel):
    name: str
    genres: list[str] = Field(default_factory=list)
    seed_idea: str = ""
    target_chapter_words: int = 2200
    bible: Bible = Field(default_factory=Bible)
    outline: list[ChapterPlan] = Field(default_factory=list)
    written: int = 0
    running_summary: str = ""  # what has happened so far, kept short for the prompt
    open_threads: list[str] = Field(default_factory=list)
    premise_doc: str = ""  # the author's own premise document, canon for the bible
    past_directions: list[str] = Field(default_factory=list)  # pitches the muse already took
    current_direction: str = ""
    author_direction: str = ""  # what the author asked for in the next chapter (Studio chat prompt); cleared once written
    continued_from: str = ""  # path of an imported manuscript this project continues


OLD_PLANS = "outline_archive.jsonl"


class Project:
    def __init__(self, root: Path, name: str):
        self.name = name
        self.dir = root / _slug(name)
        self.chapters_dir = self.dir / "chapters"
        self.state_path = self.dir / "project.json"
        self.state = ProjectState(name=name)
        self._plans: Optional[dict[int, ChapterPlan]] = None      # plans by chapter number (see plan_for)
        self._plans_size = -1
        self._old_plans: Optional[dict[int, ChapterPlan]] = None
        if self.state_path.exists():
            self.state = ProjectState(**json.loads(self.state_path.read_text(encoding="utf-8")))

    @classmethod
    def create(cls, root: Path, name: str, idea: str, genres: builtins.list[str], words: int,
               premise_doc: str = "") -> Project:
        p = cls(root, name)
        if p.state_path.exists():
            raise FileExistsError(f"project already exists: {p.dir}")
        p.state.seed_idea = idea
        p.state.genres = genres
        p.state.target_chapter_words = words
        p.state.premise_doc = premise_doc
        p.save()
        if premise_doc:
            (p.dir / "premise.md").write_text(premise_doc, encoding="utf-8")
        return p

    def previous_texts(self, upto: int, limit: int = 12) -> builtins.list[str]:
        """Prose of the most recent written chapters before ``upto`` (for novelty checks)."""
        out = []
        for n in range(max(1, upto - limit), upto):
            t = self.read_chapter(n)
            if t:
                out.append(re.sub(r"^#[^\n]*\n+", "", t))  # drop the "# Chapter N" heading
        return out

    @classmethod
    def exists(cls, root: Path, name: str) -> bool:
        return (root / _slug(name) / "project.json").exists()

    @staticmethod
    def list(root: Path) -> builtins.list[str]:
        if not root.exists():
            return []
        out = []
        for d in sorted(root.iterdir()):
            if (d / "project.json").exists():
                out.append(json.loads((d / "project.json").read_text(encoding="utf-8"))["name"])
        return out

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chapters_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
        (self.dir / "bible.md").write_text(self.bible_markdown(), encoding="utf-8")
        (self.dir / "outline.md").write_text(self.outline_markdown(), encoding="utf-8")

    # ---- chapters ----------------------------------------------------------
    def chapter_path(self, n: int) -> Path:
        return self.chapters_dir / f"chapter_{n:03d}.md"

    def write_chapter(self, n: int, title: str, text: str, meta: dict[str, Any]) -> Path:
        p = self.chapter_path(n)
        front = "---\n" + "\n".join(f"{k}: {json.dumps(v)}" for k, v in meta.items()) + "\n---\n\n"
        p.write_text(f"{front}# Chapter {n}: {title}\n\n{text.strip()}\n", encoding="utf-8")
        self.state.written = max(self.state.written, n)
        return p

    def read_chapter(self, n: int) -> Optional[str]:
        p = self.chapter_path(n)
        if not p.exists():
            return None
        text = p.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            text = parts[2] if len(parts) == 3 else text
        return text.strip()

    def read_meta(self, n: int) -> dict[str, Any]:
        """Parse the front-matter a chapter was written with (score, rubric, cast, ...)."""
        p = self.chapter_path(n)
        if not p.exists():
            return {}
        text = p.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return {}
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        meta: dict[str, Any] = {}
        for line in parts[1].strip().splitlines():
            key, sep, val = line.partition(":")
            if not sep:
                continue
            try:
                meta[key.strip()] = json.loads(val.strip())
            except (json.JSONDecodeError, ValueError):
                meta[key.strip()] = val.strip()
        return meta

    # ---- draft checkpoints -------------------------------------------------
    @property
    def drafts_dir(self) -> Path:
        return self.dir / "drafts"

    def checkpoint_path(self, n: int) -> Path:
        return self.drafts_dir / f"chapter_{n:03d}.json"

    def save_checkpoint(self, n: int, stage: str, text: str, extra: Optional[dict[str, Any]] = None) -> Path:
        """Persist a raw draft before it is judged, so a crash costs nothing but the critique."""
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        p = self.checkpoint_path(n)
        data: dict[str, Any] = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data.update({"chapter": n, "stage": stage, "text": text, **(extra or {})})
        data.setdefault("history", []).append(stage)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def load_checkpoint(self, n: int) -> Optional[dict[str, Any]]:
        p = self.checkpoint_path(n)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data if data.get("text") else None

    def clear_checkpoint(self, n: int) -> None:
        p = self.checkpoint_path(n)
        if p.exists():
            p.unlink()

    def next_chapter_number(self) -> int:
        return self.state.written + 1

    def plan_for(self, n: int) -> Optional[ChapterPlan]:
        """The plan for chapter n: looked up by number, not searched for, so a long outline costs nothing.
        Plans of chapters written long ago are kept in a file beside the story and read from there."""
        if self._plans is None or self._plans_size != len(self.state.outline):
            self._plans = {c.number: c for c in self.state.outline}
            self._plans_size = len(self.state.outline)
        got = self._plans.get(n)
        if got is None and (self.dir / OLD_PLANS).exists():
            if self._old_plans is None:
                self._old_plans = {}
                for line in (self.dir / OLD_PLANS).read_text(encoding="utf-8").splitlines():
                    try:
                        c = ChapterPlan(**json.loads(line))
                        self._old_plans[c.number] = c
                    except (ValueError, TypeError):
                        continue
            got = self._old_plans.get(n)
        return got

    def put_away_old_plans(self, keep: int = 200) -> int:
        """Move the plans of chapters written more than `keep` chapters ago out of the working file, so that
        saving the story stays quick however long it gets. Returns how many were moved."""
        cut = self.state.written - keep
        old = [c for c in self.state.outline if c.number <= cut]
        if len(old) < 100:                       # not worth a write until there is a batch
            return 0
        with (self.dir / OLD_PLANS).open("a", encoding="utf-8") as fh:
            for c in old:
                fh.write(c.model_dump_json() + "\n")
        self.state.outline = [c for c in self.state.outline if c.number > cut]
        if self._old_plans is not None:
            self._old_plans.update({c.number: c for c in old})
        self._plans = None
        return len(old)

    def story_so_far(self, words: int = 600) -> str:
        """Everything before the next chapter. With a chronicle that is the layered history (sagas, volumes,
        arcs, recent chapters) followed by where things stand now; without one, the single running summary."""
        from .chronicle import DB, Chronicle

        now = self.state.running_summary
        if not (self.dir / DB).exists():
            return now
        try:
            past = Chronicle.at(self.dir).story_so_far(self.next_chapter_number(), words)
        except Exception:
            return now
        if not past:
            return now
        return past + (f"\n\nWHERE THINGS STAND NOW:\n{now}" if now else "")

    # ---- rendering ---------------------------------------------------------
    def bible_markdown(self) -> str:
        b = self.state.bible
        lines = [f"# {self.state.name} - Series Bible", "", f"**Genres:** {', '.join(self.state.genres)}", "",
                 f"**Premise:** {b.premise}", "", f"**Protagonist:** {b.protagonist}", "",
                 f"**World:** {b.world}", "", f"**Stakes:** {b.stakes}", "",
                 f"**Promise to the reader:** {b.promise}", "", f"**Tone:** {b.tone}", f"**POV:** {b.pov}", ""]
        if b.characters:
            lines += ["## Characters"] + [f"- **{k}:** {v}" for k, v in b.characters.items()] + [""]
        if b.rules:
            lines += ["## Rules of the world"] + [f"- {r}" for r in b.rules] + [""]
        if b.arc:
            lines += ["## Arc"] + [f"{i}. {s}" for i, s in enumerate(b.arc, 1)] + [""]
        return "\n".join(lines)

    def outline_markdown(self) -> str:
        lines = [f"# {self.state.name} - Outline", ""]
        for c in self.state.outline:
            mark = "x" if c.number <= self.state.written else " "
            lines.append(f"- [{mark}] **Ch {c.number}: {c.title}** - {c.goal}")
            lines.append(f"    - hook: {c.hook}")
            lines.append(f"    - turn: {c.turn}")
            lines.append(f"    - cliffhanger: {c.cliffhanger}")
        return "\n".join(lines) + "\n"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"


def title_from_text(text: str, n: int = 4) -> str:
    """First few words of an idea as a project name, without a dangling article or preposition."""
    import re as _re

    words = [w for w in _re.findall(r"[A-Za-z0-9'\-]+", text)][:n]
    tail = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "who", "that", "with", "for", "by", "from"}
    while words and words[-1].lower() in tail:
        words.pop()
    return " ".join(words).strip() or "My Story"
