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


class Project:
    def __init__(self, root: Path, name: str):
        self.name = name
        self.dir = root / _slug(name)
        self.chapters_dir = self.dir / "chapters"
        self.state_path = self.dir / "project.json"
        self.state = ProjectState(name=name)
        if self.state_path.exists():
            self.state = ProjectState(**json.loads(self.state_path.read_text(encoding="utf-8")))

    @classmethod
    def create(cls, root: Path, name: str, idea: str, genres: builtins.list[str], words: int) -> Project:
        p = cls(root, name)
        if p.state_path.exists():
            raise FileExistsError(f"project already exists: {p.dir}")
        p.state.seed_idea = idea
        p.state.genres = genres
        p.state.target_chapter_words = words
        p.save()
        return p

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

    def next_chapter_number(self) -> int:
        return self.state.written + 1

    def plan_for(self, n: int) -> Optional[ChapterPlan]:
        return next((c for c in self.state.outline if c.number == n), None)

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
