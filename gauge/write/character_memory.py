"""Persistent character memory: a character who appears in more than one book carries scars.

Per-project character state resets when a new serial starts. This store keeps, across projects, the
few things that should *not* reset: lasting conditions (scars, debts), ingrained habits (voice tics),
grudges (relationships that ended in low trust or high fear), and a one-line note on who the
character became. When the same character is created again, those are seeded back in, so depth
accumulates instead of starting from zero every book.

It lives under ``models/character_memory/<slug>.json`` and is opt-in: nothing reads it unless a
project asks to carry a character in.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unnamed"


class CharacterLedger(BaseModel):
    name: str
    books: list[str] = Field(default_factory=list)        # project names the character has appeared in
    scars: list[str] = Field(default_factory=list)        # conditions that persist (a limp, a debt, a lost eye)
    habits: list[str] = Field(default_factory=list)       # voice tics and behaviours that define them
    grudges: dict[str, str] = Field(default_factory=dict)  # name -> why the relationship curdled
    became: str = ""                                       # one line on who they grew into
    wound: str = ""                                        # the core wound, carried between books


class GlobalCharacterMemory:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir) / "character_memory"

    def _path(self, name: str) -> Path:
        return self.dir / f"{_slug(name)}.json"

    def get(self, name: str) -> Optional[CharacterLedger]:
        p = self._path(name)
        if not p.exists():
            return None
        try:
            return CharacterLedger(**json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return None

    def all(self) -> list[CharacterLedger]:
        if not self.dir.exists():
            return []
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append(CharacterLedger(**json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return out

    def save(self, led: CharacterLedger) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._path(led.name)
        p.write_text(led.model_dump_json(indent=2), encoding="utf-8")
        return p

    def absorb(self, character: Any, project_name: str) -> CharacterLedger:
        """Fold the lasting parts of a finished character into their cross-book ledger."""
        led = self.get(character.name) or CharacterLedger(name=character.name)
        if project_name and project_name not in led.books:
            led.books.append(project_name)
        st = getattr(character, "state", None)
        if st is not None:
            for c in getattr(st, "condition", []) or []:
                if c not in led.scars:
                    led.scars.append(str(c))
            if getattr(st, "arc_progress", ""):
                led.became = str(st.arc_progress)[:200]
        voice = getattr(character, "voice", None)
        if voice is not None:
            for t in getattr(voice, "tics", []) or []:
                if t not in led.habits:
                    led.habits.append(str(t))
        matrix = getattr(character, "matrix", None)
        if matrix is not None and getattr(matrix, "wound", ""):
            led.wound = str(matrix.wound)[:200]
        for other, rel in (getattr(character, "relationships", {}) or {}).items():
            trust = getattr(rel, "trust", 1.0)
            fear = getattr(rel, "fear", 0.0)
            if trust < 0.2 or fear > 0.6:
                led.grudges[str(other)] = (getattr(rel, "stance", "") or "trust broke")[:120]
        led.scars = led.scars[-12:]
        led.habits = led.habits[-12:]
        self.save(led)
        return led

    def carry_into(self, character: Any) -> bool:
        """Seed a freshly created character with what the ledger remembers. Returns True if anything was carried."""
        led = self.get(character.name)
        if led is None:
            return False
        st = getattr(character, "state", None)
        if st is not None:
            for s in led.scars:
                if s not in st.condition:
                    st.condition.append(s)
        voice = getattr(character, "voice", None)
        if voice is not None:
            for h in led.habits:
                if h not in (voice.tics or []):
                    voice.tics.append(h)
        matrix = getattr(character, "matrix", None)
        if matrix is not None and led.wound and not getattr(matrix, "wound", ""):
            matrix.wound = led.wound
        if led.grudges:
            from .characters import Relationship
            for other, why in led.grudges.items():
                rel = character.relationships.setdefault(other, Relationship())
                if not rel.stance:
                    rel.stance = why
                rel.trust = min(rel.trust, 0.2)
        return True
