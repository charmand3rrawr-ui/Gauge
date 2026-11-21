"""The author persona: who is writing, as distinct from what the data likes.

Built from your own imported novels (or a project's accepted chapters): a measured voice
fingerprint, a few sample passages, and a model-written description of stances, humour,
taboos and favourite moves. ``write.persona_blend`` sets how much of the system prompt is
"write like me" versus "write like the data"."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..features import pacing_features, voice_features
from ..llm.client import LLM, generate_json
from ..text import head

PERSONA_PROMPT = """[persona:describe]
Describe the author of these passages as a writing persona a co-writer could imitate.

PASSAGES:
{samples}

Respond with ONLY JSON: {{"name": "a short handle for this voice", "stances": ["3-5 things this author clearly believes or keeps returning to"],
"humor": "how (and whether) this author is funny", "taboos": ["things this author never does on the page"],
"favorite_moves": ["3-6 recurring craft moves"], "sentence_music": "one line on rhythm and length", "register": "one line",
"never": ["2-4 phrases or habits to avoid when imitating this author"]}}"""


class AuthorPersona(BaseModel):
    name: str = ""
    source: str = ""  # novel key or project the persona was learned from
    fingerprint: dict[str, float] = Field(default_factory=dict)
    samples: list[str] = Field(default_factory=list)
    stances: list[str] = Field(default_factory=list)
    humor: str = ""
    taboos: list[str] = Field(default_factory=list)
    favorite_moves: list[str] = Field(default_factory=list)
    sentence_music: str = ""
    register_style: str = ""
    never: list[str] = Field(default_factory=list)

    def prompt_block(self, blend: float) -> str:
        """Text for the system prompt. blend 0 = ignore persona, 1 = persona dominates."""
        if blend <= 0.0 or not (self.stances or self.favorite_moves or self.fingerprint):
            return ""
        strength = "This matters more than the data-driven rules below." if blend >= 0.7 else (
            "Balance this with the data-driven rules below." if blend >= 0.4 else "Keep this as a light flavour under the data-driven rules below.")
        fp = self.fingerprint
        voice = []
        if fp:
            pov = "first person" if fp.get("first_person_ratio", 0) > fp.get("third_person_ratio", 0) else "third person"
            voice.append(f"{pov}; sentences average {fp.get('sent_len_mean', 0):.0f} words; dialogue about {fp.get('dialogue_ratio', 0):.0%}; "
                         f"adverbs {'rare' if fp.get('adverb_ratio', 0) < 0.02 else 'present'}")
        lines = [f"AUTHOR PERSONA ({self.name or 'the author'}). {strength}"]
        if voice:
            lines.append("Voice: " + "; ".join(voice) + (f". {self.sentence_music}" if self.sentence_music else ""))
        if self.register_style:
            lines.append(f"Register: {self.register_style}")
        if self.stances:
            lines.append("Keeps returning to: " + "; ".join(self.stances))
        if self.humor:
            lines.append(f"Humour: {self.humor}")
        if self.favorite_moves:
            lines.append("Favourite moves: " + "; ".join(self.favorite_moves))
        if self.taboos or self.never:
            lines.append("Never: " + "; ".join(self.taboos + self.never))
        if self.samples and blend >= 0.4:
            lines.append("A sample of the voice (match its energy, not its content): " + self.samples[0][:600])
        return "\n".join(lines)


def fingerprint_from_texts(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {}
    keys = ("first_person_ratio", "third_person_ratio", "adverb_ratio", "type_token_ratio", "long_word_ratio")
    acc: dict[str, float] = {}
    for t in texts:
        v = voice_features(t)
        p = pacing_features(t)
        for k in keys:
            acc[k] = acc.get(k, 0.0) + v[k]
        for k in ("sent_len_mean", "sent_len_std", "dialogue_ratio", "one_line_par_ratio"):
            acc[k] = acc.get(k, 0.0) + p[k]
    return {k: round(v / len(texts), 4) for k, v in acc.items()}


def build_persona(llm: Optional[LLM], texts: list[str], source: str, max_samples: int = 3) -> AuthorPersona:
    persona = AuthorPersona(source=source, fingerprint=fingerprint_from_texts(texts))
    persona.samples = [head(t, 180) for t in texts[:max_samples]]
    if llm is not None and texts:
        try:
            d = generate_json(llm, PERSONA_PROMPT.format(samples="\n\n---\n\n".join(head(t, 350) for t in texts[:4])),
                              temperature=0.4, max_tokens=700, purpose="persona")
        except Exception:
            d = {}
        persona.name = str(d.get("name", "") or "")[:60]
        for k in ("stances", "taboos", "favorite_moves", "never"):
            v = d.get(k)
            if isinstance(v, list):
                setattr(persona, k, [str(x)[:160] for x in v][:6])
        for k, attr in (("humor", "humor"), ("sentence_music", "sentence_music"), ("register", "register_style")):
            if d.get(k):
                setattr(persona, attr, str(d[k])[:200])
    return persona


def persona_path(models_dir: Path) -> Path:
    return models_dir / "persona.json"


def load_persona(models_dir: Path) -> Optional[AuthorPersona]:
    p = persona_path(models_dir)
    if not p.exists():
        return None
    try:
        return AuthorPersona(**json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return None


def save_persona(models_dir: Path, persona: AuthorPersona) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    p = persona_path(models_dir)
    p.write_text(persona.model_dump_json(indent=2), encoding="utf-8")
    (models_dir / "persona.md").write_text("# Author persona\n\n" + persona.prompt_block(1.0) + "\n", encoding="utf-8")
    return p


def persona_dict(p: AuthorPersona) -> dict[str, Any]:
    return p.model_dump()
