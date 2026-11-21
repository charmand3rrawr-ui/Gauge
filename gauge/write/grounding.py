"""A grounding dossier: the real sensory and procedural texture of a scene, from research.

The Scout fetches facts; this asks it the questions that make a scene *feel* real — what a place actually
sounds and smells like, how a job is really done, what the physical routine of an activity is — and folds
the answers into a short brief for the drafter. Accurate, specific texture is what separates a scene that
was there from one that was imagined. It needs the Scout (and therefore the network); offline it returns
nothing and the writer falls back on invention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_STOP = {"the", "a", "an", "and", "of", "in", "on", "at", "to", "with", "her", "his", "their", "they",
         "she", "he", "it", "into", "from", "for", "as", "by"}


def grounding_queries(scene_text: str, setting: str = "", max_q: int = 3) -> list[str]:
    """Turn a scene into a few 'what is this really like' research questions."""
    subject = setting.strip()
    if not subject:
        caps = re.findall(r"(?<=[a-z,;:]\s)[A-Z][a-z]{3,}(?:\s[A-Z][a-z]{3,})?", scene_text)
        subject = caps[0] if caps else " ".join(w for w in scene_text.split()[:6] if w.lower() not in _STOP)
    subject = subject[:60] or "this setting"
    return [f"what does {subject} actually look, sound and smell like",
            f"what is the physical routine or procedure in {subject}",
            f"specific concrete details a person would notice in {subject}"][:max_q]


@dataclass
class GroundingDossier:
    subject: str
    notes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.notes:
            return ""
        body = "\n".join(f"- {n}" for n in self.notes[:8])
        return f"### Grounding (real-world texture for this scene)\n{body}"


def build_dossier(scout: Any, scene_text: str, setting: str = "", max_queries: int = 3) -> GroundingDossier:
    """`scout` is any object with `.search(query) -> iterable of objects having .snippets/.text or str`.
    Returns a dossier of concrete texture notes. Resilient: any failure yields an empty dossier."""
    subject = (setting or "").strip() or "the scene"
    notes: list[str] = []
    sources: list[str] = []
    if scout is None:
        return GroundingDossier(subject, notes, sources)
    for q in grounding_queries(scene_text, setting, max_queries):
        try:
            findings = scout.search(q)
        except Exception:
            continue
        for f in (findings or [])[:2]:
            snippets = getattr(f, "snippets", None)
            if snippets:
                for s in snippets[:2]:
                    text = getattr(s, "text", None) or (s.get("text") if isinstance(s, dict) else str(s))
                    if text:
                        notes.append(str(text).strip()[:200])
                        src = getattr(s, "url", "") or (s.get("url", "") if isinstance(s, dict) else "")
                        if src:
                            sources.append(src)
            else:
                text = getattr(f, "text", None) or (str(f) if f else "")
                if text:
                    notes.append(str(text).strip()[:200])
    # De-duplicate while preserving order.
    seen: set[str] = set()
    uniq = [n for n in notes if not (n in seen or seen.add(n))]
    return GroundingDossier(subject, uniq[:8], list(dict.fromkeys(sources))[:8])
