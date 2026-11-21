"""Scene improvisation: the present characters play the scene out, each answering only as
themselves, with the World interjecting, before a single line of prose is written. The transcript
is raw material for the drafter, who is told to rewrite it, never to copy it."""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..llm.client import LLM
from .characters import Character
from .world import WorldAgent, WorldState

log = logging.getLogger(__name__)

TURN_PROMPT = """[improv:turn]
The scene so far (an improvisation; nothing is final):
{transcript}

Situation: {situation}
You are in this scene. Give your next beat as {name}: one or two sentences of what you do and/or say, in
your own voice, reacting to what just happened. No narration of other people's feelings. Prose, no labels."""

WORLD_PROMPT = """[improv:world]
The scene so far:
{transcript}

Situation: {situation}
Interject as the World in one or two sentences: weather, a sound, a passerby, a mechanism, a consequence
arriving. Concrete and physical. No dialogue for named characters."""


def _clean(text: str, name: str) -> str:
    t = text.strip()
    for prefix in (f"{name}:", f"{name} -", "World:"):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].strip()
    return " ".join(t.split()[:70])


def run_improv(llm: LLM, cast: list[Character], situation: str, rounds: int = 2, world: Optional[WorldState] = None,
               world_level: float = 0.0, max_cast: int = 4) -> list[dict[str, str]]:
    """Returns a transcript of {speaker, beat} entries."""
    cast = cast[:max_cast]
    if not cast:
        return []
    transcript: list[dict[str, str]] = []

    def text_so_far() -> str:
        return "\n".join(f"{t['speaker']}: {t['beat']}" for t in transcript) or "(the scene is about to begin)"

    for r in range(max(1, rounds)):
        for c in cast:
            prompt = TURN_PROMPT.format(transcript=text_so_far(), situation=situation, name=c.name)
            try:
                beat = llm.generate(prompt, system=c.system_prompt(), temperature=0.95, max_tokens=120, purpose="improv:turn")
            except Exception as e:
                log.warning("improv turn failed for %s: %s", c.name, e)
                continue
            beat = _clean(beat, c.name)
            if beat:
                transcript.append({"speaker": c.name, "beat": beat})
        if world is not None and world_level >= 0.35 and r == 0:
            try:
                wb = llm.generate(WORLD_PROMPT.format(transcript=text_so_far(), situation=situation),
                                  system=WorldAgent(llm, world).system_prompt(), temperature=0.8, max_tokens=90, purpose="improv:world")
                wb = _clean(wb, "World")
                if wb:
                    transcript.append({"speaker": "The World", "beat": wb})
            except Exception as e:
                log.warning("improv world interjection failed: %s", e)
    return transcript


def render_improv(transcript: list[dict[str, Any]]) -> str:
    if not transcript:
        return ""
    lines = ["IMPROV TRANSCRIPT (the cast played the scene as themselves; raw material to rewrite, never to copy):"]
    lines += [f"- {t['speaker']}: {t['beat']}" for t in transcript]
    return "\n".join(lines)
