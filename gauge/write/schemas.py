"""JSON schemas for every structured answer the writer asks the model for.

Passed to the server as a constrained-output format when supported, and used to validate what
comes back. Keeping them here, next to the prompts, means a prompt and its contract change
together.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BibleOut(BaseModel):
    premise: str = ""
    protagonist: str = ""
    world: str = ""
    stakes: str = ""
    promise: str = ""
    tone: str = ""
    pov: str = ""
    characters: dict[str, str] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)
    arc: list[str] = Field(default_factory=list)


class ChapterPlanOut(BaseModel):
    number: int = 0
    title: str = ""
    goal: str = ""
    hook: str = ""
    turn: str = ""
    cliffhanger: str = ""
    beats: list[str] = Field(default_factory=list)


class OutlineOut(BaseModel):
    chapters: list[ChapterPlanOut] = Field(default_factory=list)


class CriticOut(BaseModel):
    hook: float = 5
    momentum: float = 5
    payoff: float = 5
    cliffhanger: float = 5
    clarity: float = 5
    notes: str = ""


class SummaryOut(BaseModel):
    summary: str = ""
    open_threads: list[str] = Field(default_factory=list)


class MatrixOut(BaseModel):
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    dominance: float = 0.5
    warmth: float = 0.5
    impulsivity: float = 0.5
    honesty: float = 0.5
    courage: float = 0.5
    humor: float = 0.5
    ambition: float = 0.5
    loyalty: float = 0.5
    curiosity: float = 0.5
    cynicism: float = 0.5
    values: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    wants: str = ""
    needs: str = ""
    wound: str = ""
    flaw: str = ""
    lines_wont_cross: list[str] = Field(default_factory=list)


class VoiceOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    register_: str = Field("casual", alias="register")
    sentence_length: str = "medium"
    vocabulary: str = ""
    tics: list[str] = Field(default_factory=list)
    catchphrases: list[str] = Field(default_factory=list)
    swears: bool = False
    humor_style: str = ""
    avoids_talking_about: list[str] = Field(default_factory=list)


class StateOut(BaseModel):
    mood: str = "steady"
    goal: str = ""
    knows: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)


class RelationshipOut(BaseModel):
    trust: float = 0.5
    affection: float = 0.5
    respect: float = 0.5
    fear: float = 0.0
    stance: str = ""


class SheetOut(BaseModel):
    name: str = ""
    aliases: list[str] = Field(default_factory=list)
    role: str = "supporting"
    concept: str = ""
    appearance: str = ""
    backstory: str = ""
    matrix: MatrixOut = Field(default_factory=MatrixOut)
    voice: VoiceOut = Field(default_factory=VoiceOut)
    state: StateOut = Field(default_factory=StateOut)
    relationships: dict[str, RelationshipOut] = Field(default_factory=dict)


class BriefOut(BaseModel):
    goal: str = ""
    mood: str = ""
    actions: list[str] = Field(default_factory=list)
    dialogue: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    hides: list[str] = Field(default_factory=list)
    refuses: list[str] = Field(default_factory=list)
    conflict_with_plan: str = ""


class ReviewIssueOut(BaseModel):
    quote: str = ""
    problem: str = ""
    fix: str = ""


class ReviewOut(BaseModel):
    consistency: float = 10
    issues: list[ReviewIssueOut] = Field(default_factory=list)


class RelationshipDeltaOut(BaseModel):
    trust: float = 0.0
    affection: float = 0.0
    respect: float = 0.0
    fear: float = 0.0
    stance: str = ""
    note: str = ""


class MemoryOut(BaseModel):
    events: list[str] = Field(default_factory=list)
    feelings: str = ""
    mood: str = ""
    goal: str = ""
    location: str = ""
    learned: list[str] = Field(default_factory=list)
    new_secrets: list[str] = Field(default_factory=list)
    condition: list[str] = Field(default_factory=list)
    arc_progress: str = ""
    relationships: dict[str, RelationshipDeltaOut] = Field(default_factory=dict)


def schema_of(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema in the shape Ollama / OpenAI-compatible servers accept (no $defs indirection
    issues for simple models; nested models are inlined by pydantic when possible)."""
    return model.model_json_schema()
