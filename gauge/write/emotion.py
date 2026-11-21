"""A small emotional model for characters: pleasure, arousal, dominance (PAD), with decay toward a
baseline set by the personality matrix, and deterministic shading of how the character speaks."""
from __future__ import annotations

from pydantic import BaseModel


class Mood(BaseModel):
    pleasure: float = 0.0  # -1 miserable .. +1 elated
    arousal: float = 0.0  # -1 calm .. +1 agitated
    dominance: float = 0.0  # -1 cowed .. +1 in command

    def label(self) -> str:
        p, a, d = self.pleasure, self.arousal, self.dominance
        if a > 0.4 and p < -0.2:
            return "frightened" if d < 0 else "furious"
        if a > 0.4 and p > 0.2:
            return "exhilarated" if d > 0 else "giddy"
        if a < -0.3 and p < -0.2:
            return "defeated" if d < 0 else "grim"
        if a < -0.3 and p > 0.2:
            return "content"
        if p < -0.4:
            return "miserable"
        if p > 0.4:
            return "pleased"
        if d > 0.5:
            return "commanding"
        if d < -0.5:
            return "cowed"
        return "steady"


def baseline(matrix) -> Mood:
    """Where a character's mood settles when nothing is happening."""
    return Mood(pleasure=round(0.3 * (matrix.warmth - matrix.cynicism) + 0.2 * (0.5 - matrix.neuroticism), 3),
                arousal=round(0.6 * matrix.neuroticism - 0.3 + 0.3 * matrix.impulsivity, 3),
                dominance=round(0.8 * (matrix.dominance - 0.5) + 0.3 * (matrix.courage - 0.5), 3))


def apply_event(mood: Mood, delta: dict, base: Mood, decay: float = 0.35) -> Mood:
    """Decay toward baseline, then apply the chapter's emotional delta. Clamped to [-1, 1]."""

    def clamp(x: float) -> float:
        return max(-1.0, min(1.0, x))

    def step(cur: float, b: float, d) -> float:
        try:
            dv = float(d)
        except (TypeError, ValueError):
            dv = 0.0
        return clamp(cur + (b - cur) * decay + max(-1.0, min(1.0, dv)))

    return Mood(pleasure=round(step(mood.pleasure, base.pleasure, delta.get("pleasure", 0)), 3),
                arousal=round(step(mood.arousal, base.arousal, delta.get("arousal", 0)), 3),
                dominance=round(step(mood.dominance, base.dominance, delta.get("dominance", 0)), 3))


def speech_directives(mood: Mood, voice) -> list[str]:
    """How the current mood shades speech, independent of what the model would like to do."""
    out = []
    if mood.arousal > 0.5:
        out.append("speaks in fragments and interruptions; drops words; repeats a word when cornered")
    elif mood.arousal < -0.4:
        out.append("speaks slowly, in complete sentences, with long pauses")
    if mood.pleasure < -0.4:
        out.append("no jokes land; humour, if any, is bitter" if voice.humor_style else "flat, clipped answers")
    elif mood.pleasure > 0.4 and voice.humor_style:
        out.append(f"humour comes easily ({voice.humor_style})")
    if mood.dominance > 0.5:
        out.append("gives orders as statements; does not ask permission; interrupts")
    elif mood.dominance < -0.5:
        out.append("hedges, apologises, asks instead of tells, avoids eye contact in the prose")
    if mood.arousal > 0.3 and voice.swears:
        out.append("swears more than usual")
    if voice.tics and mood.arousal > 0.3:
        out.append(f"verbal tic surfaces: {voice.tics[0]}")
    return out


def drift_step(matrix, plan: dict[str, dict], chapter: int) -> dict[str, float]:
    """Move trait axes toward their arc targets by the remaining fraction. plan: axis -> {target, by}."""
    moved = {}
    for axis, spec in plan.items():
        try:
            target, by = float(spec["target"]), int(spec["by"])
        except (KeyError, TypeError, ValueError):
            continue
        cur = getattr(matrix, axis, None)
        if cur is None:
            continue
        remaining = max(1, by - chapter + 1)
        new = cur + (target - cur) / remaining
        new = max(0.0, min(1.0, new))
        if abs(new - cur) > 1e-6:
            setattr(matrix, axis, round(new, 3))
            moved[axis] = round(new, 3)
    return moved
