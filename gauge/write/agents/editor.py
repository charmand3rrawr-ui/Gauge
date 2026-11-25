"""The Line Editor: prose-level quality and voice drift, mostly measured, partly judged."""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from ...features import voice_features
from ...text import sentences, words
from .base import AgentReview, SceneContext, StoryAgent

SYSTEM = """You are a line editor for serial fiction. You do not touch plot. You fix sentences: filter words, limp
verbs, stacked adverbs, fancy dialogue tags, telling where showing would land, rhythm that never varies."""

FILTER_WORDS = {"felt", "saw", "heard", "noticed", "realized", "realised", "seemed", "wondered", "knew", "thought",
                "began", "started", "decided", "watched", "looked"}
FANCY_TAGS = re.compile(r"\b(exclaimed|opined|queried|intoned|ejaculated|expostulated|chortled|averred|interjected)\b", re.I)


class EditorState(BaseModel):
    fingerprint: dict[str, float] = Field(default_factory=dict)  # voice features from the first chapters
    fingerprint_chapters: int = 0
    history: list[dict] = Field(default_factory=list)


def measure(text: str) -> dict[str, float]:
    ws = [w.lower() for w in words(text)]
    n = max(1, len(ws))
    ss = sentences(text)
    lens = [len(words(s)) for s in ss] or [0]
    mean = sum(lens) / len(lens)
    var = (sum((x - mean) ** 2 for x in lens) / len(lens)) ** 0.5
    m = {
        "filter_per_1k": 1000.0 * sum(1 for w in ws if w in FILTER_WORDS) / n,
        "fancy_tags": float(len(FANCY_TAGS.findall(text))),
        "sent_len_std": var,
        "very_per_1k": 1000.0 * sum(1 for w in ws if w in {"very", "really", "quite", "suddenly", "just"}) / n,
    }
    m.update(voice_features(text))
    return m


class LineEditor(StoryAgent):
    name = "line_editor"
    title = "Line Editor"
    weight = 1.0
    state_model = EditorState

    def drift(self, m: dict[str, float]) -> float:
        fp = self.state.fingerprint
        if not fp:
            return 0.0
        keys = ("first_person_ratio", "third_person_ratio", "adverb_ratio", "type_token_ratio")
        return sum(abs(m.get(k, 0.0) - fp.get(k, 0.0)) for k in keys) / len(keys)

    def brief(self, ctx: SceneContext) -> str:
        fp = self.state.fingerprint
        if not fp:
            return ""
        pov = "first person" if fp.get("first_person_ratio", 0) > fp.get("third_person_ratio", 0) else "third person"
        return f"- keep the established voice: {pov}, adverbs {'sparse' if fp.get('adverb_ratio', 0) < 0.02 else 'moderate'}, varied sentence length"

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        m = measure(text)
        issues = []
        score = 1.0
        if m["filter_per_1k"] > 12:
            score -= 0.15
            issues.append(f"{m['filter_per_1k']:.0f} filter words per 1000 (felt/saw/realized...); show the thing, not the noticing")
        if m["fancy_tags"] > 0:
            score -= 0.1
            issues.append("replace ornate dialogue tags (exclaimed, queried...) with 'said' or an action beat")
        if m["sent_len_std"] < 4.0:
            score -= 0.15
            issues.append("sentence lengths barely vary; mix fragments with long runs for rhythm")
        if m["very_per_1k"] > 10:
            score -= 0.1
            issues.append("too many intensifiers (very, really, suddenly, just)")
        d = self.drift(m)
        if d > 0.05:
            score -= min(0.2, d)
            issues.append("narrative voice has drifted from the story's established voice (POV / adverb density); realign")
        j = self.ask_json("review", f"""Give the three most valuable line-level fixes for this chapter. Not plot. Not praise.
Respond with ONLY JSON: {{"fixes": [{{"quote": "exact words", "problem": "what is weak", "fix": "the rewrite or the move"}}],
"voice_ok": true/false, "notes": "one line"}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.2, max_tokens=600)
        if not any(k in j for k in ("fixes", "voice_ok", "notes")):
            self.no_answer()  # the measured verdict above stands on its own
        for f in (j.get("fixes") or [])[:3] if isinstance(j.get("fixes"), list) else []:
            if isinstance(f, dict) and f.get("quote"):
                issues.append(f"line: \"{str(f['quote'])[:80]}\" - {str(f.get('problem', ''))[:100]}. Try: {str(f.get('fix', ''))[:120]}")
        if j.get("voice_ok") is False:
            score -= 0.1
        return AgentReview(agent=self.name, score=max(0.0, score), issues=issues[:6], notes=str(j.get("notes", ""))[:160],
                           data={"filter_per_1k": round(m["filter_per_1k"], 1), "sent_len_std": round(m["sent_len_std"], 1), "drift": round(d, 3)})

    def remember(self, text: str, ctx: SceneContext) -> None:
        m = measure(text)
        if self.state.fingerprint_chapters < 3:
            k = self.state.fingerprint_chapters
            for key, v in m.items():
                self.state.fingerprint[key] = (self.state.fingerprint.get(key, 0.0) * k + v) / (k + 1)
            self.state.fingerprint_chapters += 1
        self.state.history.append({"chapter": ctx.number, "filter_per_1k": round(m["filter_per_1k"], 1), "drift": round(self.drift(m), 3)})
        self.state.history = self.state.history[-100:]

    def report(self) -> str:
        fp = self.state.fingerprint
        head = "Voice fingerprint: " + ", ".join(f"{k} {v:.3f}" for k, v in fp.items() if k in ("first_person_ratio", "adverb_ratio", "type_token_ratio", "sent_len_std"))
        return head + "\n" + "\n".join(f"  ch {h['chapter']}: filter {h['filter_per_1k']}/1k drift {h['drift']}" for h in self.state.history[-8:])
