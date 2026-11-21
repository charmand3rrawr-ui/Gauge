"""Draft a labelled opinion column from the author's own thesis — honest, disclosed, not news.

The author supplies the argument they want to make; Gauge drafts the commentary, labels it as opinion,
attaches the disclosed byline, checks the headline for clickbait, and flags every factual-looking claim
for the author to verify. The voice can be punchy and contrarian — what it may not be is a real outlet's
or a shared pseudonym's (the byline check enforces that), and it may not invent facts, quotes or
statistics and present them as sourced. It is an op-ed assistant, not a news wire.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm.client import LLM, generate_json
from .integrity import OPINION_DISCLAIMER, check_byline, check_headline, flag_claims

_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "body": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["title", "body"],
}

_SYSTEM = ("You are an op-ed assistant. You write clearly-labelled opinion commentary that argues the author's "
           "thesis in their voice. You may be punchy and contrarian. You must NOT invent facts, statistics, quotes "
           "or events, or present the piece as news reporting. Where the argument needs evidence, say so plainly "
           "rather than fabricating it. This is the author's disclosed opinion, not journalism.")


@dataclass
class OpinionDraft:
    title: str
    body: str           # includes the opinion label
    byline: str
    summary: str = ""
    headline_flags: list[str] = field(default_factory=list)
    claims_to_verify: list[str] = field(default_factory=list)


# Voice registers for the column. All honest and labelled; "contrarian" is the punchy, consensus-skeptical
# register (a ZeroHedge-like flavour) WITHOUT a borrowed byline and WITHOUT inventing facts.
VOICES = {
    "plain": "Write clearly and evenly; let the argument carry it.",
    "punchy": "Short sentences. Strong verbs. Confident and vivid, but accurate.",
    "contrarian": "Skeptical of the consensus and the official line, sardonic and sharp, challenging received "
                  "wisdom — but every claim honest and your own, never invented, and always labelled as opinion.",
}


def draft_opinion(llm: LLM, thesis: str, byline: str, angle: str = "", voice: str = "plain") -> OpinionDraft:
    ok, why = check_byline(byline)
    if not ok:
        raise PermissionError(why)
    voice_note = VOICES.get(voice, VOICES["plain"])
    prompt = ("Draft an opinion column (not news). Argue this thesis; do not invent facts, quotes or statistics, "
              "and mark where evidence would be needed.\n"
              f"VOICE: {voice_note}\nTHESIS: {thesis}\n" + (f"ANGLE: {angle}\n" if angle else "") +
              "Return JSON {title, body, summary}. The title must be accurate, not clickbait.")
    try:
        d: dict[str, Any] = generate_json(llm, prompt, system=_SYSTEM, schema=_SCHEMA, temperature=0.8,
                                          max_tokens=1200, purpose="opinion")
    except Exception:
        d = {"title": thesis[:80], "body": thesis, "summary": ""}
    title = str(d.get("title", "")).strip()
    body = str(d.get("body", "")).strip()
    # Prefer an eye-catching, in-context headline over a weak or clickbait one.
    try:
        from .headline import suggest_headlines
        better = suggest_headlines(llm, body, n=4, angle=angle)
        if better and (not check_headline(title).ok or better[0].score > 0.5):
            title = better[0].text
    except Exception:
        pass
    labelled = f"*{OPINION_DISCLAIMER}*\n\nBy {byline}\n\n{body}"
    hc = check_headline(title)
    return OpinionDraft(title=title, body=labelled, byline=byline, summary=str(d.get("summary", "")).strip(),
                        headline_flags=hc.flags, claims_to_verify=flag_claims(body))
