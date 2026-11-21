"""The Scout as a member of the writing room: decides what real-world detail a scene needs,
goes and gets it, and fact-checks the draft's real-world claims."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from ...research.scout import SOURCE_HINTS, Scout
from .base import AgentReview, SceneContext, StoryAgent, as_str_list

SYSTEM = """You are a research scout for a novelist. You know which concrete, checkable details make a scene feel
real: how a job is actually done, what a place smells like, what a technology can and cannot do, what the
weather is like there in that month, what people of that time and trade would say. You ask precise questions."""


class ScoutState(BaseModel):
    researched: list[dict[str, Any]] = Field(default_factory=list)  # per chapter: queries and hit counts
    claims_checked: int = 0
    claims_unsupported: int = 0


class ScoutAgent(StoryAgent):
    name = "scout"
    title = "Scout"
    weight = 0.5
    state_model = ScoutState

    def __init__(self, llm, project_dir, options=None, scout: Optional[Scout] = None):
        super().__init__(llm, project_dir, options)
        self.project_dir = project_dir
        self.scout = scout
        self.max_queries = int(self.options.get("max_queries", 3))
        self.fact_check = bool(self.options.get("fact_check", True))

    def _questions(self, ctx: SceneContext, world_hint: str) -> list[dict[str, str]]:
        sources = ", ".join(f"{k} ({v})" for k, v in SOURCE_HINTS.items() if self.scout is None or k in self.scout.sources)
        d = self.ask_json("questions", f"""Chapter {ctx.number} is planned as: {ctx.plan_text}
Setting: {world_hint or '(see bible)'}
Cast: {ctx.cast_text[:400]}
Genres: {', '.join(ctx.genres)}

Which {self.max_queries} real-world lookups would make this scene concrete and believable? Prefer specific nouns
and processes over themes. Choose the best source for each: {sources}.
Respond with ONLY JSON: {{"queries": [{{"query": "2-6 word search", "source": "<source name>", "why": "what detail it will give the scene"}}]}}""",
                         system=SYSTEM, temperature=0.5, max_tokens=400)
        out = []
        for q in d.get("queries", []) if isinstance(d.get("queries"), list) else []:
            if isinstance(q, dict) and q.get("query"):
                out.append({"query": str(q["query"])[:80], "source": str(q.get("source", "wikipedia")).lower()[:20], "why": str(q.get("why", ""))[:160]})
        return out[: self.max_queries]

    def brief(self, ctx: SceneContext) -> str:
        if self.scout is None:
            return ""
        queries = self._questions(ctx, ctx.world_text[:300])
        if not queries:
            return ""
        dossier = self.scout.research(queries, project=ctx.project.state.name, chapter=ctx.number)
        self.scout.file(self.project_dir, dossier, f"chapter {ctx.number:03d} {ctx.plan.title}")
        self.state.researched.append({"chapter": ctx.number, "queries": [f.query for f in dossier.findings],
                                      "hits": sum(len(f.snippets) for f in dossier.findings)})
        self.state.researched = self.state.researched[-100:]
        cards = dossier.cards(max_items=6, max_words=70)
        if not cards:
            return ""
        return ("real-world detail found for this scene (use the facts and texture; never copy the sentences):\n" + cards)

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        if self.scout is None or not self.fact_check:
            return None
        d = self.ask_json("claims", f"""List up to 3 checkable real-world claims in this chapter (history, geography, science, how a
job or device works). Skip anything that is clearly the story's own invention (its magic, its fictional city).
Respond with ONLY JSON: {{"claims": [{{"claim": "one sentence", "query": "2-5 word lookup"}}]}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.1, max_tokens=300)
        claims = [c for c in (d.get("claims") or []) if isinstance(c, dict) and c.get("claim") and c.get("query")] if isinstance(d.get("claims"), list) else []
        if not claims:
            return AgentReview(agent=self.name, score=1.0, data={"claims": 0})
        issues = []
        unsupported = 0
        for c in claims[:3]:
            finding = self.scout.search(str(c["query"]), "wikipedia", k=2)
            evidence = "\n".join(s.card(120) for s in finding.snippets) or "(nothing found)"
            v = self.ask_json("verify", f"""Claim: {c['claim']}
Evidence:
{evidence}

Respond with ONLY JSON: {{"verdict": "supported|contradicted|unknown", "note": "one line"}}""", system=SYSTEM, temperature=0.1, max_tokens=120)
            verdict = str(v.get("verdict", "unknown")).lower()
            self.state.claims_checked += 1
            if verdict == "contradicted":
                unsupported += 1
                issues.append(f"real-world claim looks wrong: \"{str(c['claim'])[:100]}\" - {str(v.get('note', ''))[:120]}")
        self.state.claims_unsupported += unsupported
        return AgentReview(agent=self.name, score=max(0.0, 1.0 - 0.3 * unsupported), issues=issues,
                           data={"claims": len(claims), "contradicted": unsupported})

    def report(self) -> str:
        lines = [f"claims checked {self.state.claims_checked}, contradicted {self.state.claims_unsupported}"]
        for r in self.state.researched[-8:]:
            lines.append(f"  ch {r['chapter']}: {', '.join(r['queries'])} ({r['hits']} hits)")
        return "\n".join(lines)


def extract_topics(bible_md: str, premise: str = "") -> list[str]:
    """Cheap, deterministic seed topics for a project dossier: proper nouns and job words in the bible."""
    import re

    text = f"{bible_md}\n{premise}"
    caps = re.findall(r"\b[A-Z][a-z]{3,}(?:\s[A-Z][a-z]{3,})?\b", text)
    stop = {"Series", "Bible", "Genres", "Premise", "Protagonist", "World", "Stakes", "Promise", "Tone", "Characters", "Rules", "Arc",
            "The", "She", "He", "They", "Chapter", "Story", "Point"}
    seen: list[str] = []
    for c in caps:
        if c not in stop and c not in seen:
            seen.append(c)
    return seen[:8]


def _as_list(v: Any) -> list[str]:
    return as_str_list(v)
