"""The Archivist: a canon ledger of facts with deterministic contradiction checks and retrieval."""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import AgentReview, SceneContext, StoryAgent

SYSTEM = """You are the archivist of a long serial. You record concrete, checkable facts: names, ages, eye colour,
who knows what, where objects are, distances, dates, rules stated as true. You never record opinions."""

_SPLIT = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return " ".join(_SPLIT.split(str(s).lower())).strip()


class Fact(BaseModel):
    subject: str
    attribute: str
    value: str
    chapter: int
    quote: str = ""


class CanonState(BaseModel):
    facts: list[Fact] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    names: dict[str, int] = Field(default_factory=dict)  # every proper name used so far -> how often


class Archivist(StoryAgent):
    name = "archivist"
    title = "Archivist"
    weight = 1.5
    state_model = CanonState

    # ---- ledger ------------------------------------------------------------
    def lookup(self, subject: str, attribute: Optional[str] = None) -> list[Fact]:
        s = _norm(subject)
        out = [f for f in self.state.facts if s and (s in _norm(f.subject) or _norm(f.subject) in s)]
        if attribute:
            a = _norm(attribute)
            out = [f for f in out if a in _norm(f.attribute) or _norm(f.attribute) in a]
        return out

    def search(self, text: str, limit: int = 12) -> list[Fact]:
        words = {w for w in _norm(text).split() if len(w) >= 3}
        scored = []
        for f in self.state.facts:
            hay = _norm(f"{f.subject} {f.attribute} {f.value}").split()
            hit = sum(1 for w in hay if w in words)
            if hit:
                scored.append((hit, f))
        scored.sort(key=lambda t: (-t[0], -t[1].chapter))
        return [f for _, f in scored[:limit]]

    def contradictions(self, new: list[Fact]) -> list[dict[str, Any]]:
        out = []
        for nf in new:
            for of in self.state.facts:
                if _norm(of.subject) == _norm(nf.subject) and _norm(of.attribute) == _norm(nf.attribute):
                    a, b = _norm(of.value), _norm(nf.value)
                    if a and b and a != b and a not in b and b not in a:
                        out.append({"subject": nf.subject, "attribute": nf.attribute, "was": of.value, "was_chapter": of.chapter,
                                    "now": nf.value, "quote": nf.quote})
        # The ledger above holds the latest facts only. The chronicle holds every fact the story has
        # established, so a lasting fact from thousands of chapters ago is still checked.
        try:
            from ..chronicle import DB, Chronicle
            from ..chronicle import _norm as norm

            if (self.dir.parent / DB).exists():
                chron = Chronicle.at(self.dir.parent)
                seen = {(c["subject"], c["attribute"]) for c in out}
                for nf in new:
                    if (nf.subject, nf.attribute) in seen:
                        continue
                    for old in chron.history(nf.subject, nf.attribute):
                        if old["replaced_at"] is None and old["lasting"] and old["chapter"] < nf.chapter:
                            a, b = old["nv"], norm(nf.value)
                            if a != b and a not in b and b not in a:
                                out.append({"subject": nf.subject, "attribute": nf.attribute, "was": old["value"],
                                            "was_chapter": old["chapter"], "now": nf.value, "quote": nf.quote})
                                break
        except Exception:
            pass
        return out

    def _extract(self, op: str, text: str, number: int) -> list[Fact]:
        d = self.ask_json(op, f"""Extract concrete canon facts from chapter {number}.
Respond with ONLY JSON: {{"facts": [{{"subject": "who/what", "attribute": "what about it (short, reusable, e.g. 'eye colour', 'location', 'knows')",
"value": "the fact", "quote": "<= 12 words from the text"}}]}} (6-14 facts, only things later chapters could contradict)

CHAPTER:
{text}""", system=SYSTEM, temperature=0.1, max_tokens=900)
        out = []
        for f in d.get("facts", []) if isinstance(d.get("facts"), list) else []:
            if isinstance(f, dict) and f.get("subject") and f.get("attribute") and f.get("value"):
                out.append(Fact(subject=str(f["subject"])[:60], attribute=str(f["attribute"])[:60], value=str(f["value"])[:160],
                                chapter=number, quote=str(f.get("quote", ""))[:120]))
        return out[:16]

    # ---- name index (measured; no model call) ------------------------------
    def _cast_names(self, ctx: SceneContext) -> set[str]:
        """Names known to be right: the cast list and its aliases, one word each."""
        try:
            from ..characters import Roster
            roster = Roster.load(ctx.project.dir)
            full = [c.name for c in roster.ordered()] + [a for c in roster.ordered() for a in c.aliases]
        except Exception:
            return set()
        return {w for name in full for w in str(name).split() if len(w) >= 3 and w[0].isupper()}

    def name_drift(self, text: str, ctx: SceneContext) -> list[tuple[str, str, int, int]]:
        from ..names import proper_names, spelling_drift
        cast = self._cast_names(ctx)
        now = proper_names(text, known=set(self.state.names) | cast)
        self._names_now = dict(now)
        return spelling_drift(now, self.state.names, protected=cast)

    # ---- moments -----------------------------------------------------------
    def brief(self, ctx: SceneContext) -> str:
        facts = self.search(ctx.plan_text + " " + ctx.cast_text[:400], limit=10)
        if not facts:
            return ""
        return "\n".join(f"- canon: {f.subject} / {f.attribute}: {f.value} (ch {f.chapter})" for f in facts)

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        drift = self.name_drift(text, ctx)
        drift_issues = [f"name spelling: '{v}' looks like a slip of '{c}' (used {n} times so far); use one spelling, "
                        "or add both to the cast list if they are different people" for v, c, _uses, n in drift[:3]]
        if not self.state.facts:
            if not drift:
                return None
            return AgentReview(agent=self.name, score=max(0.0, 1.0 - 0.1 * len(drift)), issues=drift_issues,
                               data={"conflicts": 0, "facts_checked": 0, "name_drift": [v for v, *_ in drift]})
        new = self._extract("review", text, ctx.number)
        conflicts = self.contradictions(new)
        issues = [f"contradiction: {c['subject']} / {c['attribute']} was '{c['was']}' (ch {c['was_chapter']}), now '{c['now']}' at \"{c['quote']}\""
                  for c in conflicts[:4]]
        return AgentReview(agent=self.name, score=max(0.0, 1.0 - 0.25 * len(conflicts) - 0.1 * len(drift)),
                           issues=issues + drift_issues,
                           data={"conflicts": len(conflicts), "facts_checked": len(new), "name_drift": [v for v, *_ in drift]})

    def remember(self, text: str, ctx: SceneContext) -> None:
        new = self._extract("remember", text, ctx.number)
        conflicts = self.contradictions(new)
        for c in conflicts:
            c["chapter"] = ctx.number
        self.state.conflicts = (self.state.conflicts + conflicts)[-50:]
        known = {(_norm(f.subject), _norm(f.attribute), _norm(f.value)) for f in self.state.facts}
        for f in new:
            if (_norm(f.subject), _norm(f.attribute), _norm(f.value)) not in known:
                self.state.facts.append(f)
        self.state.facts = self.state.facts[-2000:]
        # Name index: add this chapter's names. A slip that got through stays a rare name, so it never
        # becomes established enough to hide the next one.
        now = getattr(self, "_names_now", None)
        if now is None:
            self.name_drift(text, ctx)
            now = self._names_now
        for name, n in now.items():
            self.state.names[name] = self.state.names.get(name, 0) + int(n)
        if len(self.state.names) > 800:
            self.state.names = dict(sorted(self.state.names.items(), key=lambda kv: -kv[1])[:600])
        self._names_now = None

    def ask(self, question: str) -> str:
        facts = self.search(question, limit=15)
        if not facts:
            return "The canon ledger has nothing on that yet."
        listing = "\n".join(f"- {f.subject} / {f.attribute}: {f.value} (ch {f.chapter}, \"{f.quote}\")" for f in facts)
        d = self.ask_json("ask", f"""Answer the author's question using ONLY these canon facts. If they do not settle it, say so.

FACTS:
{listing}

QUESTION: {question}

Respond with ONLY JSON: {{"answer": "one or two sentences citing chapters", "confidence": 0-1}}""", system=SYSTEM, temperature=0.1)
        return str(d.get("answer") or "The ledger does not settle that.")

    def name_index(self, limit: int = 40) -> str:
        if not self.state.names:
            return "(no names indexed yet)"
        rows = sorted(self.state.names.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return "\n".join(f"  {n:>4}  {name}" for name, n in rows)

    def report(self) -> str:
        lines = [f"{len(self.state.facts)} facts, {len(self.state.conflicts)} recorded conflicts, {len(self.state.names)} names indexed"]
        for f in self.state.facts[-15:]:
            lines.append(f"  ch {f.chapter}: {f.subject} / {f.attribute} = {f.value}")
        for c in self.state.conflicts[-5:]:
            lines.append(f"  CONFLICT ch {c.get('chapter')}: {c['subject']} / {c['attribute']}: '{c['was']}' -> '{c['now']}'")
        return "\n".join(lines)
