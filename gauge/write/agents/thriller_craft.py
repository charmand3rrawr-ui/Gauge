"""The Thriller Craft agent: suspense, a fairly clued mystery, a consistent menacing antagonist, and
prose whose forensic/criminological/psychological/engineering register reads expert (roadmap craft).

Measured, no model calls. Attaches itself only when the genre or tone is dark (crime, thriller, mystery,
noir, horror, psychological). It improves the *storytelling* of this kind of fiction; it does not and must
not supply operational real-world methods — its register check rewards authentic vocabulary for
verisimilitude only.

The author declares the mystery's scaffolding in config (``agents.thriller_*``): the key clues the
solution rests on, the red herrings, the chapter the truth is revealed, the domains whose register the
story should carry, and (optionally) the antagonist's signature lexicon. With nothing declared it still
scores suspense and register and watches the antagonist's signature for drift.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..thriller import (
    clues_present,
    cover_collisions,
    covers_present,
    fair_play,
    leaked_secrets,
    parse_covers,
    parse_secrets,
    register_profile,
    signature_terms,
    suspense_profile,
    suspense_score,
)
from .base import AgentReview, SceneContext, StoryAgent


class ThrillerState(BaseModel):
    seen_clues: list[str] = Field(default_factory=list)       # declared clues shown to the reader so far
    resolved_herrings: list[str] = Field(default_factory=list)
    signature: list[str] = Field(default_factory=list)        # antagonist MO terms established
    aliases_seen: list[str] = Field(default_factory=list)     # cover identities that have appeared
    suspense_history: list[float] = Field(default_factory=list)
    chapters: int = 0


class ThrillerCraft(StoryAgent):
    name = "thriller_craft"
    title = "Thriller Craft"
    weight = 0.8
    state_model = ThrillerState

    # -- declared scaffolding (from config via options) ----------------------
    def _clues(self) -> list[str]:
        return [c for c in self.options.get("clues", []) if str(c).strip()]

    def _herrings(self) -> list[str]:
        return [h for h in self.options.get("red_herrings", []) if str(h).strip()]

    def _reveal(self) -> int:
        try:
            return int(self.options.get("reveal_chapter", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _domains(self) -> list[str]:
        return [d for d in self.options.get("domains", []) if str(d).strip()] or None

    def _sig_lexicon(self) -> set[str]:
        return {s.strip() for s in self.options.get("signature", []) if str(s).strip()}

    def _secrets(self):
        return parse_secrets(self.options.get("secrets", []))

    def _covers(self):
        return parse_covers(self.options.get("covers", []))

    # -- brief ---------------------------------------------------------------
    def brief(self, ctx: SceneContext) -> str:
        lines = ["- end the chapter on a hook (a question, a reveal, or a turn)",
                 "- keep tension on the page: dread, a ticking clock, or raised stakes",
                 "- make any technical/forensic work concrete and specific, never hand-waved"]
        declared = self._clues()
        if declared:
            unseen = [c for c in declared if c not in self.state.seen_clues]
            reveal = self._reveal()
            if unseen and reveal and ctx.number >= reveal - 2:
                lines.append(f"- fair play: the reveal is near (ch {reveal}); these key clues are still unshown — plant one: "
                             + "; ".join(unseen[:4]))
        if self.state.signature:
            lines.append("- keep the antagonist's signature consistent: " + ", ".join(self.state.signature[:6]))
        # What must stay hidden this chapter (a mole's name, a blown cover, the twist).
        keep_hidden = [s.phrase for s in self._secrets() if s.chapter > ctx.number]
        if keep_hidden:
            lines.append("- do not reveal yet (hold until their chapter): " + "; ".join(keep_hidden[:5]))
        # Genre tropes for the declared domains — how to dramatise them, not how to perform them.
        from ..thriller import TROPES
        for dom in (self._domains() or []):
            for trope in TROPES.get(dom, [])[:2]:
                lines.append(f"- {dom}: {trope}")
        return "\n".join(lines)

    # -- plan notes ----------------------------------------------------------
    def plan_notes(self, project) -> str:
        declared = self._clues()
        if not declared:
            return ""
        unseen = [c for c in declared if c not in self.state.seen_clues]
        if unseen:
            return "Mystery debts (plant these clues before the reveal): " + "; ".join(unseen[:6])
        return ""

    # -- review --------------------------------------------------------------
    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        from ...text import word_count
        if word_count(text) < 120:
            return None
        issues: list[str] = []

        sp = suspense_profile(text)
        issues += sp.issues
        sc = suspense_score(sp)

        rp = register_profile(text, self._domains())
        issues += rp.issues

        # Antagonist signature drift: terms established before but absent for a long stretch, or a brand-new
        # signature appearing late with no prior basis, is worth a nudge (soft, informational).
        sig_now = signature_terms(text, self._sig_lexicon() or None)

        # Fair-play bookkeeping.
        seen_now = clues_present(text, self._clues())
        self._seen_now = seen_now
        self._resolved_now = {h for h in self._herrings() if h.lower() in text.lower()}
        reveal = self._reveal()
        if reveal and ctx.number == reveal:
            # At the reveal, everything must have been played fair.
            seen = set(self.state.seen_clues) | seen_now
            resolved = set(self.state.resolved_herrings) | self._resolved_now
            fp = fair_play(self._clues(), self._herrings(), seen, resolved)
            for c in fp.missing_clues[:3]:
                issues.append(f"unfair reveal: the clue '{c}' was never shown to the reader before now; plant it earlier")
            for h in fp.unresolved_herrings[:2]:
                issues.append(f"red herring '{h}' is never dispelled; resolve it so the misdirection reads as fair")

        # Secrets that leak before their reveal chapter (a mole named too early, a cover blown ahead of time).
        leaks = leaked_secrets(text, self._secrets(), ctx.number)
        for p in leaks[:2]:
            issues.append(f"secret leaked early: '{p}' surfaces before its reveal chapter; hold it back to keep the turn")

        # Cover/legend ledger + accidental alias clash (two characters under one cover name).
        covers = self._covers()
        self._aliases_now = sorted(covers_present(text, covers))
        if ctx.number == 1:  # warn once, at the first chapter
            for alias in cover_collisions(covers):
                issues.append(f"two characters share the cover alias '{alias}'; confirm that is intended, or the legends will blur")

        self._sig_now = sorted(sig_now)
        score = round(0.6 * sc + 0.4 * (1.0 if not issues else max(0.4, 1.0 - 0.15 * len(issues))), 3)
        return AgentReview(agent=self.name, score=score, issues=issues[:3],
                           data={"suspense": sc, "ends_on_hook": sp.ends_on_hook, "ticking_clock": sp.ticking_clock,
                                 "register": {k: v for k, v in rp.hits.items() if v}, "clues_shown": sorted(seen_now),
                                 "aliases": self._aliases_now})

    # -- remember ------------------------------------------------------------
    def remember(self, text: str, ctx: SceneContext) -> None:
        for c in getattr(self, "_seen_now", set()):
            if c not in self.state.seen_clues:
                self.state.seen_clues.append(c)
        for h in getattr(self, "_resolved_now", set()):
            if h not in self.state.resolved_herrings:
                self.state.resolved_herrings.append(h)
        for s in getattr(self, "_sig_now", []):
            if s not in self.state.signature:
                self.state.signature.append(s)
        self.state.signature = self.state.signature[-40:]
        for a in getattr(self, "_aliases_now", []):
            if a not in self.state.aliases_seen:
                self.state.aliases_seen.append(a)
        sc = None
        last = getattr(self, "_seen_now", None)
        if last is not None:  # review ran
            sp = suspense_profile(text)
            sc = suspense_score(sp)
            self.state.suspense_history.append(sc)
            self.state.suspense_history = self.state.suspense_history[-200:]
        self.state.chapters = max(self.state.chapters, ctx.number)

    # -- report --------------------------------------------------------------
    def report(self) -> str:
        if not self.state.chapters:
            return "(no chapters tracked yet)"
        bits = []
        if self.state.suspense_history:
            avg = sum(self.state.suspense_history) / len(self.state.suspense_history)
            bits.append(f"mean suspense {avg:.2f} over {len(self.state.suspense_history)} ch")
        declared = self._clues()
        if declared:
            bits.append(f"clues shown {len(self.state.seen_clues)}/{len(declared)}")
            unseen = [c for c in declared if c not in self.state.seen_clues]
            if unseen:
                bits.append("still unplanted: " + "; ".join(unseen[:5]))
        if self.state.signature:
            bits.append("antagonist signature: " + ", ".join(self.state.signature[:8]))
        if self.state.aliases_seen:
            bits.append("covers in play: " + ", ".join(self.state.aliases_seen[:8]))
        secrets = self._secrets()
        if secrets:
            pending = [f"{s.phrase} (ch {s.chapter})" for s in secrets if s.chapter > self.state.chapters]
            if pending:
                bits.append("still to reveal: " + "; ".join(pending[:5]))
        return "; ".join(bits) or "(tracking)"
