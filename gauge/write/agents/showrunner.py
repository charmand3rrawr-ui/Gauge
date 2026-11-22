"""The Showrunner: a ledger of promises and payoffs, so the serial keeps rewarding the reader."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from .base import AgentReview, SceneContext, StoryAgent, as_str_list

SYSTEM = """You are the showrunner of a long-running serial. You think in setups and payoffs: every question
opened is a debt to the reader; every payoff must open a bigger promise. You never let more than a few
chapters pass without the reader getting something they were promised."""


class Promise(BaseModel):
    id: int
    text: str
    kind: str = "question"  # question / threat / goal / setup / romance / mystery
    planted: int
    due: int
    paid: Optional[int] = None
    advanced: list[int] = Field(default_factory=list)


class LedgerState(BaseModel):
    promises: list[Promise] = Field(default_factory=list)
    next_id: int = 1
    last_payoff_chapter: int = 0
    max_gap: int = 4  # chapters without a payoff before the ledger starts complaining


# ---------------------------------------------------------------------------- promise/payoff heatmap (roadmap 31)

@dataclass
class HeatCell:
    chapter: int
    planted: int        # promises opened this chapter
    advanced: int       # promises visibly moved this chapter
    paid: int           # promises paid off this chapter
    open_after: int     # promises still open at the end of this chapter
    overdue_after: int  # open promises already past their due date by this chapter


def promise_heatmap(promises: list[Promise], upto: int) -> list[HeatCell]:
    """A per-chapter view of the whole serial's promise economy — what was planted, advanced, paid, and
    how many debts are open and overdue at each point. Measured; no model calls."""
    cells: list[HeatCell] = []
    for c in range(1, max(0, upto) + 1):
        planted = sum(1 for p in promises if p.planted == c)
        advanced = sum(1 for p in promises if c in p.advanced)
        paid = sum(1 for p in promises if p.paid == c)
        open_after = sum(1 for p in promises if p.planted <= c and (p.paid is None or p.paid > c))
        overdue_after = sum(1 for p in promises if p.planted <= c and p.due <= c and (p.paid is None or p.paid > c))
        cells.append(HeatCell(c, planted, advanced, paid, open_after, overdue_after))
    return cells


def heatmap_text(promises: list[Promise], upto: int, width: int = 60) -> str:
    """Render the heatmap as compact text: one row per chapter with plant(+)/advance(.)/payoff(*) markers
    and the running open/overdue counts, then a short gantt of the promises still open or overdue."""
    cells = promise_heatmap(promises, upto)
    if not cells:
        return "(no promises tracked yet)"
    lines = ["ch  plant/adv/paid   open  overdue"]
    for cell in cells[-width:]:
        marks = ("+" * cell.planted + "." * cell.advanced + "*" * cell.paid) or "·"
        flag = f"  <-- {cell.overdue_after} overdue" if cell.overdue_after else ""
        lines.append(f"{cell.chapter:>3}  {marks:<14}  {cell.open_after:>4}  {cell.overdue_after:>5}{flag}")
    # Gantt for promises that are still open (most overdue first).
    open_now = [p for p in promises if p.paid is None]
    if open_now:
        open_now.sort(key=lambda p: (p.due, p.planted))
        lines.append("")
        lines.append("open threads (planted -> due):")
        for p in open_now[:12]:
            od = " OVERDUE" if p.due <= upto else ""
            lines.append(f"  #{p.id} [{p.kind}] {p.text[:48]}  (ch {p.planted} -> {p.due}{od})")
    return "\n".join(lines)


class Showrunner(StoryAgent):
    name = "showrunner"
    title = "Showrunner"
    weight = 1.5
    state_model = LedgerState

    def open(self) -> list[Promise]:
        return [p for p in self.state.promises if p.paid is None]

    def due(self, chapter: int) -> list[Promise]:
        return [p for p in self.open() if p.due <= chapter]

    def _ledger_text(self, chapter: int) -> str:
        due = self.due(chapter)
        others = [p for p in self.open() if p not in due][:8]
        lines = [f"- OVERDUE/DUE: {p.text} (planted ch {p.planted}, due ch {p.due})" for p in due[:6]]
        lines += [f"- open: {p.text} (due ch {p.due})" for p in others]
        return "\n".join(lines) or "- (no open promises yet)"

    def plan_notes(self, project) -> str:
        n = project.next_chapter_number()
        due = self.due(n + 2)
        gap = n - self.state.last_payoff_chapter if self.state.last_payoff_chapter else 0
        notes = []
        if due:
            notes.append("promises that must be paid or visibly advanced in this stretch: " + "; ".join(p.text for p in due[:5]))
        if gap >= self.state.max_gap:
            notes.append(f"{gap} chapters since the last real payoff: the next stretch must deliver one early")
        return "\n".join(f"- {x}" for x in notes)

    def brief(self, ctx: SceneContext) -> str:
        if not self.state.promises:
            return "- plant at least one concrete question the reader will want answered within four chapters"
        due = self.due(ctx.number)
        out = []
        if due:
            out.append("pay or clearly advance: " + "; ".join(p.text for p in due[:3]))
        gap = ctx.number - self.state.last_payoff_chapter if self.state.last_payoff_chapter else 0
        if gap >= self.state.max_gap:
            out.append(f"no payoff for {gap} chapters: give the reader something promised, then open a bigger promise")
        if not out:
            out.append("open promises: " + "; ".join(p.text for p in self.open()[:4]))
        return "\n".join(f"- {o}" for o in out)

    def review(self, text: str, ctx: SceneContext) -> Optional[AgentReview]:
        d = self.ask_json("review", f"""Chapter {ctx.number} against the ledger.

LEDGER:
{self._ledger_text(ctx.number)}

Respond with ONLY JSON: {{"advanced": ["ledger items this chapter visibly moved"], "paid": ["ledger items it paid off"],
"stalling": true/false (nothing promised was touched), "notes": "one line"}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.2)
        paid, advanced = as_str_list(d.get("paid"), 6), as_str_list(d.get("advanced"), 8)
        stalling = bool(d.get("stalling", False)) and not paid and not advanced
        gap = ctx.number - self.state.last_payoff_chapter if self.state.last_payoff_chapter else 0
        score = 1.0
        issues = []
        if stalling:
            score -= 0.4
            issues.append("nothing the reader was promised moves in this chapter; advance at least one open thread")
        overdue = [p for p in self.due(ctx.number) if not any(p.text.lower()[:30] in x.lower() for x in paid + advanced)]
        if overdue:
            score -= min(0.3, 0.1 * len(overdue))
            issues.append("overdue promises ignored: " + "; ".join(p.text for p in overdue[:3]))
        if gap >= self.state.max_gap and not paid:
            score -= 0.2
            issues.append(f"{gap} chapters without a payoff; let this chapter pay one")
        self._last = {"paid": paid, "advanced": advanced}
        return AgentReview(agent=self.name, score=max(0.0, score), issues=issues, notes=str(d.get("notes", ""))[:200],
                           data={"paid": paid, "advanced": advanced, "open": len(self.open())})

    def remember(self, text: str, ctx: SceneContext) -> None:
        d = self.ask_json("remember", f"""Update the ledger after chapter {ctx.number}.

LEDGER:
{self._ledger_text(ctx.number)}

Respond with ONLY JSON: {{"planted": [{{"text": "new question/threat/goal the chapter opened", "kind": "question|threat|goal|setup|romance|mystery", "due_in": chapters until the reader expects an answer}}],
"paid": ["ledger items now paid off"], "advanced": ["ledger items moved forward"]}}

CHAPTER:
{text}""", system=SYSTEM, temperature=0.2)
        for item in d.get("planted", []) if isinstance(d.get("planted"), list) else []:
            if not isinstance(item, dict) or not item.get("text"):
                continue
            try:
                due_in = max(1, min(30, int(item.get("due_in", 4))))
            except (TypeError, ValueError):
                due_in = 4
            text_ = str(item["text"])[:200]
            if any(text_.lower() == p.text.lower() for p in self.open()):
                continue
            self.state.promises.append(Promise(id=self.state.next_id, text=text_, kind=str(item.get("kind", "question"))[:20],
                                               planted=ctx.number, due=ctx.number + due_in))
            self.state.next_id += 1
        paid = as_str_list(d.get("paid"), 8) + getattr(self, "_last", {}).get("paid", [])
        advanced = as_str_list(d.get("advanced"), 8) + getattr(self, "_last", {}).get("advanced", [])
        for p in self.open():
            key = p.text.lower()[:30]
            if any(key in x.lower() or x.lower()[:30] in p.text.lower() for x in paid):
                p.paid = ctx.number
                self.state.last_payoff_chapter = ctx.number
            elif any(key in x.lower() or x.lower()[:30] in p.text.lower() for x in advanced):
                p.advanced.append(ctx.number)
        self.state.promises = self.state.promises[-120:]

    def heatmap(self, width: int = 60) -> str:
        upto = max((p.paid or p.due for p in self.state.promises), default=0)
        upto = max(upto, max((p.planted for p in self.state.promises), default=0))
        return heatmap_text(self.state.promises, upto, width)

    def report(self) -> str:
        lines = [f"open {len(self.open())}, paid {sum(1 for p in self.state.promises if p.paid)}, last payoff ch {self.state.last_payoff_chapter}"]
        for p in self.state.promises[-20:]:
            status = f"paid ch {p.paid}" if p.paid else f"due ch {p.due}" + (" OVERDUE" if p.due < (self.state.promises[-1].planted if self.state.promises else 0) else "")
            lines.append(f"  #{p.id} [{p.kind}] {p.text}  (planted {p.planted}, {status})")
        return "\n".join(lines)
