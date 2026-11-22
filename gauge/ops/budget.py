"""An error budget for unattended runs: how many failures the daemon may spend before it slows down.

A loop that fails the same way every cycle should not keep writing chapters, publishing and running bots
at full speed while nobody is watching. The budget counts unresolved journal errors in a rolling window:

* healthy   (under half spent)   normal pace, everything enabled as configured
* strained  (half or more spent) the daemon doubles its sleep between cycles
* exhausted (all spent)          the daemon sleeps `max_backoff` times longer and holds its optional
                                 automation (chapter writes, newsdesk, bots, auto-applying self-fixes).
                                 Learning continues, and self-heal still *proposes* fixes, since that is
                                 the way back.

The budget refills on its own as errors age out of the window, or at once when they are resolved
(`gauge self journal`). An allowance of 0 turns the budget off.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from ..util import utcnow


@dataclass
class BudgetStatus:
    enabled: bool
    allowed: int
    window_hours: int
    spent: int
    remaining: int
    state: str                      # "off" | "healthy" | "strained" | "exhausted"
    backoff: float                  # multiply the sleep between cycles by this
    by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def exhausted(self) -> bool:
        return self.state == "exhausted"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ErrorBudget:
    def __init__(self, db: Any, allowed: int = 10, window_hours: int = 24, max_backoff: int = 4):
        self.db = db
        self.allowed = max(0, int(allowed))
        self.window_hours = max(1, int(window_hours))
        self.max_backoff = max(1, int(max_backoff))

    def status(self, now: Optional[datetime] = None) -> BudgetStatus:
        if self.allowed == 0:
            return BudgetStatus(False, 0, self.window_hours, 0, 0, "off", 1.0)
        since = (now or utcnow()) - timedelta(hours=self.window_hours)
        rows = self.db.journal_since(since.isoformat())
        spent = len(rows)
        by_kind = dict(Counter(r["kind"] for r in rows).most_common(8))
        if spent >= self.allowed:
            state, backoff = "exhausted", float(self.max_backoff)
        elif spent * 2 >= self.allowed:
            state, backoff = "strained", float(min(2, self.max_backoff))
        else:
            state, backoff = "healthy", 1.0
        return BudgetStatus(True, self.allowed, self.window_hours, spent, max(0, self.allowed - spent), state, backoff, by_kind)


def budget_for(cfg: Any, db: Any) -> ErrorBudget:
    d = cfg.daemon
    return ErrorBudget(db, d.error_budget, d.error_window_hours, d.max_backoff)
