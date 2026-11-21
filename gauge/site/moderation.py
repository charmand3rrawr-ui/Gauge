"""A comment-moderation queue built on the sanitiser (roadmap item 88).

Incoming reader comments pass through the same injection scrubber that protects ingestion, plus a small
abuse check. Each lands in a queue with a verdict — allow, flag (hold for a human), or block — so nothing
unreviewed reaches the site, and obvious spam or injection is caught automatically. The author approves
or rejects what is held; approved comments are the scrubbed text, never the raw input.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..security.sanitize import find_injections, scrub

_ABUSE = re.compile(r"\b(k[i1]ll yourself|kys|die in a|n[i1]gger|f[a4]ggot|retard|rape you|\bslur\b)\b", re.I)
_SPAM = re.compile(r"(https?://\S+){3,}|\b(viagra|casino|crypto giveaway|free money|click here now)\b", re.I)


@dataclass
class Comment:
    id: str
    author: str
    text: str            # the scrubbed text (safe to display)
    status: str          # allow | flag | block | approved | rejected
    reasons: list[str] = field(default_factory=list)
    at: str = ""


def moderate(text: str, author: str = "") -> Comment:
    """Scrub and classify a raw comment. Never returns raw text — the stored text is the scrubbed version."""
    from ..util import utcnow
    res = scrub(text or "", action="scrub")
    reasons: list[str] = []
    status = "allow"
    if _ABUSE.search(text or ""):
        reasons.append("abuse/slur")
        status = "block"
    if _SPAM.search(text or ""):
        reasons.append("spam")
        status = "block" if status == "block" else "flag"
    inj = find_injections(text or "")
    if inj:
        reasons.append(f"instruction-shaped text ({inj[0]['label']})")
        status = "flag" if status != "block" else status
    if res.suspicious and "instruction-shaped text" not in " ".join(reasons):
        reasons.append("suspicious content")
        status = "flag" if status == "allow" else status
    return Comment(id=uuid.uuid4().hex[:8], author=(author or "anon")[:60], text=res.text,
                   status=status, reasons=reasons, at=utcnow().isoformat(timespec="seconds"))


class ModerationQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._items: list[Comment] = []
        if self.path.exists():
            try:
                self._items = [Comment(**d) for d in json.loads(self.path.read_text(encoding="utf-8"))]
            except Exception:
                self._items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(i) for i in self._items], indent=2, ensure_ascii=False), encoding="utf-8")

    def submit(self, text: str, author: str = "") -> Comment:
        c = moderate(text, author)
        self._items.append(c)
        self.save()
        return c

    def items(self, status: str | None = None) -> list[Comment]:
        return [c for c in self._items if status is None or c.status == status]

    def _set(self, cid: str, status: str) -> bool:
        for c in self._items:
            if c.id == cid:
                c.status = status
                self.save()
                return True
        return False

    def approve(self, cid: str) -> bool:
        return self._set(cid, "approved")

    def reject(self, cid: str) -> bool:
        return self._set(cid, "rejected")

    def published(self) -> list[Comment]:
        """Comments safe to show: explicitly approved, or auto-allowed and never flagged/blocked."""
        return [c for c in self._items if c.status in ("approved", "allow")]
