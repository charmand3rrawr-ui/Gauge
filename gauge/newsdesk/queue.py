"""The publishing queue: drafts wait here and go out on your schedule.

Every draft enters as `pending_review`. On each tick the queue publishes at most one item, and only when
the interval since the last publication has elapsed (default 30 minutes, configurable) and the daily cap
has room. An item is publishable when you have approved it — or, if you opted into auto-publishing, when
it is *clean*: no claims awaiting verification, no balance or headline flags, and no sign of copying a
source. Anything with a checkable factual claim always waits for a human.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from ..util import utcnow

STATUSES = ("pending_review", "approved", "published", "rejected")


@dataclass
class QueueItem:
    id: str
    topic: str
    title: str
    body: str
    byline: str
    status: str = "pending_review"
    created: str = ""
    published_at: str = ""
    clean: bool = False
    flags: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    originality: float = 1.0


def _parse(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


class PublishQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._items: list[QueueItem] = []
        if self.path.exists():
            try:
                self._items = [QueueItem(**d) for d in json.loads(self.path.read_text(encoding="utf-8"))]
            except Exception:
                self._items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(i) for i in self._items], indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- adding and reviewing ----------------------------------------------------------------------
    def has_topic(self, topic: str) -> bool:
        key = topic.strip().lower()
        return any(i.topic.strip().lower() == key and i.status != "rejected" for i in self._items)

    def add(self, c: Any) -> QueueItem:
        flags = list(c.claims_to_verify) + list(c.balance_flags) + list(c.headline_flags) + list(c.originality_flags) + list(getattr(c, "diversity_flags", []))
        item = QueueItem(id=uuid.uuid4().hex[:8], topic=c.topic, title=c.title, body=c.body, byline=c.byline,
                         created=utcnow().isoformat(timespec="seconds"), clean=bool(c.clean), flags=flags,
                         sources=list(c.sources), originality=float(c.originality))
        self._items.append(item)
        self.save()
        return item

    def get(self, item_id: str) -> Optional[QueueItem]:
        return next((i for i in self._items if i.id == item_id), None)

    def _set(self, item_id: str, status: str) -> bool:
        it = self.get(item_id)
        if it is None or it.status == "published":
            return False
        it.status = status
        self.save()
        return True

    def approve(self, item_id: str) -> bool:
        return self._set(item_id, "approved")

    def reject(self, item_id: str) -> bool:
        return self._set(item_id, "rejected")

    def items(self, status: Optional[str] = None) -> list[QueueItem]:
        return [i for i in self._items if status is None or i.status == status]

    # ---- scheduling ------------------------------------------------------------------------------
    def last_published_at(self) -> Optional[datetime]:
        times = [t for t in (_parse(i.published_at) for i in self._items if i.status == "published") if t]
        return max(times) if times else None

    def published_since(self, since: datetime) -> int:
        return sum(1 for i in self._items if i.status == "published" and (_parse(i.published_at) or since) >= since)

    def is_due(self, now: datetime, interval_minutes: int) -> bool:
        last = self.last_published_at()
        return last is None or now - last >= timedelta(minutes=max(1, interval_minutes))

    def next_publishable(self, auto_publish: bool) -> Optional[QueueItem]:
        approved = sorted(self.items("approved"), key=lambda i: i.created)
        if approved:
            return approved[0]
        if auto_publish:
            clean = sorted((i for i in self.items("pending_review") if i.clean), key=lambda i: i.created)
            if clean:
                return clean[0]
        return None

    def mark_published(self, item_id: str, at: Optional[datetime] = None) -> None:
        it = self.get(item_id)
        if it is not None:
            it.status = "published"
            it.published_at = (at or utcnow()).isoformat(timespec="seconds")
            self.save()

    def tick(self, now: datetime, interval_minutes: int, auto_publish: bool, max_per_day: int) -> Optional[QueueItem]:
        """Return the item to publish now, or None (not due, daily cap reached, or nothing publishable)."""
        if not self.is_due(now, interval_minutes):
            return None
        if self.published_since(now - timedelta(days=1)) >= max(1, max_per_day):
            return None
        return self.next_publishable(auto_publish)
