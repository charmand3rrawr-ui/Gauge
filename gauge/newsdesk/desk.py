"""One newsdesk cycle: discover -> research -> translate -> write original commentary -> queue -> publish due.

The pieces are deliberately separable (each is testable on its own); this module just runs them in order
with the settings from `newsdesk` in config.json. Publishing means adding the piece to the site project's
articles, so the next `gauge site build` (or your existing deploy) carries it.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..util import utcnow
from .commentary import write_commentary
from .queue import PublishQueue, QueueItem
from .translate import detect_script, translate
from .trends import SourceItem, Topic, discover, rss_items


@dataclass
class CycleReport:
    topics_found: int = 0
    drafted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    published: Optional[str] = None
    waiting_review: int = 0
    notes: list[str] = field(default_factory=list)


def make_fetcher(cfg: Any) -> Any:
    from ..config import IngestConfig
    from ..ingest.http import Fetcher
    return Fetcher(IngestConfig(user_agent=cfg.ingest.user_agent + " (newsdesk)",
                                request_delay_seconds=max(1.0, cfg.research.request_delay_seconds),
                                respect_robots_txt=True))  # the newsdesk always honours robots.txt


def queue_for(cfg: Any) -> PublishQueue:
    return PublishQueue(cfg.root / "newsdesk" / "queue.json")


def gather_sources(fetcher: Any, llm: Any, topic: str, feeds: list[str], target: str, limit: int = 6) -> list[SourceItem]:
    """Feed items whose headline or summary mentions the topic, translated for research."""
    words = {w for w in re.findall(r"\w+", topic.lower()) if len(w) > 3}
    found: list[SourceItem] = []
    for feed in feeds:
        for item in rss_items(fetcher, feed, 30):
            text = f"{item.title} {item.summary}".lower()
            if words and any(w in text for w in words):
                if detect_script(item.title + item.summary) != "latin" or (item.lang and item.lang not in ("en", "")):
                    item.translated = translate(llm, f"{item.title}\n{item.summary}", target, item.lang)
                found.append(item)
            if len(found) >= limit:
                return found
    return found


def publish_item(item: QueueItem, articles_dir: Path) -> Path:
    """Write the piece into the site project's articles as labelled opinion, ready for `site build`."""
    articles_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", item.title.lower()).strip("-")[:60] or item.id
    path = articles_dir / f"{slug}.json"
    path.write_text(json.dumps({"slug": slug, "title": item.title, "body": item.body, "byline": item.byline,
                                "kind": "opinion", "summary": f"Commentary on {item.topic}",
                                "published_at": utcnow().date().isoformat(), "thumbnail": ""}, indent=2,
                               ensure_ascii=False), encoding="utf-8")
    return path


def run_cycle(cfg: Any, llm: Any, fetcher: Any = None, topics: Optional[list[Topic]] = None,
              now: Optional[datetime] = None, stance: Any = None, draft: bool = True) -> CycleReport:
    nd = cfg.newsdesk
    rep = CycleReport()
    q = queue_for(cfg)
    if not nd.byline:
        rep.notes.append("no byline set (newsdesk.byline); nothing drafted — publish under your own name or owned pen name")
        draft = False
    if draft:
        if topics is None:
            if os.environ.get("GAUGE_OFFLINE"):
                topics = []
                rep.notes.append("offline: trend discovery skipped")
            else:
                fetcher = fetcher or make_fetcher(cfg)
                topics = discover(fetcher, nd.languages, nd.feeds)
        rep.topics_found = len(topics)
        if stance is None:
            from ..site.stance import load_stance
            stance = load_stance(cfg.root / "stance.json")
        for t in topics:
            if len(rep.drafted) >= nd.drafts_per_cycle:
                break
            if q.has_topic(t.title):
                rep.skipped.append(t.title)
                continue
            sources = gather_sources(fetcher, llm, t.title, nd.feeds, nd.target_language) if (fetcher and nd.feeds) else []
            if t.url and not any(s.url == t.url for s in sources):
                sources.append(SourceItem(title=f"{t.title} (Wikipedia, {t.lang})", summary="", url=t.url, lang=t.lang))
            try:
                c = write_commentary(llm, t.title, sources, nd.byline, stance=stance, voice=nd.voice,
                                     comedic=nd.comedic, max_overlap=nd.max_source_overlap)
            except PermissionError as e:
                rep.notes.append(str(e))
                break
            q.add(c)
            rep.drafted.append(c.title)
    item = q.tick(now or utcnow(), nd.interval_minutes, nd.auto_publish, nd.max_per_day)
    if item is not None:
        if nd.project:
            from ..write.project import Project
            if Project.exists(cfg.projects_dir, nd.project):
                publish_item(item, Project(cfg.projects_dir, nd.project).dir / "articles")
                q.mark_published(item.id, now)
                rep.published = item.title
            else:
                rep.notes.append(f"newsdesk.project {nd.project!r} does not exist; nothing published")
        else:
            rep.notes.append("newsdesk.project is not set; approved pieces wait until you name the site project")
    rep.waiting_review = len(q.items("pending_review"))
    return rep
