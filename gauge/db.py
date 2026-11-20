"""SQLite storage. Stdlib only, single file, safe to copy around on Windows."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .models import ChapterRecord, NovelRecord
from .util import utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS novels (
    key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    url TEXT,
    title TEXT NOT NULL,
    author TEXT,
    blurb TEXT,
    genres TEXT,
    tags TEXT,
    status TEXT,
    followers INTEGER, favorites INTEGER, views INTEGER, average_views INTEGER,
    rating REAL, rating_count INTEGER, comments INTEGER, chapter_count INTEGER, pages INTEGER,
    first_published TEXT, last_updated TEXT, fetched_at TEXT,
    user_rating REAL
);
CREATE TABLE IF NOT EXISTS chapters (
    novel_key TEXT NOT NULL,
    idx INTEGER NOT NULL,
    source_chapter_id TEXT,
    url TEXT,
    title TEXT,
    text TEXT NOT NULL,
    published_at TEXT,
    words INTEGER,
    views INTEGER,
    comments INTEGER,
    fetched_at TEXT,
    content_hash TEXT,
    PRIMARY KEY (novel_key, idx)
);
CREATE TABLE IF NOT EXISTS features (
    owner_key TEXT NOT NULL,
    level TEXT NOT NULL,
    values_json TEXT NOT NULL,
    computed_at TEXT,
    PRIMARY KEY (owner_key, level)
);
CREATE TABLE IF NOT EXISTS metric_history (
    novel_key TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    followers INTEGER, favorites INTEGER, views INTEGER, rating REAL, chapter_count INTEGER
);
CREATE TABLE IF NOT EXISTS follows (
    url TEXT PRIMARY KEY,
    source TEXT,
    added_at TEXT,
    last_checked TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    summary TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, project TEXT, chapter INTEGER, purpose TEXT, model TEXT,
    prompt_tokens INTEGER, output_tokens INTEGER, duration_ms INTEGER, ok INTEGER, retries INTEGER
);
CREATE TABLE IF NOT EXISTS generated (
    project TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    score REAL,
    features_json TEXT,
    created_at TEXT,
    PRIMARY KEY (project, chapter)
);
"""


def _dt(v: Optional[datetime]) -> Optional[str]:
    return v.isoformat() if v else None


def _parse_dt(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the first release to existing databases."""
        wanted = {"novels": {"user_rating": "REAL"}, "chapters": {"content_hash": "TEXT"}}
        for table, cols in wanted.items():
            have = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            for col, typ in cols.items():
                if col not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ---- novels ------------------------------------------------------------
    def upsert_novel(self, n: NovelRecord) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO novels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                  url=excluded.url, title=excluded.title, author=excluded.author,
                  blurb=excluded.blurb, genres=excluded.genres, tags=excluded.tags,
                  status=excluded.status, followers=excluded.followers,
                  favorites=excluded.favorites, views=excluded.views,
                  average_views=excluded.average_views, rating=excluded.rating,
                  rating_count=excluded.rating_count, comments=excluded.comments,
                  chapter_count=excluded.chapter_count, pages=excluded.pages,
                  first_published=COALESCE(excluded.first_published, novels.first_published),
                  last_updated=excluded.last_updated, fetched_at=excluded.fetched_at,
                  user_rating=COALESCE(excluded.user_rating, novels.user_rating)""",
                (
                    n.key, n.source, n.source_id, n.url, n.title, n.author, n.blurb,
                    json.dumps(n.genres), json.dumps(n.tags), n.status, n.followers, n.favorites,
                    n.views, n.average_views, n.rating, n.rating_count, n.comments,
                    n.chapter_count, n.pages, _dt(n.first_published), _dt(n.last_updated),
                    _dt(n.fetched_at), n.user_rating,
                ),
            )
            c.execute(
                "INSERT INTO metric_history VALUES (?,?,?,?,?,?,?)",
                (n.key, _dt(n.fetched_at), n.followers, n.favorites, n.views, n.rating,
                 n.chapter_count),
            )

    def _row_to_novel(self, r: sqlite3.Row) -> NovelRecord:
        return NovelRecord(
            source=r["source"], source_id=r["source_id"], url=r["url"] or "", title=r["title"],
            author=r["author"] or "", blurb=r["blurb"] or "",
            genres=json.loads(r["genres"] or "[]"), tags=json.loads(r["tags"] or "[]"),
            status=r["status"] or "", followers=r["followers"], favorites=r["favorites"],
            views=r["views"], average_views=r["average_views"], rating=r["rating"],
            rating_count=r["rating_count"], comments=r["comments"],
            chapter_count=r["chapter_count"], pages=r["pages"],
            first_published=_parse_dt(r["first_published"]),
            last_updated=_parse_dt(r["last_updated"]),
            fetched_at=_parse_dt(r["fetched_at"]) or utcnow(),
            user_rating=r["user_rating"],
        )

    def get_novel(self, key: str) -> Optional[NovelRecord]:
        r = self._conn.execute("SELECT * FROM novels WHERE key=?", (key,)).fetchone()
        return self._row_to_novel(r) if r else None

    def novels(self, source: Optional[str] = None) -> list[NovelRecord]:
        q, args = "SELECT * FROM novels", ()
        if source:
            q, args = q + " WHERE source=?", (source,)
        return [self._row_to_novel(r) for r in self._conn.execute(q + " ORDER BY title", args)]

    def metric_history(self, key: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM metric_history WHERE novel_key=? ORDER BY observed_at", (key,))
        return [dict(r) for r in rows]

    # ---- chapters ----------------------------------------------------------
    def upsert_chapter(self, ch: ChapterRecord) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO chapters VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(novel_key, idx) DO UPDATE SET
                  source_chapter_id=excluded.source_chapter_id, url=excluded.url,
                  title=excluded.title, text=excluded.text, published_at=excluded.published_at,
                  words=excluded.words, views=COALESCE(excluded.views, chapters.views),
                  comments=COALESCE(excluded.comments, chapters.comments),
                  fetched_at=excluded.fetched_at, content_hash=excluded.content_hash""",
                (ch.novel_key, ch.index, ch.source_chapter_id, ch.url, ch.title, ch.text,
                 _dt(ch.published_at), ch.words, ch.views, ch.comments, _dt(ch.fetched_at), ch.content_hash),
            )

    def chapters(self, novel_key: str) -> list[ChapterRecord]:
        rows = self._conn.execute(
            "SELECT * FROM chapters WHERE novel_key=? ORDER BY idx", (novel_key,))
        return [
            ChapterRecord(
                novel_key=r["novel_key"], index=r["idx"], source_chapter_id=r["source_chapter_id"] or "",
                url=r["url"] or "", title=r["title"] or "", text=r["text"],
                published_at=_parse_dt(r["published_at"]), words=r["words"] or 0,
                views=r["views"], comments=r["comments"], content_hash=r["content_hash"] or "",
                fetched_at=_parse_dt(r["fetched_at"]) or utcnow(),
            )
            for r in rows
        ]

    def chapter_count(self, novel_key: str) -> int:
        r = self._conn.execute("SELECT COUNT(*) FROM chapters WHERE novel_key=?", (novel_key,)).fetchone()
        return int(r[0])

    def known_chapter_ids(self, novel_key: str) -> set:
        rows = self._conn.execute(
            "SELECT source_chapter_id FROM chapters WHERE novel_key=?", (novel_key,))
        return {r[0] for r in rows if r[0]}

    def chapter_hashes(self, novel_key: str) -> dict[int, str]:
        rows = self._conn.execute("SELECT idx, content_hash FROM chapters WHERE novel_key=?", (novel_key,))
        return {int(r[0]): (r[1] or "") for r in rows}

    def delete_chapters_from(self, novel_key: str, first_idx: int) -> int:
        """Remove chapters with idx >= first_idx (a local novel shrank). Also drops their features."""
        with self.tx() as c:
            cur = c.execute("DELETE FROM chapters WHERE novel_key=? AND idx>=?", (novel_key, first_idx))
            c.execute("DELETE FROM features WHERE level='chapter' AND owner_key LIKE ? AND "
                      "CAST(substr(owner_key, length(?) + 2) AS INTEGER) >= ?",
                      (f"{novel_key}#%", novel_key, first_idx))
        return cur.rowcount

    def set_user_rating(self, novel_key: str, stars: Optional[float]) -> bool:
        with self.tx() as c:
            cur = c.execute("UPDATE novels SET user_rating=? WHERE key=?", (stars, novel_key))
        return cur.rowcount > 0

    def total_chapters(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0])

    # ---- features ----------------------------------------------------------
    def save_features(self, owner_key: str, level: str, values: dict[str, float]) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO features VALUES (?,?,?,?)",
                (owner_key, level, json.dumps(values), utcnow().isoformat()),
            )

    def features(self, level: str) -> dict[str, dict[str, float]]:
        rows = self._conn.execute("SELECT owner_key, values_json FROM features WHERE level=?", (level,))
        return {r[0]: json.loads(r[1]) for r in rows}

    def delete_feature(self, owner_key: str, level: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM features WHERE owner_key=? AND level=?", (owner_key, level))

    def feature_for(self, owner_key: str, level: str) -> Optional[dict[str, float]]:
        r = self._conn.execute(
            "SELECT values_json FROM features WHERE owner_key=? AND level=?", (owner_key, level)).fetchone()
        return json.loads(r[0]) if r else None

    # ---- follows -----------------------------------------------------------
    def follow(self, url: str, source: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO follows VALUES (?,?,?,NULL)",
                      (url, source, utcnow().isoformat()))

    def unfollow(self, url: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM follows WHERE url=?", (url,))

    def follows(self) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self._conn.execute("SELECT url, source FROM follows ORDER BY added_at")]

    def touch_follow(self, url: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE follows SET last_checked=? WHERE url=?", (utcnow().isoformat(), url))

    # ---- runs / generated --------------------------------------------------
    def log_run(self, kind: str, started: datetime, summary: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO runs (kind, started_at, finished_at, summary) VALUES (?,?,?,?)",
                      (kind, started.isoformat(), utcnow().isoformat(), json.dumps(summary)))

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["summary"] = json.loads(d["summary"] or "{}")
            out.append(d)
        return out

    # ---- llm telemetry -----------------------------------------------------
    def log_call(self, purpose: str, model: str, prompt_tokens: int, output_tokens: int, duration_ms: int, ok: bool,
                 retries: int = 0, project: str = "", chapter: Optional[int] = None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO llm_calls (ts, project, chapter, purpose, model, prompt_tokens, output_tokens, "
                      "duration_ms, ok, retries) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (utcnow().isoformat(), project, chapter, purpose, model, prompt_tokens, output_tokens, duration_ms,
                       1 if ok else 0, retries))

    def call_summary(self, project: Optional[str] = None, limit_days: Optional[int] = None) -> list[dict[str, Any]]:
        """Totals by purpose: calls, seconds, tokens, failures, average seconds per call."""
        q = ("SELECT purpose, model, COUNT(*) AS calls, SUM(duration_ms)/1000.0 AS seconds, SUM(prompt_tokens) AS prompt_tokens, "
             "SUM(output_tokens) AS output_tokens, SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) AS failed, SUM(retries) AS retries "
             "FROM llm_calls")
        where, args = [], []
        if project:
            where.append("project=?")
            args.append(project)
        if limit_days:
            where.append("ts >= datetime('now', ?)")
            args.append(f"-{int(limit_days)} days")
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " GROUP BY purpose, model ORDER BY seconds DESC"
        out = []
        for r in self._conn.execute(q, args):
            d = dict(r)
            d["avg_seconds"] = round(d["seconds"] / max(1, d["calls"]), 1)
            d["seconds"] = round(d["seconds"], 1)
            out.append(d)
        return out

    def record_generated(self, project: str, chapter: int, score: float, feats: dict[str, float]) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO generated VALUES (?,?,?,?,?)",
                      (project, chapter, score, json.dumps(feats), utcnow().isoformat()))

    def generated(self, project: str) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM generated WHERE project=? ORDER BY chapter", (project,))
        return [dict(r) for r in rows]
