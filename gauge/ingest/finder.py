"""The book finder: Gauge goes online and finds novels for its library by itself.

It decides what to look for from what you are writing (the genres of your stories, the books you rated
highly, any topics you name), searches a catalogue, chooses novels it does not already have, downloads
them, splits them into chapters and adds them to the library, where learning and GaugeLM's training
pick them up.

Where it looks, and why only there. A model may only be trained on text you have the right to use, so
the finder searches **Project Gutenberg**, asking only for books that are out of copyright: through the
Gutendex catalogue service when it answers, otherwise through the catalogue file Project Gutenberg
publishes for programs. A book whose own first page says it is still in copyright is dropped. It cannot be pointed at a web-novel site or any other address: the hosts it may
talk to are fixed in `ALLOWED_HOSTS` and every request is checked against them. Sites that host other
people's copyrighted novels are not something it can reach.

Manners and limits:

* a few books per run (`library.finder_books`), not more often than every `library.finder_hours`.
  Project Gutenberg asks that programs do not crawl its site; a handful of spaced-out downloads a day is
  what a person would do, and anyone wanting hundreds of books should use its mirrors instead;
* the site's robots.txt is obeyed and requests are spaced out (the shared `Fetcher`);
* a ceiling on how much text it may add in total (`library.finder_max_mb`);
* nothing about you is sent: a search is a genre word such as "adventure";
* everything it adds, and why, is written to `finder_log.jsonl`; dry-run and offline mode stop it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

from ..models import ChapterRecord, NovelRecord

log = logging.getLogger(__name__)

ALLOWED_HOSTS = {"gutendex.com", "www.gutenberg.org", "gutenberg.org"}
CATALOGUE = "https://gutendex.com/books"
# Project Gutenberg's own catalogue file, which it publishes so that programs do not have to crawl the site.
# Used when the catalogue service does not answer: fetched at most once a month and searched on this computer.
OFFLINE_CATALOGUE = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
OFFLINE_FILE, OFFLINE_DAYS, PAGE = "pg_catalog.csv", 30, 32
TEXT_LINK = "https://www.gutenberg.org/ebooks/{id}.txt.utf-8"
STATE, JOURNAL = "finder_state.json", "finder_log.jsonl"
LICENCE = "public domain in the US (Project Gutenberg); check your own country"
MIN_WORDS = 20_000           # shorter than this is a story or a pamphlet, not a novel

# What to ask the catalogue for, from the words Gauge uses for genres.
GENRE_TOPICS: dict[str, list[str]] = {
    "adventure": ["adventure"], "action": ["adventure"], "fantasy": ["fantasy", "fairy tales"],
    "sci-fi": ["science fiction"], "science fiction": ["science fiction"], "scifi": ["science fiction"],
    "mystery": ["detective", "mystery"], "crime": ["crime", "detective"], "thriller": ["detective", "spy stories"],
    "horror": ["horror", "gothic fiction", "ghost stories"], "gothic": ["gothic fiction"],
    "romance": ["love stories"], "historical": ["historical fiction"], "war": ["war stories"],
    "western": ["western stories"], "comedy": ["humor"], "humor": ["humor"], "sea": ["sea stories"],
    "drama": ["domestic fiction"], "slice of life": ["domestic fiction"], "children": ["children's stories"],
    "litrpg": ["adventure", "fantasy"], "progression": ["adventure"], "isekai": ["fantasy"], "psychological": ["psychological fiction"],
}
DEFAULT_TOPICS = ["adventure", "science fiction", "detective", "fantasy", "gothic fiction", "love stories",
                  "historical fiction", "sea stories", "humor", "western stories"]
_FICTION = re.compile(r"\b(fiction|stories|novels?|tales|romances?|adventure|mystery|fantasy)\b", re.I)
_NOT_NOVELS = re.compile(r"\b(poetry|poems|drama|plays|periodicals|essays|sermons|dictionar|biograph|history and criticism|"
                         r"correspondence|cookery|juvenile literature -- periodicals|index)\b", re.I)


def allowed(url: str) -> bool:
    """Only the catalogue and Project Gutenberg itself, over https."""
    p = urlparse(url)
    return p.scheme == "https" and (p.hostname or "").lower() in ALLOWED_HOSTS


def topics_for(genres: list[str]) -> list[str]:
    out: list[str] = []
    for g in genres:
        low = str(g).strip().lower()
        for key, topics in GENRE_TOPICS.items():
            if key == low or key in low:
                out.extend(topics)
    return list(dict.fromkeys(out))


def is_novel(book: dict[str, Any]) -> bool:
    """A work of prose fiction, going by how the catalogue files it."""
    filed = " | ".join(list(book.get("subjects", [])) + list(book.get("bookshelves", [])))
    if not _FICTION.search(filed):
        return False
    return not _NOT_NOVELS.search(filed)


def parse_catalogue(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The catalogue's answer, reduced to what the finder needs. Books still in copyright are dropped here
    as well as in the request, so a mistake on either side cannot let one through."""
    books = []
    for b in payload.get("results", []) or []:
        if b.get("copyright") is not False or b.get("media_type", "Text") != "Text":
            continue
        fmts = b.get("formats", {}) or {}
        text_url = next((u for k, u in fmts.items() if k.startswith("text/plain") and allowed(str(u))), None)
        if not text_url or not b.get("id"):
            continue
        books.append({"id": int(b["id"]), "title": str(b.get("title", "")).strip() or "Untitled",
                      "authors": [str(a.get("name", "")) for a in b.get("authors", []) or []],
                      "subjects": [str(s) for s in b.get("subjects", []) or []],
                      "bookshelves": [str(s) for s in b.get("bookshelves", []) or []],
                      "languages": list(b.get("languages", []) or []), "downloads": int(b.get("download_count", 0) or 0),
                      "text_url": str(text_url)})
    return books


def parse_offline_catalogue(text: str, languages: list[str]) -> list[dict[str, Any]]:
    """Project Gutenberg's catalogue file (columns Text#, Type, Issued, Title, Language, Authors, Subjects,
    LoCC, Bookshelves) as the same records the catalogue service gives. It has no download counts, so
    books are ordered by number: the lowest numbers are the classics that were added first."""
    import csv
    import io

    want = {x.lower() for x in languages} or {"en"}
    books = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            num = int(str(row.get("Text#", "")).strip())
        except ValueError:
            continue
        if (row.get("Type") or "").strip() != "Text":
            continue
        langs = [x.strip().lower() for x in re.split(r"[;,]", row.get("Language") or "") if x.strip()]
        if not want & set(langs):
            continue
        split = lambda v: [x.strip() for x in (v or "").split(";") if x.strip()]  # noqa: E731
        books.append({"id": num, "title": " ".join((row.get("Title") or "Untitled").split()),
                      "authors": [re.sub(r",?\s*\d{3,4}\??-\d{0,4}\??$", "", a).strip() for a in split(row.get("Authors"))],
                      "subjects": split(row.get("Subjects")), "bookshelves": split(row.get("Bookshelves")),
                      "languages": langs, "downloads": 0, "text_url": TEXT_LINK.format(id=num)})
    books.sort(key=lambda b: b["id"])
    return books


def says_copyrighted(raw: str) -> bool:
    """A few Project Gutenberg books are still in copyright and say so at the top. Those are never kept."""
    head = raw[:6000]
    return bool(re.search(r"COPYRIGHTED\s+Project Gutenberg|This is a COPYRIGHTED|copyrighted work\b.*Project Gutenberg", head, re.I | re.S))


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower().split(";")[0].split(":")[0]).strip()


class BookFinder:
    def __init__(self, cfg: Any, db: Any, fetcher: Any = None):
        self.cfg = cfg
        self.db = db
        self._fetcher = fetcher
        self.dir = Path(cfg.models_dir)
        self._service_down = False
        self._books: Optional[list[dict[str, Any]]] = None

    # ---- state and journal ---------------------------------------------------------
    def state(self) -> dict[str, Any]:
        f = self.dir / STATE
        try:
            d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except (OSError, ValueError):
            d = {}
        d.setdefault("runs", 0)
        d.setdefault("pages", {})           # topic -> the next catalogue page to read
        d.setdefault("passed_over", [])     # catalogue ids looked at and not wanted, so they are not fetched again
        d.setdefault("chars_added", 0)
        d.setdefault("last_run", 0.0)
        return d

    def _save(self, state: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state["passed_over"] = state["passed_over"][-5000:]
        (self.dir / STATE).write_text(json.dumps(state, indent=1), encoding="utf-8")

    def journal(self, last: int = 50) -> list[dict[str, Any]]:
        f = self.dir / JOURNAL
        if not f.exists():
            return []
        rows = []
        for line in f.read_text(encoding="utf-8").splitlines()[-last:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _note(self, row: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / JOURNAL).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- what to look for ------------------------------------------------------------
    def interests(self) -> list[str]:
        """Catalogue topics worth searching, most relevant first: what you named, what you write, what you
        rated highly, then a general spread of fiction so the library never stops growing."""
        from ..write.project import Project

        genres: list[str] = []
        try:
            for name in Project.list(Path(self.cfg.projects_dir)):
                genres.extend(Project(Path(self.cfg.projects_dir), name).state.genres)
        except Exception as e:
            log.warning("finder: could not read your stories' genres: %s", e)
        try:
            for n in self.db.novels():
                if (n.user_rating or 0) >= 4:
                    genres.extend(list(n.genres) + list(n.tags))
        except Exception as e:
            log.warning("finder: could not read your ratings: %s", e)
        named = [t.strip().lower() for t in self.cfg.library.finder_topics if t.strip()]
        return list(dict.fromkeys(named + topics_for(genres) + DEFAULT_TOPICS))

    def due(self) -> bool:
        return time.time() - float(self.state()["last_run"]) >= float(self.cfg.library.finder_hours) * 3600

    def _have(self) -> tuple[set[str], set[str]]:
        keys, titles = set(), set()
        for n in self.db.novels():
            keys.add(n.key)
            titles.add(_norm_title(n.title))
        return keys, titles

    # ---- the network, on a short lead --------------------------------------------------
    def _get(self, url: str) -> Optional[str]:
        if not allowed(url):
            log.warning("finder: refusing %s (not the catalogue or Project Gutenberg)", url)
            return None
        if self._fetcher is None:
            from .http import Fetcher

            self._fetcher = Fetcher(self.cfg.ingest)
        return self._fetcher.get(url)

    def search(self, topic: str, page: int = 1) -> list[dict[str, Any]]:
        """One page of out-of-copyright books on a topic. The catalogue service is asked first (it knows
        which books are most read); when it does not answer, Project Gutenberg's own catalogue file is used."""
        q = urlencode({"topic": topic, "languages": ",".join(self.cfg.library.finder_languages or ["en"]),
                       "copyright": "false", "sort": "popular", "page": page})
        raw = None if self._service_down else self._get(f"{CATALOGUE}?{q}")
        if raw:
            try:
                return parse_catalogue(json.loads(raw))
            except (ValueError, TypeError, AttributeError):
                pass
        if not self._service_down and page > 1:
            return []                            # past the last page of a topic: the service is fine, the topic is used up
        self._service_down = True                # do not keep asking a service that is not answering this run
        books = self._offline()
        low = topic.lower()
        hits = [b for b in books if low in " | ".join(b["subjects"] + b["bookshelves"]).lower()]
        return hits[(page - 1) * PAGE:page * PAGE]

    def _offline(self) -> list[dict[str, Any]]:
        """Project Gutenberg's catalogue file, kept on this computer and refreshed at most once a month."""
        if self._books is not None:
            return self._books
        f = self.dir / OFFLINE_FILE
        fresh = f.exists() and time.time() - f.stat().st_mtime < OFFLINE_DAYS * 86400
        if not fresh:
            raw = self._get(OFFLINE_CATALOGUE)
            if raw and raw.lstrip().startswith("Text#"):
                self.dir.mkdir(parents=True, exist_ok=True)
                f.write_text(raw, encoding="utf-8")
        try:
            text = f.read_text(encoding="utf-8") if f.exists() else ""
        except OSError:
            text = ""
        self._books = parse_offline_catalogue(text, list(self.cfg.library.finder_languages or ["en"])) if text else []
        return self._books

    def fetch(self, book: dict[str, Any]) -> Optional[list[tuple[str, str]]]:
        """The book's text as chapters, or None if it cannot be had or is not a novel-length book."""
        from ..research.sources import strip_gutenberg_boilerplate
        from .local import split_single_file

        raw = self._get(book["text_url"])
        if not raw:
            return None
        cap = int(float(self.cfg.library.finder_max_book_mb) * 1_000_000)
        if len(raw) > cap or says_copyrighted(raw):
            return None
        text = strip_gutenberg_boilerplate(raw)
        if len(text.split()) < MIN_WORDS:
            return None
        chapters = [(t, b) for t, b in split_single_file(text) if len(b.split()) >= 100]
        return chapters[: int(self.cfg.ingest.max_chapters_per_novel)] or None

    def store(self, book: dict[str, Any], chapters: list[tuple[str, str]], topic: str) -> NovelRecord:
        novel = NovelRecord(source="gutenberg", source_id=str(book["id"]), url=f"https://www.gutenberg.org/ebooks/{book['id']}",
                            title=book["title"], author=", ".join(book["authors"]), genres=[topic],
                            tags=[s.split(" -- ")[0] for s in book["subjects"][:8]], status="completed",
                            chapter_count=len(chapters))
        self.db.upsert_novel(novel)
        for i, (title, body) in enumerate(chapters, 1):
            self.db.upsert_chapter(ChapterRecord(novel_key=novel.key, index=i, source_chapter_id=str(i), title=title, text=body))
        return novel

    # ---- a run ---------------------------------------------------------------------------
    def run(self, books: Optional[int] = None, topics: Optional[list[str]] = None, plan_only: bool = False,
            should_stop: Optional[Callable[[], bool]] = None,
            progress: Optional[Callable[[dict[str, Any]], None]] = None) -> dict[str, Any]:
        """Find and add up to `books` novels. Returns what was added and what was looked at."""
        from .. import dryrun

        want = int(books or self.cfg.library.finder_books)
        search_topics = [t.strip().lower() for t in (topics or []) if t.strip()] or self.interests()
        base = {"added": [], "looked_at": 0, "passed_over": 0, "topics": []}
        if os.environ.get("GAUGE_OFFLINE"):
            return {**base, "message": "Gauge is in offline mode, so it did not look for novels."}
        state = self.state()
        budget = int(float(self.cfg.library.finder_max_mb) * 1_000_000) - int(state["chars_added"])
        if budget <= 0:
            return {**base, "message": f"The finder has added its limit of {self.cfg.library.finder_max_mb:g} MB of books. "
                                       "Raise the limit in Switches to let it add more."}
        # Start with a different topic each run, so the library grows in every direction you care about.
        start = state["runs"] % len(search_topics)
        order = search_topics[start:] + search_topics[:start]
        if plan_only or dryrun.is_dry_run():
            if dryrun.is_dry_run():
                dryrun.hold("find novels online", ", ".join(order[:3]))
            return {**base, "topics": order[:3],
                    "message": f"Would search Project Gutenberg for up to {want} novels on: {', '.join(order[:3])}. Nothing was downloaded."}
        stop = should_stop or (lambda: False)
        have_keys, have_titles = self._have()
        passed = set(state["passed_over"])
        added: list[dict[str, Any]] = []
        looked = skipped = 0
        used_topics: list[str] = []
        for topic in order:
            if len(added) >= want or stop() or budget <= 0:
                break
            page = int(state["pages"].get(topic, 1))
            found = self.search(topic, page)
            used_topics.append(topic)
            if not found:
                state["pages"][topic] = 1            # past the last page, or the catalogue did not answer: start over next time
                continue
            fresh = 0
            for book in sorted(found, key=lambda b: -b["downloads"]):
                if len(added) >= want or stop() or budget <= 0:
                    break
                key = f"gutenberg:{book['id']}"
                if key in have_keys or _norm_title(book["title"]) in have_titles or book["id"] in passed:
                    continue
                looked += 1
                fresh += 1
                if not is_novel(book):
                    passed.add(book["id"])
                    skipped += 1
                    continue
                if progress:
                    progress({"reading": book["title"], "added": len(added)})
                chapters = self.fetch(book)
                if not chapters:
                    passed.add(book["id"])
                    skipped += 1
                    continue
                chars = sum(len(b) for _, b in chapters)
                if chars > budget:
                    skipped += 1
                    continue
                novel = self.store(book, chapters, topic)
                have_keys.add(key)
                have_titles.add(_norm_title(book["title"]))
                budget -= chars
                state["chars_added"] = int(state["chars_added"]) + chars
                row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "title": book["title"], "author": ", ".join(book["authors"]),
                       "id": book["id"], "url": novel.url, "topic": topic, "chapters": len(chapters),
                       "words": sum(len(b.split()) for _, b in chapters), "licence": LICENCE,
                       "why": f"a much-read {topic} novel you did not have ({book['downloads']:,} downloads)"}
                self._note(row)
                added.append(row)
            if fresh == 0 or len(added) < want:
                state["pages"][topic] = page + 1     # this page is used up: read the next one next time
        state["runs"] += 1
        state["last_run"] = time.time()
        state["passed_over"] = sorted(passed)
        self._save(state)
        if added:
            names = "; ".join(f"{a['title']} by {a['author'] or 'unknown'}" for a in added[:4]) + ("; ..." if len(added) > 4 else "")
            msg = (f"Found and added {len(added)} novel{'' if len(added) == 1 else 's'} from Project Gutenberg: {names}. "
                   "They are out of copyright. Gauge will learn from them next time it studies or trains.")
        elif not used_topics or looked == 0:
            msg = "Looked for novels online but found nothing new. The catalogue may not be answering; it will try again later."
        else:
            msg = f"Looked at {looked} book{'' if looked == 1 else 's'} online and none was a novel worth adding this time."
        return {"added": added, "looked_at": looked, "passed_over": skipped, "topics": used_topics, "message": msg}

    def run_if_due(self, **kw: Any) -> Optional[dict[str, Any]]:
        """What 'by itself' means: called at the start of a training session and by the daemon. Does
        nothing unless the switch is on and enough time has passed."""
        if not self.cfg.library.finder or not self.due():
            return None
        try:
            return self.run(**kw)
        except Exception as e:  # finding books is a bonus; it must never stop the work it was called from
            log.warning("finder: %s", e)
            return None
