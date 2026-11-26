"""The chronicle: a story's memory that still holds at chapter 10,000.

No model remembers a long serial. A model sees only what is put in front of it, and what fits in front
of it is a few thousand words. Until now what was put there was one summary of about 250 words,
rewritten after every chapter (so a detail from chapter 40 has been rewritten away four hundred times
by chapter 440), the last dozen chapters for repetition checks, and a list of facts that kept only the
latest 2,000. That is a memory of about a hundred and fifty chapters.

The chronicle is a small database kept beside the story (`chronicle.db`) that forgets nothing and
hands the writer, for each new chapter, only the part that matters to it:

* **A digest of every chapter**, written once and never rewritten.
* **Roll-ups in layers.** Every 10 chapters are summarised into an arc, every 10 arcs into a volume,
  every 10 volumes into a saga. "The story so far" for chapter 9,997 is the nine finished sagas, the
  finished volumes of the current saga, the finished arcs of the current volume and the chapters of
  the current arc: about thirty-six summaries however long the story is, trimmed to the room there is.
* **Every fact, with its history.** Who is whose sister, what colour the door is, who is dead. A fact
  that is meant to last (a name, a parent, a death) contradicts a different later value; a fact that is
  meant to change (where someone is, what they carry) is replaced, and the old value is kept as history.
* **Who appeared where.** So when someone walks back in after four thousand chapters, the writer is
  told when they were last seen, what happened then, and everything known about them.

For each chapter `brief()` gathers, within a size limit: the facts about the people and places in the
plan, reminders for anyone returning after a long absence, the past chapters most like this one, and
any unresolved contradictions. After each chapter `record()` files the digest, and `check()` reports
anything in it that contradicts what was established, however long ago.

Everything is indexed, so the cost of a chapter does not grow with the length of the story.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from .names import proper_names

log = logging.getLogger(__name__)

DB = "chronicle.db"
SPAN = 10                                   # each layer summarises ten of the layer below
LEVELS = {1: "arc", 2: "volume", 3: "saga", 4: "age"}       # 10, 100, 1,000, 10,000 chapters
LONG_ABSENCE = 50                           # chapters away after which a returning name gets a reminder

# Attributes that are meant to last: a different value later is a contradiction, not a development.
_LASTING = re.compile(r"\b(name|born|birth|parent|mother|father|sister|brother|sibling|child|son|daughter|species|race|"
                      r"eye|hair|height|scar|handed|origin|homeland|native|blood|gender|sex|dead|death|died|killed|fate|"
                      r"real name|true name|age at|founded|built|rule|law|price of|cost of)\b", re.I)
_DEAD = re.compile(r"\b(dead|died|dies|killed|slain|murdered|executed|perished|deceased|drowned|did not survive)\b", re.I)
_ALIVE_AGAIN = re.compile(r"\b(alive|survived|revived|resurrected|returned from the dead|not dead|faked)\b", re.I)
_ACTS = (r"(?:said|says|asked|asks|replied|answered|shouted|whispered|laughed|smiled|nodded|shrugged|walked|ran|stood|sat|"
         r"turned|looked|reached|grabbed|drew|opened|stepped|came|went|took|gave|frowned|sighed|muttered|grinned)")
_MEMORY = re.compile(r"\b(remember(?:ed|s)?|memory|memories|ghost|grave|tomb|funeral|the late|portrait|dream(?:ed|t)?|"
                     r"used to|years ago|before (?:he|she|they) died|flashback|statue|letter from)\b", re.I)


_TITLES = {"the", "old", "young", "captain", "lord", "lady", "sir", "mr", "mrs", "miss", "dr", "master", "uncle", "aunt",
           "brother", "sister", "father", "mother", "king", "queen", "prince", "princess", "general", "doctor", "saint"}


def _name_words(name: str) -> list[str]:
    """The words of a name a text would use for the person: 'Old Nan' is called Nan, 'Captain Hale' is called Hale."""
    words = [w for w in name.split() if len(w) >= 3 and w.lower() not in _TITLES]
    return words or name.split()[:1] or [name]


def _norm(s: Any) -> str:
    return " ".join(re.split(r"[^a-z0-9]+", str(s).lower())).strip()


def _trim(text: str, words: int) -> str:
    ws = str(text or "").split()
    return " ".join(ws) if len(ws) <= words else " ".join(ws[:words]).rstrip(",;:") + "..."


_STATE_OF_LIFE = re.compile(r"\b(status|state|fate|condition|alive|life|living|death|dead|died|killed|end)\b", re.I)


def says_dead(attribute: str, value: str) -> bool:
    """A fact that says someone has died. The attribute has to be about whether they live ('status: dead',
    'fate: killed at the wall'); a place called the Drowned Quarter is not a death."""
    return bool(_STATE_OF_LIFE.search(attribute) and _DEAD.search(f"{attribute} {value}"))


def lasting(attribute: str, value: str = "", flag: Any = None) -> bool:
    """Is this the kind of fact that should never change? The digest may say; otherwise the wording decides."""
    if isinstance(flag, bool):
        return flag
    return bool(_LASTING.search(attribute) or says_dead(attribute, value))


class Chronicle:
    _open: dict[str, Chronicle] = {}
    _guard = threading.Lock()

    def __init__(self, project_dir: Path):
        self.dir = Path(project_dir)
        self.path = self.dir / DB
        self._lock = threading.RLock()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.fts = True
        self._setup()

    @classmethod
    def at(cls, project_dir: Path) -> Chronicle:
        """The chronicle for a story, opened once per program."""
        key = str(Path(project_dir).resolve())
        with cls._guard:
            c = cls._open.get(key)
            if c is None or not c.path.exists():
                c = cls._open[key] = cls(Path(project_dir))
            return c

    def close(self) -> None:
        with self._lock:
            self.db.close()
        with self._guard:
            self._open.pop(str(self.dir.resolve()), None)

    def _setup(self) -> None:
        with self._lock, self.db:
            self.db.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS chapters(n INTEGER PRIMARY KEY, title TEXT, summary TEXT, words INTEGER);
                CREATE TABLE IF NOT EXISTS rollups(level INTEGER, idx INTEGER, first INTEGER, last INTEGER, summary TEXT,
                                                   PRIMARY KEY(level, idx));
                CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, attribute TEXT, value TEXT,
                                                 lasting INTEGER, chapter INTEGER, quote TEXT, ns TEXT, na TEXT, nv TEXT,
                                                 replaced_at INTEGER);
                CREATE INDEX IF NOT EXISTS facts_subject ON facts(ns, na);
                CREATE TABLE IF NOT EXISTS mentions(name TEXT, chapter INTEGER, times INTEGER, PRIMARY KEY(name, chapter));
                CREATE INDEX IF NOT EXISTS mentions_chapter ON mentions(chapter);
                CREATE TABLE IF NOT EXISTS people(name TEXT PRIMARY KEY, first INTEGER, last INTEGER, chapters INTEGER, times INTEGER);
                CREATE TABLE IF NOT EXISTS conflicts(id INTEGER PRIMARY KEY AUTOINCREMENT, chapter INTEGER, subject TEXT,
                                                     attribute TEXT, was TEXT, was_chapter INTEGER, now TEXT, quote TEXT,
                                                     settled INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS memories(person TEXT, chapter INTEGER, events TEXT, PRIMARY KEY(person, chapter));
            """)
            try:
                self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(summary, content='')")
            except sqlite3.OperationalError:          # this Python's SQLite has no full-text search: matching is by name only
                self.fts = False

    # ---- recording ---------------------------------------------------------------------
    def record(self, n: int, title: str, text: str, summary: str = "", facts: Optional[list[dict[str, Any]]] = None,
               known_names: Optional[set[str]] = None, summarise: Optional[Callable[[list[str], int, int], str]] = None,
               memories: Optional[dict[str, list[str]]] = None) -> dict[str, Any]:
        """File one finished chapter: its digest, its facts, who was in it. Returns the contradictions found
        against everything established before, and any roll-ups that became due. Recording the same
        chapter again replaces what was filed for it."""
        summary = _trim(summary, 90) or _trim(" ".join(text.split()[:40]) + " ... " + " ".join(text.split()[-40:]), 90)
        with self._lock, self.db:
            again = self.db.execute("SELECT 1 FROM chapters WHERE n=?", (n,)).fetchone() is not None
            if again:
                self._forget(n)
            self.db.execute("INSERT OR REPLACE INTO chapters(n, title, summary, words) VALUES(?,?,?,?)",
                            (n, title, summary, len(text.split())))
            if self.fts:
                self.db.execute("INSERT INTO search(rowid, summary) VALUES(?,?)", (n, f"{title}. {summary}"))
            names = proper_names(text, known_names or set())
            for name, times in names.items():
                self.db.execute("INSERT OR REPLACE INTO mentions(name, chapter, times) VALUES(?,?,?)", (name, n, int(times)))
                self.db.execute("""INSERT INTO people(name, first, last, chapters, times) VALUES(?,?,?,1,?)
                                   ON CONFLICT(name) DO UPDATE SET last=MAX(last, excluded.last), first=MIN(first, excluded.first),
                                   chapters=chapters+1, times=times+excluded.times""", (name, n, n, int(times)))
            for person, events in (memories or {}).items():
                if events:
                    self.db.execute("INSERT OR REPLACE INTO memories(person, chapter, events) VALUES(?,?,?)",
                                    (person, n, "; ".join(str(e) for e in events)[:600]))
            conflicts = self._file_facts(n, facts or [])
        rolled = self._roll_up(n, summarise)
        return {"chapter": n, "conflicts": conflicts, "rolled_up": rolled, "names": len(names)}

    def retitle(self, n: int, title: str) -> None:
        """A chapter's title can change after it is filed (the packager or an A/B test picks a better one)."""
        with self._lock, self.db:
            self.db.execute("UPDATE chapters SET title=? WHERE n=?", (title, n))

    def _forget(self, n: int) -> None:
        """Take back what one chapter filed, so that rewriting a chapter does not leave its old facts behind."""
        self.db.execute("UPDATE facts SET replaced_at=NULL WHERE replaced_at=?", (n,))
        self.db.execute("DELETE FROM facts WHERE chapter=?", (n,))
        self.db.execute("DELETE FROM conflicts WHERE chapter=?", (n,))
        for row in self.db.execute("SELECT name, times FROM mentions WHERE chapter=?", (n,)).fetchall():
            self.db.execute("UPDATE people SET chapters=chapters-1, times=times-? WHERE name=?", (row["times"], row["name"]))
        self.db.execute("DELETE FROM mentions WHERE chapter=?", (n,))
        self.db.execute("DELETE FROM memories WHERE chapter=?", (n,))
        if self.fts:
            old = self.db.execute("SELECT title, summary FROM chapters WHERE n=?", (n,)).fetchone()
            if old:
                self.db.execute("INSERT INTO search(search, rowid, summary) VALUES('delete', ?, ?)",
                                (n, f"{old['title']}. {old['summary']}"))

    def _file_facts(self, n: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for f in facts[:24]:
            if not isinstance(f, dict):
                continue
            subject, attribute, value = (str(f.get(k, "") or "").strip() for k in ("subject", "attribute", "value"))
            if not subject or not attribute or not value:
                continue
            ns, na, nv = _norm(subject), _norm(attribute), _norm(value)
            keep = lasting(attribute, value, f.get("lasting"))
            current = self.db.execute("SELECT * FROM facts WHERE ns=? AND na=? AND replaced_at IS NULL ORDER BY chapter DESC",
                                      (ns, na)).fetchall()
            same = [r for r in current if r["nv"] == nv or r["nv"] in nv or nv in r["nv"]]
            if same:
                continue                                  # already known
            for r in current:
                if r["lasting"] and r["chapter"] < n:     # what was set down to last is contradicted by any other value
                    c = {"chapter": n, "subject": subject, "attribute": attribute, "was": r["value"], "was_chapter": r["chapter"],
                         "now": value, "quote": str(f.get("quote", ""))[:160]}
                    self.db.execute("INSERT INTO conflicts(chapter, subject, attribute, was, was_chapter, now, quote) "
                                    "VALUES(?,?,?,?,?,?,?)", (n, subject, attribute, r["value"], r["chapter"], value, c["quote"]))
                    found.append(c)
                else:                                     # a fact that is allowed to move on: the old value becomes history
                    self.db.execute("UPDATE facts SET replaced_at=? WHERE id=?", (n, r["id"]))
            if not any(c["subject"] == subject and c["attribute"] == attribute for c in found):
                self.db.execute("INSERT INTO facts(subject, attribute, value, lasting, chapter, quote, ns, na, nv) "
                                "VALUES(?,?,?,?,?,?,?,?,?)", (subject[:80], attribute[:80], value[:240], int(keep), n,
                                                              str(f.get("quote", ""))[:160], ns, na, nv))
        return found

    def _roll_up(self, n: int, summarise: Optional[Callable[[list[str], int, int], str]]) -> list[str]:
        """When chapter n completes an arc (and perhaps a volume, a saga), summarise the layer below."""
        made = []
        size = SPAN
        for level in sorted(LEVELS):
            if n % size:
                break
            first, idx = n - size + 1, n // size
            with self._lock:
                if level == 1:
                    rows = self.db.execute("SELECT summary FROM chapters WHERE n BETWEEN ? AND ? ORDER BY n", (first, n)).fetchall()
                else:
                    rows = self.db.execute("SELECT summary FROM rollups WHERE level=? AND first>=? AND last<=? ORDER BY idx",
                                           (level - 1, first, n)).fetchall()
            parts = [r["summary"] for r in rows if r["summary"]]
            if not parts:
                break
            text = ""
            if summarise is not None:
                try:
                    text = _trim(summarise(parts, first, n), 150)
                except Exception as e:
                    log.warning("chronicle: could not summarise chapters %d-%d (%s); keeping the first lines", first, n, e)
            if not text:                                  # no model to hand: keep the first sentence of each part
                each = max(8, 140 // len(parts))
                text = " ".join(_trim(re.split(r"(?<=[.!?])\s", p)[0], each) for p in parts)
            with self._lock, self.db:
                self.db.execute("INSERT OR REPLACE INTO rollups(level, idx, first, last, summary) VALUES(?,?,?,?,?)",
                                (level, idx, first, n, text))
            made.append(f"{LEVELS[level]} {idx} (chapters {first}-{n})")
            size *= SPAN
        return made

    # ---- reading -----------------------------------------------------------------------
    def written(self) -> int:
        with self._lock:
            r = self.db.execute("SELECT MAX(n) AS m FROM chapters").fetchone()
        return int(r["m"] or 0)

    def titles(self) -> dict[int, str]:
        with self._lock:
            return {int(r["n"]): str(r["title"]) for r in self.db.execute("SELECT n, title FROM chapters")}

    def story_so_far(self, before: int, words: int = 600) -> str:
        """Everything before chapter `before`, in layers: finished sagas, then the finished volumes of the
        current saga, the finished arcs of the current volume, and the chapters of the current arc. The
        number of pieces does not grow with the story; they are trimmed, oldest hardest, to fit `words`."""
        done = before - 1
        if done <= 0:
            return ""
        pieces: list[tuple[str, str, int]] = []           # (label, text, nearness: higher is nearer the present)
        covered = 0
        with self._lock:
            for level in sorted(LEVELS, reverse=True):
                size = SPAN ** level
                upto = (done // size) * size
                rows = self.db.execute("SELECT * FROM rollups WHERE level=? AND first>? AND last<=? ORDER BY idx",
                                       (level, covered, upto)).fetchall()
                for r in rows:
                    pieces.append((f"Chapters {r['first']}-{r['last']}", r["summary"], 5 - level))
                if rows:
                    covered = max(covered, rows[-1]["last"])
            for r in self.db.execute("SELECT * FROM chapters WHERE n>? AND n<=? ORDER BY n", (covered, done)).fetchall():
                pieces.append((f"Chapter {r['n']}", r["summary"], 5))
        if not pieces:
            return ""
        # Share the room: the nearer to now, the more words a piece may keep.
        weight = sum(p[2] for p in pieces)
        out = []
        for label, text, near in pieces:
            out.append(f"{label}: {_trim(text, max(10, int(words * near / weight)))}")
        return "\n".join(out)

    def facts_about(self, name: str, limit: int = 12) -> list[dict[str, Any]]:
        """What is true of someone or something now: lasting facts first, then the latest of the rest."""
        ns = _norm(name)
        if not ns:
            return []
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM facts WHERE replaced_at IS NULL AND (ns=? OR ns LIKE ? OR ns LIKE ? OR ns LIKE ?) "
                "ORDER BY lasting DESC, chapter DESC LIMIT ?", (ns, ns + " %", "% " + ns, "% " + ns + " %", limit)).fetchall()
        return [dict(r) for r in rows]

    def history(self, name: str, attribute: str = "") -> list[dict[str, Any]]:
        """Every value a fact has had, oldest first."""
        ns, na = _norm(name), _norm(attribute)
        with self._lock:
            rows = self.db.execute("SELECT * FROM facts WHERE ns=?" + (" AND na=?" if na else "") + " ORDER BY chapter",
                                   (ns, na) if na else (ns,)).fetchall()
        return [dict(r) for r in rows]

    def person(self, name: str) -> Optional[dict[str, Any]]:
        with self._lock:
            r = self.db.execute("SELECT * FROM people WHERE name=?", (name,)).fetchone()
            if r is None:
                r = self.db.execute("SELECT * FROM people WHERE name LIKE ? ORDER BY times DESC LIMIT 1", (name.split()[0] + "%",)).fetchone()
        return dict(r) if r else None

    def known_names(self, min_chapters: int = 1) -> set[str]:
        with self._lock:
            return {r["name"] for r in self.db.execute("SELECT name FROM people WHERE chapters>=?", (min_chapters,))}

    def dead(self) -> dict[str, dict[str, Any]]:
        """Everyone the story has said is dead and has not since said is alive: name -> the fact."""
        with self._lock:
            rows = self.db.execute("SELECT * FROM facts WHERE replaced_at IS NULL ORDER BY chapter").fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            if not _STATE_OF_LIFE.search(r["attribute"]):
                continue
            if _ALIVE_AGAIN.search(r["value"]):
                out.pop(r["subject"], None)
            elif says_dead(r["attribute"], r["value"]):
                out[r["subject"]] = dict(r)
        return out

    def similar(self, text: str, before: int, k: int = 3, skip_last: int = 12) -> list[dict[str, Any]]:
        """The earlier chapters most like `text`, leaving out the recent ones the writer already has."""
        words = [w for w in dict.fromkeys(re.findall(r"[A-Za-z]{4,}", text)) if w.lower() not in _COMMON][:24]
        if not words or not self.fts:
            return []
        query = " OR ".join(f'"{w}"' for w in words)
        try:
            with self._lock:
                rows = self.db.execute(
                    "SELECT c.n, c.title, c.summary FROM search JOIN chapters c ON c.n = search.rowid "
                    "WHERE search MATCH ? AND c.n < ? ORDER BY bm25(search) LIMIT ?", (query, before - skip_last, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]

    def open_conflicts(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM conflicts WHERE settled=0 ORDER BY chapter DESC LIMIT ?", (limit,))]

    def settle(self, conflict_id: int, keep_new: bool = False) -> None:
        """The author's ruling on a contradiction: the earlier value stands, or the later one replaces it."""
        with self._lock, self.db:
            c = self.db.execute("SELECT * FROM conflicts WHERE id=?", (conflict_id,)).fetchone()
            if c is None:
                return
            if keep_new:
                self.db.execute("UPDATE facts SET replaced_at=? WHERE ns=? AND na=? AND replaced_at IS NULL",
                                (c["chapter"], _norm(c["subject"]), _norm(c["attribute"])))
                self.db.execute("INSERT INTO facts(subject, attribute, value, lasting, chapter, quote, ns, na, nv) VALUES(?,?,?,?,?,?,?,?,?)",
                                (c["subject"], c["attribute"], c["now"], 1, c["chapter"], c["quote"], _norm(c["subject"]),
                                 _norm(c["attribute"]), _norm(c["now"])))
            self.db.execute("UPDATE conflicts SET settled=1 WHERE id=?", (conflict_id,))

    # ---- what the writer is told before a chapter ------------------------------------------
    def brief(self, n: int, plan_text: str, cast: Optional[list[str]] = None, words: int = 350) -> str:
        """What the story's past says that matters to chapter `n`: facts about who and what is in the plan,
        reminders for anyone coming back after a long time, the earlier chapters most like this one, and
        contradictions still waiting to be put right. Sized to `words`."""
        if self.written() == 0:
            return ""
        known = self.known_names()
        names = list(dict.fromkeys(list(cast or []) + [w for w in proper_names(plan_text, known) if w in known]))
        lines: list[str] = []
        dead = self.dead()
        for name in names[:10]:
            first = name.split()[0]
            facts = self.facts_about(name, 6) or (self.facts_about(first, 6) if first != name else [])
            who = self.person(name)
            gone = dead.get(name) or dead.get(first)
            if gone:
                lines.append(f"- {name} is DEAD (chapter {gone['chapter']}: {gone['value']}). Only as a memory, a body or a ghost.")
            if who and n - int(who["last"]) > LONG_ABSENCE:
                last = self.chapter(int(who["last"]))
                lines.append(f"- {name} was last seen in chapter {who['last']}, {n - int(who['last'])} chapters ago"
                             + (f": {_trim(last['summary'], 30)}" if last else "") + ".")
            for f in facts:
                lines.append(f"- {f['subject']}, {f['attribute']}: {f['value']} (chapter {f['chapter']}"
                             + (", must not change" if f["lasting"] else "") + ")")
            for mem in self.remembers(name, [x for x in names if x != name], n):
                lines.append(f"- {name} remembers (chapter {mem['chapter']}): {_trim(mem['events'], 24)}")
        for past in self.similar(plan_text, n):
            lines.append(f"- Like this before, chapter {past['n']} ({past['title']}): {_trim(past['summary'], 28)}")
        for c in self.open_conflicts(3):
            lines.append(f"- UNRESOLVED: {c['subject']}, {c['attribute']} was '{c['was']}' (chapter {c['was_chapter']}) but chapter "
                         f"{c['chapter']} said '{c['now']}'. Keep to the earlier one unless the story explains the change.")
        out, used = [], 0
        for line in dict.fromkeys(lines):
            cost = len(line.split())
            if used + cost > words:
                break
            out.append(line)
            used += cost
        return "\n".join(out)

    def chapter(self, n: int) -> Optional[dict[str, Any]]:
        with self._lock:
            r = self.db.execute("SELECT * FROM chapters WHERE n=?", (n,)).fetchone()
        return dict(r) if r else None

    def remembers(self, person: str, others: list[str], before: int, k: int = 2) -> list[dict[str, Any]]:
        """Old memories of `person` that involve someone else in the scene, from before the recent ones they already carry."""
        if not others:
            return []
        with self._lock:
            rows = self.db.execute("SELECT * FROM memories WHERE person=? AND chapter<? ORDER BY chapter DESC LIMIT 400",
                                   (person, before - 40)).fetchall()
        hits = [dict(r) for r in rows if any(o.split()[0] in r["events"] for o in others)]
        return hits[:k]

    # ---- checking a draft ---------------------------------------------------------------------
    def check(self, text: str, facts: Optional[list[dict[str, Any]]] = None, n: int = 0) -> list[str]:
        """What in a draft contradicts the story so far, in words a writer can act on. Without `facts` only
        the dead are checked (no model needed); with them, every lasting fact is."""
        issues: list[str] = []
        for name, f in self.dead().items():
            who = "|".join(re.escape(w) for w in _name_words(name))
            acts = re.search(rf"\b(?:{who})\b(?:\s+\w+)?\s+{_ACTS}\b|{_ACTS}\s+(?:{who})\b", text)
            if acts:
                around = text[max(0, acts.start() - 240):acts.end() + 240]
                if not _MEMORY.search(around):
                    issues.append(f"{name} died in chapter {f['chapter']} ({f['value']}) but acts here: \"{_trim(text[acts.start():acts.end() + 40], 12)}\". "
                                  "If this is a memory or a ghost, say so in the text; otherwise use someone living.")
        for nf in facts or []:
            if not isinstance(nf, dict) or not nf.get("subject") or not nf.get("attribute") or not nf.get("value"):
                continue
            nv = _norm(nf["value"])
            with self._lock:
                rows = self.db.execute("SELECT * FROM facts WHERE ns=? AND na=? AND replaced_at IS NULL AND lasting=1",
                                       (_norm(nf["subject"]), _norm(nf["attribute"]))).fetchall()
            for r in rows:
                if r["nv"] != nv and r["nv"] not in nv and nv not in r["nv"] and (not n or r["chapter"] < n):
                    issues.append(f"{nf['subject']}, {nf['attribute']}: chapter {r['chapter']} established '{r['value']}', "
                                  f"but this says '{nf['value']}'. Keep to chapter {r['chapter']}.")
        return issues[:6]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            one = lambda q: int(self.db.execute(q).fetchone()[0] or 0)  # noqa: E731
            return {"chapters": one("SELECT COUNT(*) FROM chapters"), "facts": one("SELECT COUNT(*) FROM facts"),
                    "current_facts": one("SELECT COUNT(*) FROM facts WHERE replaced_at IS NULL"),
                    "people": one("SELECT COUNT(*) FROM people"), "rollups": one("SELECT COUNT(*) FROM rollups"),
                    "conflicts": one("SELECT COUNT(*) FROM conflicts WHERE settled=0"),
                    "words": one("SELECT SUM(words) FROM chapters"), "bytes": self.path.stat().st_size if self.path.exists() else 0}


_COMMON = {"that", "this", "with", "from", "have", "will", "they", "them", "then", "than", "when", "what", "where", "which",
           "there", "their", "about", "into", "over", "after", "before", "chapter", "must", "should", "would", "could", "been",
           "were", "goal", "hook", "turn", "ends", "opens", "while", "because", "through", "again", "still", "only", "back"}


def digest_from(d: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """The chapter's own summary and its facts, out of the digest the model returned."""
    facts = d.get("facts") if isinstance(d.get("facts"), list) else []
    return str(d.get("chapter") or d.get("chapter_summary") or ""), [f for f in facts if isinstance(f, dict)]


def export_json(chron: Chronicle, limit: int = 200) -> str:
    """The newest facts and every open contradiction, for a person to read."""
    with chron._lock:
        facts = [dict(r) for r in chron.db.execute("SELECT subject, attribute, value, lasting, chapter FROM facts "
                                                   "WHERE replaced_at IS NULL ORDER BY chapter DESC LIMIT ?", (limit,))]
    return json.dumps({"stats": chron.stats(), "facts": facts, "conflicts": chron.open_conflicts()}, indent=1)
