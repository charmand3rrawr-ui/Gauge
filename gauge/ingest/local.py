"""Learn from novels on your own disk.

Three shapes are understood, and told apart automatically:

* **A novel folder**: chapter files (``.txt`` ``.md`` ``.html`` ``.docx``) sorted naturally, or a
  single file / ``.epub`` / ``.docx`` holding the whole book, split into chapters on headings.
* **A library folder**: a folder whose children are novel folders and/or whole-book files.
  ``gauge import LIBRARY`` imports every novel inside it, recursively.
* **A watched folder**: ``gauge watch PATH`` registers a novel or library folder so every
  ``gauge ingest`` / daemon cycle re-scans it and picks up new or edited chapters.

Metadata and engagement labels come from, in order of priority:

1. ``novel.json`` (or ``metadata.json`` / ``gauge.json``) inside the novel folder, or
   ``<book>.json`` next to a single-file novel;
2. a ``library.json`` / ``library.csv`` at the library root, keyed by folder or file name;
3. EPUB / DOCX embedded metadata (title, author, subjects, description);
4. the folder or file name, with ``Author - Title`` and ``Title by Author`` patterns recognised;
5. a ``Title`` / ``Author:`` line at the top of the first chapter.

Numbers you can supply: ``followers``, ``favorites``, ``views``, ``average_views``, ``rating``,
``rating_count``, ``comments``, ``user_rating`` (your own 1-5), per-chapter ``views`` / ``comments``.
Unlabelled novels are still analysed and still provide style exemplars; they just cannot
train the engagement model unless you give them a ``user_rating`` (``gauge rate``).
"""
from __future__ import annotations

import csv
import json
import logging
import re
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

from ..models import ChapterRecord, NovelRecord
from ..text import normalize_whitespace

log = logging.getLogger(__name__)

CHAPTER_EXTS = {".txt", ".md", ".markdown", ".html", ".htm", ".docx"}
BOOK_EXTS = {".epub", ".docx", ".txt", ".md", ".html", ".htm"}
META_NAMES = ("novel.json", "metadata.json", "gauge.json")
LIBRARY_META = ("library.json", "library.csv")
SKIP_NAMES = {"readme", "notes", "cover", "metadata", "novel", "library", "outline", "bible",
              "changelog", "license", "toc", "contents", "index"}
MIN_CHAPTER_WORDS = 120          # shorter files are treated as front/back matter
WHOLE_BOOK_WORDS = 12000         # a file this long is a whole novel, not a chapter
PSEUDO_CHAPTER_WORDS = 2500      # split heading-less books into pieces this long

_NUM_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
              "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|hundred")
_CHAPTER_HEAD = re.compile(
    rf"^[ \t]*(?:#{{1,4}}[ \t]*)?(?:\*\*)?(?:chapter|chap\.?|ch\.?|episode|ep\.?|part|book|arc|volume|vol\.?)"
    rf"[ \t]*(?:\d+|[ivxlcdm]+|(?:{_NUM_WORDS})(?:[ \-](?:{_NUM_WORDS}))?)\b[^\n]{{0,80}}$",
    re.I | re.M)
_MD_HEAD = re.compile(r"^[ \t]*#{1,3}[ \t]+\S[^\n]{0,80}$", re.M)
_BARE_NUMBER_HEAD = re.compile(r"^[ \t]*(\d{1,4})\.?[ \t]*$", re.M)
_AUTHOR_TITLE = re.compile(r"^(?P<author>[^-–—]{2,60}?)\s*[-–—]\s*(?P<title>.{2,120})$")
_TITLE_BY = re.compile(r"^(?P<title>.{2,120}?)\s+by\s+(?P<author>[^(]{2,60})$", re.I)
_TRAILING_TAGS = re.compile(r"\s*[\(\[][^\)\]]{1,40}[\)\]]\s*$")  # "(v1.2)", "[complete]"


# --------------------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------------------
def read_text(path: Path) -> str:
    """Read text with sensible encoding fallbacks (Windows exports are often cp1252)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def html_to_text(html: str) -> tuple[str, str]:
    """Return (heading, body) from an HTML fragment/document; headings become '# ' lines."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for bad in soup.select("script, style, nav, header, footer"):
        bad.decompose()
    body = soup.body or soup
    lines: list[str] = []
    heading = ""
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "div"]):
        if el.find(["p", "div", "h1", "h2", "h3"]):
            continue  # container; its children are visited separately
        t = el.get_text(" ", strip=True)
        if not t:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            if not heading:
                heading = t
            lines.append(f"# {t}")
        else:
            lines.append(t)
    if not lines:
        return heading, normalize_whitespace(body.get_text("\n", strip=True))
    return heading, normalize_whitespace("\n\n".join(lines))


def read_docx(path: Path) -> tuple[dict[str, Any], str]:
    """Extract (metadata, text) from a .docx with the stdlib only. Headings become '# ' lines."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    meta: dict[str, Any] = {}
    with zipfile.ZipFile(path) as z:
        if "docProps/core.xml" in z.namelist():
            core = ET.fromstring(z.read("docProps/core.xml"))
            for tag, key in (("{http://purl.org/dc/elements/1.1/}title", "title"),
                             ("{http://purl.org/dc/elements/1.1/}creator", "author"),
                             ("{http://purl.org/dc/elements/1.1/}description", "blurb"),
                             ("{http://purl.org/dc/elements/1.1/}subject", "subject")):
                el = core.find(tag)
                if el is not None and (el.text or "").strip():
                    meta[key] = el.text.strip()
        doc = ET.fromstring(z.read("word/document.xml"))
    paras: list[str] = []
    for p in doc.iter(f"{{{ns['w']}}}p"):
        text = "".join(t.text or "" for t in p.iter(f"{{{ns['w']}}}t")).strip()
        if not text:
            continue
        style = p.find("w:pPr/w:pStyle", ns)
        val = (style.get(f"{{{ns['w']}}}val") if style is not None else "") or ""
        if val.lower().startswith(("heading", "title")) or re.fullmatch(r"[Hh]\d", val):
            paras.append(f"# {text}")
        else:
            paras.append(text)
    return meta, normalize_whitespace("\n\n".join(paras))


def read_epub(path: Path) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Extract (metadata, [(heading, text), ...] in spine order) from an EPUB, stdlib + bs4 only."""
    meta: dict[str, Any] = {}
    docs: list[tuple[str, str]] = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        opf_path = None
        if "META-INF/container.xml" in names:
            cont = ET.fromstring(z.read("META-INF/container.xml"))
            rf = cont.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
            if rf is not None:
                opf_path = rf.get("full-path")
        if not opf_path:
            opf_path = next((n for n in names if n.lower().endswith(".opf")), None)
        if not opf_path:
            raise ValueError(f"{path.name}: no OPF package found")
        opf = ET.fromstring(z.read(opf_path))
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        dc = "{http://purl.org/dc/elements/1.1/}"
        for tag, key in ((f"{dc}title", "title"), (f"{dc}creator", "author"), (f"{dc}description", "blurb")):
            el = opf.find(f".//{tag}")
            if el is not None and (el.text or "").strip():
                meta[key] = re.sub(r"<[^>]+>", " ", el.text).strip()
        subjects = [s.text.strip() for s in opf.findall(f".//{dc}subject") if (s.text or "").strip()]
        if subjects:
            meta["genres"] = subjects
        manifest: dict[str, tuple[str, str]] = {}
        for item in opf.iter():
            if item.tag.endswith("}item") or item.tag == "item":
                manifest[item.get("id", "")] = (item.get("href", ""), item.get("media-type", ""))
        spine = [ir.get("idref", "") for ir in opf.iter() if ir.tag.endswith("}itemref") or ir.tag == "itemref"]
        hrefs = [manifest[i] for i in spine if i in manifest]
        if not hrefs:
            hrefs = [v for v in manifest.values() if "html" in v[1]]
        for href, mt in hrefs:
            if "html" not in mt and not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            full = (base + href).split("#")[0]
            full = re.sub(r"[^/]+/\.\./", "", full)
            if full not in names:
                alt = next((n for n in names if n.endswith(href.split("#")[0].split("/")[-1])), None)
                if alt is None:
                    continue
                full = alt
            heading, text = html_to_text(z.read(full).decode("utf-8", errors="replace"))
            if text:
                docs.append((heading, text))
    return meta, docs


# --------------------------------------------------------------------------------------
# Splitting text into chapters
# --------------------------------------------------------------------------------------
def _words(s: str) -> int:
    return len(s.split())


def _split_on(text: str, heads: list[re.Match]) -> list[tuple[str, str]]:
    out = []
    for i, m in enumerate(heads):
        start = m.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        title = re.sub(r"^[#\s*]+|[*\s]+$", "", m.group(0)).strip()
        out.append((title, text[start:end].strip()))
    return [c for c in out if c[1]]


def split_single_file(text: str, pseudo_words: int = PSEUDO_CHAPTER_WORDS) -> list[tuple[str, str]]:
    """Split one document into (title, body) chapters.

    Tries, in order: 'Chapter N' style headings (numbers, roman numerals or words), Markdown
    headings, bare numbers on their own line, and finally fixed-size pseudo-chapters for long
    heading-less books. Text before the first heading is dropped if short (front matter)."""
    text = normalize_whitespace(text)
    for pattern in (_CHAPTER_HEAD, _MD_HEAD, _BARE_NUMBER_HEAD):
        heads = list(pattern.finditer(text))
        if len(heads) >= 2:
            parts = _split_on(text, heads)
            if len(parts) >= 2:
                lead = text[: heads[0].start()].strip()
                if _words(lead) >= MIN_CHAPTER_WORDS * 3:
                    parts.insert(0, ("Prologue", lead))
                return parts
    if _words(text) > pseudo_words * 2:
        ws = text.split("\n\n")
        parts, cur, n = [], [], 0
        for par in ws:
            cur.append(par)
            n += _words(par)
            if n >= pseudo_words:
                parts.append((f"Part {len(parts) + 1}", "\n\n".join(cur)))
                cur, n = [], 0
        if cur:
            if parts and n < pseudo_words // 3:
                t, b = parts[-1]
                parts[-1] = (t, b + "\n\n" + "\n\n".join(cur))
            else:
                parts.append((f"Part {len(parts) + 1}", "\n\n".join(cur)))
        return parts
    # One heading (or none): a single chapter, with the heading line used as its title.
    for pattern in (_CHAPTER_HEAD, _MD_HEAD):
        heads = list(pattern.finditer(text))
        if len(heads) == 1 and heads[0].start() == 0:
            parts = _split_on(text, heads)
            if parts:
                return parts
    return [("Chapter 1", text)] if text else []


def chapters_from_docs(docs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """EPUB/HTML documents: one doc per chapter when they look like chapters, else re-split."""
    real = [(h or f"Chapter {i}", t) for i, (h, t) in enumerate(docs, 1) if _words(t) >= MIN_CHAPTER_WORDS]
    if len(real) >= 2:
        return [(h, re.sub(r"^#\s[^\n]*\n+", "", t)) for h, t in real]
    joined = "\n\n".join(t for _, t in docs)
    return split_single_file(joined)


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------
_NUMERIC = {"followers", "favorites", "views", "average_views", "rating_count", "comments"}
_FLOAT = {"rating", "user_rating"}
_LISTS = {"genres", "tags"}


def normalize_meta(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (raw or {}).items():
        k = str(k).strip().lower().replace(" ", "_")
        if k in {"subject", "subjects", "genre"}:
            k = "genres"
        if k in {"tag"}:
            k = "tags"
        if k in {"description", "synopsis", "summary"}:
            k = "blurb"
        if k in {"my_rating", "stars", "score"}:
            k = "user_rating"
        if v in (None, ""):
            continue
        if k in _NUMERIC:
            try:
                out[k] = int(float(str(v).replace(",", "")))
            except ValueError:
                continue
        elif k in _FLOAT:
            try:
                out[k] = float(str(v).replace(",", "."))
            except ValueError:
                continue
        elif k in _LISTS:
            out[k] = [s.strip() for s in (v if isinstance(v, list) else re.split(r"[;,|/]", str(v))) if str(s).strip()]
        elif k == "chapters" and isinstance(v, dict):
            out[k] = {str(i): d for i, d in v.items() if isinstance(d, dict)}
        else:
            out[k] = v if isinstance(v, (str, int, float, bool, list, dict)) else str(v)
    return out


def _sidecar_for(path: Path) -> dict[str, Any]:
    cands = [path / n for n in META_NAMES] if path.is_dir() else [path.with_suffix(".json"), path.parent / f"{path.stem}.json"]
    for c in cands:
        if c.exists():
            try:
                return normalize_meta(json.loads(read_text(c)))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("bad sidecar %s: %s", c, e)
    return {}


def load_library_meta(root: Path) -> dict[str, dict[str, Any]]:
    """library.json: {"Folder Name": {...}} ; library.csv: a 'name' (or 'folder'/'file') column + fields."""
    out: dict[str, dict[str, Any]] = {}
    j = root / "library.json"
    if j.exists():
        try:
            data = json.loads(read_text(j))
            if isinstance(data, dict):
                out.update({str(k).lower(): normalize_meta(v) for k, v in data.items() if isinstance(v, dict)})
            elif isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        name = str(row.get("name") or row.get("folder") or row.get("file") or "").lower()
                        if name:
                            out[name] = normalize_meta(row)
        except json.JSONDecodeError as e:
            log.warning("bad library.json: %s", e)
    c = root / "library.csv"
    if c.exists():
        with c.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name") or row.get("folder") or row.get("file") or "").lower()
                if name:
                    out[name] = normalize_meta({k: v for k, v in row.items() if k not in {"name", "folder", "file"}})
    return out


def infer_from_name(name: str) -> dict[str, str]:
    base = _TRAILING_TAGS.sub("", name).replace("_", " ").strip()
    base = re.sub(r"\s+", " ", base)
    m = _TITLE_BY.match(base)
    if m:
        return {"title": m.group("title").strip(), "author": m.group("author").strip()}
    m = _AUTHOR_TITLE.match(base)
    if m and not re.search(r"\d", m.group("author")):
        return {"title": m.group("title").strip(), "author": m.group("author").strip()}
    return {"title": base.title() if base.islower() else base}


def infer_from_text(first_chapter: str) -> dict[str, str]:
    """Pick up 'Title' / 'by Author' / 'Author: X' lines from the first few lines of a book."""
    out: dict[str, str] = {}
    lines = [ln.strip(" #*") for ln in first_chapter.splitlines()[:8] if ln.strip()]
    for ln in lines:
        m = re.match(r"^(?:author|by)\s*:?\s+(.{2,60})$", ln, re.I)
        if m and "author" not in out:
            out["author"] = m.group(1).strip()
        m = re.match(r"^title\s*:\s*(.{2,120})$", ln, re.I)
        if m and "title" not in out:
            out["title"] = m.group(1).strip()
    return out


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80] or "novel"


# --------------------------------------------------------------------------------------
# Novel folders / files
# --------------------------------------------------------------------------------------
@dataclass
class ImportedNovel:
    novel: NovelRecord
    chapters: list[ChapterRecord]
    source_path: Path
    warnings: list[str] = field(default_factory=list)


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]


def _chapter_files(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in CHAPTER_EXTS
             and not p.name.startswith((".", "_")) and p.stem.lower() not in SKIP_NAMES]
    return sorted(files, key=_natural_key)


def _book_files(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in BOOK_EXTS
            and not p.name.startswith((".", "_")) and p.stem.lower() not in SKIP_NAMES]


def _file_chapters(path: Path) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    ext = path.suffix.lower()
    if ext == ".epub":
        meta, docs = read_epub(path)
        return meta, chapters_from_docs(docs)
    if ext == ".docx":
        meta, text = read_docx(path)
        return meta, split_single_file(text)
    if ext in {".html", ".htm"}:
        heading, text = html_to_text(read_text(path))
        meta = {"title": heading} if heading else {}
        return meta, split_single_file(text)
    return {}, split_single_file(read_text(path))


def is_library(path: Path) -> bool:
    """A library is a folder of novels rather than a folder of chapters."""
    if not path.is_dir():
        return False
    if any((path / n).exists() for n in META_NAMES):
        return False
    if any((path / n).exists() for n in LIBRARY_META):
        return True
    subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
    novel_dirs = [d for d in subdirs if _chapter_files(d) or _book_files(d) or any(x.is_dir() for x in d.iterdir())]
    files = _chapter_files(path)
    books = [p for p in _book_files(path) if p.suffix.lower() in {".epub", ".docx"}]
    if novel_dirs and not files:
        return True
    if len(books) >= 2 and not files:
        return True
    if files and len(files) >= 2:
        # Several long text files side by side are several novels, not several chapters.
        sizes = [p.stat().st_size for p in files[:12]]
        if sum(sizes) / len(sizes) > WHOLE_BOOK_WORDS * 6 and not any(re.search(r"ch(apter)?[\s_\-]*\d", p.stem, re.I) for p in files):
            return True
    return False


def import_novel(path: Path | str, meta_override: Optional[dict[str, Any]] = None,
                 slug: Optional[str] = None) -> ImportedNovel:
    """Import one novel from a folder of chapter files or a single book file."""
    path = Path(path)
    warnings: list[str] = []
    meta: dict[str, Any] = {}
    chapters: list[tuple[str, str]] = []
    if path.is_dir():
        files = _chapter_files(path)
        if not files:
            raise ValueError(f"{path}: no chapter files ({', '.join(sorted(CHAPTER_EXTS))})")
        if len(files) == 1 and files[0].suffix.lower() in BOOK_EXTS:
            fmeta, chapters = _file_chapters(files[0])
            meta.update(fmeta)
        else:
            for f in files:
                ext = f.suffix.lower()
                if ext == ".docx":
                    fmeta, text = read_docx(f)
                elif ext in {".html", ".htm"}:
                    _, text = html_to_text(read_text(f))
                else:
                    text = normalize_whitespace(read_text(f))
                if _words(text) < MIN_CHAPTER_WORDS:
                    warnings.append(f"skipped {f.name}: {_words(text)} words (front/back matter?)")
                    continue
                title = re.sub(r"^#\s*", "", text.splitlines()[0]).strip() if text.startswith("#") else f.stem.replace("_", " ").replace("-", " ").strip()
                if text.startswith("#"):
                    text = re.sub(r"^#[^\n]*\n+", "", text)
                chapters.append((title, text))
        name_meta = infer_from_name(path.name)
    else:
        fmeta, chapters = _file_chapters(path)
        meta.update(fmeta)
        name_meta = infer_from_name(path.stem)
    if not chapters:
        raise ValueError(f"{path}: no chapter text found")

    text_meta = infer_from_text(chapters[0][1])
    sidecar = _sidecar_for(path)
    merged: dict[str, Any] = {}
    for layer in (name_meta, text_meta, meta, sidecar, normalize_meta(meta_override or {})):
        merged.update({k: v for k, v in layer.items() if v not in (None, "", [], {})})

    slug = slug or slugify(merged.get("title") or path.stem)
    novel = NovelRecord(
        source="local", source_id=slug, url=str(path.resolve()),
        title=str(merged.get("title") or path.stem), author=str(merged.get("author", "")),
        blurb=str(merged.get("blurb", "")), genres=list(merged.get("genres", [])), tags=list(merged.get("tags", [])),
        status=str(merged.get("status", "")), followers=merged.get("followers"), favorites=merged.get("favorites"),
        views=merged.get("views"), average_views=merged.get("average_views"), rating=merged.get("rating"),
        rating_count=merged.get("rating_count"), comments=merged.get("comments"),
        user_rating=merged.get("user_rating"), chapter_count=len(chapters),
    )
    per_chapter = merged.get("chapters", {}) if isinstance(merged.get("chapters"), dict) else {}
    recs = []
    for i, (title, body) in enumerate(chapters, start=1):
        extra = per_chapter.get(str(i), {})
        recs.append(ChapterRecord(novel_key=novel.key, index=i, source_chapter_id=str(i), title=title, text=body,
                                  views=extra.get("views"), comments=extra.get("comments")))
    return ImportedNovel(novel=novel, chapters=recs, source_path=path, warnings=warnings)


def iter_library(root: Path | str) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield (novel_path, library_meta_for_it) for every novel under a library folder, recursively."""
    root = Path(root)
    lib_meta = load_library_meta(root)
    entries = sorted([p for p in root.iterdir() if not p.name.startswith((".", "_"))], key=_natural_key)
    for p in entries:
        if p.is_dir():
            if is_library(p):
                yield from iter_library(p)
            elif _chapter_files(p):
                yield p, lib_meta.get(p.name.lower(), {})
        elif p.suffix.lower() in BOOK_EXTS and p.stem.lower() not in SKIP_NAMES:
            # Loose book files at library level: epub/docx always; text only when long enough.
            if p.suffix.lower() in {".epub", ".docx"} or p.stat().st_size > WHOLE_BOOK_WORDS * 5:
                yield p, lib_meta.get(p.name.lower(), lib_meta.get(p.stem.lower(), {}))


def import_path(path: Path | str, slug: Optional[str] = None, meta_override: Optional[dict[str, Any]] = None,
                library: Optional[bool] = None) -> list[ImportedNovel]:
    """Import a novel or a whole library. ``library=None`` auto-detects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    treat_as_library = is_library(path) if library is None else library
    if treat_as_library:
        out = []
        for novel_path, lib_meta in iter_library(path):
            merged = dict(lib_meta)
            merged.update(meta_override or {})
            try:
                out.append(import_novel(novel_path, merged))
            except Exception as e:  # one bad book must not stop the library
                log.warning("skipping %s: %s", novel_path, e)
        return out
    return [import_novel(path, meta_override, slug)]
