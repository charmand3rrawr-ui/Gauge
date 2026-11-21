"""Export a serial's chapters into the formats other platforms want (roadmap item 77).

Royal Road, Scribble Hub and most forums take pasted rich text or a markup dialect. This renders each
chapter into a chosen format — Markdown, HTML, BBCode, or clean plain text — so the author can paste or
upload it without reformatting by hand. It reads the chapters already on disk; it does not post anywhere.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

FORMATS = ("markdown", "html", "bbcode", "text")
_EXT = {"markdown": "md", "html": "html", "bbcode": "txt", "text": "txt"}


def _paras(body: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]


def format_chapter(number: int, title: str, body: str, fmt: str) -> str:
    heading = f"Chapter {number}: {title}" if title else f"Chapter {number}"
    paras = _paras(body)
    if fmt == "markdown":
        return f"# {heading}\n\n" + "\n\n".join(paras) + "\n"
    if fmt == "html":
        return f"<h1>{html.escape(heading)}</h1>\n" + "\n".join(f"<p>{html.escape(p)}</p>" for p in paras) + "\n"
    if fmt == "bbcode":
        return f"[b][size=5]{heading}[/size][/b]\n\n" + "\n\n".join(paras) + "\n"
    if fmt == "text":
        return f"{heading}\n{'=' * len(heading)}\n\n" + "\n\n".join(paras) + "\n"
    raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")


def export_project(project: Any, fmt: str, out_dir: Path) -> list[Path]:
    """Write one file per written chapter in the chosen format, plus a combined file. Returns the files."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = _EXT[fmt]
    written: list[Path] = []
    combined: list[str] = []
    n = 1
    while True:
        text = project.read_chapter(n)
        if text is None:
            break
        meta = project.read_meta(n) if hasattr(project, "read_meta") else {}
        title = meta.get("title") or ""
        rendered = format_chapter(n, title, text, fmt)
        p = out / f"chapter_{n:03d}.{ext}"
        p.write_text(rendered, encoding="utf-8")
        written.append(p)
        combined.append(rendered)
        n += 1
    if combined:
        name = getattr(getattr(project, "state", None), "name", "serial")
        allp = out / f"{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'serial'}.{ext}"
        allp.write_text(("\n\n".join(combined)), encoding="utf-8")
        written.append(allp)
    return written
