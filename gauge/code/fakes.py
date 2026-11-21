"""Canned Engineer / Self-Programmer answers for the offline FakeLLM."""
from __future__ import annotations

import re

FAKE_BOT_FILES = {
    "main.py": '''"""Word-count report: one CSV row per text file in a folder. Deterministic; no model."""
import argparse
import sys
from pathlib import Path

from gauge.code import toolkit


def report(folder: Path, out: Path) -> int:
    rows = []
    for p in toolkit.list_files(folder, ("*.md", "*.txt")):
        stats = toolkit.text_stats(toolkit.strip_front_matter(toolkit.read_text(p)))
        rows.append({"file": p.name, **stats})
    toolkit.write_csv(out, rows, ["file", "words", "sentences", "paragraphs", "avg_sentence_words", "dialogue_lines", "reading_minutes"])
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="word counts for every chapter file in a folder")
    ap.add_argument("--folder", default=".")
    ap.add_argument("--out", default="wordcount.csv")
    a = ap.parse_args()
    folder = Path(a.folder)
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2
    n = report(folder, Path(a.out))
    print(f"{n} files -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
    "test_main.py": '''from pathlib import Path

from main import report


def test_report(tmp_path: Path):
    (tmp_path / "a.md").write_text("One two three. Four five!", encoding="utf-8")
    (tmp_path / "b.txt").write_text('"Hi," she said.', encoding="utf-8")
    out = tmp_path / "wc.csv"
    assert report(tmp_path, out) == 2
    text = out.read_text(encoding="utf-8-sig")
    assert "a.md" in text and "b.txt" in text and text.splitlines()[0].startswith("file,words")
''',
    "README.md": "# wordcount\n\nCounts words per chapter file. Run: `python main.py --folder <dir> --out wc.csv`. No model needed.\n",
}


def fake_proposal(prompt: str) -> dict:
    """Append a comment line to the first current file shown in the prompt (a harmless, test-passing change),
    or create a doc file if no file was shown."""
    m = re.search(r"===== (\S+) =====\n(.*?)(?=\n===== |\n\nREFERENCE:)", prompt, re.S)
    if m:
        rel, body = m.group(1), m.group(2)
        new = body.rstrip("\n") + "\n\n# touched by a self-proposal\n"
        return {"files": {rel: new}, "rationale": "Adds a harmless marker comment to demonstrate the proposal path.", "tests_added": []}
    return {"files": {"docs/SELF_NOTE.md": "# Self note\n\nProposed by the self-programmer.\n"}, "rationale": "doc only", "tests_added": []}
