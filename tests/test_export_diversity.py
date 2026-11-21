"""Roadmap items 77 (serial-platform export) and 84 (newsdesk source diversity)."""

import pytest
from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.llm import FakeLLM
from gauge.newsdesk.commentary import source_diversity, write_commentary
from gauge.newsdesk.trends import SourceItem
from gauge.site.export import FORMATS, export_project, format_chapter


def test_format_chapter_per_format():
    assert format_chapter(1, "The Seal", "a\n\nb", "markdown").startswith("# Chapter 1: The Seal")
    assert "[b]" in format_chapter(1, "T", "a", "bbcode")
    assert "<h1>" in format_chapter(1, "T", "a", "html") and "<p>a</p>" in format_chapter(1, "T", "a", "html")
    assert "=====" in format_chapter(1, "The Seal", "a", "text")
    with pytest.raises(ValueError):
        format_chapter(1, "T", "a", "nope")


def test_export_project_all_formats(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.ensure_dirs()
    from gauge.write.project import Project
    p = Project.create(cfg.projects_dir, "Ten Seconds", "idea", ["Sci-fi"], 300)
    p.write_chapter(1, "The Seal", "She ran.\n\nThe door slammed.", {"title": "The Seal"})
    p.write_chapter(2, "The Chase", "He followed.", {"title": "The Chase"})
    for fmt in FORMATS:
        files = export_project(p, fmt, tmp_path / fmt)
        assert len(files) == 3  # two chapters + a combined file
        combined = next(f for f in files if "ten-seconds" in f.name)
        assert "The Seal" in combined.read_text() and "The Chase" in combined.read_text()


def test_source_diversity():
    one = [SourceItem("a", "", "https://cnn.com/1"), SourceItem("b", "", "https://cnn.com/2"),
           SourceItem("c", "", "https://www.cnn.com/3"), SourceItem("d", "", "https://cnn.com/4")]
    share, flags = source_diversity(one)
    assert share == 1.0 and flags and "cnn.com" in flags[0]
    mix = [SourceItem("a", "", "https://cnn.com/1"), SourceItem("b", "", "https://bbc.co.uk/2"),
           SourceItem("c", "", "https://reuters.com/3")]
    assert source_diversity(mix)[1] == []
    assert source_diversity([])[1] == []  # nothing to judge


def test_commentary_carries_diversity_flag():
    srcs = [SourceItem("a", "", "https://cnn.com/1"), SourceItem("b", "", "https://cnn.com/2"),
            SourceItem("c", "", "https://cnn.com/3"), SourceItem("d", "", "https://cnn.com/4")]
    c = write_commentary(FakeLLM(), "A topic", srcs, "Jane Doe")
    assert c.diversity_flags and not c.clean  # one-outlet sourcing keeps it out of auto-publish


def test_cli_export(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.ensure_dirs()
    cfg.save()
    from gauge.write.project import Project
    p = Project.create(cfg.projects_dir, "Ten Seconds", "idea", ["Sci-fi"], 300)
    p.write_chapter(1, "The Seal", "She ran.", {"title": "The Seal"})
    out = tmp_path / "out"
    r = CliRunner(env={"COLUMNS": "200"}).invoke(main, ["--data-dir", data, "export", "Ten Seconds", "--format", "bbcode", "--out", str(out)])
    assert r.exit_code == 0 and "exported" in r.output and list(out.glob("*.txt"))
