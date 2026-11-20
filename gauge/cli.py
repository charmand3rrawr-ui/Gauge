"""Gauge command line."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import __version__
from .config import GaugeConfig
from .db import Database

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s",
                        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)])
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class Ctx:
    def __init__(self, data_dir: Optional[str]):
        self.cfg = GaugeConfig.load(data_dir)
        self.cfg.ensure_dirs()
        self.db = Database(self.cfg.db_path)


@click.group()
@click.option("--data-dir", envvar="GAUGE_DATA_DIR", default=None, help="Where Gauge keeps everything (default ./data).")
@click.option("-v", "--verbose", is_flag=True)
@click.version_option(__version__, prog_name="gauge")
@click.pass_context
def main(ctx: click.Context, data_dir: Optional[str], verbose: bool) -> None:
    """Gauge: learn what makes web novels addictive, then write them. Local, open source."""
    _setup_logging(verbose)
    ctx.obj = Ctx(data_dir)


# ---------------------------------------------------------------------------- setup
@main.command()
@click.option("--provider", type=click.Choice(["ollama", "openai", "fake"]), default="ollama")
@click.option("--model", default="llama3.1:8b", show_default=True)
@click.option("--base-url", default=None, help="LLM server URL (default depends on provider).")
@click.pass_obj
def init(c: Ctx, provider: str, model: str, base_url: Optional[str]) -> None:
    """Create the data directory and config.json."""
    c.cfg.llm.provider = provider
    c.cfg.llm.model = model
    if base_url:
        c.cfg.llm.base_url = base_url
    elif provider == "openai":
        c.cfg.llm.base_url = "http://localhost:1234/v1"  # LM Studio default
    path = c.cfg.save()
    console.print(f"[green]Initialised[/] {c.cfg.root.resolve()}  (config: {path})")
    console.print("Next: [bold]gauge doctor[/], then [bold]gauge discover royalroad best-rated --pages 2[/] and [bold]gauge ingest[/].")


@main.command()
@click.pass_obj
def doctor(c: Ctx) -> None:
    """Check the LLM server, embeddings, data directory and corpus."""
    from .llm import make_embedder, make_llm
    from .report import corpus_summary

    llm = make_llm(c.cfg.llm)
    ok = llm.available()
    console.print(f"LLM        : {llm.describe()}  {'[green]reachable[/]' if ok else '[red]NOT reachable[/]'}")
    if not ok and c.cfg.llm.provider == "ollama":
        console.print("             Install Ollama from https://ollama.com/download and run: ollama pull " + c.cfg.llm.model)
    if ok and hasattr(llm, "models"):
        have = llm.models()  # type: ignore[attr-defined]
        if have and c.cfg.llm.model not in have and not any(m.startswith(c.cfg.llm.model) for m in have):
            console.print(f"             [yellow]model {c.cfg.llm.model} not pulled[/]; run: ollama pull {c.cfg.llm.model}")
    emb = make_embedder(c.cfg.llm)
    console.print(f"Embeddings : {emb.describe()}")
    console.print(f"Data dir   : {c.cfg.root.resolve()}")
    console.print(f"Corpus     : {corpus_summary(c.db)}")
    try:
        import sklearn  # noqa: F401

        console.print("scikit-learn: installed (gradient boosting enabled)")
    except ImportError:
        console.print("scikit-learn: not installed (ridge only; `pip install scikit-learn` for a stronger model)")


# ---------------------------------------------------------------------------- ingest
@main.command()
@click.argument("source", type=click.Choice(["royalroad", "scribblehub"]))
@click.argument("listing", default="best-rated")
@click.option("--pages", default=1, show_default=True)
@click.pass_obj
def discover(c: Ctx, source: str, listing: str, pages: int) -> None:
    """Follow novels from a site listing (royalroad: best-rated, trending, rising-stars, popular, new;
    scribblehub: popular, rating, favorites, trending)."""
    from .ingest import IngestRunner

    urls = IngestRunner(c.cfg, c.db).discover(source, listing, pages)
    console.print(f"[green]Following {len(urls)} novels[/] from {source}/{listing}. Run [bold]gauge ingest[/].")


@main.command()
@click.argument("url")
@click.pass_obj
def follow(c: Ctx, url: str) -> None:
    """Follow one novel URL so every `gauge ingest` picks up its new chapters."""
    from .ingest import IngestRunner

    src = IngestRunner(c.cfg, c.db).follow(url)
    console.print(f"[green]Following[/] {url} ({src})")


@main.command()
@click.option("--url", default=None, help="Ingest just this novel instead of all follows.")
@click.option("--max-novels", default=None, type=int)
@click.option("--max-chapters", default=None, type=int, help="Per-novel chapter cap for this run.")
@click.pass_obj
def ingest(c: Ctx, url: Optional[str], max_novels: Optional[int], max_chapters: Optional[int]) -> None:
    """Fetch followed novels and any chapters not yet stored (incremental; safe to re-run)."""
    from .ingest import IngestRunner

    r = IngestRunner(c.cfg, c.db)
    res = r.ingest_url(url, max_chapters) if url else r.run(max_novels)
    console.print(f"[green]Ingested[/] {res}")


@main.command(name="import")
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def import_(c: Ctx, path: str) -> None:
    """Import a local novel: a folder of chapter .txt/.md files, a single file, or an .epub.
    Optional novel.json sidecar supplies metadata and engagement numbers."""
    from .ingest import IngestRunner

    n = IngestRunner(c.cfg, c.db).import_local(path)
    console.print(f"[green]Imported[/] {n.title} ({c.db.chapter_count(n.key)} chapters) as {n.key}")


# ---------------------------------------------------------------------------- learn
@main.command()
@click.option("--all", "redo", is_flag=True, help="Recompute features for every chapter.")
@click.pass_obj
def analyze(c: Ctx, redo: bool) -> None:
    """Compute hook / pacing / payoff / cliffhanger features for every chapter and novel."""
    from .learn import Trainer

    res = Trainer(c.cfg, c.db).analyze(only_new=not redo)
    console.print(f"[green]Analyzed[/] {res}")


@main.command()
@click.option("--target", default=None, type=click.Choice(["log_followers", "log_views", "log_avg_views", "rating", "log_favorites"]))
@click.pass_obj
def train(c: Ctx, target: Optional[str]) -> None:
    """Full learning cycle: features -> engagement model -> exemplars -> style guide -> datasets."""
    from .learn import Trainer

    if target:
        c.cfg.learn.target = target
        c.cfg.save()
    t = Trainer(c.cfg, c.db)
    res = t.cycle()
    console.print_json(json.dumps(res, default=str))
    m = t.load_model()
    if m:
        console.print(m.explain())
        console.print(f"\nStyle guide: {t.guide_path}\nOllama Modelfile: {c.cfg.models_dir / 'Modelfile'}")


@main.command()
@click.option("--top", default=15, show_default=True)
@click.option("--novel", default=None, help="Explain one novel by key, e.g. royalroad:12345")
@click.pass_obj
def report(c: Ctx, top: int, novel: Optional[str]) -> None:
    """What has Gauge learned? Corpus stats, feature importances, leaderboard."""
    from .learn import Trainer
    from .report import corpus_summary, explain_novel, leaderboard

    t = Trainer(c.cfg, c.db)
    m = t.load_model()
    if novel:
        console.print(explain_novel(c.db, m, novel))
        return
    console.print(f"Corpus: {corpus_summary(c.db)}")
    if m:
        console.print("\n[bold]Engagement model[/]")
        console.print(m.explain())
    else:
        console.print("\n[yellow]No model yet.[/] Ingest labelled novels and run `gauge train`.")
    rows = leaderboard(c.db, c.cfg.learn.target, top)
    if rows:
        table = Table(title=f"Top novels by {c.cfg.learn.target}")
        for k in rows[0].keys():
            table.add_column(str(k))
        for r in rows:
            table.add_row(*[str(v) for v in r.values()])
        console.print(table)
    runs = c.db.recent_runs(5)
    if runs:
        console.print("\n[bold]Recent runs[/]")
        for r in runs:
            console.print(f"  {r['finished_at'][:19]}  {r['kind']:<13} {r['summary']}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def score(c: Ctx, path: str) -> None:
    """Score any chapter text file against what Gauge has learned (hook, cliffhanger, percentiles)."""
    from .features import chapter_features
    from .learn import Trainer
    from .llm import FakeLLM
    from .models import ChapterRecord
    from .report import compare_to_corpus
    from .write.critic import Critic

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    feats = chapter_features(ChapterRecord(novel_key="file", index=1, title=Path(path).stem, text=text))
    t = Trainer(c.cfg, c.db)
    crit = Critic(FakeLLM(), t.load_model())
    measured, issues = crit._measured(feats)
    console.print(f"[bold]{path}[/]  words={int(feats['words'])}  hook={feats['hook']:.2f}  cliff={feats['cliff']:.2f}  "
                  f"dialogue={feats['dialogue_ratio']:.0%}  measured engagement score={measured:.2f}")
    for i in issues:
        console.print(f"  - {i}")
    comp = compare_to_corpus(feats, c.db.features("chapter"))
    if comp:
        table = Table(title="vs. ingested corpus")
        for k in comp[0]:
            table.add_column(k)
        for r in comp:
            table.add_row(*[str(v) for v in r.values()])
        console.print(table)


# ---------------------------------------------------------------------------- write
@main.command()
@click.argument("name")
@click.option("--idea", required=True, help="Your seed idea in a sentence or a paragraph.")
@click.option("--genre", "genres", multiple=True, help="Repeatable, e.g. --genre LitRPG --genre Progression")
@click.option("--words", default=None, type=int, help="Target words per chapter.")
@click.pass_obj
def new(c: Ctx, name: str, idea: str, genres: tuple, words: Optional[int]) -> None:
    """Create a writing project: bible + first outline."""
    from .write import WritePipeline

    pipe = WritePipeline(c.cfg, c.db)
    if not pipe.llm.available():
        console.print(f"[red]LLM not reachable[/] ({pipe.llm.describe()}). Run `gauge doctor`.")
        sys.exit(1)
    p = pipe.new_project(name, idea, list(genres) or ["Fantasy"], words)
    console.print(f"[green]Created[/] {p.dir}\n")
    console.print(p.bible_markdown())
    console.print(p.outline_markdown())


@main.command()
@click.argument("name")
@click.option("--chapters", default=1, show_default=True)
@click.pass_obj
def write(c: Ctx, name: str, chapters: int) -> None:
    """Write the next chapter(s) of a project: draft -> score -> revise -> save."""
    from .write import WritePipeline

    pipe = WritePipeline(c.cfg, c.db)
    if not pipe.llm.available():
        console.print(f"[red]LLM not reachable[/] ({pipe.llm.describe()}). Run `gauge doctor`.")
        sys.exit(1)
    p = pipe.open_project(name)
    for res in pipe.write_many(p, chapters):
        console.print(f"[green]Chapter {res.number}[/] {res.title}  score={res.score:.2f} "
                      f"(measured {res.critique.measured:.2f} / judged {res.critique.judged:.2f}) "
                      f"revisions={res.revisions}\n  -> {res.path}")
        if res.critique.notes:
            console.print(f"  editor: {res.critique.notes}")


@main.command()
@click.argument("name", required=False)
@click.pass_obj
def projects(c: Ctx, name: Optional[str]) -> None:
    """List projects, or show one project's status."""
    from .write.project import Project

    if not name:
        names = Project.list(c.cfg.projects_dir)
        console.print("\n".join(names) if names else "(no projects yet; `gauge new NAME --idea ...`)")
        return
    from .write import WritePipeline

    pipe = WritePipeline(c.cfg, c.db)
    console.print_json(json.dumps(pipe.status(pipe.open_project(name)), default=str))


# ---------------------------------------------------------------------------- daemon
@main.command()
@click.option("--once", is_flag=True, help="Run one cycle and exit (good for Windows Task Scheduler).")
@click.option("--write/--no-write", "do_write", default=None, help="Also write a chapter of the active project.")
@click.option("--project", default=None, help="Set the active project for writing.")
@click.option("--every", default=None, type=int, help="Cycle length in minutes.")
@click.pass_obj
def daemon(c: Ctx, once: bool, do_write: Optional[bool], project: Optional[str], every: Optional[int]) -> None:
    """Continuous learning loop: ingest new chapters -> analyze -> retrain -> (write)."""
    from .daemon import Daemon

    changed = False
    if project:
        c.cfg.daemon.active_project = project
        changed = True
    if every:
        c.cfg.daemon.cycle_minutes = every
        changed = True
    if do_write is not None:
        c.cfg.daemon.write_after_learning = do_write
        changed = True
    if changed:
        c.cfg.save()
    d = Daemon(c.cfg, c.db)
    if once:
        console.print_json(json.dumps(d.once(do_write), default=str))
    else:
        console.print(f"Daemon running every {c.cfg.daemon.cycle_minutes} min. Ctrl+C to stop.")
        d.run_forever()


@main.command()
@click.pass_obj
def config(c: Ctx) -> None:
    """Print the active configuration."""
    console.print_json(c.cfg.model_dump_json())
    console.print(f"\nEdit: {c.cfg.config_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
