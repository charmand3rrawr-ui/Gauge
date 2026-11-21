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
@click.option("--provider", type=click.Choice(["ollama", "openai", "native", "fake"]), default="ollama")
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
    elif provider == "native":
        c.cfg.llm.model = "gaugelm"
        c.cfg.llm.fast_model = ""
        c.cfg.llm.num_ctx = c.cfg.native.block_size
        c.cfg.llm.native_dir = str(c.cfg.native_dir)
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
    fast = make_llm(c.cfg.llm, "fast")
    console.print(f"Fast model : {fast.model if c.cfg.llm.fast_model else '(same as main; set llm.fast_model to a 3B model for speed)'}")
    console.print(f"Execution  : num_ctx={c.cfg.llm.num_ctx} parallel={c.cfg.llm.parallel} retries={c.cfg.llm.retries} "
                  f"keep_alive={c.cfg.llm.keep_alive} structured_output={c.cfg.llm.structured_output}")
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
@click.option("--library/--novel", "library", default=None,
              help="Force treating PATH as a library of novels, or as one novel. Default: auto-detect.")
@click.option("--watch", is_flag=True, help="Also keep re-scanning this path on every `gauge ingest` / daemon cycle.")
@click.option("--title", default=None)
@click.option("--author", default=None)
@click.option("--genre", "genres", multiple=True, help="Repeatable.")
@click.option("--followers", type=int, default=None)
@click.option("--views", type=int, default=None)
@click.option("--rating", type=float, default=None, help="Site rating, e.g. 4.5")
@click.option("--stars", type=float, default=None, help="YOUR 1-5 verdict (user_rating target).")
@click.pass_obj
def import_(c: Ctx, path: str, library: Optional[bool], watch: bool, title: Optional[str], author: Optional[str],
            genres: tuple, followers: Optional[int], views: Optional[int], rating: Optional[float],
            stars: Optional[float]) -> None:
    """Import a novel or a whole library from disk.

    PATH may be a folder of chapter files (.txt .md .html .docx), a single book (.txt .md .epub .docx),
    or a library folder containing many of those. Metadata comes from novel.json / library.csv /
    EPUB+DOCX metadata / file names. Re-running only touches new or edited chapters."""
    from .ingest import IngestRunner

    override = {k: v for k, v in {"title": title, "author": author, "genres": list(genres) or None,
                                   "followers": followers, "views": views, "rating": rating,
                                   "user_rating": stars}.items() if v is not None}
    r = IngestRunner(c.cfg, c.db)
    results = r.import_local(path, library=library, meta_override=override or None)
    if not results:
        console.print("[yellow]Nothing imported.[/] Is this a folder of chapter files or a book file?")
        return
    table = Table(title=f"Imported from {path}")
    for col in ("key", "title", "chapters", "new", "changed", "removed", "labelled"):
        table.add_column(col)
    for res in results:
        table.add_row(*[str(res[k]) for k in ("key", "title", "chapters", "new", "changed", "removed", "labelled")])
    console.print(table)
    for res in results:
        for w in res["warnings"]:
            console.print(f"  [dim]{res['title']}: {w}[/]")
    unlabelled = sum(1 for res in results if not res["labelled"])
    if unlabelled:
        console.print(f"[dim]{unlabelled} novel(s) have no engagement numbers; they still feed exemplars. "
                      f"Give them your own verdict with `gauge rate KEY --stars N` to train on your taste.[/]")
    if watch:
        console.print(f"[green]Watching[/] {r.watch(path)}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def watch(c: Ctx, path: str) -> None:
    """Keep a local novel or library folder in sync: every ingest / daemon cycle re-scans it."""
    from .ingest import IngestRunner

    console.print(f"[green]Watching[/] {IngestRunner(c.cfg, c.db).watch(path)}  (run `gauge ingest` to scan now)")


@main.command()
@click.argument("key")
@click.option("--stars", type=float, required=True, help="Your 1-5 verdict; use 0 to clear.")
@click.pass_obj
def rate(c: Ctx, key: str, stars: float) -> None:
    """Give a novel your own 1-5 rating. Train on it with `gauge train --target user_rating`."""
    if not (0 <= stars <= 5):
        raise click.BadParameter("stars must be between 0 and 5")
    ok = c.db.set_user_rating(key, None if stars == 0 else stars)
    console.print(f"[green]Rated[/] {key} = {stars}" if ok else f"[red]unknown novel[/] {key}; see `gauge novels`")


@main.command()
@click.option("--source", default=None, help="royalroad | scribblehub | local")
@click.pass_obj
def novels(c: Ctx, source: Optional[str]) -> None:
    """List ingested novels with their keys and labels."""
    rows = c.db.novels(source)
    if not rows:
        console.print("(no novels yet)")
        return
    table = Table(title=f"{len(rows)} novels")
    for col in ("key", "title", "author", "chapters", "followers", "rating", "stars"):
        table.add_column(col)
    for n in rows:
        table.add_row(n.key, n.title[:40], n.author[:20], str(c.db.chapter_count(n.key)),
                      str(n.followers or ""), str(n.rating or ""), str(n.user_rating or ""))
    console.print(table)


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
@click.option("--target", default=None, type=click.Choice(["log_followers", "log_views", "log_avg_views", "rating", "log_favorites", "user_rating"]))
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
@click.option("--project", default=None, help="Show the per-project model learned from your readers.")
@click.pass_obj
def report(c: Ctx, top: int, novel: Optional[str], project: Optional[str]) -> None:
    """What has Gauge learned? Corpus stats, feature importances, leaderboard."""
    from .learn import Trainer
    from .report import corpus_summary, explain_novel, leaderboard

    t = Trainer(c.cfg, c.db)
    m = t.load_model()
    if novel:
        console.print(explain_novel(c.db, m, novel))
        return
    if project:
        from .learn.project_model import ProjectModel
        from .write.project import Project

        if not Project.exists(c.cfg.projects_dir, project):
            console.print(f"[red]no project named {project!r}[/]")
            sys.exit(1)
        pm = ProjectModel(Project(c.cfg.projects_dir, project).dir)
        console.print(f"[bold]Your readers' model for {project}[/]\n{pm.explain()}")
        fb = c.db.feedback(project)
        console.print(f"labelled chapters: {len(fb)}")
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
    from .learn.ledger import evaluate

    led = evaluate(c.db, c.cfg.learn.target, older_than_days=7)
    if led.get("pairs"):
        console.print(f"\n[bold]Prediction ledger[/] (predictions >= 7 days old vs today's numbers): {led}")
    calls = c.db.call_summary(limit_days=30)
    if calls:
        table = Table(title="LLM time by purpose (last 30 days)")
        for col in ("purpose", "model", "calls", "seconds", "avg_seconds", "prompt_tokens", "output_tokens", "failed", "retries"):
            table.add_column(col)
        for r in calls:
            table.add_row(*[str(r[k]) for k in ("purpose", "model", "calls", "seconds", "avg_seconds", "prompt_tokens",
                                                 "output_tokens", "failed", "retries")])
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
@click.option("--idea", default=None, help="Your seed idea in a sentence or a paragraph.")
@click.option("--premise-file", type=click.Path(exists=True), default=None,
              help="A premise document (.md/.txt): names, rules, arc. Everything in it is canon.")
@click.option("--continue-from", "continue_from", type=click.Path(exists=True), default=None,
              help="An existing manuscript (folder, .txt/.md/.docx/.epub). Gauge continues it from the next chapter.")
@click.option("--genre", "genres", multiple=True, help="Repeatable, e.g. --genre LitRPG --genre Progression")
@click.option("--words", default=None, type=int, help="Target words per chapter.")
@click.pass_obj
def new(c: Ctx, name: str, idea: Optional[str], premise_file: Optional[str], continue_from: Optional[str], genres: tuple,
        words: Optional[int]) -> None:
    """Create a writing project from an idea, a premise document and/or an existing manuscript."""
    from .write import WritePipeline

    if not (idea or premise_file or continue_from):
        raise click.UsageError("give --idea, --premise-file or --continue-from (any combination)")
    pipe = WritePipeline(c.cfg, c.db)
    if not pipe.llm.available():
        console.print(f"[red]LLM not reachable[/] ({pipe.llm.describe()}). Run `gauge doctor`.")
        sys.exit(1)
    premise = Path(premise_file).read_text(encoding="utf-8", errors="replace") if premise_file else ""
    p = pipe.new_project(name, idea or "", list(genres) or ["Fantasy"], words, premise_doc=premise, continue_from=continue_from)
    console.print(f"[green]Created[/] {p.dir}" + (f"  (continuing from chapter {p.state.written + 1})" if continue_from else "") + "\n")
    console.print(p.bible_markdown())
    console.print(p.outline_markdown())
    w = pipe.world(p)
    if w is not None:
        console.print(f"World: {w.name}  rules={len(w.physics_rules) + len(w.power_rules) + len(w.social_rules)}  places={len(w.locations)}")


@main.command()
@click.argument("premise")
@click.option("--name", default=None, help="Project name (default: first words of the premise).")
@click.option("--chapters", default=3, show_default=True)
@click.option("--genre", "genres", multiple=True)
@click.option("--words", default=None, type=int)
@click.pass_obj
def start(c: Ctx, premise: str, name: Optional[str], chapters: int, genres: tuple, words: Optional[int]) -> None:
    """One command from premise to chapters: PREMISE may be text or a path to a premise file."""
    from .write import WritePipeline

    pipe = WritePipeline(c.cfg, c.db)
    if not pipe.llm.available():
        console.print(f"[red]LLM not reachable[/] ({pipe.llm.describe()}). Run `gauge doctor`.")
        sys.exit(1)
    text = Path(premise).read_text(encoding="utf-8", errors="replace") if Path(premise).exists() else premise
    from .write.project import title_from_text

    title = name or title_from_text(text)
    p = pipe.new_project(title, text if len(text) < 300 else "", list(genres) or ["Fantasy"], words,
                         premise_doc=text if len(text) >= 300 else "")
    console.print(f"[green]Created[/] {p.state.name}: {p.state.bible.premise}\n")
    for res in pipe.write_many(p, chapters):
        console.print(f"[green]Chapter {res.number}[/] {res.title}  score={res.score:.2f}  -> {res.path}")
    console.print(f"\nContinue with: gauge write \"{p.state.name}\" --chapters 5")


@main.command()
@click.option("--chapters", "n", default=None, type=int, help="How many chapters to absorb this run.")
@click.pass_obj
def absorb(c: Ctx, n: Optional[int]) -> None:
    """Read the strongest unread chapters and extract reusable craft techniques (needs the LLM)."""
    from .learn import Trainer
    from .learn.techniques import Absorber
    from .llm import make_llm

    fast = make_llm(c.cfg.llm, "fast")
    if not fast.available():
        console.print(f"[red]LLM not reachable[/] ({fast.describe()}).")
        sys.exit(1)
    t = Trainer(c.cfg, c.db)
    a = Absorber(c.db, fast, t.embedder, c.cfg.models_dir, c.cfg.learn.target)
    res = a.run(n or c.cfg.learn.absorb_per_cycle)
    console.print(f"[green]Absorbed[/] {res}")
    kinds = a.library.kinds_count()
    if kinds:
        console.print("library by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
        for t_ in a.library.items[-3:]:
            console.print(f"  - {t_.card()}")


@main.command()
@click.argument("name")
@click.option("--chapters", default=1, show_default=True)
@click.option("--fresh", is_flag=True, help="Ignore a checkpointed draft and generate from scratch.")
@click.pass_obj
def write(c: Ctx, name: str, chapters: int, fresh: bool) -> None:
    """Write the next chapter(s) of a project: draft -> score -> revise -> save."""
    from .write import WritePipeline

    pipe = WritePipeline(c.cfg, c.db)
    if not pipe.llm.available():
        console.print(f"[red]LLM not reachable[/] ({pipe.llm.describe()}). Run `gauge doctor`.")
        sys.exit(1)
    p = pipe.open_project(name)
    for res in pipe.write_many(p, chapters, fresh=fresh):
        console.print(f"[green]Chapter {res.number}[/] {res.title}  score={res.score:.2f} "
                      f"(measured {res.critique.measured:.2f} / judged {res.critique.judged:.2f}) "
                      f"revisions={res.revisions}{'  [resumed from checkpoint]' if res.resumed else ''}\n  -> {res.path}")
        steps = "  ".join(f"{k} {v:.0f}s" for k, v in res.timings.items() if k != "total")
        console.print(f"  timing: {steps}  | total {res.timings.get('total', 0):.0f}s, "
                      f"{res.llm_calls} LLM calls ({res.llm_seconds:.0f}s in model)")
        if res.cast:
            console.print(f"  cast: {', '.join(res.cast)}  in-character consistency="
                          f"{sum(r.consistency for r in res.character_reviews) / len(res.character_reviews):.2f}"
                          if res.character_reviews else f"  cast: {', '.join(res.cast)}")
        for r in res.character_reviews:
            for i in r.issues[:1]:
                console.print(f"  {r.name}: \"{i.quote}\" - {i.problem}")
        extras = []
        if res.world_review is not None:
            extras.append(f"world {res.world_review.consistency:.2f}")
        if res.novelty is not None:
            extras.append(f"novelty {res.novelty.score:.2f}")
        if extras:
            console.print("  " + "  ".join(extras))
        if res.novelty and res.novelty.overused:
            console.print("  worn phrases: " + ", ".join(f'"{p}"' for p in res.novelty.overused[:4]))
        if res.agent_reviews:
            console.print("  room: " + "  ".join(f"{r.agent} {r.score:.2f}" for r in res.agent_reviews))
            for r in res.agent_reviews:
                for i in r.issues[:1]:
                    console.print(f"    {r.agent}: {i[:140]}")
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


# ---------------------------------------------------------------------------- characters
@main.group()
def character() -> None:
    """Characters as independent agents: personality matrix, voice, memory, relationships."""


def _roster(c: Ctx, project: str):
    from .write.characters import Roster
    from .write.project import Project

    if not Project.exists(c.cfg.projects_dir, project):
        console.print(f"[red]no project named {project!r}[/]")
        sys.exit(1)
    p = Project(c.cfg.projects_dir, project)
    return p, Roster.load(p.dir)


def _pipeline_or_exit(c: Ctx):
    from .write import WritePipeline

    pipe = WritePipeline(c.cfg, c.db)
    if not pipe.llm.available():
        console.print(f"[red]LLM not reachable[/] ({pipe.llm.describe()}). Run `gauge doctor`.")
        sys.exit(1)
    return pipe


@character.command("new")
@click.argument("project")
@click.argument("name")
@click.option("--concept", required=True, help="Who they are, in a line or a paragraph.")
@click.option("--role", type=click.Choice(["protagonist", "antagonist", "supporting", "minor"]), default="supporting")
@click.option("--seed", type=int, default=None, help="Roll the trait matrix from this seed, then let the LLM flesh it out.")
@click.pass_obj
def character_new(c: Ctx, project: str, name: str, concept: str, role: str, seed: Optional[int]) -> None:
    """Create a character with a full personality matrix, voice and relationships."""
    pipe = _pipeline_or_exit(c)
    p = pipe.open_project(project)
    ch = pipe.create_character(p, name, concept, role, seed)
    console.print(ch.sheet_markdown())
    console.print(f"[green]Saved[/] {p.dir / 'characters' / (ch.name.lower().replace(' ', '-') + '.json')}")


@character.command("roll")
@click.argument("project")
@click.argument("name")
@click.option("--role", default="supporting")
@click.option("--seed", type=int, default=None)
@click.option("--concept", default="")
@click.pass_obj
def character_roll(c: Ctx, project: str, name: str, role: str, seed: Optional[int], concept: str) -> None:
    """Create a character from a random trait matrix, no LLM needed. Edit the JSON or `set` traits later."""
    from .write.characters import CharacterCreator

    p, roster = _roster(c, project)
    ch = CharacterCreator(None).roll(name, role, seed, concept)  # type: ignore[arg-type]
    roster.add(ch)
    roster.save()
    console.print(ch.sheet_markdown())


@character.command("sync")
@click.argument("project")
@click.pass_obj
def character_sync(c: Ctx, project: str) -> None:
    """Create sheets for every character named in the bible that does not have one yet."""
    pipe = _pipeline_or_exit(c)
    created = pipe.sync_characters(pipe.open_project(project))
    console.print(f"[green]Created[/] {', '.join(created)}" if created else "Everyone in the bible already has a sheet.")


@character.command("list")
@click.argument("project")
@click.pass_obj
def character_list(c: Ctx, project: str) -> None:
    """List a project's characters with role, mood and last appearance."""
    _, roster = _roster(c, project)
    if not roster.characters:
        console.print("(no characters yet; `gauge character new PROJECT NAME --concept ...` or `gauge character sync PROJECT`)")
        return
    table = Table(title=f"{project}: {len(roster.characters)} characters")
    for col in ("name", "role", "mood", "goal", "last seen", "wants"):
        table.add_column(col)
    for ch in roster.ordered():
        table.add_row(ch.name, ch.role, ch.state.mood, ch.state.goal[:40], str(ch.state.last_seen_chapter or "-"), ch.matrix.wants[:40])
    console.print(table)


@character.command("show")
@click.argument("project")
@click.argument("name")
@click.pass_obj
def character_show(c: Ctx, project: str, name: str) -> None:
    """Print one character's full sheet: matrix, tendencies, voice, state, relationships, memories."""
    _, roster = _roster(c, project)
    ch = roster.get(name)
    if ch is None:
        console.print(f"[red]no character {name!r}[/]; have: {', '.join(roster.names())}")
        sys.exit(1)
    console.print(ch.sheet_markdown())


@character.command("set")
@click.argument("project")
@click.argument("name")
@click.option("--trait", "traits", multiple=True, help="axis=value, e.g. --trait courage=0.9 (repeatable)")
@click.option("--alias", "aliases", multiple=True)
@click.option("--mood", default=None)
@click.option("--goal", default=None)
@click.option("--role", default=None)
@click.pass_obj
def character_set(c: Ctx, project: str, name: str, traits: tuple, aliases: tuple, mood: Optional[str],
                  goal: Optional[str], role: Optional[str]) -> None:
    """Adjust a character by hand: trait axes, aliases, mood, goal, role."""
    from .write.characters import TRAIT_AXES

    _, roster = _roster(c, project)
    ch = roster.get(name)
    if ch is None:
        console.print(f"[red]no character {name!r}[/]")
        sys.exit(1)
    for t in traits:
        axis, _, val = t.partition("=")
        if axis not in TRAIT_AXES:
            raise click.BadParameter(f"unknown axis {axis!r}; axes: {', '.join(TRAIT_AXES)}")
        setattr(ch.matrix, axis, max(0.0, min(1.0, float(val))))
    if aliases:
        ch.aliases = sorted(set(ch.aliases) | set(aliases))
    if mood:
        ch.state.mood = mood
    if goal:
        ch.state.goal = goal
    if role:
        ch.role = role
    roster.save()
    console.print(ch.sheet_markdown())


@character.command("remove")
@click.argument("project")
@click.argument("name")
@click.pass_obj
def character_remove(c: Ctx, project: str, name: str) -> None:
    """Delete a character sheet."""
    _, roster = _roster(c, project)
    ok = roster.remove(name)
    if ok:
        roster.save()
    console.print(f"[green]Removed[/] {name}" if ok else f"[red]no character {name!r}[/]")


@character.command("chat")
@click.argument("project")
@click.argument("name")
@click.option("--ask", default=None, help="One question; omit for an interactive session.")
@click.pass_obj
def character_chat(c: Ctx, project: str, name: str, ask: Optional[str]) -> None:
    """Talk to a character directly. They answer as themselves, with everything they remember."""
    from .write.characters import CharacterAgent

    pipe = _pipeline_or_exit(c)
    p = pipe.open_project(project)
    roster = pipe.roster(p)
    ch = roster.get(name)
    if ch is None:
        console.print(f"[red]no character {name!r}[/]; have: {', '.join(roster.names())}")
        sys.exit(1)
    agent = CharacterAgent(pipe.llm, ch)
    history: list = []
    if ask:
        console.print(f"[bold]{ch.name}:[/] {agent.chat(ask)}")
        return
    console.print(f"Talking to [bold]{ch.name}[/]. Empty line to stop.")
    while True:
        q = click.prompt("You", default="", show_default=False)
        if not q.strip():
            break
        a = agent.chat(q, history)
        history.append((q, a))
        console.print(f"[bold]{ch.name}:[/] {a}")


@character.command("brief")
@click.argument("project")
@click.argument("name")
@click.option("--chapter", type=int, default=None, help="Chapter number from the outline (default: next).")
@click.pass_obj
def character_brief(c: Ctx, project: str, name: str, chapter: Optional[int]) -> None:
    """Preview how a character would approach a planned chapter (what the drafter will be told)."""
    from .write.characters import CharacterAgent

    pipe = _pipeline_or_exit(c)
    p = pipe.open_project(project)
    roster = pipe.roster(p)
    ch = roster.get(name)
    if ch is None:
        console.print(f"[red]no character {name!r}[/]")
        sys.exit(1)
    n = chapter or p.next_chapter_number()
    plan = pipe.planner.ensure_plan(p, n)
    prev = p.read_chapter(n - 1) if n > 1 else None
    scene = {"number": n, "title": plan.title, "goal": plan.goal, "hook": plan.hook, "turn": plan.turn,
             "cliffhanger": plan.cliffhanger, "beats": plan.beats, "summary": p.state.running_summary,
             "prev_tail": " ".join(prev.split()[-250:]) if prev else "", "others": [o for o in roster.names() if o != ch.name]}
    console.print(CharacterAgent(pipe.llm, ch).brief(scene).render())


# ---------------------------------------------------------------------------- bench
@main.command()
@click.option("--tokens", default=200, show_default=True, help="Tokens to generate per measurement.")
@click.option("--characters", "n_chars", default=4, show_default=True, help="Cast size to assume for the estimate.")
@click.pass_obj
def bench(c: Ctx, tokens: int, n_chars: int) -> None:
    """Measure your models' speed and estimate minutes per chapter before you commit to a model."""
    from .llm import make_llm
    from .write.drafter import output_tokens_for

    main_llm = make_llm(c.cfg.llm, "main")
    if not main_llm.available():
        console.print(f"[red]LLM not reachable[/] ({main_llm.describe()}).")
        sys.exit(1)
    fast_llm = make_llm(c.cfg.llm, "fast")
    console.print(f"Measuring [bold]{main_llm.model}[/] (prose) ...")
    m = main_llm.throughput(tokens)  # type: ignore[attr-defined]
    if fast_llm.model != main_llm.model:
        console.print(f"Measuring [bold]{fast_llm.model}[/] (JSON work) ...")
        f = fast_llm.throughput(tokens)  # type: ignore[attr-defined]
    else:
        f = m
    table = Table(title="Throughput")
    for col in ("model", "generation tok/s", "prompt tok/s", "load s"):
        table.add_column(col)
    table.add_row(main_llm.model, str(m["generation_tps"]), str(m["prompt_tps"]), str(m["load_seconds"]))
    if f is not m:
        table.add_row(fast_llm.model, str(f["generation_tps"]), str(f["prompt_tps"]), str(f["load_seconds"]))
    console.print(table)

    # Estimate one chapter: draft + revisions on the main model, JSON work on the fast model.
    w = c.cfg.write
    gen = max(0.1, m["generation_tps"])
    fgen = max(0.1, f["generation_tps"])
    prompt_tps = max(50.0, m["prompt_tps"] or 500.0)
    fprompt_tps = max(50.0, f["prompt_tps"] or 500.0)
    out_tokens = w.target_chapter_words * 1.35
    prompt_tokens = min(c.cfg.llm.num_ctx - output_tokens_for(w.target_chapter_words), 4500)
    draft_s = out_tokens / gen + prompt_tokens / prompt_tps
    revise_s = (out_tokens / gen + (prompt_tokens + out_tokens) / prompt_tps) * w.max_revisions
    json_calls = 1 + w.max_revisions  # critic
    json_calls += n_chars * (1 + 1 + 1) if w.characters_enabled else 0  # brief, review, memory
    json_calls += 1  # summary
    per_json = 250 / fgen + (1500 + out_tokens) / fprompt_tps
    json_s = json_calls * per_json / max(1, min(c.cfg.llm.parallel, max(1, n_chars)))
    total = draft_s + revise_s + json_s
    console.print(f"\nEstimated per chapter ({w.target_chapter_words} words, {w.max_revisions} revisions, {n_chars} characters):")
    console.print(f"  draft {draft_s / 60:.1f} min  revisions {revise_s / 60:.1f} min  JSON work {json_s / 60:.1f} min  "
                  f"-> [bold]about {total / 60:.0f} min[/] (worst case; good drafts skip revisions)")
    if f is m and c.cfg.llm.provider == "ollama":
        console.print("  tip: set llm.fast_model to a 3B model (e.g. llama3.2:3b) to cut the JSON work by 2-3x.")
    if c.cfg.llm.num_ctx < 8192:
        console.print(f"  [yellow]num_ctx={c.cfg.llm.num_ctx} is small; prompts will be trimmed. 8192 is recommended.[/]")


@character.command("arc")
@click.argument("project")
@click.argument("name")
@click.option("--trait", "traits", multiple=True, required=True, help="axis=target, e.g. --trait courage=0.9 (repeatable)")
@click.option("--by", "by_chapter", type=int, required=True, help="Chapter by which the trait should reach its target.")
@click.pass_obj
def character_arc(c: Ctx, project: str, name: str, traits: tuple, by_chapter: int) -> None:
    """Plan personality drift: trait axes move gradually toward targets as chapters are written."""
    from .write.characters import TRAIT_AXES

    _, roster = _roster(c, project)
    ch = roster.get(name)
    if ch is None:
        console.print(f"[red]no character {name!r}[/]")
        sys.exit(1)
    for t in traits:
        axis, _, val = t.partition("=")
        if axis not in TRAIT_AXES:
            raise click.BadParameter(f"unknown axis {axis!r}")
        ch.state.arc_plan[axis] = {"target": max(0.0, min(1.0, float(val))), "by": by_chapter}
    roster.save()
    console.print(f"[green]arc set[/] for {ch.name}: " + "; ".join(f"{k} -> {v['target']} by ch {v['by']}" for k, v in ch.state.arc_plan.items()))


# ---------------------------------------------------------------------------- world
@main.group()
def world() -> None:
    """The World agent: physics, weather, crowds, consequences, present in every scene."""


def _world_or_exit(c: Ctx, project: str):
    from .write.project import Project
    from .write.world import load_world

    if not Project.exists(c.cfg.projects_dir, project):
        console.print(f"[red]no project named {project!r}[/]")
        sys.exit(1)
    p = Project(c.cfg.projects_dir, project)
    w = load_world(p.dir)
    if w is None:
        console.print("[yellow]no world yet[/]; run `gauge world build PROJECT`")
        sys.exit(1)
    return p, w


@world.command("show")
@click.argument("project")
@click.pass_obj
def world_show(c: Ctx, project: str) -> None:
    """Print the world: rules, clock and weather, places, events, consequences, recurring extras."""
    _, w = _world_or_exit(c, project)
    console.print(w.markdown())


@world.command("build")
@click.argument("project")
@click.pass_obj
def world_build(c: Ctx, project: str) -> None:
    """(Re)build the world state from the bible and premise."""
    pipe = _pipeline_or_exit(c)
    w = pipe.build_world(pipe.open_project(project))
    console.print(w.markdown())


@world.command("set")
@click.argument("project")
@click.option("--rule", "rules", multiple=True, help="Add a physics/power/social rule (repeatable).")
@click.option("--time", "time_", default=None)
@click.option("--weather", default=None)
@click.option("--location", default=None)
@click.option("--event", "events", multiple=True, help="Add an active world event.")
@click.option("--agency", type=float, default=None, help="0-1: how much the world intervenes (saved to config).")
@click.pass_obj
def world_set(c: Ctx, project: str, rules: tuple, time_: Optional[str], weather: Optional[str], location: Optional[str],
              events: tuple, agency: Optional[float]) -> None:
    """Edit the world by hand."""
    from .write.world import save_world

    p, w = _world_or_exit(c, project)
    w.physics_rules.extend(rules)
    if time_:
        w.time = time_
    if weather:
        w.weather = weather
    if location:
        w.current_location = location
    w.active_events.extend(events)
    save_world(p.dir, w)
    if agency is not None:
        c.cfg.write.world_agency = max(0.0, min(1.0, agency))
        c.cfg.save()
    console.print(w.describe())


@world.command("chat")
@click.argument("project")
@click.option("--ask", required=True)
@click.pass_obj
def world_chat(c: Ctx, project: str, ask: str) -> None:
    """Ask the world a question: weather, who is in the market, what the law would do."""
    from .write.world import WorldAgent

    pipe = _pipeline_or_exit(c)
    p, w = _world_or_exit(c, project)
    console.print(f"[bold]World:[/] {WorldAgent(pipe.fast_llm, w, c.cfg.write.world_agency).chat(ask)}")


@world.command("extras")
@click.argument("project")
@click.pass_obj
def world_extras(c: Ctx, project: str) -> None:
    """List nameless background figures and which ones recur enough to deserve a name."""
    _, w = _world_or_exit(c, project)
    if not w.background_figures:
        console.print("(no background figures yet)")
        return
    table = Table(title="Background figures")
    for col in ("tag", "trait", "seen", "last ch", "promote?"):
        table.add_column(col)
    for f in sorted(w.background_figures, key=lambda x: -x.appearances):
        table.add_row(f.tag, f.trait, str(f.appearances), str(f.last_seen), "yes" if f.appearances >= 3 else "")
    console.print(table)


@world.command("promote")
@click.argument("project")
@click.argument("tag")
@click.option("--name", required=True, help="The name this extra gets as a full character.")
@click.option("--role", default="minor")
@click.pass_obj
def world_promote(c: Ctx, project: str, tag: str, name: str, role: str) -> None:
    """Turn a recurring nameless extra into a full character with a personality matrix."""
    pipe = _pipeline_or_exit(c)
    p, w = _world_or_exit(c, project)
    f = w.figure(tag)
    if f is None:
        console.print(f"[red]no background figure {tag!r}[/]; see `gauge world extras`")
        sys.exit(1)
    concept = f"Formerly '{f.tag}' ({f.trait}). Seen {f.appearances} times. " + " ".join(f.actions[-3:])
    ch = pipe.create_character(p, name, concept, role)
    ch.aliases = sorted(set(ch.aliases) | {f.tag})
    roster = pipe.roster(p)
    roster.add(ch)
    roster.save()
    console.print(f"[green]Promoted[/] {f.tag} -> {ch.name} ({ch.role})")
    console.print(ch.sheet_markdown())


# ---------------------------------------------------------------------------- agents (the writing room)
@main.group()
def agents() -> None:
    """The writing room: reader panel, showrunner, archivist, schemer, line editor, packager, theme keeper,
    sensitivity reader, pacing director, librarian, comedian."""


def _council(c: Ctx, project: str):
    pipe = _pipeline_or_exit(c)
    p = pipe.open_project(project)
    return pipe, p, pipe.council(p)


@agents.command("list")
@click.argument("project", required=False)
@click.pass_obj
def agents_list(c: Ctx, project: Optional[str]) -> None:
    """Which agents are on, and (for a project) a one-line state each."""
    from .write.agents import ALL_AGENTS

    a = c.cfg.agents
    table = Table(title=f"Agents (profile: {a.profile}, council weight {a.weight})")
    for col in ("agent", "enabled", "state"):
        table.add_column(col)
    council = None
    if project:
        _, _, council = _council(c, project)
    for name, cls in ALL_AGENTS.items():
        on = getattr(a, name, True) and a.profile != "off"
        state = ""
        if council is not None:
            ag = council.get(name)
            state = (ag.report().splitlines() or [""])[0][:90] if ag else "(not attached)"
        table.add_row(cls.title, "yes" if on else "no", state)
    console.print(table)


@agents.command("show")
@click.argument("project")
@click.argument("name")
@click.pass_obj
def agents_show(c: Ctx, project: str, name: str) -> None:
    """Full report of one agent's state for a project."""
    _, _, council = _council(c, project)
    ag = council.get(name)
    if ag is None:
        console.print(f"[red]no agent {name!r}[/]; have: {', '.join(council.names())}")
        sys.exit(1)
    console.print(f"[bold]{ag.title}[/]\n{ag.report()}")


@agents.command("enable")
@click.argument("name")
@click.pass_obj
def agents_enable(c: Ctx, name: str) -> None:
    """Turn an agent on (saved to config). NAME may also be a profile: full, lean, off."""
    _toggle(c, name, True)


@agents.command("disable")
@click.argument("name")
@click.pass_obj
def agents_disable(c: Ctx, name: str) -> None:
    """Turn an agent off (saved to config)."""
    _toggle(c, name, False)


def _toggle(c: Ctx, name: str, on: bool) -> None:
    from .write.agents import ALL_AGENTS

    if name in ("full", "lean", "off"):
        c.cfg.agents.profile = name
    elif name in ALL_AGENTS:
        setattr(c.cfg.agents, name, on)
    else:
        raise click.BadParameter(f"unknown agent {name!r}; agents: {', '.join(ALL_AGENTS)}; profiles: full, lean, off")
    c.cfg.save()
    console.print(f"[green]saved[/] agents.{name if name not in ('full', 'lean', 'off') else 'profile'} = {on if name not in ('full', 'lean', 'off') else name}")


@agents.command("setup")
@click.argument("project")
@click.pass_obj
def agents_setup(c: Ctx, project: str) -> None:
    """Attach the enabled agents to an existing project (schemer, theme keeper build their state)."""
    pipe = _pipeline_or_exit(c)
    names = pipe.setup_agents(pipe.open_project(project))
    console.print(f"[green]attached[/] {', '.join(names)}")


@main.command()
@click.argument("project")
@click.argument("question", required=False)
@click.option("--subject", default=None, help="List facts about a subject instead of asking.")
@click.pass_obj
def canon(c: Ctx, project: str, question: Optional[str], subject: Optional[str]) -> None:
    """Ask the Archivist what is true in the story, or list canon facts."""
    _, _, council = _council(c, project)
    ag = council.get("archivist")
    if ag is None:
        console.print("[yellow]archivist is disabled[/]")
        sys.exit(1)
    if subject:
        for f in ag.lookup(subject):
            console.print(f"ch {f.chapter}: {f.subject} / {f.attribute} = {f.value}")
        return
    if not question:
        console.print(ag.report())
        return
    console.print(ag.ask(question))


@main.command()
@click.argument("project")
@click.pass_obj
def ledger(c: Ctx, project: str) -> None:
    """The Showrunner's ledger of promises and payoffs."""
    _, _, council = _council(c, project)
    ag = council.get("showrunner")
    console.print(ag.report() if ag else "[yellow]showrunner is disabled[/]")


@main.command()
@click.argument("project")
@click.pass_obj
def readers(c: Ctx, project: str) -> None:
    """What the Reader Panel predicted for recent chapters, and how calibrated it is."""
    _, _, council = _council(c, project)
    ag = council.get("reader_panel")
    console.print(ag.report() if ag else "[yellow]reader panel is disabled[/]")


@main.command()
@click.argument("project")
@click.argument("chapter", type=int)
@click.option("--views", type=float, default=None)
@click.option("--comments", type=float, default=None)
@click.option("--likes", type=float, default=None)
@click.pass_obj
def feedback(c: Ctx, project: str, chapter: int, views: Optional[float], comments: Optional[float], likes: Optional[float]) -> None:
    """Record real reader numbers for a posted chapter. Feeds the Reader Panel's calibration, the prompt
    bandit (retention as delayed reward) and the per-project engagement model."""
    pipe = _pipeline_or_exit(c)
    out = pipe.record_feedback(pipe.open_project(project), chapter, views, comments, likes)
    console.print_json(json.dumps(out, default=str))


@main.command()
@click.argument("project")
@click.pass_obj
def bandit(c: Ctx, project: str) -> None:
    """How the prompt-settings bandit is doing for a project (posterior means per knob value)."""
    pipe = _pipeline_or_exit(c)
    b = pipe.bandit(pipe.open_project(project))
    if b is None:
        console.print("[yellow]bandit disabled[/] (learn.bandit_enabled)")
        return
    console.print(b.report())
    console.print(f"current best: {b.best()}")


@main.command()
@click.argument("project", required=False)
@click.pass_obj
def lessons(c: Ctx, project: Optional[str]) -> None:
    """Distil what the writing room keeps complaining about into lessons (shown to the writer)."""
    from .learn.lessons import write_lessons

    summary = write_lessons(c.db, c.cfg.models_dir / "lessons.md", project)
    console.print((c.cfg.models_dir / "lessons.md").read_text(encoding="utf-8"))
    console.print(f"[dim]{summary['total']} notes considered; saved to {c.cfg.models_dir / 'lessons.md'}[/]")


@main.command("sync-edits")
@click.argument("project")
@click.pass_obj
def sync_edits(c: Ctx, project: str) -> None:
    """Learn from your hand edits: compare chapters on disk with what Gauge accepted; record preference pairs."""
    from .learn.finetune import sync_author_edits
    from .write.project import Project

    if not Project.exists(c.cfg.projects_dir, project):
        console.print(f"[red]no project named {project!r}[/]")
        sys.exit(1)
    changes = sync_author_edits(Project(c.cfg.projects_dir, project), c.cfg.models_dir / "datasets")
    if not changes:
        console.print("no new edits found")
        return
    for ch in changes:
        console.print(f"  ch {ch['chapter']}: {ch['words_changed']} words changed -> author_edits.jsonl")
    console.print(f"[green]{len(changes)} pair(s) recorded[/]")


@main.command()
@click.option("--run", "do_run", is_flag=True, help="Actually train (needs CUDA + unsloth); default is to show the plan.")
@click.option("--base", default="unsloth/llama-3.1-8b-instruct", show_default=True)
@click.pass_obj
def finetune(c: Ctx, do_run: bool, base: str) -> None:
    """Check the fine-tuning datasets and run (or print) the LoRA training command."""
    from .learn import finetune as ft

    ds = c.cfg.models_dir / "datasets"
    p = ft.plan(ds, base, c.cfg.models_dir / "lora")
    table = Table(title="Fine-tuning data")
    table.add_column("file")
    table.add_column("rows")
    for k, v in p["datasets"].items():
        table.add_row(k, str(v))
    console.print(table)
    console.print(f"preference pairs: {p['preference_pairs']}  enough data: {p['enough_data']}  cuda: {p['cuda']}  unsloth: {p['unsloth']}  ollama: {p['ollama']}")
    console.print(f"\ncommand:\n  {p['command']}")
    if not do_run:
        console.print("\nAdd --run to train here, or copy the command to a machine with an NVIDIA GPU (WSL2 works).")
        return
    if not (p["cuda"] and p["unsloth"]):
        console.print("[red]cannot train here[/]: needs CUDA and `pip install unsloth trl datasets`.")
        sys.exit(1)
    sys.exit(ft.run(ds, base, c.cfg.models_dir / "lora"))


@main.group()
def persona() -> None:
    """The author persona: who is writing, learned from your own novels."""


@persona.command("build")
@click.option("--from", "source", required=True, help="A novel key (e.g. local:my-serial) or a project name to learn the voice from.")
@click.option("--blend", type=float, default=None, help="0 = write like the data, 1 = write like you (saved to config).")
@click.pass_obj
def persona_build(c: Ctx, source: str, blend: Optional[float]) -> None:
    """Build the persona from your own chapters: measured fingerprint + a described voice."""
    from .llm import make_llm
    from .write.persona import build_persona, save_persona
    from .write.project import Project

    texts: list[str] = []
    novel = c.db.get_novel(source)
    if novel is not None:
        texts = [ch.text for ch in c.db.chapters(novel.key)]
    elif Project.exists(c.cfg.projects_dir, source):
        p = Project(c.cfg.projects_dir, source)
        texts = [t for n in range(1, p.state.written + 1) for t in [p.read_chapter(n)] if t]
    if not texts:
        console.print(f"[red]nothing to learn from[/]: {source!r} is neither a novel key (see `gauge novels`) nor a project with chapters")
        sys.exit(1)
    llm = make_llm(c.cfg.llm, "fast")
    per = build_persona(llm if llm.available() else None, texts, source)
    save_persona(c.cfg.models_dir, per)
    if blend is not None:
        c.cfg.write.persona_blend = max(0.0, min(1.0, blend))
        c.cfg.save()
    console.print(per.prompt_block(1.0))
    console.print(f"\n[green]saved[/] {c.cfg.models_dir / 'persona.json'}  blend={c.cfg.write.persona_blend}")


@persona.command("show")
@click.pass_obj
def persona_show(c: Ctx) -> None:
    """Print the persona and the current blend."""
    from .write.persona import load_persona

    per = load_persona(c.cfg.models_dir)
    console.print(per.prompt_block(1.0) if per else "(no persona yet; `gauge persona build --from local:my-serial`)")
    console.print(f"blend: {c.cfg.write.persona_blend}")


@persona.command("blend")
@click.argument("value", type=float)
@click.pass_obj
def persona_blend(c: Ctx, value: float) -> None:
    """Set how much the persona outweighs the data (0..1)."""
    c.cfg.write.persona_blend = max(0.0, min(1.0, value))
    c.cfg.save()
    console.print(f"[green]persona blend[/] = {c.cfg.write.persona_blend}")





@main.command()
@click.argument("project")
@click.pass_obj
def package(c: Ctx, project: str) -> None:
    """Blurb, tagline and tags for the serial, from the Packager."""
    _, p, council = _council(c, project)
    ag = council.get("packager")
    if ag is None:
        console.print("[yellow]packager is disabled[/]")
        sys.exit(1)
    out = ag.package(p)
    console.print(f"[bold]{out['tagline']}[/]\n\n{out['blurb']}\n\nTags: {', '.join(out['tags'])}")


@main.command()
@click.argument("project")
@click.argument("question")
@click.pass_obj
def research(c: Ctx, project: str, question: str) -> None:
    """Search your project's research/ notes and premise (the Librarian)."""
    _, _, council = _council(c, project)
    ag = council.get("librarian")
    console.print(ag.ask(question) if ag else "[yellow]librarian is disabled[/]")


# ---------------------------------------------------------------------------- scout (research)
@main.group()
def scout() -> None:
    """The Scout: open-source research that makes scenes concrete (Wikipedia, Wikivoyage, Gutenberg, arXiv, climate)."""


def _scout(c: Ctx):
    from .research import Scout

    s = Scout(c.cfg, c.db)
    if not s.sources:
        console.print("[yellow]no research sources available[/] (GAUGE_OFFLINE set, or research.sources empty in config)")
    return s


@scout.command("ask")
@click.argument("query")
@click.option("--source", default="wikipedia", help="wikipedia | wikivoyage | gutenberg | arxiv | climate | searx")
@click.option("--project", default=None, help="Also file the findings into this project's research folder.")
@click.option("-k", default=3, show_default=True)
@click.pass_obj
def scout_ask(c: Ctx, query: str, source: str, project: Optional[str], k: int) -> None:
    """Look something up: `gauge scout ask "medieval courier guild"`, `gauge scout ask "Lisbon in November" --source climate`."""
    from .research.scout import Dossier

    s = _scout(c)
    f = s.search(query, source, k=k)
    c.db.research_log(project or "", None, f.query, f.source, len(f.snippets), f.cached)
    if not f.snippets:
        console.print("(nothing found)")
        return
    for sn in f.snippets:
        console.print(f"[bold]{sn.title}[/]  [dim]{sn.url}[/]\n{sn.text}\n[dim]{sn.license}[/]\n")
    if project:
        from .write.project import Project

        if not Project.exists(c.cfg.projects_dir, project):
            console.print(f"[red]no project named {project!r}[/]")
            sys.exit(1)
        path = s.file(Project(c.cfg.projects_dir, project).dir, Dossier([f]), query)
        console.print(f"[green]filed[/] {path}")


@scout.command("dossier")
@click.argument("project")
@click.option("--topic", "topics", multiple=True, help="Topics to look up (default: proper nouns and trades from the bible).")
@click.option("--source", "sources", multiple=True, default=("wikipedia",), show_default=True)
@click.pass_obj
def scout_dossier(c: Ctx, project: str, topics: tuple, sources: tuple) -> None:
    """Build or extend the project's background dossier in research/scout/."""
    pipe = _pipeline_or_exit(c)
    p = pipe.open_project(project)
    path = pipe.build_dossier(p, list(topics) or None, tuple(sources))
    console.print(f"[green]dossier[/] {path}" if path else "(nothing found)")


@scout.command("gutenberg")
@click.argument("query")
@click.option("--import", "do_import", is_flag=True, help="Import the matching public-domain book(s) into the corpus.")
@click.option("--max-books", default=1, show_default=True)
@click.pass_obj
def scout_gutenberg(c: Ctx, query: str, do_import: bool, max_books: int) -> None:
    """Find public-domain books on Project Gutenberg; with --import, add them as novels Gauge learns from."""
    s = _scout(c)
    src = s.sources.get("gutenberg")
    if src is None:
        console.print("[yellow]gutenberg source disabled[/]")
        sys.exit(1)
    books = src.find_books(query, k=max(max_books, 5))
    if not books:
        console.print("(no books found)")
        return
    table = Table(title=f"Project Gutenberg: {query}")
    for col in ("id", "title", "authors", "downloads", "text"):
        table.add_column(col)
    for b in books:
        table.add_row(str(b["id"]), b["title"][:60], ", ".join(b["authors"])[:40], str(b["downloads"]), "yes" if b["text_url"] else "no")
    console.print(table)
    if do_import:
        keys = s.import_gutenberg(None, query, max_books)
        console.print(f"[green]imported[/] {', '.join(keys) or 'nothing (too short or no text)'}  -> run `gauge train` / `gauge absorb`")


@scout.command("log")
@click.option("--project", default=None)
@click.pass_obj
def scout_log(c: Ctx, project: Optional[str]) -> None:
    """What the Scout has looked up recently."""
    rows = c.db.research_history(project, limit=30)
    if not rows:
        console.print("(no research yet)")
        return
    table = Table(title="Research log")
    for col in ("when", "project", "ch", "source", "query", "hits", "cached"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["at"][:16], r["project"] or "", str(r["chapter"] or ""), r["source"], r["query"], str(r["hits"]), "yes" if r["cached"] else "")
    console.print(table)


# ---------------------------------------------------------------------------- native model (GaugeLM)
@main.group()
def native() -> None:
    """GaugeLM: Gauge's own language model. Pure Python + numpy, trained on your corpus, no outside model code."""


def _native_docs(c: Ctx, since: Optional[str] = None):
    from .native import gather_texts

    return gather_texts(c.db, c.cfg.projects_dir, c.cfg.models_dir / "datasets", since=since)


@native.command("train")
@click.option("--steps", type=int, default=None)
@click.option("--layers", type=int, default=None)
@click.option("--embd", type=int, default=None)
@click.option("--heads", type=int, default=None)
@click.option("--block", type=int, default=None)
@click.option("--vocab", type=int, default=None)
@click.option("--batch", type=int, default=None)
@click.option("--resume", is_flag=True, help="Continue from the saved weights instead of starting over.")
@click.option("--minutes", type=float, default=None, help="Stop after this many minutes (checkpoint is saved).")
@click.pass_obj
def native_train(c: Ctx, steps, layers, embd, heads, block, vocab, batch, resume, minutes) -> None:
    """Train (or keep training) GaugeLM on everything Gauge has read plus your own chapters."""
    from .native import ModelConfig, NativeTrainer, TrainConfig

    n = c.cfg.native
    mcfg = ModelConfig(vocab_size=vocab or n.vocab_size, block_size=block or n.block_size, n_layer=layers or n.n_layer,
                       n_head=heads or n.n_head, n_embd=embd or n.n_embd)
    tcfg = TrainConfig(steps=steps or n.steps, batch_size=batch or n.batch_size, lr=n.lr)
    docs = _native_docs(c)
    if not docs:
        console.print("[red]nothing to train on[/]: ingest or import some novels first")
        sys.exit(1)
    chars = sum(len(t) for _, t in docs)
    console.print(f"corpus: {len(docs)} documents, {chars:,} characters. model: {mcfg.n_layer} layers x {mcfg.n_embd} dims, block {mcfg.block_size}")
    console.print("Training on CPU with numpy. This is slow by design: a few thousand tokens per second. Ctrl+C keeps the last checkpoint.")
    tr = NativeTrainer(c.cfg.native_dir)

    def progress(rec):
        console.print(f"  step {rec['step']:>6}  loss {rec['loss']:.3f}  val {rec['val_loss']:.3f}  {rec['tok_per_s']} tok/s")

    try:
        res = tr.train(docs, mcfg, tcfg, resume=resume, progress=progress, time_budget_s=minutes * 60 if minutes else None)
    except KeyboardInterrupt:
        console.print("[yellow]stopped; last checkpoint kept[/]")
        return
    console.print_json(json.dumps(res))
    console.print(f"[green]saved[/] {c.cfg.native_dir}. Use it with: gauge init --provider native")


@native.command("continue")
@click.option("--steps", type=int, default=None)
@click.option("--minutes", type=float, default=None)
@click.pass_obj
def native_continue(c: Ctx, steps, minutes) -> None:
    """Keep learning from new material (what the daemon does each cycle)."""
    from .native import NativeTrainer

    tr = NativeTrainer(c.cfg.native_dir)
    docs = _native_docs(c)
    res = tr.continue_training(docs, steps or c.cfg.native.daemon_steps, time_budget_s=(minutes or c.cfg.native.daemon_minutes) * 60,
                               progress=lambda r: console.print(f"  step {r['step']} loss {r['loss']:.3f} val {r['val_loss']:.3f}"))
    console.print_json(json.dumps(res))


@native.command("status")
@click.pass_obj
def native_status(c: Ctx) -> None:
    """Size, steps, losses of the native model."""
    from .native import NativeTrainer

    console.print_json(json.dumps(NativeTrainer(c.cfg.native_dir).status(), default=str))


@native.command("sample")
@click.argument("prompt", default="Write the opening of a chapter about a courier in the rain.")
@click.option("--tokens", type=int, default=150, show_default=True)
@click.option("--temperature", type=float, default=0.9, show_default=True)
@click.pass_obj
def native_sample(c: Ctx, prompt: str, tokens: int, temperature: float) -> None:
    """Generate text with GaugeLM."""
    from .native import NativeLLM

    llm = NativeLLM(c.cfg.llm, c.cfg.native_dir)
    if not llm.available():
        console.print("[red]no native model yet[/]; run `gauge native train`")
        sys.exit(1)
    console.print(llm.generate(prompt, max_tokens=tokens, temperature=temperature, purpose="sample"))


@native.command("tokenizer")
@click.argument("text")
@click.pass_obj
def native_tokenizer(c: Ctx, text: str) -> None:
    """Show how the trained tokenizer splits a text."""
    from .native import BPETokenizer

    path = c.cfg.native_dir / "tokenizer.json"
    if not path.exists():
        console.print("[red]no tokenizer yet[/]")
        sys.exit(1)
    tok = BPETokenizer.load(path)
    ids = tok.encode(text)
    console.print(f"{len(ids)} tokens (vocab {tok.vocab_size}): " + " | ".join(tok.decode([i]) for i in ids))


# ---------------------------------------------------------------------------- gui
@main.command()
@click.pass_obj
def gui(c: Ctx) -> None:
    """Open Gauge Studio: a big-button window for starting a story and reading chapters."""
    from .gui import run_studio

    run_studio(c.cfg, c.db)


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
