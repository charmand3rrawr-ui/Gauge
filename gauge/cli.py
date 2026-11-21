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
    from .learn.retention import shape as _ret_shape
    from .write.agents.readability import measure as _readability
    rs = _ret_shape(text)
    console.print(f"retention curve: opening={rs.opening:.2f} middle={rs.middle:.2f} ending={rs.ending:.2f} "
                  f"sag={rs.sag:.2f} -> shape={rs.score:.2f}")
    for i in rs.issues:
        console.print(f"  - {i}")
    rm = _readability(text)
    console.print(f"readability: Flesch={rm['flesch']:.0f} burstiness={rm['burstiness']:.2f} "
                  f"syll/word={rm['syllables_per_word']:.2f} walls={rm['wall_paragraph_ratio']:.0%}")
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
@click.option("--seed", type=int, default=None, help="Make this run reproducible (seeds RNGs and the bandit).")
@click.pass_obj
def write(c: Ctx, name: str, chapters: int, fresh: bool, seed: Optional[int]) -> None:
    """Write the next chapter(s) of a project: draft -> score -> revise -> save."""
    from .write import WritePipeline

    if seed is not None:
        c.cfg.seed = seed
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


@character.command("mood")
@click.argument("project")
@click.argument("name")
@click.pass_obj
def character_mood(c: Ctx, project: str, name: str) -> None:
    """Audit a character's emotional continuity: flag mood jumps the chapters never earned."""
    from .write.emotion_continuity import audit, summary

    _, roster = _roster(c, project)
    ch = roster.get(name)
    if ch is None:
        console.print(f"[red]no character {name!r}[/]")
        sys.exit(1)
    hist = ch.state.mood_history
    console.print(f"[bold]{ch.name}[/]: {summary(hist)}")
    for f in audit(hist):
        console.print(f"  - ch {f.chapter} [{f.kind}] {f.detail}")


@character.command("remember-across")
@click.argument("project")
@click.pass_obj
def character_remember_across(c: Ctx, project: str) -> None:
    """Fold this project's characters into the cross-book memory (scars, habits, grudges)."""
    from .write.character_memory import GlobalCharacterMemory

    _, roster = _roster(c, project)
    mem = GlobalCharacterMemory(c.cfg.models_dir)
    for ch in roster.ordered():
        led = mem.absorb(ch, project)
        console.print(f"[green]remembered[/] {ch.name}: {len(led.scars)} scars, {len(led.habits)} habits, {len(led.grudges)} grudges")


@character.command("ledger")
@click.argument("name", required=False)
@click.pass_obj
def character_ledger(c: Ctx, name: Optional[str]) -> None:
    """Show the cross-book memory for one character, or list everyone remembered."""
    from .write.character_memory import GlobalCharacterMemory

    mem = GlobalCharacterMemory(c.cfg.models_dir)
    if name:
        led = mem.get(name)
        if led is None:
            console.print(f"[red]nobody named {name!r} in the cross-book memory[/]")
            return
        console.print_json(led.model_dump_json())
    else:
        all_ = mem.all()
        if not all_:
            console.print("(cross-book memory is empty; run `gauge character remember-across <project>`)")
            return
        for led in all_:
            console.print(f"{led.name}: books={', '.join(led.books) or '-'}; scars={len(led.scars)}; grudges={len(led.grudges)}")


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


@main.command(name="feedback-import")
@click.argument("project")
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def feedback_import(c: Ctx, project: str, path: str) -> None:
    """Bulk-import real reader numbers you exported yourself (CSV or JSON) as rows of
    chapter, views, comments, likes. One row per chapter; feeds the same calibration as `feedback`."""
    import csv as _csv
    import json as _json

    p = Path(path)
    rows: list[dict] = []
    if p.suffix.lower() == ".json":
        data = _json.loads(p.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("chapters", [])
    else:
        with p.open(encoding="utf-8", newline="") as fh:
            rows = list(_csv.DictReader(fh))
    pipe = _pipeline_or_exit(c)
    proj = pipe.open_project(project)
    n = 0
    for row in rows:
        try:
            ch = int(row.get("chapter") or row.get("chapter_number"))
        except (TypeError, ValueError):
            continue
        def _num(r: dict, k: str) -> Optional[float]:
            return float(r[k]) if r.get(k) not in (None, "") else None
        pipe.record_feedback(proj, ch, _num(row, "views"), _num(row, "comments"), _num(row, "likes"))
        n += 1
    console.print(f"[green]imported[/] {n} chapter(s) of feedback for {project}")


@main.command()
@click.argument("project")
@click.pass_obj
def evaluate(c: Ctx, project: str) -> None:
    """Does Gauge's chapter score predict real retention? Correlates stored scores with the retention
    your recorded feedback implies, and reports the per-project model's cross-validated fit."""
    from .learn.evaluation import evaluate_project

    pipe = _pipeline_or_exit(c)
    proj = pipe.open_project(project)
    fb = c.db.feedback(project)
    panel = pipe.council(proj).get("reader_panel")
    panel_corr = panel.calibration().get("correlation") if panel is not None else None
    pm = pipe.project_model(proj)
    fitted = pm.fit(proj, fb, min_chapters=c.cfg.learn.project_model_min_chapters)
    r2cv = round(fitted.r2_cv, 3) if fitted and fitted.r2_cv == fitted.r2_cv else None
    ev = evaluate_project(proj, fb, panel_corr=panel_corr, project_model_r2cv=r2cv)
    console.print(f"[bold]{project}[/]: {ev.chapters_with_feedback} chapter(s) with retention data")
    if ev.retention_mean is not None:
        console.print(f"  mean retention {ev.retention_mean}, trend {ev.retention_trend}")
    console.print(f"  score vs real retention: {ev.score_vs_retention}  |  panel correlation: {ev.panel_correlation}  |  project model R2(cv): {ev.project_model_r2cv}")
    console.print(f"  [bold]{ev.verdict()}[/]")
    if ev.worst_chapters:
        console.print("  lowest-retention chapters: " + ", ".join(f"ch{n} ({r})" for n, r in ev.worst_chapters))
    for note in ev.notes:
        console.print(f"  - {note}")


@main.group()
def newsdesk() -> None:
    """Find trending topics worldwide (any language) and write ORIGINAL, sourced commentary on a schedule.
    It never rewrites other outlets' articles; drafts wait for your approval unless they pass every check."""


@newsdesk.command("trends")
@click.option("--lang", "langs", multiple=True, help="Wikipedia editions to watch (repeatable), e.g. --lang en --lang ja.")
@click.option("--feed", "feeds", multiple=True, help="RSS/Atom feeds you choose (repeatable).")
@click.option("--top", type=int, default=15, show_default=True)
@click.pass_obj
def newsdesk_trends(c: Ctx, langs: tuple, feeds: tuple, top: int) -> None:
    """What the world is reading: Wikipedia's most-viewed pages per language, plus your feeds."""
    from .newsdesk.desk import make_fetcher
    from .newsdesk.trends import discover

    nd = c.cfg.newsdesk
    topics = discover(make_fetcher(c.cfg), list(langs) or nd.languages, list(feeds) or nd.feeds)
    if not topics:
        console.print("no trends found (offline, robots.txt refused, or no feeds configured)")
        return
    for t in topics[:top]:
        console.print(f"  {t.score:5.2f}  {t.title}  [{', '.join(t.langs) or t.source}]")


@newsdesk.command("draft")
@click.argument("topic")
@click.option("--byline", default=None, help="Defaults to newsdesk.byline.")
@click.pass_obj
def newsdesk_draft(c: Ctx, topic: str, byline: Optional[str]) -> None:
    """Draft original commentary on TOPIC (sourced from your feeds) and add it to the review queue."""
    from .newsdesk.commentary import write_commentary
    from .newsdesk.desk import gather_sources, make_fetcher, queue_for
    from .site.stance import load_stance

    nd = c.cfg.newsdesk
    by = byline or nd.byline
    if not by:
        console.print("[red]set a byline[/] (--byline or newsdesk.byline): your name or an owned pen name")
        sys.exit(1)
    pipe = _pipeline_or_exit(c)
    sources = gather_sources(make_fetcher(c.cfg), pipe.llm, topic, nd.feeds, nd.target_language) if nd.feeds else []
    try:
        com = write_commentary(pipe.llm, topic, sources, by, stance=load_stance(c.cfg.root / "stance.json"),
                               voice=nd.voice, comedic=nd.comedic, max_overlap=nd.max_source_overlap)
    except PermissionError as e:
        console.print(f"[red]refused:[/] {e}")
        sys.exit(1)
    item = queue_for(c.cfg).add(com)
    console.print(f"[bold]{com.title}[/]  (queued as {item.id}, {'clean' if com.clean else 'needs review'})")
    console.print(f"  originality {com.originality:.0%}; {len(com.sources)} source(s) linked")
    for f in com.originality_flags + com.balance_flags + com.headline_flags + com.claims_to_verify[:6]:
        console.print(f"  [yellow]- {f}[/]")


@newsdesk.command("queue")
@click.option("--status", type=click.Choice(["pending_review", "approved", "published", "rejected"]), default=None)
@click.pass_obj
def newsdesk_queue(c: Ctx, status: Optional[str]) -> None:
    """List drafts and their status."""
    from .newsdesk.desk import queue_for

    items = queue_for(c.cfg).items(status)
    if not items:
        console.print("(queue is empty)")
        return
    for i in items:
        mark = "clean" if i.clean else f"{len(i.flags)} flag(s)"
        console.print(f"  {i.id}  {i.status:<15} {mark:<10} {i.title[:70]}")


@newsdesk.command("show")
@click.argument("item_id")
@click.pass_obj
def newsdesk_show(c: Ctx, item_id: str) -> None:
    """Read a queued draft in full, with its flags, before approving it."""
    from .newsdesk.desk import queue_for

    it = queue_for(c.cfg).get(item_id)
    if it is None:
        console.print(f"[red]no queued item {item_id}[/]")
        sys.exit(1)
    console.print(f"[bold]{it.title}[/]  ({it.status}, originality {it.originality:.0%})\n")
    console.print(it.body)
    for f in it.flags:
        console.print(f"[yellow]- {f}[/]")


@newsdesk.command("approve")
@click.argument("item_id")
@click.pass_obj
def newsdesk_approve(c: Ctx, item_id: str) -> None:
    """Approve a draft for publication on the next due slot."""
    from .newsdesk.desk import queue_for

    ok = queue_for(c.cfg).approve(item_id)
    console.print(f"[green]approved[/] {item_id}" if ok else f"[red]cannot approve {item_id}[/]")


@newsdesk.command("reject")
@click.argument("item_id")
@click.pass_obj
def newsdesk_reject(c: Ctx, item_id: str) -> None:
    """Reject a draft; it will never be published."""
    from .newsdesk.desk import queue_for

    ok = queue_for(c.cfg).reject(item_id)
    console.print(f"[yellow]rejected[/] {item_id}" if ok else f"[red]cannot reject {item_id}[/]")


@newsdesk.command("run")
@click.option("--no-draft", is_flag=True, help="Only publish what is due; do not discover or draft.")
@click.pass_obj
def newsdesk_run(c: Ctx, no_draft: bool) -> None:
    """One cycle: discover trends, draft new commentary into the queue, publish one due piece."""
    from .newsdesk.desk import run_cycle

    pipe = _pipeline_or_exit(c)
    rep = run_cycle(c.cfg, pipe.llm, draft=not no_draft)
    console.print(f"trends {rep.topics_found}; drafted {len(rep.drafted)}; skipped {len(rep.skipped)} already-queued; "
                  f"waiting review {rep.waiting_review}; published: {rep.published or 'nothing due'}")
    for d in rep.drafted:
        console.print(f"  + {d}")
    for n in rep.notes:
        console.print(f"  [yellow]- {n}[/]")


@newsdesk.command("watch")
@click.option("--max-minutes", type=int, default=None, help="Stop after this long (default: run until Ctrl+C).")
@click.pass_obj
def newsdesk_watch(c: Ctx, max_minutes: Optional[int]) -> None:
    """Run the desk continuously: a cycle every newsdesk.interval_minutes (default 30). Ctrl+C to stop."""
    import time as _time

    from .newsdesk.desk import run_cycle

    pipe = _pipeline_or_exit(c)
    interval = max(1, c.cfg.newsdesk.interval_minutes)
    started = _time.monotonic()
    console.print(f"newsdesk watching: a cycle every {interval} min. Ctrl+C to stop.")
    try:
        while True:
            rep = run_cycle(c.cfg, pipe.llm)
            console.print(f"[{utcnow_str()}] drafted {len(rep.drafted)}, published {rep.published or '-'}, "
                          f"waiting review {rep.waiting_review}")
            if max_minutes is not None and (_time.monotonic() - started) / 60 >= max_minutes:
                break
            _time.sleep(interval * 60)
    except KeyboardInterrupt:
        console.print("stopped")


def utcnow_str() -> str:
    from .util import utcnow
    return utcnow().strftime("%Y-%m-%d %H:%M")


@main.group()
def site() -> None:
    """Publish the author's own work to a static website: the serial and a labelled-opinion blog.
    A byline is required and cannot impersonate an outlet; opinion is labelled; clickbait is refused."""


def _article_dir(c: Ctx, project: str):
    from .write.project import Project
    if not Project.exists(c.cfg.projects_dir, project):
        console.print(f"[red]no project named {project!r}[/]")
        sys.exit(1)
    p = Project(c.cfg.projects_dir, project)
    d = p.dir / "articles"
    d.mkdir(parents=True, exist_ok=True)
    return p, d


@site.command("stance")
@click.option("--set", "set_json", type=click.Path(exists=True), default=None, help="Replace the stance from a JSON file.")
@click.pass_obj
def site_stance(c: Ctx, set_json: Optional[str]) -> None:
    """Show (or set) the platform's declared editorial stance. It is a transparent point of view, not a
    claim of neutrality; it shapes labelled opinion and is published on the site's editorial page."""
    import json as _json

    from .site.stance import EditorialStance, editorial_page_body, load_stance, save_stance

    path = c.cfg.root / "stance.json"
    if set_json:
        d = _json.loads(Path(set_json).read_text(encoding="utf-8"))
        save_stance(path, EditorialStance(**{k: d[k] for k in ("name", "principles", "lean_rule", "tone", "rails") if k in d}))
        console.print(f"[green]saved stance[/] -> {path}")
    stance = load_stance(path)
    console.print(f"[bold]{stance.name}[/]")
    for p in stance.principles:
        console.print(f"  - {p}")
    console.print(f"lean: {stance.lean_rule}")
    console.print(f"tone: {stance.tone}")
    console.print("\n[dim]editorial page:[/]\n" + editorial_page_body(stance))


@site.command("opinion")
@click.argument("project")
@click.option("--thesis", required=True, help="The argument you want the column to make.")
@click.option("--byline", required=True, help="Your name or an owned pen name (not an outlet or shared pseudonym).")
@click.option("--angle", default="", help="Optional angle or framing.")
@click.option("--voice", type=click.Choice(["plain", "punchy", "contrarian"]), default="plain", help="Register: plain, punchy, or contrarian (honest, your byline).")
@click.option("--stance/--no-stance", default=True, help="Write from the platform's declared editorial stance.")
@click.option("--comedic/--no-comedic", default=True, help="Punchy, comedic tone (a joke never fakes a fact).")
@click.pass_obj
def site_opinion(c: Ctx, project: str, thesis: str, byline: str, angle: str, voice: str, stance: bool, comedic: bool) -> None:
    """Draft a labelled opinion column from your thesis. Saved for the next `site build`; claims are
    flagged for you to verify, and the piece is marked Opinion under your byline."""
    import json as _json
    import re as _re

    from .site.opinion import draft_opinion
    from .site.stance import load_stance
    from .util import utcnow

    p, d = _article_dir(c, project)
    pipe = _pipeline_or_exit(c)
    editorial = load_stance(c.cfg.root / "stance.json") if stance else None
    try:
        draft = draft_opinion(pipe.llm, thesis, byline, angle, voice=voice, stance=editorial, comedic=comedic)
    except PermissionError as e:
        console.print(f"[red]refused:[/] {e}")
        sys.exit(1)
    slug = _re.sub(r"[^a-z0-9]+", "-", draft.title.lower()).strip("-")[:60] or "opinion"
    (d / f"{slug}.json").write_text(_json.dumps({
        "slug": slug, "title": draft.title, "body": draft.body, "byline": draft.byline, "kind": "opinion",
        "summary": draft.summary, "published_at": utcnow().date().isoformat(), "thumbnail": ""}, indent=2), encoding="utf-8")
    console.print(f"[bold]{draft.title}[/]\nby {draft.byline} (Opinion)")
    if draft.headline_flags:
        console.print("[yellow]headline flags:[/] " + "; ".join(draft.headline_flags))
    if draft.claims_to_verify:
        console.print("[yellow]verify before publishing:[/]")
        for cl in draft.claims_to_verify[:8]:
            console.print(f"  - {cl}")
    for b in draft.balance_flags:
        console.print(f"[yellow]balance:[/] {b}")
    console.print(f"[green]saved[/] -> {d / (slug + '.json')}; include it with `gauge site build {project} --byline ...`")


@site.command("build")
@click.argument("project")
@click.option("--byline", required=True, help="The byline for the landing page and chapters.")
@click.option("--url", default="", help="Your site's base URL (for the RSS feed links).")
@click.option("--chapters/--no-chapters", default=True, help="Include the serial's chapters as pages.")
@click.pass_obj
def site_build(c: Ctx, project: str, byline: str, url: str, chapters: bool) -> None:
    """Build the static site: a landing page, the serial's chapters, and any opinion columns you drafted.
    Writes a folder you upload to any host. Refuses an impersonating byline or a clickbait opinion title."""
    import json as _json

    from .market.launchkit import build_launch_kit
    from .site.publisher import Article, build_site
    from .util import utcnow

    p, d = _article_dir(c, project)
    pipe = _pipeline_or_exit(c)
    kit = build_launch_kit(pipe.llm, pipe.open_project(project))
    articles: list[Article] = [Article("about", p.state.name, (kit.blurb_long or kit.blurb_short or p.state.name),
                                        byline, kind="post", summary=kit.tagline or kit.blurb_short[:160],
                                        published_at=utcnow().date().isoformat())]
    from .site.stance import editorial_page_body, load_stance
    st = load_stance(c.cfg.root / "stance.json")  # the saved stance, or the evidence-first default
    articles.append(Article("editorial-stance", "Where we stand", editorial_page_body(st), byline, kind="post",
                            summary="Our declared editorial stance — a point of view, not a claim of neutrality.",
                            published_at=utcnow().date().isoformat()))
    if chapters:
        for n in range(1, p.state.written + 1):
            text = p.read_chapter(n)
            if not text:
                continue
            meta = p.read_meta(n)
            title = meta.get("title") or f"Chapter {n}"
            articles.append(Article(f"chapter-{n:03d}", f"Chapter {n}: {title}", text, byline, kind="chapter",
                                    summary=f"Chapter {n} of {p.state.name}", published_at=meta.get("written_at", "")[:10]))
    for f in sorted(d.glob("*.json")):
        try:
            a = _json.loads(f.read_text(encoding="utf-8"))
            articles.append(Article(**{k: a.get(k, "") for k in ("slug", "title", "body", "byline", "kind", "summary", "published_at", "thumbnail")}))
        except Exception as e:
            console.print(f"[yellow]skipped {f.name}: {e}[/]")
    out = p.dir / "site"
    try:
        written = build_site(p.state.name, articles, out, base_url=url)
    except PermissionError as e:
        console.print(f"[red]refused:[/] {e}")
        sys.exit(1)
    console.print(f"[green]built[/] {len(written)} files -> {out}")
    console.print(f"  landing + {sum(1 for a in articles if a.kind == 'chapter')} chapter(s) + "
                  f"{sum(1 for a in articles if a.kind == 'opinion')} opinion column(s). Upload the folder to your host.")


@site.command("moderate")
@click.argument("action", type=click.Choice(["submit", "list", "approve", "reject"]))
@click.argument("value", required=False)
@click.option("--author", default="anon")
@click.pass_obj
def site_moderate(c: Ctx, action: str, value: Optional[str], author: str) -> None:
    """Moderate reader comments through the sanitiser: submit text, list the queue, approve/reject by id."""
    from .site.moderation import ModerationQueue

    q = ModerationQueue(c.cfg.root / "comments.json")
    if action == "submit":
        cm = q.submit(value or "", author)
        console.print(f"[{'green' if cm.status == 'allow' else 'yellow'}]{cm.status}[/] {cm.id}"
                      + (f"  ({'; '.join(cm.reasons)})" if cm.reasons else ""))
    elif action == "list":
        for cm in q.items():
            console.print(f"  {cm.id}  {cm.status:<9} {cm.author}: {cm.text[:60]}" + (f"  [{', '.join(cm.reasons)}]" if cm.reasons else ""))
    elif action == "approve":
        console.print("approved" if q.approve(value or "") else "not found")
    else:
        console.print("rejected" if q.reject(value or "") else "not found")


@site.command("headline")
@click.argument("path", type=click.Path(exists=True))
@click.option("--angle", default="", help="Optional angle or framing.")
@click.option("--n", type=int, default=6, show_default=True)
@click.pass_obj
def site_headline(c: Ctx, path: str, angle: str, n: int) -> None:
    """Suggest eye-catching but honest headlines for an article file. Clickbait is excluded and each
    headline is checked against the body so it cannot take the piece out of context."""
    from .site.headline import suggest_headlines

    body = Path(path).read_text(encoding="utf-8", errors="replace")
    pipe = _pipeline_or_exit(c)
    ranked = suggest_headlines(pipe.llm, body, n=n, angle=angle)
    if not ranked:
        console.print("no honest headline candidates (the model's suggestions were clickbait or off-topic); try again")
        return
    for h in ranked:
        ctx = "" if h.in_context else f"  [yellow](check: {', '.join(h.unsupported[:3])} not in the article)[/]"
        console.print(f"  {h.score:.2f}  {h.text}{ctx}")


@main.group()
def market() -> None:
    """Market a novel the legitimate way: assets and a disclosed plan you post yourself.
    Gauge never creates accounts, posts to other platforms, or messages strangers — see `market policy`."""


@market.command("policy")
@click.pass_obj
def market_policy(c: Ctx, **_: object) -> None:
    """What Gauge will and will not do to promote a book."""
    from .market.policy import policy_text

    console.print(policy_text())


@market.command("channels")
@click.pass_obj
def market_channels(c: Ctx, **_: object) -> None:
    """Where authors legitimately share a web novel, and each venue's self-promotion rules."""
    from .market.channels import CHANNELS

    for ch in CHANNELS:
        console.print(f"[bold]{ch.name}[/] ({ch.kind}) — {ch.audience}")
        console.print(f"  rules: {ch.rules}")
        console.print(f"  disclosure: {ch.disclosure}")
        console.print(f"  do it right: {ch.do_it_right}\n")


@market.command("kit")
@click.argument("project")
@click.pass_obj
def market_kit(c: Ctx, project: str) -> None:
    """Generate the marketing launch kit (tagline, logline, blurbs, tags, author's note)."""
    from .market.launchkit import build_launch_kit

    pipe = _pipeline_or_exit(c)
    proj = pipe.open_project(project)
    kit = build_launch_kit(pipe.llm, proj)
    out = proj.dir / "launch_kit.md"
    out.write_text(kit.to_markdown(), encoding="utf-8")
    console.print(kit.to_markdown())
    console.print(f"\n[green]saved[/] -> {out}")


@market.command("plan")
@click.argument("project")
@click.option("--url", default="", help="The link readers should follow (your story's URL).")
@click.pass_obj
def market_plan(c: Ctx, project: str, url: str) -> None:
    """Build a posting plan you execute yourself: per channel, the rules, disclosure and a draft."""
    from .market.channels import CHANNELS
    from .market.launchkit import build_launch_kit, posting_plan

    pipe = _pipeline_or_exit(c)
    proj = pipe.open_project(project)
    kit = build_launch_kit(pipe.llm, proj)
    plan = posting_plan(kit, url=url)
    out = proj.dir / "posting_plan.md"
    out.write_text(plan, encoding="utf-8")
    console.print(f"[green]saved[/] a posting plan for {len(CHANNELS)} channels -> {out}")
    console.print("Gauge drafted these; you post them yourself, under your own account, following each venue's rules.")


@market.command("links")
@click.argument("url")
@click.option("--campaign", default="", help="Campaign tag (e.g. launch).")
@click.pass_obj
def market_links(c: Ctx, url: str, campaign: str) -> None:
    """Build UTM-tagged links for each channel so you can see which one sends readers (no reader tracking)."""
    from .market.channels import CHANNELS
    from .market.links import tagged_links

    sources = [ch.key for ch in CHANNELS]
    for src, link in tagged_links(url, sources, campaign=campaign).items():
        console.print(f"  {src}: {link}")


@market.command("social")
@click.argument("project")
@click.option("--account", "accounts", multiple=True, help="An account you OWN, as platform:handle (repeatable).")
@click.option("--weeks", type=int, default=4, show_default=True)
@click.option("--per-week", type=int, default=3, show_default=True)
@click.option("--url", default="", help="Your story's link.")
@click.option("--send", "do_send", is_flag=True, help="Actually publish via each platform's official API (needs credentials in the environment).")
@click.pass_obj
def market_social(c: Ctx, project: str, accounts: tuple, weeks: int, per_week: int, url: str, do_send: bool) -> None:
    """Plan a social content calendar for accounts you OWN: distinct posts, varied by angle, disclosed.
    This previews what would post (a dry run); Gauge ships no live poster and never posts to accounts you
    don't own or varies content to dodge spam filters. Live posting plugs in via each platform's official API."""
    from .market.launchkit import build_launch_kit
    from .market.social import OwnedAccount, calendar_markdown, content_calendar, publish

    if not accounts:
        console.print("[yellow]name at least one account you own[/], e.g. --account mastodon:yourhandle")
        return
    owned = []
    for a in accounts:
        platform, _, handle = a.partition(":")
        if not handle:
            console.print(f"[red]bad account {a!r}[/]; use platform:handle")
            sys.exit(1)
        owned.append(OwnedAccount(platform.strip(), handle.strip(), official_api=True, i_own_this=True))
    pipe = _pipeline_or_exit(c)
    proj = pipe.open_project(project)
    kit = build_launch_kit(pipe.llm, proj)
    try:
        cal = content_calendar(kit, owned, weeks=weeks, per_week=per_week, url=url)
    except PermissionError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)
    out = proj.dir / "social_calendar.md"
    out.write_text(calendar_markdown(cal), encoding="utf-8")
    from .market.posters import RoutingPoster
    poster = RoutingPoster(dry_run=not do_send)
    results = publish(cal, poster)
    sent = sum(1 for r in results if r.get("posted"))
    previewed = sum(1 for r in results if r.get("dry_run"))
    console.print(f"[green]planned[/] {len(cal)} posts across {len(owned)} of your accounts -> {out}")
    console.print(f"{'sent ' + str(sent) + ' post(s)' if do_send else 'dry run: ' + str(previewed) + ' previewed'}, "
                  f"{'' if do_send else '0 sent'}")
    for r in results[:6]:
        if r.get("refused"):
            console.print(f"  [yellow]refused:[/] {r['refused']}")
        elif r.get("error"):
            console.print(f"  [red]error:[/] {r['error']}")
        elif r.get("note"):
            console.print(f"  [{r.get('platform')}] not sent: {r['note']}")
        else:
            console.print(f"  [{r.get('platform')}] {r.get('angle', '')}: {r.get('preview', r.get('id', 'ok'))}"[:90])
    if not do_send:
        console.print("This was a dry run. Add --send with your platform credentials in the environment to publish to your own accounts.")


@market.group()
def subscribers() -> None:
    """Your opt-in mailing list: people who signed up. Import, list, and honour unsubscribes."""


@subscribers.command("import")
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def subscribers_import(c: Ctx, path: str) -> None:
    """Import a list you exported of people who signed up (CSV/JSON). Rows must carry consent (a
    signup/consent date or an opt-in flag); rows without it are refused, not stored."""
    import csv as _csv
    import json as _json

    p = Path(path)
    if p.suffix.lower() == ".json":
        rows = _json.loads(p.read_text(encoding="utf-8"))
        rows = rows if isinstance(rows, list) else rows.get("subscribers", [])
    else:
        with p.open(encoding="utf-8", newline="") as fh:
            rows = list(_csv.DictReader(fh))
    from .market.email_list import import_subscribers

    s = import_subscribers(c.db, rows, source=p.name)
    console.print(f"[green]added {s['added']}[/]  already {s['already']}  kept-unsubscribed {s['kept_unsubscribed']}  "
                  f"[yellow]refused: no-consent {s['no_consent']}, invalid-email {s['invalid_email']}[/]")


@subscribers.command("list")
@click.pass_obj
def subscribers_list(c: Ctx) -> None:
    """Counts by status."""
    counts = c.db.subscriber_counts()
    console.print("subscribers: " + (", ".join(f"{k} {v}" for k, v in counts.items()) or "(none yet)"))


@subscribers.command("unsubscribe")
@click.argument("email")
@click.pass_obj
def subscribers_unsub(c: Ctx, email: str) -> None:
    """Mark one address unsubscribed (it will never be re-added by a later import)."""
    from .market.email_list import unsubscribe

    ok = unsubscribe(c.db, email)
    console.print(f"[green]unsubscribed[/] {email}" if ok else f"[yellow]no such subscriber[/] {email}")


@market.command("newsletter")
@click.argument("project")
@click.option("--url", default="", help="The link to the latest chapter / the story.")
@click.option("--chapter", "chapter_title", default="", help="Latest chapter title, if announcing one.")
@click.option("--from-name", "from_name", default="", help="Sender name (required to send).")
@click.option("--from-email", "from_email", default="", help="Sender email (required to send).")
@click.option("--address", default="", help="Your postal address (CAN-SPAM requires one in commercial email).")
@click.option("--unsubscribe-url", "unsub", default="", help="Base URL for one-click unsubscribe.")
@click.option("--send", "do_send", is_flag=True, help="Actually send via SMTP (needs GAUGE_SMTP_* in the environment).")
@click.pass_obj
def market_newsletter(c: Ctx, project: str, url: str, chapter_title: str, from_name: str, from_email: str,
                      address: str, unsub: str, do_send: bool) -> None:
    """Email your opt-in list about a new chapter. Dry run by default: previews to consented subscribers,
    sends nothing. Every message carries your identity and an unsubscribe link."""
    from .market.email_list import Sender, build_newsletter, send_newsletter
    from .market.launchkit import build_launch_kit

    pipe = _pipeline_or_exit(c)
    proj = pipe.open_project(project)
    kit = build_launch_kit(pipe.llm, proj)
    nl = build_newsletter(kit, url=url, chapter_title=chapter_title)
    sender = Sender(from_name or "the author", from_email or "", postal_address=address)
    subscribed = len(c.db.subscribers(status="subscribed"))
    console.print(f"[bold]{nl.subject}[/]")
    if not do_send:
        from .market.email_list import render_message
        preview = render_message(nl, "subscriber@example.com", sender, unsub or "https://your-site/unsubscribe")
        console.print("--- preview (not sent) ---")
        console.print(preview)
        console.print("--------------------------")
        console.print(f"would go to {subscribed} consented, subscribed recipient(s). DRY RUN, 0 sent.")
        console.print("To send: --from-name, --from-email, --unsubscribe-url, and --send with GAUGE_SMTP_* in the environment.")
        return
    import os
    host = os.environ.get("GAUGE_SMTP_HOST")
    if not host:
        console.print("[red]--send needs GAUGE_SMTP_HOST (and GAUGE_SMTP_USER / GAUGE_SMTP_PASSWORD) in the environment[/]")
        sys.exit(1)
    from .market.email_list import SMTPMailer
    mailer = SMTPMailer(host, int(os.environ.get("GAUGE_SMTP_PORT", "587")),
                        os.environ.get("GAUGE_SMTP_USER", ""), os.environ.get("GAUGE_SMTP_PASSWORD", ""))
    try:
        res = send_newsletter(c.db, nl, sender, unsub, mailer)
    except PermissionError as e:
        console.print(f"[red]refused:[/] {e}")
        sys.exit(1)
    console.print(f"sent {res['sent']} of {res['recipients']} consented subscriber(s); {res['failed']} failed.")


@main.command()
@click.option("--project", default=None, help="Limit to one project.")
@click.option("--days", type=int, default=None, help="Only the last N days.")
@click.pass_obj
def profile(c: Ctx, project: Optional[str], days: Optional[int]) -> None:
    """Where the time and tokens go: per-purpose LLM cost from telemetry, plus the slowest chapters."""
    rows = c.db.call_summary(project=project, limit_days=days)
    if not rows:
        console.print("no telemetry yet; write a few chapters first")
        return
    table = Table(title="LLM cost by purpose")
    for col in ("purpose", "model", "calls", "seconds", "avg_seconds", "prompt_tokens", "output_tokens", "failed"):
        table.add_column(col, justify="right" if col not in ("purpose", "model") else "left")
    tot_s = tot_c = tot_pt = tot_ot = 0.0
    for r in rows:
        table.add_row(r["purpose"], r["model"] or "-", str(r["calls"]), f"{r['seconds']:.1f}", f"{r['avg_seconds']:.1f}",
                      str(r["prompt_tokens"] or 0), str(r["output_tokens"] or 0), str(r["failed"]))
        tot_s += r["seconds"] or 0
        tot_c += r["calls"] or 0
        tot_pt += r["prompt_tokens"] or 0
        tot_ot += r["output_tokens"] or 0
    console.print(table)
    console.print(f"total: {int(tot_c)} calls, {tot_s:.0f}s in model, {int(tot_pt)} prompt + {int(tot_ot)} output tokens")
    # Slowest chapters, from each chapter's stored timing/seconds, when a project is given.
    if project:
        from .write.project import Project
        if Project.exists(c.cfg.projects_dir, project):
            proj = Project(c.cfg.projects_dir, project)
            chapters = []
            for n in range(1, proj.state.written + 1):
                meta = proj.read_meta(n)
                if isinstance(meta.get("seconds"), (int, float)):
                    chapters.append((n, float(meta["seconds"])))
            if chapters:
                chapters.sort(key=lambda t: -t[1])
                console.print("slowest chapters: " + ", ".join(f"ch{n} ({s:.0f}s)" for n, s in chapters[:5]))


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


@main.command("build-sft")
@click.argument("project")
@click.pass_obj
def build_sft(c: Ctx, project: str) -> None:
    """Curate instruction-tuning data from a serial's own accepted chapters into datasets/sft.jsonl,
    so GaugeLM can learn from what Gauge itself judged good."""
    from .learn.finetune import build_sft_from_project
    from .write.project import Project

    if not Project.exists(c.cfg.projects_dir, project):
        console.print(f"[red]no project named {project!r}[/]")
        sys.exit(1)
    n = build_sft_from_project(Project(c.cfg.projects_dir, project), c.cfg.models_dir / "datasets")
    console.print(f"[green]wrote {n}[/] SFT row(s) from accepted chapters -> datasets/sft.jsonl")


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


# ---------------------------------------------------------------------------- corpus (training library)
@main.group()
def corpus() -> None:
    """Compile the training library the AI learns from — only text you may train on, licence-tracked."""


@corpus.command("stats")
@click.option("--include-third-party", is_flag=True, help="Also count scraped third-party fiction (copyrighted; off by default).")
@click.pass_obj
def corpus_stats(c: Ctx, include_third_party: bool) -> None:
    """Show what the training library would contain, by source and licence, with excluded counts."""
    from .learn.corpus import compile_corpus

    rep = compile_corpus(c.db, c.cfg.projects_dir, include_third_party=include_third_party)
    console.print(f"[bold]Training library[/]: {len(rep.docs)} documents, ~{rep.est_tokens:,} tokens")
    for src, info in sorted(rep.by_source().items(), key=lambda kv: -kv[1]["chars"]):
        console.print(f"  {src:<12} {info['docs']:>4} docs  {info['chars'] // 1000:>6}k chars  [{info['permission']}]  {info['note']}")
    if rep.excluded:
        console.print(f"[yellow]excluded[/] {len(rep.excluded)} third-party docs from {', '.join(sorted({d.source for d in rep.excluded}))} "
                      "(copyrighted; use --include-third-party only for text you hold the rights to)")


@corpus.command("build")
@click.option("--out", default=None, help="Output directory (default: data/models/corpus).")
@click.option("--include-third-party", is_flag=True, help="Include scraped third-party fiction (copyrighted; your legal call).")
@click.pass_obj
def corpus_build(c: Ctx, out: Optional[str], include_third_party: bool) -> None:
    """Compile and export the training library (corpus.jsonl + corpus.txt + a licence MANIFEST)."""
    from .learn.corpus import compile_corpus, export_corpus

    rep = compile_corpus(c.db, c.cfg.projects_dir, include_third_party=include_third_party)
    if not rep.docs:
        console.print("[yellow]nothing to compile[/]: import public-domain books (`gauge scout gutenberg`), your own "
                      "files (`gauge import`), or write chapters first.")
        return
    dest = Path(out) if out else (c.cfg.models_dir / "corpus")
    manifest = export_corpus(rep, dest)
    console.print(f"[green]compiled[/] {manifest['documents']} documents (~{manifest['est_tokens']:,} tokens) -> {dest}")
    console.print(f"  sources: {', '.join(manifest['by_source'])}")
    if include_third_party:
        console.print("[yellow]note:[/] third-party copyrighted text was included at your request; ensure you hold the rights.")


# ---------------------------------------------------------------------------- native model (GaugeLM)
@main.group()
def native() -> None:
    """GaugeLM: Gauge's own language model. Pure Python + numpy, trained on your corpus, no outside model code."""


def _native_docs(c: Ctx, since: Optional[str] = None):
    from .native import gather_texts

    return gather_texts(c.db, c.cfg.projects_dir, c.cfg.models_dir / "datasets", since=since)


@native.command("train")
@click.option("--size", type=click.Choice(["tiny", "small", "medium", "large"]), default=None, help="Preset architecture.")
@click.option("--steps", type=int, default=None)
@click.option("--layers", type=int, default=None)
@click.option("--embd", type=int, default=None)
@click.option("--heads", type=int, default=None)
@click.option("--kv-heads", type=int, default=None, help="Grouped-query attention: fewer key/value heads.")
@click.option("--block", type=int, default=None)
@click.option("--vocab", type=int, default=None)
@click.option("--batch", type=int, default=None)
@click.option("--accum", type=int, default=None, help="Gradient accumulation steps.")
@click.option("--arch", type=click.Choice(["llama", "gpt"]), default=None)
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default=None)
@click.option("--resume", is_flag=True, help="Continue from the saved weights instead of starting over.")
@click.option("--minutes", type=float, default=None, help="Stop after this many minutes (checkpoint is saved).")
@click.option("--distributed", is_flag=True, help="After local training, run model-averaging rounds with live workers.")
@click.option("--rounds", type=int, default=3, show_default=True, help="Distributed rounds (with --distributed).")
@click.pass_obj
def native_train(c: Ctx, size, steps, layers, embd, heads, kv_heads, block, vocab, batch, accum, arch, device, resume, minutes,
                 distributed, rounds) -> None:
    """Train (or keep training) GaugeLM on everything Gauge has read plus your own chapters."""
    from .native import ModelConfig, NativeTrainer, TrainConfig, preset_config

    n = c.cfg.native
    if size or (n.size and not any([layers, embd, heads, block])):
        mcfg = preset_config(size or n.size, vocab)
    else:
        mcfg = ModelConfig(vocab_size=vocab or n.vocab_size, block_size=block or n.block_size, n_layer=layers or n.n_layer,
                           n_head=heads or n.n_head, n_embd=embd or n.n_embd, n_kv_head=n.n_kv_head)
    if layers:
        mcfg.n_layer = layers
    if embd:
        mcfg.n_embd = embd
    if heads:
        mcfg.n_head = heads
    if kv_heads is not None:
        mcfg.n_kv_head = kv_heads
    if block:
        mcfg.block_size = block
    if vocab:
        mcfg.vocab_size = vocab
    mcfg.arch = arch or n.arch
    tcfg = TrainConfig(steps=steps or n.steps, batch_size=batch or n.batch_size, lr=n.lr, accum_steps=accum or n.accum_steps,
                       chat_weight=n.chat_weight, device=device or n.device)
    docs = _native_docs(c)
    if not docs:
        console.print("[red]nothing to train on[/]: ingest or import some novels first")
        sys.exit(1)
    chars = sum(len(t) for _, t in docs)
    console.print(f"corpus: {len(docs)} documents, {chars:,} characters. model: {mcfg.arch}, {mcfg.n_layer} layers x {mcfg.n_embd} dims, "
                  f"{mcfg.n_head} heads ({mcfg.kv_heads} kv), block {mcfg.block_size}, effective batch {tcfg.batch_size * tcfg.accum_steps}")
    console.print(f"device: {tcfg.device}. Run `gauge native bench --size {size or n.size}` first to see how long this will take. Ctrl+C keeps the last checkpoint.")
    tr = NativeTrainer(c.cfg.native_dir)

    def progress(rec):
        console.print(f"  step {rec['step']:>6}  loss {rec['loss']:.3f}  val {rec['val_loss']:.3f}  {rec['tok_per_s']} tok/s")

    try:
        res = tr.train(docs, mcfg, tcfg, resume=resume, progress=progress, time_budget_s=minutes * 60 if minutes else None)
    except KeyboardInterrupt:
        console.print("[yellow]stopped; last checkpoint kept[/]")
        return
    console.print_json(json.dumps(res))
    if distributed:
        from .compute import DistributedTrainer, Pool
        from .native.train import encode_corpus

        pool = Pool(c.db, c.cfg.compute.token)
        model, tok = tr.load(device=tcfg.device)
        tokens = encode_corpus(tok, docs, tcfg.max_corpus_chars)
        dt = DistributedTrainer(pool, tcfg.device)
        for r in range(rounds):
            try:
                out = dt.round(model, tokens, steps=max(10, (steps or n.steps) // 10), lr=n.lr / 3, batch_size=tcfg.batch_size)
            except Exception as e:
                console.print(f"[yellow]distributed round {r + 1} skipped[/]: {e}")
                break
            console.print(f"  round {r + 1}: participants {out['participants']} loss {out['loss']}")
            model.save(c.cfg.native_dir, {"step": res["step"], "distributed_rounds": r + 1})
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
                               progress=lambda r: console.print(f"  step {r['step']} loss {r['loss']:.3f} val {r['val_loss']:.3f}"),
                               device=c.cfg.native.device)
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

    llm = NativeLLM(c.cfg.llm, c.cfg.native_dir, device=c.cfg.native.device)
    if not llm.available():
        console.print("[red]no native model yet[/]; run `gauge native train`")
        sys.exit(1)
    console.print(llm.generate(prompt, max_tokens=tokens, temperature=temperature, purpose="sample"))


@native.command("presets")
def native_presets() -> None:
    """The preset sizes and their parameter counts."""
    from .native.evaluate import preset_table

    table = Table(title="GaugeLM presets")
    for col in ("size", "layers", "embd", "heads", "kv", "block", "vocab", "params (M)"):
        table.add_column(col)
    for r in preset_table():
        table.add_row(r["size"], str(r["n_layer"]), str(r["n_embd"]), str(r["n_head"]), str(r["n_kv_head"]), str(r["block_size"]),
                      str(r["vocab_size"]), str(r["params_millions"]))
    console.print(table)
    console.print("tiny trains on a CPU in hours; small needs a day on CPU or an hour on a GPU; medium and large want a GPU (pip install cupy-cuda12x).")


@native.command("bench")
@click.option("--size", type=click.Choice(["tiny", "small", "medium", "large"]), default="tiny", show_default=True)
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default=None)
@click.option("--batch", type=int, default=4, show_default=True)
@click.pass_obj
def native_bench(c: Ctx, size: str, device: Optional[str], batch: int) -> None:
    """Measure training and generation speed for a preset on this machine, and the hours a run would take."""
    from .native.evaluate import bench

    console.print(f"benchmarking {size} on {device or c.cfg.native.device} ...")
    console.print_json(json.dumps(bench(size, device or c.cfg.native.device, batch_size=batch)))


@native.command("distill")
@click.option("--rows", type=int, default=None, help="Teacher answers to collect (default native.distill_rows).")
@click.pass_obj
def native_distill(c: Ctx, rows: Optional[int]) -> None:
    """Collect answers from the big local model on Gauge's own tasks, so GaugeLM can learn the behaviour."""
    from .llm import make_llm
    from .native.distill import distill, distill_summary

    if c.cfg.llm.provider == "native":
        console.print("[red]the teacher must be a bigger model[/]: set llm.provider to ollama/openai (gauge init --provider ollama) for distillation")
        sys.exit(1)
    teacher = make_llm(c.cfg.llm, "main")
    if not teacher.available():
        console.print(f"[red]teacher not reachable[/] ({teacher.describe()})")
        sys.exit(1)
    out = c.cfg.models_dir / "datasets" / "distill.jsonl"
    res = distill(c.db, teacher, out, rows or c.cfg.native.distill_rows,
                  progress=lambda r: console.print(f"  {r['written']} rows ({r['failed']} failed)"))
    console.print_json(json.dumps(res))
    console.print(f"by task: {distill_summary(out)}\nNext: gauge native train --resume  (instruction rows are upweighted)")


@native.command("eval")
@click.option("--probe/--no-probe", default=True, help="Also run the instruction-shape probes.")
@click.pass_obj
def native_eval(c: Ctx, probe: bool) -> None:
    """Held-out perplexity and instruction-shape probes for the native model."""
    from .native import NativeLLM
    from .native.evaluate import evaluate_model

    texts = [ch.text for n in c.db.novels()[:20] for ch in c.db.chapters(n.key)[-1:]]
    llm = NativeLLM(c.cfg.llm, c.cfg.native_dir, device=c.cfg.native.device) if probe else None
    if llm is not None and not llm.available():
        console.print("[red]no native model yet[/]")
        sys.exit(1)
    console.print_json(json.dumps(evaluate_model(c.cfg.native_dir, texts, llm, device=c.cfg.native.device), default=str))


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


# ---------------------------------------------------------------------------- compute (GPUs and your other machines)
@main.group()
def compute() -> None:
    """Sense GPUs on this machine and use the other machines you own (gauge worker)."""


@compute.command("scan")
@click.option("--apply", is_flag=True, help="Save the best device into native.device.")
@click.pass_obj
def compute_scan(c: Ctx, apply: bool) -> None:
    """List CPU and GPUs, which backend could drive each, and what to install if one is missing."""
    from .compute import scan, summary

    devs = scan()
    s = summary(devs)
    table = Table(title="Compute on this machine")
    for col in ("kind", "vendor", "name", "memory", "backend", "ready", "source"):
        table.add_column(col)
    for d in devs:
        table.add_row(d.kind, d.vendor, d.name[:48], f"{d.memory_mb} MB" if d.memory_mb else "", d.backend, "yes" if d.backend_ready else "no", d.source)
    console.print(table)
    for d in devs:
        if d.install_hint:
            console.print(f"  [yellow]{d.name}[/]: {d.install_hint}")
    console.print(f"GaugeLM would use: [bold]{s['use']}[/] ({s['chosen']})  backends: {s['backends']}")
    if apply:
        c.cfg.native.device = s["use"]
        c.cfg.save()
        console.print(f"[green]saved[/] native.device = {s['use']}")


@compute.command("watch")
@click.option("--interval", type=float, default=30.0, show_default=True)
@click.option("--max-minutes", type=float, default=None)
@click.pass_obj
def compute_watch(c: Ctx, interval: float, max_minutes: Optional[float]) -> None:
    """Wait for a usable GPU to appear (e.g. an external GPU plugged in), then switch GaugeLM to it."""
    from .compute import summary, watch

    def on_change(devs):
        console.print("devices now: " + ", ".join(f"{d.name}{' (ready)' if d.backend_ready else ''}" for d in devs if d.kind == "gpu"))

    console.print(f"watching for a GPU every {interval:.0f}s ... Ctrl+C to stop")
    try:
        devs = watch(on_change, interval_s=interval, max_seconds=max_minutes * 60 if max_minutes is not None else None)
    except KeyboardInterrupt:
        return
    if devs:
        use = summary(devs)["use"]
        c.cfg.native.device = use
        c.cfg.save()
        console.print(f"[green]GPU ready[/]; native.device = {use}")
    else:
        console.print("no usable GPU appeared")


@compute.command("token")
@click.option("--rotate", is_flag=True, help="Generate a new token (restart workers with it).")
@click.option("--reveal", is_flag=True, help="Print the full token (default shows only the last 4 characters).")
@click.pass_obj
def compute_token(c: Ctx, rotate: bool, reveal: bool) -> None:
    """Generate, rotate or show the shared token your workers must be started with."""
    import secrets

    if rotate or not c.cfg.compute.token:
        c.cfg.compute.token = secrets.token_urlsafe(24)
        c.cfg.save()
        try:
            _audit(c).event("token_rotated" if rotate else "token_created", "worker token", 2)
        except Exception:
            pass
    tok = c.cfg.compute.token
    shown = tok if reveal else "*" * 8 + tok[-4:]
    console.print(f"token: {shown}" + ("" if reveal else "   (add --reveal to print it)") + "\nStart a worker on another machine you own with:\n"
                  "  gauge worker --token <token> --name \"office-pc\"")


@compute.command("discover")
@click.option("--seconds", type=float, default=None)
@click.pass_obj
def compute_discover(c: Ctx, seconds: Optional[float]) -> None:
    """Listen for your workers' announcements on the LAN and register them (no scanning, no probing)."""
    from .compute import Pool

    if not c.cfg.compute.token:
        console.print("[yellow]no token yet[/]: run `gauge compute token` first and start workers with it")
        sys.exit(1)
    found = Pool(c.db, c.cfg.compute.token).discover_and_register(seconds or c.cfg.compute.discover_seconds, c.cfg.compute.announce_port)
    if not found:
        console.print("no workers announced themselves (are they started with the same token, on this network?)")
        return
    for f in found:
        console.print(f"[green]found[/] {f['name']} at {f['url']}  llm={f['llm'] or '-'}  devices={[d.get('name') for d in f['devices']]}")


@compute.command("add")
@click.argument("url")
@click.option("--name", default="")
@click.pass_obj
def compute_add(c: Ctx, url: str, name: str) -> None:
    """Register a worker by URL (for machines on another network segment)."""
    from .compute import Pool, WorkerClient

    if not c.cfg.compute.token:
        console.print("[yellow]no token yet[/]: run `gauge compute token`")
        sys.exit(1)
    h = WorkerClient(url, c.cfg.compute.token).health()
    if h is None:
        console.print(f"[red]no worker answered at {url}[/] (wrong token, or not running)")
        sys.exit(1)
    Pool(c.db, c.cfg.compute.token).add(url, name or h.get("name", url), h.get("devices", []), h.get("llm", ""))
    console.print(f"[green]added[/] {h.get('name')} at {url}: {h.get('use')} {h.get('llm') or ''}")


@compute.command("list")
@click.pass_obj
def compute_list(c: Ctx) -> None:
    """Registered workers and whether they answer right now."""
    from .compute import Pool

    pool = Pool(c.db, c.cfg.compute.token)
    alive = {w["url"] for w in pool.refresh()} if c.cfg.compute.token else set()
    rows = pool.workers()
    if not rows:
        console.print("(no workers; `gauge compute discover` or `gauge compute add URL`)")
        return
    table = Table(title="Compute pool")
    for col in ("name", "url", "alive", "llm", "devices", "last seen"):
        table.add_column(col)
    for w in rows:
        devs = ", ".join(d.get("name", "?")[:24] for d in json.loads(w["devices_json"] or "[]"))
        table.add_row(w["name"], w["url"], "yes" if w["url"] in alive else "no", (w["llm"] or "")[:30], devs, (w["last_seen"] or "")[:16])
    console.print(table)


@compute.command("remove")
@click.argument("url")
@click.pass_obj
def compute_remove(c: Ctx, url: str) -> None:
    """Forget a worker."""
    from .compute import Pool

    Pool(c.db, c.cfg.compute.token).remove(url)
    console.print(f"[green]removed[/] {url}")


@main.command()
@click.option("--token", required=True, help="The shared token from `gauge compute token` on the coordinator.")
@click.option("--name", default=None)
@click.option("--port", type=int, default=None)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--no-llm", is_flag=True, help="Lend compute for training only; do not serve a model.")
@click.option("--no-announce", is_flag=True)
@click.pass_obj
def worker(c: Ctx, token: str, name: Optional[str], port: Optional[int], host: str, no_llm: bool, no_announce: bool) -> None:
    """Lend this machine to your other Gauge installation: serves its model and runs training shards."""
    import socket

    from .compute import Worker
    from .llm import make_llm

    llm = None
    if not no_llm:
        cand = make_llm(c.cfg.llm)
        llm = cand if cand.available() else None
    w = Worker(c.cfg, name or socket.gethostname(), token, host=host, port=port or c.cfg.compute.worker_port, llm=llm,
               announce=not no_announce, announce_port=c.cfg.compute.announce_port)
    console.print(f"worker [bold]{w.state.name}[/] on {host}:{w.port}  model: {llm.describe() if llm else '(none, training only)'}  "
                  f"devices: {[d.name for d in w.state.devices]}")
    console.print("announcing on the LAN; the coordinator finds this with `gauge compute discover`. Ctrl+C to stop.")
    w.serve()


@main.command()
@click.option("--out", default=None, help="Output path (default: gauge-backup-<date>.tar.gz).")
@click.option("--include-models", is_flag=True, help="Include model weights (large; reproducible by retraining).")
@click.pass_obj
def backup(c: Ctx, out: Optional[str], include_models: bool) -> None:
    """Back up the data directory to a verified tar.gz (projects, config, security seal, optionally models)."""
    from datetime import date

    from .ops import make_backup

    dest = Path(out) if out else Path(f"gauge-backup-{date.today().isoformat()}.tar.gz")
    res = make_backup(c.cfg.root, dest, include_models=include_models)
    console.print(f"[green]backed up[/] {res.files} files ({res.bytes // 1024} KiB) -> {res.path}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--into", default=None, help="Restore destination (default: a new ./restored-<name> dir).")
@click.option("--verify-only", is_flag=True, help="Only check integrity; do not write anything.")
@click.option("--overwrite", is_flag=True, help="Allow restoring into a non-empty directory.")
@click.pass_obj
def restore(c: Ctx, path: str, into: Optional[str], verify_only: bool, overwrite: bool) -> None:
    """Verify a backup's integrity and restore it. Refuses a corrupt archive."""
    from .ops import restore_backup, verify_backup

    v = verify_backup(Path(path))
    console.print(f"integrity: {'[green]OK[/]' if v.ok else '[red]FAILED[/]'}  ({v.files} files, created {v.created})")
    if not v.ok:
        console.print(f"  mismatched: {v.mismatched[:5]}  missing: {v.missing[:5]}")
        sys.exit(1)
    if verify_only:
        return
    dest = Path(into) if into else Path(f"restored-{Path(path).stem}")
    r = restore_backup(Path(path), dest, overwrite=overwrite)
    console.print(f"[green]restored[/] {r['restored']} files -> {r['into']}")


@main.command()
@click.pass_obj
def audit(c: Ctx) -> None:
    """Plain-language report of every automation: what is on, what it can do, what bounds it."""
    from .ops import render_report

    console.print(render_report(c.cfg))


@main.command(name="export")
@click.argument("project")
@click.option("--format", "fmt", type=click.Choice(["markdown", "html", "bbcode", "text"]), default="markdown", show_default=True)
@click.option("--out", default=None, help="Output directory (default: ./export-<project>-<format>).")
@click.pass_obj
def export_chapters(c: Ctx, project: str, fmt: str, out: Optional[str]) -> None:
    """Export a serial's chapters to Markdown, HTML, BBCode or plain text for Royal Road, Scribble Hub, etc."""
    from .site.export import export_project
    from .write.project import Project

    if not Project.exists(c.cfg.projects_dir, project):
        console.print(f"[red]no project named {project!r}[/]")
        sys.exit(1)
    dest = Path(out) if out else Path(f"export-{project.lower().replace(' ', '-')}-{fmt}")
    files = export_project(Project(c.cfg.projects_dir, project), fmt, dest)
    console.print(f"[green]exported[/] {len(files)} file(s) in {fmt} -> {dest}" if files
                  else "no chapters to export yet")


@main.command()
@click.option("--out", default="gauge-dashboard.html", show_default=True, help="Output HTML path.")
@click.pass_obj
def dashboard(c: Ctx, out: str) -> None:
    """Write a static telemetry dashboard (cost by purpose, chapter scores over time) you open in a browser."""
    from .ops import dashboard_html

    html_text = dashboard_html(c.db, c.cfg.projects_dir)
    Path(out).write_text(html_text, encoding="utf-8")
    console.print(f"[green]wrote[/] {out}  (open it in a browser)")


@main.command()
@click.option("--host", default=None, help="Bind address. Default 127.0.0.1 (loopback). 0.0.0.0 exposes it to the network.")
@click.option("--port", type=int, default=None, help="Port (default 8800 or server.port).")
@click.option("--token", default=None, help="API token. Default: server.token, else a generated one printed at start.")
@click.pass_obj
def serve(c: Ctx, host: Optional[str], port: Optional[int], token: Optional[str]) -> None:
    """Run Gauge headless with an HTTP control API and a web dashboard, so you can drive it from another
    machine, a phone, a server or a container. Loopback and token-gated by default; see docs/DEPLOY.md."""
    from .server import serve as run_server

    try:
        run_server(c.cfg, host=host, port=port, token=token)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)


# ---------------------------------------------------------------------------- code: bots, knowledge, self
def _kb(c: Ctx, with_self: bool = False):
    from .code import KnowledgeBase, repo_root

    kb = KnowledgeBase(extra_dir=c.cfg.knowledge_dir)
    if with_self:
        kb.index_self(repo_root() / "gauge")
    return kb


@main.group()
def bot() -> None:
    """Bots: standalone programs Gauge writes and tests; they run without a model unless the task needs one."""


@bot.command("new")
@click.argument("name")
@click.option("--task", required=True, help="What the bot should do, in plain words.")
@click.option("--language", type=click.Choice(["python", "powershell", "batch", "javascript"]), default=None)
@click.option("--needs-ai/--no-ai", default=None, help="Override the Engineer's judgement about needing a model.")
@click.option("--schedule", default=None, help="hourly | daily | every:N (minutes) | on-cycle | '' ")
@click.pass_obj
def bot_new(c: Ctx, name: str, task: str, language: Optional[str], needs_ai: Optional[bool], schedule: Optional[str]) -> None:
    """Have the Engineer write, test and register a bot."""
    from .code import Engineer
    from .llm import make_llm

    llm = make_llm(c.cfg.llm, "main")
    if not llm.available():
        console.print(f"[red]LLM not reachable[/] ({llm.describe()}); the Engineer needs a model to write code (the bot itself may not).")
        sys.exit(1)
    eng = Engineer(llm, _kb(c), c.cfg.bots_dir, max_fix_rounds=c.cfg.code.max_fix_rounds, scan_dependencies=c.cfg.security.scan_dependencies)
    res = eng.build(name, task, needs_ai=needs_ai, language=language, schedule=schedule,
                    progress=lambda r: console.print(f"  attempt {r['attempt']}: tests {'passed' if r['passed'] else 'failed'}"))
    console.print_json(json.dumps(res))
    if not res["tests_passed"]:
        console.print("[yellow]the bot's tests did not pass; it is saved for you to inspect but will not be scheduled[/]")


@bot.command("list")
@click.pass_obj
def bot_list(c: Ctx) -> None:
    """Registered bots, schedules and last results."""
    from .code import BotRegistry

    rows = BotRegistry(c.cfg.bots_dir).list()
    if not rows:
        console.print("(no bots; `gauge bot new NAME --task ...`)")
        return
    table = Table(title="Bots")
    for col in ("name", "language", "ai", "schedule", "tests", "runs", "last run", "status", "task"):
        table.add_column(col)
    for m in rows:
        table.add_row(m["name"], m.get("language", ""), "yes" if m.get("needs_ai") else "no", m.get("schedule") or "-",
                      "ok" if m.get("tests_passed") else "FAIL", str(m.get("runs", 0)), (m.get("last_run") or "")[:16],
                      str(m.get("last_status") if m.get("last_status") is not None else ""), m.get("task", "")[:50])
    console.print(table)


@bot.command("run")
@click.argument("name")
@click.argument("args", nargs=-1)
@click.pass_obj
def bot_run(c: Ctx, name: str, args: tuple) -> None:
    """Run a bot now (deterministic bots cannot reach the model)."""
    from .code import BotRegistry

    reg = BotRegistry(c.cfg.bots_dir)
    if not reg.exists(name):
        console.print(f"[red]no bot {name!r}[/]")
        sys.exit(1)
    res = reg.run(name, list(args), env_extra={"GAUGE_DATA_DIR": str(c.cfg.root)})
    console.print(res["stdout"] or "")
    if res["stderr"]:
        console.print(f"[dim]{res['stderr']}[/]")
    console.print(f"exit {res['exit']} in {res['seconds']}s")
    sys.exit(res["exit"])


@bot.command("test")
@click.argument("name")
@click.pass_obj
def bot_test(c: Ctx, name: str) -> None:
    """Run a bot's own tests."""
    from .code import BotRegistry, run_tests

    reg = BotRegistry(c.cfg.bots_dir)
    ok, report = run_tests(reg.path(name))
    console.print(report)
    console.print("[green]passed[/]" if ok else "[red]failed[/]")
    sys.exit(0 if ok else 1)


@bot.command("schedule")
@click.argument("name")
@click.argument("schedule")
@click.pass_obj
def bot_schedule(c: Ctx, name: str, schedule: str) -> None:
    """Set when the daemon runs a bot: hourly | daily | every:N | on-cycle | off."""
    from .code import BotRegistry

    reg = BotRegistry(c.cfg.bots_dir)
    if not reg.exists(name):
        console.print(f"[red]no bot {name!r}[/]")
        sys.exit(1)
    man = reg.manifest(name)
    man["schedule"] = "" if schedule == "off" else schedule
    (reg.path(name) / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    console.print(f"[green]{name}[/] schedule = {man['schedule'] or 'off'}")


@bot.command("remove")
@click.argument("name")
@click.pass_obj
def bot_remove(c: Ctx, name: str) -> None:
    """Delete a bot."""
    from .code import BotRegistry

    console.print("[green]removed[/]" if BotRegistry(c.cfg.bots_dir).remove(name) else f"[red]no bot {name!r}[/]")


@main.group()
def code() -> None:
    """What Gauge knows about programming: reference cards, your notes, and its own source."""


@code.command("cards")
@click.pass_obj
def code_cards(c: Ctx) -> None:
    """List the knowledge cards."""
    kb = _kb(c)
    for card in kb.cards:
        console.print(f"[bold]{card.key}[/]  {card.title}  [dim]({card.source}, {len(card.text)} chars)[/]")


@code.command("search")
@click.argument("query")
@click.option("-k", default=2, show_default=True)
@click.option("--self", "with_self", is_flag=True, help="Include Gauge's own source index.")
@click.pass_obj
def code_search(c: Ctx, query: str, k: int, with_self: bool) -> None:
    """Show the cards most relevant to a question."""
    kb = _kb(c, with_self)
    console.print(kb.render(kb.search(query, k)))


@code.command("learn")
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def code_learn(c: Ctx, path: str) -> None:
    """Add your own notes or a code file to the knowledge base (split on # headings)."""
    cards = _kb(c).learn_file(Path(path))
    console.print(f"[green]learned[/] {len(cards)} card(s): " + ", ".join(x.key for x in cards))


@code.command("lang")
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def code_lang(c: Ctx, path: str) -> None:
    """Read any source file: detect the language, measure its shape, flag smells, show idioms."""
    from .code.languages import analyze, tips

    code_text = Path(path).read_text(encoding="utf-8", errors="replace")
    shape = analyze(Path(path).name, code_text)
    console.print(f"[bold]{path}[/]  language={shape.language or 'unknown'}  lines={shape.lines}  tokens={shape.tokens}  "
                  f"functions={shape.functions}  classes={shape.classes}  comments={shape.comment_ratio:.0%}  max-nesting={shape.max_indent}")
    for s in shape.smells:
        console.print(f"  - {s}")
    if shape.language:
        console.print(tips(shape.language))


@code.command("binary")
@click.argument("path", type=click.Path(exists=True))
@click.option("--bytes", "n_bytes", type=int, default=256, show_default=True, help="How many bytes to dump.")
@click.pass_obj
def code_binary(c: Ctx, path: str, n_bytes: int) -> None:
    """Read any file at the binary level: identify it, hexdump it, measure its entropy."""
    from .code.binary import entropy, hexdump, identify

    data = Path(path).read_bytes()
    console.print(f"[bold]{path}[/]  {len(data)} bytes  type={identify(data)}  entropy={entropy(data[:65536])} bits/byte")
    console.print(hexdump(data, limit=n_bytes))


@code.command("disasm")
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def code_disasm(c: Ctx, path: str) -> None:
    """Disassemble a Python source file to bytecode."""
    from .code.binary import disassemble

    d = disassemble(Path(path).read_text(encoding="utf-8", errors="replace"))
    console.print(f"[bold]{d.name}[/]: {len(d.instructions)} instructions")
    console.print(d.text[:4000])


@main.group(name="self")
def self_group() -> None:
    """Self-modification with guardrails: propose -> test in isolation -> apply (human-approved by default) -> rollback."""


def _selfp(c: Ctx):
    from .code import SelfProgrammer
    from .llm import make_llm

    llm = make_llm(c.cfg.llm, "main")
    return SelfProgrammer(llm, _kb(c, with_self=True), c.cfg.self_dir,
                          mutation_check=c.cfg.code.mutation_check, mutation_min_score=c.cfg.code.mutation_min_score,
                          mutation_max_mutants=c.cfg.code.mutation_max_mutants)


@self_group.command("propose")
@click.argument("instruction")
@click.option("--file", "files", multiple=True, help="Restrict to these repo-relative files.")
@click.pass_obj
def self_propose(c: Ctx, instruction: str, files: tuple) -> None:
    """Draft a change to Gauge's own source as a reviewable proposal (nothing is modified)."""
    sp = _selfp(c)
    if not sp.llm.available():
        console.print(f"[red]LLM not reachable[/] ({sp.llm.describe()})")
        sys.exit(1)
    prop = sp.propose(instruction, list(files) or None)
    console.print(f"[bold]{prop.id}[/]  files: {list(prop.files)}  rationale: {prop.rationale}")
    if prop.meta.get("refused"):
        console.print("[yellow]refused parts:[/] " + "; ".join(prop.meta["refused"]))
    console.print(prop.diff[:4000] or "(no diff)")
    console.print(f"next: gauge self test {prop.id}")


@self_group.command("test")
@click.argument("proposal_id")
@click.pass_obj
def self_test(c: Ctx, proposal_id: str) -> None:
    """Run ruff and the whole test suite on an isolated copy with the proposal applied."""
    sp = _selfp(c)
    prop = sp.load(proposal_id)
    ok, report = sp.test(prop)
    console.print(report[-3000:])
    console.print("[green]passed[/] -> gauge self apply " + proposal_id if ok else "[red]failed[/]; not applicable")
    sys.exit(0 if ok else 1)


@self_group.command("apply")
@click.argument("proposal_id")
@click.option("--no-commit", is_flag=True)
@click.pass_obj
def self_apply(c: Ctx, proposal_id: str, no_commit: bool) -> None:
    """Apply a proposal that passed its tests; originals are backed up for rollback."""
    sp = _selfp(c)
    prop = sp.load(proposal_id)
    try:
        res = sp.apply(prop, commit=not no_commit)
    except PermissionError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)
    console.print_json(json.dumps(res))


@self_group.command("rollback")
@click.argument("proposal_id")
@click.pass_obj
def self_rollback(c: Ctx, proposal_id: str) -> None:
    """Restore the files a proposal changed."""
    console.print_json(json.dumps(_selfp(c).rollback(proposal_id)))


@self_group.command("list")
@click.pass_obj
def self_list(c: Ctx) -> None:
    """Proposals and their state."""
    rows = _selfp(c).list()
    if not rows:
        console.print("(no proposals)")
        return
    table = Table(title="Self proposals")
    for col in ("id", "source", "tested", "applied", "instruction"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["id"], r.get("source") or "", str(r.get("tested")), str(r.get("applied")), (r.get("instruction") or "")[:60])
    console.print(table)


@self_group.command("journal")
@click.option("--all", "show_all", is_flag=True)
@click.pass_obj
def self_journal(c: Ctx, show_all: bool) -> None:
    """Errors Gauge recorded about itself (what self-heal works from)."""
    rows = c.db.journal(unresolved_only=not show_all)
    if not rows:
        console.print("(journal is empty)")
        return
    for r in rows[:30]:
        console.print(f"{r['at'][:19]}  [{r['kind']}] {r['message'][:120]}")
    rec = c.db.recurring_errors(c.cfg.code.min_recurrences)
    if rec:
        console.print("\nrecurring: " + "; ".join(f"{x['message'][:60]} (x{x['n']})" for x in rec))


@self_group.command("heal")
@click.option("--apply", "do_apply", is_flag=True, help="Apply proposals that pass tests (otherwise they wait for you).")
@click.pass_obj
def self_heal(c: Ctx, do_apply: bool) -> None:
    """Turn recurring journal errors into tested proposals."""
    sp = _selfp(c)
    if not sp.llm.available():
        console.print(f"[red]LLM not reachable[/] ({sp.llm.describe()})")
        sys.exit(1)
    entries = c.db.recurring_errors(c.cfg.code.min_recurrences)
    if not entries:
        console.print("nothing recurring in the journal")
        return
    for r in sp.heal(entries, auto_apply=do_apply):
        console.print(f"proposal {r['proposal']}: files {r['files']} tested={r['tested']} applied={r.get('applied')}")


# ---------------------------------------------------------------------------- settings (the switches)
@main.group()
def settings() -> None:
    """The switches: flip any behaviour on or off. Same catalog as the Studio's Switches panel."""


@settings.command("list")
@click.option("--group", default=None, help="Writing | The room | Learning | Model | Automation")
@click.pass_obj
def settings_list(c: Ctx, group: Optional[str]) -> None:
    """Every switch with its current value."""
    from . import settings as st

    table = Table(title="Switches")
    for col in ("key", "value", "label", "group", "note"):
        table.add_column(col)
    for row in st.state(c.cfg):
        if group and row["group"].lower() != group.lower():
            continue
        note = ("[red]! " + row["warning"] + "[/]") if row["warning"] else row["description"]
        table.add_row(row["key"], str(row["value"]), row["label"], row["group"], note[:90])
    console.print(table)
    console.print("gauge settings on KEY | off KEY | set KEY VALUE | reset KEY | show KEY")


@settings.command("show")
@click.argument("key")
@click.pass_obj
def settings_show(c: Ctx, key: str) -> None:
    """Full description of one switch."""
    from . import settings as st

    sw = st.BY_KEY.get(key)
    if sw is None:
        console.print(f"[red]unknown switch {key!r}[/]")
        sys.exit(1)
    console.print(f"[bold]{sw.label}[/] ({sw.key}) = {st.get(c.cfg, key)}\n{sw.description}")
    if sw.kind in ("float", "int"):
        console.print(f"range {sw.minimum}..{sw.maximum}")
    if sw.choices:
        console.print("choices: " + ", ".join(sw.choices))
    if sw.warning:
        console.print(f"[red]! {sw.warning}[/]")


@settings.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_obj
def settings_set(c: Ctx, key: str, value: str) -> None:
    """Set a switch (validated, saved to config.json)."""
    from . import settings as st

    try:
        v = st.set_value(c.cfg, key, value)
    except (KeyError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)
    sw = st.BY_KEY[key]
    console.print(f"[green]{key}[/] = {v}" + (f"\n[red]! {sw.warning}[/]" if sw.warning and v not in (False, 0) else ""))


@settings.command("on")
@click.argument("key")
@click.pass_context
def settings_on(ctx: click.Context, key: str) -> None:
    """Turn a boolean switch on."""
    ctx.invoke(settings_set, key=key, value="on")


@settings.command("off")
@click.argument("key")
@click.pass_context
def settings_off(ctx: click.Context, key: str) -> None:
    """Turn a boolean switch off."""
    ctx.invoke(settings_set, key=key, value="off")


@settings.command("reset")
@click.argument("key")
@click.pass_obj
def settings_reset(c: Ctx, key: str) -> None:
    """Put a switch back to its default."""
    from . import settings as st

    try:
        v = st.reset(c.cfg, key)
    except KeyError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)
    console.print(f"[green]{key}[/] reset to {v}")


# ---------------------------------------------------------------------------- security
@main.group()
def security() -> None:
    """Defences: seal and verify Gauge's code, audit log, threat status, lockdown, code scan, token rotation."""


def _audit(c: Ctx):
    from .security import AuditLog

    return AuditLog(c.cfg.security_dir / "audit.jsonl")


@security.command("seal")
@click.pass_obj
def security_seal(c: Ctx) -> None:
    """Record signed hashes of Gauge's source, installers and models (run after you update or approve changes)."""
    from .code.selfmod import repo_root
    from .security import seal

    res = seal(repo_root(), c.cfg.security_dir, [c.cfg.native_dir])
    _audit(c).event("seal", f"{res['files']} files sealed", 1)
    console.print(f"[green]sealed[/] {res['files']} files; signature {res['signature']}...  (key: {c.cfg.security_dir / 'integrity.key'})")


@security.command("verify")
@click.pass_obj
def security_verify(c: Ctx) -> None:
    """Compare the current files with the seal."""
    from .code.selfmod import repo_root
    from .security import verify

    v = verify(repo_root(), c.cfg.security_dir, [c.cfg.native_dir])
    colour = "green" if v["status"].startswith("ok") else "red"
    console.print(f"[{colour}]{v['status']}[/]  signature_ok={v['signature_ok']}  sealed_at={v.get('sealed_at')}")
    for k in ("changed", "removed", "added"):
        if v[k]:
            console.print(f"  {k}: " + ", ".join(v[k][:20]) + (" ..." if len(v[k]) > 20 else ""))
    sys.exit(0 if v["status"].startswith("ok") or v["status"] == "unsealed" else 1)


@security.command("status")
@click.pass_obj
def security_status(c: Ctx) -> None:
    """Threat level, recent event counts, lock state, audit chain integrity."""
    from .code.selfmod import repo_root
    from .security import assess, verify

    audit = _audit(c)
    v = verify(repo_root(), c.cfg.security_dir, [c.cfg.native_dir])
    t = assess(audit, v["status"])
    names = {0: "calm", 1: "watch", 2: "elevated", 3: "LOCKDOWN"}
    console.print(f"threat: [bold]{names[t.level]}[/] ({t.level})  integrity: {v['status']}  locked: {c.cfg.security.locked}"
                  + (f" ({c.cfg.security.locked_reason})" if c.cfg.security.locked else ""))
    for r in t.reasons:
        console.print(f"  - {r}")
    console.print(f"events (24h): {t.counts or '{}'}")
    chain = audit.verify_chain()
    console.print(f"audit chain: {'intact' if chain['ok'] else 'BROKEN at line ' + str(chain.get('broken_at'))} ({chain['entries']} entries)")
    risky = [k for k in ("code.auto_apply", "code.self_heal", "security.allow_public_workers") if __import__("gauge.settings", fromlist=["get"]).get(c.cfg, k)]
    if risky:
        console.print(f"[yellow]risky switches on:[/] {', '.join(risky)}")
    if not c.cfg.research.respect_robots_txt:
        console.print("[yellow]robots.txt is not honoured for research[/]")


@security.command("log")
@click.option("--min", "min_sev", type=int, default=1, show_default=True)
@click.option("--limit", type=int, default=40, show_default=True)
@click.pass_obj
def security_log(c: Ctx, min_sev: int, limit: int) -> None:
    """The tamper-evident audit log."""
    rows = _audit(c).read(limit=limit, min_severity=min_sev)
    if not rows:
        console.print("(no events)")
        return
    for r in rows:
        console.print(f"{r['at']}  sev{r['severity']}  {r['kind']:<16} {r['detail'][:140]}", markup=False, highlight=False)


@security.command("lockdown")
@click.argument("reason", default="manual")
@click.pass_obj
def security_lockdown(c: Ctx, reason: str) -> None:
    """Switch off self-heal, auto-apply, bots, worker pool, research and daemon writing until unlocked."""
    from .security import lockdown

    console.print_json(json.dumps(lockdown(c.cfg, _audit(c), reason)))


@security.command("unlock")
@click.option("--no-restore", is_flag=True, help="Leave the switched-off automation off.")
@click.pass_obj
def security_unlock(c: Ctx, no_restore: bool) -> None:
    """End a lockdown. Previously-on switches come back except auto-apply, which you must turn on again yourself."""
    from .security import unlock

    console.print_json(json.dumps(unlock(c.cfg, _audit(c), restore=not no_restore)))


@security.command("scan")
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def security_scan(c: Ctx, path: str) -> None:
    """Static-scan a script or folder (Python, PowerShell, batch, JavaScript) for dangerous constructs."""
    from .security import report, scan_files, worst

    p = Path(path)
    files = {str(f.relative_to(p)): f.read_text(encoding="utf-8", errors="replace") for f in (p.rglob("*") if p.is_dir() else [p]) if f.is_file()} \
        if p.is_dir() else {p.name: p.read_text(encoding="utf-8", errors="replace")}
    findings = scan_files(files)
    console.print(report(findings))
    sys.exit(1 if worst(findings) >= 3 else 0)


@security.command("check-text")
@click.argument("path", type=click.Path(exists=True))
@click.option("--semantic", is_flag=True, help="Also ask the model whether the text is an injection (slower).")
@click.pass_obj
def security_check_text(c: Ctx, path: str, semantic: bool) -> None:
    """Show instruction-shaped text the sanitiser would flag in a file."""
    from .security import find_injections

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    hits = find_injections(text)
    if hits:
        for h in hits:
            console.print(f"sev{h['severity']} {h['label']}: {h['match']}")
    else:
        console.print("regex layer: clean")
    if semantic or c.cfg.security.semantic_injection:
        from .llm import make_llm
        from .security.semantic import SemanticInjectionDetector
        try:
            v = SemanticInjectionDetector(make_llm(c.cfg.llm, "fast")).assess(text)
            console.print(f"semantic layer: injection={v.injection} confidence={v.confidence} {v.category} — {v.why}")
        except Exception as e:
            console.print(f"[yellow]semantic layer unavailable: {e}[/]")


@security.command("deps")
@click.argument("path", type=click.Path(exists=True))
@click.pass_obj
def security_deps(c: Ctx, path: str) -> None:
    """Supply-chain scan: what a Python file imports, flagging unvetted or typosquat packages."""
    from .security.deps import report, scan_imports, worst

    findings = scan_imports(Path(path).read_text(encoding="utf-8", errors="replace"))
    console.print(report(findings))
    sys.exit(1 if worst(findings) >= 3 else 0)


@security.command("range")
@click.pass_obj
def security_range(c: Ctx) -> None:
    """Defensive cyber-range: fire synthetic attacks at Gauge's own defences and score what is caught."""
    from .security.range import run_range

    rep = run_range(max_body=c.cfg.security.max_request_bytes, audit=_audit(c))
    console.print(f"[bold]Cyber-range[/]: {rep.detected}/{rep.total} probes caught (score {rep.score:.0%})")
    for defence, (hit, tot) in sorted(rep.by_defence().items()):
        mark = "[green]ok[/]" if hit == tot else "[yellow]gap[/]"
        console.print(f"  {mark} {defence}: {hit}/{tot}")
    for lesson in rep.to_lessons():
        console.print(f"  [yellow]- {lesson}[/]")


@security.command("drill")
@click.option("--rounds", type=int, default=None, help="How many adversarial rounds (default: config).")
@click.option("--seed", type=int, default=None, help="Base seed for reproducibility (default: random each run).")
@click.option("--show", type=int, default=6, show_default=True, help="How many evasions to list.")
@click.pass_obj
def security_drill(c: Ctx, rounds: Optional[int], seed: Optional[int], show: int) -> None:
    """Adversarial self-play: mutate the attacks and vary the defence parameters every round, then
    report which mutated attacks slipped through so they can be hardened. All inward; nothing is run or sent."""
    from .security.adversary import run_drill

    n = rounds if rounds is not None else c.cfg.security.drill_rounds
    rep = run_drill(rounds=n, base_seed=seed, audit=_audit(c))
    console.print(f"[bold]Adversarial drill[/]: {rep.detected}/{rep.fired} mutated attacks caught "
                  f"(detection {rep.score:.1%}) over {len(rep.rounds)} rounds; base seed {rep.base_seed}")
    for r in rep.rounds:
        console.print(f"  round {r.index}: {r.detected}/{r.fired} caught; cats={','.join(r.plan.categories)}; "
                      f"depth={r.plan.depth}; cap={r.defense.body_cap // 1024}k, lockout={r.defense.lockout_failures}, action={r.defense.injection_action}")
    if rep.evasions:
        console.print(f"[yellow]{len(rep.evasions)} evasion(s) to harden:[/]")
        for e in rep.evasions[:show]:
            console.print(f"  [yellow]- {e.category} via {e.mutation}: {e.payload[:80]!r}[/]")
    else:
        console.print("[green]no evasions: the defences held against every mutated attack this drill[/]")


@security.command("defense")
@click.option("--plant", default=None, help="Plant a honeytoken with this label (e.g. api-key).")
@click.option("--evidence", "show_evidence", is_flag=True, help="List captured evidence.")
@click.pass_obj
def security_defense(c: Ctx, plant: Optional[str], show_evidence: bool) -> None:
    """Show (or arm) the graduated active defence. Everything here stays on your own machine."""
    from .security.defense import ActiveDefense, level_for_threat, posture
    from .security.sentinel import assess

    sec = c.cfg.security
    level = sec.active_defense_level
    if sec.active_defense_auto:
        threat = assess(_audit(c))
        level = max(level, level_for_threat(threat.level))
    ad = ActiveDefense(c.cfg.security_dir, level=level, audit=_audit(c))
    p = posture(level)
    console.print(f"[bold]Active defence[/]: level {level} — {p.note}")
    console.print(f"  tarpit {p.tarpit_seconds}s, honeytokens {'on' if p.honeytokens else 'off'}, "
                  f"decoys {'on' if p.decoys else 'off'}, evidence {'on' if p.evidence else 'off'}")
    console.print(f"  {len(ad.honeytokens())} honeytoken(s) planted; {len(ad.evidence())} evidence bundle(s)")
    if plant:
        tok = ad.plant_honeytoken(plant)
        console.print(f"[green]planted[/] honeytoken for {plant}: {tok[:20]}... (if this ever appears in traffic, it was exfiltrated)")
    if show_evidence:
        for e in ad.evidence(20):
            console.print(f"  {e['at']} {e['kind']} from {e.get('peer', '?')}: {e['payload'][:80]}")


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
