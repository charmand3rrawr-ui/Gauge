# Gauge

**An open-source, local-first engine that learns what makes web novels addictive, and writes them.**

Gauge runs entirely on a Windows desktop (Linux and macOS work too). It continuously ingests
serial fiction and its engagement numbers, measures what the chapters that readers binge have
in common, and feeds those lessons into a local LLM that plans, drafts, critiques and revises
your own serial, chapter after chapter.

No cloud. No API bills. Your drafts never leave your machine.

> Honest framing: Gauge is not a "superintelligence" and nothing you can download today is.
> It is a disciplined learning loop: data in, measurable patterns out, prose scored against those
> patterns. The loop gets smarter the longer you let it run, and every part of it is readable code.

---

## What it does

```
          ┌────────────────────────────────────────────────────────────────┐
          │                      gauge daemon (every N hours)              │
          │                                                                │
  Royal   │  ingest ──► analyze ──► train ──► distil ──► write ──► score   │
  Road,   │  (new     (hook,      (engage-  (style     (plan,    (model +  │
  Scribble│  chapters  pacing,     ment      guide,     draft)    editor   │
  Hub,    │  only)     payoff,     model,    Modelfile)           rubric)  │
  local   │            cliff)      exemplars)                              │
  files   └────────────────────────────────────────────────────────────────┘
```

1. **Ingest** - follows novels on Royal Road and Scribble Hub (public pages, rate-limited,
   robots.txt honoured) and imports your own files or EPUBs. Re-running only fetches chapters
   it has not seen, so the corpus tracks novels as they update.
2. **Analyze** - turns every chapter into ~60 transparent features: hook strength in the first
   200 words, cliffhanger strength in the last 200, dialogue ratio, sentence-length rhythm,
   tension curve across the chapter, reveal / progression / warmth density, chapter length,
   release cadence, blurb shape, and more. See [docs/LEARNING.md](docs/LEARNING.md).
3. **Train** - fits an engagement model (ridge regression, plus gradient boosting if
   scikit-learn is installed) that predicts followers / views / rating from those features, with
   cross-validation so it cannot lie to you. Builds an **exemplar library** of the openings and
   endings from the most engaging chapters. Distils everything into a plain-English
   **style guide** and an Ollama **Modelfile**. Exports preference pairs and SFT rows for
   optional LoRA fine-tuning.
4. **Write** - a project has a series bible, a rolling outline, and a running memory. Each
   chapter is planned, drafted with retrieved exemplars in context, scored by the engagement
   model *and* an LLM editor rubric, revised until it clears the bar, then saved as Markdown.
5. **Repeat** - `gauge daemon` runs the whole cycle on a schedule, or Windows Task Scheduler
   runs `gauge daemon --once` for you.

## Install on Windows (5 minutes)

Requirements: Windows 10/11, Python 3.10+, ~8 GB free disk for models. A GPU with 8 GB VRAM
makes writing fast; CPU-only works for everything, just slower.

```powershell
git clone https://github.com/charmand3rrawr-ui/gauge.git
cd gauge
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The installer creates a venv, installs Gauge, installs [Ollama](https://ollama.com) via winget
if missing, pulls `llama3.1:8b` and `nomic-embed-text`, and runs `gauge doctor`.
Full details and troubleshooting: [docs/WINDOWS.md](docs/WINDOWS.md).

Other platforms: `pip install -e ".[ml,epub,dev]"` then `gauge init`.

## Quick start

```powershell
# 1. Follow the top-rated and rising novels on Royal Road, then fetch them (incremental, polite)
.\gauge.bat discover royalroad best-rated --pages 3
.\gauge.bat discover royalroad rising-stars --pages 2
.\gauge.bat ingest

# 2. Add your own novel or anything you exported (folder of .txt/.md, one file, or .epub)
.\gauge.bat import "C:\Writing\MySerial"

# 3. Learn
.\gauge.bat train
.\gauge.bat report                      # feature importances, leaderboard, recent runs
type data\models\style_guide.md         # what the data says works, in English

# 4. Write
.\gauge.bat new "Ten Seconds" --idea "a broke courier can rewind ten seconds, at the cost of a memory each time" --genre Sci-fi --genre Thriller
.\gauge.bat write "Ten Seconds" --chapters 3
type data\projects\ten-seconds\chapters\chapter_001.md

# 5. Score any chapter you wrote by hand against the corpus
.\gauge.bat score "C:\Writing\draft.txt"

# 6. Let it keep learning (and optionally keep writing) on its own
.\gauge.bat daemon --project "Ten Seconds" --write --every 180
```

## Commands

| Command | What it does |
| --- | --- |
| `gauge init` / `gauge doctor` / `gauge config` | set up, health-check, show config |
| `gauge discover SOURCE LISTING` | follow novels from a site listing (`best-rated`, `trending`, `rising-stars`, ...) |
| `gauge follow URL` | follow one novel |
| `gauge ingest [--url U]` | fetch followed novels; only new chapters |
| `gauge import PATH` | import local folder / file / epub (+ optional `novel.json` metrics) |
| `gauge analyze [--all]` | compute features |
| `gauge train [--target T]` | full learning cycle |
| `gauge report [--novel KEY]` | what was learned; explain one novel |
| `gauge score FILE` | score a chapter file; percentiles vs corpus |
| `gauge new NAME --idea ... --genre ...` | create a project: bible + outline |
| `gauge write NAME [--chapters N]` | draft, score, revise, save |
| `gauge projects [NAME]` | list projects / status |
| `gauge daemon [--once] [--write] [--project P] [--every MIN]` | continuous loop |

## Using a different local model

Any Ollama model works: `gauge init --model qwen2.5:14b`. Any OpenAI-compatible server (LM
Studio, llama.cpp `llama-server`, vLLM) works too:
`gauge init --provider openai --base-url http://localhost:1234/v1 --model your-model`.

After `gauge train`, `data/models/Modelfile` bakes the learned style guide into a model:

```powershell
ollama create gauge-writer -f data\models\Modelfile
.\gauge.bat init --model gauge-writer
```

For real fine-tuning on the exported datasets, see [docs/LEARNING.md](docs/LEARNING.md#fine-tuning).

## Layout

```
gauge/
  features.py      feature extraction (hook, momentum, payoff, cliffhanger, voice, packaging)
  lexicon.py       the small transparent word lists the features count
  ingest/          royalroad.py, scribblehub.py, local.py, polite http.py, incremental runner.py
  learn/           engagement.py (model), exemplars.py, style.py, dataset.py, trainer.py
  write/           project.py, planner.py, drafter.py, critic.py, pipeline.py, prompts.py
  llm/             ollama / openai-compatible / fake clients, embeddings
  daemon.py  report.py  cli.py  db.py (sqlite)  config.py
docs/              ARCHITECTURE, LEARNING, WINDOWS, ETHICS, ROADMAP
tests/             pytest suite, runs offline with a fake LLM
```

## Ethics in one paragraph

Gauge reads public pages slowly, with a clear user agent, and honours robots.txt. It learns
*patterns* (numbers about hooks, pacing and cadence) and keeps only short excerpts as style
exemplars. It is built to help you write your own better serial, not to reproduce anyone
else's. Please read [docs/ETHICS.md](docs/ETHICS.md) before pointing it at a site, and never
publish generated text as your own without reading and owning every line of it.

## Contributing

Issues and PRs welcome. Adding a site connector is four small functions; see
[CONTRIBUTING.md](CONTRIBUTING.md). Run `pytest` and `ruff check gauge tests` before you push.

MIT licensed.
