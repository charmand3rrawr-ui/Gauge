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
   robots.txt honoured) and imports whole libraries from your disk (`.txt` `.md` `.html`
   `.docx` `.epub`, with metadata and labels from sidecars, EPUB/DOCX metadata or file names).
   Re-running only touches chapters it has not seen or that changed, so the corpus tracks
   novels, and your own drafts, as they update.
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
4. **Write** - a project has a series bible, a rolling outline, a running memory, a cast of
   **characters with independent personality matrices**, and a **World agent** that owns the
   physics, weather, crowds and consequences and acts in every scene through nameless passerby
   ([docs/WORLD.md](docs/WORLD.md)). A **muse** proposes divergent directions before each outline,
   **technique cards** absorbed from the corpus feed the drafter, and a **novelty tracker** punishes
   repetition ([docs/CREATIVITY.md](docs/CREATIVITY.md)). A **writing room** of eleven more
   independent agents reviews every chapter: a calibrated Reader Panel, a Showrunner with a
   promise ledger, an Archivist with a canon ledger, a Schemer running the opposition, a Line
   Editor, a Packager, a Theme Keeper, a Sensitivity Reader, a Pacing Director, a Research
   Librarian and a Comedian ([docs/AGENTS.md](docs/AGENTS.md)). A **Scout** searches open
   sources (Wikipedia, Wikivoyage, Project Gutenberg, arXiv, real climate data, your own SearXNG)
   for the real-world detail each scene needs, files it into the project, and fact-checks drafts
   ([docs/RESEARCH.md](docs/RESEARCH.md)). It **learns from your own serial**: real reader numbers
   calibrate the panel, tune prompt settings with a bandit, and fit a per-project model; your hand
   edits and its own improved revisions become fine-tuning data; a prediction ledger keeps the
   corpus model honest; recurring complaints become lessons in every prompt. Characters carry an
   emotional model, planned personality arcs, beliefs about each other, a scene improv before every
   draft, and a Voice Coach, with an author persona that can outweigh the data
   ([docs/LEARNING.md](docs/LEARNING.md), [docs/CHARACTERS.md](docs/CHARACTERS.md)). When a character is in a scene their
   agent is activated: they brief the drafter on what they want, would do, would say and would
   refuse, review the draft for out-of-character moments, and remember the chapter afterwards.
   Each chapter is planned, drafted with retrieved exemplars and the cast's briefs in context,
   scored by the engagement model, an LLM editor rubric and in-character consistency, revised
   until it clears the bar, then saved as Markdown. See [docs/CHARACTERS.md](docs/CHARACTERS.md).
5. **Repeat** - `gauge daemon` runs the whole cycle on a schedule, or Windows Task Scheduler
   runs `gauge daemon --once` for you.

## Install for everyone (double-click)

Download the ZIP, unzip it, double-click **Install Gauge.bat**, wait. It installs Python and
Ollama if needed, downloads the models, puts a **Gauge Studio** shortcut on the Desktop and opens
it: one big box for your idea, **START MY STORY**, **WRITE THE NEXT CHAPTER**, click a chapter to
read, **Talk to a character**. No accounts, no keys, nothing leaves the computer.
Details and troubleshooting: [installer/README.md](installer/README.md).

## Install on Windows (command line, 5 minutes)

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

Other platforms: `pip install -e ".[ml,dev]"` then `gauge init`.

## Quick start

```powershell
# 1. Follow the top-rated and rising novels on Royal Road, then fetch them (incremental, polite)
.\gauge.bat discover royalroad best-rated --pages 3
.\gauge.bat discover royalroad rising-stars --pages 2
.\gauge.bat ingest

# 2. Learn from your own disk too: a novel folder, a book file, or a whole library of them
.\gauge.bat import "C:\Writing\MySerial"
.\gauge.bat import "D:\Books" --watch          # library of folders/epubs/docx, kept in sync
.\gauge.bat rate local:my-serial --stars 5     # label novels with your own taste

# 3. Learn
.\gauge.bat train
.\gauge.bat report                      # feature importances, leaderboard, recent runs
type data\models\style_guide.md         # what the data says works, in English

# 4. Write: from an idea, a premise document, or an existing manuscript
.\gauge.bat new "Ten Seconds" --idea "a broke courier can rewind ten seconds, at the cost of a memory each time" --genre Sci-fi --genre Thriller
.\gauge.bat new "Ash" --premise-file "D:\Writing\ash-premise.md"       # your premise is canon
.\gauge.bat new "Book 2" --continue-from "D:\Writing\Book1"             # continue a manuscript
.\gauge.bat start "a courier who can rewind ten seconds" --chapters 3    # one command, premise to chapters
.\gauge.bat absorb                                                       # learn craft moves from the corpus
.\gauge.bat write "Ten Seconds" --chapters 3
type data\projects\ten-seconds\chapters\chapter_001.md

# 4b. Meet the cast; add your own; interview them
.\gauge.bat character list "Ten Seconds"
.\gauge.bat character new "Ten Seconds" "Marina Vale" --concept "a fence who sells rewound goods" --role antagonist
.\gauge.bat character chat "Ten Seconds" Rin --ask "What are you not telling your sister?"

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
| `gauge import PATH [--watch] [--stars N]` | import a novel folder, a book (`.txt .md .html .docx .epub`), or a whole library; see [docs/LOCAL_LIBRARY.md](docs/LOCAL_LIBRARY.md) |
| `gauge watch PATH` | re-scan a local folder every cycle (new / edited / removed chapters) |
| `gauge novels [--source S]` / `gauge rate KEY --stars N` | list novels; give one your own 1-5 label (`--target user_rating`) |
| `gauge analyze [--all]` | compute features |
| `gauge train [--target T]` | full learning cycle |
| `gauge report [--novel KEY]` | what was learned; explain one novel |
| `gauge score FILE` | score a chapter file; percentiles vs corpus |
| `gauge new NAME [--idea] [--premise-file F] [--continue-from PATH]` | create a project from an idea, a canon premise document, and/or an existing manuscript |
| `gauge start PREMISE [--chapters N]` | premise (text or file) straight to written chapters |
| `gauge absorb [--chapters N]` | extract craft technique cards from the strongest unread chapters |
| `gauge world show/build/set/chat/extras/promote` | the World agent; see [docs/WORLD.md](docs/WORLD.md) |
| `gauge agents list/show/enable/disable/setup` | the writing room; see [docs/AGENTS.md](docs/AGENTS.md) |
| `gauge canon`, `gauge ledger`, `gauge readers`, `gauge feedback`, `gauge package`, `gauge research` | ask the Archivist, read the Showrunner's ledger, see Reader Panel predictions, record real reader numbers, get blurb and tags, search your notes |
| `gauge scout ask/dossier/gutenberg/log` | open-source research; import public-domain books; see [docs/RESEARCH.md](docs/RESEARCH.md) |
| `gauge feedback`, `gauge bandit`, `gauge lessons`, `gauge sync-edits`, `gauge finetune` | learn from your own serial: real numbers, prompt tuning, lessons, edit pairs, LoRA plan |
| `gauge persona build/show/blend` | the author persona learned from your chapters |
| `gauge character arc` | planned personality drift |
| `gauge native presets/bench/train/distill/eval/continue/status/sample/tokenizer` | GaugeLM, the built-in model; see [docs/NATIVE.md](docs/NATIVE.md) |
| `gauge gui` | open Gauge Studio (the big-button window) |
| `gauge write NAME [--chapters N] [--fresh]` | draft, score, revise, save; resumes checkpointed drafts |
| `gauge bench` | measure model speed; estimate minutes per chapter |
| `gauge projects [NAME]` | list projects / status |
| `gauge character new/roll/sync/list/show/set/brief/chat/remove` | characters as independent agents with personality matrices; see [docs/CHARACTERS.md](docs/CHARACTERS.md) |
| `gauge daemon [--once] [--write] [--project P] [--every MIN]` | continuous loop |

## Execution: speed and reliability

- **Context window is explicit.** Every call sends `llm.num_ctx` (default 8192) and every draft
  prompt is assembled against it: plan and previous ending first, then bible, summary, cast
  briefs and exemplars while they fit. Nothing is left for the server to truncate.
- **Two models.** Set `llm.fast_model` to a small model (e.g. `llama3.2:3b`) and all JSON work
  (scene briefs, in-character reviews, memory, critic, summaries) runs there, while the main
  model only writes and revises prose.
- **Parallel characters.** Briefs, reviews and memory updates run `llm.parallel` at a time; the
  installer sets `OLLAMA_NUM_PARALLEL=4`. Characters who are only name-dropped skip review, and
  revisions re-review only the characters who had complaints.
- **Retries and schema output.** Transient failures retry with backoff; JSON answers are
  schema-constrained when the server supports it, fall back automatically when it does not, and
  get one repair retry when they still fail to parse.
- **Checkpoints.** Raw drafts are saved to `projects/<slug>/drafts/` before they are judged, so a
  crash never loses a generation; the next `gauge write` resumes from them (`--fresh` to ignore).
- **Telemetry.** Every call is logged with purpose, tokens and duration. `gauge write` prints
  per-step timings, `gauge report` shows LLM time by purpose, and `gauge bench` measures your
  models and estimates minutes per chapter before you commit to a model size.

## Your own model, no outside code

Gauge ships **GaugeLM**, a complete language model in pure Python and numpy: BPE tokenizer,
Llama-style transformer (RMSNorm, rotary positions, SwiGLU, grouped-query attention),
hand-written backprop, AdamW trainer with curriculum, distillation from the big local model,
and a sampler. Runs on CPU or, with CuPy, on an NVIDIA GPU; sizes from 4M to 310M parameters.
It trains on everything Gauge has read plus your own chapters, keeps learning every daemon
cycle, and plugs into the whole pipeline as `--provider native`. Small, and honest about it;
see [docs/NATIVE.md](docs/NATIVE.md).

```powershell
.\gauge.bat native bench --size small
.\gauge.bat native train --size small --minutes 240
.\gauge.bat native distill --rows 400        # learn behaviour from the big model
.\gauge.bat native train --resume --steps 800
.\gauge.bat init --provider native
```

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
  ingest/          royalroad.py, scribblehub.py, local.py (folders, libraries, epub, docx), http.py, runner.py
  learn/           engagement.py (model), exemplars.py, techniques.py (absorber), style.py, dataset.py, trainer.py
  research/        sources.py (Wikipedia, Wikivoyage, Gutenberg, arXiv, climate, SearXNG), scout.py (cache, filing, dossiers)
  native/          GaugeLM: tokenizer.py, model.py (numpy transformer + backprop), train.py, client.py
  write/           project, characters (matrix, agents), world (physics, crowds), muse (directions, novelty),
                   agents/ (the writing room: 11 agents + council), planner, drafter, critic, pipeline, prompts, schemas
  llm/             ollama / openai-compatible / fake clients, embeddings
  daemon.py  report.py  cli.py  db.py (sqlite)  config.py
docs/              ARCHITECTURE, LEARNING, LOCAL_LIBRARY, CHARACTERS, WORLD, CREATIVITY, AGENTS, RESEARCH, NATIVE, WINDOWS, ETHICS, ROADMAP
installer/         double-click installer (bootstrap.ps1) and Gauge Studio GUI launcher
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
