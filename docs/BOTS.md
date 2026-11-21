# Bots and the Engineer

Most chores around a serial do not need a language model: counting words, exporting chapters to
HTML or EPUB, backing up the data folder, watching a folder for new drafts, posting formats,
tracking numbers in a CSV. Gauge now writes those as **bots**: plain programs that run on their
own, with a model involved only while they are being written.

## How a bot is made

```powershell
.\gauge.bat bot new wordcount --task "count words in every chapter file and write a CSV, daily"
```

1. **Classify.** The Engineer decides whether the task needs a model at run time (judgement about
   meaning or open-ended text) or is deterministic (files, counts, formats, schedules, data,
   HTTP), and which language fits: Python by default, PowerShell for Windows shell work, batch
   for launchers, JavaScript for browser userscripts. Override with `--no-ai` / `--needs-ai` and
   `--language`.
2. **Write.** With the relevant knowledge cards and the toolkit API in the prompt, it writes
   `main.py` (argparse, clear errors, non-zero exit on failure, idempotent), `test_main.py`
   (pytest, fixture data, offline), a `README.md`, and a manifest.
3. **Test.** The bot's tests run in a subprocess with a timeout. Failures are fed back and the
   bot is rewritten, up to `code.max_fix_rounds` times. A bot whose tests do not pass is kept for
   inspection but never scheduled.
4. **Register.** Bots live under `data/bots/<name>/`. `gauge bot run NAME -- --args`,
   `gauge bot schedule NAME hourly|daily|every:N|on-cycle|off`, `gauge bot list`, `gauge bot test`,
   `gauge bot remove`. The daemon runs due bots every cycle.

Deterministic bots run with `GAUGE_NO_LLM=1`, and `make_llm` refuses to hand them a model. That
is enforced in code, not just promised in a prompt.

## What the Engineer knows

`gauge code cards` lists the built-in reference cards: Python core and traps, PowerShell, batch,
Bash, JavaScript, SQL/SQLite, regular expressions, HTML/CSS, HTTP and data formats, testing and
quality, Windows automation, and Gauge's own internals. `gauge code search "sqlite upsert"` shows
the cards a question would pull. `gauge code learn notes.md` adds your own notes (split on `#`
headings) and they persist under `data/models/knowledge/`.

## The toolkit

`gauge.code.toolkit` is the deterministic standard library bots build on: encoding-safe file
reading, atomic writes, natural-order listing, hashing, backups with pruning, folder watching by
polling, word counts and text statistics, Markdown to HTML and a clean HTML page, JSON/JSONL/CSV
helpers, polite cached HTTP, and subprocess running. All of it is tested and has no model
anywhere.

## Examples of bots that need no model

- export a project to one HTML file or an EPUB
- back up `data/` nightly, keeping the last ten copies
- watch a drafts folder and import new chapters
- a CSV of words, dialogue share and reading time per chapter
- convert chapter Markdown to the HTML a serial site accepts for pasting
- a launcher that runs the daemon when the PC starts
