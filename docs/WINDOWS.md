# Running Gauge on Windows

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.10 or newer (`winget install Python.Python.3.12`, or python.org; tick "Add to PATH")
- ~8 GB disk for a 8B model; ~20 GB if you want a 14B model
- Optional GPU: NVIDIA (8 GB+ VRAM recommended) or AMD via Ollama's ROCm build. CPU-only
  works for everything; writing a 2,000-word chapter takes a few minutes instead of seconds.

## Install

The easy way is one file: download `installer\Get Gauge.bat` and double-click it (see
[installer/README.md](../installer/README.md)). It needs nothing installed first and picks a model that
fits the machine. The steps below are the command-line route, for when you already use git.

```powershell
git clone https://github.com/charmand3rrawr-ui/gauge.git
cd gauge
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Options: `-Model "qwen2.5:14b"` (bigger, better prose), `-WithML` (installs scikit-learn for
the stronger engagement model), `-SkipOllama` (you run LM Studio or another server instead).

If PowerShell refuses to run scripts, the `-ExecutionPolicy Bypass` flag above is enough; you
do not need to change the system policy.

## Choosing a model

| VRAM / RAM | Suggested `--model` | Notes |
| --- | --- | --- |
| 6-8 GB VRAM | `llama3.1:8b`, `qwen2.5:7b` | default; fast, decent prose |
| 12-16 GB VRAM | `qwen2.5:14b`, `gemma2:9b` | noticeably better dialogue and structure |
| 24 GB VRAM | `qwen2.5:32b`, `llama3.1:70b-instruct-q2_K` | slow but strong |
| CPU only, 16 GB RAM | `llama3.2:3b`, `qwen2.5:7b` | patience required |

Change at any time: `.\gauge.bat init --model qwen2.5:14b` then `ollama pull qwen2.5:14b`.

## Making it fast

1. `.\gauge.bat bench` measures your models and estimates minutes per chapter.
2. Add a small model for JSON work: `ollama pull llama3.2:3b`, then set `"fast_model": "llama3.2:3b"`
   under `llm` in `data\config.json`. Briefs, reviews, memory, critic and summaries move there;
   the big model only writes prose. Typically 2-3x less time per chapter.
3. Concurrency: the installer sets `OLLAMA_NUM_PARALLEL=4` for your user (restart the Ollama app
   once). `llm.parallel` in the config should match it.
4. Context: `llm.num_ctx` defaults to 8192. On an 8 GB card with a 8B model, set the user
   environment variables `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` so the KV
   cache fits; otherwise Ollama will offload layers to CPU and slow down sharply.
5. `llm.keep_alive` is `30m` so the daemon does not reload weights every cycle; `-1` keeps them
   loaded forever.
6. If a chapter dies halfway, just run `gauge write` again: the draft was checkpointed and is
   resumed instead of regenerated.

## LM Studio instead of Ollama

Start LM Studio's local server (default `http://localhost:1234`), load a model, then:

```powershell
.\gauge.bat init --provider openai --base-url http://localhost:1234/v1 --model <model id shown in LM Studio>
```

Embeddings fall back to an offline hashing embedder with non-Ollama providers; retrieval still
works, just less semantically.

## Keep it learning in the background

Option A, a console window:

```powershell
.\scripts\run-daemon.bat --project "Ten Seconds" --write --every 180
```

Option B, Windows Task Scheduler (survives reboots, no window):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-task.ps1 -Hours 6 -Project "Ten Seconds" -Write
# remove later:
powershell -ExecutionPolicy Bypass -File .\scripts\install-task.ps1 -Remove
```

Each run does one full cycle (`gauge daemon --once`). Ollama must be running for the write
step; the ingest and learning steps need no LLM at all.

## Where things live

```
data\config.json            settings (edit by hand, or `gauge config` to view)
data\gauge.sqlite3          novels, chapters, features, metric history, runs
data\models\                engagement_*.json, style_guide.md, Modelfile, exemplars.*, datasets\
data\projects\<slug>\       project.json, bible.md, outline.md, chapters\chapter_001.md ...
```

Set `GAUGE_DATA_DIR` to put `data` somewhere else (for example on a bigger drive).

## Troubleshooting

- **`LLM NOT reachable`** in `gauge doctor`: open the Ollama app (tray icon) or run `ollama serve`.
  Check `http://localhost:11434` in a browser; it should say "Ollama is running".
- **`model ... not pulled`**: `ollama pull <name>`.
- **Writing is very slow**: pick a smaller model, or lower `write.target_chapter_words` and
  `llm.max_tokens` in `data\config.json`.
- **`robots.txt disallows`** warnings: the site asked not to be crawled there; Gauge skips it.
  Import exports with `gauge import` instead.
- **Site parsing returns nothing**: site HTML changed. Open an issue with the URL, or fix the
  selectors in `gauge\ingest\<site>.py` (fixtures in `tests\fixtures` show the expected shape).
- **Antivirus flags the venv**: add the project folder to exclusions; this is common with
  freshly created Python environments.
