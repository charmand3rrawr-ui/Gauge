# Installing Gauge the easy way

## Windows: one file

1. Download **Get Gauge.bat** (it is in this folder).
2. Double-click it.
3. Wait. The first time takes 10-20 minutes because it downloads the AI models (a few GB).
4. A window called **Gauge Studio** opens, and a **Gauge Studio** shortcut appears on your Desktop.

That is all. No git, no accounts, no API keys, nothing leaves your computer. Gauge is put in
`C:\Users\<you>\Gauge`. Already have a copy of Gauge unzipped somewhere? Double-click
**Install Gauge.bat** in it instead; it does the same thing without the download.

## macOS and Linux: one line

```
curl -fsSL https://raw.githubusercontent.com/charmand3rrawr-ui/Gauge/ccr-24dc418f-hsneyx/install.sh | bash
```

Gauge is put in `~/Gauge`, and a `gauge` command is added to `~/.local/bin`. From a copy of Gauge you
already have, run `./install.sh` instead. On Linux it may ask for your password to install Python or
Ollama through the system package manager; on macOS it uses Homebrew when something is missing.

## What the installer does

1. **Looks at your computer** and picks a writer model that fits:

   | your computer | writer model |
   | --- | --- |
   | NVIDIA graphics card with 12 GB or more | `qwen2.5:14b` |
   | NVIDIA graphics card with 6-11 GB | `llama3.1:8b` |
   | no such card, 16 GB of memory or more (32 GB on Apple Silicon gets `qwen2.5:14b`) | `llama3.1:8b` |
   | less than that | `llama3.2:3b` |

   It also checks free disk space, and steps down to the small model rather than run the disk out.
2. Installs **Python** (per-user, no admin) if you do not have it.
3. Installs **Ollama**, the program that runs the AI on your own computer.
4. Installs **Gauge** into its own private Python environment.
5. Downloads the models: the writer, a small helper (`llama3.2:3b`) for the bookkeeping jobs, and an
   embedding model for memory.
6. Writes Gauge's settings.
7. **Asks the AI a real test question**, so you know it works rather than merely that it installed, and
   prints a checklist. Anything that failed is listed with the one command that fixes it.

Running it again is safe: it updates Gauge and keeps your stories and your settings. Existing settings
are only changed if you name a model or engine yourself. Everything it did on Windows is logged in
`installer\install.log`.

## Using Gauge Studio

Type what your story is about in the big box. Press **START MY STORY**. Then press
**WRITE THE NEXT CHAPTER** whenever you want more. Click a chapter on the left to read it.
**Talk to a character** lets you ask the people in your story questions, and they answer as
themselves. **Save story as one file** makes one document you can print or share. **Switches** opens
every setting as checkboxes and sliders, with a warning on the ones that matter.

## Choosing for yourself

| you want | Windows | macOS / Linux |
| --- | --- | --- |
| a specific writer model | `set GAUGE_MODEL=qwen2.5:14b` then run **Get Gauge.bat** | `./install.sh --model qwen2.5:14b` |
| a different folder | `set GAUGE_DIR=D:\Gauge` then run it | `./install.sh --dir /path` |
| no Ollama: GaugeLM, the built-in model | `set GAUGE_ENGINE=native` then run it | `./install.sh --engine native` |
| Gauge only, no AI engine or models | (not offered) | `./install.sh --skip-engine` |

From an unzipped copy on Windows the same choices are flags:
`powershell -ExecutionPolicy Bypass -File installer\bootstrap.ps1 -Model llama3.2:3b` (also `-FastModel`,
`-Engine native`, `-NoLaunch`, `-NoShortcut`).

GaugeLM ships untrained. If you choose it, the installer finishes by telling you the two commands that
fetch a public-domain corpus and train it; see [docs/NATIVE.md](../docs/NATIVE.md) for what a model that
small can and cannot do.

## If something goes wrong

- **Blue "Windows protected your PC" screen** when double-clicking the .bat: click *More info*, then
  *Run anyway*. Windows shows this for any script downloaded from the internet. The scripts are plain
  text; you can open them in Notepad and read them first.
- **Windows asks "Do you want to allow this app to make changes?"** That is the Ollama or Python
  installer. Say yes.
- **The checklist shows `[FAIL] The AI answers`.** Open the Ollama app (the llama icon in the Start
  menu), then run the installer again. If the computer is short on memory, pick the small model:
  `set GAUGE_MODEL=llama3.2:3b` and run it again.
- **"Story engine is OFF"** in Gauge Studio: open the Ollama app, then open Gauge Studio again.
- **A model did not download.** The checklist names it. Run `ollama pull <that model>` in a terminal,
  or just run the installer again; it only fetches what is missing.
- **Linux: `[ -- ] Gauge Studio window (optional)`.** The window needs your system's `python3-tk`
  package. Everything on the command line works without it.

## For the curious

`gauge.bat` (Windows) and the `gauge` command (macOS/Linux) run the command line; everything in
`docs/` applies. `gauge gui` opens the Studio from a terminal, and `gauge doctor` re-checks an install.
