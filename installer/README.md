# Installing Gauge the easy way

1. Download this folder (green **Code** button on GitHub, then **Download ZIP**) and unzip it anywhere.
2. Double-click **Install Gauge.bat**.
3. Wait. The first time takes 10-20 minutes because it downloads the AI models (a few GB).
4. A window called **Gauge Studio** opens, and a **Gauge Studio** shortcut appears on your Desktop.

That is all. No accounts, no API keys, nothing leaves your computer.

## Using Gauge Studio

Type what your story is about in the big box. Press **START MY STORY**. Then press
**WRITE THE NEXT CHAPTER** whenever you want more. Click a chapter on the left to read it.
**Talk to a character** lets you ask the people in your story questions, and they answer as
themselves. **Save story as one file** makes one document you can print or share. **Switches** opens every setting as checkboxes and sliders, with a warning on the ones that matter.

## What the installer does

- Installs Python (per-user, no admin) if you do not have it.
- Installs Ollama, the program that runs the AI on your own computer.
- Installs Gauge and downloads three models: a writer (`llama3.1:8b`), a small helper
  (`llama3.2:3b`) for the bookkeeping jobs, and an embedding model for memory.
- Writes Gauge's settings and puts **Gauge Studio** on your Desktop.

Everything it did is logged in `installer\install.log`.

## If something goes wrong

- **Windows asks "Do you want to allow this app to make changes?"** That is the Ollama or Python
  installer. Say yes.
- **"Story engine is OFF"** in Gauge Studio: open the Ollama app (the llama icon in the Start menu),
  then open Gauge Studio again.
- **Blue "Windows protected your PC" screen** when double-clicking the .bat: click *More info*, then
  *Run anyway*. The scripts are plain text; you can read them.
- **Slow computer?** Use a smaller writer model:
  `powershell -ExecutionPolicy Bypass -File installer\bootstrap.ps1 -Model llama3.2:3b`

## For grown-ups

The installer accepts `-Model`, `-FastModel`, `-NoLaunch`, `-NoShortcut`. The command line
(`gauge.bat`) and everything in `docs/` still work; `gauge gui` opens the Studio from a terminal.
