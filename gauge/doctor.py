"""Gauge's doctor: a health check of everything Gauge depends on, each problem with what to do about it.

Studio's **Check for problems** button and `gauge doctor` both use it. Every check:

* is quick, and asks nothing of any service outside this computer except, when Claude is the writer,
  one free question to Anthropic (with a short time limit);
* is fenced off from the others, so one that breaks says "could not check" and the rest still run;
* never puts a key in its report, so the report can be copied and shown to someone helping.

A finding is `ok`, `warn` (works, but something is missing or will cause trouble), `problem` (something
does not work) or `info` (worth knowing). Where Studio has a window that fixes it, the finding names it
in `action`: engine (Change the AI), teach (Teach Gauge), switches, devices, claude, helpers.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

MARK = {"ok": "✔", "warn": "!", "problem": "✖", "info": "•"}
ORDER = {"problem": 0, "warn": 1, "info": 2, "ok": 3}
ACTIONS = {"engine": "Open Change the AI", "teach": "Open Teach Gauge", "switches": "Open Switches",
           "devices": "Set up the graphics card", "claude": "Open Claude", "helpers": "Open Free helpers"}


@dataclass
class Finding:
    name: str
    status: str            # ok | warn | problem | info
    says: str              # what was found, in a sentence
    fix: str = ""          # what to do about it, when there is something to do
    action: str = ""       # the Studio window that fixes it, when there is one

    def line(self) -> str:
        return f"{MARK.get(self.status, '?')} {self.name}: {self.says}" + (f"\n    What to do: {self.fix}" if self.fix else "")


# ---------------------------------------------------------------------------------- the checks
def _python(cfg: Any, db: Any) -> list[Finding]:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 10):
        return [Finding("Python", "problem", f"Python {ver} is too old; Gauge needs 3.10 or newer.",
                        "Run the Gauge installer again; it installs a suitable Python.")]
    return [Finding("Python", "ok", f"Python {ver}.")]


def _version(cfg: Any, db: Any) -> list[Finding]:
    from . import __version__

    marker = Path(__file__).resolve().parents[1] / "installer" / "installed-version.txt"
    try:
        parts = marker.read_text(encoding="utf-8-sig").split()
        installed = parts[-1][:7] if parts else ""
    except OSError:
        installed = ""
    says = f"Gauge {__version__}" + (f", installed copy {installed}." if installed else " (not installed by the kit).")
    return [Finding("Gauge", "info", says, "To get the newest version, double-click 'Update Gauge.bat'." if installed else "")]


def _data(cfg: Any, db: Any) -> list[Finding]:
    out: list[Finding] = []
    root = Path(cfg.root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".doctor-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        out.append(Finding("Data folder", "ok", f"{root.resolve()} can be written to."))
    except OSError as e:
        return [Finding("Data folder", "problem", f"Gauge cannot write to {root.resolve()} ({e.strerror or e}).",
                        "Make sure the folder is not read-only and the disk is not full, or move Gauge to a folder in "
                        "your user folder.")]
    free = shutil.disk_usage(root).free / 2**30
    if free < 0.3:
        out.append(Finding("Disk space", "problem", f"Only {free:.1f} GB free on the disk with Gauge's data.",
                           "Free some space: training checkpoints and the downloaded models need several GB."))
    elif free < 2:
        out.append(Finding("Disk space", "warn", f"{free:.1f} GB free on the disk with Gauge's data.",
                           "Free some space before training GaugeLM or downloading a bigger model."))
    else:
        out.append(Finding("Disk space", "ok", f"{free:.0f} GB free."))
    if cfg.config_path.exists():
        try:
            json.loads(cfg.config_path.read_text(encoding="utf-8-sig"))
            out.append(Finding("Settings", "ok", "The settings file reads correctly."))
        except (OSError, ValueError) as e:
            out.append(Finding("Settings", "problem", f"The settings file cannot be read ({str(e)[:100]}).",
                               f"Open {cfg.config_path} in Notepad and undo the last change, or rename it so Gauge "
                               "starts with fresh settings."))
    return out


def _database(cfg: Any, db: Any) -> list[Finding]:
    if db is None:
        return []
    from .report import corpus_summary

    row = db._conn.execute("PRAGMA quick_check(1)").fetchone()
    if row and str(row[0]).lower() != "ok":
        return [Finding("Library database", "problem", f"The database reports damage ({str(row[0])[:120]}).",
                        "Close Gauge and restore a backup (gauge backup restore), or ask for help with the copied report.")]
    s = corpus_summary(db)
    if not s["chapters"]:
        return [Finding("Library", "warn", "The library is empty, so Gauge has nothing to learn from.",
                        "Press Teach Gauge and add some books or your own writing.", "teach")]
    return [Finding("Library", "ok", f"{s['novels']} book{'' if s['novels'] == 1 else 's'}, {s['chapters']:,} chapters.")]


def _ollama_models(cfg: Any) -> Optional[list[str]]:
    """The models installed in Ollama, or None when Ollama is not answering."""
    from .llm.client import OllamaLLM

    base = cfg.llm if cfg.llm.provider == "ollama" else cfg.llm.model_copy(update={"provider": "ollama",
                                                                                     "base_url": "http://localhost:11434"})
    o = OllamaLLM(base, base.model or "x")
    return o.models() if o.available() else None


def _has(installed: list[str], name: str) -> bool:
    return bool(name) and any(m == name or m == f"{name}:latest" or m.startswith(name + ":") for m in installed)


def _writer(cfg: Any, db: Any) -> list[Finding]:
    from .llm import make_llm

    p = cfg.llm.provider.lower()
    if p == "fake":
        return [Finding("Writer", "info", "Gauge is set to its test model, which writes the same few lines every time.",
                        "Press Change the AI and choose the downloaded model or GaugeLM.", "engine")]
    if p == "native":
        from .native import NativeTrainer

        st = NativeTrainer(cfg.native_dir).status()
        if not st.get("trained"):
            return [Finding("Writer", "problem", "GaugeLM is the writer but has not been trained yet, so it cannot write.",
                            "Add material with Teach Gauge, then press Change the AI and Start training; or choose the "
                            "downloaded model to write meanwhile.", "engine")]
        return [Finding("Writer", "ok", f"GaugeLM writes; it has trained {int(st.get('step', 0) or 0):,} steps.")]
    if p == "claude":
        from .llm import claude

        if not claude.installed():
            return [Finding("Writer", "problem", "Claude is the writer but Anthropic's library is not installed.",
                            "Press Change the AI, then Claude..., and Connect.", "claude")]
        chk = claude.ClaudeLLM(cfg.llm, cfg.llm.model).check()
        st = claude.status(cfg)
        out = [Finding("Writer", "ok" if chk["ok"] else "problem", chk["message"],
                       "" if chk["ok"] else "Press Change the AI, then Claude..., and connect again.", "" if chk["ok"] else "claude")]
        if st["limit"] > 0 and st["left"] == 0:
            out.append(Finding("Claude's daily limit", "warn", "Today's spending limit has been reached, so Claude will not write "
                               "again until tomorrow.", "Raise the limit in Change the AI, Claude..., or wait until tomorrow.", "claude"))
        return out
    if p == "ollama":
        installed = _ollama_models(cfg)
        if installed is None:
            return [Finding("Writer", "problem", f"Ollama, which runs the downloaded model {cfg.llm.model}, is not answering.",
                            "Open the Ollama app (the llama icon) and wait a few seconds. If it is not installed, get it "
                            "from ollama.com/download.")]
        out = []
        if _has(installed, cfg.llm.model):
            out.append(Finding("Writer", "ok", f"The downloaded model {cfg.llm.model} is installed and Ollama is answering."))
        else:
            out.append(Finding("Writer", "problem", f"The model {cfg.llm.model} is not installed in Ollama.",
                               f"Open a Command Prompt and run: ollama pull {cfg.llm.model}   (or choose another model in "
                               "Change the AI).", "engine"))
        if cfg.llm.fast_model and not _has(installed, cfg.llm.fast_model):
            out.append(Finding("Fast model", "warn", f"The fast model {cfg.llm.fast_model} is not installed, so bookkeeping "
                               "falls back to the main model and is slower.", f"Run: ollama pull {cfg.llm.fast_model}"))
        if not _has(installed, cfg.llm.embed_model):
            out.append(Finding("Embeddings", "warn", f"The embedding model {cfg.llm.embed_model} is not installed, so finding "
                               "similar passages uses a rougher method.", f"Run: ollama pull {cfg.llm.embed_model}"))
        return out
    llm = make_llm(cfg.llm)
    ok = llm.available()
    return [Finding("Writer", "ok" if ok else "problem", f"{llm.describe()} is {'answering' if ok else 'not answering'}.",
                    "" if ok else "Start the server (LM Studio, llama.cpp ...) at the address in the settings.")]


def _gaugelm(cfg: Any, db: Any) -> list[Finding]:
    from .native import NativeTrainer

    tr = NativeTrainer(cfg.native_dir)
    if cfg.llm.provider == "native":
        return []                                   # already said under Writer
    if not tr.exists():
        return [Finding("GaugeLM", "info", "Gauge's own model has not been trained yet. That is fine while another model writes.",
                        "When you are ready: Change the AI, Start training.", "engine")]
    st = tr.status()
    return [Finding("GaugeLM", "ok", f"Trained {int(st.get('step', 0) or 0):,} steps; it keeps learning while the other model writes.")]


def _teachers(cfg: Any, db: Any) -> list[Finding]:
    if cfg.llm.provider == "fake":
        return []
    from .native import school

    pool = school.teachers(cfg)
    if not pool:
        return [Finding("Teachers", "warn", "No model is available to teach GaugeLM, so it only studies on its own.",
                        "Open the Ollama app, or connect a free helper (Change the AI, Free helpers...).", "helpers")]
    names = ", ".join(t.name for t in pool[:4]) + (" and others" if len(pool) > 4 else "")
    return [Finding("Teachers", "ok", f"{len(pool)} teacher{'' if len(pool) == 1 else 's'} for GaugeLM: {names}.")]


def _devices(cfg: Any, db: Any, sensed: Any = None) -> list[Finding]:
    from .compute import scan
    from .compute.detect import backend_status

    devs = sensed if sensed is not None else scan()
    gpus = [d for d in devs if getattr(d, "kind", "") == "gpu"]
    if not gpus:
        return [Finding("Graphics card", "info", "No graphics card was found; GaugeLM trains on the processor, which is slower.")]
    nvidia = [d for d in gpus if str(getattr(d, "vendor", "")).lower() == "nvidia"]
    names = ", ".join(str(getattr(d, "name", "graphics card")) for d in gpus[:2])
    st = backend_status()
    if nvidia and not (st["cupy"] and st["cupy_devices"] > 0):
        return [Finding("Graphics card", "warn", f"Found {names}, but GaugeLM cannot use it yet (a library is missing).",
                        "Press Switches, then Set up the graphics card. It is a large download.", "devices")]
    if st["cupy"] and st["cupy_devices"] > 0:
        return [Finding("Graphics card", "ok", f"{names}: GaugeLM can train on it.")]
    return [Finding("Graphics card", "info", f"Found {names}. GaugeLM has no library for this kind of card yet, so it uses the processor.")]


def _modes(cfg: Any, db: Any) -> list[Finding]:
    from . import dryrun, limit

    out = [Finding("Capacity limit", "info", limit.describe(cfg), "", "switches")]
    if dryrun.is_dry_run() or getattr(cfg, "dry_run", False):
        out.append(Finding("Dry run", "warn", "Dry run is on: nothing is posted, published or sent to an outside service.",
                           "Switch dry run off in Switches if that is not what you want.", "switches"))
    if os.environ.get("GAUGE_OFFLINE"):
        out.append(Finding("Offline", "info", "Gauge is set to stay offline (GAUGE_OFFLINE): no searching, finding books or helpers."))
    if getattr(cfg.security, "locked", False):
        out.append(Finding("Lockdown", "warn", "Security lockdown is on, so some automation is switched off"
                           + (f" ({cfg.security.locked_reason})." if cfg.security.locked_reason else "."),
                           "Once you are sure all is well: gauge security unlock"))
    return out


def _online(cfg: Any, db: Any) -> list[Finding]:
    """Claude and the free helpers, from what is recorded here: nothing is asked of them."""
    out: list[Finding] = []
    from .llm import claude, helpers

    if cfg.llm.provider != "claude" and claude.installed() and (cfg.claude.teach or claude.saved_key()):
        st = claude.status(cfg)
        say = ("Claude teaches GaugeLM. " if st["teaching"] else "Claude is set up but not teaching. ") + st["spent_line"]
        out.append(Finding("Claude", "info", say, "", "claude"))
    on = [r for r in helpers.status(cfg) if r["connected"]]
    resting = [r["name"] for r in on if r["resting"] > 0]
    spent = [r["name"] for r in on if r["used"] >= r["allowance"]]
    if on:
        names = ", ".join(r["name"] for r in on)
        st = "warn" if resting or spent else "info"
        say = f"Connected: {names}."
        if resting:
            say += f" Resting after 'too many requests': {', '.join(resting)}."
        if spent:
            say += f" Out of today's requests: {', '.join(spent)}."
        out.append(Finding("Free helpers", st, say, "", "helpers"))
    return out


def _stories(cfg: Any, db: Any) -> list[Finding]:
    root = Path(cfg.projects_dir)
    if not root.exists():
        return []
    bad, good = [], 0
    for d in sorted(root.iterdir()):
        f = d / "project.json"
        if not f.exists():
            continue
        try:
            json.loads(f.read_text(encoding="utf-8"))["name"]
            good += 1
        except (OSError, ValueError, KeyError):
            bad.append(d.name)
    out = []
    if bad:
        out.append(Finding("Stories", "problem", f"{len(bad)} stor{'y' if len(bad) == 1 else 'ies'} cannot be opened: "
                           + ", ".join(bad[:5]) + ".",
                           "Their chapters are safe in their folders. Restore project.json from a backup, or ask for help with the report."))
    out.append(Finding("Stories", "ok" if good else "info", f"{good} stor{'y' if good == 1 else 'ies'} can be opened."
                       if good else "No story yet. Type an idea and press START MY STORY."))
    return out


CHECKS: list[tuple[str, Callable[..., list[Finding]]]] = [
    ("Python", _python), ("Gauge", _version), ("Data folder", _data), ("Library", _database), ("Writer", _writer),
    ("GaugeLM", _gaugelm), ("Teachers", _teachers), ("Graphics card", _devices), ("Modes", _modes),
    ("Online", _online), ("Stories", _stories),
]


def run(cfg: Any, db: Any = None, sensed: Any = None, progress: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Every check, in order. Returns the findings, a summary line and the report as plain text."""
    findings: list[Finding] = []
    t0 = time.monotonic()
    for name, check in CHECKS:
        if progress:
            progress(name)
        try:
            got = check(cfg, db, sensed) if check is _devices else check(cfg, db)
        except Exception as e:                       # a broken check must not hide the others
            got = [Finding(name, "warn", f"Could not check this ({type(e).__name__}: {str(e)[:120]}).")]
        findings.extend(got)
    problems = [f for f in findings if f.status == "problem"]
    warns = [f for f in findings if f.status == "warn"]
    if problems:
        summary = (f"{len(problems)} problem{'' if len(problems) == 1 else 's'}"
                   + (f" and {len(warns)} warning{'' if len(warns) == 1 else 's'}" if warns else "")
                   + f" found. Start with: {problems[0].name}: {problems[0].says}")
    elif warns:
        summary = f"Gauge works. {len(warns)} thing{'' if len(warns) == 1 else 's'} could be better; see below."
    else:
        summary = "Everything looks fine."
    ordered = sorted(findings, key=lambda f: ORDER.get(f.status, 9))
    actions = list(dict.fromkeys(f.action for f in ordered if f.action and f.status in ("problem", "warn")))
    text = (f"Gauge check, {time.strftime('%Y-%m-%d %H:%M')}\n{summary}\n\n" + "\n".join(f.line() for f in ordered)
            + "\n\n(This report contains no keys or passwords.)")
    return {"ok": not problems, "summary": summary, "findings": [asdict(f) for f in ordered], "actions": actions,
            "problems": len(problems), "warnings": len(warns), "seconds": round(time.monotonic() - t0, 1), "text": text,
            "message": summary}
