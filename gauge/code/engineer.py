"""The Engineer: writes standalone bots (plain programs) and only reaches for a model when a task needs judgement.

Workflow for `gauge bot new NAME --task "..."`:
1. classify: does this task need a language model at run time, or is it deterministic? Which language?
2. write: main.py (+ a PowerShell/batch launcher when asked), test_main.py, manifest.json, README.md,
   drafted with the knowledge cards and the toolkit API in the prompt.
3. test: run the bot's tests in a subprocess with a timeout; on failure feed the errors back and
   rewrite, up to `max_fix_rounds` times.
4. register: the bot lives under data/bots/<name>/ with a schedule the daemon honours.

A bot never imports the model unless its manifest says `needs_ai: true`; the runner sets
GAUGE_BOT=1 so library code can refuse model calls in deterministic bots.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..llm.client import LLM, generate_json
from .knowledge import KnowledgeBase

log = logging.getLogger(__name__)

SYSTEM = """You are a careful senior engineer writing small, dependency-free automation for a Windows desktop user.
You prefer plain code to clever code, the standard library to packages, and you never call a language model from a
program unless the task genuinely needs judgement about natural language. Every program you write has a test, handles
errors with clear messages, is idempotent, and exits non-zero on failure."""

DETERMINISTIC_HINTS = ("count", "rename", "copy", "backup", "convert", "export", "csv", "json", "schedule", "watch", "report",
                       "statistics", "stats", "import", "move", "zip", "format", "html", "epub", "sync", "clean", "delete", "list")
AI_HINTS = ("write a", "summarize", "summarise", "rewrite", "judge", "decide if", "tone", "sentiment", "translate", "paraphrase",
            "story", "chapter", "critique", "name for", "ideas")

TOOLKIT_API = """from gauge.code import toolkit  # deterministic helpers you may use:
toolkit.read_text(path) -> str                  toolkit.write_text_atomic(path, text)
toolkit.list_files(folder, patterns=("*.md",), recursive=True) -> list[Path]   toolkit.natural_key(path)
toolkit.file_hash(path) -> sha256 hex           toolkit.backup(src, dest_dir, keep=10) -> Path
toolkit.watch_folder(folder, callback, interval_s=5, patterns=("*",), iterations=None)
toolkit.word_count(text) -> int                 toolkit.text_stats(text) -> {words, sentences, paragraphs, avg_sentence_words, dialogue_lines, reading_minutes}
toolkit.strip_front_matter(md) -> str           toolkit.markdown_to_html(md) -> str   toolkit.html_page(title, body_html) -> str
toolkit.read_json(path, default) / write_json(path, obj) / append_jsonl(path, obj)
toolkit.read_csv(path) -> list[dict] / write_csv(path, rows, fieldnames=None)
toolkit.now_iso() -> str                        toolkit.http_get(url, cache_dir=None, max_age_s=3600) -> str
toolkit.run(cmd: list[str], cwd=None, timeout=600) -> (code, out, err)        toolkit.python_exe() -> str
Gauge data: from gauge.config import GaugeConfig; cfg = GaugeConfig.load(os.environ.get("GAUGE_DATA_DIR", "data"));
from gauge.db import Database; db = Database(cfg.db_path); projects under cfg.projects_dir/<slug>/chapters/chapter_NNN.md"""


@dataclass
class BotSpec:
    name: str
    task: str
    needs_ai: bool = False
    language: str = "python"
    schedule: str = ""  # "" | "hourly" | "daily" | "every:N" (minutes) | "on-cycle"
    files: dict[str, str] = field(default_factory=dict)
    notes: str = ""


class BotRegistry:
    def __init__(self, bots_dir: Path):
        self.dir = bots_dir

    def path(self, name: str) -> Path:
        return self.dir / _slug(name)

    def exists(self, name: str) -> bool:
        return (self.path(name) / "manifest.json").exists()

    def manifest(self, name: str) -> dict[str, Any]:
        return json.loads((self.path(name) / "manifest.json").read_text(encoding="utf-8"))

    def save(self, spec: BotSpec, extra: Optional[dict[str, Any]] = None) -> Path:
        d = self.path(spec.name)
        d.mkdir(parents=True, exist_ok=True)
        for fname, content in spec.files.items():
            if ".." in fname or fname.startswith(("/", "\\")):
                continue
            (d / fname).write_text(content, encoding="utf-8")
        man = {"name": spec.name, "task": spec.task, "needs_ai": spec.needs_ai, "language": spec.language, "schedule": spec.schedule,
               "entry": "main.py" if spec.language == "python" else next(iter(spec.files), "main.py"), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "runs": 0, "last_run": None, "last_status": None, **(extra or {})}
        (d / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        return d

    def list(self) -> list[dict[str, Any]]:
        if not self.dir.exists():
            return []
        out = []
        for d in sorted(self.dir.iterdir()):
            if (d / "manifest.json").exists():
                try:
                    out.append(json.loads((d / "manifest.json").read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
        return out

    def remove(self, name: str) -> bool:
        d = self.path(name)
        if not d.exists():
            return False
        shutil.rmtree(d)
        return True

    def run(self, name: str, args: Optional[list[str]] = None, timeout: int = 1800, env_extra: Optional[dict[str, str]] = None) -> dict[str, Any]:
        man = self.manifest(name)
        d = self.path(name)
        entry = d / man.get("entry", "main.py")
        env = dict(os.environ)
        env["GAUGE_BOT"] = "1"
        if not man.get("needs_ai"):
            env["GAUGE_NO_LLM"] = "1"  # deterministic bots may not reach the model
        env.update(env_extra or {})
        if entry.suffix == ".py":
            cmd = [sys.executable, str(entry), *(args or [])]
        elif entry.suffix == ".ps1":
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(entry), *(args or [])]
        elif entry.suffix in (".bat", ".cmd"):
            cmd = ["cmd", "/c", str(entry), *(args or [])]
        else:
            cmd = [str(entry), *(args or [])]
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=str(d), capture_output=True, text=True, timeout=timeout, env=env, check=False)
            code, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            code, out, err = 124, "", f"timed out after {timeout}s"
        man["runs"] = int(man.get("runs", 0)) + 1
        man["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        man["last_status"] = code
        (d / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        with (d / "run.log").open("a", encoding="utf-8") as f:
            f.write(f"\n=== {man['last_run']} exit {code} ({time.monotonic() - t0:.1f}s)\n{out}\n{err}\n")
        return {"name": name, "exit": code, "stdout": out[-4000:], "stderr": err[-4000:], "seconds": round(time.monotonic() - t0, 1)}

    def due(self, now: Optional[float] = None) -> list[str]:
        """Bots whose schedule says they should run now."""
        now = now or time.time()
        out = []
        for m in self.list():
            sched = m.get("schedule") or ""
            if not sched:
                continue
            last = m.get("last_run")
            last_ts = time.mktime(time.strptime(last, "%Y-%m-%dT%H:%M:%S")) if last else 0.0
            gap = {"hourly": 3600, "daily": 86400, "on-cycle": 0}.get(sched)
            if gap is None and sched.startswith("every:"):
                try:
                    gap = int(sched.split(":")[1]) * 60
                except ValueError:
                    gap = None
            if gap is not None and now - last_ts >= gap:
                out.append(m["name"])
        return out


def run_tests(bot_dir: Path, timeout: int = 300) -> tuple[bool, str]:
    env = dict(os.environ)
    env["GAUGE_BOT"] = "1"
    env["GAUGE_NO_LLM"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(bot_dir)], capture_output=True, text=True,
                           timeout=timeout, env=env, check=False, cwd=str(bot_dir))
    except subprocess.TimeoutExpired:
        return False, f"tests timed out after {timeout}s"
    return p.returncode == 0, (p.stdout + p.stderr)[-6000:]


class Engineer:
    def __init__(self, llm: LLM, kb: KnowledgeBase, bots_dir: Path, max_fix_rounds: int = 2):
        self.llm = llm
        self.kb = kb
        self.registry = BotRegistry(bots_dir)
        self.max_fix_rounds = max_fix_rounds

    # ---- 1. classify -------------------------------------------------------------------
    def classify(self, task: str) -> dict[str, Any]:
        low = task.lower()
        heur_ai = any(h in low for h in AI_HINTS) and not any(h in low for h in ("count", "export", "backup", "csv", "html"))
        d = generate_json(self.llm, f"""[engineer:classify]
Task: {task}

Decide: does running this need a language model (judgement about meaning, style or open-ended text), or can plain code
do it deterministically (files, counts, formats, schedules, data, HTTP)? Default to code. Which language fits a Windows
desktop: python (default), powershell (Windows admin/shell tasks), batch (launchers), javascript (browser userscript).
Respond with ONLY JSON: {{"needs_ai": true/false, "language": "python|powershell|batch|javascript", "why": "one line",
"schedule": "|hourly|daily|every:N|on-cycle"}}""", temperature=0.1, max_tokens=200, purpose="engineer:classify")
        needs_ai = bool(d.get("needs_ai", heur_ai)) if isinstance(d.get("needs_ai"), bool) else heur_ai
        lang = str(d.get("language", "python")).lower()
        if lang not in ("python", "powershell", "batch", "javascript"):
            lang = "python"
        return {"needs_ai": needs_ai, "language": lang, "why": str(d.get("why", ""))[:200], "schedule": str(d.get("schedule", "") or "")}

    # ---- 2. write ------------------------------------------------------------------------
    def _prompt(self, spec: BotSpec, feedback: str = "") -> str:
        cards = self.kb.render(self.kb.search(f"{spec.language} {spec.task}", k=3), max_chars=4500)
        ai_rule = ("This bot MAY use the model: `from gauge.llm import make_llm` with `GaugeConfig.load(...)`, only for the judgement part."
                   if spec.needs_ai else "This bot must NOT import or call any language model. If GAUGE_NO_LLM is set, that is enforced.")
        files_hint = ('{"main.py": "...", "test_main.py": "...", "README.md": "..."}' if spec.language == "python" else
                      f'{{"main.{ "ps1" if spec.language == "powershell" else "bat" if spec.language == "batch" else "js"}": "...", "main.py": "<thin python wrapper that runs it via subprocess so tests work>", "test_main.py": "...", "README.md": "..."}}')
        fb = ("PREVIOUS ATTEMPT FAILED ITS TESTS. Fix these problems and return ALL files again:\n" + feedback) if feedback else ""
        return f"""[engineer:write]
Write a complete, runnable bot.

NAME: {spec.name}
TASK: {spec.task}
LANGUAGE: {spec.language}. {ai_rule}

Requirements:
- main.py takes arguments with argparse (paths etc. as options with sensible defaults), prints a short summary, exits 0 on
  success and non-zero with a clear message on failure. Idempotent: safe to run twice.
- test_main.py uses pytest and tmp_path; it must exercise the real behaviour on small fixture data and pass offline.
- Use the toolkit below where it fits; otherwise the standard library only.
- README.md: what it does, how to run it, how to schedule it (gauge bot schedule {spec.name} daily).

TOOLKIT:
{TOOLKIT_API}

REFERENCE CARDS:
{cards}
{fb}
Respond with ONLY JSON: {{"files": {files_hint}, "notes": "one line about design choices"}}"""

    def write(self, spec: BotSpec, feedback: str = "") -> BotSpec:
        d = generate_json(self.llm, self._prompt(spec, feedback), system=SYSTEM, temperature=0.3, max_tokens=4000, purpose="engineer:write")
        files = d.get("files") if isinstance(d.get("files"), dict) else {}
        clean = {}
        for name, content in files.items():
            name = str(name).strip()
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not isinstance(content, str):
                continue
            clean[name] = content
        if clean:
            spec.files = clean
        spec.notes = str(d.get("notes", ""))[:300]
        return spec

    # ---- 3. test loop + 4. register ----------------------------------------------------
    def build(self, name: str, task: str, needs_ai: Optional[bool] = None, language: Optional[str] = None, schedule: Optional[str] = None,
              progress=None) -> dict[str, Any]:
        cls = self.classify(task)
        spec = BotSpec(name=name, task=task, needs_ai=cls["needs_ai"] if needs_ai is None else needs_ai,
                       language=language or cls["language"], schedule=schedule if schedule is not None else cls["schedule"])
        feedback = ""
        passed, report = False, ""
        for attempt in range(self.max_fix_rounds + 1):
            spec = self.write(spec, feedback)
            if not spec.files or "main.py" not in spec.files:
                report = "the model returned no main.py"
                break
            bot_dir = self.registry.save(spec, {"classification": cls, "attempts": attempt + 1})
            passed, report = run_tests(bot_dir) if any(f.startswith("test_") for f in spec.files) else (True, "no tests provided")
            if progress:
                progress({"attempt": attempt + 1, "passed": passed})
            if passed:
                break
            feedback = report
        man = self.registry.manifest(name) if self.registry.exists(name) else {}
        man["tests_passed"] = passed
        man["test_report"] = report[-2000:]
        if man:
            (self.registry.path(name) / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        return {"name": name, "needs_ai": spec.needs_ai, "language": spec.language, "schedule": spec.schedule, "tests_passed": passed,
                "files": sorted(spec.files), "dir": str(self.registry.path(name)), "report": report[-1500:], "why": cls["why"], "notes": spec.notes}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "bot"
