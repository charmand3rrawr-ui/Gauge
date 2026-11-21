"""What Gauge knows about programming, offline, as compact reference cards.

Cards are dense practical notes a competent engineer would want at hand: syntax that matters,
idioms, the traps, and when to use which tool. They are retrieved by similarity for the
Engineer's prompts, can be extended from your own files (`gauge code learn PATH`), and Gauge
indexes its own source so it can reason about itself (`index_self`).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..llm.embeddings import Embedder, HashingEmbedder

CARDS: dict[str, str] = {
    "python-core": """PYTHON (3.10+). Indentation defines blocks; 4 spaces. Everything is an object; names bind to objects.
Types: int (unbounded), float, bool, str (immutable, Unicode), bytes, list (mutable, ordered), tuple (immutable), dict (ordered,
insertion), set, None. Truthiness: 0, "", [], {}, None are false. Comparisons chain: 0 < x < 10. Use `is` only for None/True/False.
Strings: f"{x:.2f}", "".join(parts) (never += in loops), s.split(), s.strip(), s.startswith(), str.format for templates.
Collections: comprehensions [f(x) for x in xs if p(x)], dict(zip(keys, vals)), collections.Counter, defaultdict(list), deque.
Functions: def f(a, b=1, *args, key=None, **kw). Default args are evaluated once (never use mutable defaults). Closures capture
variables, not values. *args/**kwargs unpack. lambda for tiny functions only.
Classes: class A(Base): __init__, __repr__, @property, @classmethod, @staticmethod, @dataclass(frozen=True) for records.
Dunder methods: __eq__, __hash__, __lt__, __len__, __iter__, __getitem__, __enter__/__exit__ (context managers).
Errors: try/except SpecificError as e: ... else: ... finally: ... Raise with `raise ValueError(f"...")`; chain with `from e`.
Never catch bare Exception unless re-raising or logging at a boundary. Custom exceptions subclass Exception.
Iteration: for i, x in enumerate(xs, 1); for k, v in d.items(); zip(); reversed(); sorted(xs, key=lambda x: x.name, reverse=True).
Generators: yield; lazy pipelines; itertools (chain, islice, groupby, product). any()/all() short-circuit.
Files: with open(p, encoding="utf-8") as f: ... Always pass encoding. pathlib.Path: p / "sub", p.read_text(), p.glob("*.md"),
p.mkdir(parents=True, exist_ok=True), p.exists(), p.stem, p.suffix, p.resolve(). Use Path over os.path.
Typing: def f(x: int) -> list[str]; Optional[T] = T | None; from typing import Any, Protocol, TypeVar. dataclasses + typing = cheap records.
Stdlib to reach for first: json, csv, re, datetime (use timezone-aware or document naive), subprocess.run([...], check=True,
capture_output=True, text=True), argparse, logging, shutil, tempfile, hashlib, sqlite3, http.server, urllib.request, zipfile,
concurrent.futures.ThreadPoolExecutor (I/O), multiprocessing (CPU), functools.lru_cache, itertools, statistics, textwrap.
Scripts: if __name__ == "__main__": main(); return exit codes with sys.exit(n); print to stderr for diagnostics.
Virtual envs: python -m venv .venv; .venv\\Scripts\\activate (Windows) / source .venv/bin/activate; pip install -r requirements.txt.""",

    "python-traps": """PYTHON TRAPS. Mutable default argument (def f(xs=[])): use None and create inside. Late-binding closures in loops:
use default arg (lambda i=i: ...). Modifying a list while iterating: iterate over a copy. Integer division // vs /. is vs ==.
Shadowing builtins (list, dict, id, input). Relative imports need a package (__init__.py) and `python -m pkg.module`.
Encoding: open() without encoding uses the locale (cp1252 on Windows): always pass encoding="utf-8". Windows paths: use raw
strings r"C:\\x" or Path; backslashes escape. CRLF line endings: open(..., newline="") for csv. Floating point: use round() only
for display; compare with math.isclose. datetime.now() is naive; utcnow() is deprecated: datetime.now(timezone.utc).
Threads do not speed CPU work (GIL); use multiprocessing or numpy. subprocess with shell=True and user input = injection.
except: pass hides bugs. Circular imports: import inside functions or restructure. json dumps of datetime/Path fails: default=str.
Sorting stability and key functions; sort() returns None (in place) vs sorted(). Chained comparison with `not x in` -> `x not in`.
Recursion limit ~1000. Large loops: prefer comprehension / builtins / numpy over Python-level loops.""",

    "powershell": """POWERSHELL (Windows 10/11 built in, 5.1; pwsh 7 optional). Case-insensitive. Variables $x; arrays @(1,2); hashtables
@{k="v"}; strings "interpolated $x $($obj.Prop)" vs 'literal'. Cmdlets Verb-Noun: Get-ChildItem (ls), Get-Content, Set-Content,
Copy-Item, Move-Item, Remove-Item -Recurse -Force, New-Item -ItemType Directory, Test-Path, Join-Path, Resolve-Path,
Invoke-WebRequest -Uri -OutFile, Invoke-RestMethod (JSON to objects), ConvertTo-Json -Depth 10 / ConvertFrom-Json,
Select-Object Name,Length, Where-Object { $_.Length -gt 1MB }, ForEach-Object { }, Sort-Object, Measure-Object, Out-File -Encoding utf8.
Comparison: -eq -ne -gt -lt -like 'a*' -match 'regex' -contains. Logic: -and -or -not. Pipelines pass objects, not text.
Functions: function Name { param([string]$Path, [int]$N = 3, [switch]$Force) ... }. Script params: param(...) at top.
Errors: $ErrorActionPreference = "Stop"; try { } catch { Write-Error $_ }; -ErrorAction SilentlyContinue.
Execution policy: run scripts with `powershell -NoProfile -ExecutionPolicy Bypass -File script.ps1`.
Scheduled tasks: Register-ScheduledTask with New-ScheduledTaskAction/-Trigger (Daily -At 9am; -Once -RepetitionInterval).
Environment: $env:PATH; [Environment]::SetEnvironmentVariable("K","v","User"). Processes: Start-Process -FilePath -ArgumentList -Wait.
Exit codes: exit 1; $LASTEXITCODE after native commands. Dates: Get-Date -Format "yyyy-MM-dd". Shortcuts via WScript.Shell COM.""",

    "batch": """WINDOWS BATCH (.bat/.cmd). @echo off. Variables set X=value (no spaces around =); use %X%; delayed expansion
setlocal EnableDelayedExpansion then !X! inside loops/blocks. %~dp0 = script's folder with trailing backslash; %1..%9 args;
%* all args; %~1 strips quotes. if exist "path" (...) else (...); if "%X%"=="" ...; if errorlevel 1 (true when >= 1).
for %%f in (*.txt) do (...); for /f "tokens=1,2 delims=," %%a in (file.csv) do ...; for /l %%i in (1,1,10) do ...
call other.bat; start "" "program.exe" args (title arg first); timeout /t 5 /nobreak >nul; pause; exit /b 0.
Redirect: > file, >> append, 2>&1, >nul. Quotes around paths with spaces always. Long logic belongs in PowerShell or Python;
batch is for launching them: python -m module %* with the venv's python.exe.""",

    "bash": """BASH (Linux/macOS/WSL/Git Bash). #!/usr/bin/env bash; set -euo pipefail at top. Variables x=1 (no spaces), "$x",
"${x:-default}", arrays a=(1 2) "${a[@]}". Quote every expansion. Tests: [[ -f "$p" ]] file, -d dir, -z empty, -n non-empty,
== string equality, -eq numbers. Loops: for f in *.txt; do ...; done; while read -r line; do ...; done < file.
Functions: name() { local v="$1"; ...; return 0; }. $? last status; $(cmd) capture; $(( 1 + 2 )) arithmetic.
Pipes and filters: grep -E, sed -E 's/a/b/g', awk '{print $2}', sort | uniq -c, cut -d, -f1, xargs, find . -name '*.py' -newer x,
tr, wc -l, head/tail -n, tee. Redirect 2>&1, >/dev/null. cron: crontab -e, "0 */6 * * * cmd". chmod +x. Exit codes 0 = ok.""",

    "javascript": """JAVASCRIPT (browser / Node 18+). let/const (never var). Types: number, string, boolean, null, undefined, object,
array, function; typeof; === strict equality always. Template literals `${x}`. Arrow functions (a, b) => a + b.
Arrays: map/filter/reduce/find/some/every/flatMap/sort((a,b)=>a-b)/slice vs splice (mutates). Objects: {...spread}, destructuring
const {a, b = 1} = obj; Object.entries/keys/values; optional chaining a?.b?.[0]; nullish ??.
Async: async function f() { const r = await fetch(url); if (!r.ok) throw new Error(r.status); return r.json(); } try/catch;
Promise.all; never block. Timers setTimeout/setInterval. JSON.parse/stringify. Modules: import x from "./x.js" (ESM) or require.
DOM: document.querySelector/All, el.textContent, el.addEventListener("click", fn), fetch, localStorage. Userscripts
(Tampermonkey) run on sites you visit: // ==UserScript== header, @match. Node: fs/promises, path, process.argv, process.env,
child_process.execFile. Node scripts: node script.js; package.json "type": "module". Avoid eval and innerHTML with untrusted data.""",

    "sql-sqlite": """SQL / SQLITE (Gauge's own store). Tables: CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT NOT NULL, n REAL, at TEXT). Upsert: INSERT INTO t VALUES (...) ON CONFLICT(key) DO UPDATE SET col=excluded.col.
Query: SELECT a, COUNT(*) AS n, AVG(x) FROM t WHERE at >= ? GROUP BY a HAVING n > 1 ORDER BY n DESC LIMIT 10. JOIN ... ON.
Params: always ? placeholders (never f-strings) -> conn.execute(sql, (v1, v2)). Transactions: with conn: (commit on success).
Dates stored as ISO text sort correctly; compare with datetime('now', '-7 days') but beware 'T' vs space: use replace().
Indexes: CREATE INDEX IF NOT EXISTS ix ON t(col). EXPLAIN QUERY PLAN. PRAGMA table_info(t) for migrations; ALTER TABLE ADD COLUMN.
sqlite3 in Python: conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row; rows -> dict(r). WAL mode for concurrency.
Gauge's DB: gauge.db.Database wraps all tables (novels, chapters, features, generated, feedback, comments, llm_calls, workers...).""",

    "regex": """REGULAR EXPRESSIONS (Python re). Literals; . any char; \\d \\w \\s and negations \\D \\W \\S; [abc] [^abc] [a-z]; anchors ^ $
(use re.M for per-line); \\b word boundary. Quantifiers * + ? {m,n}; lazy with ?. Groups (...) numbered, (?P<name>...) named,
(?:...) non-capturing; alternation a|b. Lookaround (?=...) (?!...) (?<=...) (?<!...). Flags re.I re.M re.S (dot matches newline) re.X.
API: re.search (anywhere), re.match (start), re.fullmatch, re.findall (groups or whole), re.finditer (match objects),
re.sub(pat, repl, s) with \\1 or a function, re.split, re.compile once for loops. Escape user text with re.escape.
Traps: greedy .* across lines, catastrophic backtracking with nested quantifiers, forgetting raw strings r"...", [.] vs \\. .
Prefer str methods for simple cases (startswith, in, split). Test patterns on real samples before trusting them.""",

    "html-css": """HTML/CSS basics for exporting stories and small pages. Skeleton: <!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>...</title><style>...</style></head><body>...</body></html>.
Semantic tags: article, section, h1-h3, p, em/strong, blockquote, hr, nav, header, footer. Escape &, <, > in text
(html.escape in Python). Markdown to HTML: paragraphs = blank-line separated blocks -> <p>; *italic* -> em; **bold** -> strong;
# heading -> h1. CSS: body{max-width:42rem;margin:2rem auto;font:18px/1.6 Georgia,serif;padding:0 1rem} p{margin:0 0 1em}
@media (prefers-color-scheme: dark){body{background:#111;color:#eee}}. Royal Road and most serial sites accept pasted HTML with
p/em/strong/hr; strip everything else. EPUB = zip of XHTML + OPF manifest + NCX/nav (mimetype first, uncompressed).""",

    "http-data": """HTTP AND DATA FORMATS. requests: r = requests.get(url, headers={"User-Agent": "..."}, timeout=30); r.raise_for_status();
r.json() / r.text; sessions for cookies; retries with backoff on 429/5xx; honour Retry-After; never hammer a site (sleep between).
urllib.request for zero dependencies. JSON: json.load/dump(obj, f, indent=2, ensure_ascii=False); JSONL = one object per line.
CSV: csv.DictReader/DictWriter(newline=""), encoding utf-8-sig for Excel. YAML/TOML: tomllib (3.11+) for reading TOML.
Dates: ISO 8601 strings; datetime.fromisoformat; timezone-aware for anything shared. Hashing: hashlib.sha256(b).hexdigest().
Caching: key = sha1(url); store body + timestamp; expire by age. Robots: honour robots.txt and terms; identify your client.""",

    "testing-quality": """TESTING AND QUALITY. pytest: files test_*.py, functions test_*, plain asserts; fixtures (tmp_path, monkeypatch, capsys);
parametrize; run `pytest -q`; -x stop at first failure; -k expr select. Write the test before the fix; one behaviour per test;
deterministic (seed randomness, fake time, no network: inject fakes). Assert messages explain the failure. Keep tests fast.
Lint/format: ruff check --fix; ruff format. Type hints everywhere public. Logging over print in libraries: logging.getLogger(__name__).
Errors at boundaries: validate inputs early, fail loudly with a clear message, exit code != 0 in scripts. Idempotent scripts:
re-running does no harm (check exists, upsert, atomic writes via temp file + replace). Never delete without a backup or --dry-run.
Small functions, pure where possible, side effects at the edges. A script that needs no judgement needs no model: use code.""",

    "windows-automation": """WINDOWS AUTOMATION. Task Scheduler: schtasks /create /tn "Name" /tr "C:\\path\\run.bat" /sc daily /st 09:00 or the
PowerShell cmdlets (Register-ScheduledTask). Startup folder: shell:startup. Services need admin; prefer scheduled tasks.
Notifications: PowerShell BurntToast module or msg * "text" (simple). Clipboard: Set-Clipboard / Get-Clipboard; Python pyperclip
or `clip`. File watching: Python watchdog (pip) or polling os.stat mtimes every N seconds (no dependency). Open files/urls:
os.startfile(path) (Windows only). Paths: use pathlib; %APPDATA%, %LOCALAPPDATA%, %USERPROFILE% via os.environ.
Run hidden: pythonw.exe; start "" /min. Long-running loops: handle Ctrl+C (KeyboardInterrupt), write a log file, exit cleanly.
Avoid admin rights; avoid modifying system PATH; install per-user. Antivirus may flag fresh .exe/.bat: keep scripts readable.""",

    "security-basics": """SECURITY BASICS (for every program Gauge writes). Treat all input as hostile: validate type, length and range at the
boundary; fail closed with a clear message. Never build shell commands from strings (subprocess with a list, shell=False);
never eval/exec; never unpickle or deserialise untrusted data (json only; np.load(allow_pickle=False)). Paths: resolve and
check they stay inside the allowed folder (Path.resolve(), relative_to) before reading or writing; never follow '..'. Secrets
live in files with owner-only permissions or the OS keychain, never in code, logs or URLs; redact tokens when printing
(show last 4). Least privilege: no admin, no system PATH edits, no registry writes, no scheduled tasks unless the task is exactly
that. Network: pin exact hosts, timeouts always, TLS verification on, identify the client, back off on 429/5xx. Logging: no
secrets, no raw user content at INFO. Dependencies: standard library first; pin versions; prefer pure-Python. Idempotent and
reversible: back up before overwrite, --dry-run for anything destructive. Tests include the hostile cases (empty, huge,
wrong type, path traversal, unicode).""",

    "security-threats": """THREATS AGAINST GAUGE AND THE DEFENCES. 1) Prompt injection: scraped chapters, comments and research text may
contain instructions aimed at the model ("ignore previous instructions", fake system markers, requests to run commands or
exfiltrate config.json). Defence: gauge.security.sanitize scrubs control/zero-width characters, neutralises instruction-shaped
spans, fences untrusted text in prompts, and logs findings; untrusted text never reaches the Engineer or the self-programmer.
2) Malicious generated code: a model can be steered to write a bot that deletes files, phones home or persists. Defence:
gauge.security.codescan blocks dynamic execution, shell=True, encoded payloads, credential paths, registry/task changes;
bots run in gauge.security.sandbox with a minimal env (secrets stripped, GAUGE_NO_LLM, GAUGE_NO_NETWORK + dead proxy), a
timeout, POSIX resource limits; deterministic bots are refused a model in code. 3) Worker abuse: forged or brute-forced
requests. Defence: shared token with constant-time compare, private-network peers only by default, per-peer rate limit,
failed-auth lockout, body size cap, npz without pickle, every refusal audited. 4) Tampering with Gauge's own code or models:
Defence: gauge.security.integrity seals SHA-256 hashes signed with a local HMAC key; the daemon verifies each cycle and
refuses self-modification and bot runs while tampered. 5) Covering tracks: Defence: the audit log is a hash chain; editing
or deleting an entry breaks it. 6) Runaway automation: Defence: the sentinel scores threat from audit counts and integrity
and locks down self-heal, auto-apply, bots, pool and research until a person unlocks. Limits, honestly: a Python process on
Windows is not a jail; the sandbox is defence in depth, not containment; there is no "mythical" tier.""",

    "security-incident": """INCIDENT RESPONSE FOR GAUGE. 1) Contain: `gauge security lockdown "reason"` switches off self-heal, auto-apply,
bots, worker pool, research and daemon writing; stop the daemon; stop workers (`Ctrl+C` where `gauge worker` runs). 2) Assess:
`gauge security status` (threat level, counts), `gauge security verify` (which files changed), `gauge security log --min 2`
(what happened, in order; the chain tells you if the log itself was edited), `gauge self list` (recent proposals/applications),
`gauge bot list` (what ran). 3) Recover: `gauge self rollback <id>` for applied proposals; restore changed files from git
(`git diff`, `git checkout -- path`) or from drafts/accepted copies; rotate the worker token (`gauge compute token --rotate`)
and restart workers with the new one; re-seal (`gauge security seal`) only after the tree is clean. 4) Learn: add the pattern
to the sanitiser or scanner if an injection or bad code got through; write a test that reproduces it; `gauge self unlock`
when done. Keep backups of data/ off the machine; BitLocker or an encrypted container protects it at rest.""",

    "active-defense": """ACTIVE DEFENCE, ALL INWARD (gauge.security.defense + gauge.security.range). A single dial (security.active_defense_level
0..3, auto-escalating from the sentinel when security.active_defense_auto) decides how hard Gauge pushes back on ITS OWN machine:
tarpit (slow responses to a suspicious peer), honeytokens (fake secrets that alert if they ever appear in traffic -> exfiltration proof),
decoys (a plausible-but-fake response to a rejected prober), and evidence capture (durable bundles for the owner; abuse_report_draft
drafts a report to send, never sends it). None of it reaches another system; there is no hack-back. The defensive cyber-range
(`gauge security range`) fires synthetic attacks at the sanitiser, code scanner, worker Guard and array unpacker and scores what is
caught, surfacing gaps to harden. The adversarial drill (`gauge security drill`) goes further: every round it MUTATES the attacks
(leetspeak, zero-width, keyword-splitting, synonyms, aliased imports) and RANDOMISES the defence parameters (body cap, rate limit,
lockout, injection action), so the defences cannot overfit to a fixed test set; attacks that slip past are logged as evasions to
harden next. All of it is inward: generated payloads are test inputs for Gauge's own detectors, never executed, never sent, never
written out as a tool. `gauge security defense` shows the posture and plants honeytokens.""",

    "programming-languages": """READING MANY LANGUAGES (gauge.code.languages). Gauge writes bots in Python but reads many languages to understand
dependencies and port idioms. detect(filename, code) ids the language (extension first, then Pygments); analyze(filename, code) ->
CodeShape (language, lines, tokens, comment_ratio, functions, classes, max_indent, smells); tips(lang)/idioms(lang)/gotchas(lang)
give the knowledge base for python, js, ts, rust, go, c, cpp, java, csharp, ruby, powershell, bash, sql. This is literacy, not
uniform expert authorship: strongest when writing Python, careful and structural when reading the rest. `gauge code lang <file>`.""",

    "binary-level": """SPEAKING AT THE BINARY LEVEL (gauge.code.binary). The useful, honest version of "down to the binary": identify(data)
names a blob from its magic number (ELF, PE, ZIP, PNG, PDF, SQLite...); hexdump(data) is the offset/hex/ASCII view; to_bits/from_bits
and to_hex/from_hex convert; pack_struct/unpack_struct/parse_record read and write binary records with struct; disassemble(func or
source) shows Python bytecode via dis; entropy(data) flags compressed/encrypted regions. Binary is a representation; the skill is
crossing between layers, not mysticism. `gauge code binary <file>`, `gauge code disasm <file.py>`.""",

    "mutation-testing": """MUTATION TESTING SELF-EDITS (gauge.code.mutation). A change that passes the suite might still be untested. The mutation
gate breaks the changed code on purpose (== becomes !=, + becomes -, True becomes False, a boundary flips) and confirms the tests then
fail. Surviving mutants are coverage holes. Turn it on with code.mutation_check; a proposal whose tests kill fewer than
code.mutation_min_score of mutants is blocked so a human looks first. It runs only after the normal tests pass, on changed non-test
Python files, bounded by code.mutation_max_mutants.""",

    "gauge-internals": """GAUGE INTERNALS (for bots and self-changes). Package `gauge`: config.GaugeConfig.load(data_dir) -> .db_path, .models_dir,
.projects_dir, .native_dir; db.Database(path) with novels(), chapters(key), features(level), feedback(project), call_summary(),
notes(), workers(); write.project.Project(root, name): state.outline, read_chapter(n), write_chapter(); features.chapter_features(ChapterRecord);
llm.make_llm(cfg.llm) -> .generate(prompt, system, max_tokens, json_mode, schema, purpose); llm.generate_json(llm, prompt, schema);
ingest.local.import_path(path); research.Scout; compute.scan(). CLI in gauge/cli.py (click groups). Tests in tests/ with FakeLLM
(deterministic) and the `cfg`/`db` fixtures from tests/conftest.py; GAUGE_OFFLINE=1 blocks network. Style: ruff (E,F,W,I,B,UP),
line length 100, type hints, docstrings that say why. Data lives under data/ (never committed). Prompts in write/prompts.py and
agent modules; schemas in write/schemas.py. Add a CLI command = a @main.command()/group function using Ctx (c.cfg, c.db).""",
}


@dataclass
class Card:
    key: str
    title: str
    text: str
    source: str = "builtin"


class KnowledgeBase:
    def __init__(self, embedder: Optional[Embedder] = None, extra_dir: Optional[Path] = None):
        self.embedder = embedder or HashingEmbedder(384)
        self.cards: list[Card] = [Card(k, k.replace("-", " ").title(), v) for k, v in CARDS.items()]
        self.extra_dir = extra_dir
        if extra_dir and extra_dir.exists():
            for p in sorted(extra_dir.glob("*.md")) + sorted(extra_dir.glob("*.txt")):
                self.cards.append(Card(p.stem, p.stem.replace("-", " ").title(), p.read_text(encoding="utf-8", errors="replace")[:6000], str(p)))
        self._matrix: Optional[np.ndarray] = None

    def _ensure(self) -> np.ndarray:
        if self._matrix is None or self._matrix.shape[0] != len(self.cards):
            self._matrix = self.embedder.embed([f"{c.title}\n{c.text}" for c in self.cards])
        return self._matrix

    def search(self, query: str, k: int = 3) -> list[Card]:
        if not self.cards:
            return []
        m = self._ensure()
        q = self.embedder.embed([query])[0]
        order = np.argsort(-(m @ q))
        return [self.cards[int(i)] for i in order[:k]]

    def add_card(self, key: str, text: str, source: str = "learned") -> Card:
        c = Card(key, key.replace("-", " ").title(), text[:6000], source)
        self.cards = [x for x in self.cards if x.key != key] + [c]
        self._matrix = None
        if self.extra_dir:
            self.extra_dir.mkdir(parents=True, exist_ok=True)
            (self.extra_dir / f"{key}.md").write_text(text, encoding="utf-8")
        return c

    def learn_file(self, path: Path) -> list[Card]:
        """Add a file of notes/code as one or more cards (split on top-level Markdown headings)."""
        text = path.read_text(encoding="utf-8", errors="replace")
        parts = re.split(r"^# +", text, flags=re.M)
        out = []
        if len(parts) <= 1 or not re.search(r"^# +\S", text, re.M):
            out.append(self.add_card(re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-"), text, str(path)))
        else:
            for part in parts[1:]:
                title, _, body = part.partition("\n")
                key = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
                if key:
                    out.append(self.add_card(key, body.strip(), str(path)))
        return out

    def index_self(self, package_dir: Path) -> int:
        """Index Gauge's own source files so it can reason about itself (module docstring + signatures)."""
        n = 0
        for p in sorted(package_dir.rglob("*.py")):
            if p.name == "__init__.py":
                continue
            try:
                src = p.read_text(encoding="utf-8")
            except OSError:
                continue
            doc = re.search(r'^"""(.*?)"""', src, re.S)
            sigs = re.findall(r"^(?:class|def|    def) [^\n]+", src, re.M)
            summary = (doc.group(1).strip() if doc else "") + "\n" + "\n".join(sigs[:60])
            rel = p.relative_to(package_dir.parent).as_posix()
            self.cards.append(Card(f"self:{rel}", rel, summary[:6000], rel))
            n += 1
        self._matrix = None
        return n

    def render(self, cards: list[Card], max_chars: int = 6000) -> str:
        out, used = [], 0
        for c in cards:
            t = f"### {c.title}\n{c.text}"
            if used + len(t) > max_chars:
                t = t[: max(0, max_chars - used)]
            out.append(t)
            used += len(t)
            if used >= max_chars:
                break
        return "\n\n".join(out)

    def to_json(self) -> str:
        return json.dumps([asdict(c) for c in self.cards], ensure_ascii=False)
