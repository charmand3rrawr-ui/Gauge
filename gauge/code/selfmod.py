"""The Self-Programmer: Gauge changes its own source, carefully.

A change is a *proposal*: full new contents for one to three source files, produced from the
instruction (or from an error in the journal), the current file text, and Gauge's self-index.
Proposals are tested in an isolated copy of the repository (ruff + the whole pytest suite) and
applied only when everything passes, with the originals backed up for `rollback`. By default a
human runs `gauge self apply`; `self.auto_apply` lets the daemon apply proposals that pass, and
even then the guardrails hold: protected paths cannot be touched, tests cannot be weakened, and a
proposal cannot grow or shrink a file beyond sane bounds.
"""
from __future__ import annotations

import difflib
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

PROTECTED = ("installer/", "Install Gauge.bat", "Gauge Studio.bat", "install.ps1", ".github/", "pyproject.toml", "LICENSE",
             "gauge/code/selfmod.py", "scripts/", "requirements.txt")
MAX_FILES = 3
MAX_GROWTH = 2.5  # a rewritten file may not become 2.5x longer or shorter than 40% of the original
SYSTEM = """You are the maintainer of this codebase making one careful, minimal change. You keep the existing style (type
hints, docstrings that say why, ruff-clean, line length 100), you never weaken or delete tests, you prefer the smallest
change that fully solves the problem, and you return complete files, not fragments."""


@dataclass
class Proposal:
    id: str
    instruction: str
    files: dict[str, str]  # relative path -> new full content
    rationale: str = ""
    created: str = ""
    tested: Optional[bool] = None
    test_report: str = ""
    applied: bool = False
    diff: str = ""
    source: str = "user"  # user | journal
    meta: dict[str, Any] = field(default_factory=dict)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class SelfProgrammer:
    def __init__(self, llm: LLM, kb: KnowledgeBase, self_dir: Path, root: Optional[Path] = None, test_timeout: int = 900):
        self.llm = llm
        self.kb = kb
        self.dir = self_dir
        self.root = root or repo_root()
        self.test_timeout = test_timeout
        (self.dir / "proposals").mkdir(parents=True, exist_ok=True)

    # ---- guardrails -----------------------------------------------------------------------
    def check_path(self, rel: str) -> Optional[str]:
        norm = rel.replace("\\", "/")
        if ".." in norm or norm.startswith("/") or not norm:
            return f"refused: path escapes the repository ({rel})"
        if any(norm == p.rstrip("/") or norm.startswith(p) for p in PROTECTED):
            return f"refused: {rel} is protected"
        if not (norm.endswith(".py") or norm.endswith(".md")):
            return f"refused: only .py and .md files may be changed ({rel})"
        return None

    def check_content(self, rel: str, new: str, old: Optional[str]) -> Optional[str]:
        try:
            if rel.endswith(".py"):
                compile(new, rel, "exec")
        except SyntaxError as e:
            return f"refused: {rel} does not compile ({e})"
        if rel.endswith(".py") and not rel.startswith("tests/"):
            from ..security.codescan import report as scan_report
            from ..security.codescan import scan_text, worst

            findings = scan_text(rel, new)
            old_findings = scan_text(rel, old) if old else []
            # New blocking constructs are refused; pre-existing ones (e.g. documented subprocess use) are allowed to stay.
            new_blocks = [f for f in findings if f.severity >= 3 and f.rule not in {o.rule for o in old_findings}]
            if new_blocks:
                return "refused: security scan blocks new construct(s) in " + rel + ": " + scan_report([(rel, f) for f in new_blocks])[:300]
            if worst(findings) >= 3 and not old:
                return f"refused: new file {rel} contains blocked constructs"
        if old is not None and old.strip():
            ratio = len(new) / max(1, len(old))
            if ratio > MAX_GROWTH or ratio < 0.4:
                return f"refused: {rel} would change size by {ratio:.1f}x"
        if rel.replace("\\", "/").startswith("tests/") and old is not None:
            # Tests may only grow: every existing test function must survive.
            old_tests = set(re.findall(r"^def (test_\w+)", old, re.M))
            new_tests = set(re.findall(r"^def (test_\w+)", new, re.M))
            missing = old_tests - new_tests
            if missing:
                return f"refused: would remove tests {sorted(missing)}"
            if "pytest.skip" in new and "pytest.skip" not in old:
                return "refused: would add a skip to tests"
        return None

    # ---- proposing --------------------------------------------------------------------------
    def relevant_files(self, instruction: str, k: int = 4) -> list[str]:
        cards = [c for c in self.kb.search(instruction, k=k + 4) if c.key.startswith("self:")]
        out = []
        for c in cards:
            rel = c.key[len("self:"):]
            if self.check_path(rel) is None and (self.root / rel).exists():
                out.append(rel)
            if len(out) >= k:
                break
        return out

    def files_in_context(self, context: str) -> list[str]:
        """Repo files named in a traceback or error text, most recent frame first."""
        out: list[str] = []
        for m in re.finditer(r"((?:gauge|tests)/[\w/.-]+\.py)", context.replace("\\", "/")):
            rel = m.group(1)
            if rel not in out and self.check_path(rel) is None and (self.root / rel).exists():
                out.append(rel)
        return list(reversed(out))[:MAX_FILES]

    def propose(self, instruction: str, files: Optional[list[str]] = None, source: str = "user", context: str = "") -> Proposal:
        rels = files or (self.files_in_context(context) if context else []) or self.relevant_files(instruction)
        shown = []
        for rel in rels[:MAX_FILES]:
            p = self.root / rel
            if p.exists():
                shown.append(f"===== {rel} =====\n{p.read_text(encoding='utf-8')[:14000]}")
        cards = self.kb.render([c for c in self.kb.search(instruction, k=3) if not c.key.startswith("self:")], max_chars=2500)
        if context:
            from ..security.sanitize import fence, scrub

            context = fence("error context", scrub(context).text[:3000])
        ctx_block = ("CONTEXT (error / journal entry):\n" + context) if context else ""
        prompt = f"""[self:propose]
INSTRUCTION: {instruction}
{ctx_block}
Rules: change at most {MAX_FILES} files; return each changed file COMPLETE; you may add a new test file under tests/ but never
remove or skip existing tests; do not touch protected files ({', '.join(PROTECTED)}); keep the style; smallest change that fully works.

CURRENT FILES:
{chr(10).join(shown) if shown else '(none selected; name the files you need in "files")'}

REFERENCE:
{cards}

Respond with ONLY JSON: {{"files": {{"<relative path>": "<complete new content>"}}, "rationale": "two or three sentences",
"tests_added": ["test names you added, if any"]}}"""
        d = generate_json(self.llm, prompt, system=SYSTEM, temperature=0.2, max_tokens=6000, purpose="self:propose")
        files_out: dict[str, str] = {}
        refused: list[str] = []
        for rel, content in (d.get("files") or {}).items() if isinstance(d.get("files"), dict) else []:
            rel = str(rel).replace("\\", "/")
            if not isinstance(content, str):
                continue
            why = self.check_path(rel)
            if why is None:
                old = (self.root / rel).read_text(encoding="utf-8") if (self.root / rel).exists() else None
                why = self.check_content(rel, content, old)
            if why:
                refused.append(why)
                continue
            files_out[rel] = content
            if len(files_out) >= MAX_FILES:
                break
        pid = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^a-z0-9]+", "-", instruction.lower())[:30].strip("-")
        prop = Proposal(id=pid, instruction=instruction, files=files_out, rationale=str(d.get("rationale", ""))[:600],
                        created=time.strftime("%Y-%m-%dT%H:%M:%S"), source=source, meta={"refused": refused, "tests_added": d.get("tests_added", [])})
        prop.diff = self.diff(prop)
        self.save(prop)
        return prop

    def diff(self, prop: Proposal) -> str:
        out = []
        for rel, new in prop.files.items():
            p = self.root / rel
            old = p.read_text(encoding="utf-8").splitlines(keepends=True) if p.exists() else []
            out.extend(difflib.unified_diff(old, new.splitlines(keepends=True), fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        return "".join(out)

    # ---- persistence -------------------------------------------------------------------------
    def _pdir(self, pid: str) -> Path:
        return self.dir / "proposals" / pid

    def save(self, prop: Proposal) -> None:
        d = self._pdir(prop.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "proposal.json").write_text(json.dumps({k: v for k, v in prop.__dict__.items() if k != "files"}, indent=2, default=str), encoding="utf-8")
        for rel, content in prop.files.items():
            fp = d / "files" / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
        (d / "change.diff").write_text(prop.diff, encoding="utf-8")

    def load(self, pid: str) -> Proposal:
        d = self._pdir(pid)
        meta = json.loads((d / "proposal.json").read_text(encoding="utf-8"))
        files = {}
        base = d / "files"
        if base.exists():
            for fp in base.rglob("*"):
                if fp.is_file():
                    files[fp.relative_to(base).as_posix()] = fp.read_text(encoding="utf-8")
        return Proposal(files=files, **{k: v for k, v in meta.items() if k in Proposal.__dataclass_fields__ and k != "files"})

    def list(self) -> list[dict[str, Any]]:
        out = []
        for d in sorted((self.dir / "proposals").iterdir()):
            p = d / "proposal.json"
            if p.exists():
                m = json.loads(p.read_text(encoding="utf-8"))
                out.append({k: m.get(k) for k in ("id", "instruction", "tested", "applied", "source", "created")})
        return out

    # ---- testing in isolation ---------------------------------------------------------------------
    def test(self, prop: Proposal) -> tuple[bool, str]:
        if not prop.files:
            prop.tested, prop.test_report = False, "proposal has no files"
            self.save(prop)
            return False, prop.test_report
        import tempfile

        with tempfile.TemporaryDirectory(prefix="gauge-self-") as tmp:
            copy = Path(tmp) / "repo"
            shutil.copytree(self.root, copy, ignore=shutil.ignore_patterns("data", ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                                                                            ".ruff_cache", "*.egg-info", "node_modules"))
            for rel, content in prop.files.items():
                fp = copy / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
            env = dict(os.environ)
            env["GAUGE_OFFLINE"] = "1"
            env["PYTHONPATH"] = str(copy)
            report = []
            ok = True
            try:
                lint = subprocess.run([sys.executable, "-m", "ruff", "check", "gauge", "tests"], cwd=str(copy), capture_output=True, text=True,
                                      timeout=120, env=env, check=False)
                if lint.returncode not in (0,) and "No module named ruff" not in lint.stderr:
                    ok = False
                    report.append("ruff:\n" + (lint.stdout + lint.stderr)[-3000:])
                tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=str(copy), capture_output=True,
                                       text=True, timeout=self.test_timeout, env=env, check=False)
                if tests.returncode != 0:
                    ok = False
                report.append("pytest:\n" + (tests.stdout + tests.stderr)[-4000:])
            except subprocess.TimeoutExpired:
                ok = False
                report.append("tests timed out")
        prop.tested = ok
        prop.test_report = "\n".join(report)[-6000:]
        self.save(prop)
        return ok, prop.test_report

    # ---- applying and rolling back ------------------------------------------------------------------
    def apply(self, prop: Proposal, commit: bool = True) -> dict[str, Any]:
        if not prop.tested:
            raise PermissionError("proposal has not passed tests; run `gauge self test` first")
        if prop.applied:
            return {"applied": False, "reason": "already applied"}
        backup = self.dir / "backups" / prop.id
        backup.mkdir(parents=True, exist_ok=True)
        changed = []
        for rel, content in prop.files.items():
            why = self.check_path(rel)
            if why:
                raise PermissionError(why)
            target = self.root / rel
            if target.exists():
                bp = backup / rel
                bp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, bp)
            else:
                (backup / (rel + ".ABSENT")).parent.mkdir(parents=True, exist_ok=True)
                (backup / (rel + ".ABSENT")).write_text("", encoding="utf-8")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            changed.append(rel)
        prop.applied = True
        self.save(prop)
        committed = False
        if commit and (self.root / ".git").exists() and shutil.which("git"):
            try:
                subprocess.run(["git", "add", *changed], cwd=str(self.root), check=True, capture_output=True, timeout=60)
                msg = f"self: {prop.instruction[:60]}\n\n{prop.rationale}\n\nProposal {prop.id}; tests passed in isolation."
                subprocess.run(["git", "-c", "user.name=Gauge", "-c", "user.email=noreply@users.noreply.github.com", "commit", "-q", "-m", msg],
                               cwd=str(self.root), check=True, capture_output=True, timeout=60)
                committed = True
            except (subprocess.SubprocessError, OSError) as e:
                log.warning("self: git commit failed: %s", e)
        return {"applied": True, "files": changed, "backup": str(backup), "committed": committed}

    def rollback(self, pid: str) -> dict[str, Any]:
        backup = self.dir / "backups" / pid
        if not backup.exists():
            raise FileNotFoundError(f"no backup for proposal {pid}")
        restored, removed = [], []
        for bp in backup.rglob("*"):
            if not bp.is_file():
                continue
            rel = bp.relative_to(backup).as_posix()
            if rel.endswith(".ABSENT"):
                target = self.root / rel[: -len(".ABSENT")]
                if target.exists():
                    target.unlink()
                    removed.append(rel[: -len(".ABSENT")])
                continue
            shutil.copy2(bp, self.root / rel)
            restored.append(rel)
        try:
            prop = self.load(pid)
            prop.applied = False
            self.save(prop)
        except FileNotFoundError:
            pass
        return {"restored": restored, "removed": removed}

    # ---- self-healing from the journal ------------------------------------------------------------
    def heal(self, entries: list[dict[str, Any]], auto_apply: bool = False) -> list[dict[str, Any]]:
        """Turn recurring errors into tested proposals; apply only when told to."""
        out = []
        seen: set[str] = set()
        for e in entries:
            key = str(e.get("message", ""))[:80]
            if key in seen or not key:
                continue
            seen.add(key)
            instr = f"Fix this recurring error in Gauge: {key}"
            prop = self.propose(instr, source="journal", context=str(e.get("traceback", ""))[:3000])
            ok, _ = self.test(prop) if prop.files else (False, "")
            res = {"proposal": prop.id, "files": list(prop.files), "tested": ok, "applied": False}
            if ok and auto_apply:
                res.update(self.apply(prop))
            out.append(res)
        return out
