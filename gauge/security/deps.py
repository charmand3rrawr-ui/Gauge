"""Supply-chain scan for code Gauge writes: what does this bot import, and should it?

The code scanner catches dangerous constructs; this catches dangerous *dependencies* — a generated
bot that quietly imports a package no one vetted, or one whose name is a typo of a popular package
(the classic typosquat attack, where `reqeusts` installs malware). It extracts the imports, sorts
them into standard library (safe), the packages Gauge itself is built on (allowed), unknown
third-party (worth a look), and typosquat look-alikes (block). It reads the code; it never installs
anything.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass

# Packages Gauge is built on or safely ships with; a generated bot may use these without a flag.
ALLOWED = {"requests", "bs4", "beautifulsoup4", "lxml", "numpy", "click", "pydantic", "rich", "sklearn",
           "scikit_learn", "cupy", "gauge", "pytest", "ruff", "yaml", "dateutil"}

# Popular PyPI names used for typosquat detection (a near-miss of one of these is suspicious).
POPULAR = {"requests", "numpy", "pandas", "scipy", "flask", "django", "setuptools", "pytest", "pillow",
           "urllib3", "click", "pydantic", "rich", "lxml", "cryptography", "boto3", "tensorflow", "torch",
           "matplotlib", "sqlalchemy", "beautifulsoup4", "selenium", "scikit", "aiohttp", "httpx"}


@dataclass
class DepFinding:
    module: str
    kind: str   # stdlib | allowed | unknown | typosquat
    severity: int  # 1 note, 2 warn, 3 block
    detail: str


def _edit_distance(a: str, b: str) -> int:
    """Optimal string alignment (restricted Damerau-Levenshtein): an adjacent transposition counts as
    one edit, so the most common typo — swapping two letters, as in reqeusts/requests — is caught."""
    if abs(len(a) - len(b)) > 2:
        return 3
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _stdlib() -> set[str]:
    names = getattr(sys, "stdlib_module_names", None)
    return set(names) if names else {"os", "sys", "re", "json", "math", "time", "io", "struct", "dis",
                                     "pathlib", "typing", "dataclasses", "collections", "itertools", "functools",
                                     "hashlib", "random", "subprocess", "socket", "ast"}


def top_level_imports(code: str) -> set[str]:
    """The top-level package names imported by the code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                mods.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    return mods


def scan_imports(code: str) -> list[DepFinding]:
    std = _stdlib()
    out: list[DepFinding] = []
    for mod in sorted(top_level_imports(code)):
        low = mod.lower()
        if low in std:
            out.append(DepFinding(mod, "stdlib", 1, "standard library"))
        elif low in ALLOWED:
            out.append(DepFinding(mod, "allowed", 1, "vetted dependency"))
        else:
            near = [p for p in POPULAR if p != low and _edit_distance(low, p) <= 1]
            if near:
                out.append(DepFinding(mod, "typosquat", 3, f"looks like a typo of {', '.join(near)}; likely a malicious package"))
            else:
                out.append(DepFinding(mod, "unknown", 2, "unvetted third-party import; confirm it is real and needed"))
    return out


def worst(findings: list[DepFinding]) -> int:
    return max((f.severity for f in findings), default=0)


def report(findings: list[DepFinding]) -> str:
    names = {1: "ok", 2: "WARN", 3: "BLOCK"}
    flagged = [f for f in findings if f.severity >= 2]
    if not flagged:
        return "dependencies: all standard-library or vetted"
    return "\n".join(f"[{names[f.severity]}] {f.module}: {f.detail}" for f in sorted(flagged, key=lambda f: -f.severity))
