"""Many languages, read fluently — not every language spoken fluently.

Gauge's bots and self-edits are Python, but it reads and reasons about code in many languages: to
understand a dependency, port an idiom, or review a snippet. This module is the honest version of
"knows every programming language": a registry of languages with their idioms and gotchas, plus a
lexical analyser built on Pygments that detects the language, measures its shape (comment ratio,
functions, classes, nesting), and flags obvious smells. It does not claim expert authorship in
hundreds of languages; it claims literacy — the ability to take code apart and talk about it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# A compact, honest knowledge base. `idioms` are things to prefer; `gotchas` are things that bite.
LANGUAGES: dict[str, dict] = {
    "python": {"paradigm": "multi (OO, functional, imperative)", "comment": "#", "ext": [".py", ".pyw"],
               "idioms": ["comprehensions over manual loops", "context managers for resources", "f-strings", "dataclasses"],
               "gotchas": ["mutable default arguments", "late-binding closures", "== vs is"]},
    "javascript": {"paradigm": "multi, prototype-based", "comment": "//", "ext": [".js", ".mjs", ".cjs"],
                   "idioms": ["const/let over var", "async/await", "optional chaining ?."],
                   "gotchas": ["== coercion (use ===)", "this binding", "floating-point money"]},
    "typescript": {"paradigm": "multi, typed JS", "comment": "//", "ext": [".ts", ".tsx"],
                   "idioms": ["narrow types, avoid any", "discriminated unions", "readonly"],
                   "gotchas": ["types erased at runtime", "structural typing surprises"]},
    "rust": {"paradigm": "systems, ownership", "comment": "//", "ext": [".rs"],
             "idioms": ["Result/Option over exceptions", "borrow, don't clone", "iterators"],
             "gotchas": ["the borrow checker", "lifetimes", "move semantics"]},
    "go": {"paradigm": "imperative, concurrent", "comment": "//", "ext": [".go"],
           "idioms": ["return errors explicitly", "goroutines + channels", "small interfaces"],
           "gotchas": ["nil interface != nil", "loop variable capture", "unused imports error"]},
    "c": {"paradigm": "procedural, manual memory", "comment": "//", "ext": [".c", ".h"],
          "idioms": ["check every malloc", "free what you own", "const correctness"],
          "gotchas": ["buffer overflows", "use-after-free", "undefined behaviour"]},
    "cpp": {"paradigm": "multi, RAII", "comment": "//", "ext": [".cpp", ".cc", ".hpp", ".cxx"],
            "idioms": ["RAII / smart pointers", "the rule of five", "std algorithms"],
            "gotchas": ["dangling references", "object slicing", "iterator invalidation"]},
    "java": {"paradigm": "OO", "comment": "//", "ext": [".java"],
             "idioms": ["program to interfaces", "try-with-resources", "Optional over null"],
             "gotchas": ["NullPointerException", "equals/hashCode pairing", "checked exceptions"]},
    "csharp": {"paradigm": "OO, multi", "comment": "//", "ext": [".cs"],
               "idioms": ["LINQ", "using/IDisposable", "async/await"],
               "gotchas": ["null refs (use nullable refs)", "struct vs class copy"]},
    "ruby": {"paradigm": "OO, dynamic", "comment": "#", "ext": [".rb"],
             "idioms": ["blocks and yield", "everything is an object", "duck typing"],
             "gotchas": ["monkey-patching surprises", "nil vs false"]},
    "powershell": {"paradigm": "shell, OO pipeline", "comment": "#", "ext": [".ps1", ".psm1"],
                   "idioms": ["objects down the pipeline", "approved verbs", "[CmdletBinding()]"],
                   "gotchas": ["execution policy", "$null comparisons on the left"]},
    "bash": {"paradigm": "shell", "comment": "#", "ext": [".sh", ".bash"],
             "idioms": ["quote your variables", "set -euo pipefail", "[[ ]] over [ ]"],
             "gotchas": ["word splitting", "unquoted globs", "subshell variable scope"]},
    "sql": {"paradigm": "declarative, set-based", "comment": "--", "ext": [".sql"],
            "idioms": ["think in sets not rows", "parameterise queries", "index the join keys"],
            "gotchas": ["SQL injection from string-building", "NULL three-valued logic"]},
}

ALIASES = {"js": "javascript", "ts": "typescript", "c++": "cpp", "cs": "csharp", "c#": "csharp", "py": "python", "shell": "bash", "ps1": "powershell"}


def normalize(name: str) -> str:
    n = (name or "").strip().lower()
    return ALIASES.get(n, n)


def known(language: str) -> bool:
    return normalize(language) in LANGUAGES


def idioms(language: str) -> list[str]:
    return LANGUAGES.get(normalize(language), {}).get("idioms", [])


def gotchas(language: str) -> list[str]:
    return LANGUAGES.get(normalize(language), {}).get("gotchas", [])


def tips(language: str) -> str:
    lang = normalize(language)
    info = LANGUAGES.get(lang)
    if not info:
        return f"{language}: not in the knowledge base; read the code structurally"
    return (f"{lang} ({info['paradigm']}). Prefer: {', '.join(info['idioms'])}. "
            f"Watch out for: {', '.join(info['gotchas'])}.")


def detect(filename: str, code: str = "") -> Optional[str]:
    """Best-effort language id from the filename, then the content."""
    ext = re.search(r"(\.[A-Za-z0-9+#]+)$", filename or "")
    if ext:
        e = ext.group(1).lower()
        for name, info in LANGUAGES.items():
            if e in info["ext"]:
                return name
    try:
        from pygments.lexers import guess_lexer, guess_lexer_for_filename
        try:
            lx = guess_lexer_for_filename(filename, code) if filename else guess_lexer(code)
        except Exception:
            lx = guess_lexer(code)
        return normalize(lx.aliases[0] if lx.aliases else lx.name)
    except Exception:
        return None


@dataclass
class CodeShape:
    language: Optional[str]
    lines: int
    tokens: int
    comment_ratio: float
    functions: int
    classes: int
    max_indent: int
    smells: list[str] = field(default_factory=list)


_FUNC_PATTERNS = {
    "python": r"^\s*(?:async\s+)?def\s+\w+", "ruby": r"^\s*def\s+\w+",
    "javascript": r"\bfunction\b|\b\w+\s*=\s*\([^)]*\)\s*=>", "typescript": r"\bfunction\b|=>",
    "go": r"^\s*func\s+", "rust": r"^\s*(?:pub\s+)?fn\s+", "c": r"^\w[\w\s\*]*\([^;]*\)\s*\{",
    "cpp": r"^\w[\w\s\*:<>]*\([^;]*\)\s*\{", "java": r"\b(?:public|private|protected|static|final|\s)+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{",
    "csharp": r"\b(?:public|private|protected|internal|static|\s)+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{",
    "powershell": r"^\s*function\s+\w+", "bash": r"^\s*(?:function\s+)?\w+\s*\(\)\s*\{",
}
_CLASS_PATTERNS = {
    "python": r"^\s*class\s+\w+", "ruby": r"^\s*class\s+\w+", "java": r"\b(?:class|interface|enum)\s+\w+",
    "csharp": r"\b(?:class|interface|struct|enum)\s+\w+", "cpp": r"\b(?:class|struct)\s+\w+",
    "rust": r"\b(?:struct|enum|trait|impl)\s+\w+", "go": r"\btype\s+\w+\s+struct", "typescript": r"\b(?:class|interface)\s+\w+",
    "javascript": r"\bclass\s+\w+",
}


def analyze(filename: str, code: str) -> CodeShape:
    lang = detect(filename, code)
    lines = code.splitlines()
    n_lines = len(lines)
    comment = LANGUAGES.get(lang or "", {}).get("comment", "#")
    comment_lines = sum(1 for ln in lines if ln.strip().startswith(comment))
    max_indent = 0
    for ln in lines:
        if ln.strip():
            indent = len(ln) - len(ln.lstrip(" "))
            max_indent = max(max_indent, indent // 4 + ln[:indent].count("\t"))
    funcs = len(re.findall(_FUNC_PATTERNS.get(lang or "", r"$^"), code, re.M)) if lang in _FUNC_PATTERNS else 0
    classes = len(re.findall(_CLASS_PATTERNS.get(lang or "", r"$^"), code, re.M)) if lang in _CLASS_PATTERNS else 0
    tokens = _count_tokens(filename, code, lang)
    smells: list[str] = []
    if n_lines and comment_lines / n_lines < 0.02 and n_lines > 40:
        smells.append("almost no comments in a long file")
    if max_indent >= 6:
        smells.append(f"deep nesting ({max_indent} levels); consider extracting functions")
    long_lines = sum(1 for ln in lines if len(ln) > 120)
    if long_lines > max(3, n_lines * 0.1):
        smells.append(f"{long_lines} very long lines")
    return CodeShape(lang, n_lines, tokens, round(comment_lines / n_lines, 3) if n_lines else 0.0,
                     funcs, classes, max_indent, smells)


def _count_tokens(filename: str, code: str, lang: Optional[str]) -> int:
    try:
        from pygments import lex
        from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
        try:
            lx = get_lexer_by_name(lang) if lang and known(lang) else guess_lexer_for_filename(filename, code)
        except Exception:
            return len(code.split())
        return sum(1 for _ in lex(code, lx))
    except Exception:
        return len(code.split())
