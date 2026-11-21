"""Mutation testing: prove the tests actually cover a self-edit before trusting it.

A self-written change that passes the suite might still be untested — the suite may not exercise the
new code at all. Mutation testing checks that: it makes small, deliberate breakages (a `==` becomes
`!=`, a `+` becomes a `-`, a boundary flips) and confirms the tests then *fail*. A mutant the tests
still pass ("survives") is a hole in coverage. The mutation score is the fraction of mutants killed;
a low score on a changed file means the self-programmer wrote code the tests don't guard, which is
exactly when a human should look before it lands.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

# Textual mutation operators: (compiled pattern, replacement, label). Conservative and syntax-preserving.
_OPERATORS: list[tuple[str, str, str]] = [
    (r"==", "!=", "== -> !="),
    (r"!=", "==", "!= -> =="),
    (r"<=", ">", "<= -> >"),
    (r">=", "<", ">= -> <"),
    (r"(?<![<>=!])<(?![=])", ">", "< -> >"),
    (r"(?<![<>=!])>(?![=])", "<", "> -> <"),
    (r"\bTrue\b", "False", "True -> False"),
    (r"\bFalse\b", "True", "False -> True"),
    (r"\band\b", "or", "and -> or"),
    (r"\bor\b", "and", "or -> and"),
    (r"(?<![+\-*/]) \+ ", " - ", "+ -> -"),
    (r"(?<![+\-*/]) - ", " + ", "- -> +"),
]


@dataclass
class Mutant:
    label: str
    line: int
    source: str


def _valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def generate_mutants(source: str, max_mutants: int = 12, python: bool = True) -> list[Mutant]:
    """One mutation per mutant, at distinct sites across operators. Mutants that break syntax are skipped."""
    mutants: list[Mutant] = []
    seen: set[tuple[int, str]] = set()
    for pattern, repl, label in _OPERATORS:
        for m in re.finditer(pattern, source):
            start = m.start()
            line = source.count("\n", 0, start) + 1
            key = (start, label)
            if key in seen:
                continue
            seen.add(key)
            mutated = source[:start] + re.sub(pattern, repl, source[start:m.end()], count=1) + source[m.end():]
            if mutated == source:
                continue
            if python and not _valid_python(mutated):
                continue
            mutants.append(Mutant(label, line, mutated))
            if len(mutants) >= max_mutants:
                return mutants
    return mutants


@dataclass
class MutationResult:
    total: int
    killed: int
    survivors: list[Mutant]

    @property
    def score(self) -> float:
        return round(self.killed / self.total, 3) if self.total else 1.0


def mutation_score(source: str, tests_pass: Callable[[str], bool], max_mutants: int = 12,
                   python: bool = True) -> MutationResult:
    """`tests_pass(mutated_source)` returns True when the suite still passes on the mutated file.
    A mutant is *killed* when the tests fail on it (tests_pass returns False)."""
    mutants = generate_mutants(source, max_mutants=max_mutants, python=python)
    killed = 0
    survivors: list[Mutant] = []
    for mut in mutants:
        try:
            still_passes = tests_pass(mut.source)
        except Exception:
            still_passes = False  # a crash counts as killed: the mutation was detected
        if still_passes:
            survivors.append(mut)
        else:
            killed += 1
    return MutationResult(len(mutants), killed, survivors)


def report(result: MutationResult, min_score: float = 0.5) -> Optional[str]:
    """A warning string when the file is under-tested, else None."""
    if result.total == 0:
        return None
    if result.score < min_score:
        surv = "; ".join(f"line {m.line}: {m.label}" for m in result.survivors[:5])
        return (f"mutation score {result.score:.0%} ({result.killed}/{result.total} killed) is below {min_score:.0%}; "
                f"the tests do not catch: {surv}")
    return None
