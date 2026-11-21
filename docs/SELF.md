# Self-modification, with guardrails

Gauge can change its own source. It does so the way a careful maintainer would: as a reviewable
proposal, tested in isolation, applied only when the whole suite passes, with a backup and a
rollback, and with a human approving by default.

## The flow

```powershell
.\gauge.bat self propose "make `gauge novels` show the chapter count in bold"
.\gauge.bat self test 20261121-153010-make-gauge-novels-show
.\gauge.bat self apply 20261121-153010-make-gauge-novels-show
.\gauge.bat self rollback 20261121-153010-make-gauge-novels-show   # if you change your mind
```

- **Propose.** Gauge indexes its own source (module docstrings and signatures), picks the files
  the instruction concerns (or the ones named in an error traceback), and returns complete new
  contents for at most three files with a rationale. Nothing in the working tree changes. The
  proposal, a unified diff and the rationale are saved under `data/self/proposals/<id>/`.
- **Test.** The repository is copied to a temporary directory (without data, git or virtual
  environments), the proposal is applied there, and `ruff` plus the entire test suite run with
  the network disabled. The real tree is never touched by a test.
- **Apply.** Only a proposal that passed may be applied. Originals are backed up under
  `data/self/backups/<id>/`, the files are written, and a git commit is made when the tree is a
  repository. `rollback` restores the backup.

## Guardrails that hold even in automatic mode

- Protected paths cannot be changed: the installers, `pyproject.toml`, `LICENSE`, `.github/`,
  `scripts/`, `requirements.txt`, and the self-modification module itself.
- Only `.py` and `.md` files may change; at most three per proposal.
- Existing tests may not be removed or skipped; test files may only grow.
- A rewritten file may not balloon or shrink beyond sane bounds, and must compile.
- Every proposal is tested against the full suite before it can be applied.

## Self-healing

The daemon and the writer record exceptions in a journal (`gauge self journal`). When an error
has recurred `code.min_recurrences` times, `gauge self heal` turns it into a proposal with the
traceback as context and tests it. With `code.self_heal: true` the daemon does this on its own,
and with `code.auto_apply: true` it also applies proposals that pass. Both are **off by default**.
Turn `auto_apply` on only if you are comfortable with Gauge committing to its own code while
you are not watching; the guardrails above still apply, and every applied change can be rolled
back.

## What this is not

It is not an unbounded agent rewriting itself at will. The model proposes; code decides what is
allowed; tests decide what is correct; you decide what is applied.
