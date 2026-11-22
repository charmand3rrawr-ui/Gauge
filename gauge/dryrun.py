"""A global dry-run switch: one flag that holds every outward-facing or hard-to-undo action (roadmap 97).

``gauge --dry-run <anything>`` (or ``GAUGE_DRY_RUN=1``, or ``dry_run: true`` in config) makes a first run of
any automation safe. The switch is enforced at the lowest-level effect points, not in each command, so no
command path can bypass it:

* social posting            (market.posters)      -> previewed, nothing sent
* newsletter sending        (market.email_list)   -> previewed, nothing sent
* newsdesk publishing       (newsdesk.desk)       -> the due piece stays in the queue
* self-modification         (code.selfmod.apply)  -> the proposal stays unapplied
* restoring a backup        (ops.backup)          -> verified, nothing written
* daemon chapter writes and scheduled bots (daemon) -> skipped

What still runs: reading, analysis, training, drafting into local queues, and anything the author runs
explicitly that only writes inside the data directory. Each held action is recorded so the CLI can say
exactly what a real run would have done.

The switch lives in the environment so worker and bot subprocesses inherit it.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

ENV = "GAUGE_DRY_RUN"
_TRUE = {"1", "true", "yes", "on"}
_held: list[dict[str, str]] = []

HELD_ACTIONS = ("social posting", "newsletter sending", "newsdesk publishing", "self-modification",
                "backup restore", "daemon chapter writes", "scheduled bots")


def is_dry_run() -> bool:
    return os.environ.get(ENV, "").strip().lower() in _TRUE


def set_dry_run(on: bool) -> None:
    if on:
        os.environ[ENV] = "1"
    else:
        os.environ.pop(ENV, None)


def hold(action: str, detail: str = "") -> dict[str, str]:
    """Record an action the dry run held back. Returns the record."""
    rec = {"action": action, "detail": detail}
    _held.append(rec)
    log.info("dry run: held %s%s", action, f" ({detail})" if detail else "")
    return rec


def held() -> list[dict[str, str]]:
    return list(_held)


def reset() -> None:
    _held.clear()


def summary() -> str:
    """What this run held back, for the end of a dry run."""
    if not _held:
        return "dry run: nothing needed holding back; a real run would have done the same."
    lines = [f"dry run: held {len(_held)} action(s) a real run would have taken:"]
    lines += [f"  - {h['action']}" + (f": {h['detail']}" if h["detail"] else "") for h in _held[:40]]
    return "\n".join(lines)
