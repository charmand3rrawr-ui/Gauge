"""A plain-language audit of every automation: what is on, what it can do, what it cannot.

An operator should be able to answer, in one command, "what can this thing do while I am not watching?"
This reads the config and reports each automation's state, its reach, and the guardrail that bounds it —
no jargon, no digging through switches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AutomationStatus:
    name: str
    on: bool
    does: str
    bound_by: str


def audit_report(cfg: Any) -> list[AutomationStatus]:
    c, nd, sec, srv, d = cfg.code, cfg.newsdesk, cfg.security, cfg.server, cfg.daemon
    rows = [
        AutomationStatus("daemon: write chapters", d.write_after_learning,
                         "writes the next chapter of the active project each learning cycle",
                         "local only; stops under security lockdown"),
        AutomationStatus("bots", c.bots_enabled,
                         "runs scheduled model-written programs; each is scanned and sandboxed",
                         "blocked bots never run; no network unless a bot's manifest asks; lockdown disables"),
        AutomationStatus("self-heal", c.self_heal,
                         "turns recurring errors into tested code-change proposals",
                         "proposals are tested in isolation and NOT applied unless auto-apply is on"),
        AutomationStatus("auto-apply self-fixes", c.auto_apply,
                         "applies passing proposals to Gauge's own code without asking",
                         "protected files are out of reach; rollback exists; never restored after lockdown"),
        AutomationStatus("newsdesk", nd.enabled,
                         "discovers trends and drafts original commentary into a review queue",
                         "drafts wait for approval; only clean pieces auto-publish; never rewrites others' articles"),
        AutomationStatus("newsdesk auto-publish", nd.auto_publish,
                         "publishes fully-clean commentary on a schedule without review",
                         "anything with a checkable factual claim still waits; lockdown disables; never auto-restored"),
        AutomationStatus("active defence", sec.active_defense_level > 0,
                         f"level {sec.active_defense_level}: tarpit/honeytokens/decoys/evidence on Gauge's own machine",
                         "inward only; never reaches another system"),
        AutomationStatus("control server", srv.enabled or False,
                         "exposes an HTTP control API (write chapters, run cycles)",
                         f"binds {srv.host}; token required on every call; non-loopback refused without a token"),
        AutomationStatus("research scout", cfg.research.enabled,
                         "fetches real-world detail from open sources for scenes and fact-checks",
                         "honours robots.txt; cached; lockdown disables"),
        AutomationStatus("worker (compute pool)", bool(cfg.compute.token),
                         "accepts jobs from other machines you own to share model calls / training",
                         "private-network peers only unless you opt in; token-gated; rate-limited"),
    ]
    return rows


def render_report(cfg: Any) -> str:
    from ..dryrun import HELD_ACTIONS, is_dry_run
    dry = is_dry_run() or bool(getattr(cfg, "dry_run", False))
    lines = ["Gauge automation audit", "", f"Security lockdown: {'ACTIVE' if cfg.security.locked else 'off'}",
             f"Dry run: {'ACTIVE: held = ' + ', '.join(HELD_ACTIONS) if dry else 'off (gauge --dry-run holds every outward or hard-to-undo action)'}",
             ""]
    for a in audit_report(cfg):
        lines.append(f"[{'ON ' if a.on else 'off'}] {a.name}")
        lines.append(f"      does: {a.does}")
        lines.append(f"      bound by: {a.bound_by}")
    on = sum(1 for a in audit_report(cfg) if a.on)
    lines += ["", f"{on} automation(s) active. Everything above is off by default and opt-in."]
    return "\n".join(lines)
