"""The Sentinel: watches the audit stream and the integrity check, scores the threat level, and
can lock Gauge down: automation that could do harm if driven by an attacker (self-heal, auto-apply,
bots, worker pool, research) is switched off until a person unlocks it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import GaugeConfig
from .audit import AuditLog

THRESHOLDS = {"auth_failure": 10, "injection": 25, "code_block": 3, "integrity": 1, "rate_limit": 20, "worker_rejected_peer": 5}


@dataclass
class Threat:
    level: int  # 0 calm, 1 watch, 2 elevated, 3 lockdown
    reasons: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def assess(audit: AuditLog, integrity_status: str = "ok", since_s: float = 86400.0) -> Threat:
    counts = audit.counts(since_s)
    reasons = []
    score = 0
    for kind, limit in THRESHOLDS.items():
        n = counts.get(kind, 0)
        if n >= limit:
            reasons.append(f"{kind}: {n} in the last {int(since_s // 3600)}h (limit {limit})")
            score += 2 if kind in ("integrity", "code_block") else 1
    if integrity_status in ("tampered", "corrupt"):
        reasons.append(f"integrity: {integrity_status}")
        score += 3
    level = 3 if score >= 3 else 2 if score == 2 else 1 if score == 1 else 0
    return Threat(level, reasons, counts)


LOCKDOWN_SWITCHES = {"code.self_heal": False, "code.auto_apply": False, "code.bots_enabled": False, "compute.pool_generation": False,
                     "compute.distributed_training": False, "research.enabled": False, "daemon.write_after_learning": False,
                     "newsdesk.enabled": False, "newsdesk.auto_publish": False}


def lockdown(cfg: GaugeConfig, audit: AuditLog, reason: str) -> dict[str, Any]:
    from .. import settings as st

    before = {k: st.get(cfg, k) for k in LOCKDOWN_SWITCHES}
    for k, v in LOCKDOWN_SWITCHES.items():
        st.set_value(cfg, k, v, save=False)
    cfg.security.locked = True
    cfg.security.locked_reason = reason[:300]
    cfg.security.locked_previous = {k: bool(v) for k, v in before.items()}
    cfg.save()
    audit.event("lockdown", reason, severity=3, data={"switched_off": [k for k, v in before.items() if v]})
    return {"locked": True, "reason": reason, "switched_off": [k for k, v in before.items() if v]}


def unlock(cfg: GaugeConfig, audit: AuditLog, restore: bool = True) -> dict[str, Any]:
    from .. import settings as st

    restored = []
    if restore:
        for k, v in (cfg.security.locked_previous or {}).items():
            if v and k in LOCKDOWN_SWITCHES and k not in ("code.auto_apply", "newsdesk.auto_publish"):  # unattended changes/publishing never come back automatically
                st.set_value(cfg, k, True, save=False)
                restored.append(k)
    cfg.security.locked = False
    cfg.security.locked_reason = ""
    cfg.security.locked_previous = {}
    cfg.save()
    audit.event("unlock", "unlocked by user", severity=2, data={"restored": restored})
    return {"locked": False, "restored": restored}
