"""Graduated active defence — all of it on Gauge's own machine.

A single dial, like the rebel-craft dial, that decides how hard Gauge pushes back as an attack
intensifies. It never reaches into anyone else's system. Every action here is inward:

* **tarpit** - slow Gauge's own responses to a suspicious peer, so a prober wastes its time;
* **honeytokens** - plant fake secrets; if one ever turns up in traffic or a file read, that is proof
  of exfiltration, and it raises an alert;
* **decoys** - hand a rejected prober a plausible-but-fake response instead of anything real;
* **evidence** - capture who hit Gauge and how into a durable bundle for the owner or a responder.

The level can be set by hand or escalated automatically from the Sentinel's threat level. There is no
outward offence in here by design: counter-pressure an attacker feels, nothing that touches them.
"""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..util import utcnow

# Per-level behaviour. Index = active-defence level 0..3.
TARPIT_SECONDS = [0.0, 0.5, 2.0, 5.0]
HONEYTOKEN_MARKER = "GAUGE-CANARY-"


@dataclass
class DefensePosture:
    level: int
    tarpit_seconds: float
    honeytokens: bool
    decoys: bool
    evidence: bool
    note: str = ""


def level_for_threat(threat_level: int, floor: int = 0) -> int:
    """Map the Sentinel's 0..3 threat level to an active-defence level, never below `floor`."""
    return max(floor, min(3, int(threat_level)))


def posture(level: int) -> DefensePosture:
    lvl = max(0, min(3, int(level)))
    names = ["passive (log only)", "watch (tarpit + evidence)", "elevated (+ honeytokens + decoys)", "high (maximum delay, all on)"]
    return DefensePosture(lvl, TARPIT_SECONDS[lvl], honeytokens=lvl >= 2, decoys=lvl >= 2, evidence=lvl >= 1, note=names[lvl])


class ActiveDefense:
    def __init__(self, security_dir: Path, level: int = 0, audit=None):
        self.dir = Path(security_dir)
        self.level = max(0, min(3, int(level)))
        self.audit = audit
        self.posture = posture(self.level)

    def _audit(self, kind: str, detail: str, sev: int, data: Optional[dict] = None) -> None:
        if self.audit is not None:
            try:
                self.audit.event(kind, detail, sev, data)
            except Exception:
                pass

    # ---- tarpit ------------------------------------------------------------
    def tarpit_seconds(self) -> float:
        return self.posture.tarpit_seconds

    # ---- honeytokens -------------------------------------------------------
    def _token_path(self) -> Path:
        return self.dir / "honeytokens.json"

    def plant_honeytoken(self, label: str = "api-key") -> str:
        """Create and remember a fake secret. If it ever appears in traffic, it was exfiltrated."""
        token = HONEYTOKEN_MARKER + secrets.token_hex(16)
        self.dir.mkdir(parents=True, exist_ok=True)
        tokens = self.honeytokens()
        tokens.append({"label": label, "token": token, "planted": utcnow().isoformat(timespec="seconds")})
        self._token_path().write_text(json.dumps(tokens, indent=2), encoding="utf-8")
        self._audit("honeytoken_planted", f"planted honeytoken for {label}", 1)
        return token

    def honeytokens(self) -> list[dict[str, Any]]:
        p = self._token_path()
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []

    def scan_for_leak(self, observed: str, where: str = "outbound") -> list[str]:
        """Return any planted tokens that appear in `observed` (e.g. an outbound request body). Alerts."""
        leaked = [t["token"] for t in self.honeytokens() if t["token"] in (observed or "")]
        if leaked:
            self._audit("honeytoken_tripped", f"{len(leaked)} honeytoken(s) seen in {where} — possible exfiltration", 3,
                        {"where": where, "count": len(leaked)})
        return leaked

    # ---- decoys ------------------------------------------------------------
    def decoy_response(self) -> dict[str, Any]:
        """A plausible-but-fake worker response for a rejected prober. Reveals nothing real; logs the probe."""
        self._audit("decoy_served", "served a decoy response to a rejected peer", 2)
        return {"status": "ok", "model": "gauge-worker", "result": "", "note": "queued"}

    # ---- evidence ----------------------------------------------------------
    def capture(self, kind: str, peer: str, payload: Any, meta: Optional[dict] = None) -> Optional[Path]:
        """Write a durable evidence bundle for the owner / a responder. On-disk only; no outbound action."""
        if not self.posture.evidence:
            return None
        ev_dir = self.dir / "evidence"
        ev_dir.mkdir(parents=True, exist_ok=True)
        stamp = utcnow().strftime("%Y%m%dT%H%M%S%f")
        row = {"at": utcnow().isoformat(timespec="seconds"), "kind": kind, "peer": peer,
               "payload": str(payload)[:4000], "meta": meta or {}}
        path = ev_dir / f"{stamp}-{kind}.json"
        path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        self._audit("evidence_captured", f"captured {kind} from {peer}", 2, {"file": path.name})
        return path

    def evidence(self, limit: int = 50) -> list[dict[str, Any]]:
        ev_dir = self.dir / "evidence"
        if not ev_dir.exists():
            return []
        out = []
        for p in sorted(ev_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def abuse_report_draft(self, peer: str) -> str:
        """Draft (not send) an abuse report the owner can choose to send. Reporting is the legal response."""
        rows = [e for e in self.evidence(200) if e.get("peer") == peer]
        lines = [f"Subject: Abuse report for traffic from {peer}", "",
                 f"The host {peer} generated the following activity against a private service:"]
        for e in rows[:20]:
            lines.append(f"  {e['at']}  {e['kind']}: {e['payload'][:120]}")
        lines += ["", "No action was taken against the source; this is a report for the responsible network operator.",
                  "Generated by Gauge's defensive evidence log."]
        return "\n".join(lines)
