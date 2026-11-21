"""Defensive security for Gauge: untrusted text is data, generated code is scanned and sandboxed,
the source tree is sealed and verified, every security event is in a tamper-evident log, and a
sentinel can lock automation down."""
from .audit import AuditLog
from .codescan import Finding, report, scan_files, scan_text, worst
from .integrity import seal, verify
from .sandbox import minimal_env, run_isolated
from .sanitize import ScrubResult, fence, find_injections, scrub, strip_controls
from .sentinel import LOCKDOWN_SWITCHES, Threat, assess, lockdown, unlock

__all__ = ["AuditLog", "Finding", "report", "scan_files", "scan_text", "worst", "seal", "verify", "minimal_env", "run_isolated",
           "ScrubResult", "fence", "find_injections", "scrub", "strip_controls", "LOCKDOWN_SWITCHES", "Threat", "assess", "lockdown", "unlock"]
