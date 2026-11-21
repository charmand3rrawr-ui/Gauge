"""Static scan of code before Gauge runs or installs it: generated bots, self-modification proposals, imported scripts.

Findings have a severity: 1 note, 2 warn, 3 block. Blocking findings stop a bot from being saved as
runnable and stop a proposal from being tested. The patterns are deliberately simple and readable;
this is a tripwire against the obvious, not a substitute for reading the code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PY_RULES: list[tuple[str, str, int]] = [
    (r"\b(eval|exec)\s*\(", "dynamic code execution", 3),
    (r"subprocess\.[a-zA-Z_]+\([^)]*shell\s*=\s*True", "shell=True subprocess", 3),
    (r"\b[\w.]+\.(run|Popen|call|check_output|check_call)\s*\([^)]*shell\s*=\s*True", "shell=True subprocess (aliased)", 3),
    (r"import\s+subprocess\s+as\s+\w+", "aliased subprocess import", 1),
    (r"\bos\.system\s*\(", "os.system", 3),
    (r"\b(shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir)\s*\(", "deletes files", 2),
    (r"\b(base64\.b64decode|codecs\.decode\([^)]*rot)", "decodes hidden payload", 3),
    (r"\bimport\s+(socket|ctypes|winreg|_winreg|msvcrt|pty)\b|\bfrom\s+(socket|ctypes|winreg)\s+import", "low-level system access", 2),
    (r"\b(urllib\.request|requests|http\.client|httpx|aiohttp)\b", "network access", 1),
    (r"(urlopen|requests\.(get|post|put|patch)|DownloadString|\.send)\s*\([^)]{0,160}os\.environ", "sends environment/secret over the network", 3),
    (r"(ssh|\.aws|\.ssh|credentials|id_rsa|\.netrc|wallet|keychain|Local State|Login Data)", "touches credentials", 3),
    (r"\b(HKEY_|reg add|reg delete|schtasks|sc create|netsh|bcdedit|vssadmin)\b", "system configuration change", 3),
    (r"\bopen\s*\([^)]*['\"][a-zA-Z]:\\\\Windows", "writes under Windows dir", 3),
    (r"\bpickle\.loads?\s*\(|allow_pickle\s*=\s*True", "unpickling", 3),
    (r"\b__import__\s*\(|importlib\.import_module\s*\(\s*[^'\"]", "dynamic import", 2),
    (r"\bgetattr\s*\(\s*(os|subprocess|builtins)\b", "obfuscated system call", 3),
    (r"\\x[0-9a-fA-F]{2}\\x[0-9a-fA-F]{2}\\x[0-9a-fA-F]{2}", "hex-encoded strings", 2),
    (r"GAUGE_NO_LLM|os\.environ\.(pop|__delitem__)|del os\.environ", "tampers with bot sandbox env", 3),
    (r"(token|secret|password|api[_-]?key)\s*=\s*['\"][^'\"]{8,}", "hard-coded secret", 2),
]
PS_RULES: list[tuple[str, str, int]] = [
    (r"Invoke-Expression|\biex\b|\bIEX\b", "Invoke-Expression", 3),
    (r"-EncodedCommand|-enc\s|FromBase64String", "encoded command", 3),
    (r"Invoke-WebRequest|Invoke-RestMethod|Net\.WebClient|DownloadString|DownloadFile", "network download", 2),
    (r"Remove-Item[^\n]*-Recurse", "recursive delete", 2),
    (r"Set-ItemProperty[^\n]*HKLM|New-ItemProperty[^\n]*Run\b|reg add", "registry persistence", 3),
    (r"Set-MpPreference|Disable|Add-MpPreference -Exclusion", "disables protections", 3),
    (r"Start-Process[^\n]*-Verb RunAs|-ExecutionPolicy Unrestricted", "privilege escalation", 3),
    (r"schtasks|Register-ScheduledTask", "scheduled task", 2),
]
BAT_RULES: list[tuple[str, str, int]] = [
    (r"\b(del|erase|rd|rmdir)\b[^\n]*(/s|/q)", "recursive delete", 2),
    (r"\bformat\s+[a-z]:", "formats a drive", 3),
    (r"\breg\s+(add|delete)\b|\bschtasks\b|\bnetsh\b|\bbcdedit\b|\bvssadmin\b", "system configuration change", 3),
    (r"powershell[^\n]*(-enc|-EncodedCommand|iex|Invoke-Expression)", "encoded powershell", 3),
    (r"\bcertutil\b[^\n]*(-decode|-urlcache)|\bbitsadmin\b", "download/decode utility", 3),
]
JS_RULES: list[tuple[str, str, int]] = [
    (r"\beval\s*\(|new Function\s*\(", "dynamic code execution", 3),
    (r"child_process|require\(['\"]fs['\"]\)[^\n]*(rm|unlink)", "process/file system", 2),
    (r"document\.cookie|localStorage\.getItem\(['\"](token|auth|session)", "reads credentials", 3),
    (r"\.innerHTML\s*=", "innerHTML assignment", 1),
    (r"fetch\(['\"]https?://(?!localhost|127\.0\.0\.1)", "external network", 1),
]


@dataclass
class Finding:
    rule: str
    severity: int
    line: int
    snippet: str


def _rules_for(filename: str) -> list[tuple[str, str, int]]:
    low = filename.lower()
    if low.endswith(".py"):
        return PY_RULES
    if low.endswith(".ps1"):
        return PS_RULES
    if low.endswith((".bat", ".cmd")):
        return BAT_RULES
    if low.endswith((".js", ".mjs")):
        return JS_RULES
    return []


def scan_text(filename: str, code: str) -> list[Finding]:
    out = []
    for pattern, label, sev in _rules_for(filename):
        rx = re.compile(pattern)
        for i, line in enumerate(code.splitlines(), 1):
            if rx.search(line):
                out.append(Finding(label, sev, i, line.strip()[:120]))
    return out


def scan_files(files: dict[str, str]) -> list[tuple[str, Finding]]:
    out = []
    for name, code in files.items():
        for f in scan_text(name, code):
            out.append((name, f))
    return out


def worst(findings: list[tuple[str, Finding]] | list[Finding]) -> int:
    sev = 0
    for item in findings:
        f = item[1] if isinstance(item, tuple) else item
        sev = max(sev, f.severity)
    return sev


def report(findings: list[tuple[str, Finding]]) -> str:
    if not findings:
        return "no findings"
    names = {1: "note", 2: "warn", 3: "BLOCK"}
    return "\n".join(f"[{names[f.severity]}] {name}:{f.line} {f.rule}: {f.snippet}" for name, f in sorted(findings, key=lambda t: -t[1].severity))
