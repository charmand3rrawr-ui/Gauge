"""Run generated or untrusted programs with the least they need.

What this does: a minimal environment with secrets stripped, GAUGE_NO_LLM set, GAUGE_NO_NETWORK
set unless the manifest allows network (and a dead HTTP proxy so library HTTP fails fast), a
working directory confined to the bot's folder, a wall-clock timeout, and on POSIX CPU/memory/
file-size limits via `resource`. What it cannot do: a Python process on Windows is not a jail;
there is no OS-level sandbox here. The code scanner and the test run are the other two layers.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

SECRET_KEYS = ("GAUGE_TOKEN", "OLLAMA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID",
               "GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "AZURE_OPENAI_KEY", "GOOGLE_API_KEY")
KEEP_PREFIXES = ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
                 "LANG", "LC_", "PYTHON", "VIRTUAL_ENV", "NUMBER_OF_PROCESSORS", "PROCESSOR_", "OS", "TZ", "GAUGE_DATA_DIR", "GAUGE_OFFLINE")


def minimal_env(extra: Optional[dict[str, str]] = None, allow_network: bool = False, allow_llm: bool = False) -> dict[str, str]:
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in SECRET_KEYS or "TOKEN" in k or "SECRET" in k or "PASSWORD" in k:
            continue
        if k.startswith(KEEP_PREFIXES):
            env[k] = v
    env["GAUGE_BOT"] = "1"
    if not allow_llm:
        env["GAUGE_NO_LLM"] = "1"
    if not allow_network:
        env["GAUGE_NO_NETWORK"] = "1"
        # Library HTTP (requests/urllib honour these) fails fast against a closed port.
        env["HTTP_PROXY"] = env["HTTPS_PROXY"] = env["http_proxy"] = env["https_proxy"] = "http://127.0.0.1:9"
        env["NO_PROXY"] = env["no_proxy"] = ""
    env.update(extra or {})
    return env


def _posix_limits(cpu_seconds: int, memory_mb: int):
    def pre_exec() -> None:  # pragma: no cover - runs in the child
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
            mem = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024 * 1024, 512 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            pass

    return pre_exec


def run_isolated(cmd: list[str], cwd: Path, timeout: int = 600, allow_network: bool = False, allow_llm: bool = False,
                 extra_env: Optional[dict[str, str]] = None, cpu_seconds: int = 600, memory_mb: int = 2048) -> dict[str, Any]:
    env = minimal_env(extra_env, allow_network=allow_network, allow_llm=allow_llm)
    kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        kwargs["preexec_fn"] = _posix_limits(cpu_seconds, memory_mb)
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env, check=False, **kwargs)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        code, out, err = 124, "", f"timed out after {timeout}s"
    except OSError as e:
        code, out, err = 126, "", str(e)
    return {"exit": code, "stdout": out[-8000:], "stderr": err[-8000:], "seconds": round(time.monotonic() - t0, 1),
            "network": allow_network, "llm": allow_llm}
