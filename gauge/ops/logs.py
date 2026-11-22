"""Structured logging: one JSON object per line, so a deployed Gauge can be searched and alerted on.

The console keeps its readable Rich output. Alongside it, every log record is also written to
``<data>/logs/gauge.jsonl`` as a JSON line with a UTC timestamp, level, logger, message, any structured
fields (``event``, ``step``, ``duration_s`` ...) and, for errors, the exception type, message and
traceback. The file rotates by size, so an unattended daemon cannot fill the disk.

Secrets do not belong in logs: field names that look like credentials are masked, and so are
``token=...``-shaped fragments inside messages.
"""
from __future__ import annotations

import json
import logging
import re
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

_MARK = "_gauge_structured"
_SECRET_KEY = re.compile(r"token|passw|secret|api[_-]?key|authorization|credential", re.I)
_SECRET_INLINE = re.compile(r"(?i)\b(token|password|passwd|secret|api[_-]?key|authorization)\b(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|\S+)")
_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}
MASK = "***"


def redact(value: Any) -> Any:
    """Mask anything that looks like a credential, in nested fields and inside message text."""
    if isinstance(value, dict):
        return {k: (MASK if _SECRET_KEY.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _SECRET_INLINE.sub(lambda m: f"{m.group(1)}{m.group(2)}{MASK}", value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        fields = getattr(record, "gauge", None)
        if isinstance(fields, dict):
            for k, v in redact(fields).items():
                doc.setdefault(k, v)
        if record.exc_info and record.exc_info[0] is not None:
            etype, exc, tb = record.exc_info
            doc["error"] = {"type": etype.__name__, "message": redact(str(exc)),
                            "traceback": redact("".join(traceback.format_exception(etype, exc, tb))[-4000:])}
        return json.dumps(doc, default=str, ensure_ascii=False)


def remove_structured_log() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _MARK, False):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:  # closing a log must never raise
                pass


def setup_structured_log(path: Path, max_mb: int = 5, backups: int = 3) -> logging.Handler:
    """Attach the JSON-lines file log to the root logger (replacing an earlier one). The file is created
    on the first record, and rotates at `max_mb`, keeping `backups` old files."""
    remove_structured_log()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=max(1, max_mb) * 1024 * 1024, backupCount=max(0, backups),
                                  encoding="utf-8", delay=True)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _MARK, True)
    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    return handler


def event(logger: logging.Logger, name: str, level: int = logging.INFO, msg: str = "", exc_info: Any = None,
          **fields: Any) -> None:
    """Log a named, structured event. `name` lands in the `event` field; the rest become fields."""
    logger.log(level, msg or name, exc_info=exc_info, extra={"gauge": {"event": name, **fields}})


def read_log(path: Path, tail: int = 50, min_level: Optional[str] = None, event_name: Optional[str] = None) -> list[dict]:
    """The last `tail` matching records, oldest first. Reads the rotated file too when the current one is
    short. Lines that are not valid JSON are skipped rather than failing the read."""
    path = Path(path)
    floor = _LEVELS.get((min_level or "").lower(), 0)
    rows: list[dict] = []
    for p in (path.with_name(path.name + ".1"), path):
        if not p.exists():
            continue
        with p.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if _LEVELS.get(str(rec.get("level", "")).lower(), 0) < floor:
                    continue
                if event_name and rec.get("event") != event_name:
                    continue
                rows.append(rec)
    return rows[-max(1, tail):]
