"""The continuous loop: ingest new chapters -> analyze -> retrain -> (optionally) write.

Run it with ``gauge daemon``. On Windows you can also register it as a Scheduled Task with
``scripts/install-task.ps1`` so it wakes up every few hours with the machine.
"""
from __future__ import annotations

import logging
import signal
import time
from typing import Optional

from .config import GaugeConfig
from .db import Database
from .ingest.runner import IngestRunner
from .learn.trainer import Trainer
from .util import utcnow

log = logging.getLogger(__name__)


class Daemon:
    def __init__(self, cfg: GaugeConfig, db: Database):
        self.cfg = cfg
        self.db = db
        self._stop = False

    def _handle(self, *_: object) -> None:
        log.info("stop requested; finishing current step")
        self._stop = True

    def once(self, write: Optional[bool] = None) -> dict[str, object]:
        started = utcnow()
        out: dict[str, object] = {}
        ingest = IngestRunner(self.cfg, self.db)
        out["ingest"] = ingest.run()
        trainer = Trainer(self.cfg, self.db)
        out["train"] = trainer.cycle()
        do_write = self.cfg.daemon.write_after_learning if write is None else write
        if do_write and self.cfg.daemon.active_project:
            from .write.pipeline import WritePipeline

            pipe = WritePipeline(self.cfg, self.db, trainer=trainer)
            if pipe.llm.available():
                project = pipe.open_project(self.cfg.daemon.active_project)
                res = pipe.write_next(project)
                out["write"] = {"chapter": res.number, "score": round(res.score, 3), "path": str(res.path)}
            else:
                out["write"] = f"skipped: LLM not reachable ({pipe.llm.describe()})"
        self.db.log_run("daemon_cycle", started, out)
        return out

    def run_forever(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._handle)
            signal.signal(signal.SIGTERM, self._handle)
        except (ValueError, AttributeError):  # not main thread / Windows quirks
            pass
        while not self._stop:
            t0 = time.monotonic()
            try:
                summary = self.once()
                log.info("cycle done: %s", summary)
            except Exception as e:  # never let one bad cycle kill the loop
                log.exception("cycle failed: %s", e)
            wait = max(60.0, self.cfg.daemon.cycle_minutes * 60 - (time.monotonic() - t0))
            log.info("sleeping %.0f minutes", wait / 60)
            end = time.monotonic() + wait
            while not self._stop and time.monotonic() < end:
                time.sleep(min(5.0, end - time.monotonic()))
