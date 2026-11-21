"""The continuous loop: ingest new chapters -> analyze -> retrain -> (optionally) write.

Run it with ``gauge daemon``. On Windows you can also register it as a Scheduled Task with
``scripts/install-task.ps1`` so it wakes up every few hours with the machine.
"""
from __future__ import annotations

import logging
import signal
import time
from datetime import timedelta
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

    def _journal(self, kind: str, e: Exception) -> None:
        import traceback

        try:
            self.db.add_journal(kind, f"{type(e).__name__}: {e}", traceback.format_exc())
        except Exception:  # journaling must never raise
            pass

    def _security_check(self, out: dict[str, object]) -> bool:
        """Verify the sealed tree and score the threat. Returns False when automation must stay off this cycle."""
        from .code.selfmod import repo_root
        from .security import AuditLog, assess, lockdown, verify

        audit = AuditLog(self.cfg.security_dir / "audit.jsonl")
        status = "unsealed"
        if self.cfg.security.verify_each_cycle:
            v = verify(repo_root(), self.cfg.security_dir, [self.cfg.native_dir])
            status = v["status"]
            if status in ("tampered", "corrupt"):
                audit.event("integrity", f"{status}: changed={v['changed'][:5]} removed={v['removed'][:5]}", 3, v)
        threat = assess(audit, status)
        out["security"] = {"integrity": status, "threat": threat.level, "reasons": threat.reasons}
        if threat.level >= 3 and self.cfg.security.auto_lockdown and not self.cfg.security.locked:
            out["lockdown"] = lockdown(self.cfg, audit, "; ".join(threat.reasons)[:300])
            log.error("SECURITY LOCKDOWN: %s", threat.reasons)
        return not (self.cfg.security.locked or status in ("tampered", "corrupt"))

    def once(self, write: Optional[bool] = None) -> dict[str, object]:
        started = utcnow()
        out: dict[str, object] = {}
        automation_ok = self._security_check(out)
        ingest = IngestRunner(self.cfg, self.db)
        try:
            out["ingest"] = ingest.run()
        except Exception as e:
            self._journal("ingest", e)
            out["ingest"] = f"failed: {e}"
        trainer = Trainer(self.cfg, self.db)
        out["train"] = trainer.cycle()
        # Absorb craft techniques from the strongest unread chapters (needs the LLM; skipped when offline).
        if self.cfg.learn.absorb_per_cycle > 0:
            from .learn.techniques import Absorber
            from .llm import make_llm

            fast = make_llm(self.cfg.llm, "fast")
            if fast.available():
                try:
                    out["absorb"] = Absorber(self.db, fast, trainer.embedder, self.cfg.models_dir,
                                             self.cfg.learn.target).run(self.cfg.learn.absorb_per_cycle)
                except Exception as e:
                    log.exception("absorb failed: %s", e)
                    out["absorb"] = f"failed: {e}"
            else:
                out["absorb"] = "skipped: LLM not reachable"
        # Sense compute: switch GaugeLM to a GPU that appeared, and find workers that announced themselves.
        if self.cfg.compute.auto_device:
            try:
                from .compute import scan, summary

                use = summary(scan())["use"]
                if use != self.cfg.native.device and self.cfg.native.device != "cpu":
                    log.info("compute: switching native.device %s -> %s", self.cfg.native.device, use)
                    self.cfg.native.device = use
                    self.cfg.save()
                out["device"] = use
            except Exception as e:
                log.warning("device scan failed: %s", e)
        if self.cfg.compute.token:
            try:
                from .compute import Pool

                pool = Pool(self.db, self.cfg.compute.token)
                found = pool.discover_and_register(self.cfg.compute.discover_seconds, self.cfg.compute.announce_port)
                alive = pool.refresh()
                out["workers"] = {"announced": len(found), "alive": [w["name"] for w in alive]}
            except Exception as e:
                log.warning("worker discovery failed: %s", e)
        # Keep the built-in model learning from whatever arrived this cycle.
        if self.cfg.native.daemon_steps > 0:
            from .native import NativeTrainer, gather_texts

            tr = NativeTrainer(self.cfg.native_dir)
            if tr.exists():
                try:
                    since = (started - timedelta(days=1)).isoformat()
                    docs = gather_texts(self.db, self.cfg.projects_dir, self.cfg.models_dir / "datasets", since=since) or \
                        gather_texts(self.db, self.cfg.projects_dir, self.cfg.models_dir / "datasets")
                    out["native"] = tr.continue_training(docs, self.cfg.native.daemon_steps, time_budget_s=self.cfg.native.daemon_minutes * 60,
                                                         device=self.cfg.native.device)
                    if self.cfg.compute.distributed_training and self.cfg.compute.token:
                        from .compute import DistributedTrainer, Pool
                        from .native.train import encode_corpus

                        pool = Pool(self.db, self.cfg.compute.token)
                        if pool.clients():
                            model, tok = tr.load(device=self.cfg.native.device)
                            tokens = encode_corpus(tok, docs, 5_000_000)
                            rnd = DistributedTrainer(pool, self.cfg.native.device).round(model, tokens, steps=max(10, self.cfg.native.daemon_steps // 4),
                                                                                       lr=self.cfg.native.lr / 3)
                            model.save(self.cfg.native_dir, {"step": tr.status().get("step", 0)})
                            out["distributed"] = rnd
                except Exception as e:
                    log.exception("native training failed: %s", e)
                    out["native"] = f"failed: {e}"
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
        # Scheduled bots (plain programs; deterministic ones cannot reach the model).
        if self.cfg.code.bots_enabled and automation_ok:
            try:
                from .code import BotRegistry

                reg = BotRegistry(self.cfg.bots_dir)
                ran = {}
                for name in reg.due():
                    if not reg.manifest(name).get("tests_passed", True):
                        continue
                    ran[name] = reg.run(name, env_extra={"GAUGE_DATA_DIR": str(self.cfg.root)})["exit"]
                if ran:
                    out["bots"] = ran
            except Exception as e:
                self._journal("bots", e)
        # Self-healing: recurring errors become tested proposals; applied only if you allow it.
        if self.cfg.code.self_heal and automation_ok:
            try:
                entries = self.db.recurring_errors(self.cfg.code.min_recurrences)
                if entries:
                    from .code import KnowledgeBase, SelfProgrammer, repo_root
                    from .llm import make_llm

                    llm = make_llm(self.cfg.llm, "main")
                    if llm.available():
                        kb = KnowledgeBase(extra_dir=self.cfg.knowledge_dir)
                        kb.index_self(repo_root() / "gauge")
                        out["self_heal"] = SelfProgrammer(
                            llm, kb, self.cfg.self_dir, mutation_check=self.cfg.code.mutation_check,
                            mutation_min_score=self.cfg.code.mutation_min_score, mutation_max_mutants=self.cfg.code.mutation_max_mutants,
                        ).heal(entries, auto_apply=self.cfg.code.auto_apply)
            except Exception as e:
                self._journal("self_heal", e)
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
                self._journal("cycle", e)
            wait = max(60.0, self.cfg.daemon.cycle_minutes * 60 - (time.monotonic() - t0))
            log.info("sleeping %.0f minutes", wait / 60)
            end = time.monotonic() + wait
            while not self._stop and time.monotonic() < end:
                time.sleep(min(5.0, end - time.monotonic()))
