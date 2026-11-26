"""Gauge Studio: a window with big buttons, for people who do not want a terminal.

Built on tkinter (ships with Python on Windows) so the installer needs nothing extra. All the
real work happens in ``StudioController``, which has no UI in it and is what the tests cover;
the window just calls it from a background thread so it never freezes.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from .config import GaugeConfig
from .db import Database

log = logging.getLogger(__name__)

# What "Teach Gauge" can read.
MATERIAL_EXTS = {".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".epub"}
TEACH_INTRO = ("Give Gauge books or your own writing to learn from. It keeps the strongest passages as examples and "
               "learns craft techniques from them, then uses both every time it writes. Everything stays on this computer.\n"
               "Add files: each file is one book (.txt .md .docx .epub .html). Add a folder: a folder of chapter files "
               "is one book; a folder of books is a library.\n"
               "Use only writing that is yours or that you have the right to use. This teaches Gauge what to aim for; it "
               "does not rebuild the downloaded AI model itself.")
GROW_ASK = ("Grow GaugeLM?\n\nNow: {now}\nAfter: {after}\n\n"
            "It keeps everything it has learned, and its answers are checked to be unchanged before anything is replaced. "
            "The extra room is empty until it trains. It will have about {times:g} times the weights, so training and "
            "writing become several times slower. "
            "The smaller model is kept, so you can undo this.")
VOCAB_ASK = ("Grow GaugeLM's vocabulary from {now} to {after} tokens?\n\n"
             "Tokens are the pieces of words it reads and writes. With more of them the same text takes fewer pieces, so it "
             "sees further back and writes faster. It keeps every token it knows. The new ones are untrained, so it gets a "
             "little worse until it has trained on them: start a training session afterwards. You can undo this.")
SHRINK_ASK = ("Shrink GaugeLM?\n\n"
              "It goes back to the size it had before it last grew (or the next size down), which is faster to train and to "
              "write with. The bigger model teaches the smaller one for about 10 minutes first. A smaller model holds less, "
              "so some of what it learned is lost; Gauge will say how much. The bigger model is kept, so you can undo this.")
DEVICES_NOTE = ("Switch on the devices Gauge may use; more than one can be on. GaugeLM trains and writes on them, and several "
                "graphics cards share the work. The downloaded model picks its own graphics card, but stays on the processor "
                "if no graphics card is switched on here.")
SETUP_GPU_ASK = ("Set up the graphics card for GaugeLM?\n\n"
                 "Gauge found an NVIDIA graphics card, but GaugeLM needs one more library before it can use it. This downloads "
                 "several hundred megabytes and can take several minutes. Your card's normal NVIDIA driver must be installed. "
                 "The downloaded model already uses the graphics card and is not affected.")
FIND_ASK = ("Let Gauge go online and find novels?\n\n"
            "It searches Project Gutenberg, a library of books that are out of copyright, for novels in the genres of your "
            "stories, and adds a few it does not have. It only ever talks to that library; it cannot visit other sites. "
            "Nothing about you or your writing is sent: a search is a word such as 'adventure'.")
CHAT_HINT = ("Type below and press Ask Gauge to talk about your story: ideas, what a character would do, what is "
             "still unresolved. Or type what should happen and press 'Use for the next chapter'.\n")
CHARACTER_INTRO = ("Teach Gauge how to build a character by showing it finished ones. Read the breakdown to see everything "
                   "a new character needs. Save a blank sheet, fill it in with any text editor, then add it here. Gauge checks "
                   "each sheet against the breakdown and tells you what is still missing. The examples you add are shown to "
                   "the writer every time it creates a character, and GaugeLM studies them when it trains.")
ENGINE_NOTE = ("GaugeLM is Gauge's own small model. It has to be trained before it can write, and it writes far more "
               "roughly than the downloaded model. It learns from your library, from every chapter written for your "
               "stories, and from the downloaded model, which teaches it worked examples each time training starts. "
               "So it keeps learning while the downloaded model does the writing. Training takes hours in total: "
               "Start training whenever you like, Pause training when you need the computer, and it carries on from "
               "where it stopped.")
NOT_TRAINED = ("GaugeLM has not been trained yet. Press Teach Gauge and add books or your own writing, "
               "then press Change the AI and Start training.")
UNTIL_PAUSED = "until I pause"


class StudioController:
    """Everything the window can do, as plain methods returning plain data."""

    def __init__(self, cfg: GaugeConfig, db: Database):
        self.cfg = cfg
        self.db = db
        self._pipe = None
        self._pause = threading.Event()          # set by pause_training(); the trainer checks it after each step
        self._training: dict[str, Any] = {"running": False, "step": 0, "loss": None}
        self._chat: dict[str, list[tuple[str, str]]] = {}   # the conversation with Gauge, per story
        self._sensed: Any = None                             # the devices found on this computer (looked for once)

    def status(self) -> dict[str, Any]:
        from .llm import make_llm
        from .write.project import Project

        llm = make_llm(self.cfg.llm)
        eng = self.engine()
        return {"llm_ok": eng["ready"], "llm": llm.describe(), "model": self.cfg.llm.model,
                "projects": Project.list(self.cfg.projects_dir), "engine": eng["label"], "is_gaugelm": eng["is_gaugelm"],
                "capacity": self.capacity(), "version": self.version()}

    def capacity(self) -> str:
        """One sentence on how much of this computer Gauge may use."""
        from . import limit

        return limit.describe(self.cfg)

    def version(self) -> str:
        """The installed version (first seven characters), or '' for a copy that was not installed by the kit."""
        marker = Path(__file__).resolve().parents[1] / "installer" / "installed-version.txt"
        try:
            parts = marker.read_text(encoding="utf-8-sig").split()
        except OSError:
            return ""
        return parts[-1][:7] if parts else ""

    # ---- which AI writes ---------------------------------------------------------------
    def engine(self) -> dict[str, Any]:
        """Which model does the writing, in words, and whether it can write right now."""
        if self.cfg.llm.provider == "native":
            from .native import NativeTrainer

            st = NativeTrainer(self.cfg.native_dir).status()
            trained = bool(st.get("trained"))
            steps = int(st.get("step", 0) or 0)
            label = f"GaugeLM, Gauge's own model (trained {steps} steps)" if trained else "GaugeLM, Gauge's own model (not trained yet)"
            return {"provider": "native", "is_gaugelm": True, "ready": trained, "steps": steps, "label": label}
        from .llm import make_llm

        ready = make_llm(self.cfg.llm).available()
        return {"provider": self.cfg.llm.provider, "is_gaugelm": False, "ready": ready, "steps": 0,
                "label": f"the downloaded model {self.cfg.llm.model}"}

    def _engine_memo(self) -> Path:
        return self.cfg.root / "studio_engine.json"

    def use_gaugelm(self, on: bool = True) -> dict[str, Any]:
        """Point the writer at GaugeLM (on) or back at the downloaded model (off).

        Switching to GaugeLM also selects the lean writing room and shorter chapters, which is what a
        model this small needs; the previous choices are remembered and restored when switching back."""
        import json

        llm, memo = self.cfg.llm, self._engine_memo()
        if on:
            if llm.provider != "native":
                memo.write_text(json.dumps({"provider": llm.provider, "model": llm.model, "fast_model": llm.fast_model,
                                            "num_ctx": llm.num_ctx, "profile": self.cfg.agents.profile,
                                            "words": self.cfg.write.target_chapter_words,
                                            "best_of": self.cfg.write.best_of}), encoding="utf-8")
                llm.provider, llm.model, llm.fast_model = "native", "gaugelm", ""
                llm.num_ctx = self.cfg.native.block_size
                llm.native_dir = str(self.cfg.native_dir)
                self.cfg.agents.profile = "lean"
                self.cfg.write.target_chapter_words = min(self.cfg.write.target_chapter_words, 800)
                self.cfg.write.best_of = max(self.cfg.write.best_of, 4)   # it is fast and uneven: write four, keep the best
        elif llm.provider == "native":
            prev: dict[str, Any] = {}
            if memo.exists():
                try:
                    prev = json.loads(memo.read_text(encoding="utf-8-sig"))
                except ValueError:
                    prev = {}
            llm.provider = prev.get("provider") or "ollama"
            llm.model = prev.get("model") or "llama3.1:8b"
            llm.fast_model = prev.get("fast_model", "")
            llm.num_ctx = int(prev.get("num_ctx") or 8192)
            self.cfg.agents.profile = prev.get("profile") or "full"
            self.cfg.write.target_chapter_words = int(prev.get("words") or 2200)
            self.cfg.write.best_of = int(prev.get("best_of") or 1)
        self.cfg.save()
        self._pipe = None
        eng = self.engine()
        if eng["is_gaugelm"]:
            msg = "Gauge now writes with GaugeLM, its own model, using the lean writing room and shorter chapters. "
            msg += "It is ready to write." if eng["ready"] else "It is not trained yet: add material, then press Start training."
        else:
            msg = f"Gauge now writes with {eng['label']}."
            if not eng["ready"]:
                msg += " It is not answering: open the Ollama app."
        return {**eng, "message": msg}

    # ---- a stronger writer, and more thinking -------------------------------------------
    def stronger_model(self) -> dict[str, Any]:
        """Could this computer run a stronger writing model than the one in use?"""
        from .llm import advisor
        from .llm.client import OllamaLLM

        if self.cfg.llm.provider == "native":
            return {"better": False, "best": None, "message": "GaugeLM is writing. Switch to the downloaded model to use a stronger one."}
        if self.cfg.llm.provider != "ollama":
            return {"better": False, "best": None, "message": f"Gauge is writing with {self.cfg.llm.provider}; this advice is for Ollama models."}
        installed = OllamaLLM(self.cfg.llm, self.cfg.llm.model or "x").models()
        hw = advisor.hardware()
        return advisor.advise(self.cfg.llm.model, hw["vram_gb"], hw["ram_gb"], installed)

    def get_stronger_model(self) -> dict[str, Any]:
        """Download the strongest model this computer can run, and write with it from now on."""
        from .llm import advisor

        adv = self.stronger_model()
        if not adv.get("better"):
            return {"changed": False, "message": adv["message"]}
        if adv.get("needs_download"):
            res = advisor.fetch(adv["best"])
            if not res["ok"]:
                return {"changed": False, "message": res["message"]}
        self.cfg.llm.model = adv["best"]
        self.cfg.save()
        self._pipe = None
        return {"changed": True, "model": adv["best"], "message": f"Gauge now writes with {adv['best']}."}

    def effort(self, level: Optional[str] = None) -> dict[str, Any]:
        """The thinking-effort dial: what it is set to, and (with `level`) setting it."""
        from .write import effort

        if level:
            res = effort.apply(self.cfg, level)
            self.cfg.save()
            self._pipe = None
            return res
        now = effort.current(self.cfg)
        return {"level": now, "message": f"Thinking effort is {now}" + (f": {effort.SAYS[now]}." if now in effort.SAYS else ".")}

    # ---- the device switch -------------------------------------------------------------
    def devices(self, refresh: bool = False) -> dict[str, Any]:
        """The devices this computer has, which are switched on, and a sentence saying what that means."""
        from .compute import devices as dv
        from .compute import scan

        if refresh or self._sensed is None:
            self._sensed = scan()
        return {"rows": dv.sensed(self.cfg, self._sensed), "automatic": not dv.chosen_keys(self.cfg),
                "line": dv.describe(self.cfg, self._sensed), "auto_line": dv.AUTO_LINE}

    def set_device(self, key: str, on: bool) -> dict[str, Any]:
        """Switch one device on or off. Several can be on at once. At least one stays on."""
        from .compute import devices as dv

        now = self.devices()
        current = [r["key"] for r in now["rows"] if r["on"]]
        wanted = [k for k in current if k != key] + ([key] if on else [])
        if not wanted:
            return {**now, "changed": False, "message": "At least one device has to stay on. Switch another one on first."}
        return self._choose_devices([r["key"] for r in now["rows"] if r["key"] in wanted], dv)

    def set_devices_automatic(self) -> dict[str, Any]:
        from .compute import devices as dv

        return self._choose_devices([], dv)

    def _choose_devices(self, keys: list[str], dv: Any) -> dict[str, Any]:
        if self._training["running"]:
            return {**self.devices(), "changed": False,
                    "message": "GaugeLM is training. Press Pause training, wait for it to stop, then change the devices."}
        try:
            res = dv.choose(self.cfg, keys, self._sensed)
        except ValueError as e:
            return {**self.devices(), "changed": False, "message": str(e)}
        self._pipe = None
        return {**self.devices(), "changed": True, "restart": res["restart"], "message": res["message"]}

    def setup_gpu(self) -> dict[str, Any]:
        """Install the library GaugeLM needs to use the graphics card, then look at the devices again."""
        from .compute import devices as dv

        res = dv.setup()
        self._pipe = None
        return {**self.devices(refresh=True), **res}

    # ---- growing GaugeLM ---------------------------------------------------------------
    def gaugelm_size(self) -> dict[str, Any]:
        """How big GaugeLM is now, and what growing it would give."""
        from .native import can_grow

        info = can_grow(self.cfg.native_dir)
        if "now" not in info:
            line = info["reason"]
        elif info["ok"]:
            line = (f"GaugeLM is {info['now']}. It can grow to {info['next']}: about {info['times']:g} times the "
                    "weights, so slower, with room to learn more.")
        else:
            line = f"GaugeLM is {info['now']}. It cannot grow further: {info['reason']}"
        if "vocab" in info:
            line += f" Its vocabulary is {info['vocab']} tokens and can grow to {info['next_vocab']}."
        return {**info, "line": line}

    def _resize(self, work: Callable[[], dict[str, Any]], flag: str) -> dict[str, Any]:
        """Run one change of GaugeLM's size: refused while it trains, settings kept in step afterwards."""
        from .native import GrowError
        from .native.grow import apply_to_config

        if self._training["running"]:
            return {flag: False, "message": "GaugeLM is training. Press Pause training, wait for it to stop, then try again."}
        try:
            res = work()
        except GrowError as e:
            return {flag: False, "message": str(e)}
        apply_to_config(self.cfg, res.get("to") or res["config"])
        self.cfg.save()
        self._pipe = None                       # the writer loads the changed model next time
        return {flag: True, **res}

    def grow_vocabulary(self, vocab_size: Optional[int] = None) -> dict[str, Any]:
        """Give GaugeLM a larger vocabulary, learned from the library. It keeps every token it knows; the
        new ones are untrained, so it needs a training session afterwards."""
        from .learn.corpus import permitted_texts
        from .native import grow_vocab_checkpoint

        texts = [t for _, t in permitted_texts(self.db, self.cfg.projects_dir)]
        res = self._resize(lambda: grow_vocab_checkpoint(self.cfg.native_dir, texts, vocab_size), "grown")
        if res["grown"]:
            res["message"] += " To go back, press Undo the last size change."
        return res

    def shrink_gaugelm(self, to: Optional[str] = None, minutes: Optional[float] = 10,
                       progress: Optional[Callable[[dict[str, Any]], None]] = None) -> dict[str, Any]:
        """Make GaugeLM smaller and faster. The smaller model is cut out of the bigger one and taught by it
        for about `minutes` (Pause training ends the teaching early). Some of what it knew is lost."""
        from .learn.corpus import permitted_texts
        from .native import shrink_checkpoint

        if self._training["running"]:
            return {"shrunk": False, "message": "GaugeLM is training. Press Pause training, wait for it to stop, then try again."}
        docs = permitted_texts(self.db, self.cfg.projects_dir)
        self._pause.clear()

        def work() -> dict[str, Any]:
            return shrink_checkpoint(self.cfg.native_dir, docs, to, steps=1_000_000 if minutes else 300, minutes=minutes,
                                     batch_size=self.cfg.native.batch_size, lr=self.cfg.native.lr,
                                     should_stop=self._pause.is_set, progress=progress)

        try:
            res = self._resize(lambda: self._while_training(work), "shrunk")
        finally:
            self._pause.clear()
        if res["shrunk"]:
            res["message"] += " To go back, press Undo the last size change."
        return res

    def _while_training(self, work: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Run `work` with the training flag up, so Pause training reaches it and nothing else starts."""
        self._training.update(running=True, step=0, loss=None)
        try:
            return work()
        finally:
            self._training["running"] = False

    def grow_gaugelm(self, to: Optional[str] = None) -> dict[str, Any]:
        """Grow GaugeLM to the next size (or a named one) without losing what it learned."""
        from .native import grow_checkpoint

        res = self._resize(lambda: grow_checkpoint(self.cfg.native_dir, to), "grown")
        if res["grown"]:
            res["message"] += " To go back, press Undo the last size change."
        return res

    def undo_grow_gaugelm(self) -> dict[str, Any]:
        """Put GaugeLM back to how it was before its size last changed (grown, vocabulary, or shrunk)."""
        from .native import undo_grow

        return self._resize(lambda: undo_grow(self.cfg.native_dir), "undone")

    def pause_training(self) -> dict[str, Any]:
        """Ask a running training session to stop. It stops after the step it is on, with everything saved;
        starting again carries on from there."""
        if not self._training["running"]:
            return {"running": False, "message": "Nothing is training right now."}
        self._pause.set()
        return {"running": True, "message": "Pausing: finishing the current step and saving. This takes a few seconds."}

    def training(self) -> dict[str, Any]:
        """Is GaugeLM training right now, and how far has it got?"""
        return dict(self._training, pause_requested=self._pause.is_set())

    def train_gaugelm(self, minutes: Optional[float] = 30, progress: Optional[Callable[[dict[str, Any]], None]] = None,
                      teach: bool = True) -> dict[str, Any]:
        """Start training GaugeLM, or carry on from wherever it stopped. It runs for about `minutes`, or with
        `minutes=None` until `pause_training()` is called. Either way a pause stops it cleanly.

        It works whichever model is doing the writing; that is the point. GaugeLM learns from three things:
        the library; every chapter written for your stories, whoever wrote it; and worked examples of
        Gauge's own tasks. When the downloaded model is the writer and is running, `teach` has it answer
        a few of those tasks first, so the big model teaches the small one.

        Only text that is safe to train on is used: what you added yourself, public-domain books, the
        chapters written for you and the examples made on this machine. Scraped third-party fiction is
        left out."""
        from .ingest.finder import BookFinder
        from .learn.corpus import permitted_texts

        found = BookFinder(self.cfg, self.db).run_if_due()     # only when switched on and a day has passed
        if found and found["added"]:
            log.info("the finder added %d novel(s) before training", len(found["added"]))
        prose = permitted_texts(self.db, self.cfg.projects_dir)
        if not prose:
            return {"trained": False, "step": 0, "lessons": 0, "paused": False,
                    "message": "There is nothing to train on yet. Add books or your own writing first."}
        self._pause.clear()
        self._training.update(running=True, step=0, loss=None)
        try:
            return self._train_session(prose, minutes, progress, teach)
        finally:
            self._training["running"] = False
            self._pause.clear()

    def _student(self) -> Any:
        """GaugeLM as it is now, for the tutor to set tasks to. None until it has been trained, or when
        tutoring is switched off."""
        from .llm import make_llm
        from .native import NativeTrainer

        if not self.cfg.native.tutor or not NativeTrainer(self.cfg.native_dir).exists():
            return None
        try:
            return make_llm(self.cfg.llm.model_copy(update={"provider": "native", "model": "gaugelm", "fast_model": "",
                                                            "native_dir": str(self.cfg.native_dir)}), "main")
        except Exception as e:
            log.warning("could not load GaugeLM for tutoring: %s", e)
            return None

    def teachers(self) -> list[str]:
        """Which models teach GaugeLM right now."""
        from .native import school

        return [t.line() for t in school.teachers(self.cfg)]

    def report_card(self) -> dict[str, Any]:
        from .native import school

        return school.report_card(self.cfg.models_dir / "datasets")

    def understanding(self) -> dict[str, Any]:
        """How well GaugeLM understands: how much it uses context, and the words it is most lost on."""
        from .learn.corpus import permitted_texts
        from .native import NativeTrainer, understand

        tr = NativeTrainer(self.cfg.native_dir)
        if not tr.exists():
            return {"ok": False, "message": "GaugeLM has not been trained yet, so there is nothing to measure."}
        text = "\n".join(t for _, t in permitted_texts(self.db, self.cfg.projects_dir))[-400_000:]
        if len(text.split()) < 200:
            return {"ok": False, "message": "There is too little text in the library to measure on. Add a book first."}
        model, tok = tr.load(device=self.cfg.native.device)
        rep = understand.report(model, tok, text)
        return {"ok": True, **rep, "message": rep["line"]}

    def school_line(self) -> str:
        """Who teaches GaugeLM and how it is doing, in a sentence or two."""
        try:
            names = [t.split("  -  ")[0] for t in self.teachers()]
        except Exception:
            names = []
        if not names:
            who = "No teacher model is running, so GaugeLM studies on its own. Open the Ollama app and it will be taught as well."
        elif len(names) == 1:
            who = f"Taught by {names[0]} at the start of each training session. Install more models in Ollama and they take turns."
        else:
            who = "Taught by " + ", ".join(names[:-1]) + f" and {names[-1]}, taking turns at the start of each training session."
        card = self.report_card()
        return who + (" " + card["line"] if card["rows"] else "")

    def _train_session(self, prose: list[tuple[str, str]], minutes: Optional[float],
                       progress: Optional[Callable[[dict[str, Any]], None]], teach: bool) -> dict[str, Any]:
        from .native import (
            ModelConfig,
            NativeTrainer,
            TrainConfig,
            dedup,
            instruction_rows,
            preset_config,
        )

        datasets = self.cfg.models_dir / "datasets"
        lessons, taught = 0, ""
        if teach:
            # Free models teach it first: whichever chat models are installed take turns, answering Gauge's
            # tasks, correcting GaugeLM's own attempts and writing simple practice scenes.
            from .native import school

            try:
                pool = school.teachers(self.cfg)
                if pool:
                    want = 6 if minutes is None else max(4, min(60, round(float(minutes) / 3)))
                    got = school.session(self.cfg, [t for _, t in prose], want, should_stop=self._pause.is_set, pool=pool,
                                         student=self._student(), step=int(NativeTrainer(self.cfg.native_dir).status().get("step", 0) or 0))
                    lessons, taught = int(got["rows"]), str(got["message"])
            except Exception as e:  # a teacher that stumbles must not stop the training
                log.warning("could not collect lessons from the teacher models: %s", e)
        try:  # the character school's lessons are part of every session, even before any example is added
            self.school().export_training()
        except Exception as e:
            log.warning("could not write the character lessons: %s", e)
        try:  # the starter rows are there only while their switch is on
            from .native import seed_lessons

            seed_lessons.sync(datasets, self.cfg.native.seed_data)
        except Exception as e:
            log.warning("could not set out the starter lessons: %s", e)
        docs = dedup(prose + instruction_rows(datasets))
        examples = sum(1 for kind, _ in docs if kind.startswith("chat:"))
        n = self.cfg.native
        mcfg = preset_config(n.size) if n.size else ModelConfig(vocab_size=n.vocab_size, block_size=n.block_size, n_layer=n.n_layer,
                                                                 n_head=n.n_head, n_embd=n.n_embd, n_kv_head=n.n_kv_head)
        mcfg.arch = n.arch
        tcfg = TrainConfig(steps=n.steps, batch_size=n.batch_size, lr=n.lr, accum_steps=n.accum_steps, chat_weight=n.chat_weight,
                           device=n.device, eval_every=25, checkpoint_every=50)
        from .compute import devices as _devices

        tcfg.devices = _devices.train_devices(self.cfg)      # several chosen devices share every step
        trainer = NativeTrainer(self.cfg.native_dir)
        resumed = trainer.exists()
        started = time.monotonic()
        deadline = None if minutes is None else started + max(0.005, float(minutes)) * 60

        def note(rec: dict[str, Any]) -> None:
            self._training.update(step=rec.get("step", 0), loss=rec.get("loss"))
            if progress:
                progress(rec)

        res: dict[str, Any] = {}
        try:
            # One pass of the trainer is `native.steps` long. Keep going, pass after pass, until the time is
            # up or a pause is asked for; every pass carries on from the checkpoint the last one saved.
            while not self._pause.is_set():
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    break
                res = trainer.train(docs, mcfg, tcfg, resume=trainer.exists(), progress=note, time_budget_s=left,
                                    should_stop=self._pause.is_set)
                self._training.update(step=int(res["step"]), loss=res.get("loss"))
                if res.get("stopped") or deadline is None and self._pause.is_set():
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
        except ValueError as e:  # the corpus is too small to cut into training windows
            return {"trained": False, "step": 0, "lessons": lessons, "paused": False,
                    "message": f"Not enough text to train on yet ({e}). Add more material, a few books is a good start, and try again."}
        if not res:
            return {"trained": False, "step": int(trainer.status().get("step", 0) or 0), "lessons": lessons, "paused": True,
                    "message": "Paused before training began. Press Start training to carry on."}
        self._pipe = None
        paused = bool(res.get("stopped")) or self._pause.is_set()
        chars = sum(len(t) for _, t in docs)
        mins = max(1, round((time.monotonic() - started) / 60))
        msg = ("Training paused" if paused else "Trained GaugeLM") + f" after {mins} minute{'' if mins == 1 else 's'} on " \
              f"{len(docs) - examples} chapters and {examples} worked example{'' if examples == 1 else 's'} ({chars:,} characters). "
        if lessons:
            msg += taught + " "
        msg += (f"It is at step {res['step']}" + (f", loss {res['loss']} (lower is better)" if res.get("loss") is not None else "") + ". "
                + ("It carried on from where it stopped. " if resumed else "")
                + ("Progress is saved; press Start training to carry on." if paused
                   else "Start training again whenever you like; it keeps improving for many hours."))
        return {"trained": True, "step": int(res["step"]), "loss": res.get("loss"), "seconds": round(time.monotonic() - started, 1),
                "resumed": resumed, "documents": len(docs), "examples": examples, "lessons": lessons, "paused": paused,
                "message": msg}

    def _require_engine(self) -> None:
        """A clear sentence instead of a stack trace when GaugeLM is selected but has never been trained."""
        if self.cfg.llm.provider == "native":
            from .native import NativeTrainer

            if not NativeTrainer(self.cfg.native_dir).exists():
                raise ValueError(NOT_TRAINED)

    def pipeline(self):
        if self._pipe is None:
            from .write import WritePipeline

            self._pipe = WritePipeline(self.cfg, self.db)
        return self._pipe

    def start_story(self, idea: str, name: Optional[str] = None, genres: Optional[list[str]] = None) -> str:
        idea = idea.strip()
        if not idea:
            raise ValueError("Tell me your story idea first.")
        self._require_engine()
        from .write.project import Project, title_from_text

        title = name or title_from_text(idea)

        base, k = title, 2
        while Project.exists(self.cfg.projects_dir, title):
            title = f"{base} {k}"
            k += 1
        long = len(idea) >= 300
        p = self.pipeline().new_project(title, "" if long else idea, genres or ["Adventure"], premise_doc=idea if long else "")
        return p.state.name

    def chapter_words(self, name: Optional[str] = None) -> int:
        """How long chapters are asked to be: this story's own number, or the default for new stories."""
        from .write.project import Project

        if name and Project.exists(self.cfg.projects_dir, name):
            return int(Project(self.cfg.projects_dir, name).state.target_chapter_words)
        return int(self.cfg.write.target_chapter_words)

    @staticmethod
    def read_words(text: Any) -> int:
        """The number typed in the words box. Accepts '1,500', '1500 words', '2k'. Raises ValueError with
        a sentence the window can show."""
        from .write.pipeline import MAX_CHAPTER_WORDS, MIN_CHAPTER_WORDS

        raw = str(text or "").strip().lower().replace(",", "").replace("words", "").replace("word", "").strip()
        try:
            n = int(float(raw[:-1]) * 1000) if raw.endswith("k") else int(float(raw))
        except ValueError:
            raise ValueError("Type how many words the chapter should be, as a number: for example 1500.") from None
        if not MIN_CHAPTER_WORDS <= n <= MAX_CHAPTER_WORDS:
            raise ValueError(f"A chapter can be from {MIN_CHAPTER_WORDS:,} to {MAX_CHAPTER_WORDS:,} words.")
        return n

    def write_next(self, name: str, words: Any = None) -> dict[str, Any]:
        """Write the next chapter. `words` (a number, or what was typed in the box) says how long it should
        be; the story remembers it for the chapters after."""
        self._require_engine()
        asked = self.read_words(words) if words not in (None, "") else None
        pipe = self.pipeline()
        res = pipe.write_next(pipe.open_project(name), words=asked)
        info = {"number": res.number, "title": res.title, "score": round(res.score, 2), "path": str(res.path),
                "seconds": round(res.timings.get("total", 0.0)), "gaugelm": "", "words": res.words,
                "words_asked": res.words_asked}
        info["record"] = self.record_line(name, res.number)
        # GaugeLM keeps learning while another model writes: the chapter just written is new material for it.
        after = self.cfg.native.after_chapter_minutes
        if after > 0:
            try:
                t = self.train_gaugelm(after, teach=False)
                info["gaugelm"] = f"GaugeLM trained {after} more minute{'' if after == 1 else 's'} (step {t['step']})." if t["trained"] else t["message"]
            except Exception as e:  # training is a bonus; never lose the chapter over it
                log.warning("after-chapter GaugeLM training failed: %s", e)
                info["gaugelm"] = "GaugeLM could not be trained this time."
        return info

    def record_line(self, name: str, chapter: int = 0) -> str:
        """A sentence about the story's record: any contradiction the chapter just written made with what
        was established earlier, however long ago."""
        from .write.chronicle import DB, Chronicle
        from .write.project import Project

        try:
            p = Project(self.cfg.projects_dir, name)
            if not (p.dir / DB).exists():
                return ""
            waiting = [c for c in Chronicle.at(p.dir).open_conflicts(5) if not chapter or c["chapter"] == chapter]
        except Exception as e:
            log.warning("could not read the story's record: %s", e)
            return ""
        if not waiting:
            return ""
        c = waiting[0]
        more = f" (and {len(waiting) - 1} more)" if len(waiting) > 1 else ""
        return (f"The story's record found a contradiction{more}: {c['subject']}, {c['attribute']} was '{c['was']}' in chapter "
                f"{c['was_chapter']} but this chapter says '{c['now']}'.")

    def chapters(self, name: str) -> list[tuple[int, str]]:
        from .write.project import Project

        p = Project(self.cfg.projects_dir, name)
        titles: dict[int, str] = {}
        try:                                     # the story's record has every title in one read, however many there are
            from .write.chronicle import DB, Chronicle

            if (p.dir / DB).exists():
                titles = Chronicle.at(p.dir).titles()
        except Exception as e:
            log.warning("could not read chapter titles from the chronicle: %s", e)
        out = []
        for n in range(1, p.state.written + 1):
            title = titles.get(n)
            if title is None:
                plan = p.plan_for(n)
                title = plan.title if plan else f"Chapter {n}"
            out.append((n, title))
        return out

    def read(self, name: str, n: int) -> str:
        from .write.project import Project

        return Project(self.cfg.projects_dir, name).read_chapter(n) or "(this chapter has not been written yet)"

    def characters(self, name: str) -> list[str]:
        from .write.characters import Roster
        from .write.project import Project

        return Roster.load(Project(self.cfg.projects_dir, name).dir).names()

    def chat(self, name: str, character: str, question: str) -> str:
        from .write.characters import CharacterAgent

        pipe = self.pipeline()
        p = pipe.open_project(name)
        c = pipe.roster(p).get(character)
        if c is None:
            return f"I don't know anyone called {character}."
        return CharacterAgent(pipe.fast_llm, c).chat(question)

    # ---- the chat prompt: ask Gauge, or say what should happen next ------------------
    def ask(self, message: str, name: Optional[str] = None) -> dict[str, Any]:
        """Ask Gauge something. With a story open it answers knowing the bible, the story so far, the cast
        and how the last chapter ended. It talks; it does not change the story."""
        from .llm.tokens import estimate_tokens, truncate_to_tokens
        from .write import prompts
        from .write.project import Project

        message = " ".join((message or "").split())
        if not message:
            return {"answer": "", "message": "Type a question first."}
        self._require_engine()
        pipe = self.pipeline()
        llm = pipe.llm
        max_out = 500
        room = max(200, llm.context_window() - max_out - estimate_tokens(prompts.STUDIO_CHAT_SYSTEM)
                   - estimate_tokens(message) - 200)
        story = ""
        key = name or ""
        if name and Project.exists(self.cfg.projects_dir, name):
            p = pipe.open_project(name)
            st = p.state
            last = p.read_chapter(st.written) if st.written else None
            parts = [f"STORY: {st.name}", truncate_to_tokens(p.bible_markdown(), max(100, room // 3)),
                     f"CHAPTERS WRITTEN: {st.written}",
                     "WHAT HAS HAPPENED SO FAR: " + (p.story_so_far(350) or "(nothing yet)")]
            if st.open_threads:
                parts.append("STILL UNRESOLVED: " + "; ".join(st.open_threads[:8]))
            cast = self.characters(name)
            if cast:
                parts.append("CAST: " + ", ".join(cast[:12]))
            if last:
                parts.append("THE LAST CHAPTER ENDED WITH: " + " ".join(last.split()[-180:]))
            if st.author_direction:
                parts.append("THE AUTHOR ALREADY ASKED FOR THIS IN THE NEXT CHAPTER: " + st.author_direction)
            story = truncate_to_tokens("\n".join(parts), max(150, room * 3 // 4)) + "\n\n"
        turns = self._chat.setdefault(key, [])
        history, left = "", room - estimate_tokens(story)
        for q, a in reversed(turns[-6:]):               # the most recent turns first, while they fit
            block = f"The author said: {q}\nYou answered: {a}\n"
            if estimate_tokens(block) > left:
                break
            history, left = block + history, left - estimate_tokens(block)
        prompt = prompts.STUDIO_CHAT_PROMPT.format(story=story, history=(history + "\n") if history else "", message=message)
        answer = llm.generate(prompt, system=prompts.STUDIO_CHAT_SYSTEM, temperature=0.7, max_tokens=max_out,
                              purpose="studio:chat").strip()
        answer = answer or "I have nothing to say to that. Try asking it another way."
        turns.append((message, answer))
        del turns[:-20]
        return {"answer": answer, "message": answer}

    def clear_chat(self, name: Optional[str] = None) -> None:
        self._chat.pop(name or "", None)

    def direction(self, name: Optional[str]) -> str:
        """What the author has asked for in the next chapter, if anything."""
        from .write.project import Project

        if not name or not Project.exists(self.cfg.projects_dir, name):
            return ""
        return Project(self.cfg.projects_dir, name).state.author_direction

    def direct_next(self, name: Optional[str], text: str) -> dict[str, Any]:
        """Say what should happen in the next chapter. It is followed once, by the next chapter written,
        and then forgotten. An empty text takes back an instruction that has not been used yet."""
        from .write.project import Project

        text = " ".join((text or "").split())
        if not name or not Project.exists(self.cfg.projects_dir, name):
            return {"set": False, "chapter": 0, "direction": "", "message": "Start or open a story first."}
        if not text and not self.direction(name):
            return {"set": False, "chapter": 0, "direction": "",
                    "message": "Type what should happen in the next chapter, then press 'Use for the next chapter'."}
        pipe = self.pipeline()
        p = pipe.open_project(name)
        n = pipe.direct_next(p, text)
        kept = p.state.author_direction
        if not kept:
            return {"set": False, "chapter": n, "direction": "",
                    "message": f"Taken back. Chapter {n} will be written the way Gauge planned it."}
        return {"set": True, "chapter": n, "direction": kept,
                "message": f"Chapter {n} will follow this: \"{kept}\"  Press WRITE THE NEXT CHAPTER when you are ready. "
                           "To take it back, press 'Use for the next chapter' with the box empty."}

    # ---- switches ------------------------------------------------------------------
    def switches(self) -> list[dict[str, Any]]:
        from . import settings as st

        return st.state(self.cfg)

    def set_switch(self, key: str, value: Any) -> Any:
        from . import settings as st

        v = st.set_value(self.cfg, key, value)
        self._pipe = None  # the next job picks up the new settings
        return v

    # ---- finding novels online ---------------------------------------------------------
    def find_novels(self, books: Optional[int] = None, progress: Optional[Callable[[dict[str, Any]], None]] = None) -> dict[str, Any]:
        """Go online and add out-of-copyright novels that fit your stories. Project Gutenberg only."""
        from .ingest.finder import BookFinder

        res = BookFinder(self.cfg, self.db).run(books=books, progress=progress)
        if res["added"]:
            res["message"] += " Press 'Learn from the library now' to study them straight away."
            self._pipe = None
        return res

    def finder_by_itself(self, on: Optional[bool] = None) -> dict[str, Any]:
        """Whether Gauge looks for novels by itself, and (with `on`) switching that."""
        if on is not None:
            self.cfg.library.finder = bool(on)
            self.cfg.save()
        lib = self.cfg.library
        return {"on": lib.finder,
                "message": (f"Gauge looks for novels by itself: up to {lib.finder_books} about every {lib.finder_hours:g} hours, "
                            "when a training session starts." if lib.finder else "Gauge only looks for novels when you press the button.")}

    # ---- teaching character building -------------------------------------------------
    def school(self):
        from .learn.character_school import CharacterSchool

        return CharacterSchool(self.cfg.models_dir)

    def character_breakdown(self) -> str:
        """Everything a newly built character needs, as plain text to read."""
        from .write import blueprint as bp

        return bp.breakdown_markdown().replace("**", "").replace("*Test:*", "Test:").replace("## ", "").replace("# ", "")

    def save_blank_sheet(self, path: str, name: str = "", role: str = "supporting") -> dict[str, Any]:
        from .write import blueprint as bp

        out = Path(path)
        if not out.suffix:
            out = out.with_suffix(".txt")
        out.write_text(bp.blank_sheet(name, role), encoding="utf-8")
        return {"path": str(out), "message": f"Saved a blank character sheet to {out}. Open it in Notepad, fill it in, "
                                             "then press 'Add character sheets'."}

    def character_examples(self) -> list[str]:
        return [e.line() for e in self.school().examples()]

    def teach_characters(self, paths: list[str]) -> dict[str, Any]:
        """Add filled-in character sheets (files, or folders of them) as examples to learn from."""
        school = self.school()
        results: list[dict[str, Any]] = []
        for raw in paths:
            results.extend(school.add_path(Path(raw)))
        return self._school_report(results, "Nothing was added. Choose a filled-in character sheet.")

    def learn_from_cast(self, name: str) -> dict[str, Any]:
        """Learn from the characters of one of your own stories."""
        from .write.characters import Roster
        from .write.project import Project

        if not name or not Project.exists(self.cfg.projects_dir, name):
            return {"added": 0, "refused": 0, "message": "Open or start a story first."}
        p = Project(self.cfg.projects_dir, name)
        results = self.school().promote(Roster.load(p.dir), name, genres=list(p.state.genres))
        return self._school_report(results, f"{name} has no characters to learn from yet. Write a chapter first.")

    def _school_report(self, results: list[dict[str, Any]], nothing: str) -> dict[str, Any]:
        added = [r for r in results if r["added"]]
        refused = [r for r in results if not r["added"]]
        if not results:
            msg = nothing
        else:
            msg = f"Added {len(added)} character{'' if len(added) == 1 else 's'} for Gauge to learn from." if added else ""
            if refused:
                msg += ("  " if msg else "") + "  ".join(r["message"] for r in refused[:3])
                if len(refused) > 3:
                    msg += f"  ...and {len(refused) - 3} more."
            st = self.school().stats()
            msg += f"  The school now holds {st['examples']} example{'' if st['examples'] == 1 else 's'}."
        self._pipe = None
        return {"added": len(added), "refused": len(refused), "message": msg.strip(),
                "names": [r["name"] for r in added]}

    # ---- teaching: add material, then learn from it ----------------------------------
    def library(self) -> list[dict[str, Any]]:
        """Every book Gauge has to learn from: title, where it came from, chapters, your star rating."""
        rows = []
        for n in self.db.novels():
            rows.append({"key": n.key, "title": n.title, "source": n.source,
                         "chapters": len(self.db.chapter_hashes(n.key)), "stars": n.user_rating})
        return sorted(rows, key=lambda r: (r["source"] != "local", r["title"].lower()))

    @staticmethod
    def library_line(row: dict[str, Any]) -> str:
        stars = f"  ({row['stars']:g} stars)" if row.get("stars") else ""
        where = "yours" if row["source"] == "local" else row["source"]
        return f"{row['title']}  -  {row['chapters']} chapter{'' if row['chapters'] == 1 else 's'}, {where}{stars}"

    def add_material(self, paths: list[str], stars: Optional[float] = None) -> dict[str, Any]:
        """Add books or your own writing to the library. Each path is a book file (.txt .md .docx .epub
        .html), a folder of chapter files (one book), or a folder of books. One bad file never stops
        the rest. `stars` is your own 1-5 verdict on this material; it teaches Gauge your taste."""
        from .ingest import IngestRunner
        from .learn.character_school import looks_like_sheet

        if stars is not None and not 1 <= float(stars) <= 5:
            raise ValueError("Stars go from 1 to 5.")
        override = {"user_rating": float(stars)} if stars is not None else None
        runner = IngestRunner(self.cfg, self.db)
        added: list[dict[str, Any]] = []
        problems: list[str] = []
        sheets: list[dict[str, Any]] = []
        for raw in paths:
            path = Path(raw)
            if not path.exists():
                problems.append(f"{path.name}: not found")
                continue
            if looks_like_sheet(path):      # a character sheet is a lesson in building people, not a book
                sheets.extend(self.school().add_path(path))
                continue
            if path.is_file() and path.suffix.lower() not in MATERIAL_EXTS:
                problems.append(f"{path.name}: not a kind of file Gauge can read ({', '.join(sorted(MATERIAL_EXTS))})")
                continue
            try:
                got = runner.import_local(str(path), meta_override=override)
            except Exception as e:  # a broken book must not stop the others
                log.warning("could not import %s: %s", path, e)
                problems.append(f"{path.name}: {e}")
                continue
            if not got:
                problems.append(f"{path.name}: nothing readable found in it")
            added.extend(got)
        chapters = sum(int(r["chapters"]) for r in added)
        new = sum(int(r["new"]) + int(r["changed"]) for r in added)
        if added:
            msg = (f"Added {len(added)} book{'' if len(added) == 1 else 's'} ({chapters} chapters, {new} new or changed). "
                   "Press 'Learn from the library now' so Gauge studies them.")
        elif not sheets:
            msg = "Nothing was added."
        else:
            msg = ""
        if sheets:
            msg = (msg + "  " if msg else "") + self._school_report(sheets, "")["message"]
        if problems:
            msg += "  Could not read: " + "; ".join(problems[:4]) + ("; ..." if len(problems) > 4 else "")
        return {"books": len(added), "chapters": chapters, "new": new, "problems": problems, "message": msg,
                "titles": [str(r["title"]) for r in added], "characters": sum(1 for r in sheets if r["added"])}

    def learn(self) -> dict[str, Any]:
        """Study the library and keep what it teaches, so the next chapter is written with it.

        Three things come out of it, and the writer uses all of them in every prompt:
        example passages (the strongest chapters, to imitate), craft techniques (moves worth reusing,
        which need the story engine to read the chapters), and, when enough books carry reader numbers or
        your star ratings, a model of what makes a chapter work. It does not rebuild the downloaded AI
        model itself; that model stays as it is and is steered by what was learned."""
        from .learn import Trainer

        total = self.db.total_chapters()
        if total == 0:
            return {"chapters": 0, "examples": 0, "techniques": 0, "model": False,
                    "message": "There is nothing to learn from yet. Add a book or a folder of your writing first."}
        trainer = Trainer(self.cfg, self.db)
        res = trainer.cycle()
        need = self.cfg.learn.min_chapters_for_training
        if "exemplars" in res:
            examples = int(res["exemplars"])  # type: ignore[arg-type]
        else:
            # Below the size needed for the full cycle, the examples are still worth keeping: a small
            # library of the author's own choosing is exactly what they want the writer to imitate.
            examples = len(trainer.build_exemplars().items)
        model = isinstance(res.get("model"), dict)
        techniques, technique_note = 0, ""
        if self.cfg.learn.absorb_per_cycle > 0:
            from .llm import make_llm

            fast = make_llm(self.cfg.llm, "fast")
            if fast.available():
                from .learn.techniques import Absorber

                got = Absorber(self.db, fast, trainer.embedder, self.cfg.models_dir,
                               self.cfg.learn.target).run(self.cfg.learn.absorb_per_cycle)
                techniques = int(got.get("techniques_total", 0))
                technique_note = f" Learned craft techniques from {got.get('chapters_absorbed', 0)} more chapters ({techniques} in total)."
            else:
                technique_note = " The story engine is off, so craft techniques were skipped; open the Ollama app and learn again to get them."
        self._pipe = None  # the next chapter is written with what was just learned
        books = len(self.db.novels())
        msg = (f"Studied {total} chapters from {books} book{'' if books == 1 else 's'}. "
               f"Kept {examples} example passage{'' if examples == 1 else 's'} for the writer to imitate.{technique_note}")
        if model:
            msg += " Built a model of what makes a chapter work."
        elif total < need:
            msg += f" With {need} or more chapters Gauge can also measure what makes chapters work (you have {total})."
        else:
            msg += " To also learn what makes chapters work, give several books a star rating when you add them."
        if self.cfg.llm.provider == "native":
            msg += " GaugeLM itself is trained separately: press Change the AI, then Start training."
        return {"chapters": total, "books": books, "examples": examples, "techniques": techniques, "model": model,
                "message": msg}

    def export_story(self, name: str) -> Path:
        """One readable Markdown file of the whole story so far."""
        from .write.project import Project

        p = Project(self.cfg.projects_dir, name)
        parts = [f"# {p.state.name}", ""]
        for n in range(1, p.state.written + 1):
            txt = p.read_chapter(n)
            if txt:
                parts.append(txt)
                parts.append("")
        out = p.dir / f"{p.state.name}.md"
        out.write_text("\n".join(parts), encoding="utf-8")
        return out


def _fit(widget: Any, width: int, height: int) -> str:
    """A window size that fits the screen it opens on: the wanted size, or less on a small display."""
    try:
        sw, sh = int(widget.winfo_screenwidth()), int(widget.winfo_screenheight())
    except Exception:  # a display that will not say how big it is
        sw, sh = 1280, 800
    return f"{min(width, max(640, sw - 80))}x{min(height, max(480, sh - 120))}"


def run_studio(cfg: GaugeConfig, db: Database) -> None:  # pragma: no cover - needs a display
    """The Studio window.

    Layout rule, learned the hard way: nothing important may depend on the window being tall. The buttons
    are placed before the chapter list, so when space is short it is the list that shrinks, never a button
    that falls off the bottom; the window is sized to the screen; and every action is also in the menu bar,
    which is always visible."""
    try:
        import tkinter as tk
        from tkinter import messagebox, scrolledtext, simpledialog, ttk
    except ImportError as e:
        raise SystemExit("tkinter is not available in this Python. Install Python from python.org (it includes it).") from e

    ctl = StudioController(cfg, db)
    root = tk.Tk()
    version = ctl.version()
    root.title("Gauge Studio" + (f"  (version {version})" if version else ""))
    root.geometry(_fit(root, 1100, 860))
    root.minsize(820, 560)
    big = ("Segoe UI", 13)
    mid = ("Segoe UI", 11)
    small = ("Segoe UI", 10)
    huge = ("Segoe UI", 17, "bold")
    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except tk.TclError:
        pass
    style.configure("Big.TButton", font=big, padding=8)
    style.configure("Mid.TButton", font=mid, padding=6)
    style.configure("TLabel", font=big)

    state: dict[str, Any] = {"project": None, "busy": False}
    left = ttk.Frame(root, padding=14)
    left.pack(side="left", fill="y")
    right = ttk.Frame(root, padding=14)
    right.pack(side="right", fill="both", expand=True)

    # ---- left column, top: what is going on, and the idea box
    status_var = tk.StringVar(value="Checking the story engine...")
    ttk.Label(left, textvariable=status_var, wraplength=400, font=mid).pack(anchor="w")
    engine_line = tk.StringVar(value="")
    ttk.Label(left, textvariable=engine_line, wraplength=400, font=("Segoe UI", 11, "bold")).pack(anchor="w")
    capacity_line = tk.StringVar(value=ctl.capacity())
    ttk.Label(left, textvariable=capacity_line, wraplength=400, font=small).pack(anchor="w", pady=(0, 8))
    ttk.Label(left, text="What is your story about?", font=huge).pack(anchor="w")
    idea = scrolledtext.ScrolledText(left, width=40, height=4, font=big, wrap="word")
    idea.pack(fill="x", pady=4)
    idea.insert("1.0", "A kid who finds a dragon egg in the school lost-and-found...")
    start_row = ttk.Frame(left)
    start_row.pack(fill="x")

    # ---- left column, bottom: the buttons. Packed BEFORE the chapter list so they always get their space.
    bottom = ttk.Frame(left)
    bottom.pack(side="bottom", fill="x")
    words_row = ttk.Frame(bottom)
    words_row.pack(fill="x", pady=(6, 0))
    ttk.Label(words_row, text="Words in the next chapter:", font=mid).pack(side="left")
    words_var = tk.StringVar(value=str(ctl.chapter_words()))
    words_entry = ttk.Entry(words_row, textvariable=words_var, width=8, font=big)
    words_entry.pack(side="left", padx=6)
    write_row = ttk.Frame(bottom)
    write_row.pack(fill="x", pady=(4, 2))
    grid = ttk.Frame(bottom)
    grid.pack(fill="x")
    grid.columnconfigure(0, weight=1, uniform="half")
    grid.columnconfigure(1, weight=1, uniform="half")

    # ---- left column, middle: the story and its chapters. This is the part that shrinks.
    proj_var = tk.StringVar(value="No story yet")
    ttk.Label(left, textvariable=proj_var, font=("Segoe UI", 11, "italic")).pack(anchor="w", pady=(8, 0))
    progress = ttk.Progressbar(left, mode="indeterminate", length=360)
    progress.pack(fill="x", pady=4)
    chapters_box = tk.Listbox(left, font=big, height=4)
    chapters_box.pack(fill="both", expand=True, pady=4)

    # ---- right side, bottom: the chat prompt. Packed BEFORE the reader so it always keeps its space.
    chat = ttk.Frame(right)
    chat.pack(side="bottom", fill="x", pady=(8, 0))
    ttk.Label(chat, text="Ask Gauge, or say what should happen next", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    chat_log = scrolledtext.ScrolledText(chat, height=6, font=mid, wrap="word", padx=8, pady=6)
    chat_log.pack(fill="x", pady=(2, 4))
    chat_log.insert("end", CHAT_HINT)
    chat_log.configure(state="disabled")
    chat_row = ttk.Frame(chat)
    chat_row.pack(fill="x")
    chat_var = tk.StringVar(value="")
    chat_entry = ttk.Entry(chat_row, textvariable=chat_var, font=big)
    chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    ask_btn = ttk.Button(chat_row, text="Ask Gauge", style="Mid.TButton", command=lambda: on_ask())
    ask_btn.pack(side="left", padx=(0, 6))
    use_btn = ttk.Button(chat_row, text="Use for the next chapter", style="Mid.TButton", command=lambda: on_direct())
    use_btn.pack(side="left")
    direction_line = tk.StringVar(value="")
    ttk.Label(chat, textvariable=direction_line, wraplength=640, font=small).pack(anchor="w", pady=(3, 0))
    chat_entry.bind("<Return>", lambda _evt: on_ask())

    reader = scrolledtext.ScrolledText(right, font=("Georgia", 14), wrap="word", padx=16, pady=16)
    reader.pack(fill="both", expand=True)
    reader.insert("1.0", "Your chapters will appear here.\n\n1. Type your idea on the left.\n2. Press START MY STORY.\n"
                         "3. Press WRITE THE NEXT CHAPTER whenever you want more.\n4. Click a chapter to read it.\n\n"
                         "Teach Gauge: give it books or your own writing to learn from.\n"
                         "Change the AI: choose which model writes, and train Gauge's own.\n"
                         "Both are also in the menu at the top of this window.\n\n"
                         "Ask Gauge: type in the box at the bottom to talk about your story, or to say what "
                         "should happen in the next chapter.")
    buttons: list[Any] = [ask_btn, use_btn]

    def set_busy(flag: bool, msg: str = "") -> None:
        state["busy"] = flag
        progress.start(12) if flag else progress.stop()
        if msg:
            status_var.set(msg)
        for b in buttons:
            b.state(["disabled"] if flag else ["!disabled"])

    def run_bg(work: Callable[[], Any], done: Callable[[Any], None], msg: str) -> None:
        if state["busy"]:
            return
        set_busy(True, msg)

        def target() -> None:
            try:
                result = work()
                root.after(0, lambda: (set_busy(False, "Ready."), done(result)))
            except Exception as e:
                log.error("studio task failed: %s\n%s", e, traceback.format_exc())
                err = str(e)
                root.after(0, lambda: (set_busy(False, "Something went wrong."),
                                       messagebox.showerror("Gauge", f"{err}\n\nIf this keeps happening, run 'gauge doctor'.")))

        threading.Thread(target=target, daemon=True).start()

    def show_engine() -> None:
        """Say which AI writes and whether it can, on the main window."""
        eng = ctl.engine()
        engine_line.set("Writing with: " + eng["label"])
        capacity_line.set(ctl.capacity())
        if eng["ready"]:
            status_var.set("Story engine ready.")
        elif eng["is_gaugelm"]:
            status_var.set(NOT_TRAINED)
        else:
            status_var.set("Story engine is OFF. Open the Ollama app (the llama icon), then restart Gauge Studio.")

    def refresh_chapters() -> None:
        chapters_box.delete(0, "end")
        if state["project"]:
            for n, title in ctl.chapters(state["project"]):
                chapters_box.insert("end", f"Chapter {n}: {title}")

    def show_chapter(n: int) -> None:
        reader.delete("1.0", "end")
        reader.insert("1.0", ctl.read(state["project"], n))

    def say(who: str, text: str) -> None:
        """Add a line to the conversation and keep the newest in view."""
        chat_log.configure(state="normal")
        chat_log.insert("end", f"\n{who}: {text}\n")
        chat_log.see("end")
        chat_log.configure(state="disabled")

    def show_words() -> None:
        words_var.set(str(ctl.chapter_words(state["project"])))

    def show_direction() -> None:
        asked = ctl.direction(state["project"])
        direction_line.set(f"The next chapter will follow: \"{asked}\"" if asked else "")

    def on_ask() -> None:
        text = chat_var.get().strip()
        if not text:
            return
        if state["busy"]:
            status_var.set("Gauge is busy. Wait for the current job to finish, then ask again.")
            return
        chat_var.set("")
        say("You", text)
        run_bg(lambda: ctl.ask(text, state["project"]), lambda res: say("Gauge", res["answer"]), "Gauge is thinking...")

    def on_direct() -> None:
        if not state["project"]:
            messagebox.showinfo("Gauge", "Start a story first!")
            return
        text = chat_var.get().strip()
        res = ctl.direct_next(state["project"], text)
        chat_var.set("")
        if text:
            say("You", text)
        say("Gauge", res["message"])
        show_direction()

    def on_start() -> None:
        text = idea.get("1.0", "end").strip()

        def done(name: str) -> None:
            state["project"] = name
            proj_var.set(f"Story: {name}")
            refresh_chapters()
            show_direction()
            show_words()
            reader.delete("1.0", "end")
            reader.insert("1.0", f"'{name}' is ready!\n\nPress WRITE THE NEXT CHAPTER to begin.")

        run_bg(lambda: ctl.start_story(text), done, "Dreaming up your world and characters... (a minute or two)")

    def on_write() -> None:
        if not state["project"]:
            messagebox.showinfo("Gauge", "Start a story first!")
            return

        try:
            asked = ctl.read_words(words_var.get())
        except ValueError as e:
            messagebox.showinfo("Gauge", str(e))
            return

        def done(info: dict[str, Any]) -> None:
            refresh_chapters()
            show_chapter(info["number"])
            show_direction()                    # the instruction was for that chapter; it is used up now
            status_var.set(f"Chapter {info['number']} done in {info['seconds']}s: {info['words']:,} words (asked for "
                           f"{info['words_asked']:,}), score {info['score']}. {info.get('record', '')} {info.get('gaugelm', '')}".strip())

        run_bg(lambda: ctl.write_next(state["project"], asked), done,
               f"Writing the next chapter, about {asked:,} words... this takes a few minutes.")

    def on_pick(_evt: Any = None) -> None:
        sel = chapters_box.curselection()
        if sel and state["project"]:
            show_chapter(sel[0] + 1)

    def on_open() -> None:
        names = ctl.status()["projects"]
        if not names:
            messagebox.showinfo("Gauge", "No stories yet. Start one!")
            return
        pick = simpledialog.askstring("Open a story", "Which one?\n\n" + "\n".join(names), parent=root)
        if pick and pick in names:
            state["project"] = pick
            proj_var.set(f"Story: {pick}")
            refresh_chapters()
            show_direction()
            show_words()
            if ctl.chapters(pick):
                show_chapter(len(ctl.chapters(pick)))

    def on_chat() -> None:
        if not state["project"]:
            messagebox.showinfo("Gauge", "Start a story first!")
            return
        names = ctl.characters(state["project"])
        if not names:
            messagebox.showinfo("Gauge", "No characters yet.")
            return
        who = simpledialog.askstring("Talk to a character", "Who?\n\n" + "\n".join(names), parent=root)
        if not who:
            return
        q = simpledialog.askstring("Talk to a character", f"What do you want to ask {who}?", parent=root)
        if not q:
            return

        def done(answer: str) -> None:
            reader.insert("end", f"\n\n---\nYou asked {who}: {q}\n{who}: {answer}\n")
            reader.see("end")

        run_bg(lambda: ctl.chat(state["project"], who, q), done, f"{who} is thinking...")

    def on_export() -> None:
        if state["project"]:
            messagebox.showinfo("Gauge", f"Saved the whole story to:\n{ctl.export_story(state['project'])}")
        else:
            messagebox.showinfo("Gauge", "Start a story first!")

    def dialog(title: str, width: int, height: int) -> tuple[Any, Any, Callable[[], bool], Any, Callable[..., None]]:
        """A small window with a status line and a starter for background jobs that report into it."""
        win = tk.Toplevel(root)
        win.title(title)
        win.geometry(_fit(root, width, height))
        frame = ttk.Frame(win, padding=14)
        frame.pack(fill="both", expand=True)
        note = tk.StringVar(value="")

        def alive() -> bool:
            try:
                return bool(win.winfo_exists())
            except tk.TclError:
                return False

        def start(work: Callable[[], Any], msg: str, after: Optional[Callable[[], None]] = None) -> None:
            if state["busy"]:
                note.set("Gauge is busy. Wait for the current job to finish, then try again.")
                return
            note.set(msg)

            def done(res: dict[str, Any]) -> None:
                if alive():
                    note.set(res["message"])
                if after:
                    after()

            run_bg(work, done, msg)

        return win, frame, alive, note, start

    def on_teach() -> None:
        """Add books or your own writing to the library, then have Gauge learn from them."""
        from tkinter import filedialog

        win, frame, alive, note, start = dialog("Teach Gauge", 760, 640)
        ttk.Label(frame, text="Teach Gauge", font=huge).pack(anchor="w")
        ttk.Label(frame, text=TEACH_INTRO, wraplength=700, font=small).pack(anchor="w", pady=(2, 8))
        rate_row = ttk.Frame(frame)
        rate_row.pack(fill="x")
        ttk.Label(rate_row, text="How good is what you are adding? (optional)", font=mid).pack(side="left")
        stars_var = tk.StringVar(value="not rated")
        ttk.Combobox(rate_row, textvariable=stars_var, values=["not rated", "1", "2", "3", "4", "5"], state="readonly",
                     width=10).pack(side="left", padx=8)
        teach_buttons = ttk.Frame(frame)
        teach_buttons.pack(fill="x", pady=(8, 4))
        find_row = ttk.Frame(frame)
        find_row.pack(fill="x", pady=(0, 4))
        # The status line goes in before the list, at the bottom, so a short window never hides it.
        ttk.Label(frame, textvariable=note, wraplength=700, font=mid).pack(side="bottom", anchor="w", pady=(4, 0))
        ttk.Label(frame, text="In the library", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 0))
        lib_box = tk.Listbox(frame, font=mid, height=5)
        lib_box.pack(fill="both", expand=True, pady=4)

        def refresh_library() -> None:
            if not alive():
                return
            lib_box.delete(0, "end")
            rows = ctl.library()
            for r in rows:
                lib_box.insert("end", ctl.library_line(r))
            if not rows:
                lib_box.insert("end", "(nothing yet: add a book, or a folder of your writing)")

        def add(paths: Any) -> None:
            picked = [str(x) for x in (paths or []) if x]
            if picked:
                v = stars_var.get()
                stars = float(v) if v.isdigit() else None
                start(lambda: ctl.add_material(picked, stars), "Reading your files...", refresh_library)

        def on_files() -> None:
            add(filedialog.askopenfilenames(parent=win, title="Choose books or documents",
                                            filetypes=[("Books and documents", "*.txt *.md *.docx *.epub *.html *.htm"),
                                                       ("All files", "*.*")]))

        def on_folder() -> None:
            folder = filedialog.askdirectory(parent=win, title="Choose a folder of chapters or books")
            add([folder] if folder else [])

        def on_learn() -> None:
            start(ctl.learn, "Studying your library... this can take several minutes.", refresh_library)

        for text, cmd in (("Add files...", on_files), ("Add a folder...", on_folder), ("Learn from the library now", on_learn),
                          ("Characters...", on_characters)):
            ttk.Button(teach_buttons, text=text, style="Mid.TButton", command=cmd).pack(side="left", padx=(0, 8))

        def on_find() -> None:
            if messagebox.askyesno("Find novels online", FIND_ASK, parent=win):
                start(ctl.find_novels, "Looking for novels online... each book takes a little while.", refresh_library)

        def on_find_auto() -> None:
            note.set(ctl.finder_by_itself(bool(auto_find.get()))["message"])

        auto_find = tk.BooleanVar(value=ctl.finder_by_itself()["on"])
        ttk.Button(find_row, text="Find novels online", style="Mid.TButton", command=on_find).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(find_row, text="Keep finding novels by itself (about once a day)", variable=auto_find,
                        command=on_find_auto).pack(side="left")
        refresh_library()

    def on_characters() -> None:
        """Teach Gauge how to build a character: the breakdown, a blank sheet, and finished examples."""
        from tkinter import filedialog

        win, frame, alive, note, start = dialog("Teach character building", 760, 560)
        ttk.Label(frame, text="Teach character building", font=huge).pack(anchor="w")
        ttk.Label(frame, text=CHARACTER_INTRO, wraplength=700, font=small).pack(anchor="w", pady=(2, 8))
        row_a = ttk.Frame(frame)
        row_a.pack(fill="x", pady=(0, 4))
        row_b = ttk.Frame(frame)
        row_b.pack(fill="x", pady=(0, 4))
        ttk.Label(frame, textvariable=note, wraplength=700, font=mid).pack(side="bottom", anchor="w", pady=(4, 0))
        ttk.Label(frame, text="Characters Gauge is learning from", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 0))
        box = tk.Listbox(frame, font=mid, height=5)
        box.pack(fill="both", expand=True, pady=4)

        def refresh() -> None:
            if not alive():
                return
            box.delete(0, "end")
            lines = ctl.character_examples()
            for line in lines:
                box.insert("end", line)
            if not lines:
                box.insert("end", "(none yet: save a blank sheet, fill it in, and add it)")

        def on_breakdown() -> None:
            view = tk.Toplevel(win)
            view.title("What a new character needs")
            view.geometry(_fit(root, 820, 640))
            text = scrolledtext.ScrolledText(view, wrap="word", font=mid, padx=12, pady=10)
            text.pack(fill="both", expand=True)
            text.insert("1.0", ctl.character_breakdown())
            text.configure(state="disabled")

        def on_blank() -> None:
            path = filedialog.asksaveasfilename(parent=win, title="Save a blank character sheet",
                                                initialfile="my-character.txt", defaultextension=".txt",
                                                filetypes=[("Text file", "*.txt")])
            if path:
                note.set(ctl.save_blank_sheet(path)["message"])

        def on_add() -> None:
            picked = [str(x) for x in (filedialog.askopenfilenames(
                parent=win, title="Choose filled-in character sheets",
                filetypes=[("Character sheets", "*.txt *.md *.json"), ("All files", "*.*")]) or []) if x]
            if picked:
                start(lambda: ctl.teach_characters(picked), "Checking the sheets...", refresh)

        def on_cast() -> None:
            name = state["project"]
            if not name:
                note.set("Open or start a story first.")
                return
            start(lambda: ctl.learn_from_cast(name), f"Learning from the characters of {name}...", refresh)

        for parent, pairs_ in ((row_a, (("What a character needs", on_breakdown), ("Save a blank sheet...", on_blank))),
                               (row_b, (("Add character sheets...", on_add), ("Learn from this story's characters", on_cast)))):
            for text, cmd in pairs_:
                ttk.Button(parent, text=text, style="Mid.TButton", command=cmd).pack(side="left", padx=(0, 8))
        refresh()

    def start_training(note: Any, how_long: str, starter: Optional[Callable[..., None]] = None,
                       after: Optional[Callable[[], None]] = None) -> None:
        """Start (or carry on) training GaugeLM. `how_long` is a number of minutes, or UNTIL_PAUSED."""
        minutes = float(how_long) if str(how_long).replace(".", "", 1).isdigit() else None

        def progress(rec: dict[str, Any]) -> None:
            line = f"Training GaugeLM... step {rec.get('step')}, loss {rec.get('loss')} (lower is better). Pause any time."
            root.after(0, lambda: (status_var.set(line), note.set(line) if note is not None else None))

        msg = ("Training GaugeLM until you press Pause training." if minutes is None
               else f"Training GaugeLM for about {minutes:g} minutes. Pause any time.")

        def finished(res: dict[str, Any]) -> None:
            show_engine()
            status_var.set(res["message"])
            if after:
                after()

        if starter is not None:
            starter(lambda: ctl.train_gaugelm(minutes, progress), msg, lambda: (show_engine(), after() if after else None))
        else:
            run_bg(lambda: ctl.train_gaugelm(minutes, progress), finished, msg)

    def on_pause_training() -> None:
        status_var.set(ctl.pause_training()["message"])

    def on_engine() -> None:
        """Choose which AI writes, and train Gauge's own."""
        win, frame, alive, note, start = dialog("Change the AI", 760, 720)
        ttk.Label(frame, text="Which AI writes", font=huge).pack(anchor="w")
        now_var = tk.StringVar(value="Writing with: " + ctl.engine()["label"])
        ttk.Label(frame, textvariable=now_var, wraplength=700, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(2, 0))
        ttk.Label(frame, text=ENGINE_NOTE, wraplength=700, font=small).pack(anchor="w", pady=(4, 8))
        choose = ttk.Frame(frame)
        choose.pack(fill="x", pady=(0, 6))
        train = ttk.Frame(frame)
        train.pack(fill="x", pady=(0, 6))
        minutes_var = tk.StringVar(value=UNTIL_PAUSED)
        limit_var = tk.StringVar(value=ctl.capacity() + " Change it in Switches (Capacity limit).")
        ttk.Label(frame, textvariable=limit_var, wraplength=700, font=small).pack(anchor="w", pady=(0, 6))
        school_var = tk.StringVar(value=ctl.school_line())
        ttk.Label(frame, textvariable=school_var, wraplength=700, font=small).pack(anchor="w", pady=(0, 6))
        size_var = tk.StringVar(value=ctl.gaugelm_size()["line"])
        ttk.Label(frame, textvariable=size_var, wraplength=700, font=small).pack(anchor="w")
        grow_row = ttk.Frame(frame)
        grow_row.pack(fill="x", pady=(4, 2))
        shrink_row = ttk.Frame(frame)
        shrink_row.pack(fill="x", pady=(0, 6))
        ttk.Label(frame, textvariable=note, wraplength=700, font=mid).pack(anchor="w")

        def refresh() -> None:
            if alive():
                now_var.set("Writing with: " + ctl.engine()["label"])
                size_var.set(ctl.gaugelm_size()["line"])
                school_var.set(ctl.school_line())
            show_engine()

        def on_grow() -> None:
            info = ctl.gaugelm_size()
            if not info.get("ok"):
                note.set(info["line"])
                return
            if not messagebox.askyesno("Grow GaugeLM", GROW_ASK.format(now=info["now"], after=info["next"], times=info["times"]),
                                       parent=win):
                return
            start(ctl.grow_gaugelm, "Growing GaugeLM... this takes a moment.", refresh)

        def on_vocab() -> None:
            info = ctl.gaugelm_size()
            if "vocab" not in info:
                note.set(info["line"])
                return
            if messagebox.askyesno("Grow the vocabulary", VOCAB_ASK.format(now=info["vocab"], after=info["next_vocab"]),
                                   parent=win):
                start(ctl.grow_vocabulary, "Learning new tokens from your library...", refresh)

        def on_shrink() -> None:
            if "vocab" not in ctl.gaugelm_size():
                note.set(ctl.gaugelm_size()["line"])
                return
            if messagebox.askyesno("Shrink GaugeLM", SHRINK_ASK, parent=win):
                start(ctl.shrink_gaugelm, "Shrinking GaugeLM: the bigger model is teaching the smaller one (about 10 minutes; "
                                          "Pause training ends it early)...", refresh)

        def on_undo_grow() -> None:
            if messagebox.askyesno("Undo the last size change", "Put GaugeLM back to how it was before its size last changed?"
                                   "\n\nTraining it has done since then will be lost.", parent=win):
                start(ctl.undo_grow_gaugelm, "Putting the earlier GaugeLM back...", refresh)

        def on_use(gaugelm: bool) -> None:
            if state["busy"]:
                note.set("Gauge is busy. Wait for the current job to finish, then try again.")
                return
            try:
                note.set(ctl.use_gaugelm(gaugelm)["message"])
            except Exception as e:
                note.set(f"Could not switch: {e}")
            refresh()

        def on_train() -> None:
            start_training(note if alive() else None, minutes_var.get(), start, refresh)

        def on_pause() -> None:
            note.set(ctl.pause_training()["message"])

        ttk.Button(choose, text="Use GaugeLM", style="Mid.TButton", command=lambda: on_use(True)).pack(side="left", padx=(0, 8))
        ttk.Button(choose, text="Use the downloaded model", style="Mid.TButton", command=lambda: on_use(False)).pack(side="left")
        ttk.Button(train, text="Start training", style="Mid.TButton", command=on_train).pack(side="left", padx=(0, 6))
        ttk.Button(train, text="Pause training", style="Mid.TButton", command=on_pause).pack(side="left", padx=(0, 10))
        ttk.Label(train, text="Run for", font=mid).pack(side="left")
        ttk.Combobox(train, textvariable=minutes_var, values=[UNTIL_PAUSED, "10", "30", "60", "120", "240", "480"],
                     state="readonly", width=14).pack(side="left", padx=4)
        ttk.Label(train, text="(minutes)", font=small).pack(side="left")
        def on_stronger() -> None:
            adv = ctl.stronger_model()
            if not adv.get("better"):
                note.set(adv["message"])
                return
            if messagebox.askyesno("A stronger writer", adv["message"] + "\n\nGet it and write with it from now on? "
                                   "A bigger model writes better and is slower.", parent=win):
                start(ctl.get_stronger_model, f"Getting {adv['best']}... a large download can take a long time.", refresh)

        ttk.Button(choose, text="Find a stronger model", style="Mid.TButton", command=on_stronger).pack(side="left", padx=(8, 0))
        ttk.Button(train, text="How well does it understand?", style="Mid.TButton",
                   command=lambda: start(ctl.understanding, "Measuring how GaugeLM uses context...")).pack(side="left", padx=(10, 0))
        for row, pair in ((grow_row, (("Grow GaugeLM", on_grow), ("Grow the vocabulary", on_vocab))),
                          (shrink_row, (("Shrink GaugeLM", on_shrink), ("Undo the last size change", on_undo_grow)))):
            for text, cmd in pair:
                ttk.Button(row, text=text, style="Mid.TButton", command=cmd).pack(side="left", padx=(0, 8))

    def on_switches() -> None:
        """A window of every switch, grouped, built from the catalog. Changes save immediately."""
        win = tk.Toplevel(root)
        win.title("Gauge Switches")
        win.geometry(_fit(root, 760, 680))
        canvas = tk.Canvas(win, highlightthickness=0)
        scroll = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=12)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        status = tk.StringVar(value="Changes are saved as soon as you make them.")
        ttk.Label(inner, textvariable=status, font=("Segoe UI", 11, "italic")).pack(anchor="w", pady=(0, 8))

        def apply(key: str, value: Any, label: str) -> None:
            try:
                v = ctl.set_switch(key, value)
                status.set(f"{label} = {v}")
                capacity_line.set(ctl.capacity())
            except Exception as e:
                status.set(f"Could not set {label}: {e}")

        # ---- devices: one switch per device this computer has; several can be on
        ttk.Label(inner, text="Devices", font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(4, 4))
        ttk.Label(inner, text=DEVICES_NOTE, wraplength=680, font=small).pack(anchor="w")
        dev_line = tk.StringVar(value="Looking at this computer's devices...")
        ttk.Label(inner, textvariable=dev_line, wraplength=680, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(4, 2))
        dev_box = ttk.Frame(inner)
        dev_box.pack(fill="x")
        dev_vars: dict[str, Any] = {}

        def show_devices(info: dict[str, Any]) -> None:
            try:
                if not win.winfo_exists():
                    return
            except tk.TclError:
                return
            dev_line.set(info.get("message") or info["line"])
            auto_var.set(info["automatic"])
            for r in info["rows"]:
                if r["key"] not in dev_vars:
                    var = tk.BooleanVar(value=r["on"])
                    dev_vars[r["key"]] = var
                    row = ttk.Frame(dev_box)
                    row.pack(fill="x", pady=2)
                    box = ttk.Checkbutton(row, text=r["label"], variable=var,
                                          command=lambda k=r["key"], v=var: show_devices(ctl.set_device(k, bool(v.get()))))
                    box.pack(anchor="w")
                    if not r["ready"]:
                        box.state(["disabled"])
                        ttk.Label(row, text=r["note"], wraplength=660, font=small).pack(anchor="w", padx=(22, 0))
                else:
                    dev_vars[r["key"]].set(r["on"])
            if any(r["can_set_up"] for r in info["rows"]) and not setup_btn.winfo_ismapped():
                setup_btn.pack(anchor="w", pady=(4, 0))

        def on_auto() -> None:
            if auto_var.get():
                show_devices(ctl.set_devices_automatic())
            else:                                   # switching Automatic off keeps what it had picked, as a fixed choice
                picked = [r["key"] for r in ctl.devices()["rows"] if r["on"]]
                show_devices(ctl.set_device(picked[0], True) if picked else ctl.devices())

        def on_setup_gpu() -> None:
            if state["busy"]:
                dev_line.set("Gauge is busy. Wait for the current job to finish, then try again.")
                return
            if messagebox.askyesno("Set up the graphics card", SETUP_GPU_ASK, parent=win):
                dev_line.set("Installing the graphics library... this is a large download and can take several minutes.")
                run_bg(ctl.setup_gpu, show_devices, "Setting up the graphics card...")

        auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(inner, text="Automatic (let Gauge pick)", variable=auto_var, command=on_auto).pack(anchor="w", pady=(4, 0))
        setup_btn = ttk.Button(inner, text="Set up the graphics card", style="Mid.TButton", command=on_setup_gpu)

        def look() -> None:
            try:
                info = ctl.devices()
            except Exception as e:
                log.warning("could not look at the devices: %s", e)
                root.after(0, lambda: dev_line.set("Could not look at this computer's devices."))
                return
            root.after(0, lambda: show_devices(info))

        threading.Thread(target=look, daemon=True).start()   # finding the devices can take a second or two

        rows = ctl.switches()
        for group in sorted({r["group"] for r in rows}, key=lambda g: [r["group"] for r in rows].index(g)):
            ttk.Label(inner, text=group, font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(12, 4))
            for r in [x for x in rows if x["group"] == group]:
                frame = ttk.Frame(inner)
                frame.pack(fill="x", pady=3)
                if r["kind"] == "bool":
                    var = tk.BooleanVar(value=bool(r["value"]))
                    ttk.Checkbutton(frame, text=r["label"], variable=var,
                                    command=lambda k=r["key"], v=var, lab=r["label"]: apply(k, v.get(), lab)).pack(anchor="w")
                elif r["kind"] in ("float", "int"):
                    ttk.Label(frame, text=f"{r['label']}  ({r['value']})").pack(anchor="w")
                    var = tk.DoubleVar(value=float(r["value"]))
                    res = 1 if r["kind"] == "int" else 0.05
                    sc = ttk.Scale(frame, from_=r["minimum"] or 0, to=r["maximum"] or 1, variable=var, orient="horizontal", length=320)
                    sc.pack(anchor="w")
                    sc.bind("<ButtonRelease-1>", lambda e, k=r["key"], v=var, lab=r["label"], step=res: apply(k, round(v.get() / step) * step, lab))
                elif r["kind"] == "choice":
                    ttk.Label(frame, text=r["label"]).pack(anchor="w")
                    var = tk.StringVar(value=str(r["value"]))
                    cb = ttk.Combobox(frame, textvariable=var, values=r["choices"], state="readonly", width=24)
                    cb.pack(anchor="w")
                    cb.bind("<<ComboboxSelected>>", lambda e, k=r["key"], v=var, lab=r["label"]: apply(k, v.get(), lab))
                else:
                    ttk.Label(frame, text=r["label"]).pack(anchor="w")
                    var = tk.StringVar(value=str(r["value"]))
                    ent = ttk.Entry(frame, textvariable=var, width=36)
                    ent.pack(anchor="w")
                    ent.bind("<Return>", lambda e, k=r["key"], v=var, lab=r["label"]: apply(k, v.get(), lab))
                    ent.bind("<FocusOut>", lambda e, k=r["key"], v=var, lab=r["label"]: apply(k, v.get(), lab))
                ttk.Label(frame, text=r["description"], wraplength=680, font=small).pack(anchor="w")
                if r["warning"]:
                    ttk.Label(frame, text="! " + r["warning"], wraplength=680, foreground="#b00020", font=("Segoe UI", 10, "bold")).pack(anchor="w")

    def on_about() -> None:
        eng = ctl.engine()
        messagebox.showinfo("About Gauge Studio",
                            f"Gauge Studio{' version ' + version if version else ''}\n\n"
                            f"Writing with: {eng['label']}\n{ctl.capacity()}\n\n"
                            "To update Gauge, double-click 'Update Gauge.bat'.")

    # ---- buttons: START under the idea box; the rest pinned to the bottom, two to a row
    start_btn = ttk.Button(start_row, text="START MY STORY", style="Big.TButton", command=on_start)
    start_btn.pack(fill="x", pady=(2, 0))
    write_btn = ttk.Button(write_row, text="WRITE THE NEXT CHAPTER", style="Big.TButton", command=on_write)
    write_btn.pack(fill="x")
    buttons.extend([start_btn, write_btn])
    pairs = [("Talk to a character", on_chat), ("Open another story", on_open),
             ("Teach Gauge", on_teach), ("Change the AI", on_engine),
             ("Switches", on_switches), ("Save story as one file", on_export)]
    for i, (text, cmd) in enumerate(pairs):
        b = ttk.Button(grid, text=text, style="Mid.TButton", command=cmd)
        b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        buttons.append(b)
    chapters_box.bind("<<ListboxSelect>>", on_pick)

    # ---- the menu bar: the same actions, always visible whatever the window size
    menubar = tk.Menu(root)
    for title, entries in (
        ("Story", [("Start my story", on_start), ("Write the next chapter", on_write), ("Open another story...", on_open),
                   ("Ask Gauge...", lambda: chat_entry.focus_set()),
                   ("Talk to a character...", on_chat), ("Save story as one file", on_export)]),
        ("Teach", [("Add books or your own writing...", on_teach), ("Teach character building...", on_characters),
                   ("Change the AI that writes...", on_engine),
                   ("Start training GaugeLM", lambda: start_training(None, UNTIL_PAUSED)),
                   ("Pause training", on_pause_training)]),
        ("Settings", [("Switches...", on_switches), ("Devices (processor and graphics cards)...", on_switches)]),
        ("Help", [("About and version", on_about)]),
    ):
        menu = tk.Menu(menubar, tearoff=0)
        for label, cmd in entries:
            menu.add_command(label=label, command=cmd)
        menubar.add_cascade(label=title, menu=menu)
    root.config(menu=menubar)

    def check_engine() -> None:
        def done(st: dict[str, Any]) -> None:
            show_engine()
            if st["projects"]:
                proj_var.set(f"You have {len(st['projects'])} stor{'y' if len(st['projects']) == 1 else 'ies'}. "
                             "Press 'Open another story' or start a new one.")

        run_bg(ctl.status, done, "Checking the story engine...")

    root.after(200, check_engine)
    root.mainloop()
