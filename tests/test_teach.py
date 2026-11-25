"""Teach Gauge: adding books or your own writing in the Studio, then learning from them."""
import sys
import threading
import types

import pytest

from gauge.gui import MATERIAL_EXTS, TEACH_INTRO, StudioController

PROSE = "\"Run,\" she said. The blade was in the door before I understood what she meant. "


def _novel_folder(tmp_path, name="My Novel", chapters=6):
    folder = tmp_path / name
    folder.mkdir()
    for i in range(1, chapters + 1):
        (folder / f"chapter_{i:02d}.txt").write_text(f"Chapter {i}\n\n" + PROSE * 60, encoding="utf-8")
    return folder


# ------------------------------------------------------------------ the controller
def test_add_material_reads_books_and_reports_what_it_could_not(cfg, db, tmp_path):
    ctl = StudioController(cfg, db)
    folder = _novel_folder(tmp_path)
    story = tmp_path / "Short Story.txt"
    story.write_text("A quiet story about a lighthouse keeper. " * 400, encoding="utf-8")
    picture = tmp_path / "photo.png"
    picture.write_bytes(b"\x89PNG")
    res = ctl.add_material([str(folder), str(story), str(picture), str(tmp_path / "missing.txt")], stars=4)
    assert res["books"] == 2 and res["chapters"] == 7 and res["new"] == 7
    assert sorted(res["titles"]) == ["My Novel", "Short Story"]
    assert len(res["problems"]) == 2                          # the picture and the missing file, and nothing else
    assert "photo.png" in res["message"] and "missing.txt" in res["message"]
    lib = ctl.library()
    assert [(r["title"], r["chapters"], r["stars"]) for r in lib] == [("My Novel", 6, 4.0), ("Short Story", 1, 4.0)]
    assert "6 chapters, yours" in ctl.library_line(lib[0]) and "4 stars" in ctl.library_line(lib[0])
    assert "1 chapter," in ctl.library_line(lib[1])           # singular


def test_adding_the_same_book_twice_does_not_duplicate_it(cfg, db, tmp_path):
    ctl = StudioController(cfg, db)
    folder = _novel_folder(tmp_path)
    assert ctl.add_material([str(folder)])["new"] == 6
    again = ctl.add_material([str(folder)])
    assert again["books"] == 1 and again["new"] == 0 and len(ctl.library()) == 1
    assert db.total_chapters() == 6


def test_stars_are_optional_and_checked(cfg, db, tmp_path):
    ctl = StudioController(cfg, db)
    folder = _novel_folder(tmp_path)
    ctl.add_material([str(folder)])
    assert ctl.library()[0]["stars"] is None
    with pytest.raises(ValueError):
        ctl.add_material([str(folder)], stars=9)


def test_learn_with_nothing_says_so(cfg, db):
    res = StudioController(cfg, db).learn()
    assert res["chapters"] == 0 and "Add a book" in res["message"]


def test_learning_from_a_small_library_still_teaches_the_writer(cfg, db, tmp_path):
    cfg.learn.min_chapters_for_training = 30                  # the shipped default; 6 chapters is well under it
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))], stars=5)
    ctl.pipeline()                                            # a pipeline built before learning...
    res = ctl.learn()
    assert res["chapters"] == 6 and res["books"] == 1
    assert res["examples"] > 0                                # examples are kept even below the full-cycle size
    assert res["techniques"] > 0                              # the (fake) story engine read the chapters
    assert res["model"] is False and "30 or more chapters" in res["message"]
    pipe = ctl.pipeline()                                     # ...is rebuilt, so the next chapter uses what was learned
    assert pipe.exemplars is not None and pipe.exemplars.items
    assert pipe.techniques.items


def test_learning_without_the_story_engine_explains_what_was_skipped(cfg, db, tmp_path, monkeypatch):
    from gauge.llm.client import FakeLLM
    monkeypatch.setattr(FakeLLM, "available", lambda self: False)
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    res = ctl.learn()
    assert res["examples"] > 0 and res["techniques"] == 0
    assert "story engine is off" in res["message"]


def test_teach_text_is_honest_about_what_learning_is():
    assert "does not rebuild the downloaded AI model" in TEACH_INTRO
    assert "right to use" in TEACH_INTRO and "stays on this computer" in TEACH_INTRO
    assert {".txt", ".docx", ".epub", ".md"} <= MATERIAL_EXTS


# ------------------------------------------------------------------ the window, against a stand-in toolkit
class _Var:
    def __init__(self, value=None, **_):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class _Widget:
    """Accepts anything, remembers buttons by their text, and lets a test press them."""
    buttons: dict = {}
    listboxes: list = []
    menu: dict = {}

    def __init__(self, *args, **kw):
        self.kw = kw
        self.items: list = []
        if "command" in kw and "text" in kw:
            _Widget.buttons[kw["text"]] = kw["command"]

    def add_command(self, label="", command=None, **_):
        _Widget.menu[label] = command

    def insert(self, _where, text=""):
        self.items.append(text)

    def delete(self, *_):
        self.items.clear()

    def get(self, *_):
        return "\n".join(str(i) for i in self.items)

    def curselection(self):
        return ()

    def winfo_exists(self):
        return True

    def after(self, _ms, fn):
        fn()

    def __getattr__(self, _name):
        return lambda *a, **k: None


class _Listbox(_Widget):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _Widget.listboxes.append(self)


def _fake_tkinter(picked_folder):
    tk = types.ModuleType("tkinter")
    tk.Tk = tk.Toplevel = tk.Canvas = tk.Menu = _Widget
    tk.Listbox = _Listbox
    tk.StringVar = tk.BooleanVar = tk.DoubleVar = _Var
    tk.TclError = type("TclError", (Exception,), {})
    ttk = types.ModuleType("tkinter.ttk")
    for name in ("Frame", "Label", "Button", "Progressbar", "Style", "Scrollbar", "Checkbutton", "Scale", "Combobox", "Entry"):
        setattr(ttk, name, _Widget)
    scrolled = types.ModuleType("tkinter.scrolledtext")
    scrolled.ScrolledText = _Widget
    shown: list = []
    box = types.ModuleType("tkinter.messagebox")
    box.showinfo = lambda *a, **k: shown.append(("info", a))
    box.showerror = lambda *a, **k: shown.append(("error", a))
    dialog = types.ModuleType("tkinter.simpledialog")
    dialog.askstring = lambda *a, **k: None
    files = types.ModuleType("tkinter.filedialog")
    files.askdirectory = lambda **k: str(picked_folder)
    files.askopenfilenames = lambda **k: ()
    files.asksaveasfilename = lambda **k: str(picked_folder.parent / "my-character.txt")
    mods = {"tkinter": tk, "tkinter.ttk": ttk, "tkinter.scrolledtext": scrolled, "tkinter.messagebox": box,
            "tkinter.simpledialog": dialog, "tkinter.filedialog": files}
    for name, mod in mods.items():
        if "." in name:
            setattr(tk, name.split(".")[1], mod)
    return mods, shown


def _wait_for_jobs():
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(timeout=60)


def test_the_window_code_runs_the_teach_flow(cfg, db, tmp_path, monkeypatch):
    """tkinter is not installed in CI, so this drives the real window code with a stand-in toolkit. It
    proves the handlers are wired and free of plain coding errors, not that the window looks right."""
    from gauge import gui

    folder = _novel_folder(tmp_path)
    mods, shown = _fake_tkinter(folder)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    assert {"START MY STORY", "WRITE THE NEXT CHAPTER", "Teach Gauge", "Change the AI", "Switches",
            "Save story as one file", "Talk to a character", "Open another story"} <= set(_Widget.buttons)
    # The same actions are in the menu bar, which no window size can hide.
    assert {"Add books or your own writing...", "Change the AI that writes...", "Switches...", "About and version",
            "Write the next chapter"} <= set(_Widget.menu)

    _Widget.buttons["Teach Gauge"]()                          # open the Teach Gauge window
    assert {"Add files...", "Add a folder...", "Learn from the library now"} <= set(_Widget.buttons)
    library_box = _Widget.listboxes[-1]
    assert "nothing yet" in library_box.get()

    _Widget.buttons["Add a folder..."]()                      # the file dialog stand-in returns our folder
    _wait_for_jobs()
    assert db.total_chapters() == 6
    assert "My Novel" in library_box.get() and "nothing yet" not in library_box.get()

    _Widget.buttons["Add files..."]()                         # nothing picked: nothing happens, nothing breaks
    _Widget.buttons["Learn from the library now"]()
    _wait_for_jobs()
    assert (cfg.models_dir / "exemplars.json").exists() or any(cfg.models_dir.glob("exemplar*"))
    assert not [s for s in shown if s[0] == "error"], shown   # no error box was raised anywhere in the flow

    _Widget.buttons["Switches"]()                             # the neighbouring button still opens its window


def test_the_window_code_runs_the_character_teaching_flow(cfg, db, tmp_path, monkeypatch):
    from gauge import gui
    from gauge.write import blueprint as bp

    folder = _novel_folder(tmp_path)
    mods, shown = _fake_tkinter(folder)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    assert "Teach character building..." in _Widget.menu
    _Widget.buttons["Teach Gauge"]()
    _Widget.buttons["Characters..."]()                        # opens from the Teach Gauge window
    assert {"What a character needs", "Save a blank sheet...", "Add character sheets...",
            "Learn from this story's characters"} <= set(_Widget.buttons)
    box = _Widget.listboxes[-1]
    assert "none yet" in box.get()
    _Widget.buttons["What a character needs"]()               # the breakdown window
    _Widget.buttons["Save a blank sheet..."]()                # the save dialog stand-in names a file
    blank = tmp_path / "my-character.txt"
    assert blank.read_text(encoding="utf-8").startswith(bp.SHEET_HEADER)
    _Widget.buttons["Add character sheets..."]()              # nothing picked: nothing happens
    _Widget.buttons["Learn from this story's characters"]()   # no story open yet: a sentence, not an error

    # With a filled sheet picked, it lands in the list.
    from tests.test_character_school import FILLED
    sheet = tmp_path / "marina.txt"
    sheet.write_text(FILLED, encoding="utf-8")
    mods["tkinter.filedialog"].askopenfilenames = lambda **k: (str(sheet),)
    _Widget.buttons["Add character sheets..."]()
    _wait_for_jobs()
    assert "Marina Vale" in box.get() and "none yet" not in box.get()
    assert not [s for s in shown if s[0] == "error"], shown


# ------------------------------------------------------------------ which AI writes, and training GaugeLM
def _small_gaugelm(cfg):
    """A GaugeLM small enough to train in a second, so the real training code runs in the tests."""
    n = cfg.native
    n.size, n.n_layer, n.n_head, n.n_kv_head, n.n_embd, n.block_size, n.vocab_size = "", 1, 2, 2, 32, 32, 400
    n.steps, n.batch_size, n.accum_steps, n.device = 12, 4, 1, "cpu"


def test_engine_says_which_model_writes(cfg, db):
    ctl = StudioController(cfg, db)
    eng = ctl.engine()
    assert not eng["is_gaugelm"] and eng["label"].startswith("the downloaded model")
    st = ctl.status()
    assert st["engine"] == eng["label"] and st["is_gaugelm"] is False and "llm_ok" in st


def test_switching_to_gaugelm_and_back_restores_everything(cfg, db):
    from gauge.config import GaugeConfig
    cfg.llm.provider, cfg.llm.model, cfg.llm.fast_model = "ollama", "qwen2.5:14b", "llama3.2:3b"
    cfg.agents.profile, cfg.write.target_chapter_words = "full", 2200
    ctl = StudioController(cfg, db)
    on = ctl.use_gaugelm(True)
    assert on["is_gaugelm"] and not on["ready"] and "not trained yet" in on["message"]
    saved = GaugeConfig.load(cfg.data_dir)                     # it is written to disk, not just held in memory
    assert (saved.llm.provider, saved.llm.model, saved.llm.fast_model) == ("native", "gaugelm", "")
    assert saved.agents.profile == "lean" and saved.write.target_chapter_words == 800
    assert ctl.use_gaugelm(True)["is_gaugelm"]                  # pressing it twice does not lose the remembered choice
    off = ctl.use_gaugelm(False)
    assert not off["is_gaugelm"] and "qwen2.5:14b" in off["message"]
    back = GaugeConfig.load(cfg.data_dir)
    assert (back.llm.provider, back.llm.model, back.llm.fast_model, back.llm.num_ctx) == ("ollama", "qwen2.5:14b", "llama3.2:3b", 8192)
    assert back.agents.profile == "full" and back.write.target_chapter_words == 2200


def test_untrained_gaugelm_refuses_to_write_with_a_clear_sentence(cfg, db):
    from gauge.gui import NOT_TRAINED
    ctl = StudioController(cfg, db)
    ctl.use_gaugelm(True)
    with pytest.raises(ValueError) as e:
        ctl.start_story("A courier who can rewind ten seconds")
    assert str(e.value) == NOT_TRAINED and "Start training" in NOT_TRAINED
    with pytest.raises(ValueError):
        ctl.write_next("Anything")


def test_training_gaugelm_from_the_studio_trains_resumes_and_makes_it_ready(cfg, db, tmp_path):
    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    assert "nothing to train on" in ctl.train_gaugelm(0.01)["message"]
    ctl.add_material([str(_novel_folder(tmp_path))])
    ctl.use_gaugelm(True)
    seen = []
    first = ctl.train_gaugelm(0.01, progress=seen.append)
    assert first["trained"] and first["step"] > 0 and not first["resumed"]
    assert first["lessons"] == 0                               # GaugeLM is the writer: there is no bigger model to teach it
    assert ctl.engine()["ready"] and f"trained {first['step']} steps" in ctl.engine()["label"]
    second = ctl.train_gaugelm(0.01)
    assert second["resumed"] and second["step"] > first["step"] and "carried on" in second["message"]


def test_the_downloaded_model_teaches_gaugelm_while_it_does_the_writing(cfg, db, tmp_path):
    # The answer to "how does GaugeLM learn if another model always writes": that model is its teacher.
    from gauge.native import instruction_rows
    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)                             # the (fake) downloaded model is the writer
    ctl.add_material([str(_novel_folder(tmp_path))])
    res = ctl.train_gaugelm(0.01)
    assert res["trained"] and res["lessons"] >= 4               # the writer answered Gauge's own tasks first
    assert res["examples"] >= res["lessons"] and "taught it" in res["message"]
    assert len(instruction_rows(cfg.models_dir / "datasets")) >= 4
    assert not ctl.engine()["is_gaugelm"]                       # and the writer is still the downloaded model
    quiet = ctl.train_gaugelm(0.01, teach=False)
    assert quiet["lessons"] == 0 and quiet["examples"] == res["examples"]   # the earlier lessons are still used


def test_gaugelm_can_train_after_every_chapter(cfg, db, chapter_text, tmp_path, monkeypatch):
    _small_gaugelm(cfg)
    cfg.write.target_chapter_words = 300
    cfg.write.max_revisions = 0
    cfg.agents.profile = "off"
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    name = ctl.start_story("A courier who can rewind ten seconds")
    assert ctl.write_next(name)["gaugelm"] == ""                # off by default: writing is untouched
    cfg.native.after_chapter_minutes = 1
    asked = []
    real = StudioController.train_gaugelm
    monkeypatch.setattr(StudioController, "train_gaugelm",                 # a minute is asked for; the test runs a moment
                        lambda self, minutes=None, progress=None, teach=True: asked.append((minutes, teach)) or real(self, 0.01, progress, teach))
    info = ctl.write_next(name)
    assert asked == [(1, False)]                                 # one minute, and no teacher: the chapter is the lesson
    assert info["number"] == 2 and "GaugeLM trained" in info["gaugelm"]
    assert ctl.engine()["is_gaugelm"] is False                  # the downloaded model still writes
    from gauge.native import NativeTrainer
    assert NativeTrainer(cfg.native_dir).exists()               # but GaugeLM now exists and has learned from it


def test_train_after_chapter_is_a_switch():
    from gauge import settings as st
    sw = st.BY_KEY["native.after_chapter_minutes"]
    assert sw.kind == "int" and "keeps learning" in sw.description


def test_the_window_can_switch_and_train_gaugelm(cfg, db, tmp_path, monkeypatch):
    from gauge import gui
    from gauge.config import GaugeConfig
    _small_gaugelm(cfg)
    mods, shown = _fake_tkinter(_novel_folder(tmp_path))
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    _Widget.buttons["Teach Gauge"]()
    _Widget.buttons["Add a folder..."]()
    _wait_for_jobs()
    _Widget.menu["Change the AI that writes..."]()             # opened from the menu bar this time
    assert {"Use GaugeLM", "Use the downloaded model", "Start training", "Pause training"} <= set(_Widget.buttons)
    assert {"Start training GaugeLM", "Pause training"} <= set(_Widget.menu)
    _Widget.buttons["Pause training"]()                         # nothing running: a sentence, not an error
    _Widget.buttons["Use GaugeLM"]()
    assert GaugeConfig.load(cfg.data_dir).llm.provider == "native"
    from gauge.gui import StudioController as _SC
    real_train = _SC.train_gaugelm
    monkeypatch.setattr(_SC, "train_gaugelm", lambda self, minutes=None, progress=None, teach=True: real_train(self, 0.01, progress, teach))
    _Widget.buttons["Start training"]()                         # "until I pause" in the window; bounded here so the test ends
    _wait_for_jobs()
    from gauge.native import NativeTrainer
    assert NativeTrainer(cfg.native_dir).exists()
    _Widget.buttons["Use the downloaded model"]()
    assert GaugeConfig.load(cfg.data_dir).llm.provider != "native"
    assert not [s for s in shown if s[0] == "error"], shown


# ------------------------------------------------------------------ the window can never hide its buttons
def test_buttons_are_placed_before_the_chapter_list_so_a_short_window_cannot_cut_them_off():
    # The first Studio put Teach Gauge and Switches in the last row of a column taller than the window, where
    # they were simply not drawn. pack() gives space in the order widgets are packed, so the bottom block must
    # be packed before the list that is allowed to shrink.
    import inspect

    from gauge import gui
    src = inspect.getsource(gui.run_studio)
    bottom = src.index('bottom.pack(side="bottom", fill="x")')
    chapters = src.index('chapters_box.pack(fill="both", expand=True')
    assert bottom < chapters
    assert 'root.geometry(_fit(root, 1100, 860))' in src and "root.config(menu=menubar)" in src
    assert 'geometry("1000x720")' not in src                     # no fixed size that ignores the screen


def test_window_sizes_fit_the_screen():
    from gauge.gui import _fit

    class Screen:
        def __init__(self, w, h):
            self.w, self.h = w, h

        def winfo_screenwidth(self):
            return self.w

        def winfo_screenheight(self):
            return self.h

    assert _fit(Screen(1920, 1080), 1100, 860) == "1100x860"     # room to spare: the wanted size
    assert _fit(Screen(1366, 768), 1100, 860) == "1100x648"      # a small laptop: shorter, inside the screen
    assert _fit(Screen(1024, 600), 1100, 860) == "944x480"
    assert _fit(object(), 760, 600) == "760x600"                 # a display that will not answer: a sane default


def test_studio_reports_version_and_capacity(cfg, db, monkeypatch):
    from gauge import limit
    ctl = StudioController(cfg, db)
    st = ctl.status()
    assert "capacity" in st and "version" in st
    monkeypatch.delenv(limit.ENV, raising=False)
    assert "20%" in ctl.capacity()                               # the default limit is shown on the main window
    cfg.compute.capacity_percent = 100
    assert "No capacity limit" in ctl.capacity()


# ------------------------------------------------------------------ start and pause training
def test_the_trainer_stops_when_asked_and_keeps_its_progress(cfg, tmp_path):
    from gauge.native import ModelConfig, NativeTrainer, TrainConfig
    docs = [("prose", PROSE * 80)]
    tr = NativeTrainer(tmp_path / "native")
    asked = {"n": 0}

    def stop_after_three():
        asked["n"] += 1
        return asked["n"] >= 3

    res = tr.train(docs, ModelConfig(vocab_size=300, block_size=32, n_layer=1, n_head=2, n_embd=32),
                   TrainConfig(steps=500, batch_size=4, device="cpu"), should_stop=stop_after_three)
    assert res["stopped"] is True and res["step"] == 3            # stopped at the step it was on, not 500 later
    assert tr.exists() and tr.status()["step"] == 3              # and saved, so nothing is lost
    more = tr.train(docs, ModelConfig(vocab_size=300, block_size=32, n_layer=1, n_head=2, n_embd=32),
                    TrainConfig(steps=4, batch_size=4, device="cpu"), resume=True)
    assert more["step"] == 7 and more["stopped"] is False         # carries on from step 3


def test_training_runs_until_paused_and_then_carries_on(cfg, db, tmp_path):
    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    assert ctl.pause_training()["message"] == "Nothing is training right now."
    assert ctl.training()["running"] is False
    passes = []

    def progress(rec):
        passes.append(rec["step"])
        assert ctl.training()["running"] is True
        if len(passes) == 2:                                      # let it run two passes, then press Pause
            assert ctl.pause_training()["running"] is True

    first = ctl.train_gaugelm(None, progress=progress, teach=False)   # None = until I pause
    assert first["trained"] and first["paused"] is True and "Training paused" in first["message"]
    assert first["step"] == passes[-1] and len(passes) == 2       # it stopped where Pause was pressed
    assert ctl.training() == {"running": False, "step": first["step"], "loss": first["loss"], "pause_requested": False}
    again = ctl.train_gaugelm(0.01, teach=False)                  # Start again: carries on, and this time runs out the clock
    assert again["resumed"] and again["step"] > first["step"] and again["paused"] is False


def test_pause_is_noticed_while_the_downloaded_model_is_teaching(cfg, db, tmp_path, monkeypatch):
    from gauge.native import distill as distill_mod
    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    real = distill_mod.distill
    batches = []

    def teaching(db_, teacher, out, rows, **kw):
        batches.append(rows)
        ctl._training["running"] = True
        ctl.pause_training()                                      # Pause pressed during the very first lessons
        return real(db_, teacher, out, rows, **kw)

    monkeypatch.setattr(distill_mod, "distill", teaching)
    res = ctl.train_gaugelm(None)
    assert batches == [2]                                         # lessons come two at a time, so it stopped after one batch
    assert res["paused"] is True and res["trained"] is False and "Paused before training began" in res["message"]
