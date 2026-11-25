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

    def __init__(self, *args, **kw):
        self.kw = kw
        self.items: list = []
        if "command" in kw and "text" in kw:
            _Widget.buttons[kw["text"]] = kw["command"]

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
    tk.Tk = tk.Toplevel = tk.Canvas = _Widget
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
    _Widget.buttons, _Widget.listboxes = {}, []
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    assert {"START MY STORY", "WRITE THE NEXT CHAPTER", "Teach Gauge", "Switches"} <= set(_Widget.buttons)

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
    assert str(e.value) == NOT_TRAINED and "Train GaugeLM" in NOT_TRAINED
    with pytest.raises(ValueError):
        ctl.write_next("Anything")


def test_training_gaugelm_from_the_studio_trains_resumes_and_makes_it_ready(cfg, db, tmp_path):
    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    assert "nothing to train on" in ctl.train_gaugelm(0.05)["message"]
    ctl.add_material([str(_novel_folder(tmp_path))])
    ctl.use_gaugelm(True)
    seen = []
    first = ctl.train_gaugelm(0.05, progress=seen.append)
    assert first["trained"] and first["step"] > 0 and not first["resumed"]
    assert first["lessons"] == 0                               # GaugeLM is the writer: there is no bigger model to teach it
    assert ctl.engine()["ready"] and f"trained {first['step']} steps" in ctl.engine()["label"]
    second = ctl.train_gaugelm(0.05)
    assert second["resumed"] and second["step"] > first["step"] and "carried on" in second["message"]


def test_the_downloaded_model_teaches_gaugelm_while_it_does_the_writing(cfg, db, tmp_path):
    # The answer to "how does GaugeLM learn if another model always writes": that model is its teacher.
    from gauge.native import instruction_rows
    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)                             # the (fake) downloaded model is the writer
    ctl.add_material([str(_novel_folder(tmp_path))])
    res = ctl.train_gaugelm(0.05)
    assert res["trained"] and res["lessons"] >= 4               # the writer answered Gauge's own tasks first
    assert res["examples"] >= res["lessons"] and "taught it" in res["message"]
    assert len(instruction_rows(cfg.models_dir / "datasets")) >= 4
    assert not ctl.engine()["is_gaugelm"]                       # and the writer is still the downloaded model
    quiet = ctl.train_gaugelm(0.05, teach=False)
    assert quiet["lessons"] == 0 and quiet["examples"] == res["examples"]   # the earlier lessons are still used


def test_gaugelm_can_train_after_every_chapter(cfg, db, chapter_text, tmp_path):
    _small_gaugelm(cfg)
    cfg.write.target_chapter_words = 300
    cfg.write.max_revisions = 0
    cfg.agents.profile = "off"
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    name = ctl.start_story("A courier who can rewind ten seconds")
    assert ctl.write_next(name)["gaugelm"] == ""                # off by default: writing is untouched
    cfg.native.after_chapter_minutes = 1
    info = ctl.write_next(name)
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
    _Widget.buttons, _Widget.listboxes = {}, []
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    _Widget.buttons["Teach Gauge"]()
    assert {"Use GaugeLM", "Use the downloaded model", "Train GaugeLM"} <= set(_Widget.buttons)
    _Widget.buttons["Add a folder..."]()
    _wait_for_jobs()
    _Widget.buttons["Use GaugeLM"]()
    assert GaugeConfig.load(cfg.data_dir).llm.provider == "native"
    _Widget.buttons["Train GaugeLM"]()
    _wait_for_jobs()
    from gauge.native import NativeTrainer
    assert NativeTrainer(cfg.native_dir).exists()
    _Widget.buttons["Use the downloaded model"]()
    assert GaugeConfig.load(cfg.data_dir).llm.provider != "native"
    assert not [s for s in shown if s[0] == "error"], shown
