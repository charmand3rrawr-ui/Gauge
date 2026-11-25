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
