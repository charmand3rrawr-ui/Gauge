"""Gauge Studio: a window with big buttons, for people who do not want a terminal.

Built on tkinter (ships with Python on Windows) so the installer needs nothing extra. All the
real work happens in ``StudioController``, which has no UI in it and is what the tests cover;
the window just calls it from a background thread so it never freezes.
"""
from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from .config import GaugeConfig
from .db import Database

log = logging.getLogger(__name__)


class StudioController:
    """Everything the window can do, as plain methods returning plain data."""

    def __init__(self, cfg: GaugeConfig, db: Database):
        self.cfg = cfg
        self.db = db
        self._pipe = None

    def status(self) -> dict[str, Any]:
        from .llm import make_llm
        from .write.project import Project

        llm = make_llm(self.cfg.llm)
        return {"llm_ok": llm.available(), "llm": llm.describe(), "model": self.cfg.llm.model,
                "projects": Project.list(self.cfg.projects_dir)}

    def pipeline(self):
        if self._pipe is None:
            from .write import WritePipeline

            self._pipe = WritePipeline(self.cfg, self.db)
        return self._pipe

    def start_story(self, idea: str, name: Optional[str] = None, genres: Optional[list[str]] = None) -> str:
        idea = idea.strip()
        if not idea:
            raise ValueError("Tell me your story idea first.")
        from .write.project import Project, title_from_text

        title = name or title_from_text(idea)

        base, k = title, 2
        while Project.exists(self.cfg.projects_dir, title):
            title = f"{base} {k}"
            k += 1
        long = len(idea) >= 300
        p = self.pipeline().new_project(title, "" if long else idea, genres or ["Adventure"], premise_doc=idea if long else "")
        return p.state.name

    def write_next(self, name: str) -> dict[str, Any]:
        pipe = self.pipeline()
        res = pipe.write_next(pipe.open_project(name))
        return {"number": res.number, "title": res.title, "score": round(res.score, 2), "path": str(res.path),
                "seconds": round(res.timings.get("total", 0.0))}

    def chapters(self, name: str) -> list[tuple[int, str]]:
        from .write.project import Project

        p = Project(self.cfg.projects_dir, name)
        out = []
        for n in range(1, p.state.written + 1):
            plan = p.plan_for(n)
            out.append((n, plan.title if plan else f"Chapter {n}"))
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

    # ---- switches ------------------------------------------------------------------
    def switches(self) -> list[dict[str, Any]]:
        from . import settings as st

        return st.state(self.cfg)

    def set_switch(self, key: str, value: Any) -> Any:
        from . import settings as st

        v = st.set_value(self.cfg, key, value)
        self._pipe = None  # the next job picks up the new settings
        return v

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


def run_studio(cfg: GaugeConfig, db: Database) -> None:  # pragma: no cover - needs a display
    try:
        import tkinter as tk
        from tkinter import messagebox, scrolledtext, simpledialog, ttk
    except ImportError as e:
        raise SystemExit("tkinter is not available in this Python. Install Python from python.org (it includes it).") from e

    ctl = StudioController(cfg, db)
    root = tk.Tk()
    root.title("Gauge Studio")
    root.geometry("1000x720")
    root.minsize(820, 600)
    big = ("Segoe UI", 14)
    huge = ("Segoe UI", 20, "bold")
    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except tk.TclError:
        pass
    style.configure("Big.TButton", font=big, padding=10)
    style.configure("TLabel", font=big)

    state: dict[str, Any] = {"project": None, "busy": False}
    left = ttk.Frame(root, padding=16)
    left.pack(side="left", fill="y")
    right = ttk.Frame(root, padding=16)
    right.pack(side="right", fill="both", expand=True)

    status_var = tk.StringVar(value="Checking the story engine...")
    ttk.Label(left, textvariable=status_var, wraplength=360).pack(anchor="w", pady=(0, 12))
    ttk.Label(left, text="1. What is your story about?", font=huge).pack(anchor="w")
    idea = scrolledtext.ScrolledText(left, width=40, height=7, font=big, wrap="word")
    idea.pack(pady=6)
    idea.insert("1.0", "A kid who finds a dragon egg in the school lost-and-found...")
    proj_var = tk.StringVar(value="No story yet")
    ttk.Label(left, textvariable=proj_var, font=("Segoe UI", 12, "italic")).pack(anchor="w", pady=(10, 0))
    progress = ttk.Progressbar(left, mode="indeterminate", length=360)
    progress.pack(pady=8)
    chapters_box = tk.Listbox(left, font=big, height=8)
    chapters_box.pack(fill="x", pady=6)
    reader = scrolledtext.ScrolledText(right, font=("Georgia", 14), wrap="word", padx=16, pady=16)
    reader.pack(fill="both", expand=True)
    reader.insert("1.0", "Your chapters will appear here.\n\n1. Type your idea on the left.\n2. Press START MY STORY.\n"
                         "3. Press WRITE THE NEXT CHAPTER whenever you want more.\n4. Click a chapter to read it.")
    buttons: list[Any] = []

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

    def refresh_chapters() -> None:
        chapters_box.delete(0, "end")
        if state["project"]:
            for n, title in ctl.chapters(state["project"]):
                chapters_box.insert("end", f"Chapter {n}: {title}")

    def show_chapter(n: int) -> None:
        reader.delete("1.0", "end")
        reader.insert("1.0", ctl.read(state["project"], n))

    def on_start() -> None:
        text = idea.get("1.0", "end").strip()

        def done(name: str) -> None:
            state["project"] = name
            proj_var.set(f"Story: {name}")
            refresh_chapters()
            reader.delete("1.0", "end")
            reader.insert("1.0", f"'{name}' is ready!\n\nPress WRITE THE NEXT CHAPTER to begin.")

        run_bg(lambda: ctl.start_story(text), done, "Dreaming up your world and characters... (a minute or two)")

    def on_write() -> None:
        if not state["project"]:
            messagebox.showinfo("Gauge", "Start a story first!")
            return

        def done(info: dict[str, Any]) -> None:
            refresh_chapters()
            show_chapter(info["number"])
            status_var.set(f"Chapter {info['number']} done in {info['seconds']}s (score {info['score']}).")

        run_bg(lambda: ctl.write_next(state["project"]), done, "Writing the next chapter... this takes a few minutes.")

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

    def on_switches() -> None:
        """A window of every switch, grouped, built from the catalog. Changes save immediately."""
        win = tk.Toplevel(root)
        win.title("Gauge Switches")
        win.geometry("760x680")
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
            except Exception as e:
                status.set(f"Could not set {label}: {e}")

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
                ttk.Label(frame, text=r["description"], wraplength=680, font=("Segoe UI", 10)).pack(anchor="w")
                if r["warning"]:
                    ttk.Label(frame, text="! " + r["warning"], wraplength=680, foreground="#b00020", font=("Segoe UI", 10, "bold")).pack(anchor="w")

    buttons.extend([
        ttk.Button(left, text="START MY STORY", style="Big.TButton", command=on_start),
        ttk.Button(left, text="WRITE THE NEXT CHAPTER", style="Big.TButton", command=on_write),
        ttk.Button(left, text="Talk to a character", style="Big.TButton", command=on_chat),
        ttk.Button(left, text="Open another story", style="Big.TButton", command=on_open),
        ttk.Button(left, text="Save story as one file", style="Big.TButton", command=on_export),
        ttk.Button(left, text="Switches", style="Big.TButton", command=on_switches),
    ])
    for b in buttons:
        b.pack(fill="x", pady=4)
    chapters_box.bind("<<ListboxSelect>>", on_pick)

    def check_engine() -> None:
        def done(st: dict[str, Any]) -> None:
            if st["llm_ok"]:
                status_var.set(f"Story engine ready ({st['model']}).")
            else:
                status_var.set("Story engine is OFF. Open the Ollama app (the llama icon), then restart Gauge Studio.")
            if st["projects"]:
                proj_var.set(f"You have {len(st['projects'])} stor{'y' if len(st['projects']) == 1 else 'ies'}. "
                             "Press 'Open another story' or start a new one.")

        run_bg(ctl.status, done, "Checking the story engine...")

    root.after(200, check_engine)
    root.mainloop()
