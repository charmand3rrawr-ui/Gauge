"""The chat prompt in Gauge Studio: ask Gauge about the story, or say what should happen next."""
import sys

import pytest

from gauge.gui import StudioController
from gauge.llm import FakeLLM
from gauge.write.project import Project


class _Recorder(FakeLLM):
    def __init__(self):
        super().__init__()
        self.seen: list[tuple[str, str, str]] = []

    def generate(self, prompt, system="", **kw):
        self.seen.append((kw.get("purpose", ""), prompt, system))
        return super().generate(prompt, system=system, **kw)


def _studio(cfg, db):
    ctl = StudioController(cfg, db)
    rec = _Recorder()
    pipe = ctl.pipeline()
    pipe.llm = pipe.fast_llm = rec
    for part in ("planner", "drafter"):
        getattr(pipe, part).llm = rec
    return ctl, rec


def _prompts(rec, purpose):
    return [p for kind, p, _s in rec.seen if kind == purpose]


def test_asking_without_a_story_still_answers(cfg, db):
    ctl, rec = _studio(cfg, db)
    assert ctl.ask("   ")["message"] == "Type a question first." and not rec.seen
    res = ctl.ask("How do I make a villain frightening?")
    assert res["answer"].startswith("A thought about writing: How do I make a villain frightening?")
    kind, prompt, system = rec.seen[-1]
    assert kind == "studio:chat" and "You are Gauge" in system and "STORY:" not in prompt


def test_asking_with_a_story_open_knows_the_story_and_remembers_the_conversation(cfg, db):
    ctl, rec = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    ctl.write_next(name)
    res = ctl.ask("What is Rin still hiding?", name)
    assert res["answer"].startswith("A thought about your story")
    first = _prompts(rec, "studio:chat")[-1]
    for piece in ("STORY: Ten Seconds", "CHAPTERS WRITTEN: 1", "WHAT HAS HAPPENED SO FAR:", "THE LAST CHAPTER ENDED WITH:", "CAST:"):
        assert piece in first, piece
    assert first.rstrip().endswith("Answer the author.") and "The author says: What is Rin still hiding?" in first
    ctl.ask("And who knows?", name)
    second = _prompts(rec, "studio:chat")[-1]
    assert "The author said: What is Rin still hiding?" in second and "You answered: A thought about your story" in second
    # Each story has its own conversation; clearing forgets it; asking changes nothing in the story.
    ctl.ask("Unrelated", None)
    assert "What is Rin still hiding?" not in _prompts(rec, "studio:chat")[-1]
    ctl.clear_chat(name)
    ctl.ask("Fresh start?", name)
    assert "The author said:" not in _prompts(rec, "studio:chat")[-1]
    assert Project(cfg.projects_dir, name).state.written == 1 and not ctl.direction(name)


def test_a_long_conversation_and_bible_still_fit_a_small_model(cfg, db):
    ctl, rec = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    rec.context_window = lambda: 1200
    for i in range(12):
        ctl.ask(f"Question number {i} " + "and more words " * 40, name)
    from gauge.llm.tokens import estimate_tokens
    last = _prompts(rec, "studio:chat")[-1]
    assert estimate_tokens(last) < 1200 and "Question number 11" in last and "Question number 2 " not in last


def test_saying_what_happens_next_steers_the_next_chapter_once(cfg, db):
    ctl, rec = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    ctl.write_next(name)
    assert Project(cfg.projects_dir, name).plan_for(2) is not None       # chapter 2 was already planned
    asked = "Jun finds the bag and opens it before Rin can stop her"
    res = ctl.direct_next(name, "  " + asked + "  ")
    assert res["set"] and res["chapter"] == 2 and asked in res["message"] and "WRITE THE NEXT CHAPTER" in res["message"]
    assert ctl.direction(name) == asked
    assert Project(cfg.projects_dir, name).plan_for(2) is None           # so it is planned again, around the instruction
    assert "THE AUTHOR ALREADY ASKED FOR THIS" in (ctl.ask("Good idea?", name), _prompts(rec, "studio:chat")[-1])[1]

    rec.seen.clear()
    info = ctl.write_next(name)
    assert info["number"] == 2
    outline, draft = _prompts(rec, "outline"), _prompts(rec, "draft")
    assert outline and f"THE AUTHOR'S INSTRUCTION for chapter 2 (follow it; it outranks everything else): {asked}" in outline[0]
    assert draft and f"THE AUTHOR ASKED FOR THIS IN THIS CHAPTER (do it; it outranks the plan and every other note): {asked}" in draft[0]
    p = Project(cfg.projects_dir, name)
    assert p.state.author_direction == "" and ctl.direction(name) == ""  # used once, then forgotten
    assert p.read_meta(2)["author_direction"] == asked                   # the chapter records what it was asked to do

    rec.seen.clear()
    ctl.write_next(name)                                                 # chapter 3 is Gauge's own again
    assert not any("THE AUTHOR ASKED FOR THIS" in d for d in _prompts(rec, "draft"))


def test_an_instruction_can_be_taken_back_and_needs_a_story(cfg, db):
    ctl, _rec = _studio(cfg, db)
    assert ctl.direct_next(None, "anything")["message"] == "Start or open a story first."
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    assert "Type what should happen" in ctl.direct_next(name, "   ")["message"]
    ctl.direct_next(name, "The Broker arrives early")
    back = ctl.direct_next(name, "")
    assert not back["set"] and back["message"].startswith("Taken back.") and ctl.direction(name) == ""


def test_an_untrained_gaugelm_says_so_instead_of_answering(cfg, db):
    from gauge.gui import NOT_TRAINED

    ctl = StudioController(cfg, db)
    ctl.use_gaugelm(True)
    with pytest.raises(ValueError) as e:
        ctl.ask("Anything?")
    assert str(e.value) == NOT_TRAINED


def test_the_window_has_a_working_chat_prompt(cfg, db, tmp_path, monkeypatch):
    from gauge import gui
    from tests.test_teach import _Entry, _fake_tkinter, _wait_for_jobs, _Widget

    mods, shown = _fake_tkinter(tmp_path)
    made: list = []

    class _Text(_Widget):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    mods["tkinter.scrolledtext"].ScrolledText = _Text
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Widget.buttons, _Widget.listboxes, _Widget.menu = {}, [], {}
    gui.run_studio(cfg, db)
    _wait_for_jobs()
    assert {"Ask Gauge", "Use for the next chapter"} <= set(_Widget.buttons) and "Ask Gauge..." in _Widget.menu
    box = _Entry.made[0].kw["textvariable"]
    log = next(t for t in made if "press Ask Gauge" in t.get())

    _Widget.buttons["Ask Gauge"]()                             # an empty box does nothing
    box.set("How do I open a chapter?")
    _Widget.buttons["Ask Gauge"]()
    _wait_for_jobs()
    assert "You: How do I open a chapter?" in log.get() and "Gauge: A thought about writing" in log.get()
    assert box.get() == ""                                      # the box is cleared for the next question

    box.set("The dragon hatches")
    _Widget.buttons["Use for the next chapter"]()              # no story yet: a message, not an error
    assert shown[-1][0] == "info" and "Start a story first" in shown[-1][1][1]
    _Widget.menu["Ask Gauge..."]()
    assert not [s for s in shown if s[0] == "error"], shown


def test_the_chat_prompt_is_placed_before_the_reader_so_a_short_window_cannot_cut_it_off():
    import inspect

    from gauge import gui

    src = inspect.getsource(gui.run_studio)
    assert src.index('chat.pack(side="bottom"') < src.index("reader.pack(") and src.index("use_btn.pack(") < src.index("reader.pack(")
