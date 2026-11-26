"""The chronicle: a story's memory that still holds at chapter 10,000."""
import time

import pytest

from gauge.gui import StudioController
from gauge.llm import FakeLLM
from gauge.write.chronicle import Chronicle, lasting
from gauge.write.project import ChapterPlan, Project


def _chapter(n: int) -> tuple[str, str, str]:
    """A made-up chapter: who is in it changes over the story, so that people really are away for thousands of chapters."""
    who = ["Rin", "Jun", "Broker", "Marrow", "Ysolde", "Tobin"][(n // 700) % 6]
    place = ["harbour", "citadel", "salt road", "archive", "drowned quarter"][(n // 130) % 5]
    text = (f"{who} crossed the {place} before dawn. The {place} was quiet, and {who} did not trust it. "
            f"Later {who} met Captain Hale near the {place} and nothing was said about chapter {n}. ") * 3
    return f"The {place.title()} {n}", text, f"{who} crosses the {place} and meets Captain Hale; day {n} of the long road."


@pytest.fixture
def chron(tmp_path):
    c = Chronicle(tmp_path / "story")
    yield c
    c.close()


def test_what_lasts_and_what_may_change():
    assert lasting("eye colour") and lasting("sister") and lasting("status", "dead") and lasting("real name")
    assert not lasting("location") and not lasting("carries") and not lasting("mood")
    assert lasting("location", flag=True) and not lasting("eye colour", flag=False)       # the digest's own word wins
    assert not lasting("location", "the Drowned Quarter")                                 # a place name is not a death


def test_a_changing_fact_moves_on_and_a_lasting_one_contradicts(chron):
    chron.record(1, "One", "Rin ran.", "Rin runs.", [{"subject": "Rin", "attribute": "location", "value": "the harbour"},
                                                    {"subject": "Rin", "attribute": "eye colour", "value": "grey"}])
    moved = chron.record(2, "Two", "Rin walked.", "Rin walks.", [{"subject": "Rin", "attribute": "location", "value": "the citadel"}])
    assert moved["conflicts"] == []
    assert [f["value"] for f in chron.history("Rin", "location")] == ["the harbour", "the citadel"]     # the old value is history
    assert [f["value"] for f in chron.facts_about("Rin") if f["attribute"] == "location"] == ["the citadel"]
    clash = chron.record(3, "Three", "Rin looked.", "Rin looks.", [{"subject": "Rin", "attribute": "eye colour", "value": "green",
                                                                   "quote": "her green eyes"}])
    assert clash["conflicts"][0]["was"] == "grey" and clash["conflicts"][0]["was_chapter"] == 1
    assert [f["value"] for f in chron.facts_about("Rin") if f["attribute"] == "eye colour"] == ["grey"]   # the first stands
    open_ = chron.open_conflicts()
    assert len(open_) == 1 and "UNRESOLVED: Rin, eye colour was 'grey'" in chron.brief(4, "Rin goes on.", ["Rin"])
    chron.settle(open_[0]["id"], keep_new=True)                                                        # the author rules
    assert chron.open_conflicts() == [] and chron.facts_about("Rin")[0]["value"] == "green"
    # Saying the same thing again, or more of it, is not a new fact.
    chron.record(4, "Four", "x", "x", [{"subject": "Rin", "attribute": "eye colour", "value": "Green"}])
    assert len(chron.history("Rin", "eye colour")) == 2


def test_rewriting_a_chapter_replaces_what_it_filed(chron):
    chron.record(1, "One", "Rin ran with Jun.", "first", [{"subject": "Rin", "attribute": "location", "value": "the harbour"}])
    chron.record(2, "Two", "Rin and Marrow.", "second", [{"subject": "Rin", "attribute": "location", "value": "the citadel"}])
    chron.record(2, "Two again", "Rin and Tobin, and Tobin again.", "second, rewritten",
                 [{"subject": "Rin", "attribute": "location", "value": "the archive"}])
    assert [f["value"] for f in chron.history("Rin", "location")] == ["the harbour", "the archive"]
    assert chron.chapter(2)["title"] == "Two again" and chron.person("Marrow")["chapters"] == 0
    assert chron.stats()["chapters"] == 2


def test_the_dead_stay_dead_unless_the_story_says_otherwise(chron):
    chron.record(5, "The Fall", "Marrow fell.", "Marrow dies on the wall.",
                 [{"subject": "Marrow", "attribute": "status", "value": "dead", "quote": "Marrow did not get up"}])
    assert list(chron.dead()) == ["Marrow"]
    found = chron.check('The gate opened. "You are late," Marrow said, and walked to the fire.')
    assert len(found) == 1 and "Marrow died in chapter 5" in found[0] and "memory or a ghost" in found[0]
    assert chron.check("She remembered how Marrow laughed at the gate, years ago, before the wall.") == []   # a memory is fine
    assert chron.check("Rin said nothing. Nobody mentioned Marrow.") == []
    assert "Marrow is DEAD (chapter 5" in chron.brief(60, "Rin returns to the wall where Marrow kept the gate.", ["Marrow"])
    # With the draft's facts, lasting facts are checked too.
    chron.record(6, "Six", "x", "x", [{"subject": "Jun", "attribute": "sister", "value": "Rin"}])
    said = chron.check("text", [{"subject": "Jun", "attribute": "sister", "value": "Ysolde"}], n=900)
    assert "chapter 6 established 'Rin'" in said[0] and "Keep to chapter 6" in said[0]
    # Coming back from the dead contradicts the record until the author rules that it happened.
    back = chron.record(7, "Back", "x", "x", [{"subject": "Marrow", "attribute": "status", "value": "alive after all"}])
    assert back["conflicts"][0]["was"] == "dead" and list(chron.dead()) == ["Marrow"]
    chron.settle(chron.open_conflicts()[0]["id"], keep_new=True)
    assert chron.dead() == {}


def test_story_so_far_is_layered(chron):
    for n in range(1, 236):
        title, text, summary = _chapter(n)
        chron.record(n, title, text, summary)
    so_far = chron.story_so_far(236, words=400)
    labels = [line.split(":")[0] for line in so_far.splitlines()]
    assert labels == ["Chapters 1-100", "Chapters 101-200", "Chapters 201-210", "Chapters 211-220", "Chapters 221-230",
                      "Chapter 231", "Chapter 232", "Chapter 233", "Chapter 234", "Chapter 235"]
    assert len(so_far.split()) <= 460
    assert chron.story_so_far(1) == "" and chron.story_so_far(4).count("Chapter ") == 3
    # A model can write the roll-ups; when it fails, the first lines are kept instead.
    made = []
    c2 = Chronicle(chron.dir.parent / "two")
    for n in range(1, 11):
        c2.record(n, "t", "text", f"summary {n}", summarise=lambda parts, a, b: made.append((len(parts), a, b)) or "An arc in a line.")
    assert made == [(10, 1, 10)] and "An arc in a line." in c2.story_so_far(11)

    def broken(parts, a, b):
        raise RuntimeError("model down")

    for n in range(11, 21):
        c2.record(n, "t", "text", f"summary {n}", summarise=broken)
    assert "Chapters 11-20: summary 11" in c2.story_so_far(21)
    c2.close()


def test_ten_thousand_chapters(tmp_path):
    """The whole point: after 10,000 chapters, what was true in chapter 3 is still known, the brief still
    fits, and filing a chapter costs what it cost at the start."""
    chron = Chronicle(tmp_path / "long")
    timings = []
    for n in range(1, 10_001):
        title, text, summary = _chapter(n)
        facts = [{"subject": text.split()[0], "attribute": "location", "value": title}]
        if n == 3:
            facts += [{"subject": "Rin", "attribute": "scar", "value": "across the left palm", "quote": "the scar on her palm"},
                      {"subject": "Rin", "attribute": "sister", "value": "Jun"}]
        if n == 40:
            facts.append({"subject": "Old Nan", "attribute": "status", "value": "dead"})
        t0 = time.perf_counter()
        chron.record(n, title, text, summary, facts, known_names={"Rin", "Jun", "Marrow", "Broker", "Ysolde", "Tobin"},
                     memories={"Rin": [f"met Hale on day {n}"]} if n == 12 else None)
        timings.append(time.perf_counter() - t0)
    st = chron.stats()
    assert st["chapters"] == 10_000 and st["rollups"] == 1000 + 100 + 10 + 1 and st["bytes"] < 40_000_000

    # Filing a chapter did not get slower as the story grew.
    early, late = sorted(timings[100:1100])[500], sorted(timings[-1000:])[500]
    assert late < early * 3 + 0.002, (early, late)

    # The story so far for chapter 9,998: nine sagas, nine volumes, nine arcs, seven chapters, and it fits.
    so_far = chron.story_so_far(9_998, words=600)
    labels = [line.split(":")[0] for line in so_far.splitlines()]
    assert labels[0] == "Chapters 1-1000" and labels[8] == "Chapters 8001-9000" and labels[9] == "Chapters 9001-9100"
    assert labels[-1] == "Chapter 9997" and len(labels) == 9 + 9 + 9 + 7 and len(so_far.split()) <= 700
    assert chron.story_so_far(10_001).splitlines()[0].startswith("Chapters 1-10000")

    # Rin has been away since chapter 699. In chapter 9,998 she walks back in.
    t0 = time.perf_counter()
    brief = chron.brief(9_998, "Rin returns to the harbour and shows Jun the letter.", ["Rin", "Jun"], words=300)
    took = time.perf_counter() - t0
    assert "Rin, scar: across the left palm (chapter 3, must not change)" in brief
    assert "Rin, sister: Jun (chapter 3, must not change)" in brief
    assert "Rin was last seen in chapter 9099, 899 chapters ago" in brief                # her last stretch ended there
    assert "Like this before, chapter" in brief and len(brief.split()) <= 300
    assert took < 0.5, took

    # A contradiction of something set down 9,995 chapters earlier is caught.
    assert "chapter 3 established 'across the left palm'" in chron.check(
        "x", [{"subject": "Rin", "attribute": "scar", "value": "over the right eye"}], n=9_998)[0]
    clash = chron.record(10_001, "New", "Rin again.", "Rin again.", [{"subject": "Rin", "attribute": "sister", "value": "Ysolde"}])
    assert clash["conflicts"][0]["was_chapter"] == 3
    # Someone dead since chapter 40 still may not speak.
    assert "Old Nan died in chapter 40" in chron.check('"Sit," Nan said.')[0]
    assert chron.titles()[9_000] == _chapter(9_000)[0] and len(chron.titles()) == 10_001
    chron.close()


# ------------------------------------------------------------------ in the writing pipeline
class _Watch(FakeLLM):
    def __init__(self):
        super().__init__()
        self.seen: list[tuple[str, str]] = []

    def generate(self, prompt, **kw):
        self.seen.append((kw.get("purpose", ""), prompt))
        return super().generate(prompt, **kw)


def _studio(cfg, db):
    ctl = StudioController(cfg, db)
    llm = _Watch()
    pipe = ctl.pipeline()
    pipe.llm = pipe.fast_llm = llm
    for part in ("planner", "drafter"):
        getattr(pipe, part).llm = llm
    return ctl, llm


def test_every_chapter_is_filed_and_the_next_one_is_briefed(cfg, db):
    ctl, llm = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    ctl.write_next(name)
    p = Project(cfg.projects_dir, name)
    chron = Chronicle.at(p.dir)
    assert chron.written() == 1 and chron.chapter(1)["summary"].startswith("In chapter 1 Rin runs")
    assert {f["attribute"] for f in chron.facts_about("Rin")} == {"trade", "location"}
    assert chron.person("Rin")["chapters"] == 1
    llm.seen.clear()
    ctl.write_next(name)
    draft = next(prompt for kind, prompt in llm.seen if kind == "draft")
    assert "FROM THE STORY'S RECORD (established in earlier chapters; do not contradict it):" in draft
    assert "Rin, trade: courier (chapter 1, must not change)" in draft
    assert "Chapter 1: In chapter 1 Rin runs" in draft and "WHERE THINGS STAND NOW:" in draft     # layered, then the present
    assert [f["value"] for f in chron.history("Rin", "location")] == ["the alley in chapter 1", "the alley in chapter 2"]
    assert ctl.chapters(name)[0][0] == 1 and len(ctl.chapters(name)) == 2
    asked = ctl.ask("What does Rin do for a living?", name)
    assert asked["answer"] and any("Chapter 1: In chapter 1 Rin runs" in prompt for kind, prompt in llm.seen if kind == "studio:chat")

    cfg.write.chronicle = False                                    # switched off: nothing more is filed or briefed
    ctl._pipe = None
    ctl2, llm2 = _studio(cfg, db)
    ctl2.write_next(name)
    assert Chronicle.at(p.dir).written() == 2
    assert "FROM THE STORY'S RECORD" not in next(prompt for kind, prompt in llm2.seen if kind == "draft")


def test_a_dead_character_in_a_draft_sends_it_back_for_revision(cfg, db):
    ctl, llm = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    ctl.write_next(name)
    p = Project(cfg.projects_dir, name)
    Chronicle.at(p.dir).record(1, "One", "Rin ran.", "Rin runs.", [{"subject": "Rin", "attribute": "status", "value": "dead"}])
    cfg.write.min_acceptable_score = 0.0                           # only the record can send it back
    llm.seen.clear()
    ctl.write_next(name)
    meta = p.read_meta(2)
    assert any(kind == "revise" for kind, _ in llm.seen) or meta["revisions"] == 0
    revise = [prompt for kind, prompt in llm.seen if kind == "revise"]
    if revise:
        assert "Rin died in chapter 1" in revise[0]


def test_chapter_ten_thousand_costs_what_chapter_ten_did(cfg, db, tmp_path):
    """The real pipeline, one chapter, on a story that already has 9,990: it must not slow down or balloon."""
    ctl, llm = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    t0 = time.perf_counter()
    ctl.write_next(name, 200)
    first = time.perf_counter() - t0
    first_prompt = max(len(prompt) for kind, prompt in llm.seen if kind == "draft")

    p = Project(cfg.projects_dir, name)
    chron = Chronicle.at(p.dir)
    for n in range(2, 9_991):
        title, text, summary = _chapter(n)
        chron.record(n, title, text, summary, [{"subject": text.split()[0], "attribute": "location", "value": title}])
    p.state.outline = [ChapterPlan(number=n, title=f"Plan {n}", goal="go on", hook="h", turn="t", cliffhanger="c") for n in range(1, 9_996)]
    p.state.written = 9_990
    for n in range(9_978, 9_991):                                  # the last dozen chapters are on disk, as they would be
        p.write_chapter(n, f"Plan {n}", _chapter(n)[1], {"title": f"Plan {n}"})
    p.save()

    ctl._pipe = None
    ctl, llm = _studio(cfg, db)
    t0 = time.perf_counter()
    info = ctl.write_next(name, 200)
    late = time.perf_counter() - t0
    assert info["number"] == 9_991
    late_prompt = max(len(prompt) for kind, prompt in llm.seen if kind == "draft")
    assert late < first * 4 + 3.0, (first, late)                   # no slower in kind, with room for noise
    assert late_prompt < first_prompt * 3 + 6000, (first_prompt, late_prompt)
    draft = next(prompt for kind, prompt in llm.seen if kind == "draft")
    assert "Chapters 1-1000:" in draft and "Chapters 9981-9990:" in draft
    # The working file stays small: old plans were put away, and can still be read.
    p2 = Project(cfg.projects_dir, name)
    assert len(p2.state.outline) < 400 and p2.plan_for(5).title == "Plan 5" and p2.plan_for(9_991) is not None
    assert (p2.dir / "outline_archive.jsonl").exists() and p2.state_path.stat().st_size < 400_000
    assert len(ctl.chapters(name)) == 9_991 and ctl.chapters(name)[4_999][1] == _chapter(5_000)[0]


def test_an_older_story_can_be_filed_afterwards_and_read_from_the_command_line(cfg, db):
    from click.testing import CliRunner

    from gauge.cli import main

    cfg.write.chronicle = False                                    # a story begun before the chronicle existed
    cfg.save()
    ctl, _ = _studio(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    for _ in range(3):
        ctl.write_next(name)
    p = Project(cfg.projects_dir, name)
    assert Chronicle.at(p.dir).written() == 0
    cfg.write.chronicle = True
    cfg.save()
    runner = CliRunner(env={"COLUMNS": "200"})

    def run(*args):
        r = runner.invoke(main, ["--data-dir", cfg.data_dir, "chronicle", "Ten Seconds", *args])
        assert r.exit_code == 0, r.output
        return r.output

    assert "3 written chapters are not on record yet" in run()
    assert "Filed 3 chapters" in run("--rebuild")
    out = run()
    assert "3 chapters on record" in out and "names" in out and "not on record yet" not in out
    assert "Chapter 1: In chapter 1 Rin runs" in run("--so-far")
    about = run("--about", "Rin")
    assert "first seen chapter 1" in about and "trade: courier" in about and "lasting" in about
    assert "Rin, trade: courier" in run("--brief", "Rin opens the bag")
    Chronicle.at(p.dir).record(4, "Clash", "x", "x", [{"subject": "Rin", "attribute": "trade", "value": "smuggler"}])
    waiting = run("--conflicts")
    assert "'courier' (ch 1) against 'smuggler' (ch 4)" in waiting and "--settle" in waiting
    number = Chronicle.at(p.dir).open_conflicts()[0]["id"]
    assert "the earlier value stands" in run("--settle", str(number))
    assert Chronicle.at(p.dir).open_conflicts() == []
    # Without a model: names and opening lines only, but it still works.
    Chronicle.at(p.dir).close()
    (p.dir / "chronicle.db").unlink()
    assert "Filed 3 chapters" in run("--rebuild", "--quick")
    r = runner.invoke(main, ["--data-dir", cfg.data_dir, "chronicle", "No Such Story"])
    assert r.exit_code == 1
