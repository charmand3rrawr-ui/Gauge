"""World agent, creativity (muse, novelty, technique absorption), premise/continue starts, Studio controller."""
import json

from gauge.gui import StudioController
from gauge.learn import Trainer
from gauge.learn.techniques import Absorber, Technique, TechniqueLibrary
from gauge.llm import FakeLLM, HashingEmbedder
from gauge.write import WritePipeline
from gauge.write.muse import Direction, Muse, NoveltyTracker, overlap
from gauge.write.world import (
    WorldAgent,
    WorldState,
    intervention_level,
    load_world,
    save_world,
)


def _trained(cfg, db, chapter_text):
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    return t


# ------------------------------------------------------------------ world
def test_intervention_level_scales_with_tension_and_agency():
    quiet = intervention_level("Rin and Jun share tea and talk about their mother.", 0, 0.6, 1)
    loud = intervention_level("The Broker's men attack; Rin runs through fire as the crowd screams and the blade falls.", 2, 0.6, 1)
    assert loud > quiet
    assert intervention_level("attack fire blood", 0, 1.0, 3) > intervention_level("attack fire blood", 0, 0.1, 3)
    assert intervention_level("x", 0, 0.6, 5) == intervention_level("x", 0, 0.6, 5)  # deterministic per chapter
    assert 0.05 <= quiet <= 1.0 and 0.05 <= loud <= 1.0


def test_world_agent_build_brief_review_remember_chat(tmp_path):
    llm = FakeLLM()
    w = WorldState()
    agent = WorldAgent(llm, w, agency=0.6)
    agent.build("# bible", "premise")
    assert w.name == "Lowmarket" and "Noodle Alley" in w.locations and w.weather == "steady rain"
    assert llm.calls_list[-1]["purpose"] == "world:build"
    quiet_scene = {"number": 2, "title": "Tea", "goal": "Rin and Jun talk quietly", "beats": []}
    b = agent.brief(quiet_scene, "- Rin: wants tea", conflicts=0)
    assert b.intervention_level < 0.35 and b.interventions == [] and len(b.background_figures) <= 1
    assert llm.calls_list[-1]["system"].startswith("You are the World")
    loud_scene = {"number": 3, "title": "Ambush", "goal": "The Broker's men attack Rin in the alley; blood, fire, screaming crowd",
                  "turn": "the blade falls", "beats": ["fight", "flee"]}
    b2 = agent.brief(loud_scene, "- Rin: wants out; friction: would run", conflicts=1)
    assert b2.intervention_level >= 0.35 and b2.interventions and b2.background_figures[0]["tag"] == "the one-armed noodle vendor"
    rendered = b2.render()
    assert "### The World" in rendered and "background: the one-armed noodle vendor" in rendered and "must stay true" in rendered
    rev = agent.review("the sun shone", 3)
    assert rev.consistency == 0.9 and rev.violations[0]["rule"] == "it is night and raining"
    agent.remember("chapter text", 3)
    assert w.last_chapter == 3 and w.weather == "heavier rain" and "the vendor saw Rin's face" in w.consequences
    assert w.figure("The One-Armed Noodle Vendor").appearances == 1 and w.history[-1]["chapter"] == 3
    agent.remember("again", 4)
    agent.remember("again", 5)
    assert [f.tag for f in w.promotable()] == ["the one-armed noodle vendor"]
    assert "vendor" in agent.chat("who is in the market?")
    save_world(tmp_path, w)
    again = load_world(tmp_path)
    assert again.name == "Lowmarket" and again.background_figures[0].appearances == 3
    assert (tmp_path / "world.md").read_text(encoding="utf-8").startswith("# World: Lowmarket")
    assert "Recurring extras" in w.describe()


def test_world_degrades_gracefully_on_bad_llm():
    w = WorldState(name="X", current_location="here", time="noon", weather="clear")
    agent = WorldAgent(FakeLLM(canned=["garbage", "more garbage", "x", "y", "z", "w"]), w)
    b = agent.brief({"number": 1, "title": "t", "goal": "g"}, "")
    assert b.setting.startswith("here") and b.interventions == []
    assert agent.review("t", 1).consistency == 1.0
    agent.remember("the end of things", 1)
    assert w.history and w.last_chapter == 1


# ------------------------------------------------------------------ creativity: novelty + muse
def test_novelty_tracker_detects_repetition(chapter_text):
    nt = NoveltyTracker()
    prev = [chapter_text(True) for _ in range(3)]
    rep_same = nt.report(chapter_text(True), prev)
    assert rep_same.ngram_overlap > 0.5 and rep_same.opening_repeat and rep_same.ending_repeat
    assert rep_same.score < 0.4 and rep_same.overused and rep_same.notes
    fresh = ("Nobody warned the river. It came up the stairs like it had been invited, brown and patient, and took the "
             "photographs first. Marta watched from the landing with the cat under one arm and the ledger under the other. "
             "\"Well,\" she said to the water. \"You're early.\" Downstairs the clock kept going until it didn't.")
    rep_fresh = nt.report(fresh, prev)
    assert rep_fresh.score > 0.85 and not rep_fresh.opening_repeat and rep_fresh.overused == []
    assert nt.report("anything", []) .score == 1.0
    assert overlap("a b c d", "a b c d") == 1.0 and overlap("a b c d", "x y z w") == 0.0
    over = nt.overused_phrases(prev)
    assert over and all(len(p.split()) == 3 for p in over)


def test_muse_proposes_and_picks_fresh_direction():
    muse = Muse(FakeLLM(), techniques_hint="[hook] Dialogue before setting")
    dirs = muse.directions("# bible", "summary", ["who wrote her name?"], past=[])
    assert len(dirs) == 3 and dirs[1].title == "Jun Vanishes"
    rated = muse.rate(dirs, "# bible", "summary", past=["The Broker calls the debt early and Rin must rewind through a negotiation."])
    # direction 1 overlaps the past pitch heavily -> novelty penalty; direction 2 rated highest by the model
    by_title = {d.title: d for d in rated}
    assert by_title["The Debt Comes Due"].novelty < by_title["Jun Vanishes"].novelty
    assert rated[0].title == "Jun Vanishes" and rated[0].surprise == 0.9
    assert muse.choose("# bible", "s", [], []).title == "Jun Vanishes"
    assert Muse(FakeLLM(canned=["bad", "bad"])).choose("b", "s", [], []) is None
    d = Direction(title="t", pitch="p", surprise=1, fit=1, payoff=1, novelty=1)
    assert abs(d.score - (0.35 + 0.3 + 0.2 + 1.5)) < 1e-9


# ------------------------------------------------------------------ creativity: techniques
def test_technique_library_and_absorber(cfg, db, chapter_text):
    _trained(cfg, db, chapter_text)
    lib = TechniqueLibrary(HashingEmbedder(64))
    lib.add([Technique("hook", "Dialogue first", "open on a line", "\"Run,\"", "n:1", "N", 1, 5.0),
             Technique("ending", "Sound cut", "end on silence", "stopped.", "n:1", "N", 1, 5.0),
             Technique("hook", "Question open", "ask", "Why?", "n:2", "M", 2, 2.0)])
    found = lib.search("how should the chapter open with a hook", k=2)
    assert len(found) == 2 and {t.kind for t in found} == {"hook", "ending"}  # diversified by kind
    lib.save(cfg.models_dir)
    again = TechniqueLibrary(HashingEmbedder(64))
    assert again.load(cfg.models_dir) and len(again.items) == 3 and again.matrix.shape[0] == 3
    fast = FakeLLM()
    absorber = Absorber(db, fast, HashingEmbedder(64), cfg.models_dir, cfg.learn.target)
    assert len(absorber.library.items) == 3  # loaded the saved library
    cands = absorber.candidates(4)
    assert len(cands) == 4 and cands[0][2] >= cands[-1][2]  # strongest novels first
    res = absorber.run(4)
    assert res["chapters_absorbed"] == 4 and res["techniques_added"] == 12 and res["techniques_total"] == 15
    assert len(db.absorbed_keys()) == 4
    res2 = absorber.run(4)  # incremental: the next four
    assert res2["chapters_absorbed"] == 4 and len(db.absorbed_keys()) == 8
    assert all(c["purpose"] == "absorb" for c in fast.calls_list)


# ------------------------------------------------------------------ pipeline integration
def test_pipeline_world_muse_techniques_novelty(cfg, db, chapter_text):
    t = _trained(cfg, db, chapter_text)
    fast = FakeLLM()
    # a technique library on disk is picked up by the pipeline
    Absorber(db, fast, t.embedder, cfg.models_dir, cfg.learn.target).run(2)
    main = FakeLLM()
    pipe = WritePipeline(cfg, db, llm=main, trainer=t, fast_llm=fast)
    assert pipe.techniques.items
    p = pipe.new_project("Ten Seconds", "a courier who can rewind ten seconds", ["Sci-fi"], words=300)
    w = pipe.world(p)
    assert w is not None and w.name == "Lowmarket" and (p.dir / "world.md").exists()
    assert p.state.past_directions and p.state.current_direction.startswith("Jun Vanishes")  # muse ran for the outline
    assert any(c["purpose"] == "muse" for c in fast.calls_list) and any(c["purpose"] == "muse:rate" for c in fast.calls_list)
    res = pipe.write_next(p)
    draft_prompt = next(c["prompt"] for c in main.calls_list if c["purpose"] == "draft")
    assert "### The World" in draft_prompt and "CRAFT MOVES TO TRY" in draft_prompt and "[hook]" in draft_prompt
    outline_prompts = [c["prompt"] for c in main.calls_list + fast.calls_list if c["purpose"] == "outline"]
    assert outline_prompts and "Jun Vanishes" in outline_prompts[0]  # the muse's direction reached the outliner
    assert res.world_review is not None and res.world_review.consistency == 0.9
    assert res.novelty is None  # first chapter: nothing to compare with
    assert "world" in res.timings
    front = res.path.read_text(encoding="utf-8")
    assert "world_consistency: 0.9" in front and "direction:" in front
    w2 = load_world(p.dir)
    assert w2.last_chapter == 1 and w2.background_figures and w2.weather == "heavier rain"
    # chapter 2: novelty now measured against chapter 1 (fake prose is identical -> low novelty, avoid list present)
    res2 = pipe.write_next(p)
    assert res2.novelty is not None and res2.novelty.score < 0.5 and res2.novelty.opening_repeat and res2.novelty.ngram_overlap > 0.9
    draft2 = [c["prompt"] for c in main.calls_list if c["purpose"] == "draft"][-1]
    assert "PHRASES AND SHAPES TO AVOID" in draft2 and "recent chapter openings" in draft2
    res3 = pipe.write_next(p)  # with two earlier chapters, worn phrases are now identified
    assert res3.novelty.overused and "worn out" in [c["prompt"] for c in main.calls_list if c["purpose"] == "draft"][-1]
    assert res2.score < res.score  # repetition is punished


def test_world_can_be_disabled(cfg, db, chapter_text):
    cfg.write.world_enabled = False
    cfg.write.muse_enabled = False
    t = _trained(cfg, db, chapter_text)
    fast = FakeLLM()
    pipe = WritePipeline(cfg, db, llm=FakeLLM(), trainer=t, fast_llm=fast)
    p = pipe.new_project("Quiet", "a quiet story", ["Drama"], words=200)
    assert pipe.world(p) is None and not p.state.past_directions
    res = pipe.write_next(p)
    assert res.world_review is None and not any(c["purpose"].startswith("world") for c in fast.calls_list)


# ------------------------------------------------------------------ premise + continue
def test_premise_document_is_canon_and_continue_from_manuscript(cfg, db, chapter_text, tmp_path):
    t = _trained(cfg, db, chapter_text)
    main, fast = FakeLLM(), FakeLLM()
    pipe = WritePipeline(cfg, db, llm=main, trainer=t, fast_llm=fast)
    premise = "PREMISE\n" + "Rin Okafor is a courier in Lowmarket who can rewind ten seconds. " * 12 + "\nRULES: each rewind costs a memory."
    p = pipe.new_project("Premise Story", "", ["Sci-fi"], words=300, premise_doc=premise)
    bible_prompt = next(c["prompt"] for c in main.calls_list if c["purpose"] == "bible")
    assert "Every fact in the document is canon" in bible_prompt and "Rin Okafor" in bible_prompt
    assert (p.dir / "premise.md").read_text(encoding="utf-8") == premise
    # continue from an existing manuscript
    ms = tmp_path / "manuscript.txt"
    ms.write_text("Chapter 1\n\n" + chapter_text(True) + "\n\nChapter 2\n\n" + chapter_text(False), encoding="utf-8")
    p2 = pipe.new_project("Continued", "", ["Sci-fi"], words=300, continue_from=str(ms))
    assert p2.state.written == 2 and p2.read_chapter(2) and p2.state.continued_from == str(ms)
    assert p2.state.running_summary.startswith("Rin, a courier") and p2.state.open_threads == ["Who wrote her name?"]
    assert "CHARACTERS SO FAR" in p2.state.premise_doc and "Jun" in p2.state.premise_doc
    assert any(c["purpose"] == "continue:summary" for c in fast.calls_list)
    res = pipe.write_next(p2)
    assert res.number == 3 and p2.plan_for(3) is not None
    assert p2.state.outline[0].number >= 3 or any(c.number == 3 for c in p2.state.outline)


# ------------------------------------------------------------------ studio controller + CLI
def test_studio_controller_flow(cfg, db, chapter_text):
    _trained(cfg, db, chapter_text)
    ctl = StudioController(cfg, db)
    st = ctl.status()
    assert st["llm_ok"] and st["projects"] == []
    name = ctl.start_story("A kid who finds a dragon egg in the school lost-and-found")
    assert name == "A kid who finds"
    assert ctl.start_story("A kid who finds a dragon egg in the school lost-and-found") == "A kid who finds 2"  # no clobbering
    info = ctl.write_next(name)
    assert info["number"] == 1 and ctl.chapters(name) == [(1, "Chapter 1")]
    assert ctl.read(name, 1).startswith("# Chapter 1") and "not been written" in ctl.read(name, 9)
    assert "Rin" in ctl.characters(name)
    assert "One more time" in ctl.chat(name, "Rin", "why?") and "don't know anyone" in ctl.chat(name, "Zed", "?")
    out = ctl.export_story(name)
    assert out.exists() and out.read_text(encoding="utf-8").startswith(f"# {name}")
    long_idea = "Marta and the river. " * 30
    assert ctl.start_story(long_idea, name="River") == "River"
    try:
        ctl.start_story("   ")
    except ValueError:
        pass
    else:
        raise AssertionError("empty idea must be rejected")


def test_cli_start_world_absorb(tmp_path, chapter_text):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "200"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    for i in range(6):
        d = tmp_path / f"n{i}"
        d.mkdir()
        for c in range(2):
            (d / f"{c}.txt").write_text(chapter_text(i % 2 == 0), encoding="utf-8")
        (d / "novel.json").write_text(json.dumps({"title": f"N{i}", "followers": 5000 if i % 2 == 0 else 100}))
        runner.invoke(main, ["--data-dir", data, "import", str(d)])
    r = runner.invoke(main, ["--data-dir", data, "absorb", "--chapters", "2"])
    assert r.exit_code == 0 and "Absorbed" in r.output and "library by kind" in r.output, r.output
    pf = tmp_path / "premise.md"
    pf.write_text("Rin Okafor, courier. " * 40, encoding="utf-8")
    r = runner.invoke(main, ["--data-dir", data, "new", "Prem", "--premise-file", str(pf), "--genre", "Sci-fi", "--words", "200"])
    assert r.exit_code == 0 and "World: Lowmarket" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "new", "Nothing"])
    assert r.exit_code != 0
    r = runner.invoke(main, ["--data-dir", data, "start", "A courier who can rewind ten seconds", "--chapters", "1", "--words", "200"])
    assert r.exit_code == 0 and "Chapter 1" in r.output and "Continue with" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "world", "show", "Prem"])
    assert r.exit_code == 0 and "# World: Lowmarket" in r.output
    r = runner.invoke(main, ["--data-dir", data, "world", "set", "Prem", "--rule", "gravity is optional on Sundays", "--weather", "hail", "--agency", "0.9"])
    assert r.exit_code == 0 and "hail" in r.output
    assert json.loads((tmp_path / "data" / "config.json").read_text())["write"]["world_agency"] == 0.9
    r = runner.invoke(main, ["--data-dir", data, "world", "chat", "Prem", "--ask", "who is in the market?"])
    assert r.exit_code == 0 and "vendor" in r.output
    r = runner.invoke(main, ["--data-dir", data, "write", "Prem"])
    assert r.exit_code == 0 and "world 0.90" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "world", "extras", "Prem"])
    assert r.exit_code == 0 and "noodle vendor" in r.output
    r = runner.invoke(main, ["--data-dir", data, "world", "promote", "Prem", "the one-armed noodle vendor", "--name", "Old Pak"])
    assert r.exit_code == 0 and "Promoted" in r.output and "## Old Pak" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "show", "Prem", "Old Pak"])
    assert r.exit_code == 0 and "one-armed noodle vendor" in r.output
    ms = tmp_path / "book.txt"
    ms.write_text("Chapter 1\n\n" + chapter_text(True) + "\n\nChapter 2\n\n" + chapter_text(True), encoding="utf-8")
    r = runner.invoke(main, ["--data-dir", data, "new", "Cont", "--continue-from", str(ms), "--words", "200"])
    assert r.exit_code == 0 and "continuing from chapter 3" in r.output, r.output
