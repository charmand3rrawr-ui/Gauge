"""The writing room: eleven independent agents and the council that runs them."""
import json
from pathlib import Path

from gauge.learn import Trainer
from gauge.llm import FakeLLM, HashingEmbedder
from gauge.write import WritePipeline
from gauge.write.agents import (
    ALL_AGENTS,
    AgentCouncil,
    AgentReview,
    Archivist,
    Comedian,
    Librarian,
    LineEditor,
    PacingDirector,
    Packager,
    ReaderPanel,
    SceneContext,
    Schemer,
    SensitivityReader,
    Showrunner,
    ThemeKeeper,
    build_council,
)
from gauge.write.agents.archivist import Fact
from gauge.write.agents.comedian import wants_comedy
from gauge.write.agents.editor import measure
from gauge.write.agents.pacing import classify
from gauge.write.agents.packager import title_score
from gauge.write.project import ChapterPlan, Project


def _ctx(tmp_path, number=2, prev="She ran. The footsteps stopped.", goal="Rin confronts The Broker about the seal"):
    p = Project.create(tmp_path / "proj", "P", "idea", ["Thriller"], 300) if not (tmp_path / "proj" / "p" / "project.json").exists() else Project(tmp_path / "proj", "P")
    plan = ChapterPlan(number=number, title="Chapter 2", goal=goal, hook="a knock", turn="the Broker smiles", cliffhanger="Jun is gone")
    return p, SceneContext(project=p, number=number, plan=plan, prev_tail=prev, summary="Rin rewound time.", cast_text="- Rin: wants out", genres=["Thriller"])


def _trained(cfg, db, chapter_text):
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    return t


# ------------------------------------------------------------------ individual agents
def test_reader_panel_predicts_and_calibrates(tmp_path):
    p, ctx = _ctx(tmp_path)
    panel = ReaderPanel(FakeLLM(), p.dir)
    assert "hoping for" in panel.brief(ctx)
    rev = panel.review("Chapter text.", ctx)
    assert rev.agent == "reader_panel" and abs(rev.score - 0.68) < 1e-6 and rev.data["per_persona"]["hostile"] == 0.4
    assert any("hostile would stop" in i for i in rev.issues)
    panel.remember("Chapter text.", ctx)
    assert panel.state.predictions[-1].chapter == 2 and panel.state.predictions[-1].comments
    # calibration: predicted 0.68 for ch 2..4, observed retention 0.9 -> positive bias
    for n in (3, 4):
        ctx.number = n
        panel.review("x", ctx)
        panel.remember("x", ctx)
    for n in (2, 3, 4, 5):
        panel.record_actual(n, views=1000 * (0.9 ** (n - 2)))
    cal = panel.calibration()
    assert cal["pairs"] == 3 and abs(panel.state.bias - 0.22) < 0.01
    rev2 = panel.review("x", ctx)
    assert rev2.score > rev.score  # bias applied
    panel.save()
    again = ReaderPanel(FakeLLM(), p.dir)
    assert again.state.bias == panel.state.bias and "Calibration" in again.report()


def test_showrunner_ledger(tmp_path):
    p, ctx = _ctx(tmp_path)
    sr = Showrunner(FakeLLM(), p.dir)
    assert "plant at least one" in sr.brief(ctx)
    sr.remember("ch text", ctx)
    assert len(sr.open()) == 1 and sr.open()[0].due == 6 and sr.open()[0].kind == "question"
    sr.remember("ch text", ctx)  # duplicate plant ignored
    assert len(sr.open()) == 1
    ctx.number = 7  # now overdue
    assert "pay or clearly advance" in sr.brief(ctx)
    assert "must be paid" in sr.plan_notes(type("P", (), {"next_chapter_number": lambda self: 6})())
    rev = sr.review("ch text", ctx)
    assert rev.score < 1.0 and any("overdue" in i for i in rev.issues)  # fake says 'advanced the debt', not the name
    # paying it off
    sr.state.promises[0].text = "the debt"
    rev2 = sr.review("ch", ctx)
    assert rev2.score == 1.0
    sr.state.last_payoff_chapter = 1
    ctx.number = 6
    assert "chapters since the last real payoff" in sr.plan_notes(type("P", (), {"next_chapter_number": lambda self: 6})())
    assert "open" in sr.report()


def test_archivist_facts_and_contradictions(tmp_path):
    p, ctx = _ctx(tmp_path)
    ar = Archivist(FakeLLM(), p.dir)
    assert ar.review("x", ctx) is None  # empty ledger: nothing to contradict
    ar.remember("x", ctx)
    assert len(ar.state.facts) == 2 and ar.lookup("rin", "eye")[0].value == "grey"
    assert "canon: Rin / eye colour: grey" in ar.brief(_ctx(tmp_path, goal="Rin looks in the mirror")[1])
    rev = ar.review("her brown eyes", ctx)  # fake review extracts 'brown'
    assert rev.score == 0.75 and "contradiction: Rin / eye colour was 'grey'" in rev.issues[0]
    assert ar.contradictions([Fact(subject="Rin", attribute="Eye Colour", value="Grey", chapter=3)]) == []  # same value, different case
    assert "grey" in ar.ask("what colour are Rin's eyes?")
    assert "nothing on that" in ar.ask("zzzz qqqq")
    ar.remember("x", ctx)  # duplicates not re-added
    assert len(ar.state.facts) == 2
    assert "2 facts" in ar.report()


def test_schemer_plans_and_adapts(tmp_path):
    p, ctx = _ctx(tmp_path)
    sc = Schemer(FakeLLM(), p.dir)
    assert sc.brief(ctx) == "" and sc.review("x", ctx) is None  # not set up yet
    sc.setup(p)
    assert sc.state.antagonist == "The Broker" and len(sc.state.scheme) == 3
    assert "hidden: the Broker ordered the seal broken" in sc.brief(ctx)
    rev = sc.review("x", ctx)
    assert rev.score == 1.0 and rev.data["exposure"].startswith("Rin now knows")
    sc.remember("x", ctx)
    assert "break a seal to test her" in sc.state.done and "send the vendor to report" in sc.state.scheme
    assert sc.state.exposure.startswith("Rin suspects") and sc.state.moves[-1]["chapter"] == 2
    sc.save()
    assert Schemer(FakeLLM(), p.dir).state.antagonist == "The Broker"  # state persisted


def test_line_editor_measures_and_fingerprints(tmp_path, chapter_text):
    p, ctx = _ctx(tmp_path)
    ed = LineEditor(FakeLLM(), p.dir)
    flat = "I felt sad. I saw the door. I realized it was open. I noticed the cold. I heard a sound. " * 12
    rev = ed.review(flat, ctx)
    assert rev.score < 0.8 and any("filter words" in i for i in rev.issues) and any("barely vary" in i for i in rev.issues)
    assert any(i.startswith("line:") for i in rev.issues)
    good = chapter_text(True)
    rev2 = ed.review(good, ctx)
    assert rev2.score > rev.score
    for _ in range(3):
        ed.remember(good, ctx)
    assert ed.state.fingerprint_chapters == 3 and "first_person_ratio" in ed.state.fingerprint
    assert "keep the established voice" in ed.brief(ctx)
    drifted = "He walked. She talked. They argued about the weather for a very long time, very slowly. " * 20
    assert ed.drift(measure(drifted)) > 0.0
    assert "Voice fingerprint" in ed.report()


def test_packager_titles_and_blurb(tmp_path):
    p, ctx = _ctx(tmp_path)
    pk = Packager(FakeLLM(), p.dir)
    assert pk.retitle("text", ChapterPlan(number=1, title="The Door Opens"), 1) is None  # already specific
    t = pk.retitle("text", ChapterPlan(number=1, title="Chapter 1"), 1)
    assert t in {"Ten Seconds", "The Seal", "One More Time"} and pk.state.titles[1] == t
    assert title_score("Chapter 3") < title_score("The Seal Breaks") and title_score("Why Did She Run?") > title_score("Run")
    out = pk.package(p)
    assert out["tags"] and out["tagline"] and "rewind" in out["blurb"]
    assert "Tagline" in pk.report()


def test_theme_keeper(tmp_path):
    p, ctx = _ctx(tmp_path)
    tk = ThemeKeeper(FakeLLM(), p.dir)
    assert tk.brief(ctx) == ""
    tk.setup(p)
    assert tk.state.motifs == ["rain", "seals", "counting"]
    b = tk.brief(ctx)
    assert "thread this motif lightly" in b
    rev = tk.review("She counted the coins twice.", ctx)
    assert rev.score == 1.0 and "counting" in rev.data["motifs"]
    tk.remember("x", ctx)
    assert tk.state.occurrences["counting"] == [2]
    assert tk._pick(3) != "counting"  # least recently used motif rotates in
    rev2 = tk.review("nothing thematic", SceneContext(project=p, number=3, plan=ctx.plan))
    assert rev2.score == 1.0 or rev2.score == 0.7  # fake LLM says counting present


def test_sensitivity_reader_warns_or_blocks(tmp_path):
    p, ctx = _ctx(tmp_path)
    sr = SensitivityReader(FakeLLM(), p.dir, {"block_on_flags": False})
    rev = sr.review("the blade", ctx)
    assert rev.score == 1.0 and rev.issues == [] and rev.data["flags"][0]["category"] == "violence"
    sr.remember("x", ctx)
    assert sr.state.flags[2][0]["severity"] == "mild" and "mild violence" in sr.report()
    blocker = SensitivityReader(FakeLLM(canned=[json.dumps({"flags": [{"category": "hate or slurs", "severity": "severe", "quote": "x", "note": "n"}], "ok_for_general": False})]),
                                p.dir, {"block_on_flags": True})
    rev2 = blocker.review("x", ctx)
    assert rev2.score == 0.0 and rev2.issues and "severe" in rev2.issues[0]


def test_pacing_director_varies_shapes(tmp_path, chapter_text):
    p, ctx = _ctx(tmp_path)
    pd = PacingDirector(FakeLLM(), p.dir)
    assert pd.recommend() == "action" and pd.plan_notes(p) == ""
    action = chapter_text(True)
    for n in (1, 2):
        pd.remember(action, SceneContext(project=p, number=n, plan=ctx.plan))
    assert pd.state.shapes == ["action", "action"] and pd.recommend() == "quiet"
    assert "should be a quiet chapter" in pd.plan_notes(p)
    assert "shape this chapter as quiet" in pd.brief(ctx)
    rev = pd.review(action, ctx)  # third action in a row
    assert rev.score == 0.6 and "third action chapter in a row" in rev.issues[0]
    assert classify({"reveal_density": 30, "cliff_reveal": 0.5}) == "reveal" and classify({"dialogue_ratio": 0.6}) == "dialogue"
    assert "Shapes:" in pd.report()


def test_librarian_indexes_notes(tmp_path):
    p, ctx = _ctx(tmp_path)
    (p.dir / "research").mkdir()
    (p.dir / "research" / "sewers.md").write_text("The Lowmarket sewers were built by the old guild. " * 30 + "They flood at high tide. " * 10, encoding="utf-8")
    (p.dir / "premise.md").write_text("Rewinds cost a memory each. " * 40, encoding="utf-8")
    lib = Librarian(FakeLLM(), p.dir, embedder=HashingEmbedder(64))
    assert len(lib.chunks) >= 2
    b = lib.brief(SceneContext(project=p, number=2, plan=ChapterPlan(number=2, title="t", goal="Rin flees into the sewers at high tide")))
    assert "sewers.md" in b
    assert "guild" in lib.ask("who built the sewers?") and "No notes match" in Librarian(FakeLLM(), tmp_path / "empty").ask("x")
    assert "indexed chunks" in lib.report()


def test_comedian_only_for_comic_genres(tmp_path):
    assert wants_comedy(["Comedy"]) and wants_comedy([], "wry, tense") and not wants_comedy(["Horror"], "grim")
    p, ctx = _ctx(tmp_path)
    cm = Comedian(FakeLLM(), p.dir)
    assert "comic beat" in cm.brief(ctx)
    rev = cm.review("x", ctx)
    assert rev.score == 0.9 and rev.data["laughs"] == 2 and "flat joke" in rev.issues[0]
    cm.remember("x", ctx)
    assert cm.state.laughs == [2]


# ------------------------------------------------------------------ council
def test_council_merges_scores_and_respects_config(cfg, tmp_path):
    p = Project.create(tmp_path / "pp", "Comic Tale", "idea", ["Comedy"], 300)
    council = build_council(cfg, FakeLLM(), p, embedder=HashingEmbedder(32))
    assert set(council.names()) == set(ALL_AGENTS) - {"scout"}  # comedy attaches the comedian; scout needs a Scout instance
    p2 = Project.create(tmp_path / "pp2", "Grim Tale", "idea", ["Horror"], 300)
    assert "comedian" not in build_council(cfg, FakeLLM(), p2).names()
    cfg.agents.profile = "lean"
    lean = build_council(cfg, FakeLLM(), p)
    assert set(lean.names()) == {"archivist", "showrunner", "line_editor", "pacing", "librarian", "packager", "voice_coach", "readability", "suitability", "arc_keeper"}
    cfg.agents.profile = "full"
    cfg.agents.schemer = False
    assert "schemer" not in build_council(cfg, FakeLLM(), p).names()
    cfg.agents.profile = "off"
    assert build_council(cfg, FakeLLM(), p).agents == []
    reviews = [AgentReview(agent="reader_panel", score=0.5, issues=["a", "b", "c"]), AgentReview(agent="pacing", score=1.0),
               AgentReview(agent="sensitivity", score=1.0)]
    assert abs(council.score(reviews) - (0.5 * 2.0 + 1.0 * 0.5) / 2.5) < 1e-9  # weighted; weight-0 agents excluded
    assert council.score([AgentReview(agent="sensitivity", score=0.0)]) == 0.0  # a blocking flag floors the score
    assert AgentCouncil.problems(reviews) == ["[reader_panel] a", "[reader_panel] b"]
    assert council.score([]) is None


def test_pipeline_runs_the_writing_room(cfg, db, chapter_text):
    t = _trained(cfg, db, chapter_text)
    main, fast = FakeLLM(), FakeLLM()
    pipe = WritePipeline(cfg, db, llm=main, trainer=t, fast_llm=fast)
    p = pipe.new_project("Ten Seconds", "a courier who can rewind ten seconds", ["Sci-fi", "Comedy"], words=300)
    agents_dir = p.dir / "agents"
    assert (agents_dir / "schemer.json").exists() and (agents_dir / "theme_keeper.json").exists()
    assert json.loads((agents_dir / "schemer.json").read_text())["antagonist"] == "The Broker"
    res = pipe.write_next(p)
    names = {r.agent for r in res.agent_reviews}
    assert {"reader_panel", "showrunner", "schemer", "line_editor", "theme_keeper", "sensitivity", "pacing", "comedian"} <= names
    draft_prompt = next(c["prompt"] for c in main.calls_list if c["purpose"] == "draft")
    assert "NOTES FROM THE WRITING ROOM" in draft_prompt and "### Schemer" in draft_prompt and "### Pacing Director" in draft_prompt
    assert "council" in res.timings
    front = res.path.read_text(encoding="utf-8")
    assert '"reader_panel": {"score": 0.68' in front and "content_flags" in front and '"category": "violence"' in front
    assert res.title in {"Ten Seconds", "The Seal", "One More Time"}  # packager retitled the generic plan title
    purposes = {c["purpose"] for c in fast.calls_list}
    assert {"reader_panel:review", "archivist:remember", "showrunner:remember", "schemer:brief", "comedian:brief"} <= purposes
    # states persisted and the archivist now has canon; chapter 2 gets canon briefs and a ledger
    assert json.loads((agents_dir / "archivist.json").read_text())["facts"]
    assert json.loads((agents_dir / "showrunner.json").read_text())["promises"]
    res2 = pipe.write_next(p)
    draft2 = [c["prompt"] for c in main.calls_list if c["purpose"] == "draft"][-1]
    assert "canon: Rin / eye colour" in draft2 and "### Showrunner" in draft2
    assert any(r.agent == "archivist" for r in res2.agent_reviews)  # ledger non-empty -> archivist reviews
    # council plan notes reach the outline when the muse extends it
    for _ in range(2):
        pipe.write_next(p)
    outline_prompts = [c["prompt"] for c in main.calls_list + fast.calls_list if c["purpose"] == "outline"]
    assert any("Constraints from the writing room" in o or "recent chapter shapes" in o for o in outline_prompts)


def test_pipeline_with_room_off(cfg, db, chapter_text):
    cfg.agents.profile = "off"
    t = _trained(cfg, db, chapter_text)
    fast = FakeLLM()
    pipe = WritePipeline(cfg, db, llm=FakeLLM(), trainer=t, fast_llm=fast)
    p = pipe.new_project("Quiet", "a quiet story", ["Drama"], words=200)
    res = pipe.write_next(p)
    assert res.agent_reviews == [] and not (p.dir / "agents").exists()
    assert not any(":" in c["purpose"] and c["purpose"].split(":")[0] in ALL_AGENTS for c in fast.calls_list)


def test_cli_writing_room(tmp_path, chapter_text):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "220"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    r = runner.invoke(main, ["--data-dir", data, "agents", "list"])
    assert r.exit_code == 0 and "Reader Panel" in r.output and "Comedian" in r.output
    r = runner.invoke(main, ["--data-dir", data, "new", "Ten Seconds", "--idea", "a courier", "--genre", "Sci-fi", "--words", "200"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--data-dir", data, "write", "Ten Seconds"])
    assert r.exit_code == 0 and "room:" in r.output and "reader_panel" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "agents", "list", "Ten Seconds"])
    assert r.exit_code == 0 and "Schemer" in r.output and "The Broker" in r.output
    r = runner.invoke(main, ["--data-dir", data, "agents", "show", "Ten Seconds", "schemer"])
    assert r.exit_code == 0 and "The Broker" in r.output
    r = runner.invoke(main, ["--data-dir", data, "canon", "Ten Seconds", "what colour are Rin's eyes?"])
    assert r.exit_code == 0 and "grey" in r.output
    r = runner.invoke(main, ["--data-dir", data, "canon", "Ten Seconds", "--subject", "Rin"])
    assert r.exit_code == 0 and "eye colour" in r.output
    r = runner.invoke(main, ["--data-dir", data, "ledger", "Ten Seconds"])
    assert r.exit_code == 0 and "open" in r.output
    r = runner.invoke(main, ["--data-dir", data, "readers", "Ten Seconds"])
    assert r.exit_code == 0 and "click_next" in r.output
    r = runner.invoke(main, ["--data-dir", data, "feedback", "Ten Seconds", "1", "--views", "1200"])
    assert r.exit_code == 0 and "recorded" in r.output
    r = runner.invoke(main, ["--data-dir", data, "package", "Ten Seconds"])
    assert r.exit_code == 0 and "Tags:" in r.output
    r = runner.invoke(main, ["--data-dir", data, "research", "Ten Seconds", "sewers"])
    assert r.exit_code == 0
    r = runner.invoke(main, ["--data-dir", data, "agents", "disable", "schemer"])
    assert r.exit_code == 0 and json.loads(Path(data, "config.json").read_text())["agents"]["schemer"] is False
    r = runner.invoke(main, ["--data-dir", data, "agents", "enable", "lean"])
    assert r.exit_code == 0 and json.loads(Path(data, "config.json").read_text())["agents"]["profile"] == "lean"
    r = runner.invoke(main, ["--data-dir", data, "agents", "enable", "bogus"])
    assert r.exit_code != 0
