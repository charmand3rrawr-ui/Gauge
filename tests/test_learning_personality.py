"""Learning from your own serial (bandit, project model, ledger, lessons, comments, fine-tune data) and
personality depth (persona, improv, emotion + drift, voice coach, second-order beliefs)."""
import json
from pathlib import Path

from gauge.ingest.royalroad import parse_chapter_comments
from gauge.learn import Trainer
from gauge.learn.bandit import KNOBS, PromptBandit
from gauge.learn.comments import (
    analyze_comment,
    chapter_comment_features,
    persona_samples,
    reaction_hotspots,
)
from gauge.learn.finetune import (
    dataset_summary,
    plan,
    record_self_pair,
    save_accepted_copy,
    sync_author_edits,
)
from gauge.learn.ledger import evaluate, record_predictions
from gauge.learn.lessons import classify, distill, prompt_lines, write_lessons
from gauge.learn.project_model import ProjectModel, retention_labels
from gauge.llm import FakeLLM, HashingEmbedder
from gauge.models import NovelRecord
from gauge.write import WritePipeline
from gauge.write.agents import SceneContext, VoiceCoach
from gauge.write.characters import (
    Character,
    CharacterAgent,
    PersonalityMatrix,
    Relationship,
    Roster,
    Voice,
)
from gauge.write.emotion import Mood, apply_event, baseline, drift_step, speech_directives
from gauge.write.improv import render_improv, run_improv
from gauge.write.persona import AuthorPersona, build_persona, load_persona, save_persona
from gauge.write.project import ChapterPlan, Project
from gauge.write.stylometry import attribute_dialogue, distinctness, style_vector
from gauge.write.world import WorldState


def _trained(cfg, db, chapter_text):
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    return t


# ------------------------------------------------------------------ bandit
def test_bandit_learns_and_persists(tmp_path):
    b = PromptBandit(tmp_path / "bandit.json", seed=3)
    s = b.sample(1)
    assert set(s) == set(KNOBS) and s["exemplars"] in KNOBS["exemplars"]
    assert b.update(1, 0.9) and not b.update(99, 0.5)  # unknown chapter ignored
    assert b.state.pending.get(1) is not None  # immediate reward keeps the arm pending for a delayed one
    assert b.update(1, 0.2, weight=2.0, kind="retention") and 1 not in b.state.pending
    arm = b.state.arms["exemplars"][str(s["exemplars"])]
    assert arm.pulls == 2 and abs(arm.alpha - (1 + 0.9 + 0.4)) < 1e-9
    b.save()
    again = PromptBandit(tmp_path / "bandit.json")
    assert again.state.arms["exemplars"][str(s["exemplars"])].pulls == 2 and len(again.state.history) == 2
    # a knob value rewarded consistently becomes the best
    for n in range(2, 40):
        s = b.sample(n)
        b.update(n, 0.95 if s["temperature"] == 1.05 else 0.05)
    assert b.best()["temperature"] == 1.05
    assert "temperature" in b.report()


# ------------------------------------------------------------------ project model + ledger
def test_project_model_from_feedback(cfg, db, chapter_text, tmp_path):
    p = Project.create(tmp_path / "pp", "Own", "idea", ["Fantasy"], 300)
    fb = {}
    for n in range(1, 13):
        hooky = n % 2 == 0
        p.write_chapter(n, f"C{n}", chapter_text(hooky), {})
        fb[n] = {"views": 1000.0 * (0.95 if hooky else 0.7) ** n, "comments": 10.0 if hooky else 2.0}
    labels = retention_labels(fb)
    assert len(labels) == 11 and labels[2] < labels[3]  # dull chapter -> readers leave before the next
    pm = ProjectModel(p.dir)
    assert pm.fit(p, fb, min_chapters=30) is None and pm.score01({}) is None  # not enough yet
    m = pm.fit(p, fb, min_chapters=5)
    assert m is not None and m.n_train == 11 and (p.dir / "models" / "project_retention.json").exists()
    assert ProjectModel(p.dir).model is not None and "project:retention" in pm.explain()
    hooky_score = pm.score01(__import__("gauge.features", fromlist=["chapter_features"]).chapter_features(
        __import__("gauge.models", fromlist=["ChapterRecord"]).ChapterRecord(novel_key="x", index=1, text=chapter_text(True))))
    assert hooky_score is not None and 0 <= hooky_score <= 1


def test_prediction_ledger(cfg, db, chapter_text):
    t = _trained(cfg, db, chapter_text)
    m = t.load_model()
    rows = db.predictions(cfg.learn.target)
    assert rows and len(rows) == m.n_train  # the training cycle recorded one prediction per labelled novel
    # backdate predictions, then move the truth: evaluation should see pairs
    db._conn.execute("UPDATE predictions SET at = datetime('now', '-10 days')")
    db._conn.commit()
    for n in db.novels():
        n.followers = (n.followers or 0) + 100
        db.upsert_novel(n)
    ev = evaluate(db, cfg.learn.target, older_than_days=7)
    assert ev["pairs"] == m.n_train and "correlation" in ev and "mae" in ev
    assert record_predictions(db, m, db.novels(), db.features("novel")) == m.n_train
    assert evaluate(db, "rating", 7) == {"pairs": 0}


# ------------------------------------------------------------------ lessons + comments
def test_lessons_distil(cfg, db):
    db.add_notes("P", 1, [("reader_panel", "skimmer would stop at the opening"), ("line_editor", "12 filter words per 1000")])
    db.add_notes("P", 2, [("reader_panel", "binge would stop at the slow middle"), ("showrunner", "overdue promises ignored")])
    db.add_notes("P", 3, [("reader_panel", "would stop at paragraph two"), ("line_editor", "too many intensifiers")])
    assert classify("the opening lacks a hook") == "openings" and classify("world rule broken") == "world rules"
    s = distill(db, "P")
    assert s["total"] == 6 and s["lessons"][0]["theme"] in {"readers", "prose"}
    lines = prompt_lines(s, 3)
    assert lines.startswith("- ") and len(lines.splitlines()) <= 3
    out = write_lessons(db, cfg.models_dir / "lessons.md", "P")
    assert (cfg.models_dir / "lessons.md").read_text(encoding="utf-8").startswith("# Lessons") and out["total"] == 6
    assert distill(db, "nope") == {"total": 0, "themes": [], "lessons": []}


def test_comment_parsing_and_analysis(db):
    html = '''<div class="chapter-inner chapter-content"><p>text</p></div>
    <div class="comments"><div class="comment"><a class="comment-author" href="/profile/1">readerA</a><div class="comment-body">OMG that cliffhanger!!! need the next one</div><time datetime="2024-01-01">x</time></div>
    <div class="comment"><span class="username">readerB</span><div class="comment-body">wait what, how did she hear her own voice? plot hole</div></div>
    <div class="comment"><span class="username">readerC</span><div class="comment-body">calling it: the Broker wrote the name</div></div></div>'''
    comments = parse_chapter_comments(html)
    assert len(comments) == 3 and comments[0]["author"] == "readerA" and comments[0]["at"] == "2024-01-01"
    a = analyze_comment(comments[0]["text"])
    assert a["excited"] >= 1 and analyze_comment(comments[1]["text"])["confused"] >= 1 and analyze_comment(comments[2]["text"])["prediction"] == 1
    n = NovelRecord(source="royalroad", source_id="1", title="T", followers=100)
    db.upsert_novel(n)
    assert db.add_comments(n.key, 1, comments) == 3 and db.add_comments(n.key, 1, comments) == 0  # idempotent
    f = chapter_comment_features(db.comments_for(n.key, 1))
    assert f["comment_count"] == 3 and f["comment_excited"] > 0 and f["comment_prediction"] > 0
    assert reaction_hotspots(db, n.key)[0][0] == 1
    samples = persona_samples(db)
    assert "binge" in samples or "lore" in samples or "commenter" in samples
    assert chapter_comment_features([])["comment_count"] == 0.0


# ------------------------------------------------------------------ fine-tune data
def test_self_pairs_author_edits_and_plan(cfg, tmp_path):
    ds = cfg.models_dir / "datasets"
    assert record_self_pair(ds, "P", 1, "prompt", "draft", "better", 0.5, 0.7) is not None
    assert record_self_pair(ds, "P", 2, "prompt", "draft", "worse", 0.7, 0.5) is None  # revising did not help
    assert record_self_pair(ds, "P", 3, "prompt", "same", "same", 0.5, 0.9) is None
    p = Project.create(tmp_path / "pp", "P", "idea", ["Fantasy"], 300)
    p.write_chapter(1, "One", "Gauge wrote this chapter. " * 20, {})
    save_accepted_copy(p, 1, "Gauge wrote this chapter. " * 20)
    assert sync_author_edits(p, ds) == []  # unchanged
    p.write_chapter(1, "One", "The author rewrote this chapter by hand. " * 20, {})
    changes = sync_author_edits(p, ds)
    assert changes and changes[0]["chapter"] == 1
    assert sync_author_edits(p, ds) == []  # same edit not recorded twice
    s = dataset_summary(ds)
    assert s["self_preference.jsonl"] == 1 and s["author_edits.jsonl"] == 1
    pl = plan(ds, "base-model", cfg.models_dir / "lora", min_pairs=1)
    assert pl["preference_pairs"] == 2 and pl["enough_data"] and "finetune_lora.py" in pl["command"]
    assert Path(pl["script"]).exists()


# ------------------------------------------------------------------ persona
def test_persona_build_and_prompt(cfg, chapter_text):
    per = build_persona(FakeLLM(), [chapter_text(True), chapter_text(True)], "local:mine")
    assert per.name == "Wet Neon" and per.fingerprint["first_person_ratio"] > 0 and len(per.samples) == 2
    assert per.register_style == "clipped street-level" and "no epigraphs" in per.taboos
    block = per.prompt_block(0.8)
    assert block.startswith("AUTHOR PERSONA (Wet Neon)") and "matters more than the data" in block and "A sample of the voice" in block
    assert "light flavour" in per.prompt_block(0.2) and per.prompt_block(0.0) == ""
    save_persona(cfg.models_dir, per)
    again = load_persona(cfg.models_dir)
    assert again.name == "Wet Neon" and (cfg.models_dir / "persona.md").exists()
    assert build_persona(None, [], "x").prompt_block(1.0) == ""
    assert AuthorPersona(stances=["x"]).prompt_block(0.5)


# ------------------------------------------------------------------ emotion, drift, beliefs, triangles
def test_emotion_model_and_speech():
    m = PersonalityMatrix(neuroticism=0.9, warmth=0.2, cynicism=0.8, dominance=0.1, courage=0.2, impulsivity=0.8)
    b = baseline(m)
    assert b.arousal > 0.3 and b.pleasure < 0 and b.dominance < 0
    mood = apply_event(Mood(), {"pleasure": -0.5, "arousal": 0.8, "dominance": -0.4}, b)
    assert mood.label() == "frightened"
    calm = apply_event(Mood(pleasure=0.5, arousal=-0.6), {}, Mood())  # decays toward baseline
    assert abs(calm.pleasure) < 0.5 and abs(calm.arousal) < 0.6
    v = Voice(swears=True, tics=["says okay twice"], humor_style="deflecting")
    d = speech_directives(mood, v)
    assert any("fragments" in x for x in d) and any("swears more" in x for x in d) and any("tic surfaces" in x for x in d) and any("hedges" in x for x in d)
    assert apply_event(Mood(), {"pleasure": "junk"}, Mood()).pleasure == 0.0
    # drift toward an arc target
    mx = PersonalityMatrix(courage=0.2)
    plan_ = {"courage": {"target": 0.9, "by": 10}, "bogus": {"target": 1}}
    moved = drift_step(mx, plan_, 1)
    assert "courage" in moved and 0.2 < mx.courage < 0.9
    for ch in range(2, 11):
        drift_step(mx, plan_, ch)
    assert abs(mx.courage - 0.9) < 1e-6


def test_remember_applies_mood_beliefs_and_arc(tmp_path):
    c = Character(name="Rin", role="protagonist", matrix=PersonalityMatrix(courage=0.3), voice=Voice(swears=True))
    c.state.arc_plan = {"courage": {"target": 0.8, "by": 5}}
    roster = Roster(tmp_path)
    roster.add(c)
    roster.add(Character(name="Jun"))
    roster.add(Character(name="The Broker", role="antagonist"))
    CharacterAgent(FakeLLM(), c).remember("text", 1, roster)
    assert c.state.mood_vec.arousal > 0.3 and c.state.mood_vec.pleasure < 0
    assert c.relationships["Jun"].beliefs["The Broker"].startswith("Jun trusts")
    assert c.matrix.courage > 0.3
    assert "Right now you" in c.system_prompt()  # speech directives flow into the character's own prompt
    # triangle: Rin believes Jun trusts the Broker; Jun actually fears him
    roster.get("Jun").relationships["The Broker"] = Relationship(trust=0.1, affection=0.2, fear=0.8)
    tri = roster.triangles()
    assert tri and tri[0]["who"] == "Rin" and tri[0]["regarding"] == "The Broker" and "distrusts" in tri[0]["truth"]
    assert "Rin -> Jun" in roster.graph_text() and "believes" in roster.graph_text()
    roster.save()
    again = Roster.load(tmp_path)
    assert again.get("Rin").state.arc_plan and again.get("Rin").relationships["Jun"].beliefs


# ------------------------------------------------------------------ improv
def test_improv_transcript(tmp_path):
    llm = FakeLLM()
    cast = [Character(name="Rin", role="protagonist"), Character(name="Jun")]
    w = WorldState(name="Lowmarket", time="night", weather="rain")
    t = run_improv(llm, cast, "Rin and Jun argue in the alley", rounds=2, world=w, world_level=0.6)
    speakers = [x["speaker"] for x in t]
    assert speakers == ["Rin", "Jun", "The World", "Rin", "Jun"]
    assert t[0]["beat"].startswith('"Okay. Okay,"') and "shutter" in t[2]["beat"]
    assert all(c["system"].startswith("You are Rin") for c in llm.calls_list if "next beat as Rin" in c["prompt"])
    quiet = run_improv(llm, cast, "tea", rounds=1, world=w, world_level=0.2)
    assert "The World" not in [x["speaker"] for x in quiet]
    r = render_improv(t)
    assert r.startswith("IMPROV TRANSCRIPT") and "- Jun:" in r and render_improv([]) == ""
    assert run_improv(llm, [], "x") == []


# ------------------------------------------------------------------ stylometry + voice coach
def test_stylometry_and_voice_coach(tmp_path):
    text = ('"Okay. Okay. Run," Rin said.\n\n"Why would I run? I have done nothing wrong, and I will not be hurried by anyone," '
            'the Broker replied, folding his hands.\n\nRin shrugged. "Don\'t care. Move."\n\n"You misunderstand the situation entirely," '
            'said the Broker.\n\n"Nope," Rin said.\n\n"I shall explain it to you again, slowly, since it appears necessary," the Broker said.\n\n'
            '"Okay okay okay," Rin said.\n\nJun watched them both.')
    lines = attribute_dialogue(text, ["Rin", "The Broker", "Jun"])
    assert len(lines["Rin"]) == 4 and len(lines["The Broker"]) == 3 and lines["Jun"] == []
    rin, broker = style_vector(lines["Rin"]), style_vector(lines["The Broker"])
    assert rin["words_per_line"] < broker["words_per_line"] and rin["contraction_ratio"] > broker["contraction_ratio"]
    d, pairs = distinctness({"Rin": rin, "The Broker": broker})
    assert d > 0.35 and pairs[0][2] == d
    p = Project.create(tmp_path / "pp", "P", "idea", ["Fantasy"], 300)
    roster = Roster(p.dir)
    for n in ("Rin", "The Broker", "Jun"):
        roster.add(Character(name=n))
    roster.save()
    coach = VoiceCoach(FakeLLM(), p.dir)
    ctx = SceneContext(project=p, number=1, plan=ChapterPlan(number=1, title="t"))
    rev = coach.review(text, ctx)
    assert rev.score == 1.0 and rev.data["speakers"] == ["Rin", "The Broker"]
    coach.remember(text, ctx)
    assert coach.state.samples["Rin"] == 1 and "distinctness" in coach.report()
    same = ('"We should go now," Rin said.\n\n"We should go now," the Broker said.\n\n"Yes, we should go now," Rin said.\n\n'
            '"Yes, we should go now," the Broker said.\n\n"Go now," Rin said.\n\n"Go now," the Broker said.')
    rev2 = coach.review(same, ctx)
    assert rev2.score < 1.0 and "sound alike" in rev2.issues[0]
    coach.remember(same, ctx)
    assert "sounding alike" in coach.brief(ctx) or coach.brief(ctx) == ""  # depends on merged profiles crossing the threshold
    assert coach.review("no dialogue here", ctx) is None


# ------------------------------------------------------------------ pipeline integration
def test_pipeline_learns_from_itself(cfg, db, chapter_text):
    cfg.write.min_acceptable_score = 0.95  # force a revision so a self-preference pair can be recorded
    cfg.write.max_revisions = 1
    t = _trained(cfg, db, chapter_text)
    main, fast = FakeLLM(), FakeLLM()
    # make the revision text differ from the draft so the pair is real
    main.canned = []
    pipe = WritePipeline(cfg, db, llm=main, trainer=t, fast_llm=fast)
    p = pipe.new_project("Ten Seconds", "a courier who can rewind ten seconds", ["Sci-fi"], words=300)
    res = pipe.write_next(p)
    draft_prompt = next(c["prompt"] for c in main.calls_list if c["purpose"] == "draft")
    assert "IMPROV TRANSCRIPT" in draft_prompt and "- Rin:" in draft_prompt and "improv" in res.timings
    # bandit sampled and was rewarded
    b = PromptBandit(p.dir / "bandit.json")
    assert b.state.history and b.state.history[-1]["chapter"] == 1 and 1 in b.state.pending
    front = res.path.read_text(encoding="utf-8")
    assert "settings: {" in front
    # notes recorded for lessons, accepted copy saved
    assert db.notes("Ten Seconds") and (p.dir / "drafts" / "accepted" / "chapter_001.md").exists()
    # feedback -> panel calibration, delayed bandit reward, project model attempt
    pipe.write_next(p)
    pipe.record_feedback(p, 1, views=1000)
    out2 = pipe.record_feedback(p, 2, views=800)
    assert out2["bandit_reward"] == {"chapter": 1, "retention": 0.8} and "panel" in out2
    assert out2["project_model"] == "not enough labelled chapters" and db.feedback("Ten Seconds")[2]["views"] == 800
    assert 1 not in PromptBandit(p.dir / "bandit.json").state.pending
    # lessons and persona reach the system prompt of a new pipeline
    db.add_notes("Ten Seconds", 1, [("reader_panel", "would stop at the opening")] * 3)
    save_persona(cfg.models_dir, AuthorPersona(name="Wet Neon", stances=["debt is weather"]))
    pipe2 = WritePipeline(cfg, db, llm=FakeLLM(), trainer=t, fast_llm=FakeLLM())
    assert "AUTHOR PERSONA (Wet Neon)" in pipe2.guide_compact and "Lessons from your own earlier chapters" in pipe2.guide_compact
    # the schemer saw the relationship graph
    sch = next(c["prompt"] for c in fast.calls_list if c["purpose"] == "schemer:brief")
    assert "Relationship graph" in sch


def test_cli_learning_commands(tmp_path, chapter_text):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "220"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    d = tmp_path / "mine"
    d.mkdir()
    for i in range(3):
        (d / f"{i}.txt").write_text(chapter_text(True), encoding="utf-8")
    (d / "novel.json").write_text(json.dumps({"title": "Mine", "followers": 10}))
    runner.invoke(main, ["--data-dir", data, "import", str(d)])
    r = runner.invoke(main, ["--data-dir", data, "persona", "build", "--from", "local:mine", "--blend", "0.7"])
    assert r.exit_code == 0 and "Wet Neon" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "persona", "show"])
    assert r.exit_code == 0 and "blend: 0.7" in r.output
    r = runner.invoke(main, ["--data-dir", data, "new", "Ten Seconds", "--idea", "a courier", "--genre", "Sci-fi", "--words", "200"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--data-dir", data, "write", "Ten Seconds", "--chapters", "2"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--data-dir", data, "feedback", "Ten Seconds", "1", "--views", "1000"])
    assert r.exit_code == 0 and "recorded" in r.output
    r = runner.invoke(main, ["--data-dir", data, "feedback", "Ten Seconds", "2", "--views", "900", "--comments", "12"])
    assert r.exit_code == 0 and "bandit_reward" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "bandit", "Ten Seconds"])
    assert r.exit_code == 0 and "temperature" in r.output and "current best" in r.output
    r = runner.invoke(main, ["--data-dir", data, "lessons", "Ten Seconds"])
    assert r.exit_code == 0 and "Lessons" in r.output
    r = runner.invoke(main, ["--data-dir", data, "sync-edits", "Ten Seconds"])
    assert r.exit_code == 0 and "no new edits" in r.output
    r = runner.invoke(main, ["--data-dir", data, "finetune"])
    assert r.exit_code == 0 and "finetune_lora.py" in r.output and "Add --run" in r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "arc", "Ten Seconds", "Rin", "--trait", "courage=0.9", "--by", "20"])
    assert r.exit_code == 0 and "arc set" in r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "show", "Ten Seconds", "Rin"])
    assert r.exit_code == 0 and "Arc plan" in r.output and "Mood vector" in r.output
    r = runner.invoke(main, ["--data-dir", data, "report", "--project", "Ten Seconds"])
    assert r.exit_code == 0 and "labelled chapters: 2" in r.output
    r = runner.invoke(main, ["--data-dir", data, "agents", "show", "Ten Seconds", "voice_coach"])
    assert r.exit_code == 0
