"""Characters as independent agents: matrix, roster, agent, pipeline integration, CLI."""
import json

from gauge.learn import Trainer
from gauge.llm import FakeLLM, HashingEmbedder
from gauge.write import WritePipeline
from gauge.write.characters import (
    Character,
    CharacterAgent,
    CharacterCreator,
    PersonalityMatrix,
    Roster,
    Voice,
    character_from_sheet,
    render_cast,
)


def test_matrix_tendencies_and_random():
    m = PersonalityMatrix(impulsivity=0.9, conscientiousness=0.2, honesty=0.9, courage=0.1, humor=0.8)
    t = " ".join(m.tendencies())
    assert "acts before planning" in t and "cannot hold a lie" in t and "freezes or flees" in t and "jokes" in t
    r1, r2 = PersonalityMatrix.random(7), PersonalityMatrix.random(7)
    assert r1.axes() == r2.axes() and all(0 <= v <= 1 for v in r1.axes().values())
    assert PersonalityMatrix.random(1).axes() != PersonalityMatrix.random(2).axes()
    d = PersonalityMatrix.from_dict({"courage": "1.7", "values": "duty; family", "wants": "out"})
    assert d.courage == 1.0 and d.values == ["duty", "family"] and d.wants == "out"
    assert "Traits:" in m.describe() and "Tendencies:" in m.describe()


def test_mentions_and_roster_detection(tmp_path):
    rin = Character(name="Rin Okafor", aliases=["the courier"], role="protagonist")
    jun = Character(name="Jun", role="supporting")
    broker = Character(name="The Broker", role="antagonist")
    marina = Character(name="Marina", role="minor")
    assert rin.mentioned_in("Rin ran.") and rin.mentioned_in("the courier bag") and not rin.mentioned_in("Rinse it")
    assert jun.mentions("Jun said. Jun left. Junior stayed.") == 2
    roster = Roster(tmp_path)
    for c in (jun, broker, rin, marina):
        roster.add(c)
    assert "Jun" in rin.relationships and "Rin Okafor" in jun.relationships  # symmetric slots
    present = roster.present_in(["The Broker calls Jun twice. Jun answers."])
    assert [c.name for c in present][:3] == ["Rin Okafor", "Jun", "The Broker"]  # protagonist always first
    assert marina not in present
    assert roster.get("rin").name == "Rin Okafor" and roster.get("the courier") is rin and roster.get("nobody") is None
    roster.save()
    again = Roster.load(tmp_path)
    assert set(again.names()) == {"Rin Okafor", "Jun", "The Broker", "Marina"}
    assert (tmp_path / "characters.md").read_text(encoding="utf-8").startswith("# Characters")
    assert again.remove("Marina") and not (tmp_path / "characters" / "marina.json").exists()


def test_sheet_parsing_and_system_prompt():
    sheet = json.loads(FakeLLM().generate("Create a character sheet\nName: Rin Okafor\n", json_mode=True))
    c = character_from_sheet("Rin Okafor", sheet, "protagonist", "a courier")
    assert c.name == "Rin Okafor" and c.aliases == ["Rin"] and c.matrix.impulsivity == 0.85
    assert c.voice.tone == "clipped" and c.voice.swears and c.state.secrets == ["she can rewind"]
    assert c.relationships["Jun"].affection == 0.95
    sp = c.system_prompt()
    assert sp.startswith("You are Rin Okafor") and "acts before planning" in sp and "Jun: trust 0.9" in sp
    assert "## Rin Okafor" in c.sheet_markdown()


def test_agent_brief_review_remember_chat(tmp_path):
    llm = FakeLLM()
    c = Character(name="Rin", role="protagonist", matrix=PersonalityMatrix(lines_wont_cross=["abandon Jun"]),
                  voice=Voice(tone="clipped"))
    roster = Roster(tmp_path)
    roster.add(c)
    roster.add(Character(name="Jun"))
    agent = CharacterAgent(llm, c)
    brief = agent.brief({"number": 1, "title": "Ten Seconds", "goal": "escape", "beats": ["run"], "others": ["Jun"]})
    assert brief.goal == "get out with the bag" and "hand the bag over" in brief.refuses and "abandon Jun" in brief.refuses
    rendered = render_cast([brief], roster)
    assert "### Rin" in rendered and "will NOT:" in rendered and "friction with the plan" in rendered
    assert llm.calls[-1]["system"].startswith("You are Rin")  # the agent speaks as the character, not the author
    rev = agent.review("Okay, Rin said.", 1)
    assert rev.consistency == 0.8 and rev.issues[0].quote == "Okay"
    agent.remember("Rin ran. Jun waited.", 1, roster)
    st = c.state
    assert st.mood == "shaken" and st.last_seen_chapter == 1 and "someone knows my name" in st.knows
    assert st.memories[0].chapter == 1 and "bruised shoulder" in st.condition
    assert abs(c.relationships["Jun"].affection - 0.55) < 1e-9 and c.relationships["Jun"].history[0].startswith("ch 1:")
    assert "One more time" in agent.chat("Why did you run?")
    # a failing LLM degrades gracefully
    broken = FakeLLM(canned=["not json at all"])
    r2 = CharacterAgent(broken, c).review("x", 2)
    assert r2.consistency == 1.0 and r2.issues == []


def test_creator_roll_and_seeded_concept():
    creator = CharacterCreator(FakeLLM())
    rolled = creator.roll("Dice", seed=3)
    assert rolled.matrix.axes() == PersonalityMatrix.random(3).axes() and rolled.role == "supporting"
    seeded = creator.from_concept("Rin Okafor", "a courier", "protagonist", seed_matrix=PersonalityMatrix.random(3))
    assert seeded.matrix.courage == PersonalityMatrix.random(3).courage  # seed wins over the LLM's numbers
    assert seeded.voice.tone == "clipped"  # but the LLM still fleshes out voice etc.


def _train(cfg, db, chapter_text):
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    return t


def test_pipeline_activates_cast_and_updates_memory(cfg, db, chapter_text):
    cfg.write.min_acceptable_score = 0.9  # force the revision path so character notes reach the reviser
    cfg.write.max_revisions = 1
    t = _train(cfg, db, chapter_text)
    llm = FakeLLM()
    pipe = WritePipeline(cfg, db, llm=llm, trainer=t)
    p = pipe.new_project("Ten Seconds", "a courier who can rewind time", ["Sci-fi"], words=300)
    roster = pipe.roster(p)
    # bible protagonist 'Rin, a courier...' -> 'Rin'; supporting Jun and The Broker from bible.characters
    assert set(roster.names()) == {"Rin", "Jun", "The Broker"}
    assert roster.get("Rin").role == "protagonist" and roster.get("The Broker").role == "supporting"
    res = pipe.write_next(p)
    assert "Rin" in res.cast
    draft_prompt = next(c["prompt"] for c in llm.calls if "CAST IN THIS SCENE" in c["prompt"])
    assert "### Rin" in draft_prompt and "will NOT:" in draft_prompt
    assert res.character_reviews and res.character_reviews[0].consistency == 0.8
    front = res.path.read_text(encoding="utf-8")
    assert 'cast: ["Rin"' in front and "character_consistency: 0.8" in front
    roster = pipe.roster(p)  # reloaded from disk: memory persisted
    rin = roster.get("Rin")
    assert rin.state.last_seen_chapter == 1 and rin.state.mood == "shaken" and rin.state.memories
    assert (p.dir / "characters.md").exists()
    # character problems were fed into revision prompts
    revise_prompts = [c["prompt"] for c in llm.calls if "Revise the chapter below" in c["prompt"]]
    assert revise_prompts and any("out of character" in rp for rp in revise_prompts)
    # a hand-made character shows up when the outline names them
    pipe.create_character(p, "Marina Vale", "a fence who sells rewound goods", "antagonist", seed=5)
    p.state.outline[1].goal = "Marina Vale offers Rin a deal"
    p.save()
    res2 = pipe.write_next(p)
    assert "Marina Vale" in res2.cast and "Rin" in res2.cast


def test_characters_can_be_disabled(cfg, db, chapter_text):
    cfg.write.characters_enabled = False
    t = _train(cfg, db, chapter_text)
    llm = FakeLLM()
    pipe = WritePipeline(cfg, db, llm=llm, trainer=t)
    p = pipe.new_project("Quiet", "a quiet story", ["Drama"], words=200)
    assert not pipe.roster(p).characters
    res = pipe.write_next(p)
    assert res.cast == [] and res.character_reviews == []


def test_cli_character_commands(tmp_path, chapter_text):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "200"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    r = runner.invoke(main, ["--data-dir", data, "new", "Ten Seconds", "--idea", "a courier", "--genre", "Sci-fi", "--words", "200"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "list", "Ten Seconds"])
    assert r.exit_code == 0 and "Rin" in r.output and "protagonist" in r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "new", "Ten Seconds", "Marina Vale", "--concept", "a fence", "--role", "antagonist", "--seed", "9"])
    assert r.exit_code == 0 and "## Marina Vale" in r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "roll", "Ten Seconds", "Dice", "--seed", "1"])
    assert r.exit_code == 0 and "## Dice" in r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "set", "Ten Seconds", "Dice", "--trait", "courage=0.95", "--alias", "D", "--mood", "grim"])
    assert r.exit_code == 0 and "courage 0.95" in r.output and "mood grim" in r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "set", "Ten Seconds", "Dice", "--trait", "bogus=1"])
    assert r.exit_code != 0
    r = runner.invoke(main, ["--data-dir", data, "character", "show", "Ten Seconds", "rin"])
    assert r.exit_code == 0 and "**Matrix:**" in r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "chat", "Ten Seconds", "Rin", "--ask", "Why run?"])
    assert r.exit_code == 0 and "One more time" in r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "brief", "Ten Seconds", "Rin"])
    assert r.exit_code == 0 and "wants in this scene" in r.output
    r = runner.invoke(main, ["--data-dir", data, "write", "Ten Seconds"])
    assert r.exit_code == 0 and "cast:" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "remove", "Ten Seconds", "Dice"])
    assert r.exit_code == 0 and "Removed" in r.output
    r = runner.invoke(main, ["--data-dir", data, "character", "show", "Ten Seconds", "Dice"])
    assert r.exit_code != 0
