"""The character blueprint and the character school: what a new character needs, and teaching Gauge to build one."""
import json

import pytest

from gauge.gui import StudioController
from gauge.learn.character_school import MIN_SCORE, CharacterSchool, looks_like_sheet
from gauge.llm import FakeLLM
from gauge.models import ChapterRecord, NovelRecord
from gauge.write import blueprint as bp
from gauge.write.characters import Character, CharacterCreator, PersonalityMatrix

FILLED = """# Gauge character sheet
# notes like this are ignored

## Identity
name: Marina Vale
aliases: the Fence; Mar
role: antagonist
concept: a fence who launders stolen memories and has started keeping the best ones

## Outside
appearance: Silver rings on every finger of the left hand and none on the right. She wears a dead man's coat, cut down to fit.
habits: turns a ring when she is pricing something; never sits with her back to a window

## Past
backstory: She ran the night market's lost-and-found as a girl and learned that everything has a buyer. Her brother sold
 his memory of her to pay a debt, and she bought it back without telling him what it was.
wound: her brother looked at her and did not know her face

## Engine
want: to own every memory of the market before the Broker does
need: to be known by one person who was not paid for it
lie: anything that can be forgotten was never really yours
flaw: she prices people before she listens to them
stakes: the one memory she bought back is the first thing the Broker would take
fears: being forgotten; owing anyone
values: a fair price; keeping her word
lines: sell a child's memory; break a deal she set herself

## Temperament
openness: 0.7
conscientiousness: 0.9
warmth: 0.15
dominance: 0.85
honesty: 0.8
cynicism: 0.9
contradiction: she will cheat a buyer on price and never once on what she promised

## Voice
register: formal
sentence_length: long
vocabulary: ledger and auction words; never says please
tics: quotes a price when she is frightened
catchphrases: Everything has a buyer.
swears: no
humor_style: dry
avoids: her brother

## Capability
skills: values a memory by touch, but cannot read her own; knows who owes whom in the market

## Situation
mood: watchful
goal: buy the courier's debt before the week is out
knows: who broke the seal on the bag
secrets: she keeps a memory she had no right to buy

## Connections and change
relationship: Rin Okada = the courier she means to own and half wants to warn | trust 0.3, fear 0.2
arc: from owning what she cannot keep, through losing the one memory that was hers, toward giving something away for nothing
"""


def _marina() -> Character:
    return bp.character_from_text(FILLED)


# ------------------------------------------------------------------ the blueprint
def test_the_breakdown_covers_everything_and_every_item_has_a_test():
    text = bp.breakdown_markdown()
    for section in ("Identity", "Outside", "Past", "Engine", "Temperament", "Voice", "Capability", "Situation",
                    "Connections and change"):
        assert f"## {section}" in text
    for it in bp.ITEMS.values():
        assert it.label in text and it.what and it.test
    assert len(bp.ITEMS) == 33 and len(bp.SECTIONS) == 9
    # A minor character needs far less than a lead.
    lead, minor = bp.checklist("protagonist"), bp.checklist("minor")
    assert "the lie they believe" in lead and "the lie they believe" not in minor and len(minor) < len(lead) / 4


def test_a_filled_sheet_is_read_in_full():
    c = _marina()
    assert c.name == "Marina Vale" and c.role == "antagonist" and c.aliases == ["the Fence", "Mar"]
    assert c.matrix.wants.startswith("to own every memory") and c.matrix.wound.endswith("did not know her face")
    assert c.lie and c.stakes and c.contradiction and c.arc and len(c.habits) == 2 and len(c.skills) == 2
    assert "bought it back" in c.backstory                      # a value that runs on to the next line
    assert c.matrix.warmth == 0.15 and c.matrix.cynicism == 0.9 and c.matrix.extraversion == 0.5
    assert c.voice.tone == "formal" and c.voice.swears is False and c.voice.avoids_talking_about == ["her brother"]
    rel = c.relationships["Rin Okada"]
    assert rel.stance.startswith("the courier she means to own") and rel.trust == 0.3 and rel.fear == 0.2
    a = bp.audit(c, cast_size=1)
    assert a.score == 1.0 and not a.gaps(), a.summary()


def test_a_sheet_survives_being_written_out_and_read_back():
    c = _marina()
    again = bp.character_from_text(bp.sheet_text(c, notes=True))
    assert bp.sheet_json(again) == bp.sheet_json(c)
    # JSON works too, and "humor" is the trait when it is a number and the style when it is a word.
    assert bp.sheet_json(bp.character_from_text(json.dumps(bp.sheet_json(c)))) == bp.sheet_json(c)
    d = bp.parse_sheet("name: X\nhumor: dry\nhumor: 0.9\ncourage: 8\nloyalty: 15%")
    assert d["voice"]["humor_style"] == "dry" and (d["matrix"]["humor"], d["matrix"]["courage"], d["matrix"]["loyalty"]) == (0.9, 0.8, 0.15)
    with pytest.raises(ValueError):
        bp.character_from_text("role: minor\nconcept: nobody at all")


def test_the_audit_names_what_is_missing_thin_or_repeated():
    blank = bp.character_from_text(bp.blank_sheet("Nobody", "protagonist"))
    a = bp.audit(blank, cast_size=2)
    assert a.score < 0.2 and {"want", "need", "wound", "lie", "traits", "relationships", "arc"} <= {f.key for f in a.missing}
    assert "missing:" in a.summary()

    c = _marina()
    c.appearance = "Tall."                                      # too short to count
    c.matrix.needs = c.matrix.wants                             # a need that repeats the want gives no story
    for axis in ("conscientiousness", "warmth", "dominance", "cynicism", "honesty"):
        setattr(c.matrix, axis, 0.5)                            # leaves one mild lean: nobody in particular
    thin = {f.key: f.note for f in bp.audit(c, 1).thin}
    assert set(thin) == {"appearance", "need", "traits"}
    assert "repeats the want" in thin["need"] and "extreme" in thin["traits"]

    # A minor character is only asked for a name, a role, a concept and whether they swear.
    bit = Character(name="Dockhand", role="minor", concept="a dockhand who saw the bag change hands")
    assert bp.audit(bit).score == 1.0
    # Nobody to relate to yet: no relationship is demanded.
    lone = _marina()
    lone.relationships = {}
    assert "relationships" not in bp.audit(lone, cast_size=0).gaps() and "relationships" in bp.audit(lone, cast_size=1).gaps()


def test_new_fields_reach_the_sheet_and_the_character_prompt():
    c = _marina()
    md, sp = c.sheet_markdown(), c.system_prompt()
    assert "anything that can be forgotten" in md and "turns a ring" in md and "values a memory by touch" in md
    assert "anything that can be forgotten" in sp and "turns a ring" in sp


# ------------------------------------------------------------------ the school
def test_the_school_admits_complete_sheets_and_refuses_thin_ones(cfg):
    school = CharacterSchool(cfg.models_dir)
    res = school.add(_marina(), genres=["Thriller"])
    assert res["added"] and res["score"] == 1.0 and "Added Marina Vale" in res["message"]
    thin = school.add(bp.character_from_text(bp.blank_sheet("Nobody", "supporting")))
    assert not thin["added"] and thin["score"] < MIN_SCORE
    assert "was not added" in thin["message"] and "Missing:" in thin["message"] and "Want" in thin["message"]
    assert school.add(bp.character_from_text(bp.blank_sheet("Nobody", "supporting")), force=True)["added"]
    # Adding the same character again replaces it; the school is read back from disk.
    assert school.add(_marina())["replaced"]
    again = CharacterSchool(cfg.models_dir)
    assert [e.name for e in again.examples()] == ["Nobody", "Marina Vale"]
    assert again.examples()[1].line().startswith("Marina Vale  -  antagonist, written by you, 100% complete")
    assert again.remove("nobody") and not again.remove("nobody") and again.stats()["examples"] == 1


def test_examples_carry_no_story_state(cfg):
    c = _marina()
    c.created_chapter = 12
    school = CharacterSchool(cfg.models_dir)
    school.add(c, source="project:Ten Seconds")
    stored = json.loads(school.path.read_text())["examples"][0]
    assert set(stored["sheet"]["state"]) == {"mood", "goal", "knows", "secrets"} and "created_chapter" not in stored["sheet"]


def test_picking_prefers_the_same_role_then_genre_then_the_authors_own(cfg):
    school = CharacterSchool(cfg.models_dir)
    villain = _marina()
    hero = bp.character_from_text(FILLED.replace("Marina Vale", "Tomas Reyes").replace("role: antagonist", "role: protagonist"))
    side = bp.character_from_text(FILLED.replace("Marina Vale", "Old Nan").replace("role: antagonist", "role: supporting"))
    school.add(villain, genres=["thriller"])
    school.add(hero, source="project:Ten Seconds", genres=["sci-fi"])
    school.add(side, genres=["sci-fi"])
    assert [e.name for e in school.pick("antagonist", k=1)] == ["Marina Vale"]
    assert [e.name for e in school.pick("protagonist", ["sci-fi"], k=2)] == ["Tomas Reyes", "Old Nan"]
    assert [e.name for e in school.pick("minor", ["sci-fi"], k=1)] == ["Old Nan"]         # the author's sheet wins the tie
    assert "Marina Vale" not in [e.name for e in school.pick("antagonist", k=3, exclude="marina vale")]
    block = school.guidance("antagonist", ["thriller"], k=1)
    assert "WHAT A FINISHED ANTAGONIST NEEDS" in block and "EXAMPLES the author approved" in block
    assert "Do not reuse" in block and "anything that can be forgotten" in block and "Tomas Reyes" not in block
    assert school.guidance("antagonist", k=0, checklist=False) == ""


def test_training_rows_teach_the_breakdown_the_build_and_the_critique(cfg):
    from gauge.native.train import instruction_rows

    school = CharacterSchool(cfg.models_dir)
    before = school.export_training()                           # the breakdown is taught before any example exists
    assert before["rows"] > 40 and before["examples"] == 0
    school.add(_marina())
    rows = [json.loads(line) for line in school.dataset.read_text(encoding="utf-8").splitlines()]
    assert len(rows) > before["rows"] + 25 and school.stats()["rows"] == len(rows)
    by = {}
    for r in rows:
        by.setdefault(r["instruction"], []).append(r)
    whole = by["Create a character sheet for a serial web novel."][0]
    assert whole["input"].startswith("Name: Marina Vale\nRole: antagonist") and json.loads(whole["output"])["lie"]
    parts = {r["input"].split("Part: ")[1].split(".")[0]: r["output"] for r in by["Build one part of a character."]}
    assert parts["Wound"] == "her brother looked at her and did not know her face" and "The lie they believe" in parts
    assert parts["how they stand with Rin Okada"].startswith("the courier she means to own")
    checks = by["Check this character sheet against the blueprint. What is missing, and what would fix it?"]
    assert len(checks) == 3 and checks[-1]["output"].startswith("Nothing is missing.")
    for ex in checks[:2]:                                        # what the answer names is exactly what was taken away
        named = [line.split("Missing: ")[1].split(".")[0] for line in ex["output"].splitlines()]
        assert len(named) == 3
        partial = bp.character_from_text(ex["input"])
        assert sorted(bp.ITEMS[k].label for k in bp.audit(partial, 1).gaps()) == sorted(named)
    assert any(r["input"] == "Section: Engine" and "Want" in r["output"] for r in by["What does a newly built character need?"])
    # The GaugeLM trainer reads them.
    taught = instruction_rows(cfg.models_dir / "datasets")
    assert len(taught) == len(rows) and all(kind.startswith("chat:") for kind, _ in taught)


def test_sheet_files_and_folders_are_added_and_bad_ones_do_not_stop_the_rest(cfg, tmp_path):
    folder = tmp_path / "sheets"
    folder.mkdir()
    (folder / "marina.txt").write_text(FILLED, encoding="utf-8")
    (folder / "bom.txt").write_text("﻿" + FILLED.replace("Marina Vale", "Tomas Reyes"), encoding="utf-8")   # Notepad adds this
    (folder / "broken.json").write_text("{not json", encoding="utf-8")
    (folder / "empty.txt").write_text(bp.blank_sheet(), encoding="utf-8")
    (folder / "photo.png").write_bytes(b"\x89PNG")
    school = CharacterSchool(cfg.models_dir)
    results = school.add_path(folder)
    assert sorted(r["name"] for r in results if r["added"]) == ["Marina Vale", "Tomas Reyes"]
    refused = {r["file"]: r["message"] for r in results if not r["added"]}
    assert set(refused) == {"broken.json", "empty.txt"} and "no name" in refused["empty.txt"]
    assert looks_like_sheet(folder / "marina.txt") and looks_like_sheet(folder / "bom.txt")
    assert not looks_like_sheet(folder / "photo.png") and not looks_like_sheet(folder / "broken.json")
    assert not school.add_path(tmp_path / "nowhere.txt")[0]["added"]


def test_studying_a_book_needs_permission_to_train_on_it(cfg, db):
    school = CharacterSchool(cfg.models_dir)
    para = ("Rin Okada counted the exits before she sat down, the way she always did, and rubbed the scar on her palm. "
            "The bag was still sealed and the debt was still hers, and nobody in the room knew she had already run once.")
    for source, title in (("gutenberg", "The Open Book"), ("royalroad", "Someone Else's Serial")):
        n = NovelRecord(source=source, source_id=title.lower().replace(" ", "-"), title=title)
        db.upsert_novel(n)
        for i in (1, 2):
            db.upsert_chapter(ChapterRecord(novel_key=n.key, index=i, text="\n\n".join([para, "Nobody else was there at all."] * 4)))
    refused = school.study(db, FakeLLM(), "Someone Else's Serial", "Rin Okada", "protagonist")
    assert not refused["added"] and refused["permission"] == "third_party" and "not yours to train on" in refused["message"]
    assert not school.examples()
    ok = school.study(db, FakeLLM(), "open book", "Rin Okada", "protagonist")
    assert ok["added"] and ok["permission"] == "public_domain", ok["message"]
    assert school.examples()[0].source == "study:The Open Book" and school.examples()[0].name == "Rin Okada"
    assert "does not appear" in school.study(db, FakeLLM(), "The Open Book", "Zebulon")["message"]
    assert "no book matching" in school.study(db, FakeLLM(), "Missing Title", "Rin")["message"]


# ------------------------------------------------------------------ creation uses what was taught
class _Recorder(FakeLLM):
    """The test model, remembering every prompt it was given."""

    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return super().generate(prompt, **kw)


class _Sparse(_Recorder):
    """A model that forgets the deeper fields on the first pass, like a small local model does."""

    def generate(self, prompt, **kw):
        out = super().generate(prompt, **kw)
        if "create a character sheet" in prompt.lower():
            d = json.loads(out)
            for k in ("habits", "skills", "lie", "stakes", "arc", "contradiction"):
                d.pop(k, None)
            d["matrix"]["wound"] = ""
            return json.dumps(d)
        return out


def test_creation_shows_the_checklist_and_the_closest_examples(cfg):
    school = CharacterSchool(cfg.models_dir)
    school.add(_marina())
    llm = _Recorder()
    CharacterCreator(llm, school).from_concept("The Broker", "a man who buys debts", "antagonist", others=["Rin"])
    first = llm.prompts[0]
    assert "WHAT A FINISHED ANTAGONIST NEEDS" in first and "Example 1 (antagonist)" in first and "Marina Vale" in first
    # Without a school the checklist still goes in; with everything off the prompt is as it always was.
    plain = _Recorder()
    CharacterCreator(plain).from_concept("Jun", "a sister", "supporting")
    assert "WHAT A FINISHED SUPPORTING NEEDS" in plain.prompts[0] and "EXAMPLES" not in plain.prompts[0]
    bare = _Recorder()
    CharacterCreator(bare, checklist=False, repair=False).from_concept("Jun", "a sister", "supporting")
    assert "WHAT A FINISHED" not in bare.prompts[0] and len(bare.prompts) == 1


def test_gaps_are_filled_with_one_more_request_and_nothing_good_is_overwritten():
    llm = _Sparse()
    c = CharacterCreator(llm).from_concept("Rin Okada", "a courier who owes the wrong people", "protagonist", others=["Jun"])
    assert len(llm.prompts) == 2 and "Complete this character sheet" in llm.prompts[1]
    keys = llm.prompts[1].split("holding just these keys: ")[1].strip().split(", ")
    assert {"habits", "skills", "lie", "stakes", "arc", "contradiction", "wound"} <= set(keys)
    assert "want" not in keys and "name" not in keys            # what passed is not asked for again
    assert c.lie and c.stakes and c.arc and c.habits and c.skills and c.matrix.wound and c.name == "Rin Okada"
    assert c.matrix.wants == "clear the debt"                   # and is not changed
    assert bp.audit(c, 1).score == 1.0

    # Fixed trait values are never touched by a repair.
    seed = PersonalityMatrix()                                   # every dial in the middle: "missing" to the audit
    seeded = CharacterCreator(_Sparse()).from_concept("Rin Okada", "a courier", "protagonist", seed_matrix=seed)
    assert seeded.matrix.axes() == seed.axes()

    off = _Sparse()
    left = CharacterCreator(off, repair=False).from_concept("Rin Okada", "a courier", "protagonist")
    assert len(off.prompts) == 1 and not left.lie


def test_the_pipeline_creates_characters_with_the_school(cfg, db):
    from gauge.write import WritePipeline

    CharacterSchool(cfg.models_dir).add(_marina())
    pipe = WritePipeline(cfg, db)
    creator = pipe.character_creator()
    assert creator.school is not None and creator.examples == 2 and creator.checklist and creator.repair
    cfg.llm.provider = "native"                                  # GaugeLM's context is too small for examples
    small = WritePipeline(cfg, db).character_creator()
    assert small.examples == 0 and not small.checklist and not small.repair


# ------------------------------------------------------------------ the command line
def test_character_school_commands(tmp_path):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "200"})

    def run(*args):
        r = runner.invoke(main, ["--data-dir", data, *args])
        assert r.exit_code == 0, r.output
        return r.output

    run("init", "--provider", "fake")
    assert "## Engine" in run("character", "blueprint") and "the lie they believe" in run("character", "blueprint", "--role", "antagonist")
    blank = tmp_path / "blank.txt"
    run("character", "sheet", "--name", "Marina Vale", "--role", "antagonist", "--out", str(blank))
    assert blank.read_text().startswith("# Gauge character sheet") and "name: Marina Vale" in blank.read_text()
    out = run("character", "teach", str(blank))                  # an empty sheet is refused, with the reason
    assert "was not added" in out and "0 added" in out
    sheet = tmp_path / "marina.txt"
    sheet.write_text(FILLED, encoding="utf-8")
    out = run("character", "teach", str(sheet), "--genre", "thriller")
    assert "Added Marina Vale" in out and "1 added" in out
    assert "Marina Vale" in run("character", "school") and "antagonist" in run("character", "school")

    run("new", "Ten Seconds", "--idea", "a courier", "--genre", "Sci-fi", "--words", "200")
    assert "Added Marina Vale to Ten Seconds" in run("character", "add", "Ten Seconds", str(sheet))
    r = runner.invoke(main, ["--data-dir", data, "character", "add", "Ten Seconds", str(sheet)])
    assert r.exit_code == 1 and "already in" in r.output
    run("character", "roll", "Ten Seconds", "Dice", "--seed", "1")
    out = run("character", "check", "Ten Seconds")
    assert "Marina Vale (antagonist): 33 of 33" in out and "Dice (supporting)" in out and "missing" in out
    out = run("character", "check", "Ten Seconds", "Dice", "--fix")
    assert "filled Dice" in out and "Want" in out
    assert "role: antagonist" in run("character", "sheet", "--from", "Ten Seconds", "Marina Vale")
    out = run("character", "promote", "Ten Seconds")
    assert "Marina Vale" in out and "Dice" in out
    assert "Removed" in run("character", "school", "--remove", "Marina Vale", "--export")


# ------------------------------------------------------------------ the Studio
def test_the_studio_teaches_characters(cfg, db, tmp_path):
    ctl = StudioController(cfg, db)
    text = ctl.character_breakdown()
    assert "What a newly built character needs" in text and "The lie they believe" in text and "**" not in text
    saved = ctl.save_blank_sheet(str(tmp_path / "my-character"))
    assert saved["path"].endswith("my-character.txt") and "Notepad" in saved["message"]
    assert ctl.character_examples() == []
    sheet = tmp_path / "marina.txt"
    sheet.write_text(FILLED, encoding="utf-8")
    res = ctl.teach_characters([str(sheet), saved["path"]])
    assert res["added"] == 1 and res["refused"] == 1 and res["names"] == ["Marina Vale"]
    assert "Added 1 character" in res["message"] and "was not added" in res["message"] and "holds 1 example" in res["message"]
    assert ctl.character_examples()[0].startswith("Marina Vale  -  antagonist")
    assert ctl.teach_characters([])["message"].startswith("Nothing was added")
    assert ctl.learn_from_cast("")["message"] == "Open or start a story first."


def test_a_sheet_dropped_in_with_the_books_goes_to_the_school_not_the_library(cfg, db, tmp_path):
    ctl = StudioController(cfg, db)
    sheet = tmp_path / "marina.txt"
    sheet.write_text(FILLED, encoding="utf-8")
    story = tmp_path / "Short Story.txt"
    story.write_text("A quiet story about a lighthouse keeper. " * 400, encoding="utf-8")
    res = ctl.add_material([str(sheet), str(story)])
    assert res["books"] == 1 and res["characters"] == 1 and "Added 1 character" in res["message"]
    assert [r["title"] for r in ctl.library()] == ["Short Story"] and len(ctl.character_examples()) == 1


def test_learning_from_a_storys_cast(cfg, db):
    ctl = StudioController(cfg, db)
    name = ctl.start_story("A courier who can rewind ten seconds", name="Ten Seconds")
    if not ctl.characters(name):                                 # the cast is created with the first chapter
        ctl.write_next(name)
    assert ctl.characters(name)
    res = ctl.learn_from_cast(name)
    assert res["added"] >= 1 and "for Gauge to learn from" in res["message"], res
    assert all("from your story Ten Seconds" in line for line in ctl.character_examples())


def test_training_gaugelm_from_the_studio_includes_the_character_lessons(cfg, db, tmp_path):
    from tests.test_teach import _novel_folder, _small_gaugelm

    _small_gaugelm(cfg)
    ctl = StudioController(cfg, db)
    ctl.add_material([str(_novel_folder(tmp_path))])
    ctl.teach_characters([str(_write(tmp_path / "marina.txt", FILLED))])
    seen: list[dict] = []
    res = ctl.train_gaugelm(minutes=0.01, progress=seen.append, teach=False)
    assert res["trained"] and res["step"] > 0, res
    rows = (cfg.models_dir / "datasets" / "characters.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) > 70 and res["examples"] >= len(rows) - 5, (len(rows), res)   # near-duplicates are dropped once


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_the_guide_lists_every_item_of_the_blueprint():
    from pathlib import Path

    guide = (Path(__file__).parent.parent / "docs" / "CHARACTER-SCHOOL.md").read_text(encoding="utf-8")
    for it in bp.ITEMS.values():
        assert f"**{it.label}**" in guide and it.test in guide, it.key
