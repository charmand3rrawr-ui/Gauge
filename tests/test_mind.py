"""Batch B: contrastive learning, cross-book character memory, emotional continuity, reader reactions."""
from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.learn.contrastive import build_pairs, contrast, guidance, split_by_label
from gauge.learn.reactions import score_reactions
from gauge.write.character_memory import GlobalCharacterMemory
from gauge.write.characters import Character, PersonalityMatrix, Relationship, Voice
from gauge.write.emotion_continuity import audit


def test_contrastive_guidance_and_split():
    high = [{"hook": 0.8, "cliff": 0.7, "adverb_ratio": 0.01}, {"hook": 0.75, "cliff": 0.65, "adverb_ratio": 0.02}]
    low = [{"hook": 0.3, "cliff": 0.3, "adverb_ratio": 0.09}, {"hook": 0.35, "cliff": 0.25, "adverb_ratio": 0.1}]
    cs = contrast(high, low)
    assert cs and cs[0].feature in {"hook", "cliff", "adverb_ratio"} and abs(cs[0].delta) >= abs(cs[-1].delta)
    g = guidance(high, low, 3)
    assert any("open with more immediate motion" in x for x in g)
    rows = [({"hook": 0.1}, 1.0), ({"hook": 0.9}, 9.0), ({"hook": 0.5}, 5.0)]
    hi, lo = split_by_label(rows, top_frac=0.34)
    assert hi[0]["hook"] == 0.9 and lo[0]["hook"] == 0.1
    pairs = build_pairs(["hit one", "hit two"], ["miss"], "prompt")
    assert pairs and pairs[0]["chosen"] == "hit one" and pairs[0]["rejected"] == "miss"


def test_emotional_continuity_flags():
    # A calm chapter, then a huge mood jump with no event => unearned.
    hist = [
        {"chapter": 1, "pleasure": 0.1, "arousal": 0.0, "dominance": 0.0, "event": 0.1},
        {"chapter": 2, "pleasure": -0.9, "arousal": 0.8, "dominance": -0.5, "event": 0.05},
    ]
    flags = audit(hist)
    assert any(f.kind == "unearned_jump" and f.chapter == 2 for f in flags)
    # Whiplash: up hard, then down hard.
    whip = [
        {"chapter": 1, "pleasure": -0.8, "arousal": 0.0, "dominance": 0.0, "event": 0.9},
        {"chapter": 2, "pleasure": 0.8, "arousal": 0.0, "dominance": 0.0, "event": 0.9},
        {"chapter": 3, "pleasure": -0.8, "arousal": 0.0, "dominance": 0.0, "event": 0.9},
    ]
    assert any(f.kind == "whiplash" for f in audit(whip))
    assert audit([]) == []


def _char(name="Rin"):
    c = Character(name=name, matrix=PersonalityMatrix(), voice=Voice(tics=["says okay twice"]))
    c.state.condition = ["scar across the left palm", "owes the Broker"]
    c.state.arc_progress = "learned to let someone help"
    c.matrix.wound = "her mother left without a word"
    c.relationships["The Broker"] = Relationship(trust=0.05, fear=0.7, stance="the man who owns her debt")
    return c


def test_cross_book_memory_roundtrip(tmp_path):
    mem = GlobalCharacterMemory(tmp_path)
    assert mem.get("Rin") is None and mem.all() == []
    led = mem.absorb(_char(), "Book One")
    assert "scar across the left palm" in led.scars and "says okay twice" in led.habits
    assert "The Broker" in led.grudges and led.wound and "Book One" in led.books
    # A fresh character in a new book carries the scars/habits/grudges.
    fresh = Character(name="Rin", matrix=PersonalityMatrix(), voice=Voice())
    assert mem.carry_into(fresh) is True
    assert "scar across the left palm" in fresh.state.condition
    assert "says okay twice" in fresh.voice.tics
    assert fresh.relationships["The Broker"].trust <= 0.2
    assert fresh.matrix.wound == "her mother left without a word"
    assert mem.carry_into(Character(name="Nobody", matrix=PersonalityMatrix())) is False


def test_reactions_scoring():
    good = [{"text": "omg amazing chapter!!", "at": "2026-01-01T00:00:00"},
            {"text": "loved it, need the next", "at": "2026-01-01T00:30:00"},
            {"text": "epic, chills", "at": "2026-01-01T01:00:00"}]
    bad = [{"text": "boring and slow"}, {"text": "this was filler, disappointed"}, {"text": "meh, skipped"}]
    assert score_reactions(good).score > score_reactions(bad).score
    assert score_reactions(good).sentiment > 0 > score_reactions(bad).sentiment
    assert any("complain" in x for x in score_reactions(bad).lessons)
    assert score_reactions([]).count == 0


def test_cli_character_memory_and_ledger(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.ensure_dirs()
    # Seed a project with a character sheet on disk.
    from gauge.write.project import Project
    p = Project.create(cfg.projects_dir, "Book One", "a courier with a debt", ["Fantasy"], 300)
    from gauge.write.characters import Roster
    r = Roster.load(p.dir)
    r.add(_char())
    r.save()
    runner = CliRunner(env={"COLUMNS": "200"})
    res = runner.invoke(main, ["--data-dir", data, "character", "remember-across", "Book One"])
    assert res.exit_code == 0 and "remembered" in res.output, res.output
    res = runner.invoke(main, ["--data-dir", data, "character", "ledger", "Rin"])
    assert res.exit_code == 0 and "scars" in res.output and "The Broker" in res.output, res.output
