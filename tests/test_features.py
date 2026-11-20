from gauge.features import chapter_features, cliffhanger_score, hook_score, novel_features
from gauge.models import ChapterRecord, NovelRecord
from gauge.text import dialogue_ratio, flesch_reading_ease, sentences, word_count


def test_sentences_and_words():
    s = sentences("Hello there. \"Run!\" she said. Why? Because.")
    assert len(s) == 4
    assert word_count("one two three") == 3


def test_dialogue_ratio_bounds():
    assert dialogue_ratio("") == 0.0
    assert 0.3 < dialogue_ratio('"Hi there friend," he said. "Yes."') <= 1.0


def test_hook_prefers_hooky_opening(chapter_text):
    hooky = hook_score(chapter_text(True))["hook"]
    dull = hook_score(chapter_text(False))["hook"]
    assert hooky > dull


def test_cliffhanger_prefers_unresolved_ending(chapter_text):
    hooky = cliffhanger_score(chapter_text(True))["cliff"]
    dull = cliffhanger_score(chapter_text(False))["cliff"]
    assert hooky > dull


def test_chapter_features_complete(chapter_text):
    ch = ChapterRecord(novel_key="x:1", index=1, title="Chapter 1: The Door", text=chapter_text(True))
    f = chapter_features(ch)
    for key in ("hook", "cliff", "words", "dialogue_ratio", "tension_rising", "readability",
                "first_person_ratio", "title_has_subtitle"):
        assert key in f
    assert f["words"] > 50
    assert f["title_has_subtitle"] == 1.0
    assert all(isinstance(v, float) for v in f.values())


def test_novel_features_aggregate(chapter_text):
    from datetime import datetime, timedelta

    n = NovelRecord(source="local", source_id="a", title="A", blurb="What if you could rewind?", tags=["LitRPG"])
    chs = [ChapterRecord(novel_key=n.key, index=i, text=chapter_text(True)) for i in range(1, 5)]
    feats = [chapter_features(c) for c in chs]
    dates = [datetime(2024, 1, 1) + timedelta(days=2 * i) for i in range(4)]
    nf = novel_features(n, feats, dates)
    assert nf["chapters_analyzed"] == 4
    assert abs(nf["chapters_per_week"] - 3.5) < 1e-6
    assert nf["blurb_question"] == 1.0
    assert nf["tagged_litrpg"] == 1.0
    assert "early_hook" in nf and "mean_cliff" in nf


def test_readability_range():
    assert -50 <= flesch_reading_ease("The cat sat on the mat. It was warm.") <= 121
