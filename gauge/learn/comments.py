"""Reader comments as a learning signal: where readers reacted, and how.

A small transparent lexicon turns comment text into per-chapter numbers (excitement, confusion,
complaint, prediction) that feed chapter features, calibrate the Reader Panel's personas with
how real readers talk, and point the technique absorber at chapters that made people react."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..db import Database

EXCITED = {"omg", "wow", "hype", "amazing", "love", "loved", "awesome", "epic", "finally", "yes", "lets go", "let's go", "insane",
           "chills", "goosebumps", "cried", "screaming", "cliffhanger", "need the next", "more please", "best chapter"}
CONFUSED = {"confused", "confusing", "wait what", "what happened", "don't understand", "dont understand", "plot hole", "makes no sense",
            "how did", "why did", "huh", "unclear", "lost me"}
COMPLAINT = {"boring", "slow", "filler", "dragging", "drag", "rushed", "annoying", "stupid", "out of character", "ooc", "typo", "grammar",
             "too short", "dropped", "disappointed", "meh", "skip", "skipped"}
PREDICTION = {"calling it", "i bet", "i think", "theory", "predict", "probably", "gonna", "going to", "mark my words", "plot twist incoming"}
LAUGH = {"lol", "lmao", "haha", "hahaha", "rofl", "😂", "🤣", "dead", "hilarious"}


def analyze_comment(text: str) -> dict[str, int]:
    low = text.lower()
    return {
        "excited": int(any(w in low for w in EXCITED)) + low.count("!") // 2,
        "confused": int(any(w in low for w in CONFUSED)) + low.count("?") // 2,
        "complaint": int(any(w in low for w in COMPLAINT)),
        "prediction": int(any(w in low for w in PREDICTION)),
        "laugh": int(any(w in low for w in LAUGH)),
    }


def chapter_comment_features(comments: list[dict[str, Any]]) -> dict[str, float]:
    n = len(comments)
    if n == 0:
        return {"comment_count": 0.0, "comment_excited": 0.0, "comment_confused": 0.0, "comment_complaint": 0.0,
                "comment_prediction": 0.0, "comment_laugh": 0.0, "comment_len_mean": 0.0}
    tot = Counter()
    lens = []
    for c in comments:
        a = analyze_comment(c["text"])
        for k, v in a.items():
            tot[k] += 1 if v else 0
        lens.append(len(c["text"].split()))
    return {"comment_count": float(n), "comment_excited": tot["excited"] / n, "comment_confused": tot["confused"] / n,
            "comment_complaint": tot["complaint"] / n, "comment_prediction": tot["prediction"] / n, "comment_laugh": tot["laugh"] / n,
            "comment_len_mean": sum(lens) / n}


def persona_samples(db: Database, limit_each: int = 3) -> dict[str, list[str]]:
    """Real comments that sound like each persona, to ground the Reader Panel in how readers talk."""
    buckets: dict[str, list[str]] = {"binge": [], "lore": [], "commenter": [], "hostile": []}
    for c in db.all_comments(limit=4000):
        t = c["text"].strip()
        if not (8 <= len(t.split()) <= 60):
            continue
        a = analyze_comment(t)
        if a["excited"] and len(buckets["binge"]) < limit_each:
            buckets["binge"].append(t)
        elif a["confused"] and len(buckets["lore"]) < limit_each:
            buckets["lore"].append(t)
        elif a["complaint"] and len(buckets["hostile"]) < limit_each:
            buckets["hostile"].append(t)
        elif a["prediction"] and len(buckets["commenter"]) < limit_each:
            buckets["commenter"].append(t)
        if all(len(v) >= limit_each for v in buckets.values()):
            break
    return {k: v for k, v in buckets.items() if v}


def reaction_hotspots(db: Database, novel_key: str, top: int = 5) -> list[tuple[int, float]]:
    """Chapters whose comments show the most excitement: candidates for technique absorption."""
    counts = db.comment_counts(novel_key)
    scored = []
    for idx, n in counts.items():
        f = chapter_comment_features(db.comments_for(novel_key, idx))
        scored.append((idx, n * (0.5 + f["comment_excited"] + 0.5 * f["comment_laugh"])))
    scored.sort(key=lambda t: -t[1])
    return scored[:top]


_TAG = re.compile(r"<[^>]+>")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub(" ", text)).strip()
