"""Hand-built lexicons used by the feature extractor.

These are deliberately small and transparent. The learner decides how much each signal
matters; the lexicons only turn prose into countable evidence.
"""
from __future__ import annotations

# Words that signal rising tension / arousal. Used for the tension curve.
TENSION = {
    "scream", "screamed", "blood", "blade", "sword", "gun", "shot", "fire", "burn", "burning",
    "run", "ran", "chase", "attack", "attacked", "strike", "struck", "kill", "killed", "die",
    "died", "death", "dead", "danger", "fear", "terror", "panic", "roar", "roared", "crash",
    "crashed", "explode", "exploded", "explosion", "slam", "slammed", "grab", "grabbed", "fight",
    "fought", "battle", "war", "enemy", "enemies", "monster", "beast", "demon", "blast", "crack",
    "shatter", "shattered", "rip", "ripped", "tear", "tore", "stab", "stabbed", "bleed", "bleeding",
    "trap", "trapped", "ambush", "hunt", "hunted", "flee", "fled", "escape", "desperate", "frantic",
    "suddenly", "instantly", "before", "too late", "no time", "collapse", "collapsed", "fall",
    "fell", "falling", "pain", "agony", "shock", "betray", "betrayed", "secret", "lie", "lied",
}

# Markers of an unresolved ending: the single most consistent predictor of "one more chapter".
CLIFFHANGER = {
    "then", "until", "but", "before", "suddenly", "realized", "realised", "froze", "stopped",
    "stared", "whispered", "smiled", "grinned", "laughed", "door", "voice", "behind", "shadow",
    "message", "notification", "appeared", "vanished", "gone", "disappeared", "screamed", "knew",
    "wrong", "too late", "never", "last", "final", "dead", "alive", "wait", "what", "who", "why",
    "how", "impossible", "name", "remember", "forgot", "lied", "truth", "didn't", "couldn't",
    "wasn't", "unless", "except", "footsteps", "silence", "dark", "light", "eyes", "opened",
}

# Reveal / twist vocabulary.
REVEAL = {
    "revealed", "reveal", "truth", "actually", "real", "really", "secret", "hidden", "all along",
    "the whole time", "turns out", "turned out", "lied", "lying", "mask", "disguise", "identity",
    "remember", "memory", "memories", "father", "mother", "brother", "sister", "son", "daughter",
    "heir", "prophecy", "chosen", "system", "level", "awaken", "awakened", "awakening", "inherit",
}

# Progression / power-fantasy markers (LitRPG, cultivation, system, regression...).
PROGRESSION = {
    "level", "levels", "leveled", "levelled", "skill", "skills", "stat", "stats", "strength",
    "agility", "mana", "qi", "cultivation", "realm", "breakthrough", "rank", "ranked", "tier",
    "class", "title", "quest", "dungeon", "boss", "loot", "experience", "xp", "points", "system",
    "status", "window", "notification", "upgrade", "evolve", "evolved", "evolution", "awaken",
    "bloodline", "technique", "artifact", "sect", "elder", "disciple", "regress", "regression",
    "reincarnated", "reincarnation", "transmigrated", "villain", "villainess", "duke", "duchess",
}

# Emotional payoff / warmth markers (romance, slice-of-life, found family).
WARMTH = {
    "smile", "smiled", "laugh", "laughed", "warm", "warmth", "hug", "hugged", "hand", "hands",
    "kiss", "kissed", "soft", "softly", "gentle", "gently", "heart", "blush", "blushed", "love",
    "loved", "home", "safe", "together", "friend", "friends", "family", "trust", "trusted",
    "promise", "promised", "thank", "thanks", "proud", "tears", "cried", "beautiful", "cute",
}

# Hooky first-sentence openers.
OPENERS = {
    "the", "i", "when", "there", "it", "he", "she", "they", "nobody", "everyone", "no one",
    "someone", "something", "today", "tonight", "by", "on", "in", "at", "for", "after", "before",
}

HOOK_VERBS = {
    "died", "woke", "ran", "screamed", "fell", "killed", "stabbed", "exploded", "disappeared",
    "vanished", "stared", "realized", "realised", "knew", "lied", "burned", "crashed", "opened",
    "appeared", "remembered", "forgot", "whispered", "found", "lost", "broke", "bled", "awoke",
}

GENRE_KEYWORDS = {
    "litrpg": PROGRESSION,
    "action": TENSION,
    "romance": WARMTH,
    "mystery": {"murder", "clue", "detective", "case", "suspect", "body", "witness", "alibi",
                "evidence", "killer", "investigation", "disappeared", "missing"},
    "comedy": {"laughed", "joke", "ridiculous", "absurd", "snorted", "deadpan", "sighed",
               "groaned", "facepalm", "idiot", "nonsense", "sarcastic"},
    "horror": {"blood", "scream", "dark", "darkness", "whisper", "shadow", "cold", "rot",
               "flesh", "bone", "crawl", "crawled", "eyes", "teeth", "silence", "corpse"},
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "was", "were", "be", "been", "being", "are", "am", "it",
    "its", "he", "she", "they", "them", "his", "her", "their", "i", "me", "my", "we", "us", "our",
    "you", "your", "that", "this", "these", "those", "there", "here", "not", "no", "yes", "do",
    "did", "does", "have", "has", "had", "would", "could", "should", "will", "can", "just", "like",
    "into", "out", "up", "down", "over", "about", "than", "too", "very", "s", "t", "d", "ll",
}
