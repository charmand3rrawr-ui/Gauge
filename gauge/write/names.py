"""A measured index of proper names, to catch spelling drift across a serial (roadmap 28).

Over a few hundred chapters "Katherine" becomes "Katharine" and "Lowmarket" becomes "Lowmarkt". The
Archivist's fact ledger does not see that: a misspelt name is just a new subject. This module keeps a
count of every proper name the serial has used and flags a new name that sits one slip away from an
established one.

It is deliberately conservative, because two characters called Mara and Maya are not a typo:

* only a name that is already established (used several times, or on the cast list) can be drifted from;
* a variant that is itself established, or on the cast list, is a different name, not a slip;
* long names may differ by one edit (two for very long ones); short names only by a doubled or dropped
  repeated letter, or two letters swapped ("Rin"/"Rinn", "Mara"/"Maar"), never by a changed letter;
* when both spellings are each used several times in the same chapter, they are two people in a scene.

Across 25,200 pairs of distinct common first names the slip rule matches 10, half of them true spelling
variants (Brian/Bryan, Teresa/Theresa). The rest (Julia/Julie, Andrea/Andrew) are why the cast list
matters: a name on it is never reported, and a new character stops being reported once they have been
used a few times.
"""
from __future__ import annotations

import re
from collections import Counter

# Capitalised words that are not names.
_NOT_NAMES = {
    "The", "A", "An", "And", "But", "Or", "If", "Then", "When", "While", "After", "Before", "Because", "Although",
    "He", "She", "They", "We", "You", "It", "His", "Her", "Their", "Our", "Your", "Its", "My", "Me", "Him", "Them",
    "This", "That", "These", "Those", "There", "Here", "What", "Why", "How", "Who", "Where", "Which", "Whose",
    "Yes", "No", "Not", "Now", "Never", "Maybe", "Okay", "Well", "Oh", "Ah", "Please", "Thanks", "Sorry",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March",
    "April", "May", "June", "July", "August", "September", "October", "November", "December", "Chapter", "God",
    "Mr", "Mrs", "Ms", "Dr", "Sir", "Lady", "Lord", "Captain", "Doctor", "Professor", "King", "Queen", "Father",
    "Mother", "Dad", "Mom", "Mum", "Uncle", "Aunt", "North", "South", "East", "West", "English", "Just", "Even",
    "Still", "Only", "All", "Some", "One", "Two", "Three", "Every", "Each", "Nothing", "Something", "Everything",
    "Someone", "Anyone", "Nobody", "Everyone", "Let", "Come", "Go", "Look", "Listen", "Wait", "Stop", "Run",
}
_WORD = re.compile(r"[A-Z][a-zà-öø-ÿ]+(?:-[A-Z][a-zà-öø-ÿ]+)?")
_SENT_START = re.compile(r"(?:^|[.!?…]\s+|[\"“‘'(\[]\s*|\n\s*|—\s*|:\s+)$")


def proper_names(text: str, known: set[str] | None = None) -> Counter:
    """Names in a chapter with their counts. A word counts as a name when it is capitalised somewhere
    other than the start of a sentence, or is already a `known` name; its sentence-opening uses are then
    counted too. (A capital at the start of a sentence proves nothing on its own.)"""
    known = known or set()
    mid: Counter = Counter()
    start: Counter = Counter()
    for m in _WORD.finditer(text):
        w = m.group(0)
        if w in _NOT_NAMES or len(w) < 3:
            continue
        if _SENT_START.search(text[max(0, m.start() - 4):m.start()]):
            start[w] += 1
        else:
            mid[w] += 1
    names = set(mid) | (set(start) & known)
    return Counter({w: mid.get(w, 0) + start.get(w, 0) for w in names})


def _distance(a: str, b: str, limit: int = 2) -> int:
    """Edit distance counting an adjacent swap as one edit, cut off once it exceeds `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def _collapse(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s)


def _is_slip(variant: str, canon: str) -> bool:
    """Is `variant` plausibly a misspelling of `canon` rather than a different name?"""
    v, c = variant.lower(), canon.lower()
    if v == c:
        return False
    if v == c + "s" or c == v + "s":            # a plural or a possessive without its apostrophe
        return False
    n = min(len(v), len(c))
    if n < 5:
        # Short names: only a doubled/dropped repeated letter, or two letters swapped.
        if _collapse(v) == _collapse(c):
            return True
        return len(v) == len(c) and sorted(v) == sorted(c) and _distance(v, c, 1) == 1
    if v[0] != c[0] and n < 7:                   # a different first letter is usually a different name
        return False
    return _distance(v, c, 2) <= (2 if n >= 8 else 1)


def spelling_drift(now: Counter, canon: dict[str, int], protected: set[str] | None = None,
                   min_canon: int = 3) -> list[tuple[str, str, int, int]]:
    """Names in this chapter that look like a slip of an established name.

    `canon` is the serial's running name counts; `protected` are names known to be right (the cast list
    and its aliases), which count as established and are never reported as a slip themselves. Returns
    (variant, canonical, uses of the variant in this chapter, uses of the canonical name so far)."""
    protected = protected or set()
    established = {n: c for n, c in canon.items() if c >= min_canon}
    for p in protected:
        established[p] = max(established.get(p, 0), min_canon)
    out: list[tuple[str, str, int, int]] = []
    for variant, n in now.items():
        if variant in established or variant in protected:
            continue
        best = None
        for name, count in established.items():
            if n >= 3 and now.get(name, 0) >= 3:     # both spellings well used in one chapter: two people
                continue
            if count >= 3 * (canon.get(variant, 0) + 1) and _is_slip(variant, name):
                if best is None or count > best[1]:
                    best = (name, count)
        if best:
            out.append((variant, best[0], n, best[1]))
    return sorted(out, key=lambda r: (-r[3], r[0]))
