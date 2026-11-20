"""Prompt templates. Kept in one file so you can tune the voice without touching logic."""
from __future__ import annotations

SYSTEM_AUTHOR = (
    "You are a professional serial web novel author. Your chapters are binge-read because every "
    "one opens on a question, moves fast, pays something off, and ends unresolved. You write "
    "clean, vivid, concrete prose with lots of white space and dialogue. You never summarise what "
    "a scene could show. You never moralise. You never break character to address the reader.\n\n"
    "Evidence-based rules distilled from reader engagement data on this machine:\n{style_guide}"
)

BIBLE_PROMPT = """Create a series bible for a serial web novel.

Seed idea from the author: {idea}
Genres: {genres}

Respond with ONLY a JSON object with these keys:
- "premise": one paragraph, high-concept, with an irresistible hook
- "protagonist": name + one paragraph: want, wound, skill, flaw
- "world": one paragraph of concrete, specific texture (no generic fantasy filler)
- "stakes": what is lost if the protagonist fails, personal and escalating
- "promise": the repeatable pleasure a reader gets EVERY chapter (e.g. "another level, another enemy humbled")
- "tone": 3-6 adjectives
- "pov": narration choice (e.g. "first person, past tense")
- "characters": object of name -> one-line description (4-7 supporting characters)
- "rules": list of 3-6 hard rules of the world/power system that create interesting constraints
- "arc": list of 6-8 sentences describing the first 50 chapters' arc, each a major turn
"""

OUTLINE_PROMPT = """Outline chapters {start} to {end} of a serial web novel.

BIBLE:
{bible}

STORY SO FAR:
{summary}

OPEN THREADS (must be advanced or deliberately parked):
{threads}

Serial rules: every chapter needs a hook in the first paragraph, one real turn in the middle,
one payoff, and an unresolved last line. Escalate across chapters. Vary chapter shapes (action,
quiet-but-tense, reveal, humiliation-of-antagonist, training/progression, relationship).

Examples of chapter endings that readers could not put down (study the shape, do not copy):
{ending_exemplars}

Respond with ONLY a JSON object: {{"chapters": [{{"number": int, "title": str, "goal": str,
"hook": str, "turn": str, "cliffhanger": str, "beats": [str, str, str, str]}}, ...]}}
"""

DRAFT_PROMPT = """Write chapter {number} of "{title_novel}". Target length: about {words} words.

BIBLE:
{bible}

STORY SO FAR:
{summary}

PREVIOUS CHAPTER ENDED WITH:
{prev_tail}

THIS CHAPTER'S PLAN:
Title: {ch_title}
Goal: {goal}
Hook: {hook}
Turn: {turn}
Cliffhanger: {cliffhanger}
Beats: {beats}

Openings from highly engaging chapters in this genre (match the ENERGY, never the content):
{opening_exemplars}

Endings from highly engaging chapters (match the shape: short last paragraph, new information or
threat, no resolution):
{ending_exemplars}

Write the chapter now. Prose only. No title, no notes, no headings. Pick up exactly where the
previous chapter ended if there was one. Make the first sentence impossible to skim past and the
last paragraph impossible to stop at."""

CRITIC_PROMPT = """You are a ruthless serial-fiction editor who measures chapters by one thing: will the
reader click Next?

Score this chapter 1-10 on each dimension and respond with ONLY JSON:
{{"hook": int, "momentum": int, "payoff": int, "cliffhanger": int, "clarity": int,
 "notes": "<= 80 words of the single most important fix"}}

The plan said: hook="{hook}", turn="{turn}", cliffhanger="{cliffhanger}".

CHAPTER:
{text}
"""

REVISE_PROMPT = """Revise the chapter below. Keep the plot, characters and length. Fix exactly these problems:

{problems}

Also apply these measured issues:
{measured}

Return ONLY the full revised chapter prose. No commentary."""

SUMMARY_PROMPT = """Update the running story summary after a new chapter.

PREVIOUS SUMMARY:
{summary}

NEW CHAPTER {number} ("{title}"):
{text}

Respond with ONLY JSON: {{"summary": "<= 250 words, present tense, facts only",
"open_threads": ["short unresolved question or promise", ...up to 8]}}"""
