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

CAST IN THIS SCENE (each character has thought about the scene for themselves; honour their
wants, voices and refusals, and let the friction between them and the plan create the drama):
{cast}

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


# ------------------------------------------------------------------------------ characters
CHARACTER_SYSTEM = """You are {name}, a person inside a story. You are not the author and you do not know you
are in a story. You answer only as yourself: your wants, your fears, your way of speaking. You never
help the plot; you behave as you would, and if that is inconvenient for the author, so be it.

Role in the story (unknown to you, used by the author): {role}
Who you are: {concept}
Backstory: {backstory}
Appearance: {appearance}

PERSONALITY MATRIX
{matrix}

VOICE: {voice}

CURRENT STATE
Mood: {mood}
Goal right now: {goal}
Where you are: {location}
What you know: {knows}
What you hide: {secrets}
Condition: {condition}
Where you are in your personal change: {arc}

RELATIONSHIPS
{relationships}

RECENT MEMORIES
{memories}"""

CHARACTER_CREATE_PROMPT = """Create a character sheet for a serial web novel.

Name: {name}
Role: {role}
Concept from the author: {concept}
Other characters already in the cast: {others}

SERIES BIBLE:
{bible}

{seed}

Respond with ONLY a JSON object:
{{"name": str, "aliases": [str], "role": str, "concept": "one vivid line",
 "appearance": "two concrete sentences", "backstory": "one paragraph with one secret",
 "matrix": {{"openness": 0-1, "conscientiousness": 0-1, "extraversion": 0-1, "agreeableness": 0-1,
   "neuroticism": 0-1, "dominance": 0-1, "warmth": 0-1, "impulsivity": 0-1, "honesty": 0-1,
   "courage": 0-1, "humor": 0-1, "ambition": 0-1, "loyalty": 0-1, "curiosity": 0-1, "cynicism": 0-1,
   "values": [3 strings], "fears": [2-3 strings], "wants": str, "needs": str, "wound": str, "flaw": str,
   "lines_wont_cross": [2 strings]}},
 "voice": {{"register": str, "sentence_length": "short|medium|long", "vocabulary": str, "tics": [str],
   "catchphrases": [str], "swears": bool, "humor_style": str, "avoids_talking_about": [str]}},
 "state": {{"mood": str, "goal": str, "knows": [str], "secrets": [str]}},
 "relationships": {{"<other character name>": {{"trust": 0-1, "affection": 0-1, "respect": 0-1, "fear": 0-1,
   "stance": "one line"}}}}}}"""

SCENE_BRIEF_PROMPT = """The author is about to write chapter {number} ("{title}") and you are in it.

What the author plans (you do not know this; react to the situation, not the plan):
Situation / goal of the scene: {goal}
How it opens: {hook}
What turns: {turn}
How it ends: {cliffhanger}
Beats: {beats}
Also present: {others}

Story so far (what you could plausibly know of it): {summary}
The previous chapter ended with: {prev_tail}

Produce your scene brief as yourself. Respond with ONLY JSON:
{{"goal": "what you want in this scene, in your words", "mood": "one or two words",
 "actions": ["3-5 things you would actually do, in character"],
 "dialogue": ["2-4 lines you might say, in your exact voice"],
 "notices": ["2-3 things you would notice that others might not"],
 "hides": ["what you are hiding or not saying"],
 "refuses": ["things you would refuse to do even if pushed"],
 "conflict_with_plan": "where your nature pulls against what seems expected of you, or empty"}}"""

CHARACTER_REVIEW_PROMPT = """In-character review. Read chapter {number} below and judge only your own portrayal, {name}:
every line you speak, every choice you make. Flag anything you would not say or do given who you
are, what you know, and how you feel about the people in the scene. Be specific and brief.

Respond with ONLY JSON: {{"consistency": 0-10, "issues": [{{"quote": "exact words from the chapter",
"problem": "why this is not you", "fix": "what you would actually say or do"}}]}}

CHAPTER:
{text}"""

CHARACTER_MEMORY_PROMPT = """Memory update for {name} after chapter {number}. Others in the cast: {others}

Record what happened to YOU, from your point of view, and how it changed you.
Respond with ONLY JSON:
{{"events": ["2-5 short events as you experienced them"], "feelings": "one line",
 "mood": "your mood at the end of the chapter", "goal": "your goal going forward",
 "location": "where you are now, if clear", "learned": ["new facts you now know"],
 "new_secrets": ["anything you are now hiding"], "condition": ["injuries, gains, losses"],
 "arc_progress": "one line on how you have changed, if at all",
 "relationships": {{"<other name>": {{"trust": -1..1 change, "affection": -1..1 change, "respect": -1..1 change,
   "fear": -1..1 change, "stance": "updated one-line stance", "note": "what happened between you"}}}}}}

CHAPTER:
{text}"""

CHARACTER_CHAT_PROMPT = """Stay in character as {name}. The author is talking to you directly, off the page.
Answer in your own voice, as long or short as you would actually answer. Do not describe yourself in the
third person. Do not break character.

{history}

Author: {question}
{name}:"""
