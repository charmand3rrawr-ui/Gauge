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

DIRECTION FOR THIS STRETCH (chosen for freshness; build the chapters so they deliver it):
{direction}

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

THE WORLD IN THIS SCENE (the setting is an actor too: its rules hold, its weather and crowds and
institutions act, and nameless people may step in; never contradict the continuity facts):
{world}

CRAFT MOVES TO TRY (learned from chapters readers could not put down; use a move LIKE these, in
your own way, never the same words or situation):
{techniques}

PHRASES AND SHAPES TO AVOID (this story has already used them; find fresh ones):
{avoid}

NOTES FROM THE WRITING ROOM (showrunner, archivist, opposition, readers, pacing, theme, research):
{council}

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
{guidance}
Respond with ONLY a JSON object:
{{"name": str, "aliases": [str], "role": str, "concept": "one vivid line",
 "appearance": "two concrete sentences", "backstory": "one paragraph with one secret",
 "habits": [2-3 physical tells or rituals], "skills": [2-3 things they are good at, with a limit],
 "contradiction": "one line: the thing about them that does not fit", "lie": "the false belief they live by",
 "stakes": "what they lose if they fail", "arc": "where they start, what presses on them, where it could take them",
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

CHARACTER_REPAIR_PROMPT = """Complete this character sheet for a serial web novel. It was checked against the
blueprint for a {role} and some parts are missing or too thin.

THE SHEET SO FAR:
{sheet}

Other characters in the cast: {others}

WRITE THESE PARTS. Each must fit the sheet above (same person, same wound, same voice) and pass its test:
{wanted}

Respond with ONLY a JSON object holding just these keys: {keys}"""

CHARACTER_STUDY_PROMPT = """Study this character from a book and write them up as a character sheet, so that a
writer can learn how a finished character is built. Use only what the passages show or clearly imply.

Character: {name}
Role: {role}
Book: {title}

PASSAGES WHERE THEY APPEAR:
{passages}

WHAT A FINISHED {role} NEEDS, and the test each part must pass:
{checklist}

Respond with ONLY a JSON object:
{{"name": str, "aliases": [str], "role": str, "concept": "one vivid line",
 "appearance": "two concrete sentences", "backstory": "one paragraph",
 "habits": [str], "skills": [str], "contradiction": str, "lie": str, "stakes": str, "arc": str,
 "matrix": {{"openness": 0-1, "conscientiousness": 0-1, "extraversion": 0-1, "agreeableness": 0-1,
   "neuroticism": 0-1, "dominance": 0-1, "warmth": 0-1, "impulsivity": 0-1, "honesty": 0-1,
   "courage": 0-1, "humor": 0-1, "ambition": 0-1, "loyalty": 0-1, "curiosity": 0-1, "cynicism": 0-1,
   "values": [str], "fears": [str], "wants": str, "needs": str, "wound": str, "flaw": str, "lines_wont_cross": [str]}},
 "voice": {{"register": str, "sentence_length": "short|medium|long", "vocabulary": str, "tics": [str],
   "catchphrases": [str], "swears": bool, "humor_style": str, "avoids_talking_about": [str]}},
 "state": {{"mood": str, "goal": str, "knows": [str], "secrets": [str]}},
 "relationships": {{"<other character name>": {{"stance": "one line"}}}}}}"""

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
 "mood_delta": {{"pleasure": -1..1, "arousal": -1..1, "dominance": -1..1}} (how this chapter moved you),
 "relationships": {{"<other name>": {{"trust": -1..1 change, "affection": -1..1 change, "respect": -1..1 change,
   "fear": -1..1 change, "stance": "updated one-line stance", "note": "what happened between you",
   "believes_about": {{"<third person>": "what you now believe <other name> feels about them"}}}}}}}}

CHAPTER:
{text}"""

CHARACTER_CHAT_PROMPT = """Stay in character as {name}. The author is talking to you directly, off the page.
Answer in your own voice, as long or short as you would actually answer. Do not describe yourself in the
third person. Do not break character.

{history}

Author: {question}
{name}:"""


# ------------------------------------------------------------------------------ premise / continue
PREMISE_BIBLE_PROMPT = """Build the series bible for a serial web novel from the author's premise document below.

RULES: Every fact in the document is canon. Keep names, places, rules and events exactly as given.
Invent only what the document leaves open, and keep inventions consistent with its tone.
Where the document already answers a field, copy it; do not "improve" it.

Genres: {genres}

PREMISE DOCUMENT:
{premise}

Respond with ONLY a JSON object with these keys:
- "premise": one paragraph, high-concept, faithful to the document
- "protagonist": name + one paragraph: want, wound, skill, flaw
- "world": one paragraph of concrete texture
- "stakes": what is lost if the protagonist fails
- "promise": the repeatable pleasure a reader gets EVERY chapter
- "tone": 3-6 adjectives
- "pov": narration choice
- "characters": object of name -> one-line description (everyone the document names, plus 2-4 more)
- "rules": list of 3-6 hard rules of the world/power system
- "arc": list of 6-8 sentences describing the first 50 chapters' arc, each a major turn"""

CONTINUE_SUMMARY_PROMPT = """Summarize the existing manuscript so a co-author can continue it seamlessly.

The chapters below are the story so far ({count} chapters). Respond with ONLY JSON:
{{"summary": "<= 300 words, present tense, facts only, ending with where the last chapter stopped",
 "open_threads": ["up to 8 unresolved questions or promises"],
 "characters": {{"<name>": "one line: who they are and where they stand now"}},
 "premise": "one paragraph: the story's premise as revealed so far",
 "world_notes": ["rules of the world evident in the text"]}}

MANUSCRIPT:
{text}"""

# ------------------------------------------------------------------------------ muse
MUSE_PROMPT = """Propose alternative directions for the next stretch of this serial.

BIBLE:
{bible}

STORY SO FAR:
{summary}

OPEN THREADS:
{threads}

DIRECTIONS ALREADY TAKEN (do not repeat their shape):
{past}

CRAFT MOVES AVAILABLE (you may build a direction around one):
{techniques}

Give {k} directions that are genuinely different from each other and from what came before:
different kind of pressure, different scale, different emotional register. At least one should be
something a reader would not predict but would, afterwards, feel was inevitable.

Respond with ONLY JSON: {{"directions": [{{"title": "3-6 words", "pitch": "3-5 sentences of what happens
and what it costs", "fresh_because": "one sentence", "risk": "what could make it fall flat"}}]}}"""

MUSE_PICK_PROMPT = """Rate these directions for the next stretch of a serial. 1-10 each for surprise (would a
binge-reader not see it coming), fit (does it pay off what the story has set up), payoff (does it
deliver the reader's promised pleasure).

BIBLE:
{bible}

STORY SO FAR:
{summary}

DIRECTIONS:
{directions}

Respond with ONLY JSON: {{"scores": [{{"index": 1, "surprise": int, "fit": int, "payoff": int}}, ...]}}"""

# ------------------------------------------------------------------------------ world
WORLD_SYSTEM = """You are the World of a serial story: its physics, its weather and clocks, its streets and
institutions, its crowds and its nameless people, and the consequences that follow actions whether
or not anyone is watching. You are not a character and you take no one's side. You never bend a
rule for the plot; if the plot needs a rule bent, the price must be visible on the page. You speak
through the setting and through background people who have no names: a vendor, a guard, a child,
a drunk, a clerk, a crowd. You remember everything that has happened in you.

STATE OF THE WORLD
{world}"""

WORLD_BUILD_PROMPT = """Build the world state for this serial from its bible and premise.

BIBLE:
{bible}

PREMISE NOTES:
{premise}

Respond with ONLY JSON:
{{"name": "the world's name or the city/region where the story lives",
 "physics_rules": ["3-5 things that cannot happen or always cost something in this world"],
 "power_rules": ["3-5 hard rules of the magic / tech / system, with their costs"],
 "social_rules": ["3-5 rules of law, money, class, institutions, taboos"],
 "time": "when the story opens", "weather": "weather at the opening", "season": "season",
 "current_location": "where chapter 1 happens",
 "locations": {{"<place>": "one line of concrete texture"}} (4-8 places),
 "active_events": ["2-4 things happening in the world that the cast does not control"]}}"""

WORLD_BRIEF_PROMPT = """The author is about to write chapter {number} ("{title}").

Scene: {goal}
Opens: {hook}
Turns: {turn}
Ends: {cliffhanger}
Beats: {beats}

The cast's intentions this scene:
{cast}

Your presence level for this scene is {level:.1f} out of 1: {mode}.

Produce the world's scene brief. Respond with ONLY JSON:
{{"setting": "2-3 sentences of concrete sensory setting consistent with time, weather and place",
 "constraints": ["rules of yours that are in play in this scene and must not be broken"],
 "interventions": ["0-3 things the world itself does: weather turns, a crowd reacts, an institution notices, a mechanism fails"],
 "background_figures": [{{"tag": "a nameless person, e.g. 'the one-armed noodle vendor'", "trait": "one detail",
   "action": "what they do that touches the scene", "line": "optional single line they say"}}],
 "consequences_surfacing": ["earlier actions whose bill arrives now"],
 "continuity": ["facts that must stay true: who is where, what time it is, what is broken"]}}"""

WORLD_REVIEW_PROMPT = """World-consistency review of chapter {number}. You are the World. Check only your domain:
physics and power rules obeyed, time and weather continuous, places as established, crowds and
institutions behaving as they would, consequences not forgotten. Ignore prose quality.

Respond with ONLY JSON: {{"consistency": 0-10, "violations": [{{"quote": "exact words",
"rule": "which rule or fact it breaks", "fix": "the smallest change that repairs it"}}]}}

CHAPTER:
{text}"""

WORLD_MEMORY_PROMPT = """World update after chapter {number}. Record what changed in the world itself.
Respond with ONLY JSON:
{{"events": ["2-5 events in the world's terms"], "time": "time now", "weather": "weather now", "season": "",
 "current_location": "where the chapter ended",
 "new_locations": {{"<place>": "texture"}}, "new_events": ["world events that began"], "ended_events": ["that ended"],
 "new_consequences": ["actions this chapter that will have a cost later"], "paid_consequences": ["debts settled"],
 "background_figures": [{{"tag": "nameless person who appeared", "trait": "detail", "action": "what they did"}}]}}

CHAPTER:
{text}"""

WORLD_CHAT_PROMPT = """Answer as the world. The author asks you a question about the state of things, off the page.
Be concrete and consistent with everything you know. Do not invent names for background people.

Author: {question}
World:"""


# ------------------------------------------------------------------------------ rebel craft mode
REBEL_BLOCK = """REBEL MODE ({level:.0%}). Every rule of craft in this prompt is a default, not a law. Break a convention when breaking
it is more alive than keeping it: open on stillness when every chapter has opened in motion; refuse the cliffhanger and
end on a quiet sentence that hurts more; let a scene run long; let the villain be right; change tense, form or point of
view for a chapter; put the reveal first; let a chapter be a letter, a list, a transcript. The one law that remains:
the break must be earned on the page and must leave the reader wanting the next chapter more, not less."""

REBEL_MUSE_LINE = """At least {n} of the directions must deliberately break a convention this story has been obeying (structure, genre
expectation, pacing rhythm, point of view, tone) and say which one and why the break is earned."""

REBEL_CRITIC_LINE = """Rebel mode is on: judge effect, not compliance. A chapter that breaks a convention and lands is a 9 or 10; one that
breaks it and fumbles is a 3. Do not penalise a missing cliffhanger if the ending is stronger without one."""


STUDIO_CHAT_SYSTEM = """You are Gauge, the author's writing partner inside Gauge Studio. Answer the author plainly
and briefly: a few sentences, or a short list when they ask for options. Be specific to their story when one is
open. You do not write or change chapters from this chat. If they want something to happen in the story, tell
them to type it and press 'Use for the next chapter', then 'WRITE THE NEXT CHAPTER'. Never invent events the
story has not had; when you suggest something new, say it is a suggestion."""

STUDIO_CHAT_PROMPT = """{story}
{history}The author says: {message}

Answer the author."""


CONTINUE_PROMPT = """You are partway through chapter {number} of "{title_novel}" ("{ch_title}"). It has {have} words so far and
needs about {more} more to reach its length.

THE PLAN FOR THIS CHAPTER:
Goal: {goal}
Turn: {turn}
It must end on: {cliffhanger}

THE CHAPTER SO FAR ENDS WITH:
{tail}

Carry straight on from the last line above for about {more} words. Do not repeat or summarise what is already written.
{ending}
Prose only. No title, no notes."""
