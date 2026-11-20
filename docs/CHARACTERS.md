# Characters as independent agents

A serial lives or dies on its people. Gauge gives every character their own **personality
matrix**, **voice**, **relationships**, **memory** and an **agent** that speaks only as that
person. When a character appears in a planned scene, their agent is activated: they think about
the scene for themselves, the drafter writes with their intentions in hand, they review the
result for anything they would never say or do, and then they remember what happened.

Everything is stored as JSON under `data/projects/<slug>/characters/` and rendered to
`characters.md`, so you can read and edit a character by hand between chapters.

## The personality matrix

Fifteen trait axes from 0 to 1:

| | | |
| --- | --- | --- |
| openness | conscientiousness | extraversion |
| agreeableness | neuroticism | dominance |
| warmth | impulsivity | honesty |
| courage | humor | ambition |
| loyalty | curiosity | cynicism |

plus the anchors that turn a profile into a person: `values`, `fears`, `wants` (conscious goal),
`needs` (what would actually fix them), `wound`, `flaw`, and `lines_wont_cross`.

The axes are not decoration. `PersonalityMatrix.tendencies()` turns them into deterministic
behaviour rules that go into every prompt the character sees, for example:

- impulsivity high + conscientiousness low: *acts before planning; regrets later, rarely apologises first*
- honesty low: *lies smoothly and small; tells the truth only when it costs nothing*
- courage low: *freezes or flees under sudden threat, then is ashamed*
- humor high: *deflects with jokes, especially at the worst moment*

Hard lines are enforced in code: whatever the model says in a scene brief, every entry in
`lines_wont_cross` is appended to the character's refusals.

## Voice

Register, sentence length, vocabulary, verbal tics, catchphrases, swearing, humour style, and
topics they avoid. Rendered into both the character's own prompt and the drafter's cast card, so
dialogue stays recognisable across hundreds of chapters.

## State, memory and relationships

Each character tracks mood, current goal, location, what they know, what they hide, their
condition (injuries, gains, losses), how far along their personal change they are, and a list
of memories by chapter *from their own point of view*. Relationships to every other character
carry trust, affection, respect and fear (0 to 1), a one-line stance, and a short history. After
every chapter in which a character appears, their agent updates all of this from the chapter
text. Relationship changes are deltas, so trust is earned and lost gradually.

## The agent in the writing loop

```
plan chapter n
   │
   ├─ roster.present_in(plan, previous tail)   -> who is in this scene (protagonist always)
   ├─ for each: CharacterAgent.brief(scene)    -> goal, mood, would do / say / notice / hide / refuse,
   │                                              and where their nature pulls against the plan
   ├─ Drafter.draft(..., cast=briefs)          -> prose written with the cast's intentions in hand
   ├─ Critic + CharacterAgent.review(text)     -> engagement score blended with in-character consistency
   │     └─ revise with the characters' own corrections if below the bar
   └─ for each on the page: CharacterAgent.remember(text) -> mood, knowledge, relationships, memories
```

The agent's system prompt is the character and nothing else: not the author, not the plot.
That is what makes them independent. A cowardly character will tell you in the brief that they
would run, and if the plan needs them to stand their ground, the drafter sees that friction and
has to earn it on the page. The `conflict_with_plan` line in each brief is often where the best
scene comes from.

Scores: the chapter score becomes `(1 - w) * engagement + w * consistency` with
`write.character_weight` (default 0.2). Set `write.character_review` to false to skip reviews,
or `write.characters_enabled` to false to switch the whole system off.

## Commands

```powershell
.\gauge.bat new "Ten Seconds" --idea "..."                 # bible characters get sheets automatically
.\gauge.bat character list "Ten Seconds"
.\gauge.bat character show "Ten Seconds" Rin
.\gauge.bat character new "Ten Seconds" "Marina Vale" --concept "a fence who sells rewound goods" --role antagonist
.\gauge.bat character new "Ten Seconds" "Dice" --concept "..." --seed 42   # fixed random matrix, LLM fills the rest
.\gauge.bat character roll "Ten Seconds" "Extra" --seed 7                 # random matrix, no LLM
.\gauge.bat character set "Ten Seconds" Rin --trait courage=0.9 --alias "the courier" --mood wired
.\gauge.bat character sync "Ten Seconds"                                  # sheets for new names in the bible
.\gauge.bat character brief "Ten Seconds" Rin --chapter 7                 # preview their take on a planned chapter
.\gauge.bat character chat "Ten Seconds" Rin                              # interview them; they remember everything
.\gauge.bat character remove "Ten Seconds" Extra
```

Characters are detected in a scene by name and aliases (word-boundary match) in the chapter
plan and the previous chapter's ending. Give nicknames and titles as aliases so "the Broker" and
"Mr. Hale" are recognised as the same person.

## Tips

- Interview a character before chapter 1. Ten minutes of `character chat` tells you whether the
  matrix produced someone you want to spend a hundred chapters with.
- When a chapter feels flat, read the briefs (`character brief`) for the next one. If no
  character has a `conflict_with_plan`, the plan is probably too agreeable.
- Edit `characters/<name>.json` freely. The agent reads it fresh every chapter.
