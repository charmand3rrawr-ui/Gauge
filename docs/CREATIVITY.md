# Creativity: learning craft, not just numbers

The engagement model learns *what correlates* with readers staying. That keeps the writing
effective, but it cannot by itself keep the writing alive over hundreds of chapters. Three
mechanisms do that, and all three keep changing as Gauge absorbs new content.

## 1. Technique absorption (`gauge absorb`)

The absorber reads the strongest unread chapters in the corpus, newest and highest-engagement
first, and asks the fast model to name the **craft moves** in each: the misdirect, the reversal
at the two-thirds mark, the one-line paragraph that lands a reveal, the dialogue beat that hides
exposition. Each move becomes a technique card (kind, name, description, a short example, where
it came from, how popular the source was), embedded for retrieval and stored under
`data/models/techniques.*`.

The drafter is handed two or three cards that fit the planned chapter, one per kind, with a
single instruction: *use a move like this, in your own way, never the same words or situation*.
The muse (below) sees the whole library as raw material for directions.

The daemon absorbs `learn.absorb_per_cycle` chapters every cycle (default 8), so every ingest
of new chapters teaches new moves. Nothing is copied: cards are descriptions plus a 40-word
example, and the drafter is told to transform, not reuse.

## 2. The muse: divergent planning

Before the outline is extended, the muse proposes several genuinely different directions for
the next stretch (different pressure, scale and emotional register, at least one unpredictable
but inevitable-in-hindsight), rates them for surprise, fit and payoff, and penalises any that
resemble directions already taken (n-gram overlap against `past_directions`). The winner is
written into the outline prompt as the stretch's direction. Divergent first, convergent second,
instead of letting the model's first idea win every time.

`write.muse_directions` sets how many alternatives (default 3); `write.muse_enabled` turns it off.
The chosen direction is recorded in each chapter's front matter.

## 3. Novelty tracking

Every draft is compared with the project's own recent chapters:

- 4-gram overlap with the last six chapters,
- whether the opening line has the same shape as a recent chapter,
- whether the closing line echoes a recent ending,
- phrases (3-grams) that already appeared in two or more earlier chapters and appear again.

The result is a novelty score; `write.novelty_weight` (default 0.15) of the chapter score comes
from it, so repetition is punished and revisions are asked to retire worn phrases. The drafter is
also given, up front, the worn phrases and the last three chapter openings to avoid.

## Writing from a premise or a manuscript

- **A premise document**: `gauge new NAME --premise-file premise.md`, or `gauge start premise.md`.
  Everything in the document is canon; the model fills gaps and does not reinvent. The document
  is kept as `premise.md` in the project and feeds the bible, the characters and the world.
- **An existing manuscript**: `gauge new NAME --continue-from "D:\Writing\MyBook"` (folder, .txt,
  .md, .docx or .epub). The chapters are imported as already written, summarised, and Gauge
  continues from the next chapter with the characters, threads and world rules it found.
- **One command**: `gauge start "a courier who can rewind ten seconds" --chapters 3`.

## Temperature and models

Ideation runs hot (the muse at 1.0, bible and sheets at 0.9), drafting at `llm.temperature`
(0.9), judgement cold (critic, reviews, memory at 0.2). Divergence where it helps, discipline
where it counts.
