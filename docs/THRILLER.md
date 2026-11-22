# Writing crime, thrillers and mysteries with Gauge

Dark fiction lives or dies on four things: **tension** that never lets up, a mystery that plays **fair**,
an antagonist who is **consistent and genuinely menacing**, and prose whose procedural detail reads
**expert** rather than vague. The Thriller Craft agent measures all four, every chapter, with no model
calls, and attaches itself automatically whenever a project's genre or tone is dark (crime, thriller,
mystery, noir, horror, psychological, serial-killer, police procedural).

## What it checks

**Suspense.** Each chapter is scored for dread cues (something watched, something wrong, a pressing
silence), a ticking clock, stakes on the page, and whether it ends on a hook. A long lull — several
paragraphs with no tension — is flagged so you can thread menace through the quiet.

**Expert register (verisimilitude).** When a scene claims technical work, the prose should carry the
vocabulary a knowledgeable writer would use. The agent rewards the specific register of the domains you
ask for and flags hand-waving like *"did some forensics and figured it out."* Domains: `forensic`,
`criminology`, `psychology`, `deception`, `engineering`.

> This check is about how the fiction *reads*, not about real methods. Gauge improves storytelling; it
> does not generate operational instructions for real crime, deception or weapons — and it does not need
> to. Dread, fair clues and a consistent character are what make a thriller land.

**Antagonist signature.** A serial antagonist should have a consistent MO and signature. The agent
remembers the signature terms it sees and surfaces them in the brief so the killer stays recognisably
themselves across the book.

**Fair play.** Declare the mystery's scaffolding and the agent holds you to it: every key clue must be
shown to the reader *before* the reveal, and every red herring must be dispelled. Near the reveal it nudges
the planner and drafter to plant any clue still missing.

## Declaring the mystery

In `config.json` under `agents`:

```json
{
  "agents": {
    "thriller_domains": ["forensic", "criminology", "psychology"],
    "thriller_clues": ["the muddy boot print", "the torn ticket stub", "the stopped clock"],
    "thriller_red_herrings": ["the jealous brother", "the missing cash"],
    "thriller_reveal_chapter": 18,
    "thriller_signature": ["the staged tableau", "the removed left shoe", "the folded note"]
  }
}
```

- **thriller_clues** — the clues the solution genuinely rests on. Checked (case-insensitive substring)
  against each chapter; any not shown before `thriller_reveal_chapter` is flagged as an unfair reveal.
- **thriller_red_herrings** — misdirections that must be addressed and cleared by the reveal.
- **thriller_reveal_chapter** — the chapter the truth comes out (`0` = unset; fair play is then only
  tracked, not enforced at a point).
- **thriller_domains** — which expert registers the story should carry.
- **thriller_signature** — the antagonist's MO lexicon, kept consistent.

## Inspecting it

```
gauge thriller PROJECT     # suspense trend, clues shown vs. still unplanted, antagonist signature
gauge agents show PROJECT thriller_craft
```

## How it fits the room

Thriller Craft complements the agents you already have: the **Showrunner** tracks promises and payoffs,
the **Schemer** runs the opposition as a planner with hidden information, the **Pacing Director** shapes
the rhythm, and the **Sensitivity Reader** and **Suitability Checker** keep dark content within your
platform's rules. Thriller Craft adds the genre-specific layer on top.
