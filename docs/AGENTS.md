# The writing room: independent story agents

Beyond the characters and the World, Gauge runs a room of specialist intelligences. Each has
its own point of view, its own prompt, and its own persistent state in
`projects/<slug>/agents/<name>.json`. None of them sees another's prompt; the **council**
runs them in parallel at four moments of the chapter loop and merges their verdicts.

| moment | what the agents do |
| --- | --- |
| plan notes | advice to the planner before the outline is extended (showrunner debts, pacing shape) |
| brief | advice to the drafter, written into the draft prompt as *Notes from the writing room* |
| review | a 0..1 score plus concrete issues; weighted into the chapter score and fed to revisions |
| remember | update their own state once the chapter is accepted |

`agents.weight` (default 0.25) is the council's share of the chapter score; inside it, each
agent has a relative weight (Reader Panel 2.0, Showrunner and Archivist 1.5, down to 0.5 for
Theme, Pacing and Comedian). Advisory agents (Packager, Librarian, Sensitivity) carry no weight.

An agent only votes with a verdict it actually got. If its model call fails, or the answer lacks
what was asked for, `review` returns nothing and the council leaves the agent out of the score;
agents with a measured part (Line Editor, Theme, the Archivist's name check) fall back to that
part alone. Nothing is recorded for a chapter nobody read: no "clean" from the Sensitivity
Reader, no "zero laughs" from the Comedian. The agents that went unanswered are listed in the
chapter's front matter under `silent_judges`.

## The agents

**Reader Panel** (`reader_panel`). Five personas read every draft: the binge reader, the
skimmer, the lore-hound, the commenter, and the hostile reader hunting plot holes. Each reports
the probability they click Next, where they would have closed the tab, and what they would
post. The mean click-through is the score. Record real numbers with
`gauge feedback PROJECT CHAPTER --views N --comments N`; the panel compares its predictions with
observed retention (views of chapter n+1 over chapter n), learns a bias, and reports its
correlation in `gauge readers`. This is the agent that closes the loop on "addictive".

**Showrunner** (`showrunner`). A ledger of promises: every question, threat, goal or setup the
story opens, when it was planted, when the reader expects it paid. It tells the planner which
debts must be paid in the next stretch, tells the drafter what to advance, and penalises chapters
that stall or ignore overdue promises. `gauge ledger PROJECT`, and `gauge ledger PROJECT --heatmap` for a
serial-wide view of what was planted, advanced and paid each chapter, the open and overdue counts, and a
gantt of the threads still open.

**Archivist** (`archivist`). A canon ledger of checkable facts (names, ages, eye colour, who
knows what, where things are) extracted from every chapter. Contradictions are found
deterministically: same subject and attribute, different value. It briefs the drafter with the
canon relevant to the scene and flags contradictions in drafts.
`gauge canon PROJECT "what colour are Rin's eyes?"` and `gauge canon PROJECT --subject Rin`.
It also keeps a measured **name index** (no model call): every proper name the serial has used and how
often. A new name one slip away from an established one ("Katharine" after fourteen "Katherine"s) is
flagged as spelling drift. It is conservative on purpose: names on the cast list are never reported,
short names only match on a doubled or swapped letter (Mara and Maya are different people), and two
spellings each used several times in one chapter are treated as two people. `gauge canon PROJECT --names`.

**Schemer** (`schemer`). The opposition as a planner with hidden information: goal, resources,
what it knows about the protagonist, a multi-step scheme, and how much of it the protagonist has
uncovered. Every chapter it decides what the opposition does visibly and off-page, reviews the
draft for a stupid or omniscient antagonist, and adapts the scheme to what happened.

**Line Editor** (`line_editor`). Mostly measured: filter words per thousand, ornate dialogue
tags, flat sentence rhythm, intensifiers, and drift from the voice fingerprint taken from the
first three chapters. Plus three line-level fixes from the model.

**Packager** (`packager`). Replaces generic chapter titles ("Chapter 7") with a scored candidate,
and writes the blurb, tagline and tags: `gauge package PROJECT`.

**Theme Keeper** (`theme_keeper`). Central question, motifs and images from the bible; rotates
the least-used motif into each chapter's brief and checks that something thematic landed.

**Sensitivity Reader** (`sensitivity`). Flags content by category and severity against your
platform's rules (`agents.sensitivity_rules`). Flags are written into each chapter's front matter
as `content_flags`. Warns by default; `agents.block_on_sensitivity` makes severe flags force a
revision.

**Thriller Craft** (`thriller_craft`). No model calls. Attaches only for dark genres (crime, thriller,
mystery, noir, horror, psychological). It raises the craft of this fiction: it scores suspense (dread,
ticking clock, stakes, an end-of-chapter hook, and lulls), checks that the prose carries an authentic
forensic/criminological/psychological/engineering *register* instead of hand-waving ("did some
forensics"), keeps a serial antagonist's signature consistent, and — when you declare the mystery's
scaffolding — enforces **fair play**: every key clue the solution rests on must be shown to the reader
before the reveal, and every red herring must be dispelled. Declare it in config under `agents.thriller_*`
(`clues`, `red_herrings`, `reveal_chapter`, `domains`, `signature`). `gauge thriller PROJECT`. See
THRILLER.md. It improves storytelling only — it never supplies real-world criminal, deception or weapon
methods.

**World Auditor** (`world_auditor`). No model calls. Guards the slow physical state of the world across
the whole serial: an absolute timeline built from time markers, season-vs-weather consistency, who is hurt
(and whether they are back in hard action too soon), props that were destroyed yet reappear, and
instant travel between declared places. Author-declared world rules are checked each chapter —
`agents.world_rules`, one per line, `never: <phrase>` or `always: <phrase> when <trigger>`; name places in
`agents.known_places` to sharpen the travel check. `gauge world audit PROJECT`. In the lean profile.

**Thread Keeper** (`thread_keeper`). No model calls. Tracks the weave the Showrunner's big promises miss:
author-declared **subplots** (`agents.subplots`, `label: keyword, keyword`) and **foreshadow**
(`agents.foreshadow`, `plant -> payoff`). It flags a subplot gone quiet too long (`agents.subplot_gap`, a
dropped thread), a payoff that lands with no earlier setup (unearned), and reminds the planner which
plants are still waiting to pay off. `gauge threads PROJECT`. In the lean profile.

**Pacing Director** (`pacing`). No model calls. Classifies each chapter's shape from its
features (action, quiet, reveal, progression, dialogue), recommends the next shape so the serial
breathes, complains about a third identical shape in a row, and plans a deliberate shape rhythm for the
next stretch (`arc_schedule`, shown in `gauge agents show pacing`).

**Research Librarian** (`librarian`). Indexes `premise.md` and everything in the project's
`research/` folder (`.md`, `.txt`) with embeddings; briefs the drafter with the notes relevant
to the scene; `gauge research PROJECT "who built the sewers?"`.

**Voice Coach** (`voice_coach`). No model calls. Attributes dialogue to speakers, measures each
character's style, and penalises chapters where two characters sound alike. See CHARACTERS.md.

**Scout** (`scout`). Open-source research for the scene and fact-checks of real-world claims. See
RESEARCH.md.

**Comedian** (`comedian`). Attaches itself only when the genres or tone are comic. Plans comic
beats from character and situation, counts laughs, keeps running gags for callbacks.

## Controlling the room

```powershell
.\gauge.bat agents list [PROJECT]            # who is on; one-line state per agent
.\gauge.bat agents show PROJECT schemer      # full state
.\gauge.bat agents disable reader_panel      # saved to config
.\gauge.bat agents enable lean               # profiles: full | lean | off
.\gauge.bat agents setup PROJECT             # attach agents to an older project
```

`lean` keeps only the cheap agents (archivist, showrunner, line editor, pacing, librarian,
packager). Each full-room chapter adds roughly ten fast-model calls, run in parallel.

## Adding an agent

Subclass `StoryAgent` in `gauge/write/agents/`, give it a `name`, `title`, `weight` and a
pydantic `state_model`, implement any of `setup`, `plan_notes`, `brief`, `review`, `remember`,
`report`, start every prompt with `self.marker(op)`, register it in `council.ALL_AGENTS`, add a
canned answer to `agents/fakes.py` so it runs offline, and add a flag to `AgentsConfig`.
