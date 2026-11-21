# Realism: grounding prose in the senses and a consistent world

Realism in fiction is mostly a scaffolding problem, not a base-model problem. These measured checks catch
what makes AI prose read as unreal — no texture, feelings announced instead of shown, the reader held at
arm's length, time of day that contradicts itself — and feed concrete fixes to the reviser.

## Two measured agents (no model calls)

- **Sensory Editor** (`agents.sensory`). Scores each chapter on how many senses it engages, whether
  emotion is shown or told ("she was afraid"), how often filter verbs intrude (saw/heard/felt/noticed),
  and how specific the detail is (a named thing vs "a car"). It briefs the drafter with last chapter's
  gaps. `gauge.write.realism.sensory_profile`.
- **Continuity Auditor** (`agents.continuity`). Guards logistics the way the Archivist guards facts: it
  flags a chapter whose time of day contradicts itself with no passage of time, and a season or time of
  day that flips between consecutive chapters. `gauge.write.realism.logistics_check`.

## Grounding from research

With `write.grounding` on (and the Scout enabled), each scene gets a **grounding dossier** — the real
sensory and procedural texture of its setting, fetched as research and folded into the draft brief, so
detail is accurate rather than generic. `gauge.write.grounding.build_dossier`. It needs the network;
offline, the writer falls back on invention.

## The realism dial and world mode

- **`write.realism_weight`** (default 0) adds a share of the chapter score from sensory grounding and
  show-don't-tell. It is off by default on purpose: **realism trades against pace**, and the most "real"
  version of a scene is not always the most bingeable. Turn it up for literary work, leave it low for
  fast serial fiction.
- **`write.world_realism`** tells the World agent to ground geography, weather and procedure in the
  actual world instead of inventing them — for contemporary or historical settings rather than fantasy.

All of this is additive: the agents join the writing room, the dial and world mode are opt-in, and
nothing changes the default behaviour of an existing project until you turn it on.
