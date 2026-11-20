# The World agent

Characters decide what they want. The **World** decides what is true around them, and it is in
every scene.

## What it owns

- **Rules**: physics (what cannot happen, what always costs), power (magic / tech / system rules
  and their prices), society (law, money, class, institutions, taboos).
- **The clock and the sky**: time, season, weather, continuous across chapters.
- **Places**: a registry with one line of texture each, growing as the story visits them.
- **Active events**: things happening whether or not the cast is involved (a strike, a siege, a
  festival, a plague).
- **Consequences**: actions whose bill has not arrived yet. The World remembers them and brings
  them back.
- **Background figures**: nameless people with one trait each. The World speaks through them.

All of it lives in `projects/<slug>/world.json`, rendered to `world.md`, built from the bible
and premise when a project is created, and updated after every chapter.

## How it acts in a scene

Before each chapter, after the characters have briefed the drafter, the World produces its own
brief: the sensory setting, which rules are in play, what the world *does*, which nameless
people step in, which consequences surface, and the continuity facts that must hold.

How much it does depends on the scene. `intervention_level` is computed from the tension in the
plan, from how many characters are pulling against the plan, from a little per-chapter variety,
all scaled by `write.world_agency` (0 to 1, default 0.6):

| level | the World... |
| --- | --- |
| below 0.35, *ambient* | texture, weather, one passerby at most, no interventions |
| 0.35 to 0.7, *active* | a crowd reacts, a vendor refuses, a mechanism fails, an institution notices |
| above 0.7, *forceful* | the world turns against or around the cast; consequences land now |

After the draft, the World reviews it in its own domain only: rules obeyed, time and weather
continuous, places as established, consequences not forgotten. Violations go into the revision
pass and `write.world_weight` (default 0.1) of the chapter score comes from this consistency.
Then the World remembers: clock and weather move on, new places are registered, events begin and
end, consequences are added or paid, and every nameless figure who appeared is counted.

## Nameless people and promotion

Background figures are tracked by tag ("the one-armed noodle vendor"), never by name. The World
is told not to name them. When one keeps appearing, it shows up in `gauge world extras` as
promotable, and `gauge world promote PROJECT "the one-armed noodle vendor" --name "Old Pak"`
turns them into a full character with a personality matrix, keeping the tag as an alias so
earlier chapters still connect.

## Commands

```powershell
.\gauge.bat world show PROJECT                     # rules, clock, places, events, consequences, extras
.\gauge.bat world build PROJECT                    # (re)build from the bible and premise
.\gauge.bat world set PROJECT --rule "..." --weather hail --time "dawn, day 4" --event "..." --agency 0.8
.\gauge.bat world chat PROJECT --ask "who is in the market right now?"
.\gauge.bat world extras PROJECT
.\gauge.bat world promote PROJECT "<tag>" --name "<Name>" [--role minor]
```

Set `write.world_enabled` to false to turn the World off.
