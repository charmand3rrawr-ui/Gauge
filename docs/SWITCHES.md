# The switches

Every behaviour a person might want to flip is a **switch**: a labelled setting with a plain
description, a valid range, and a red warning where it affects other people, your machine, or
safety. One catalog (`gauge/settings.py`) drives both places you can flip them:

- **Gauge Studio -> Switches**: a window of checkboxes, sliders and menus grouped as Writing,
  The room, Learning, Model and Automation. Changes save as soon as you make them and the next
  job uses them.
- **The command line**: `gauge settings list`, `gauge settings on KEY`, `off KEY`, `set KEY VALUE`,
  `show KEY`, `reset KEY`.

## Rebel craft mode

`write.rebel` makes every rule of craft a default rather than a law. The muse must propose
directions that deliberately break a convention the story has been obeying (structure, genre
expectation, pacing rhythm, point of view, tone) and say why the break is earned. The writer's
system prompt invites opening on stillness, refusing the cliffhanger, letting the villain be
right, changing form for a chapter. The critic judges effect, not compliance: a break that lands
is a 9 or 10, a break that fumbles is a 3. `write.rebel_level` sets how much of each outline
must break something. This touches only rules of craft; it changes nothing about how Gauge
treats other people's servers, work or machines.

## The catalog

| group | switches |
| --- | --- |
| Writing | rebel mode and level, persona blend, novelty weight, quality bar, revision limit |
| The room | characters, character weight, improv, world, world agency and weight, muse, room size (full/lean/off), room weight, content flags block |
| Learning | prompt bandit, fine-tuning pairs, lessons in prompt, Scout research, honour robots.txt (!), fact-check |
| Model | provider, writer model, fast model, GaugeLM device, auto GPU switch, workers for generation, workers for training |
| Security | instruction-shaped text handling, verify sealed code each cycle, automatic lockdown, scan generated code (!), accept public workers (!) |
| Automation | daemon writes chapters, scheduled bots, self-heal proposals (!), auto-apply self-fixes (!) |

Switches marked (!) carry a warning you will see when turning them on:

- **Honour robots.txt** off: fetches from sites whose robots.txt says no (arXiv, Open-Meteo), on
  your decision.
- **Self-heal**: Gauge writes proposals to change its own source; they are tested in isolation and
  wait for you.
- **Auto-apply self-fixes**: Gauge commits changes to its own code while you are not watching.
  Protected files and tests stay out of reach and every change can be rolled back, but keep it
  off unless you trust it.

What has no switch, by design: scanning networks or using machines that have not been set up as
your workers, touching the installers, packaging or the self-modification module, and removing or
skipping tests. Those are not behaviours of a writing tool, so there is nothing to turn on.
