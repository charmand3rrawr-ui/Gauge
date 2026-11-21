# Ethics and good citizenship

Gauge touches two sensitive things: other people's stories and other people's servers.

## Servers

- One clear user agent (`Gauge/0.1 (open-source web novel research tool)`), a delay between
  requests (2.5 s default), exponential back-off on 429/5xx, robots.txt honoured by default.
- Only public pages. No logins, no paywalls, no premium chapters, no bypassing anything.
- Incremental by design: a chapter is fetched once, ever.
- Keep `max_novels_per_run` and `max_chapters_per_novel` reasonable. The defaults fetch a few
  hundred pages per cycle, spread over minutes, which is less than a human binge-reader.

If a site's terms forbid automated access, do not point Gauge at it. Use `gauge import` with
files you are allowed to have.

## Research sources

The Scout reads Wikipedia and Wikivoyage (CC BY-SA), Project Gutenberg (public domain), arXiv
abstracts and Open-Meteo data through their public APIs, with a clear user agent, a delay and a
cache so nothing is fetched twice. The drafter is told to use facts and texture and never to
copy sentences; the bibliography in each project's `research/sources.md` records what was
consulted. General web search is only available through a SearXNG instance you run yourself.

## Other people's computers

The compute pool uses only machines where you have started `gauge worker` with your token.
Discovery listens for those workers' announcements; it does not scan address ranges, probe
ports or try credentials, and Gauge never enrols in cloud services on its own. Lending someone
else's machine to the pool requires their consent and their action: they must run the worker.

## Code that writes code

Bots are plain programs you can read; deterministic ones are refused a model at run time.
Self-modification is proposal-based, tested in isolation, and applied by you unless you switch
on automatic mode; protected files and existing tests are out of reach either way, and every
change is backed up and reversible. Gauge does not install software, change system settings or
touch files outside its repository and data folder on its own.

## Stories

Gauge is built to learn **patterns**, not to copy text:

- The engagement model sees only numbers (ratios, densities, lengths, cadence).
- Exemplars are 180-word openings and endings, shown to the model as *shape* references with an
  explicit "match the energy, never the content" instruction. They are retrieval context, not
  training data, and they are not published anywhere.
- The SFT export uses your own local imports by default. The preference export uses short
  excerpts. Fine-tuning on other authors' full chapters is a choice you make, with its own legal
  and ethical weight depending on your jurisdiction; Gauge does not make it for you.

## What you publish

A model can produce fluent prose that is derivative, inconsistent, or quietly wrong. Read every
chapter. Change what is not yours in spirit. Serial readers are sharp and loyal; they reward
authors who show up, not pipelines that ship. Gauge's best use is as an editor that has read ten
thousand chapters and can tell you, with numbers, that your ending resolves too much.

## Defending itself

Gauge defends itself only in ways that stay on its own machine: it scrubs
prompt injection out of scraped text, scans code before it runs it, sandboxes
its bots, seals and verifies its own files, keeps a tamper-evident audit log and
switches its own automation off when something looks wrong. It never scans,
probes, counter-attacks or deceives anyone else, and nothing in the security
package can be turned toward another system. See `docs/SECURITY.md` for what
Gauge is entitled to refuse and what it does when it is attacked.

## Disclosure

Many platforms now ask or require that AI-assisted work be labelled. Follow the rules of the
platform you post on.
