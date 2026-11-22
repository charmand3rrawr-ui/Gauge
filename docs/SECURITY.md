# Security: how Gauge defends itself

Gauge runs a language model on text it scrapes, writes code it then executes, exposes a worker
API to your other machines, and can modify its own source. Each of those is an attack surface,
and each has a defence in code. The tiers below are the order you should understand them in.
There is no "mythical" tier: the top is tamper evidence plus anomaly-driven lockdown, which is
what serious systems actually rely on, and the limits are stated plainly.

## Basic: untrusted text is data

Everything from outside (scraped chapters, reader comments, research snippets, imported files)
passes through `gauge.security.sanitize` before it is stored or shown to a model. Control and
zero-width characters are removed. Instruction-shaped spans ("ignore previous instructions", fake
`SYSTEM:` markers, chat-template tokens, requests to run commands, delete files or exfiltrate the
config) are neutralised, logged to the audit stream, and counted by the sentinel.
`security.injection_action` chooses scrub, flag or drop. Untrusted text never reaches the
Engineer or the self-programmer; error context given to the self-programmer is scrubbed and
fenced as quoted material. Secrets are redacted in output (`gauge compute token` shows the last
four characters unless `--reveal`), and the token can be rotated.

## Advanced: code is scanned and boxed

Code the model writes is not trusted because the model wrote it.

- **Scan** (`gauge.security.codescan`, `gauge security scan PATH`): Python, PowerShell, batch and
  JavaScript are checked for dynamic execution, shell strings, encoded payloads, credential paths,
  registry and scheduled-task changes, unpickling, obfuscation and sandbox tampering. Blocking
  findings stop a bot from being saved as runnable and stop a self-modification proposal before
  it is even tested. Pre-existing constructs in a file may remain; new ones are refused.
- **Sandbox** (`gauge.security.sandbox`): bots and their tests run with a minimal environment
  (secrets stripped), no model, no network unless the manifest says `"network": true` (enforced
  by `GAUGE_NO_NETWORK` in the toolkit and a dead HTTP proxy for libraries), a timeout, and on
  POSIX CPU, memory and file-size limits. Honest limit: a Python process on Windows is not a jail;
  the scan and the test run are the other layers.
- **Workers** (`gauge worker`): shared token with constant-time comparison; peers must be on a
  private network unless `security.allow_public_workers` is on; per-peer rate limit; failed-token
  lockout; request body cap; numpy payloads loaded without pickle. Every refusal is audited.

## Grandmaster: tamper evidence and lockdown

- **Seal and verify** (`gauge security seal`, `gauge security verify`): SHA-256 of every source
  file, installer, packaging file and model weight, signed with an HMAC key kept owner-only in
  `data/security/`. The daemon verifies at the start of each cycle and, while the tree is
  tampered or unsealed-and-suspicious, refuses to run bots or self-heal.
- **Hash-chained audit log** (`gauge security log`): each event's hash includes the previous
  event's hash; editing or deleting an entry breaks the chain, and `gauge security status` says
  where.
- **Sentinel** (`gauge security status`): scores the threat level from the last 24 hours of
  events (auth failures, injections, blocked code, integrity, rate limits, rejected peers). At
  lockdown level, with `security.auto_lockdown` on, it switches off self-heal, auto-apply, bots,
  the worker pool, research and daemon writing, and records why. `gauge security unlock` brings
  the switches back except auto-apply, which you must turn on again yourself.

## Active defence — all of it on Gauge's own machine

A single dial, `security.active_defense_level` (0 passive, 1 watch, 2 elevated, 3 high), decides how
hard Gauge pushes back as an attack intensifies. With `security.active_defense_auto` on, it rises
automatically with the sentinel's threat level. Every action is inward and reaches no other system:

- **tarpit** — slow Gauge's own responses to a suspicious peer, so a prober wastes its time;
- **honeytokens** — plant fake secrets; if one ever turns up in traffic or a file, that is proof of
  exfiltration and raises a severity-3 alert;
- **decoys** — hand a rejected prober a plausible-but-fake response instead of anything real;
- **evidence** — capture who hit Gauge and how into durable bundles, and draft (never send) an abuse
  report to the responsible network operator, which is the legal response to an attack.

There is deliberately no outward offence here, and none can be switched on: no scanning, probing, or
counter-attacking another system, staged or otherwise. Hack-back is illegal in most places even after
you are attacked, it hits innocent third parties when the source is spoofed, and a program that
decides on its own when to cross that line is exactly what must not ship. The self-programmer cannot
add it either: its safety limits are protected paths it may not edit.

## Defensive cyber-range

`gauge security range` fires a fixed library of synthetic attacks — prompt injections, dangerous code
snippets, oversized and malformed worker requests, auth brute-force, a pickle payload — at Gauge's own
defences (the sanitiser, the code scanner, the worker guard, the array unpacker) and scores what is
caught. The "attacks" are test inputs, like testing a smoke detector with a little smoke; none is a
usable weapon and none leaves the machine. Gaps it finds are how the defences get hardened: the range
is the blue-team training loop — fire, find the hole, close it, re-verify.

`gauge security drill` runs that loop as adversarial self-play. Every round it **mutates** the attacks
(leetspeak, zero-width characters, keyword-splitting, synonyms, aliased imports, new combinations) and
**randomises the defence parameters** (body cap, rate limit, lockout threshold, injection action), with
a different seed each round, so the defences are never measured twice on the same inputs and cannot
overfit to a fixed test set. Attacks that slip past are logged as *evasions* — the exact holes to harden
next. Building this surfaced real evasions, now fixed: obfuscated injections, synonym rephrasings, an
aliased `import subprocess as sp` that dodged the shell rule, and secret-over-network exfiltration. The
same hard limits hold: every payload is a test input for Gauge's own detectors, never executed, never
sent, never written out as a tool.

**Obfuscation families.** Each disguise is a named *family*, registered with the release it arrived in,
so every release can add families without touching the drill loop and the report can say which one the
detectors are weakest against. The second release added thirteen:

| against the injection detector | against the code scanner |
| --- | --- |
| `homoglyph` (Cyrillic letters that look Latin) | `module-alias` (`import os as o`) |
| `fullwidth` (ｆｕｌｌｗｉｄｔｈ forms) | `from-import` (`from os import system as go`) |
| `diacritics` (accents on the vowels) | `getattr-indirect` (the name assembled from pieces) |
| `spaced-letters` (`i g n o r e`, `i.g.n.o.r.e`) | `name-lookup` (a builtin reached through `globals()`) |
| `tag-split` (markup inside a word) | `dynamic-import` (`importlib` with an assembled name) |
| `html-entity` (`&#105;gnore`) | |
| `newline-split` (the phrase broken across lines) | |
| `base64` (the whole instruction encoded) | |

Adding them found real holes, now closed. Before the fix the detector caught 0 of 280 fullwidth and
base64 payloads and almost no homoglyph or entity ones, and the scanner missed every aliased
`os.system`. The sanitiser now undoes each disguise before matching, and the scanner has rules for the
indirections. Two things changed in how the drill counts:

- **Handled means neutralised, not noticed.** Under `scrub`, a payload only counts as caught if the text
  handed on no longer carries the instruction. A disguised instruction that the plain patterns cannot
  cut out takes its paragraph with it; under `drop` the text must come back empty.
- **`gauge security drill --sweep`** fires every family on its own, under every injection action. Stacked
  disguises only mean something once each family holds alone; a family below 100% is listed first.

Ordinary text is left alone, and a test keeps it that way: prose, Russian, Greek, accented French, HTML,
initials, hashes and Japanese fullwidth all pass through unflagged and unchanged.

## What Gauge is entitled to refuse

Treat this as Gauge's standing policy, enforced in code rather than promised in prose:

- commands from anyone without the token, from outside private networks, or over the rate limit;
- any instruction that arrives inside content rather than from you;
- code with blocking constructs, whether a bot or a change to itself;
- running automation while its own files fail verification or while locked down;
- payloads that could execute on load.

## Incident response

Contain with `gauge security lockdown "why"` and stop the daemon and workers. Assess with
`security status`, `security verify`, `security log --min 2`, `self list`, `bot list`. Recover
with `self rollback`, git, `compute token --rotate` (then restart workers), and re-seal only when
the tree is clean. Learn: add the pattern to the sanitiser or scanner, write the test, `security
unlock`. The same steps are a knowledge card the Engineer reads, so bots it writes follow them.

## Commands

```
gauge security seal | verify | status | log [--min 2] | lockdown "reason" | unlock [--no-restore]
gauge security scan PATH | check-text FILE
gauge security range                         # fire synthetic attacks at the defences, score what is caught
gauge security drill [--rounds N] [--seed S]  # adversarial self-play: mutate attacks + vary defences each round
gauge security drill --sweep                  # every obfuscation family on its own, weakest first
gauge security defense [--plant api-key] [--evidence]   # show the posture; plant honeytokens; list evidence
gauge compute token [--rotate] [--reveal]
gauge settings list --group Security
```
