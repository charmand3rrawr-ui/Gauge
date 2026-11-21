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
gauge compute token [--rotate] [--reveal]
gauge settings list --group Security
```
