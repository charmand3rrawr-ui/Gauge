# Marketing: assets and a plan you run yourself

Gauge helps an author market their own web novel. It generates the material and a posting plan; the
author posts it, under their own identity, following each community's rules. Gauge does not create
accounts, post to third-party platforms, or message anyone — and there is no code in it that could.

## What it does

- **`gauge market kit PROJECT`** — a launch kit from the project's bible: tagline, logline, short and
  long blurbs, tags, and an author's note. Saved to `projects/<slug>/launch_kit.md`.
- **`gauge market plan PROJECT --url <link>`** — a posting plan: for each channel, the venue's
  self-promotion rules, the disclosure it expects, and a ready-to-adapt draft that always carries the
  line *"Posted by the author. This is self-promotion of my own web novel."* Saved to
  `projects/<slug>/posting_plan.md`.
- **`gauge market channels`** — where authors legitimately share a serial (Royal Road, genre
  subreddits, Discord promo channels, your own newsletter and social accounts, writing forums), with
  each one's rules and the one habit that keeps you welcome rather than flagged as spam.
- **`gauge market policy`** — the boundary, below, printed from the code that enforces it.

## What it will not do, and why

The promotion policy lives in `gauge/market/policy.py` as a single gate, `check_action`, and the
refused actions have no implementation anywhere in the package:

| refused | why |
| --- | --- |
| create accounts | automated sign-up defeats the CAPTCHA/verification that exists to stop spam, and breaks every platform's terms |
| auto-post to platforms | posting from automation or fake identities is spam; undisclosed promotion is astroturfing |
| mass-DM people | messaging strangers in bulk is unsolicited spam |
| scrape contacts | harvesting people "with interest" to target them is surveillance of private individuals |
| sockpuppet / fake reviews | posing as readers or fabricating reviews is fraud, and it is detectable |

These are not throttled or gated-behind-a-switch; they are absent. The legitimate version — assets plus
a rules-respecting plan a human executes with disclosure — is the version that actually gets a book
read instead of banned.

## Doing it right

The durable channel is the one you own: a steady release schedule and a mailing list grown from a link
in your own chapters. Promotion in communities works only in the threads and channels that allow it,
with flair and disclosure, from someone who takes part rather than drops a link and leaves.
