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

## Social PR on accounts you own

`gauge market social PROJECT --account platform:handle [--weeks N] [--per-week K] --url LINK` plans a
content calendar for accounts **you own**: distinct posts varied by angle (teaser, quote, logline,
invitation), scheduled across the weeks, each carrying your story link. It is a dry run — it previews
what would post and sends nothing. Live posting plugs in per platform through that platform's official
API, behind the same guard.

The guard (`gauge/market/social.py`) will only post to an account the caller affirms it owns
(`i_own_this`) and that uses the official API; there is no code for engagement, replies, DMs, or other
people's communities. The variety between posts exists because repeating one post is low-quality
marketing — it is not, and the policy refuses, variation to slip past spam detection
(`evade_spam_detection`). Buying or botting engagement (`buy_engagement`), reply-bots (`reply_bot`) and
auto-posting into communities you don't own (`cross_community_post`) are refused and unimplemented.

This is social-media management of your own presence, the way a scheduling tool works — not outreach
into communities, and not anything that pretends automated promotion is organic activity.

### Publishing for real

Adapters ship for platforms with clean official posting APIs: Mastodon, Bluesky, Telegram (your own
channel), Discord (a webhook on your own server), and a generic webhook to any endpoint you run. Add
`--send` to `gauge market social`, with the platform's credentials in the environment:

```
GAUGE_MASTODON_URL, GAUGE_MASTODON_TOKEN
GAUGE_BLUESKY_APP_PASSWORD
GAUGE_TELEGRAM_BOT_TOKEN, GAUGE_TELEGRAM_CHAT_ID
GAUGE_DISCORD_WEBHOOK_URL
GAUGE_WEBHOOK_URL
```

Credentials live in the environment, never in `config.json`. Each adapter runs the owned-account guard
on every post and, without its credentials, stays a dry run and says what is missing. Platforms whose
APIs need your own registered OAuth app (X/Twitter, Meta, LinkedIn, Reddit) are reached through a
webhook to a small service you run that holds those tokens — Gauge never holds them.

## Your opt-in mailing list

The one channel you own outright. `gauge market subscribers import FILE` imports a list you exported of
people who **signed up** — each row must carry consent (a signup/consent date, or an opt-in flag), and
rows without it are refused, not stored. `gauge market subscribers unsubscribe EMAIL` marks someone
gone, and a later import will not resurrect them.

`gauge market newsletter PROJECT --url LINK --chapter TITLE` composes a new-chapter email and, by
default, previews it to your consented subscribers without sending. Every message carries your sender
identity and a working unsubscribe link — `send_newsletter` refuses to run without both, and only ever
sends to addresses still marked subscribed. To send for real, pass `--from-name`, `--from-email`,
`--unsubscribe-url` and `--send`, with your own SMTP server in `GAUGE_SMTP_HOST` / `GAUGE_SMTP_USER` /
`GAUGE_SMTP_PASSWORD`. Gauge will not import a list without consent (`email_nonconsent`), and bought or
scraped lists (`purchase_list`) are refused outright.

## Doing it right

The durable channel is the one you own: a steady release schedule and a mailing list grown from a link
in your own chapters. Promotion in communities works only in the threads and channels that allow it,
with flair and disclosure, from someone who takes part rather than drops a link and leaves.
