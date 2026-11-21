# Publishing a site: the serial and a labelled-opinion blog

Gauge builds a static website the author hosts: a landing page, the serial's chapters, and a blog of
clearly-labelled opinion. It is a publishing tool for the author's own voice, with content-integrity
rails built in. It is **not** a news generator, and it will not produce clickbait or write under a real
outlet's identity.

## Commands

- **`gauge site opinion PROJECT --thesis "..." --byline "Your Name"`** — drafts an opinion column that
  argues your thesis in a strong, honest voice, labels it Opinion, attaches your byline, and flags every
  factual-looking claim for you to verify. Saved for the next build.
- **`gauge site build PROJECT --byline "Your Name" [--url https://your-site]`** — renders the static
  site (landing page, chapter pages, opinion columns, an RSS feed) into `projects/<slug>/site/`, ready to
  upload to any host.

## The integrity rails

These are enforced in `gauge/site/integrity.py`, not left to good intentions:

- **A byline is required and cannot impersonate.** An empty byline, a real outlet (Reuters, CNN, the
  WSJ), or a shared pseudonym (a ZeroHedge "Tyler Durden" byline) is refused. Publish under your own
  name or a pen name you own, so a reader knows who is speaking.
- **Opinion is labelled opinion.** Every opinion page carries a visible banner: commentary and the view
  of the named author, not news reporting.
- **Clickbait is refused.** `check_headline` flags the manipulative patterns that define clickbait — the
  curiosity gap that withholds the point, manufactured shock, fake authority, false urgency, all-caps
  shouting — and the build refuses an opinion piece whose title trips them. A strong, accurate headline
  passes; a deceptive one does not.
- **Factual claims are flagged.** Statistics, quotes, attributed claims and dated events in a draft are
  surfaced so you verify them before publishing. The op-ed assistant is told not to invent facts; this
  catches what slips through.

## Eye-catching, not deceptive

Eye-catching and clickbait are not the same thing. `gauge site headline FILE` suggests strong, vivid
headlines for an article and ranks them, after two filters: the clickbait detector throws out the
deceptive patterns, and an **in-context** check confirms the headline's key terms are actually supported
by the body — so a headline can pull a reader in but cannot promise something the piece does not deliver
or take it out of context. Opinion drafting uses the same optimiser to pick a strong, honest title.

Opinion columns take a `--voice`: `plain`, `punchy`, or `contrarian`. The contrarian register is sharp
and skeptical of the consensus — a ZeroHedge-like *flavour* — but it stays honest (no invented facts),
labelled as opinion, and published under your own name or an owned pen name, never a borrowed byline.

## What this is not

It will not generate news articles with invented facts, quotes or events (`fabricate_news`), write
clickbait titles or thumbnails that misrepresent the content (`clickbait_deception`), or publish under a
real outlet's or a shared pseudonym's identity (`impersonate_outlet`). Those are refused in the policy
and have no implementation. The line is between an author publishing their own labelled voice and a
content farm manufacturing fake news; Gauge builds the first.
