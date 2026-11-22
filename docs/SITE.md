# Publishing a site: the serial and a labelled-opinion blog

Gauge builds a static website the author hosts: a landing page, the serial's chapters, and a blog of
clearly-labelled opinion. It is a publishing tool for the author's own voice, with content-integrity
rails built in. It is **not** a news generator, and it will not produce clickbait or write under a real
outlet's identity.

## Commands

- **`gauge site opinion PROJECT --thesis "..." --byline "Your Name"`** — drafts an opinion column that
  argues your thesis in a strong, honest voice, labels it Opinion, attaches your byline, and flags every
  factual-looking claim for you to verify. Saved for the next build.
- **`gauge site build PROJECT --byline "Your Name" [--url https://your-site] [--theme classic|paper|slate|noir]`**
  — renders the static site (landing page, chapter pages, opinion columns, feeds) into
  `projects/<slug>/site/`, ready to upload to any host.

## Themes and the light/dark toggle

Four palettes: `classic` (the default serif), `paper` (warm, bookish), `slate` (cool, sans-serif) and
`noir` (high contrast, suits thrillers). Each has a light and a dark variant, and in both the body text,
muted text and link colour clear WCAG AA contrast against the background — a test enforces that for
every theme.

Pages follow the reader's system preference. A **Dark mode / Light mode** button at the top of every page
overrides it and remembers the choice in the reader's browser. The button only appears when JavaScript is
available, so nobody gets a dead control; without it the system preference still applies.

## Feeds

Every build writes:

| file | what |
| --- | --- |
| `feed.xml` | RSS 2.0, everything, newest first |
| `atom.xml` | Atom 1.0, everything |
| `feeds/serial.xml` | RSS, the serial's chapters only (when there are any) |
| `feeds/opinion.xml` | RSS, opinion columns only (when there are any) |

Each item carries its byline and a category (Chapter, Opinion, Post). Opinion stays labelled as opinion
in a feed reader: its summary is prefixed `[Opinion]`. Pages advertise the feeds for auto-discovery.
Pass `--url` so feed links, ids and the self-link are absolute; without it the feeds are still
well-formed but their links are relative, which most readers handle poorly.

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

## A declared editorial stance

A platform has a point of view; the honest thing is to declare it rather than pretend to be neutral.
`gauge site stance` shows the stance and `--set FILE` customises it. The default is an evidence-first
centrist desk: fiscal prudence, compassion for people, and following the evidence rather than the
tribe — leaning conservative or liberal only where the evidence and those principles point, and saying
plainly which way a piece lands and why.

`gauge site opinion` writes from that stance by default (`--no-stance` to turn it off), in a punchy,
comedic register (`--no-comedic` to drop the jokes). Two things keep "moderate and logical" honest
rather than merely asserted:

- **A balance check.** A finished column that argues only one side is flagged: it should state the
  strongest opposing view and answer it. Steelman, not strawman.
- **The facts stay separate.** The stance shapes opinion, not reporting. It is no licence to bend facts:
  claims are still flagged for verification, opinion is labelled, satire is labelled, and nothing is
  invented.

`gauge site build` publishes a **Where we stand** page from the stance, so every reader sees the lens.
It says in as many words that it is a point of view, not a claim of neutrality. There is no "pure
objectivity" on offer here — a stated, consistent, evidence-first lens is the honest substitute.

## What this is not

It will not generate news articles with invented facts, quotes or events (`fabricate_news`), write
clickbait titles or thumbnails that misrepresent the content (`clickbait_deception`), or publish under a
real outlet's or a shared pseudonym's identity (`impersonate_outlet`). Those are refused in the policy
and have no implementation. The line is between an author publishing their own labelled voice and a
content farm manufacturing fake news; Gauge builds the first.
