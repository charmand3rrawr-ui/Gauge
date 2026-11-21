# The newsdesk: trending topics, any language, original commentary on a schedule

The newsdesk finds what the world is reading, in many languages, and writes **original** commentary on
it — in the platform's declared stance and comedic voice, citing and linking its sources — then publishes
on a schedule you set. It is a commentary desk, not a content farm: it never rewrites other outlets'
articles, and anything asserting a checkable fact waits for you.

## How a cycle works

1. **Discover.** Wikipedia's most-viewed pages per language (`newsdesk.languages`, default
   en/es/fr/de/ja/pt) via the public Wikimedia pageviews API, plus RSS/Atom feeds you choose
   (`newsdesk.feeds`). A topic trending in several languages ranks higher. Everything goes through
   Gauge's polite fetcher: robots.txt honoured, rate-limited. Only headlines, summaries and links are
   read — never article bodies.
2. **Research and translate.** Feed items that mention the topic become research. Non-English coverage
   is translated by your main model (`newsdesk.target_language`). GaugeLM, the small built-in model,
   does not "learn" many languages — the main model translates, honestly.
3. **Write original commentary.** The writer gets the sources as research and is told to produce its own
   analysis, in the declared stance (see SITE.md), engaging the strongest opposing view, without copying
   or inventing facts. Every piece is labelled Opinion, bylined, and ends with a **Sources** list linking
   the originals.
4. **Check.** An **originality check** measures how much of each source's wording was reproduced (shared
   five-word sequences as a share of the source); past `newsdesk.max_source_overlap` (25%) the draft is
   flagged as too close. The claim flagger, the balance check and the headline check run too.
5. **Queue and publish.** Drafts enter as `pending_review`. At most one piece goes out per
   `newsdesk.interval_minutes` (default **30**, change it any time) and no more than `newsdesk.max_per_day`.
   A piece is published when you approve it — or, if you turned on `newsdesk.auto_publish`, when it is
   completely clean: no claims awaiting verification, no balance, headline or originality flags.
   Publishing adds it to the site project (`newsdesk.project`) for the next `gauge site build`.

## Commands

```
gauge newsdesk trends [--lang en --lang ja] [--feed URL]   # what the world is reading
gauge newsdesk draft "TOPIC"                               # original commentary into the review queue
gauge newsdesk queue [--status pending_review]              # what is waiting
gauge newsdesk show ID                                      # read a draft with its flags
gauge newsdesk approve ID | reject ID
gauge newsdesk run [--no-draft]                             # one cycle
gauge newsdesk watch [--max-minutes N]                      # a cycle every interval, until Ctrl+C
```

Turn on `newsdesk.enabled` to run a cycle inside the daemon instead. Set `newsdesk.byline` (your name or
an owned pen name — outlets and shared pseudonyms are refused) and `newsdesk.project` first.

## What it will not do

- **Rewrite other outlets' articles** (`rewrite_republish`). Paraphrase-spinning someone else's reporting
  into your article is plagiarism, and at scale copyright infringement — the content-farm mechanic. Facts
  are free to use; someone's article is not. The originality check enforces this.
- **Publish checkable claims with no human review** (`auto_publish_unreviewed_news`). Auto-publishing even
  allows only fully clean drafts; a piece with a statistic, a quote or a dated event waits for you.
- **Run during a lockdown.** The security sentinel switches the newsdesk and auto-publishing off with the
  rest of the automation, and auto-publishing never comes back on by itself.

At a 30-minute cadence the honest workflow is: the desk drafts and queues continuously, you skim the
queue a few times a day, approve what is right, and the schedule spaces out what you approved.
