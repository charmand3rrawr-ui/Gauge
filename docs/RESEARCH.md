# The Scout: research that makes it real

Characters, the World and the writing room all work from what the story has already said.
The **Scout** brings in what the story does not know yet: how a trade is actually practised,
what a place is like in that month, what a technology can and cannot do, how people of a period
spoke. It searches open sources, caches everything, files what it finds into the project, and
fact-checks the draft's real-world claims.

## Sources (no keys, no accounts)

| source | what it is for | licence note given to the drafter |
| --- | --- | --- |
| `wikipedia` | facts: history, technology, professions, biology, geography, institutions | CC BY-SA: use facts and texture, never copy sentences |
| `wikivoyage` | place texture: streets, markets, food, how people move around | CC BY-SA |
| `gutenberg` | period prose and vocabulary from public-domain books, via Gutendex | public domain: quote sparingly, transform freely |
| `arxiv` | abstracts for hard-science plausibility | abstract only; use the science |
| `climate` | real weather for a real place and month, via Open-Meteo ("Lisbon in November") | open data |
| `searx` | general web search through **your own** SearXNG instance (`research.searx_url`) | verify before relying on it |

Requests go through the same polite fetcher as ingestion (clear user agent, delay, robots.txt
honoured). Wikipedia and Wikivoyage are read through `api.wikimedia.org`, the Wikimedia
developer API host, whose robots.txt permits clients; the wiki sites themselves disallow
automated access to their API paths and are not touched. Gutendex allows our user agent.

**arXiv and Open-Meteo** publish a robots.txt that disallows all automated access, even
though their API terms welcome slow, identified clients. Gauge therefore does not enable them
by default. If you decide to use them, add them to `research.sources` and set
`research.respect_robots_txt: false` yourself; that choice is yours, not the software's.

Every answer is cached in the database for `research.cache_days` (30), so repeated questions
cost nothing. `GAUGE_OFFLINE=1` switches all sources off.

## In the writing room

As a member of the room (`agents.scout`), before each chapter the Scout asks the fast model
which two or three concrete lookups would make the planned scene believable and which source
fits each, runs them, and files the results under `projects/<slug>/research/scout/` with a
bibliography in `research/sources.md`. The drafter gets the findings as *real-world detail
found for this scene*, with the instruction to use facts and texture and never copy sentences.
The Librarian indexes the same folder, so everything the Scout learns is available to every
later chapter.

After drafting, the Scout extracts up to three checkable real-world claims from the text
(skipping the story's own inventions), looks them up, and asks the model for a verdict.
Contradicted claims lower the chapter score and go into the revision prompt.

When a project is created, the Scout also builds a **project dossier** from the proper nouns
and trades in the bible (`research.dossier_on_create`).

## Public-domain books as learning material

`gauge scout gutenberg "dracula" --import` pulls a public-domain novel into the corpus as a
novel of source `gutenberg`. It is then analysed like any other, feeds the exemplar library and
`gauge absorb`, and can be used for fine-tuning without the questions that hang over modern
copyrighted text.

## Commands

```powershell
.\gauge.bat scout ask "medieval courier guild"                      # look something up
.\gauge.bat scout ask "Lisbon in November" --source climate
.\gauge.bat scout ask "night market" --source wikivoyage --project "Ten Seconds"   # and file it
.\gauge.bat scout dossier "Ten Seconds" --topic "sealed letters" --topic "debt bondage"
.\gauge.bat scout gutenberg "Dracula" --import
.\gauge.bat scout log --project "Ten Seconds"
.\gauge.bat research "Ten Seconds" "who built the sewers?"          # ask the Librarian about filed notes
```

## Settings

`research.sources` (default wikipedia, wikivoyage, gutenberg), `research.respect_robots_txt`, `research.searx_url`, `research.max_queries_per_chapter` (3),
`research.results_per_query` (3), `research.cache_days` (30), `research.fact_check`,
`research.dossier_on_create`, `research.request_delay_seconds` (0.7). Turn the whole thing off
with `research.enabled: false` or `gauge agents disable scout`.
