# The training library: text the AI may lawfully learn from

`gauge corpus` compiles the library GaugeLM and the learning models train on — and compiles it only from
sources you have the right to train on, tracking the licence and provenance of every document.

## Commands

```
gauge corpus stats                    # what the library would contain, by source and licence
gauge corpus fetch --source gutenberg --query "<topic>"   # pull public-domain books into the library
gauge corpus fetch --source wikisource --title "On Liberty"
gauge corpus build [--out DIR]        # export corpus.jsonl + corpus.txt + a licence MANIFEST
```

`stats` and `build` take `--include-third-party` to add scraped third-party fiction — off by default
(see below). `fetch` needs network access, so run it on your own machine or the daemon (it no-ops under
`GAUGE_OFFLINE`); it never reaches a scraped fiction site.

## The book finder: Gauge looks for novels by itself

```
gauge corpus find                     # go online now and add a few novels that fit your stories
gauge corpus find --topic "sea stories" --books 5
gauge corpus find --plan              # say what it would search for; download nothing
gauge corpus find --log               # what it has added so far, and why
gauge corpus find --by-itself on      # do this by itself, about once a day
```

In the Studio: **Teach Gauge**, **Find novels online**, with a box to **Keep finding novels by itself**.

**What it looks for.** Catalogue topics taken from the genres of your stories and of the books you
rated four stars or more, then any topics you name (`library.finder_topics`), then a general spread
of fiction. It starts from a different topic on each run. It takes novels only (prose fiction of at
least 20,000 words), most-read first, that the library does not already have under any source, and
splits each into chapters.

**Where it looks, and nowhere else.** Project Gutenberg, whose books are out of copyright in the US:

* the Gutendex catalogue service is asked first, with `copyright=false`, because it knows which books
  are most read;
* when that service does not answer (it is often slow), Project Gutenberg's own catalogue file is
  used instead, fetched at most once a month and searched on your computer;
* a book whose own first page says it is still in copyright is dropped.

The addresses it may talk to are fixed in the code (`gutendex.com` and `gutenberg.org`) and every
request is checked against them. It cannot be pointed at a web-novel site, and a book address that
leads anywhere else is refused before any request is made. Sites that host other people's
copyrighted novels are out of its reach by construction, not by a setting.

**Manners and limits.** Project Gutenberg asks that programs do not crawl its site and use its
catalogue files instead; the finder takes a few books per run (default 3), by itself at most once a
day, obeys robots.txt and spaces its requests. For hundreds of books, use Project Gutenberg's mirrors
rather than this. A ceiling (`library.finder_max_mb`, default 300 MB) stops it filling the disk.
Nothing about you is sent: a search is a word such as "adventure". Everything it adds is written to
`data/models/finder_log.jsonl` with the reason. Dry-run and offline mode stop it.

**By itself.** Off until you switch it on. Then it runs when a training session starts in the Studio
and in each daemon cycle, if a day has passed since the last run.

**Not run against the real sites.** From the machine this was built on, both the catalogue service
and Project Gutenberg answered "service unavailable", so the finder is tested against stand-in
answers in the shape those sites document, not live. The first real run is on your computer.

## What goes in, and what does not

| source | licence | trained on by default |
| --- | --- | --- |
| Project Gutenberg (`gauge corpus find`, or `gauge corpus fetch --source gutenberg`) | public domain | yes |
| Wikisource (`gauge corpus fetch --source wikisource`) | public domain (verify the work) | yes |
| Standard Ebooks / CC works (`gauge import --source-tag`) | public domain / Creative Commons | yes |
| files you imported (`gauge import`) | yours to use | yes |
| your own written chapters | yours | yes |
| scraped sites (Royal Road, Scribble Hub, …) | someone else's copyright | **no** |

Training a model on copyrighted fiction you scraped is a legal problem, so the compiler **excludes
third-party sources by default** and tells you how many documents it left out and from where. You can
add them with `--include-third-party`, but that is your legal decision and they are labelled in the
manifest so your training set is always auditable. The exported `MANIFEST.json` records the document
count, token estimate, and the licence and provenance of every source.

## Building the library up

- **Public domain:** `gauge corpus fetch --source gutenberg "<topic or author>"` pulls Project Gutenberg
  books (full text, boilerplate stripped); `--source wikisource --title "<page>"` pulls a Wikisource page.
- **Standard Ebooks & Creative Commons:** these ship as EPUB/text files — download them, then
  `gauge import <file> --source-tag standardebooks` (or `cc-by`, `cc0`, `publicdomain`) so the compiler
  knows the licence is safe to train on.
- **Your own files:** `gauge import <folder>` (see LOCAL_LIBRARY.md) imports text you hold the rights to.
- **Your serials:** every chapter Gauge writes is already part of the library.

Then `gauge corpus build`, and point training at the result. Honest note: at GaugeLM's size the model
learns style and structure from this corpus, not facts; a small public-domain-plus-your-own library is
enough to shape its voice, not to rival a large model.
