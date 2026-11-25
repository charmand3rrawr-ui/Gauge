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

## What goes in, and what does not

| source | licence | trained on by default |
| --- | --- | --- |
| Project Gutenberg (`gauge corpus fetch --source gutenberg`) | public domain | yes |
| Wikisource (`gauge corpus fetch --source wikisource`) | public domain (verify the work) | yes |
| Standard Ebooks / CC works (`gauge import --source-tag`) | public domain / Creative Commons | yes |
| files you imported (`gauge import`) | yours to use | yes |
| your own written chapters | yours | yes, unless unreadable (below) |
| scraped sites (Royal Road, Scribble Hub, …) | someone else's copyright | **no** |

One of your own chapters that is not readable prose (under `write.coherence_floor`: word salad a weak
model wrote and an older version accepted) is left out, counted under `unreadable` in the MANIFEST, and
reported by `stats` and `build`. `gauge chapters check PROJECT` shows the file so you can rewrite or
delete it; the compiler never deletes it.

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
