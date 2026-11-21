# The training library: text the AI may lawfully learn from

`gauge corpus` compiles the library GaugeLM and the learning models train on — and compiles it only from
sources you have the right to train on, tracking the licence and provenance of every document.

## Commands

```
gauge corpus stats                    # what the library would contain, by source and licence
gauge corpus build [--out DIR]        # export corpus.jsonl + corpus.txt + a licence MANIFEST
```

Both take `--include-third-party` to add scraped third-party fiction — off by default (see below).

## What goes in, and what does not

| source | licence | trained on by default |
| --- | --- | --- |
| Project Gutenberg (`gauge scout gutenberg`) | public domain | yes |
| files you imported (`gauge import`) | yours to use | yes |
| your own written chapters | yours | yes |
| scraped sites (Royal Road, Scribble Hub, …) | someone else's copyright | **no** |

Training a model on copyrighted fiction you scraped is a legal problem, so the compiler **excludes
third-party sources by default** and tells you how many documents it left out and from where. You can
add them with `--include-third-party`, but that is your legal decision and they are labelled in the
manifest so your training set is always auditable. The exported `MANIFEST.json` records the document
count, token estimate, and the licence and provenance of every source.

## Building the library up

- **Public domain:** `gauge scout gutenberg "<topic or author>"` imports Project Gutenberg books.
- **Your own files:** `gauge import <folder>` (see LOCAL_LIBRARY.md) imports text you hold the rights to.
- **Your serials:** every chapter Gauge writes is already part of the library.

Then `gauge corpus build`, and point training at the result. Honest note: at GaugeLM's size the model
learns style and structure from this corpus, not facts; a small public-domain-plus-your-own library is
enough to shape its voice, not to rival a large model.
