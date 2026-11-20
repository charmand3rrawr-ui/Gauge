# Learning from your own folders

Everything Gauge learns from Royal Road it can learn from a folder on your disk: your own
drafts, exports from sites without a connector, EPUBs you own, a decade of saved serials.
Local novels get the same features, feed the same exemplar library, and can train the same
engagement model when you give them labels.

## What you can point it at

```powershell
.\gauge.bat import "D:\Books\My Serial"          # one novel: a folder of chapter files
.\gauge.bat import "D:\Books\Ash and Salt.epub"  # one novel: a single book file
.\gauge.bat import "D:\Books"                    # a whole library (auto-detected)
.\gauge.bat import "D:\Books" --watch            # ...and keep it in sync every cycle
```

Formats: `.txt` `.md` `.html` `.docx` as chapter files or whole books, `.epub` as whole books.
No extra packages are needed for EPUB or DOCX. Encodings are detected (UTF-8, UTF-8 with BOM,
Windows-1252, Latin-1).

### A novel folder

```
My Serial/
  novel.json          optional metadata + numbers (see below)
  chapter_001.txt     one chapter per file, sorted naturally (1, 2, ..., 10, not 1, 10, 2)
  chapter_002.md      Markdown heading on line 1 becomes the chapter title
  chapter_003.docx
  README.md           ignored (README, notes, cover, outline, bible, toc ... are skipped)
```

A folder with a single `.epub` / `.docx` / long `.txt` is also a novel; the book is split into
chapters on headings.

### A single book file

Chapters are found by, in order: `Chapter 7` / `Chapter Seven` / `CHAPTER VII` / `Part 2` /
`Episode 3` headings, Markdown `#` headings, bare numbers on their own line, and finally fixed
2,500-word pieces for a heading-less book. Text before the first heading becomes a Prologue when
it is long enough, and is dropped when it is front matter.

### A library folder

```
Books/
  library.csv                         optional labels for everything below
  Ann Author - First Serial/          folder name gives author + title
    chapter_1.txt ...
  second_serial/
    novel.json                        per-novel metadata wins over library.csv
    c1.md c2.md ...
  Third Book by E. Pub.epub           loose book files are novels too
  Completed/                          nested libraries are walked recursively
    fourth/
      1.txt 2.txt
```

Detection is automatic: a folder is a library when its children are novel folders or book
files rather than chapter files. Force it with `--library` or `--novel`.

## Metadata and labels

Priority, highest first: command-line flags, `novel.json` (also `metadata.json`,
`gauge.json`; or `<book>.json` next to a single file), `library.json` / `library.csv` at the
library root, EPUB/DOCX embedded metadata, the folder or file name
(`Author - Title`, `Title by Author`), then `Title:` / `by Author` lines at the top of the text.

`novel.json`:

```json
{
  "title": "First Serial", "author": "Ann Author",
  "blurb": "What if every lie you told came true the next morning?",
  "genres": ["Fantasy", "Mystery"], "tags": ["Female Lead", "Progression"],
  "followers": 15000, "favorites": 2400, "views": 850000, "average_views": 12000,
  "rating": 4.6, "rating_count": 310, "comments": 5200,
  "user_rating": 5,
  "chapters": {"1": {"views": 90000, "comments": 400}, "2": {"views": 61000}}
}
```

`library.csv` (any subset of columns; `name` is the folder or file name):

```csv
name,title,author,followers,rating,stars,genres
Ann Author - First Serial,,,15000,4.6,5,Fantasy|Action
Third Book by E. Pub.epub,,,,,2,
```

Synonyms are accepted: `genre`/`subject` for genres, `description`/`synopsis` for blurb,
`stars`/`my_rating`/`score` for `user_rating`. Numbers may contain commas.

### Your own taste as a label

Most local novels have no public engagement numbers. Give them yours:

```powershell
.\gauge.bat novels --source local            # find the key
.\gauge.bat rate local:fourth --stars 4.5
.\gauge.bat train --target user_rating       # learn what *you* find addictive
```

Five or more rated novels are enough to fit a model; thirty gives you coefficients worth
reading. `user_rating` also works as a column in `library.csv` or `--stars` on import.

Unlabelled novels are never wasted: they are analysed, contribute to corpus percentiles in
`gauge score`, and provide style exemplars (ranked a little below labelled hits).

## Keeping a folder in sync

`gauge watch PATH` (or `import --watch`) registers the folder. Every `gauge ingest` and every
daemon cycle re-scans it:

- unchanged chapters are skipped (content hash),
- edited chapters are replaced and their cached features invalidated,
- new chapter files are added,
- chapters deleted on disk are removed.

So you can keep writing in your editor; the next cycle sees the new chapter, re-analyses, and
the critic scores your newest chapter against everything else it has read.

## Learning your voice

`gauge train` writes `data/models/datasets/sft.jsonl` from your **local** novels only. That is
the file to fine-tune on if you want the model to write like you rather than like the corpus.
See [LEARNING.md](LEARNING.md#fine-tuning).
