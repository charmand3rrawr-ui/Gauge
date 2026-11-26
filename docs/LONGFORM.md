# Very long serials: staying consistent to chapter 10,000

No model remembers a long serial. A model sees only what is put in front of it for each chapter, and
what fits is a few thousand words. So consistency over thousands of chapters is not something the
model does. It is something the system around the model has to do: keep everything, and put the
right small part of it in front of the writer each time.

## What there was, and why it could not last

Before the chronicle, a story's memory was:

- one summary of about 250 words, rewritten after every chapter. By chapter 440 a detail from
  chapter 40 had been rewritten away four hundred times;
- the last dozen chapters, for repetition checks;
- a list of facts that kept only the latest 2,000, which at a dozen facts a chapter is about 160
  chapters;
- each character's last 40 memories.

That is a working memory of roughly a hundred and fifty chapters.

## The chronicle

`gauge/write/chronicle.py` keeps a small database beside each story (`chronicle.db`). It forgets
nothing, and everything in it is indexed, so a chapter costs the same at 10,000 as at 10.

| it keeps | so that |
| --- | --- |
| a digest of every chapter, written once and never rewritten | nothing is summarised away |
| roll-ups in layers: every 10 chapters an arc, every 10 arcs a volume, every 10 volumes a saga | "the story so far" is about 36 short pieces however long the story is |
| every fact with its history, each marked as lasting (a name, a parent, a scar, a death, a rule) or changing (where someone is, what they carry) | a lasting fact contradicts a different later value; a changing fact moves on and the old value becomes history |
| who appeared in which chapter | someone returning after thousands of chapters comes with when they were last seen, what happened then, and everything known about them |
| every character memory | an old memory involving someone in the scene can be brought back |

### Before each chapter

The writer and the planner are given:

1. **The story so far, in layers.** For chapter 9,998: the nine finished sagas, the finished volumes
   of the current saga, the finished arcs of the current volume, the chapters of the current arc,
   then where things stand now. Nearer pieces get more words.
2. **From the story's record**, about the people and places in this chapter's plan: the facts in
   force (lasting ones marked "must not change"), who is dead, reminders for anyone away more than
   50 chapters, the earlier chapters most like this one, and any contradiction still unresolved.

Both are sized to the model's reading length.

### After each chapter

The same single call that used to update the running summary now also returns the chapter's own
digest and six to twelve facts, each marked lasting or changing. They are filed. If a lasting fact
contradicts one already on record, from any chapter, it is logged, shown in the Studio's status
line, and put in front of the writer for the next chapters until you rule on it.

### Checking a draft

- **The dead.** Without any model: if someone on record as dead speaks or acts in a draft, and the
  passage is not marked as a memory, a dream or a ghost, the draft is sent back for revision with
  the chapter in which they died.
- **Lasting facts.** With the full writing room, the Archivist draws facts out of the draft and now
  checks them against every fact ever recorded, not only the latest 2,000.

### Your rulings

Stories do change their own facts on purpose: people come back from the dead. The record treats that
as a contradiction until you say it happened.

```powershell
.\gauge.bat chronicle "My Serial"                       # how much is on record; contradictions waiting
.\gauge.bat chronicle "My Serial" --about Rin           # everything on record about someone, with history
.\gauge.bat chronicle "My Serial" --so-far              # the layered story so far, as the writer gets it
.\gauge.bat chronicle "My Serial" --brief "Rin returns to the harbour"   # what the writer would be told
.\gauge.bat chronicle "My Serial" --settle 12           # ruling: the earlier value stands
.\gauge.bat chronicle "My Serial" --settle 12 --keep-new   # ruling: the later value replaces it
.\gauge.bat chronicle "My Serial" --rebuild             # file the chapters of a story begun before this existed
```

`--rebuild --quick` files an older story without a model: names and opening and closing lines, no
facts.

## What was tested, and what was not

A test files 10,000 made-up chapters in which people are away for thousands of chapters at a time,
then checks, at chapter 9,998:

- a scar and a sister established in chapter 3 are in the brief, marked "must not change";
- the returning character comes with "last seen in chapter 9,099, 899 chapters ago";
- the story so far is 34 pieces and fits its word limit;
- a draft that gives her a different scar is caught, citing chapter 3;
- someone dead since chapter 40 still may not speak;
- filing a chapter is no slower at the end than at the start, and the brief takes under half a second.

A second test runs the real writing pipeline for one chapter on a story that already has 9,990:
the time and the size of the request stay in the same range as for chapter 1, and the working file
stays small because the plans of old chapters are put away.

What that does not show:

- **Real prose at that length.** The 10,000 chapters were made up and the model was a stand-in. The
  machinery holds; nobody has read a 10,000-chapter Gauge novel.
- **How well facts are drawn out.** The record is only as good as the model that digests each
  chapter. A capable local model does this well; a small GaugeLM will not, and should not be the
  model doing it.
- **Whether the writer obeys.** The brief only helps a writer that reads its prompt. The downloaded
  model does. A small GaugeLM barely uses context yet (`gauge native understand`), so with it the
  record mostly catches mistakes after the fact rather than preventing them.
- **Meaning.** The checks compare facts as worded. "Grey eyes" against "eyes like slate" is not
  caught; "sister: Jun" against "sister: Ysolde" is.

## Cost

One extra short call every tenth chapter to write an arc summary (and one more at every hundredth
and thousandth). The database for 10,000 chapters in the test was a few megabytes.

## Switch

`write.chronicle` (on by default): keep the record and brief the writer from it.
