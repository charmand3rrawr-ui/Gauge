# Installing Gauge the easy way

## Windows: one file

1. Download **Get Gauge.bat** (it is in this folder).
2. Double-click it.
3. Wait. The first time takes 10-20 minutes because it downloads the AI models (a few GB).
4. A window called **Gauge Studio** opens, and a **Gauge Studio** shortcut appears on your Desktop.

That is all. No git, no accounts, no API keys, nothing leaves your computer. Gauge is put in
`C:\Users\<you>\Gauge`. Already have a copy of Gauge unzipped somewhere? Double-click
**Install Gauge.bat** in it instead; it does the same thing without the download.

## macOS and Linux: one line

```
curl -fsSL https://raw.githubusercontent.com/charmand3rrawr-ui/Gauge/ccr-24dc418f-hsneyx/install.sh | bash
```

Gauge is put in `~/Gauge`, and a `gauge` command is added to `~/.local/bin`. From a copy of Gauge you
already have, run `./install.sh` instead. On Linux it may ask for your password to install Python or
Ollama through the system package manager; on macOS it uses Homebrew when something is missing.

## What the installer does

1. **Looks at your computer** and picks a writer model that fits:

   | your computer | writer model |
   | --- | --- |
   | NVIDIA graphics card with 12 GB or more | `qwen2.5:14b` |
   | NVIDIA graphics card with 6-11 GB | `llama3.1:8b` |
   | no such card, 16 GB of memory or more (32 GB on Apple Silicon gets `qwen2.5:14b`) | `llama3.1:8b` |
   | less than that | `llama3.2:3b` |

   It also checks free disk space, and steps down to the small model rather than run the disk out.
2. Installs **Python** (per-user, no admin) if you do not have it.
3. Installs **Ollama**, the program that runs the AI on your own computer.
4. Installs **Gauge** into its own private Python environment.
5. Downloads the models: the writer, a small helper (`llama3.2:3b`) for the bookkeeping jobs, and an
   embedding model for memory.
6. Writes Gauge's settings.
7. **Asks the AI a real test question**, so you know it works rather than merely that it installed, and
   prints a checklist. Anything that failed is listed with the one command that fixes it.

Running it again is safe: it keeps your stories and your settings, leaves alone any model that is
already on the computer, and only changes existing settings if you name a model or engine yourself.
Everything it did on Windows is logged in `installer\install.log`.

## Updating

To get a newer Gauge, use the updater, not the installer.

- **Windows:** double-click **Update Gauge.bat** (in this folder; it also works from anywhere, such as
  your Desktop).
- **macOS and Linux:** `~/Gauge/install.sh --update`

The updater first asks GitHub one small question: is there a newer version? If not, it says so and
downloads nothing. If there is, it fetches only Gauge's own program (about 1 MB), puts it in place and
lists what is new. Python packages are reinstalled only when the new version needs different ones.

It never reinstalls Python or Ollama, never downloads an AI model, and never touches your stories, your
library or your settings. It cannot: none of that machinery is in it. If Gauge Studio is open, close it
and open it again afterwards. It refuses to run on a git checkout, where `git pull` is the right tool.

## Using Gauge Studio

Type what your story is about in the big box. Press **START MY STORY**. Then press
**WRITE THE NEXT CHAPTER** whenever you want more. Click a chapter on the left to read it.
**Talk to a character** lets you ask the people in your story questions, and they answer as
themselves. **Save story as one file** makes one document you can print or share. **Switches** opens
every setting as checkboxes and sliders, with a warning on the ones that matter.

Every button is also in the menu bar at the top of the window (**Story**, **Teach**, **Settings**,
**Help**). The window sizes itself to your screen, and when it is short it is the chapter list that
shrinks, never a button. The top of the window says which AI is writing and how much of the computer
Gauge may use; the title bar shows the installed version.

## How long a chapter is

Above **WRITE THE NEXT CHAPTER** there is a box, **Words in the next chapter**. Type a number (for
example `1500`, or `2k`) and press the button. The story remembers the number for the chapters after,
and each story keeps its own.

A model tends to stop when it feels finished, often well short of what was asked. Gauge now checks
the length and, if the chapter is short, has the model carry on from its own last lines until it is
within about a tenth of the number you typed. Chapters can be from 100 to 20,000 words; long ones
take proportionally longer. The switch **Write chapters to the asked length** turns the carrying-on
off.

## Writing a whole novel in one go

Under **WRITE THE NEXT CHAPTER** there is **Write the whole novel...**. Type how many chapters the
novel should have (for example 500) and the shortest a chapter may be (for example 2000 words), and
press **Start (or carry on)**.

- Gauge first plans the whole novel in volumes, so the story builds toward an ending and reaches it
  at the last chapter.
- Then it writes chapter after chapter without being asked. The line under the buttons shows which
  chapter it is on and roughly how many hours are left.
- No chapter will be shorter than the number you gave.
- **Stop after this chapter** stops it cleanly; nothing is left half-written. **Start** carries on
  from the next chapter, today or next week. You can close Gauge in between.

This takes a long time. A chapter of 2,000 words can take many minutes on a home computer, so 500
chapters is days of running. Raising the capacity limit in **Switches** makes it faster.

## Staying consistent in a very long story

No AI remembers a long story; it only sees what is put in front of it for each chapter. Gauge now
keeps a permanent record of every story: a short digest of every chapter, every fact that was
established (who is whose sister, who has died, what the rules are), and who appeared where. Before
each chapter it gives the writer the layered story so far and everything on record about the people
in that chapter, including anyone who has not appeared for a long time. After each chapter it checks
what was written against the record.

If a chapter contradicts something established earlier, however long ago, the line under the buttons
says so. A character who has died cannot speak or act again unless the passage is a memory, a dream
or a ghost; such a draft is sent back to be rewritten.

For a story you began before this update, file its chapters once from a Command Prompt in the Gauge
folder: `gauge.bat chronicle "Your Story" --rebuild`. More in
[docs/LONGFORM.md](../docs/LONGFORM.md).

## Several attempts, the best one kept

A small model's writing varies a lot from one attempt to the next: one attempt invents people your
story has never heard of, another repeats itself. With GaugeLM as the writer, Gauge now writes each
piece of a chapter four times and keeps the attempt that stays closest to your story: fewest
invented names, least repetition, nearest the length you asked for. It takes four times as long to
write, which GaugeLM can afford. The switch **Attempts to choose between** changes the number (1 to
8); with the downloaded model it stays at 1 unless you raise it, because that model is slow and
steadier.

## Asking Gauge, and saying what happens next

At the bottom of the right-hand side there is a box to type in, with two buttons.

- **Ask Gauge** (or press Enter) talks to the AI about your story. With a story open it knows the
  world and characters, what has happened so far, what is still unresolved and how the last chapter
  ended. Ask for ideas, what a character would do, or what a reader is waiting for. Asking never
  changes the story.
- **Use for the next chapter** takes what you typed as an instruction, for example *"Jun finds the bag
  and opens it before Rin can stop her"*. The next chapter is planned and written around it, and it
  outranks everything else Gauge had in mind. It is used once, for that chapter only. The line under
  the box shows what is waiting; press the button with the box empty to take it back.

## Teaching Gauge with your own material

Press **Teach Gauge** to give it books or your own writing to learn from.

1. *(Optional)* choose how good the material is, 1 to 5 stars. Ratings teach Gauge your taste.
2. Press **Add files...** (each file is one book: `.txt .md .docx .epub .html`) or **Add a folder...**
   (a folder of chapter files is one book; a folder of books is a whole library).
3. Press **Learn from the library now**. This can take several minutes.

What learning does: Gauge studies the library, keeps the strongest passages as examples for the writer
to imitate, and learns craft techniques from the chapters. Both are used every time it writes. With 30
or more chapters, and star ratings or reader numbers on several books, it also builds a model of what
makes a chapter work.

What it does not do: it does not rebuild the downloaded AI model itself. That model stays as it is and
is steered by what was learned. (Only GaugeLM, the built-in model, is trained on your text directly;
see [docs/NATIVE.md](../docs/NATIVE.md).)

Use only writing that is yours or that you have the right to use, such as your own work or
public-domain books. Everything stays on your computer. Adding the same book again updates it rather
than duplicating it.

The same thing from the command line: `gauge.bat import "C:\path\to\folder"`, then `gauge.bat train`
and `gauge.bat absorb`.

## Letting Gauge find novels by itself

In **Teach Gauge**, press **Find novels online**. Gauge goes to Project Gutenberg, a library of books
that are out of copyright, looks for novels in the genres of your stories, and adds a few it does not
already have. They appear in the library list; press **Learn from the library now** to study them.

- Tick **Keep finding novels by itself** and it does this about once a day, when a training session
  starts.
- It only ever talks to Project Gutenberg and its catalogue. It cannot visit other websites, and it
  cannot fetch novels that are still in copyright.
- Nothing about you or your writing is sent. A search is a word such as "adventure".
- It adds three novels at a time and stops at 300 MB in total; both can be changed in **Switches**.

## Teaching Gauge how to build characters

In the **Teach Gauge** window press **Characters...** (or use the **Teach** menu, **Teach character
building...**).

1. Press **What a character needs** to read the breakdown: nine sections covering who the character
   is, how they look and move, their past, what drives them, their temperament, their voice, what
   they can do, where they stand when the story starts, and how they connect and change.
2. Press **Save a blank sheet...**, open the file in Notepad, and fill it in. Every line has a note
   saying what belongs there and the test a good answer passes.
3. Press **Add character sheets...** and choose your filled-in sheets. Gauge checks each one and tells
   you what is still missing; a complete sheet is kept as an example.
4. Or press **Learn from this story's characters** to keep the main characters of the open story.

From then on, every time Gauge creates a character it is shown your closest examples and the
checklist, and it fills in anything it left out. GaugeLM also studies the examples when it trains.
More in [docs/CHARACTER-SCHOOL.md](../docs/CHARACTER-SCHOOL.md).

## Which AI writes: the downloaded model or GaugeLM

Gauge can write with either of two models, and the Studio's top line always says which one is in use
("Story engine ready: ...").

- **The downloaded model** (for example `llama3.1:8b`, run by Ollama). Ready as soon as it is installed.
  This is what writes good chapters today.
- **GaugeLM**, Gauge's own model. It is very small and ships untrained. It knows only what it has been
  trained on, so it must be trained before it can write at all, and for a long time its text is far
  rougher than the downloaded model's. After a minute or two of training it produces nonsense; it needs
  many hours and a good-sized library before the text becomes readable.

Press **Change the AI** (or **Teach > Change the AI that writes...** in the menu):

- **Use GaugeLM** / **Use the downloaded model** switches the writer. Switching to GaugeLM also selects
  the lean writing room and shorter chapters, which a model that small needs; switching back restores
  what you had.
- **Start training** trains GaugeLM, carrying on from wherever it stopped. Choose **until I pause** or a
  number of minutes. It works whichever model is writing.
- **Pause training** stops it after the step it is on, usually within a few seconds, with everything
  saved. Press **Start training** again and it carries on from that step. Both are also in the
  **Teach** menu, so training can be paused from the main window.

While training runs, the other buttons wait; pause it when you want to write.

### How GaugeLM learns while the downloaded model does the writing

Using the downloaded model does not hold GaugeLM back: it is GaugeLM's teacher. GaugeLM learns from

1. your library;
2. every chapter written for your stories, whichever model wrote it;
3. worked examples: each time training starts while the downloaded model is running, that model first
   answers a few of Gauge's own tasks, and GaugeLM learns to imitate the answers.

It only learns when training actually runs. Press **Start training** now and then, or turn on the switch
**Train GaugeLM after each chapter (minutes)** in **Switches** so it trains a little every time a chapter
is written. A sensible path: write with the downloaded model, train GaugeLM alongside, and switch the
writer to GaugeLM once its text reads well enough for you.

Only text that is safe to train on is used: what you added yourself, public-domain books, the chapters
written for you, and the examples made on your own machine.

### How big GaugeLM can be

GaugeLM's sizes now go all the way up to twenty trillion weights (`20t`), and **Grow GaugeLM** steps
from each size to the next. Before growing, Gauge checks that the bigger model fits in your
computer's memory and tells you the numbers if it does not; a home computer stops at around `large`.
The sizes above that are there for a machine that could hold them. Nothing about Gauge's code is the
limit; the computer is.

### A stronger writer, and more thinking

- **Find a stronger model** (in **Change the AI**) checks whether your computer could run a better
  writing model than the one it has, and offers to download it and switch. A bigger model writes
  better and is slower, and the download is large.
- **Thinking effort** (in **Switches**) sets how hard Gauge works on each chapter: *quick*, *normal*,
  *deep* or *maximum*. More effort means more attempts and more rounds of revision, and takes
  proportionally longer (deep about two and a half times, maximum about five).

These get the most out of your computer. They do not make a home computer into one of the giant AI
systems; nothing can.

### Free AI helpers (optional)

Some companies let anyone with a free account send a limited number of requests a day to large AI
models. Gauge can link to them so they **teach GaugeLM** and **check what it writes**.

1. Make a free account and a key at one of them: Groq (console.groq.com), OpenRouter (openrouter.ai),
   Google Gemini (aistudio.google.com) or Mistral (console.mistral.ai).
2. In Studio: **Change the AI**, then **Free helpers...**
3. Choose the helper, paste the key, press **Connect helper**, and answer the question it asks.

They are free but not private: passages from your library and GaugeLM's writing are sent to them, and
some free plans keep what you send. Gauge sends each one only a set number of requests a day and
leaves it alone when it says it has had too many. After every chapter GaugeLM writes, a checker marks
it and points at the faults; **Story > Check this chapter with another AI** does the same for any
chapter. Full details: [docs/HELPERS.md](../docs/HELPERS.md).

### Claude as a teacher (optional, paid)

GaugeLM can also be taught by **Claude**, Anthropic's model, which is far stronger than anything your
computer can run. This is the one part of Gauge that is not free and not private: you need your own
Anthropic API key, every request is paid for, and passages from your library are sent to Anthropic.

1. Make a key at console.anthropic.com (Settings, API keys).
2. In Studio: **Change the AI**, then **Claude...**
3. Paste the key, press **Connect**, choose a model and a daily limit, press **Save**.
4. Press **Let Claude teach GaugeLM** and read the question before you answer it. It covers the cost,
   what is sent, and Anthropic's terms about training other AI models, which are yours to judge.

Gauge stops asking Claude once the day's limit is reached (2 dollars unless you change it), and the
free teachers carry on. **Remove Claude's lessons** takes out everything it wrote; **Disconnect**
deletes the saved key. The same window can make Claude the writer of your stories, which costs much
more: roughly a dollar a chapter. Full details: [docs/CLAUDE.md](../docs/CLAUDE.md).

### Who teaches GaugeLM

At the start of each training session the free models on your computer teach GaugeLM before it studies
on its own. **Change the AI** says who is teaching and how it is doing.

- Every chat model installed in Ollama takes a turn. To add a teacher, open a Command Prompt and run
  `ollama pull qwen2.5:7b` (or `phi4`, `mistral`); Gauge finds it by itself.
- A teacher answers Gauge's writing tasks on your library, marks and corrects GaugeLM's own attempts,
  and writes short, simple practice scenes for it.
- GaugeLM practises most what it is marked lowest on. The marks out of 10 are a teacher's opinion:
  watch whether they rise, not the exact number.
- Nothing leaves your computer. Teachers on the internet are possible but are off unless you add
  them to the settings yourself; see [docs/NATIVE.md](../docs/NATIVE.md).

To make it learn faster, the biggest step is the capacity limit in **Switches**: at 50% instead of
20% it should train roughly two and a half times as fast.

### The starter lessons

Gauge comes with a small set of ready-made lessons for GaugeLM: 36 very short scenes, worked
examples of each writing task, and ten words explained. They give it well-formed examples from its
first training session, before any teacher model has run. They are a seed, not a library; most of
what it learns still comes from your books and your teacher models.

They are **off until you switch them on** (**Switches**, **Use the starter lessons**). They were
written for this project by Claude, an AI model made by Anthropic, and Anthropic's terms limit using
Claude's output to train AI models that compete with its services. Whether your own small story
model is within those terms is for you to decide, which is why Gauge does not decide it for you.

### How well GaugeLM understands

A language model has no dictionary. It learns what a word means from the words around it, over many
passages, and it learns to use context (what came earlier in the passage) only gradually. In
**Change the AI**, press **How well does it understand?** to measure it:

- how many next word-pieces it is hesitating between when it can see the passage, compared with
  when it can only see the last four pieces. If the two numbers are close, it is barely using
  context yet;
- how often its first guess for the next piece is right;
- the words in your library it is most lost on.

The teachers now explain exactly those words, in the sentence where your library uses them, at the
start of training sessions. What raises understanding, in order: more training, more text, a longer
reading length and a larger model (**Grow GaugeLM**), and a larger vocabulary so common words are one
piece each (**Grow the vocabulary**).

### Growing and shrinking GaugeLM

GaugeLM's size is fixed when it is first trained, and more training only improves it up to what that
size can hold. In **Change the AI**, a line says how big it is now and what it could become, with four
buttons. Pause training before using any of them.

- **Grow GaugeLM** moves it to the next size. It keeps everything it has learned: Gauge checks that
  the bigger model gives the same answers as the smaller one before it replaces anything. The extra
  room is empty until it trains, so press **Start training** afterwards. A bigger model is slower; on
  a test machine, going from tiny to small made training about 4.5 times slower and writing about 3
  times slower.
- **Grow the vocabulary** teaches it more word-pieces from your library, so the same text takes fewer
  pieces and it sees further back. It keeps every piece it knows, but the new ones are untrained, so
  it gets a little worse until it has had a short training session.
- **Shrink GaugeLM** makes it smaller and faster again. The smaller model is cut out of the bigger one
  and then taught by it for about ten minutes (**Pause training** ends that early). A smaller model
  holds less, so some of what it learned is lost; Gauge says how much.
- **Undo the last size change** puts back the model from before the last of these. Training done
  since then is lost. Press it again to step back further.

## How much of the computer Gauge uses

Gauge uses at most **20%** of each device: this computer, and any other computers you have linked to it.
The Studio shows the limit under the status line, and **Switches > Capacity limit** changes it (100
removes it).

| what | how it is held to the limit |
| --- | --- |
| the processor, for Gauge itself | Gauge pins itself to that share of the cores and lowers its priority. It cannot use a core it is not pinned to. |
| the AI engine (Ollama) on the processor | each request tells it to use only that many threads, and requests go one at a time |
| the graphics card | a card cannot be made to run at a fifth of its speed, so Gauge rests between requests: after one that took 10 seconds it waits 40, so the card is busy a fifth of the time |
| GaugeLM on a graphics card | its memory is capped at the share |
| linked computers | each applies the same to itself, and this computer also spaces out what it sends them |

Two things to know. It makes everything proportionally slower: at 20%, a chapter or a training session
takes roughly three to five times as long. And it cannot shrink a model: a loaded model needs its full
size in memory whatever the limit says. On a 16 GB computer, 20% is about 3 GB, and `llama3.1:8b` needs
about 5.5 GB. `gauge.bat compute limit` shows the numbers for your computer, including whether your
model fits inside the share; `llama3.2:3b` does on most machines.

## Which devices Gauge uses (processor and graphics cards)

Open **Switches**. At the top is a **Devices** section with a switch for each device Gauge found on
your computer: the processor and each graphics card. Switch on the ones Gauge may use; more than one
can be on. **Automatic** lets Gauge pick.

- The downloaded model already uses your graphics card by itself.
- Gauge's own model, GaugeLM, needs one extra library before it can use an NVIDIA graphics card. If
  the card shows greyed out, press **Set up the graphics card** (a large download, once). The
  installer and **Update Gauge.bat** now do this for you when they find an NVIDIA card.
- With two graphics cards switched on, they share GaugeLM's training.
- With only the processor switched on, the downloaded model is kept off the graphics card too.
- After changing which graphics cards are on, close and reopen Gauge Studio.

This was built on a computer without a graphics card, so the graphics-card paths are tested with
stand-ins and have not run on a real one. If training does not speed up after setup, tell me what
**Switches** shows in the Devices section.

## Choosing for yourself

| you want | Windows | macOS / Linux |
| --- | --- | --- |
| a specific writer model | `set GAUGE_MODEL=qwen2.5:14b` then run **Get Gauge.bat** | `./install.sh --model qwen2.5:14b` |
| a different folder | `set GAUGE_DIR=D:\Gauge` then run it | `./install.sh --dir /path` |
| no Ollama: GaugeLM, the built-in model | `set GAUGE_ENGINE=native` then run it | `./install.sh --engine native` |
| Gauge only, no AI engine or models | (not offered) | `./install.sh --skip-engine` |

From an unzipped copy on Windows the same choices are flags:
`powershell -ExecutionPolicy Bypass -File installer\bootstrap.ps1 -Model llama3.2:3b` (also `-FastModel`,
`-Engine native`, `-NoLaunch`, `-NoShortcut`).

GaugeLM ships untrained. If you choose it, the installer finishes by telling you the two commands that
fetch a public-domain corpus and train it; see [docs/NATIVE.md](../docs/NATIVE.md) for what a model that
small can and cannot do.

## If something goes wrong

- **Blue "Windows protected your PC" screen** when double-clicking the .bat: click *More info*, then
  *Run anyway*. Windows shows this for any script downloaded from the internet. The scripts are plain
  text; you can open them in Notepad and read them first.
- **Windows asks "Do you want to allow this app to make changes?"** That is the Ollama or Python
  installer. Say yes.
- **The checklist shows `[FAIL] The AI answers`.** Open the Ollama app (the llama icon in the Start
  menu), then run the installer again. If the computer is short on memory, pick the small model:
  `set GAUGE_MODEL=llama3.2:3b` and run it again.
- **"Story engine is OFF"** in Gauge Studio: open the Ollama app, then open Gauge Studio again.
- **A model did not download.** The checklist names it. Run `ollama pull <that model>` in a terminal,
  or just run the installer again; it only fetches what is missing.
- **Linux: `[ -- ] Gauge Studio window (optional)`.** The window needs your system's `python3-tk`
  package. Everything on the command line works without it.

## For the curious

`gauge.bat` (Windows) and the `gauge` command (macOS/Linux) run the command line; everything in
`docs/` applies. `gauge gui` opens the Studio from a terminal, and `gauge doctor` re-checks an install.
