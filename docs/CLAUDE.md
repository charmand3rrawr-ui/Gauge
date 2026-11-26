# Claude: a paid teacher for GaugeLM, and an optional writer

Everything else in Gauge runs on your computer, for free, with nothing leaving it. This is the one
exception, and it is off until you set it up: Gauge can use **Claude**, Anthropic's model, through
Anthropic's API with your own key.

- **As GaugeLM's teacher.** Claude leads the school (`docs/NATIVE.md`): it writes the practice scenes,
  answers Gauge's writing tasks about passages from your library, marks and corrects GaugeLM's own
  attempts, and explains the words GaugeLM is lost on. GaugeLM then trains on those answers. A far
  stronger teacher than any model a home computer can run.
- **As the writer.** Claude can do everything the downloaded model does: plan, draft, review, revise.

## Three things to know first

1. **It costs money.** Every request is paid for from your Anthropic account. Gauge writes each
   request's tokens and an estimated cost to a ledger and stops asking Claude once today's estimate
   reaches your daily limit (2 dollars unless you change it).
2. **Your text leaves your computer.** Whatever is in a request is sent to Anthropic: passages from
   your library and GaugeLM's attempts when it teaches; your story's notes and chapters when it writes.
3. **Anthropic's terms are yours to read.** They do not allow using Claude to build a competing
   product or service, including training competing AI models. Whether a small personal model like
   GaugeLM counts is for you to judge from the terms themselves (<https://www.anthropic.com/legal>);
   Gauge cannot decide that for you, and this page is not legal advice. Teaching stays off until you
   answer yes to that question yourself.

Gauge is plain with Claude about what it is doing. Every teaching request begins by saying that the
answers are saved as examples a small model called GaugeLM is trained on. Claude can decline a
request. If it declines one because it is about training another model, Gauge says so, stops asking
Claude for that session, and lets the free teachers carry on. It never rewords a request to get round
a refusal.

## Setting it up in Gauge Studio

1. Make an API key at <https://console.anthropic.com/settings/keys> (you need an Anthropic account
   with some credit on it).
2. In Studio press **Change the AI**, then **Claude...**
3. Paste the key in the **API key** box and press **Connect**. The first time, this installs
   Anthropic's library (a small download). Gauge then checks the key with Anthropic, which is free.
4. Choose a **Model** and a **Daily limit**, and press **Save**.
5. Press **Let Claude teach GaugeLM**, read the question, and answer it.

From then on Claude teaches at the start of every training session. **Have a lesson now** asks for
eight short lessons straight away. Nothing is sent to Claude before step 5.

| button | what it does |
|---|---|
| Connect / Disconnect | keep and check a key / delete the saved key and stop using Claude for anything |
| Save | the model and the daily limit |
| Let Claude teach GaugeLM / Stop Claude teaching | the teaching switch |
| Have a lesson now | eight lessons now; GaugeLM learns from them the next time it trains |
| Write with Claude / Write with the downloaded model | who writes your stories |
| Remove Claude's lessons | take every lesson Claude wrote out of what GaugeLM trains on |

### Where the key is kept

In one file in your user folder: `C:\Users\<you>\.gauge\claude-key.txt` on Windows,
`~/.gauge/claude-key.txt` elsewhere. It is not in Gauge's settings or data, so it is never part of a
backup, an export, or an update, and Gauge never shows it again. It is a plain text file: anyone who
can read your user folder can read it. **Disconnect** deletes it. If you would rather not have Gauge
keep it, leave the box empty and set the `ANTHROPIC_API_KEY` environment variable instead; Gauge
then uses that.

Never paste a key into a chat, an email or a story.

## Which model, and what it costs

| model | dollars per million tokens, in / out | about one lesson | |
|---|---|---|---|
| `claude-opus-5-5` (default) | 4 / 20 | 2 cents | very strong, mid-priced |
| `claude-sonnet-5-5` | 2 / 10 | 1 cent | half the price, a little less strong |
| `claude-haiku-5-5` | 0.10 / 0.50 | well under a cent | by far the cheapest; fine for short lessons |
| `claude-fable-5-2` | 8 / 40 | 4 cents | the strongest, at twice the price of Opus |

"About one lesson" assumes about 600 tokens in and 900 out, thinking included. It is a guess from the
size of the prompts, not a measurement. A training session asks for 4 to 60 lessons depending on how
long it is set to run (about one for every three minutes), so a half-hour session with Opus is around
20 cents. The limit is what actually bounds it.

**Writing costs much more than teaching.** Counted with a stand-in model, one chapter through the
full writing room is about 35 requests carrying about 28,000 tokens of prompts (the lean room: about
23 requests and 21,000). With a real 2,000-word chapter the prompts are larger, because the draft
itself is read by each reviewer, and the answers and Claude's thinking are paid for too. Expect on
the order of a dollar a chapter with Opus, more at higher thinking effort; that figure is an estimate
and has not been measured against the real service. A 500-chapter novel would be hundreds of
dollars. When the daily limit is reached in the middle of a chapter the chapter stops with a message,
and can be written again the next day or after you raise the limit.

The cost figures Gauge shows are **estimates from list prices**. Your Anthropic invoice is the truth.
The ledger is `claude_usage.json` in Gauge's data folder: a line per day with requests, tokens,
estimated dollars (teaching and writing separately) and how many requests Claude declined.

## What Gauge sends, exactly

- Each request is a system text and one message. No files, no tools, nothing about you or your
  computer.
- When teaching, the passages come only from text that is permitted for training: your own writing,
  public-domain books, and your Gauge chapters (the same rule as for every other teacher).
- Claude's thinking is always on for these models; Gauge sets how hard it thinks (`low` when teaching,
  `medium` when writing) and leaves the rest at Anthropic's defaults.
- In a dry run (`gauge --dry-run ...`) no request is sent to Claude at all.
- Refusal fallbacks are on: if Claude declines a request, Anthropic may re-run it on the model it
  recommends for that case, at that model's price. `claude.fallbacks: false` in `config.json` turns
  this off. (Claude Haiku has no such fallback; a declined request simply stays declined.)

## From the command line

```
.\gauge.bat claude status [--check]      # connected? teaching? spent today; the price table
.\gauge.bat claude connect               # asks for the key at a hidden prompt, installs the library, checks the key
.\gauge.bat claude set --model claude-sonnet-5-5 --limit 3
.\gauge.bat claude teach on              # asks the question; `off` stops it
.\gauge.bat claude lesson --rows 8       # a lesson now
.\gauge.bat claude write on              # Claude writes your stories; `off` goes back
.\gauge.bat claude forget                # remove every lesson Claude wrote
.\gauge.bat claude disconnect            # delete the saved key, stop teaching and writing
```

Settings (`config.json`, under `claude`): `model`, `effort`, `teach_effort`, `daily_usd` (0 = no
limit), `max_tokens`, `fallbacks`, `teach`, `terms_ok`. Teaching needs both `teach` and `terms_ok`;
Studio and `gauge claude teach on` set them together when you answer yes.

## Who teaches when

While Claude's teaching is on and today's limit has not been reached, Claude takes every school
session. When the limit is reached (even in the middle of a session) the free models on your computer
carry on, taking turns as before. Every row records who wrote it (`"teacher": "claude-opus-5-5"`) and
under what terms, so Claude's lessons can be told apart and removed.

Removing the lessons takes them out of future training. What GaugeLM already learned from them stays
in its weights; to undo that as well, go back to a checkpoint from before, or train a fresh model.

## What has and has not been tested

- Gauge's calls were run through Anthropic's real Python library (version 2.14) against a stand-in
  server on the test machine: the library accepts the requests as Gauge shapes them, and Gauge reads
  the streamed answer, a refusal and a rejected key correctly (`tests/test_claude_sdk.py`).
- The key handling, the daily limit, the refusal rules, the school rotation and the Studio window's
  handlers are tested with a stand-in for the library (`tests/test_claude.py`).
- **Nothing here has been run against Anthropic's real service**: no key was available where Gauge is
  developed. The first real request you make is the first one anyone has made with this code. If it
  fails, the message says what Anthropic answered.
- How much faster GaugeLM improves with Claude as its teacher has not been measured. A model as small
  as GaugeLM is limited mostly by its size and its hours of training, so expect better lessons, not a
  different class of model.
- The Studio window has not been seen on a real screen.
