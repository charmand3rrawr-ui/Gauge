# Free AI helpers: other models teach GaugeLM and check its work

Several companies let anyone with a free account send a limited number of requests a day to large AI
models. Gauge can link to them, so that GaugeLM learns from more than the models on your computer and
has its writing checked by something that can tell good from bad.

- **Teaching.** A connected helper joins GaugeLM's school (`docs/NATIVE.md`). It takes turns with the
  free models on your computer: writing practice scenes, answering Gauge's writing tasks, correcting
  GaugeLM's own attempts, explaining words it is lost on.
- **Checking.** A helper reads what GaugeLM wrote, marks it out of 10, points at the faults, and
  rewrites the passage correctly. The corrections become lessons.

Nothing here is used until you connect it. These services are **free, but not private and not
unlimited**; the rest of this page is about those two things.

## Setting one up in Gauge Studio

1. Choose a helper from the table below and make a free account and a key on its website.
2. In Studio: **Change the AI**, then **Free helpers...** (or **Teach > Free AI helpers...**).
3. Choose the helper, paste the key in **Its key**, press **Connect helper**. Gauge tries the key by
   asking the provider which models it offers, which is free, and picks one.
4. Read the question that appears and answer it. Only a yes lets Gauge send that helper anything.

From then on it teaches at the start of training sessions and checks GaugeLM's chapters. You can
connect several; they take turns.

| button | what it does |
|---|---|
| Connect helper / Disconnect helper | try and keep a key / delete it and stop using that helper |
| Use this model | pick another of the helper's free models from the list |
| Teaching on / off | may its answers be kept as lessons for GaugeLM |
| Checking on / off | may it check what GaugeLM writes |
| Try it | one short request, to see it answer |
| Check the chapter I am reading | a check now, of the chapter on screen, by whoever can check |
| Remove its lessons | take everything that helper wrote out of what GaugeLM trains on |

## The helpers Gauge knows

What each free plan offers and keeps, **as read in November 2026**. Plans change; each provider's own
pages are the truth, and Gauge shows this summary again in the question it asks you.

| helper | free plan | what it does with what you send | requests a day Gauge sends |
|---|---|---|---|
| **Groq** (`groq`) | free plan, no payment card; open models such as Llama, very fast | says it does not keep it or train on it | 300 |
| **OpenRouter** (`openrouter`) | one key for many companies' models; Gauge uses only those offered at no charge (names ending `:free`), about 50 requests a day on an account that never bought credits | each free model is run by a different company with its own rule; some keep it | 40 |
| **Google Gemini** (`gemini`) | free allowance, set per model | on the free allowance Google uses it, and the answers, to improve its products, and people may read it | 100 |
| **Mistral** (`mistral`) | free "Experiment" plan (asks for a phone number) | may train its models on it unless you switch that off in its privacy settings | 200 |

Keys: [Groq](https://console.groq.com/keys), [OpenRouter](https://openrouter.ai/settings/keys),
[Google](https://aistudio.google.com/apikey), [Mistral](https://console.mistral.ai/api-keys).

**Gemini starts as a checker only.** Google's terms for the Gemini API do not allow using it to
develop models that compete with it. Whether GaugeLM counts is yours to judge, so Gauge leaves its
teaching off to begin with: it marks GaugeLM's work and points at faults, and none of its answers is
kept as a lesson. You can switch its teaching on yourself.

**Any other service** that speaks the same OpenAI-style protocol can be added by its address:
`gauge helpers connect myhost --url https://.../v1`. A key is only ever sent to an `https://` address
(or to your own computer or home network).

Models come and go, so Gauge never trusts a list written into its code: it asks the provider which
models it has, leaves out the ones that are not for conversation (speech, images, embeddings), on
OpenRouter leaves out every model that costs anything, and picks from what remains. Its pick may not
be the best one; **Use this model** lets you choose.

## Staying free

- Gauge counts the requests it sends each helper and stops at the number in the table for the day.
  Change it per helper with `gauge helpers set NAME --daily N`.
- Requests to one helper are spaced a few seconds apart.
- When a helper answers "too many requests", Gauge leaves it alone for 30 minutes and the next
  teacher carries on. When one answers "payment required", Gauge stops using it for a day; nothing is
  paid.
- **Gauge cannot see your account.** A free plan is free because the account has no payment method or
  is on the free tier. If you add a card at a provider, that provider's charges apply to whatever
  Gauge sends. On OpenRouter only the no-charge models are ever requested, whatever the account holds.

## What is sent, and under whose terms

- Each request is a short system text and one message: a passage from text you are allowed to train
  on (your own writing, public-domain books, your Gauge chapters), or something GaugeLM wrote. Nothing
  about you or your computer.
- The system text says plainly what the answer is for: that GaugeLM is a small model being trained,
  and that the answer may be saved as an example for it.
- **Terms are yours to read.** Some providers, and the licences of some of the models they run, do not
  allow answers to be used to train another AI model. Gauge cannot decide that for you; the question
  it asks before switching a helper on links to that provider's terms. Every lesson records which
  helper and model wrote it, so they can be told apart and removed.
- Keys are kept in one file in your user folder (`C:\Users\<you>\.gauge\helper-keys.json`), not in
  Gauge's settings or data. It is a plain text file. A provider's usual environment variable
  (`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`) works too.
- In a dry run (`gauge --dry-run ...`), and with `GAUGE_OFFLINE` set, nothing is sent to any helper.

## Checking GaugeLM's work

GaugeLM cannot tell whether what it wrote is any good. A check takes three passages of a chapter (the
opening, the ending and one between) and has each read by up to two checkers. Each gives:

- marks out of 10 for clarity, grammar, sense (does it hold together) and interest;
- the worst faults, each with the exact words, what is wrong, and how to put it right;
- the passage rewritten with those faults put right.

What Gauge does with it:

- **A fault is reported only if the quoted words are really in the passage.** Checkers invent things
  too; this way every fault shown can be found in the text.
- The marks are averaged and remembered. The trend over many chapters ("6.1 lately, 4.3 at first")
  shows whether GaugeLM's writing is improving. They are a model's opinion, useful as a trend and not
  a measurement.
- The corrected passage becomes a lesson (`checks.jsonl`: "rewrite this so it is clear and correct"),
  but only from a checker whose teaching is on.
- The whole check is saved beside the story, in `checks/chapter_0001.json`.

**Who checks**, in order: helpers switched on for checking, Claude if it is teaching
(`docs/CLAUDE.md`), then the free models on your computer. So checking works with no helper at all,
as long as Ollama is running. GaugeLM never checks itself. A check by Claude is paid for like any other
Claude request and counts toward its daily limit.

**When.** Automatically after each chapter GaugeLM writes with **Write the next chapter**
(`helpers.check_after_chapter`; not during a whole-novel run, whose chapters you can check afterwards), and
whenever you ask: **Story > Check this chapter with another AI**, which works for a chapter written by
any model.

## From the command line

```
.\gauge.bat helpers list                       # every helper: its free plan, and where each stands
.\gauge.bat helpers connect groq               # asks for the key at a hidden prompt, tries it, then asks the question
.\gauge.bat helpers models groq                # the free models it offers, Gauge's guess first
.\gauge.bat helpers set groq --model NAME --daily 100 --teach on --check on
.\gauge.bat helpers try groq                   # one short request
.\gauge.bat helpers check "My Story" --chapter 3
.\gauge.bat helpers disconnect groq --forget   # delete its key; --forget also removes its lessons
```

Settings (`config.json`, under `helpers`): `providers` (what you connected), `daily_requests` (one
number for every helper; 0 = each helper's own), `rest_minutes`, `check_after_chapter`,
`check_pieces`, `check_panel`.

## What has and has not been tested

- Each real provider's address was contacted from the development machine with a made-up key, and each
  refused it and Gauge reported that correctly. OpenRouter's real model list was read and the
  free-model filter picked 12 free models out of 474.
- Teaching, checking, the daily allowance, the pauses and the Studio window's handlers are tested
  against a stand-in server that speaks the same protocol, through Gauge's real HTTP client
  (`tests/test_helpers.py`).
- **No real key was available, so no real model has answered a request from this code.** The first
  lesson you ask for is the first. Free models differ in small ways (some think out loud, some wrap
  JSON in extra words); Gauge handles the ones it knows of, and a reply it cannot read is skipped
  rather than kept.
- The limits in the table come from the providers' pages and from write-ups of them, not from Gauge
  hitting them.
- How much faster GaugeLM improves with these teachers has not been measured. A model as small as
  GaugeLM is limited mostly by its size and its hours of training.
- The Studio window has not been seen on a real screen.
