# How Gauge learns

## The question

Web serials live or die on one reader decision, repeated thousands of times: *click Next, or
close the tab.* Gauge treats "addictive" as a measurable property of text and schedule, then
lets data, not taste, decide which properties matter.

## Features (what is measured)

Per chapter (`gauge/features.py`):

| Group | Features | Intuition |
| --- | --- | --- |
| Hook | `hook`, first-sentence length, opens on dialogue, early question, hook verb, opening tension | Does the first paragraph create a question? |
| Cliffhanger | `cliff`, short last paragraph, punchy last line, ends on question/dialogue/ellipsis, cliff lexicon, reveal lexicon, final tension, proper-noun ratio in the last paragraph | Does the ending refuse to resolve? |
| Momentum | sentence length mean/std, short-sentence ratio, paragraph length mean/std, one-line-paragraph ratio, dialogue ratio, exclamations, readability | Does the prose move? |
| Tension curve | mean, peak, peak position, rising (second half minus first half), final | Where is the big beat? |
| Payoff | reveal, progression, warmth, tension densities per 1000 words | What reward does the reader get? |
| Voice | first/third person ratio, type-token ratio, long-word ratio, adverb ratio | Register and texture |
| Title | word count, numbered-only, has subtitle, question | Packaging |

Per novel (`novel_features`): mean and std of the above across chapters, `early_hook` /
`early_cliff` / `early_words` for chapters 1-3, chapter count, **chapters per week** and
release irregularity (from chapter dates), blurb length / question / progression / tension
signals, title length, tag count, genre flags.

Everything is counted with small, visible word lists in `gauge/lexicon.py`. They are meant to
be edited. If your genre has its own tension vocabulary, add it.

## Targets (what "addictive" means numerically)

`log_followers` (default), `log_views`, `log_avg_views`, `rating`, `log_favorites`, and
`user_rating`: your own 1-5 verdict on local novels (`gauge rate`), so the model can learn your
taste instead of the crowd's. See [LOCAL_LIBRARY.md](LOCAL_LIBRARY.md).
Log scales because engagement is heavy-tailed. Switch with `gauge train --target log_avg_views`.
Average views per chapter is the closest public proxy to retention; followers to intent to
return; rating to satisfaction. They disagree in interesting ways, so train on more than one.

## Model

Ridge regression on standardised novel features. Reported with training R2 and 5-fold
cross-validated R2. If CV R2 is near zero, the model has learned nothing you should trust, and
the report will say so. With scikit-learn installed and >= 40 labelled novels, a gradient
boosted regressor is blended in for *scoring* drafts (it catches interactions such as "short
chapters are fine only when cadence is high"); the ridge coefficients remain the explanation.

Confounders you should keep in mind: popular novels are older (more chapters, more time to
accumulate followers), genre strongly shapes both features and audience, and listing position
is self-reinforcing. `log_chapters` and genre flags are in the feature set to absorb some of
this, and `gauge report --novel KEY` shows the per-feature contributions so you can sanity-check.

## Exemplars

The first and last 180 words of chapters from high-engagement novels, ranked by
`engagement x (0.5 + local hook/cliff score)`, at most three per novel per kind, embedded with
`nomic-embed-text` (or a hashing embedder offline). The drafter retrieves the few most similar
to the current chapter's plan and shows them as *shape* references with an explicit instruction
to match energy, never content.

## Style guide and Modelfile

`data/models/style_guide.md` is regenerated each cycle: the top positive and negative
coefficients, translated to writing advice, with median values among the top 20% of novels as
targets ("Top novels average 2,100 words per chapter", "aim for about 0.52 cliffhanger score").
The compact bullet version becomes the system prompt for the writer and is baked into
`data/models/Modelfile` so `ollama create gauge-writer -f data/models/Modelfile` gives you a
model that carries the lessons without any prompt engineering.

## Fine-tuning

`gauge train` exports to `data/models/datasets/`:

- `preference.jsonl`: `{prompt, chosen, rejected}` pairs. `chosen` is a short opening or ending
  from a top-third novel, `rejected` a matched passage from a bottom-third novel. Use with DPO /
  ORPO trainers.
- `sft.jsonl`: `{instruction, input, output}` rows built from your **local** imports only by
  default (your own chapters), so you fine-tune on what you own.

Recommended Windows route: [Unsloth](https://github.com/unslothai/unsloth) under WSL2 with
an NVIDIA GPU, LoRA on a Llama 3.1 8B or Qwen 2.5 7B base, then export to GGUF and
`ollama create`. LLaMA-Factory has a GUI if you prefer one. Gauge stays agnostic; it just
produces clean data and consumes whatever Ollama model you point it at.

## The continuous part

`gauge daemon` repeats: ingest new chapters -> recompute features for new chapters only ->
refit the model -> rebuild exemplars and the style guide -> export datasets -> optionally write
the next chapter of your active project with the fresh guide. `metric_history` records every
observation of every novel's metrics, so growth-based targets (followers gained per chapter
released) are a short addition when you want them. See `docs/ROADMAP.md`.
