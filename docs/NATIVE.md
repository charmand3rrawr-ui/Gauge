# GaugeLM: Gauge's own language model

`gauge/native/` is a complete language model with no outside model code: a byte-level BPE
tokenizer, a Llama-style transformer, hand-written backpropagation, AdamW with warmup and cosine
decay, sampling with a KV cache, distillation from the big local model, and a client that plugs
into the whole writing pipeline. The only dependencies are numpy and, optionally, CuPy for a GPU.
No PyTorch, no Ollama, no downloaded weights.

## What "more powerful" can and cannot mean here

A frontier model is trained on trillions of tokens across tens of thousands of GPUs. Nothing in
this repository, or any repository, turns a desktop into that, and GaugeLM will not reason or
hold a hundred-chapter plot the way an 8B model does, let alone a frontier one. What *is* in
reach is making a small model as strong as a small model can be, and that is what this version
does:

| lever | what it buys |
| --- | --- |
| Llama-style architecture (RMSNorm, rotary positions, SwiGLU, grouped-query attention, no biases) | better loss per parameter and per step than the 2019 GPT recipe; rotary positions generalise to longer context |
| GPU through CuPy (`pip install cupy-cuda12x`, `native.device: auto`) | the same code runs 10-100x faster, so `small` and `medium` sizes become realistic |
| gradient accumulation, dedup, curriculum (prose first, instruction rows late and upweighted) | stabler training and better instruction following from the few rows that teach behaviour |
| **distillation** (`gauge native distill`) | the big local model answers Gauge's real tasks (continue, open, end, title, score, outline, brief, dialogue) on excerpts from your corpus; GaugeLM learns the behaviour, not just the prose |
| decoding: no-repeat n-grams, min-p, repetition penalty, minimum length | far less looping and babble from a small model |
| presets and a benchmark | pick the largest size your machine can actually train in the time you have |

## Sizes

```
gauge native presets
```

| size | layers | width | heads (kv) | context | params | where it trains |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 4 | 256 | 4 (2) | 256 | ~4M | CPU, hours |
| small | 8 | 512 | 8 (4) | 512 | ~27M | CPU in a day; GPU in an hour |
| medium | 12 | 768 | 12 (4) | 1024 | ~88M | GPU |
| large | 24 | 1024 | 16 (8) | 2048 | ~310M | GPU, overnight |

`gauge native bench --size small` measures training and generation tokens per second on your
machine and prints how many hours 2,000 steps would take. Run it before committing to a size.

## Training

```powershell
.\gauge.bat native bench --size small
.\gauge.bat native train --size small --minutes 240          # presets; Ctrl+C keeps the checkpoint
.\gauge.bat native train --resume --steps 1000                # keep going
.\gauge.bat native distill --rows 400                         # needs llm.provider = ollama/openai (the teacher)
.\gauge.bat native train --resume --steps 800                 # instruction rows now upweighted
.\gauge.bat native eval                                       # held-out perplexity + instruction-shape probes
.\gauge.bat native continue --steps 200                       # what the daemon does each cycle
.\gauge.bat native sample "Write the opening of a chapter."
```

The corpus is every ingested chapter (sites, local imports, Gutenberg), every project chapter,
and the instruction rows under `data/models/datasets/` (SFT, the chosen side of self-preference
and author-edit pairs, the distilled teacher answers, and the character school's lessons in building
a character; see [CHARACTER-SCHOOL.md](CHARACTER-SCHOOL.md)), de-duplicated, with instruction rows
repeated `native.chat_weight` times and concentrated in the last third of training.

## The school: free models teach it

A small model learns behaviour far faster from a stronger model than from raw prose. Three things
now happen at the start of every training session in the Studio, and on demand with
`gauge native teach` (`gauge/native/school.py`):

| | what the teacher does | why it helps |
| --- | --- | --- |
| worked answers | answers Gauge's real tasks (continue, open, end, title, score, outline, brief, dialogue) on excerpts of your library | GaugeLM learns the behaviour, not just the prose |
| the tutor | GaugeLM attempts a task itself; the teacher marks it out of 10 and writes the answer it should have given | practice goes where it is weakest: the lower the running mark for a kind of task, the more often it is set |
| practice scenes | writes very short, simple scenes on fresh subjects, harder as GaugeLM trains | clean text at a level a small model can absorb |
| word study | explains the words GaugeLM is most surprised by, in the sentence where your library uses them | teaching goes to the words it understands least |

**More than one teacher.** Every chat model installed in Ollama teaches, taking turns session by
session (one teacher per session, so the engine is not swapping models in and out of memory).
`ollama pull qwen2.5:7b` or `ollama pull phi4` adds a teacher; nothing else needs setting. They run
on your computer and cost nothing. GaugeLM is taught this way even when it is itself the writer.

```powershell
.\gauge.bat native teachers                      # who teaches, and each model's licence
.\gauge.bat native teachers --use "qwen2.5:7b, phi4"   # choose; --use all for everyone installed
.\gauge.bat native teach --rows 60               # a school session now
.\gauge.bat native train --resume                # learn from what was collected
.\gauge.bat native report                        # the tutor's marks per kind of task, at first and now
```

**Licences.** Each row records which teacher wrote it and the licence that model is published under.
Qwen (most sizes), Mistral and Phi are Apache-2.0 or MIT. Llama and Gemma have their own terms: fine
for learning on your own computer, worth reading before you share a model trained with them.
`gauge native teachers` shows this for whatever is installed. It is a guide, not legal advice.

**Teachers that are not on your computer.** Free hosted models can teach too, through any
OpenAI-compatible address, but they are never used unless you list them yourself, because excerpts of
your library are sent to them and because some providers do not allow their answers to be used for
training another model:

```json
"native": {"outside_teachers": [{"name": "my-free-host", "base_url": "https://.../v1", "model": "...",
                                 "api_key_env": "MY_KEY", "terms_ok": true}]}
```

`terms_ok` is your statement that you have read that provider's terms and they allow it. The key is
read from the environment variable you name, not stored in the settings.

**What is and is not measured.** The mechanics are tested with a stand-in teacher. How much faster
GaugeLM improves depends on the teachers and has not been measured here; the report card is a teacher
model's opinion, useful as a trend rather than a measurement. A teacher is only ever shown text that
is permitted for training (your own writing, public-domain books, your Gauge chapters).

### What it understands: word meaning and context

A language model has no dictionary. It learns what a word means from the company the word keeps, and
two things inside it carry that. Each token has a row of numbers (its embedding), and words used in
similar surroundings end up with similar rows: that is as near as it gets to meaning. Attention lets
each position look back over the passage, so the same word can be read differently in different
surroundings: that is context. Neither can be handed to it. Both are learned, and context is learned
late, because guessing from the last two or three words pays off first.

`gauge/native/understand.py` measures both, so it is plain whether it is learning:

```powershell
.\gauge.bat native understand     # how much it uses context, how often it guesses right, its hardest words
.\gauge.bat native alike courier  # the words it treats as most like "courier"
```

The context measure predicts the same next tokens twice, once seeing the passage and once seeing only
the last four tokens, and reports each as the number of next pieces it is hesitating between. On a
small test model:

| training | hesitating between, with the passage | with only four tokens | context makes it surer by |
| --- | --- | --- | --- |
| 20 steps | 824 | 854 | 1% |
| 500 steps | 107 | 114 | 1% |
| 2,000 steps | 43 | 46 | 2% |
| 7,000 steps (a separate run) | 46 | 70 | 10% |

It got far better at predicting long before it started using context: through 2,000 steps it was in
effect looking at the last few tokens only. That is the "it does not understand" you can feel in its
writing, as a number. The same model with attention switched off scores exactly 0%, which is the
check that the measure means what it says.

**What raises it**, in order of effect: more training steps; more text; a longer reading length and a
bigger model (`gauge native grow`), since it cannot use what it cannot see; a larger vocabulary
(`gauge native grow --vocab`), so a common word is one token with one meaning instead of several
pieces.

**Word study.** `hard_words` finds the ordinary words in your library GaugeLM is most surprised by
(names, numbers and words seen once are left out). The school has a teacher explain each one in the
sentence where your library uses it and then use it three other ways; a word is taught once. This aims
the teaching at what it understands least. Its effect on the measures above has not been measured: it
needs a real teacher model, which the build machine does not have.

**Tried and left out: context drills.** Practice passages were built so that the ending depends on a
detail given earlier (who has the key, what colour the door is), with ordinary sentences in between,
on the theory that they would force it to look back. In two trials they made things worse: at 7,000
steps the model trained with them hesitated between 56 next pieces on real text instead of 46, used
context less (5.5% against 10%), and answered the drills themselves barely above chance (15 of 120).
At this size the time is better spent on real prose, so they are not in Gauge.

### Other ways to make it learn faster

In rough order of effect:

1. **Give it more of the machine.** The capacity limit is 20% by default; at 50% it should train roughly
   two and a half times as fast (`gauge compute limit 50`, or Switches in the Studio).
2. **A graphics card.** GaugeLM uses it once its driver library is installed (`gauge compute setup-gpu`, or
   Switches in the Studio); choose which devices with `gauge compute use`. See [COMPUTE.md](COMPUTE.md).
3. **More, cleaner text.** `gauge corpus fetch` adds public-domain books; your own chapters count most.
4. **Train after every chapter** (`native.after_chapter_minutes`), so it never falls behind.
5. **Grow it only when it has stopped improving** at its current size; a bigger model trains slower.
6. **Other computers you own** can join in (`gauge worker`; see [COMPUTE.md](COMPUTE.md)).

## Growing the model

A model's size is fixed when it is created; training only changes the values of its weights. Once a
size has learned what it can, the way to more capacity used to be starting again. `gauge native grow`
instead builds a larger model that gives **exactly the same answers** as the trained one, with the
extra capacity switched off, and training carries on from the same step.

```powershell
.\gauge.bat native grow --plan            # what it is now, and what it would become; changes nothing
.\gauge.bat native grow                   # the next size up (tiny -> small -> medium -> large)
.\gauge.bat native grow --to medium       # or a named size
.\gauge.bat native grow --layers 6        # or only deeper / --width 384 / --context 512
.\gauge.bat native train --resume         # carry on; the new capacity fills in as it trains
.\gauge.bat native grow --undo            # put back the model from before the last growth
```

How it stays the same model (`gauge/native/grow.py`):

| grows | how |
| --- | --- |
| deeper | new layers start with zero output weights, so they add nothing yet; they are spread evenly between the old ones |
| wider | the old model lives in the first dimensions and the new ones start empty; what is written into the stream is scaled up and the norm gains scaled down so the norms see exactly what they saw before |
| more heads | the head size is kept, so a wider model has more heads; old heads keep their weights and still read the same key/value head (copied where the grouping gets tighter); new heads write zero |
| feed-forward | new hidden units have zero output weights |
| context | positions are rotary, so nothing is stored per position; the limit simply moves |

The weights that read *into* the new parts are random, so gradients reach them from the first step.
Before any file is replaced the grown model is run against the old one on the same input and the
growth is refused if they differ; the old checkpoint is kept in `native/before-grow/`.

The head size can also grow, to a whole multiple of what it was (`--width 128 --heads 4` on a model
with heads of 16 gives heads of 32): rotary pair *i* of a small head turns at the same rate as pair
*k x i* of a head *k* times the size, so the old head sits inside the new one exactly. An older
GPT-style checkpoint can grow deeper and read more at once, but not wider, because its norm layers
subtract the mean over every dimension and empty new ones cannot be hidden from that. A bigger model
is also slower. Measured on a 4-core test machine, going from tiny (3.9 million weights) to small
(25.2 million with the same vocabulary) made training about 4.5 times slower per token and writing
about 3 times slower; run `gauge native bench --size small` to measure yours first.

A measured run on a small test model (trained 1,600 steps, then 400 more steps each way):

| | training loss | held-out loss |
| --- | --- | --- |
| grown at step 1,600, then 400 steps | 3.53 | 4.230 |
| not grown, 400 more steps | 3.63 | 4.238 |
| a new model of the larger size, 400 steps from nothing | 4.60 | 4.831 |

The loss was identical before and after the growth (5.63018 both ways in an earlier run), and growing
was far ahead of starting over. On that small corpus the gain over not growing was modest: a bigger
model needs more text as well as more steps to pull ahead.

### Growing the vocabulary

```powershell
.\gauge.bat native grow --vocab            # the next step up: 4096 -> 8192 -> 16384 -> 32000
.\gauge.bat native grow --vocab 6000       # or a number
.\gauge.bat native grow --to small --vocab # shape and vocabulary together
```

The tokenizer learns further merges on the library as it already cuts it, so every existing token keeps
its number and its meaning, and the model gets a new row for each new token, starting as the average of
the two tokens it joins. Text cut the old way scores every old token exactly as before (checked before
anything is replaced). Unlike a bigger shape this is **not free**: text is now cut into fewer, longer
tokens that the model has never seen, so it is worse until it trains on them. Measured on the small
test model (vocabulary 1,200 to 2,400, 17% fewer tokens for the same text), in loss per character so
the two vocabularies can be compared:

| | larger vocabulary | vocabulary left alone |
| --- | --- | --- |
| before | 1.715 | 1.715 |
| straight after growing | 1.902 (11% worse) | |
| after 100 more steps | 1.724 | 1.725 |
| after 400 more steps | 1.641 | 1.683 |

So expect a short dip, level again after a brief session, and ahead after that.

### Shrinking

```powershell
.\gauge.bat native shrink                  # back to the shape before it last grew, else the next size down
.\gauge.bat native shrink --to tiny --minutes 20
```

A smaller model holds less, so shrinking cannot be lossless. What it does is keep as much as it can:
the small model is **cut out of the big one** (growth run backwards, which on a freshly grown model
returns exactly the original weights) and then **taught by it**, training on the library with the big
model's guesses as targets alongside the real next token. Measured on the same test model, loss per
character on text it never trained on:

| | loss per character |
| --- | --- |
| the small model before it grew | 1.716 |
| the big model after 800 more steps | 1.597 |
| cut down, not taught | 1.656 |
| cut down and taught for 300 steps | 1.585 |
| a new small model taught for 300 steps from nothing | 2.182 |

Most of what the big model learned survived the cut, and teaching closed the rest on that corpus. On
a larger library, where the big model has learned things a small one cannot hold, expect a real loss;
the command reports it. A shape the big model could not have been grown from (heads that do not
divide evenly, for example) is not cut but started fresh and taught, which keeps much less.

`gauge native grow --undo` steps back through every change of size in turn: a growth, a larger
vocabulary or a shrink. The model from before each change is kept in `native/before-grow/`.

What still cannot be done: the vocabulary cannot shrink, heads cannot grow by a fraction, an older
GPT-style model cannot widen or shrink, and nothing here makes a model better by itself. Growing adds
room; only training fills it.

## From Gauge Studio

The Studio's **Teach Gauge** window has the same controls without a terminal: **Use GaugeLM** / **Use the
downloaded model** to choose the writer, and **Start training** / **Pause training**. Training runs until
paused or for a set number of minutes, stops within a few seconds of a pause with everything saved, and
carries on from that step next time. When the downloaded model is running, starting training first has
it answer a few of Gauge's tasks (distillation), so the big model teaches the small one while it carries
on doing the writing. **Grow GaugeLM**, **Grow the vocabulary**, **Shrink GaugeLM** and **Undo the last size change** are in the same window. Training respects the capacity limit (`gauge compute limit`). The switch
`native.after_chapter_minutes` trains GaugeLM a little after every chapter. See
[installer/README.md](../installer/README.md).

## Using it as the writer

```powershell
.\gauge.bat init --provider native
.\gauge.bat write "My Serial"
```

Structured answers come from **schema-guided filling**: each field is generated on its own and
typed in code (nested objects and arrays resolved through the schema), so the planner, critic and
writing room never see unparseable output. Quality tracks the model; with `tiny` use the `lean`
agent profile and short chapters.

## Implementation

- `tokenizer.py`: byte-level BPE, GPT-2 style pre-tokenization, special tokens, JSON on disk.
- `backend.py`: numpy or CuPy; the model never imports either directly.
- `model.py`: `arch="llama"` (RMSNorm, RoPE, SwiGLU, GQA) and `arch="gpt"` (legacy checkpoints);
  every backward pass by hand, verified against finite differences for both architectures and
  for grouped heads; KV-cache generation with top-k, top-p, min-p, repetition penalty, no-repeat
  n-grams, minimum length; `AdamW`.
- `train.py`: corpus gathering and dedup, prose/instruction streams, curriculum, gradient
  accumulation, schedule, held-out evaluation, checkpoints, time budgets, `continue_training`.
- `distill.py`: task-shaped prompts over corpus excerpts answered by the teacher model.
- `school.py`: the pool of teachers, the tutor and its marks, graded practice scenes, word study, the report card.
- `understand.py`: how much it uses context, the words it is most lost on, the words it treats as alike.
- `grow.py`: function-preserving growth to a larger shape, vocabulary growth, shrinking by cutting down and
  teaching, each checked or measured before it replaces anything, with undo.
- `evaluate.py`: presets table, benchmark, perplexity, instruction-shape probes.
- `client.py`: the `NativeLLM` provider.
