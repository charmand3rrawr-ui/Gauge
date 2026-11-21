# GaugeLM: Gauge's own language model

`gauge/native/` is a complete language model with no outside model code: a byte-level BPE
tokenizer, a GPT-style transformer, hand-written backpropagation, AdamW with warmup and cosine
decay, sampling with a KV cache, and a client that plugs into the whole writing pipeline. The
only dependency is numpy. No PyTorch, no Ollama, no downloaded weights. It trains on the text
Gauge has read and on your own chapters, and the daemon keeps training it as new chapters
arrive.

## What to expect, honestly

A model you can train on a desktop CPU in hours is small: the default is 4 layers, 256 wide,
about 6 million parameters, roughly a thousand times smaller than the models Ollama runs.
It learns your corpus's vocabulary, rhythm and shape, and it will produce prose that reads like
your sources at a distance; it will not reason, follow long instructions or hold a plot the way
an 8B model does. Think of it as the model that is entirely yours, improving as long as you feed
it, next to the bigger borrowed one. Both are first-class providers; switch with `gauge init`.

Throughput on an ordinary CPU with numpy is a few thousand tokens per second in training and
tens of tokens per second in generation for the default size. A 2,000-step first training on a
few million characters takes an hour or two; the daemon's continue-training cycles are capped by
`native.daemon_minutes`.

## Training

```powershell
.\gauge.bat native train                         # defaults from config (native.*)
.\gauge.bat native train --layers 6 --embd 384 --steps 5000 --minutes 180
.\gauge.bat native train --resume --steps 1000   # keep going from the checkpoint
.\gauge.bat native continue --steps 200          # what the daemon does every cycle
.\gauge.bat native status
.\gauge.bat native sample "Write the opening of a chapter about a courier in the rain."
.\gauge.bat native tokenizer "The courier ran."
```

The corpus is: every ingested chapter (sites, local imports, Gutenberg), every chapter of every
project, and the instruction rows under `data/models/datasets/` (SFT rows, the chosen side of
self-preference and author-edit pairs) in a prompt/response format with special tokens, so the
model learns to answer a prompt and not only to continue text. Checkpoints are written every
`checkpoint_every` steps and on Ctrl+C; `--minutes` stops on a budget.

## Using it as the writer

```powershell
.\gauge.bat init --provider native
.\gauge.bat doctor
.\gauge.bat write "My Serial"
```

Structured answers (bibles, outlines, scores, character sheets) come from **schema-guided
filling**: each field in the schema is generated on its own with a short budget and typed in code
(strings, lists split on separators, numbers parsed or defaulted, nested objects recursed). It is
slower than one call but never yields unparseable output, so the planner, critic and the writing
room keep working. Quality tracks the model: with the small default you will want the `lean`
agent profile and short chapters while it learns.

## The implementation

- `tokenizer.py`: GPT-2 style pre-tokenization, byte-level merges learned on your text, five
  special tokens (`<pad> <bos> <eos> <prompt> <response>`), JSON on disk.
- `model.py`: pre-norm blocks (LayerNorm, causal multi-head attention, GELU MLP), learned
  positions, tied embeddings; `forward` caches activations and `backward` computes every
  gradient by hand (verified against finite differences in the tests to 1e-4 relative error);
  `generate` with temperature, top-k, top-p, repetition penalty, stop tokens, minimum length, and
  a KV cache that re-primes when the context fills; `AdamW` with gradient clipping and decoupled
  weight decay.
- `train.py`: corpus gathering, encoding, random-window batches, LR schedule, evaluation on a
  held-out tail, checkpointing, `continue_training`.
- `client.py`: the `NativeLLM` provider with schema-guided structured output and call telemetry.

Sizes are in `config.json` under `native`; `gauge native train` flags override them.
