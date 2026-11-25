# Architecture

Gauge is a single Python package with five layers and one SQLite file. Each layer can be used
on its own; the CLI and the daemon just wire them together.

```
config.py        GaugeConfig (JSON on disk)                data/config.json
db.py            Database (sqlite3, stdlib)                data/gauge.sqlite3
models.py        NovelRecord, ChapterRecord                 (pydantic)

ingest/          Source protocol -> RoyalRoad, ScribbleHub, local importer
                 Fetcher: UA, delay, robots.txt, retries
                 IngestRunner: follows -> novels -> only-new chapters

features.py      chapter_features(ChapterRecord) -> {name: float}
lexicon.py       novel_features(NovelRecord, [chapter feats], dates) -> {name: float}

learn/           EngagementModel (ridge + optional GBM), ExemplarLibrary (embeddings),
                 distil_style_guide, dataset exports, Trainer.cycle()

llm/             LLM protocol: OllamaLLM, OpenAICompatLLM, FakeLLM
                 Embedder: OllamaEmbedder, HashingEmbedder (offline fallback)

write/           Project (bible, outline, memory on disk) -> Planner -> Drafter -> Critic
                 characters.py: PersonalityMatrix, Voice, Relationship, CharacterState, Character,
                 Roster (who is in the scene), CharacterAgent (brief / review / remember / chat),
                 CharacterCreator
                 WritePipeline.write_next(): plan, activate cast, draft, critique + in-character
                 review, revise, save, update story memory and each character's memory

daemon.py        once(): ingest.run -> trainer.cycle -> (write_next)   run_forever(): loop
report.py        summaries, leaderboard, explain_novel, compare_to_corpus
cli.py           click commands
```

## Data flow

1. `IngestRunner.run()` walks the `follows` table. For each URL the matching `Source` fetches
   the novel page (metrics + chapter list), compares chapter ids against `chapters`, and fetches
   only the missing ones. Every novel fetch appends a row to `metric_history`, so follower
   growth over time is available for future targets.
2. `Trainer.analyze()` computes `chapter_features` for chapters without a `features` row and
   recomputes `novel_features` for every novel (cheap; aggregates plus cadence and blurb).
3. `Trainer.train_model()` builds a table of novel features vs. a target (`log_followers` by
   default), standardises, fits ridge, reports train and 5-fold R2. If scikit-learn is present
   and n >= 40 a GradientBoostingRegressor is blended in for scoring; ridge coefficients remain
   the explanation.
4. `Trainer.build_exemplars()` takes the first and last 180 words of every chapter, ranks them
   by `novel engagement x (0.5 + local hook/cliff score)`, keeps a diverse top-N, embeds them.
5. `Trainer.distil()` writes `style_guide.md` (top positive/negative coefficients translated to
   writing advice, with median values among top-quintile novels as concrete targets) and an
   Ollama `Modelfile` with the compact guide as system prompt.
6. `WritePipeline.write_next()`:
   - `Planner.ensure_plan(n)` extends the outline (5 chapters at a time) when needed, using
     retrieved *ending* exemplars as shape references.
   - `Roster.present_in()` finds the characters named in the plan and the previous chapter's
     tail; each `CharacterAgent.brief()` answers as that person (goal, mood, actions, lines,
     refusals, friction with the plan). The briefs become the CAST section of the draft prompt.
   - `Drafter.draft()` writes the chapter with the bible, running summary, previous chapter
     tail, the plan, and retrieved opening/ending exemplars.
   - `Critic.critique()` first measures coherence (`prose.coherence()`, no model): text that is not
     prose at all (word salad, markup debris, a stuck loop) scores 0 and no other judge is asked.
     Otherwise it computes chapter features, projects them into novel-feature space and
     scores with the engagement model (or heuristic targets if no model), and asks the LLM for a
     1-10 rubric (hook, momentum, payoff, cliffhanger, clarity). Blended 50/50.
   - Each present character's `CharacterAgent.review()` scores their own portrayal; the chapter
     score is blended with that consistency (`write.character_weight`).
   - If below `min_acceptable_score`, `Drafter.revise()` is called with the editor's notes, the
     measured issues and the characters' own corrections, up to `max_revisions` times. The best
     version is kept. An unreadable draft is drafted again rather than revised; if every attempt is
     unreadable, `ChapterRejected` is raised: the last attempt goes to `drafts/rejected/`, and the
     steps below (memory, save, learning) do not run.
   - Every character on the page runs `remember()`: memories, mood, knowledge and relationship
     deltas are written back to `characters/<name>.json`.
   - The chapter is saved as Markdown with front matter; the running summary and open threads
     are updated by the LLM; the score is logged in `generated`.

## World, muse, techniques

- `write/world.py`: `WorldState` (rules, clock, weather, places, events, consequences, nameless
  figures, history) and `WorldAgent` (build / brief / review / remember / chat).
  `intervention_level()` decides how hard the world pushes a scene. Persisted as `world.json`.
- `write/muse.py`: `Muse` proposes and rates alternative directions (surprise, fit, payoff, plus a
  deterministic novelty penalty against past directions); `Planner.direction_for` hooks it into
  outline extension. `NoveltyTracker` scores drafts against the project's own earlier chapters.
- `learn/techniques.py`: `Absorber` turns the strongest unread chapters into `Technique` cards
  (LLM extraction, embedded); `TechniqueLibrary.search` diversifies by kind. The `absorbed` table
  makes it incremental; the daemon runs it each cycle.
- `WritePipeline.write_next` order: plan (muse direction if the outline needs extending) ->
  character briefs -> world brief -> technique cards + avoid list -> draft -> critic + character
  reviews + world review + novelty -> revise -> character memory + world memory + story summary.
- `gui.py`: `StudioController` (all logic, tested) and `run_studio` (tkinter window).

## The writing room (`write/agents/`)

`StoryAgent` (base.py) defines the four moments (`plan_notes`, `brief`, `review`, `remember`)
plus `setup` and `report`, with JSON state under `projects/<slug>/agents/`. `AgentCouncil`
(council.py) runs the enabled agents in parallel at each moment, merges weighted scores, and
collects issues for the reviser; `build_council` reads `AgentsConfig`. Every agent prompt
starts with `[agent:<name>:<op>]` so the offline `FakeLLM` can answer from `agents/fakes.py`.
Agents: readers, showrunner, archivist, schemer, editor, packager, theme, sensitivity, pacing,
librarian, comedian.

## Research (`research/`)

`sources.py` holds one class per open source with pure parsing functions (testable offline);
`scout.py` runs queries through them with a SQLite cache (`research_cache`), logs them
(`research_log`), files Markdown dossiers under `projects/<slug>/research/scout/` plus a
bibliography, and can import Project Gutenberg books as `gutenberg:` novels. The `ScoutAgent`
in the writing room decides the per-chapter questions and fact-checks drafts; the Librarian
indexes what the Scout files. `GAUGE_OFFLINE=1` disables all sources.

## Learning from your own serial and personality depth

- `learn/bandit.py` (factored Thompson sampling over prompt knobs, per project), `learn/project_model.py`
  (ridge on your chapters' features vs retention from `feedback`), `learn/ledger.py` (predictions
  table, evaluation of old predictions), `learn/lessons.py` (themes from `chapter_notes`),
  `learn/comments.py` (comment lexicon; Royal Road comments are parsed from chapter pages),
  `learn/finetune.py` (self-preference pairs, author-edit pairs, plan/run for `scripts/finetune_lora.py`).
- `write/persona.py` (author persona in the system prompt), `write/improv.py` (turn-based scene
  improv), `write/emotion.py` (PAD mood, decay, speech directives, arc drift), `write/stylometry.py`
  + `agents/voice_coach.py` (dialogue attribution and distinctness), `Relationship.beliefs` and
  `Roster.triangles()` in `characters.py` (second-order beliefs for the Schemer).
- `WritePipeline.write_next` order now: bandit draw -> plan -> character briefs -> world brief ->
  improv -> technique cards + avoid list -> council briefs -> draft -> critic (+ project model) +
  reviews -> revise -> memories -> notes, preference pair, accepted copy, bandit reward.

## GaugeLM (`native/`)

A self-contained model provider: `BPETokenizer` (pure Python), `GaugeLM` (numpy forward and
backward, KV-cache generation, save/load as npz+json), `AdamW`, `NativeTrainer` (corpus from the
database, projects and datasets; schedule; checkpoints; continue-training), and `NativeLLM`
(the `LLM` protocol with schema-guided structured output). `make_llm` returns it for
`llm.provider = "native"`; the daemon calls `continue_training` each cycle when a model exists.

## Execution model

- `llm/tokens.py`: token estimation and `PromptBudget`; `Drafter.build_draft_prompt` fits every
  section into `num_ctx - output - margin` in priority order (plan, previous tail, bible, summary,
  cast, exemplars) and logs what it trimmed.
- `llm/client.py`: `num_ctx` and `keep_alive` on every call, retries with backoff on 429/5xx and
  connection errors, JSON-schema `format` with automatic fallback to plain JSON mode on a 400, and
  a `CallRecord` per call into a thread-safe `CallLog` whose sink writes `llm_calls` rows.
  `generate_json()` adds one low-temperature repair retry. `make_llm(cfg, "fast")` returns a
  client on `fast_model` sharing the same log.
- `llm/parallel.py`: `run_parallel` for briefs, reviews and memory updates, order-preserving,
  one failure never cancels the rest.
- `write/schemas.py`: pydantic models for every structured answer; prompts and contracts live
  side by side.
- Checkpoints: `Project.save_checkpoint` after the draft and after each revision; `write_next`
  resumes when the checkpoint's plan title matches, clears it on acceptance.
- Timings: `ChapterResult.timings` (briefs, draft, critique, revise, memory, total), plus call and
  model-second counts; `gauge bench` uses measured tokens/s to estimate a chapter.

## Why these choices

- **SQLite + JSON + Markdown**: zero services to run, trivially backed up, hand-editable.
- **Ridge regression first**: a few hundred labelled novels cannot support a deep model; a
  linear model with cross-validation gives honest, readable coefficients.
- **Transparent features**: every number the model uses can be explained to a writer. The
  style guide is literally the coefficients translated to English.
- **Local LLM via HTTP**: Ollama and LM Studio are the two easiest Windows installs; both are
  supported without SDK dependencies.
- **FakeLLM everywhere**: the whole pipeline is testable offline in about a second.

## Extending

- New site: implement `matches`, `discover`, `fetch_novel`, `list_chapters`, `fetch_chapter`
  (see `ingest/base.py`) and add it to `build_sources`.
- New feature: add a function in `features.py` and merge its dict into `chapter_features` or
  `novel_features`. The model picks it up on the next `gauge train`.
- New target: add a lambda to `learn/engagement.py: TARGETS`.
- New prompt style: edit `write/prompts.py`.
