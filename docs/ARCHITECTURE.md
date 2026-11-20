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
   - `Critic.critique()` computes chapter features, projects them into novel-feature space and
     scores with the engagement model (or heuristic targets if no model), and asks the LLM for a
     1-10 rubric (hook, momentum, payoff, cliffhanger, clarity). Blended 50/50.
   - Each present character's `CharacterAgent.review()` scores their own portrayal; the chapter
     score is blended with that consistency (`write.character_weight`).
   - If below `min_acceptable_score`, `Drafter.revise()` is called with the editor's notes, the
     measured issues and the characters' own corrections, up to `max_revisions` times. The best
     version is kept.
   - Every character on the page runs `remember()`: memories, mood, knowledge and relationship
     deltas are written back to `characters/<name>.json`.
   - The chapter is saved as Markdown with front matter; the running summary and open threads
     are updated by the LLM; the score is logged in `generated`.

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
