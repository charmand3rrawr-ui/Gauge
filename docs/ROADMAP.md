# Roadmap

Things the current design makes easy, roughly in order of value.

- **Growth targets.** `metric_history` already records every observation. Add a target for
  followers gained per chapter released over the last 30 days, which isolates *recent* chapter
  quality from a novel's age.
- **Chapter-level engagement.** Royal Road exposes per-chapter comment counts on chapter pages
  and some sites expose per-chapter views. Record them (`ChapterRecord.views/comments` exist)
  and train a chapter-level model: which chapters inside a popular novel were the dips?
- **Comment sentiment.** Reader comments are the closest thing to a label for "I could not stop".
  A small classifier over comment text would be a strong target.
- **More connectors.** Wattpad, WebNovel, Tapas, AO3 each need four functions. Respect their
  terms.
- **Author-style fingerprinting.** Reuse `voice_features` to keep a project's voice consistent
  across hundreds of chapters; flag drift.
- **Series-level planning.** Arc-level payoff scheduling (promise/progress/payoff spacing)
  learned from the position of reveal-dense chapters in popular novels.
- **In-process LoRA.** Optional module that runs Unsloth training on the exported datasets when
  a CUDA GPU is present, then registers the GGUF with Ollama.
- **Local web UI.** A small FastAPI + HTMX page over the same SQLite file: leaderboard, feature
  importances, project view, "write next chapter" button.
