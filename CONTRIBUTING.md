# Contributing

Thanks for helping. The bar is simple: tests pass, lint passes, the ethics doc still holds.

```powershell
pip install -e ".[ml,epub,dev]"
pytest -q
ruff check gauge tests
```

## Adding a site connector

1. Create `gauge/ingest/<site>.py` with pure parsing functions (`parse_novel_page`,
   `parse_chapter_list`, `parse_chapter_page`, `parse_listing`) and a class implementing the
   `Source` protocol in `ingest/base.py`.
2. Add a small HTML fixture to `tests/fixtures/` and a test in `tests/test_parsers.py`. Keep
   fixtures synthetic (hand-written HTML with the right structure), not copied pages.
3. Register it in `ingest/runner.py: build_sources`.
4. Confirm the site's robots.txt and terms allow it, and say so in the PR.

## Adding a feature

Add a function in `gauge/features.py` returning `{name: float}`, merge it in
`chapter_features` or `novel_features`, add a test showing it moves in the expected direction on
the synthetic chapters in `tests/conftest.py`. If it needs a word list, put it in `lexicon.py`.
Prefix related features consistently so the style guide can describe them (`learn/style.py`).

## Style

- Python 3.10+, type hints, docstrings that explain *why*.
- No new runtime dependencies without a strong reason; the install must stay painless on a
  plain Windows machine.
- Prompts live in `write/prompts.py` only.
