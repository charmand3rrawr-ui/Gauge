"""Execution: context budgeting, retries, schema fallback, JSON repair, parallelism, fast model,
checkpoints, telemetry and bench."""
import copy
import json
import threading
import time
from types import SimpleNamespace

import pytest

from gauge.config import LLMConfig
from gauge.learn import Trainer
from gauge.llm import (
    FakeLLM,
    HashingEmbedder,
    LLMError,
    OllamaLLM,
    PromptBudget,
    estimate_tokens,
    generate_json,
    make_llm,
    run_parallel,
)
from gauge.llm.tokens import truncate_to_tokens
from gauge.write import WritePipeline
from gauge.write.drafter import Drafter
from gauge.write.project import ChapterPlan, Project


# ------------------------------------------------------------------ item 1: context budget
def test_token_estimate_and_budget():
    assert estimate_tokens("") == 0
    assert 60 < estimate_tokens("word " * 50) < 80
    cut = truncate_to_tokens("word " * 500, 100)
    assert estimate_tokens(cut) <= 110 and cut.endswith("...")
    b = PromptBudget(num_ctx=1000, reserved_output=300, margin=100)
    assert b.capacity == 600
    b.take("x " * 100)
    assert b.fit("a", "y " * 100) and not b.dropped
    assert b.fit("b", "z " * 2000, shrinkable=False) == "" and "b" in b.dropped
    kept = b.fit_list("ex", ["e " * 100, "f " * 100, "g " * 100, "h " * 100])
    assert kept.count("\n\n") < 3 and any(d.startswith("ex[") for d in b.dropped)
    assert not b.over


def _project(tmp_path, long_summary=True):
    p = Project.create(tmp_path, "Budget", "idea", ["Sci-fi"], 300)
    p.state.bible.premise = "premise " * 200
    if long_summary:
        p.state.running_summary = "summary " * 400
    return p


class _Lib:
    """Stand-in exemplar library returning k long passages."""

    def search(self, q, kind=None, k=4, genres=None):
        return [SimpleNamespace(novel_title=f"N{i}", text=f"{kind} passage " * 90) for i in range(k)]


def test_draft_prompt_is_trimmed_to_context_window(tmp_path):
    plan = ChapterPlan(number=1, title="T", goal="g", hook="h", turn="t", cliffhanger="c", beats=["a"])
    big = FakeLLM(LLMConfig(provider="fake", num_ctx=32000))
    small = FakeLLM(LLMConfig(provider="fake", num_ctx=2500))
    p = _project(tmp_path)
    prev = "previous chapter words " * 300
    d_big = Drafter(big, _Lib(), "guide", exemplars_in_prompt=4)
    prompt_big, max_out = d_big.build_draft_prompt(p, plan, prev, 300, cast="cast " * 300)
    assert not d_big.last_budget.dropped and prompt_big.count("passage") >= 8 * 90
    d_small = Drafter(small, _Lib(), "guide", exemplars_in_prompt=4)
    prompt_small, _ = d_small.build_draft_prompt(p, plan, prev, 300, cast="cast " * 300)
    assert estimate_tokens(prompt_small) + estimate_tokens(d_small.system) + max_out <= 2500
    assert d_small.last_budget.dropped  # exemplars went first
    assert "there is no previous" not in prompt_small and "previous chapter words" in prompt_small  # continuity kept
    assert "Hook: h" in prompt_small  # plan is fixed
    assert d_small.draft(p, plan, prev, 300).startswith('"Don\'t open it,"')


# ------------------------------------------------------------------ item 3: retries, schema fallback, repair
class _Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def test_ollama_sends_num_ctx_retries_and_falls_back_from_schema(monkeypatch):
    cfg = LLMConfig(num_ctx=4096, keep_alive="1h", retries=2, retry_backoff_seconds=0.0, fast_model="tiny")
    seen = []
    answers = [_Resp(503, text="busy"), _Resp(400, text="format not supported"),
               _Resp(200, {"response": '{"a": 1}', "prompt_eval_count": 10, "eval_count": 5, "eval_duration": 500_000_000})]

    def fake_post(url, json=None, timeout=None, headers=None):
        seen.append(copy.deepcopy(json))
        return answers.pop(0)

    monkeypatch.setattr("gauge.llm.client.requests.post", fake_post)
    monkeypatch.setattr("gauge.llm.client.time.sleep", lambda s: None)
    llm = make_llm(cfg, "fast")
    assert isinstance(llm, OllamaLLM) and llm.model == "tiny"
    out = llm.generate("hi", schema={"type": "object"}, purpose="t")
    assert out == '{"a": 1}'
    assert seen[0]["options"]["num_ctx"] == 4096 and seen[0]["keep_alive"] == "1h"
    assert seen[0]["format"] == {"type": "object"} and seen[1]["format"] == {"type": "object"} and seen[2]["format"] == "json"
    assert llm._schema_unsupported  # remembered for later calls
    rec = llm.calls.records[-1]
    assert rec.ok and rec.prompt_tokens == 10 and rec.output_tokens == 5 and rec.tokens_per_second == 10.0 and rec.retries == 2  # one 503 retry + one schema fallback
    # exhaustion raises LLMError and records a failed call
    answers.extend([_Resp(500), _Resp(500), _Resp(500)])
    with pytest.raises(LLMError):
        llm.generate("again", purpose="t2")
    assert llm.calls.records[-1].ok is False
    assert llm.calls.summary()["t2"]["failed"] == 1


def test_generate_json_repairs_bad_output():
    llm = FakeLLM(canned=["Sure! Here it is: not json", '{"fixed": true}'])
    assert generate_json(llm, "give me json", purpose="x") == {"fixed": True}
    assert [c["purpose"] for c in llm.calls_list] == ["x", "x:repair"]
    assert "not valid JSON" in llm.calls_list[1]["prompt"]
    with pytest.raises(ValueError):
        generate_json(FakeLLM(canned=["nope", "still nope"]), "p", purpose="y")


# ------------------------------------------------------------------ item 2: parallelism & fast model
def test_run_parallel_preserves_order_and_isolates_failures():
    seen_threads = set()

    def work(i):
        seen_threads.add(threading.current_thread().name)
        time.sleep(0.02)
        if i == 2:
            raise RuntimeError("boom")
        return i * 10

    out = run_parallel(work, [0, 1, 2, 3], workers=4)
    assert out == [0, 10, None, 30]
    assert len(seen_threads) > 1
    assert run_parallel(work, [], workers=4) == []
    assert run_parallel(lambda x: x, [5], workers=1) == [5]


def _trained(cfg, db, chapter_text):
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    return t


def test_pipeline_uses_fast_model_parallel_and_skips_light_mentions(cfg, db, chapter_text):
    cfg.llm.parallel = 3
    t = _trained(cfg, db, chapter_text)
    main, fast = FakeLLM(delay=0.05), FakeLLM(delay=0.05)
    pipe = WritePipeline(cfg, db, llm=main, trainer=t, fast_llm=fast)
    assert main.calls is fast.calls is pipe.calls
    p = pipe.new_project("Ten Seconds", "a courier", ["Sci-fi"], words=300)
    # 3 bible characters -> sheets were created with the main model
    assert sum(1 for c in main.calls_list if c["purpose"] == "character:create") == 3
    # Name every character in the plan so all three are activated; Jun is only name-dropped in the prose.
    p.state.outline[0].goal = "Rin, Jun and The Broker collide. The Broker speaks. Jun listens."
    p.save()
    before = len(pipe.calls.records)
    res = pipe.write_next(p)
    purposes = [c["purpose"] for c in fast.calls_list]
    assert purposes.count("character:brief") == 3 and set(res.cast) == {"Rin", "Jun", "The Broker"}
    # prose only mentions Rin -> only Rin reviewed and remembered (protagonist), others skipped
    assert purposes.count("character:review") >= 1 and all(c["system"].startswith("You are Rin") for c in fast.calls_list if c["purpose"] == "character:review")
    assert purposes.count("character:memory") == 1
    assert "critic" in purposes and "summary" in purposes
    main_purposes = {c["purpose"] for c in main.calls_list}
    assert "draft" in main_purposes and not {"critic", "summary", "character:brief"} & main_purposes
    # briefs ran in parallel: 3 x 50ms delay would take 150ms sequentially
    assert res.timings["briefs"] < 0.12
    assert res.llm_calls == len(pipe.calls.records) - before and res.llm_seconds >= 0
    assert set(res.timings) >= {"briefs", "draft", "critique", "memory", "total"}
    # telemetry landed in the database, tagged with the project and chapter
    rows = db.call_summary(project="Ten Seconds")
    assert {r["purpose"] for r in rows} >= {"draft", "critic", "character:brief", "summary"}
    assert db._conn.execute("SELECT COUNT(*) FROM llm_calls WHERE chapter=1").fetchone()[0] >= 6


def test_revision_only_re_reviews_flagged_characters(cfg, db, chapter_text):
    cfg.write.min_acceptable_score = 0.95
    cfg.write.max_revisions = 1
    t = _trained(cfg, db, chapter_text)
    fast = FakeLLM()
    pipe = WritePipeline(cfg, db, llm=FakeLLM(), trainer=t, fast_llm=fast)
    p = pipe.new_project("Ten Seconds", "a courier", ["Sci-fi"], words=300)
    # Make The Broker appear heavily in prose via canned draft+revision text so he is reviewed too.
    broker_text = chapter_text(True).replace("she said", "The Broker said").replace("I counted", "The Broker counted") + "\n\nRin ran."
    pipe.llm.canned = [broker_text, broker_text]
    fast.canned = []
    p.state.outline[0].goal = "Rin faces The Broker"
    p.save()
    # The fake review always returns one issue for everyone, so both get re-reviewed; verify counts are consistent.
    res = pipe.write_next(p)
    reviews = [c for c in fast.calls_list if c["purpose"] == "character:review"]
    assert res.revisions == 1
    assert len(reviews) == 4  # 2 characters x (initial + re-review because both had issues)
    assert len(res.character_reviews) == 2


# ------------------------------------------------------------------ item 4: checkpoints
def test_crash_after_draft_resumes_from_checkpoint(cfg, db, chapter_text):
    t = _trained(cfg, db, chapter_text)
    boom = FakeLLM(fail_on=lambda prompt: "Score this chapter" in prompt)
    pipe = WritePipeline(cfg, db, llm=boom, trainer=t, fast_llm=boom)
    p = pipe.new_project("Ten Seconds", "a courier", ["Sci-fi"], words=300)
    # the critic swallows LLM errors; make the drafter's follow-up (summary) the crash instead
    boom.fail_on = lambda prompt: "running story summary" in prompt.lower() and False
    good = FakeLLM()
    pipe2 = WritePipeline(cfg, db, llm=good, trainer=t, fast_llm=good)
    # simulate: draft generated, then process dies before acceptance -> checkpoint remains
    plan = pipe2.planner.ensure_plan(p, 1)
    p.save_checkpoint(1, "draft", "CHECKPOINTED DRAFT " * 60, {"plan_title": plan.title, "cast": ["Rin"], "revisions": 0})
    assert p.load_checkpoint(1)["stage"] == "draft"
    res = pipe2.write_next(p)
    assert res.resumed and "CHECKPOINTED DRAFT" in res.path.read_text(encoding="utf-8")
    assert not any(c["purpose"] == "draft" for c in good.calls_list)  # no re-generation
    assert not any(c["purpose"] == "character:brief" for c in good.calls_list)
    assert p.load_checkpoint(1) is None  # cleared on acceptance
    assert "resumed: true" in res.path.read_text(encoding="utf-8")
    # a stale checkpoint for a different plan title is ignored
    p.save_checkpoint(2, "draft", "STALE " * 60, {"plan_title": "something else", "cast": [], "revisions": 0})
    res2 = pipe2.write_next(p)
    assert not res2.resumed and "STALE" not in res2.path.read_text(encoding="utf-8")
    # --fresh ignores a valid checkpoint
    plan3 = pipe2.planner.ensure_plan(p, 3)
    p.save_checkpoint(3, "draft", "FRESH-IGNORED " * 60, {"plan_title": plan3.title, "cast": [], "revisions": 0})
    res3 = pipe2.write_next(p, fresh=True)
    assert not res3.resumed and "FRESH-IGNORED" not in res3.path.read_text(encoding="utf-8")


def test_checkpoint_written_before_critique(cfg, db, chapter_text):
    """If the critic step dies, the draft is already on disk."""
    t = _trained(cfg, db, chapter_text)
    llm = FakeLLM()
    pipe = WritePipeline(cfg, db, llm=llm, trainer=t, fast_llm=llm)
    p = pipe.new_project("Ten Seconds", "a courier", ["Sci-fi"], words=300)

    def explode(*a, **k):
        raise RuntimeError("critic died")

    pipe.critic.critique = explode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        pipe.write_next(p)
    ck = p.load_checkpoint(1)
    assert ck and ck["stage"] == "draft" and ck["text"].startswith('"Don\'t open it,"') and ck["cast"] == ["Rin"]


# ------------------------------------------------------------------ item 5: CLI telemetry + bench
def test_cli_write_timing_report_and_bench(tmp_path):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "220"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    r = runner.invoke(main, ["--data-dir", data, "doctor"])
    assert r.exit_code == 0 and "num_ctx=8192" in r.output and "parallel=4" in r.output
    r = runner.invoke(main, ["--data-dir", data, "new", "Ten Seconds", "--idea", "a courier", "--genre", "Sci-fi", "--words", "200"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--data-dir", data, "write", "Ten Seconds", "--fresh"])
    assert r.exit_code == 0 and "timing:" in r.output and "LLM calls" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "report"])
    assert r.exit_code == 0 and "LLM time by purpose" in r.output and "draft" in r.output
    r = runner.invoke(main, ["--data-dir", data, "bench", "--tokens", "50"])
    assert r.exit_code == 0 and "Estimated per chapter" in r.output and "about" in r.output, r.output
    cfg_path = tmp_path / "data" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    assert cfg["llm"]["num_ctx"] == 8192 and cfg["write"]["checkpoint_drafts"] is True
