"""GaugeLM: tokenizer, model math (gradient check), training, generation, the native provider, CLI."""
import json

import numpy as np

from gauge.config import GaugeConfig, LLMConfig
from gauge.llm import make_llm
from gauge.native import (
    AdamW,
    BPETokenizer,
    GaugeLM,
    ModelConfig,
    NativeLLM,
    NativeTrainer,
    TrainConfig,
    gather_texts,
)
from gauge.native.train import encode_corpus, lr_at
from gauge.write.schemas import CriticOut, SheetOut, schema_of


# ------------------------------------------------------------------ tokenizer
def test_bpe_roundtrip_and_specials(tmp_path):
    text = "The courier ran. The courier ran again, and again, through the rain. Ünïcode too: 你好!"
    tok = BPETokenizer.train(text * 20, vocab_size=320)
    assert 256 < tok.vocab_size <= 320 + 5
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert len(ids) < len(text.encode("utf-8"))  # merges learned
    chat = tok.encode_chat("prompt here", "the answer")
    assert chat[0] == tok.bos and chat[1] == tok.special["<prompt>"] and chat[-1] == tok.eos and tok.special["<response>"] in chat
    assert tok.decode(chat) == "prompt herethe answer"  # specials vanish from text
    tok.save(tmp_path / "tok.json")
    again = BPETokenizer.load(tmp_path / "tok.json")
    assert again.encode(text) == ids and again.vocab_size == tok.vocab_size
    assert BPETokenizer().encode("abc") == [97, 98, 99]  # untrained = raw bytes


# ------------------------------------------------------------------ model math
def test_gradients_match_finite_differences():
    cfg = ModelConfig(vocab_size=37, block_size=8, n_layer=2, n_head=2, n_embd=16)
    m = GaugeLM(cfg, seed=1)
    for k in m.p:
        m.p[k] = m.p[k].astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, cfg.vocab_size, size=(2, 6))
    tgt = rng.integers(0, cfg.vocab_size, size=(2, 6))
    tgt[0, 2] = -1  # ignored position
    _, loss, caches = m.forward(idx, tgt)
    grads = m.backward(caches)
    assert loss > 0 and set(grads) == set(m.p)
    worst = 0.0
    for name in ["wte", "wpe", "lnf_g", "l0.qkv", "l0.proj", "l1.fc", "l1.fc2", "l0.ln2_b", "l1.qkv_b", "l0.fc_b"]:
        p = m.p[name]
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p[i]
            p[i] = old + 1e-5
            lp = m.forward(idx, tgt)[1]
            p[i] = old - 1e-5
            lm = m.forward(idx, tgt)[1]
            p[i] = old
            num = (lp - lm) / 2e-5
            worst = max(worst, abs(num - grads[name][i]) / max(1e-7, abs(num) + abs(grads[name][i])))
    assert worst < 1e-4


def test_model_save_load_generate_and_kv_cache(tmp_path):
    cfg = ModelConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2, n_embd=16)
    m = GaugeLM(cfg, seed=3)
    assert m.n_params() == sum(v.size for v in m.p.values())
    m.save(tmp_path / "m", {"step": 7})
    again = GaugeLM.load(tmp_path / "m")
    assert all(np.array_equal(m.p[k], again.p[k]) for k in m.p)
    assert json.loads((tmp_path / "m" / "model.json").read_text())["step"] == 7
    # greedy generation is deterministic and the KV-cache path matches full recomputation
    prompt = [1, 2, 3]
    out = m.generate(prompt, 5, temperature=0.0)
    assert out[:3] == prompt and len(out) == 8
    full_logits, _, _ = m.forward(np.asarray([out[:-1]]))
    assert int(full_logits[0, -1].argmax()) == out[-1]
    # min_tokens keeps the stop token away; stop token ends early otherwise
    stop = {out[3]}
    out2 = m.generate(prompt, 5, temperature=0.0, stop_tokens=stop)
    assert len(out2) == 4
    out3 = m.generate(prompt, 5, temperature=0.0, stop_tokens=stop, min_tokens=3)
    assert len(out3) >= 6
    # sampling past the block size re-primes the cache without crashing
    long = m.generate(prompt, 40, temperature=1.0, rng=np.random.default_rng(1))
    assert len(long) == 43
    opt = AdamW(m.p, lr=1e-3)
    grads = {k: np.ones_like(v) for k, v in m.p.items()}
    before = m.p["wte"].copy()
    norm = opt.step(m.p, grads)
    assert norm > 0 and not np.array_equal(before, m.p["wte"])


# ------------------------------------------------------------------ training
def test_training_lowers_loss_and_resumes(tmp_path, chapter_text):
    docs = [("prose", chapter_text(i % 2 == 0)) for i in range(30)] + [("chat:Write the opening.", chapter_text(True))] * 6
    tr = NativeTrainer(tmp_path / "native")
    assert not tr.exists() and tr.status() == {"trained": False, "n_params": None, "config": None}
    mcfg = ModelConfig(vocab_size=500, block_size=32, n_layer=1, n_head=2, n_embd=32)
    tcfg = TrainConfig(steps=60, batch_size=8, lr=3e-3, min_lr=3e-4, warmup=5, eval_every=30, checkpoint_every=30, seed=0)
    seen = []
    res = tr.train(docs, mcfg, tcfg, progress=seen.append)
    assert res["step"] == 60 and res["val_loss"] is not None and tr.exists()
    assert len(seen) == 2 and seen[-1]["loss"] < seen[0]["loss"] * 0.8  # it learns
    assert tr.status()["step"] == 60 and tr.status()["n_params"] > 0
    res2 = tr.continue_training(docs, steps=10)
    assert res2["step"] == 70 and tr.status()["step"] == 70
    assert lr_at(0, tcfg) < lr_at(5, tcfg) and lr_at(60, tcfg) == tcfg.min_lr
    tok = BPETokenizer.load(tmp_path / "native" / "tokenizer.json")
    arr = encode_corpus(tok, docs[:2], max_chars=10_000)
    assert arr.dtype == np.int32 and arr[0] == tok.bos and tok.eos in arr
    res3 = tr.train(docs, mcfg, TrainConfig(steps=1000, eval_every=10, checkpoint_every=10, warmup=1), resume=True, time_budget_s=0.01)
    assert res3["step"] < 1070  # the time budget stopped it and kept a checkpoint


def test_gather_texts_from_db_projects_and_datasets(cfg, db, chapter_text, tmp_path):
    from gauge.models import ChapterRecord, NovelRecord
    from gauge.write.project import Project

    n = NovelRecord(source="local", source_id="a", title="A")
    db.upsert_novel(n)
    db.upsert_chapter(ChapterRecord(novel_key=n.key, index=1, text=chapter_text(True)))
    p = Project.create(cfg.projects_dir, "Own", "idea", ["Fantasy"], 300)
    p.write_chapter(1, "One", "my own chapter " * 30, {})
    p.save()
    ds = cfg.models_dir / "datasets"
    ds.mkdir(parents=True)
    (ds / "sft.jsonl").write_text(json.dumps({"instruction": "write", "input": "x", "output": "the output"}) + "\n", encoding="utf-8")
    (ds / "author_edits.jsonl").write_text(json.dumps({"prompt": "p", "chosen": "kept", "rejected": "old"}) + "\n", encoding="utf-8")
    docs = gather_texts(db, cfg.projects_dir, ds)
    kinds = [k.split(":")[0] for k, _ in docs]
    assert kinds.count("prose") == 2 and kinds.count("chat") == 2
    assert any(t == "kept" for _, t in docs) and not any(t == "old" for _, t in docs)
    assert gather_texts(db, None, None, since="2999-01-01") == []


# ------------------------------------------------------------------ provider
def test_native_llm_provider_and_schema_fill(tmp_path, chapter_text):
    docs = [("prose", chapter_text(i % 2 == 0)) for i in range(30)]
    d = tmp_path / "native"
    NativeTrainer(d).train(docs, ModelConfig(vocab_size=400, block_size=32, n_layer=1, n_head=2, n_embd=32),
                           TrainConfig(steps=40, batch_size=8, lr=3e-3, warmup=5, eval_every=40, checkpoint_every=40))
    cfg = LLMConfig(provider="native", temperature=0.8, native_dir=str(d))
    llm = make_llm(cfg)
    assert isinstance(llm, NativeLLM) and llm.available() and "params" in llm.describe() and llm.context_window() == 32
    out = llm.generate("Write the opening of a chapter.", max_tokens=30, purpose="t")
    assert isinstance(out, str) and out and llm.calls.records[-1].purpose == "t" and llm.calls.records[-1].ok
    crit = json.loads(llm.generate("Score this chapter", schema=schema_of(CriticOut), purpose="critic"))
    assert set(crit) == {"hook", "momentum", "payoff", "cliffhanger", "clarity", "notes"} and isinstance(crit["notes"], str)
    sheet = json.loads(llm.generate("Create a character sheet", schema=schema_of(SheetOut), purpose="sheet"))
    assert "matrix" in sheet and "openness" in sheet["matrix"] and isinstance(sheet["aliases"], list) and sheet["relationships"] == {}
    assert isinstance(sheet["voice"]["swears"], bool)
    generic = json.loads(llm.generate("anything", json_mode=True, purpose="j"))
    assert "text" in generic
    missing = NativeLLM(cfg, tmp_path / "nowhere")
    assert not missing.available() and "not trained" in missing.describe() and missing.context_window() == 256
    tp = llm.throughput(20)
    assert tp["generation_tps"] >= 0


def test_native_pipeline_runs_without_outside_models(cfg, db, chapter_text, monkeypatch):
    """The whole writing pipeline on GaugeLM alone: slow and crude, but no outside code."""
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    docs = [("prose", chapter_text(i % 2 == 0)) for i in range(30)]
    NativeTrainer(cfg.native_dir).train(docs, ModelConfig(vocab_size=400, block_size=48, n_layer=1, n_head=2, n_embd=32),
                                        TrainConfig(steps=30, batch_size=8, lr=3e-3, warmup=5, eval_every=30, checkpoint_every=30))
    cfg.llm.provider = "native"
    cfg.llm.native_dir = str(cfg.native_dir)
    cfg.llm.num_ctx = 48
    cfg.write.max_revisions = 0
    cfg.write.improv_rounds = 1
    cfg.agents.profile = "lean"
    cfg.write.characters_enabled = False
    cfg.write.world_enabled = False
    cfg.write.muse_enabled = False
    from gauge.write import WritePipeline

    pipe = WritePipeline(cfg, db)
    assert isinstance(pipe.llm, NativeLLM) and isinstance(pipe.fast_llm, NativeLLM)
    p = pipe.new_project("Native Tale", "a courier", ["Sci-fi"], words=40)
    assert p.state.outline  # the outline came back as a filled schema
    res = pipe.write_next(p)
    assert res.path.exists() and res.number == 1


def test_cli_native(tmp_path, chapter_text):
    from click.testing import CliRunner

    from gauge.cli import main

    data = str(tmp_path / "data")
    runner = CliRunner(env={"COLUMNS": "200"})
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    r = runner.invoke(main, ["--data-dir", data, "native", "status"])
    assert r.exit_code == 0 and '"trained": false' in r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "train", "--steps", "5"])
    assert r.exit_code != 0 and "nothing to train on" in r.output
    d = tmp_path / "novel"
    d.mkdir()
    for i in range(6):
        (d / f"{i}.txt").write_text(chapter_text(i % 2 == 0) * 2, encoding="utf-8")
    runner.invoke(main, ["--data-dir", data, "import", str(d)])
    r = runner.invoke(main, ["--data-dir", data, "native", "train", "--steps", "20", "--layers", "1", "--embd", "32", "--heads", "2",
                             "--block", "32", "--vocab", "400", "--batch", "4"])
    assert r.exit_code == 0 and "saved" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "status"])
    assert r.exit_code == 0 and '"step": 20' in r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "sample", "Write the opening.", "--tokens", "20"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "tokenizer", "The courier ran."])
    assert r.exit_code == 0 and "tokens (vocab" in r.output
    r = runner.invoke(main, ["--data-dir", data, "native", "continue", "--steps", "5", "--minutes", "1"])
    assert r.exit_code == 0 and '"step": 25' in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "init", "--provider", "native"])
    assert r.exit_code == 0
    assert json.loads((tmp_path / "data" / "config.json").read_text())["llm"]["provider"] == "native"
    r = runner.invoke(main, ["--data-dir", data, "doctor"])
    assert r.exit_code == 0 and "native GaugeLM" in r.output and "reachable" in r.output
    assert GaugeConfig.load(data).llm.native_dir.endswith("native")
