"""Batch C: multi-language reading, binary-level tooling, and the mutation gate for self-edits."""

import pytest
from click.testing import CliRunner

from gauge.cli import main
from gauge.code.binary import (
    Field,
    disassemble,
    entropy,
    from_bits,
    from_hex,
    hexdump,
    identify,
    pack_struct,
    parse_record,
    to_bits,
    to_hex,
    unpack_struct,
)
from gauge.code.languages import analyze, detect, gotchas, idioms, known, normalize, tips
from gauge.code.mutation import generate_mutants, mutation_score, report
from gauge.config import GaugeConfig


def test_language_detect_and_analyze():
    assert detect("main.rs", "fn main() {}") == "rust"
    assert detect("a.go", "package main") == "go"
    assert normalize("C++") == "cpp" and known("python") and not known("cobol-77")
    assert idioms("python") and gotchas("rust") and "python" in tips("python")
    shape = analyze("m.py", "import os\n\nclass A:\n    def f(self, x):\n        if x == 1 and x > 0:\n            return True\n        return False\n")
    assert shape.language == "python" and shape.functions >= 1 and shape.classes == 1 and shape.tokens > 0
    deep = analyze("d.py", "def f():\n" + "".join("    " * i + "if True:\n" for i in range(1, 8)) + "    " * 8 + "pass\n")
    assert any("nesting" in s for s in deep.smells)


def test_binary_tooling():
    assert identify(b"\x7fELF\x02\x01") == "ELF executable (Linux)"
    assert identify(b"PK\x03\x04") == "ZIP archive (or .jar/.docx/.xlsx)"
    assert "hello" in hexdump(b"hello world")
    assert from_bits(to_bits(b"Gauge")) == b"Gauge"
    assert from_hex(to_hex(b"\x00\xff")) == b"\x00\xff"
    rec = parse_record([Field("magic", "4s"), Field("version", "H"), Field("count", "I")],
                       pack_struct("<4sHI", b"GLM1", 7, 42))
    assert rec == {"magic": b"GLM1", "version": 7, "count": 42}
    assert unpack_struct("<I", b"\x01\x00\x00\x00\xff")[0] == 1
    with pytest.raises(ValueError):
        unpack_struct("<I", b"\x01")
    assert entropy(b"\x00" * 100) < entropy(bytes(range(256)))


def test_disassemble_python():
    d = disassemble("x = 1 + 2\n")
    assert d.instructions and any(i["op"].startswith("LOAD") or "CONST" in i["op"] for i in d.instructions)

    def fn(a, b):
        return a + b
    d2 = disassemble(fn)
    assert d2.name == "fn" and d2.instructions


def test_mutation_generation_and_scoring():
    src = "def f(a, b):\n    return a == b and a > 0\n"
    muts = generate_mutants(src)
    labels = {m.label for m in muts}
    assert "== -> !=" in labels and any("and" in x for x in labels)
    for m in muts:  # every mutant must still be valid Python
        compile(m.source, "<m>", "exec")
    # Tests that catch every mutation => score 1.0; tests that catch none => 0.0 with a warning.
    strict = mutation_score(src, tests_pass=lambda s: False)
    assert strict.score == 1.0 and report(strict) is None
    blind = mutation_score(src, tests_pass=lambda s: True)
    assert blind.score == 0.0 and blind.survivors and "below" in report(blind)


def test_mutation_gate_blocks_undertested_selfedit(tmp_path):
    # A proposal whose "tests" never fail on mutations must be blocked when the gate is on.
    from gauge.code import KnowledgeBase, SelfProgrammer

    sp = SelfProgrammer(None, KnowledgeBase(), tmp_path / "self", root=tmp_path, mutation_check=True, mutation_min_score=0.5)

    captured = {}

    def fake_gate(prop, copy, env):
        captured["ran"] = True
        return 0.0, "gauge/x.py: 0/4 mutants killed (0%) - mutation score 0% is below 50%"

    sp._mutation_gate = fake_gate  # avoid spinning up a real pytest subprocess in the unit test
    from gauge.code.selfmod import Proposal
    prop = Proposal(id="p1", instruction="x", files={"gauge/x.py": "def f():\n    return 1 == 1\n"})

    # Monkeypatch the subprocess-based normal test run to "pass", so the gate is what decides.
    import gauge.code.selfmod as sm
    real_run = sm.subprocess.run

    class _OK:
        returncode = 0
        stdout = ""
        stderr = ""

    sm.subprocess.run = lambda *a, **k: _OK()
    try:
        ok, rep = sp.test(prop)
    finally:
        sm.subprocess.run = real_run
    assert captured.get("ran") and ok is False and "below" in rep and prop.mutation_score == 0.0


def test_cli_code_lang_and_binary(tmp_path):
    data = str(tmp_path / "data")
    GaugeConfig(data_dir=data).save()
    py = tmp_path / "s.py"
    py.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", data, "code", "lang", str(py)])
    assert r.exit_code == 0 and "language=python" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "code", "binary", str(py), "--bytes", "32"])
    assert r.exit_code == 0 and "entropy=" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "code", "disasm", str(py)])
    assert r.exit_code == 0 and "instructions" in r.output, r.output
