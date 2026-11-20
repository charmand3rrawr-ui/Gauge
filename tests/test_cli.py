from click.testing import CliRunner

from gauge.cli import main


def test_cli_init_doctor_report_score(tmp_path, chapter_text):
    runner = CliRunner()
    data = str(tmp_path / "data")
    r = runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "data" / "config.json").exists()
    r = runner.invoke(main, ["--data-dir", data, "doctor"])
    assert r.exit_code == 0 and "fake llm" in r.output
    r = runner.invoke(main, ["--data-dir", data, "report"])
    assert r.exit_code == 0 and "No model yet" in r.output
    f = tmp_path / "ch.txt"
    f.write_text(chapter_text(True), encoding="utf-8")
    r = runner.invoke(main, ["--data-dir", data, "score", str(f)])
    assert r.exit_code == 0 and "hook=" in r.output
    r = runner.invoke(main, ["--data-dir", data, "projects"])
    assert r.exit_code == 0 and "no projects" in r.output


def test_cli_import_analyze_train(tmp_path, chapter_text):
    runner = CliRunner()
    data = str(tmp_path / "data")
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    import json

    for i in range(8):
        d = tmp_path / f"novel{i}"
        d.mkdir()
        hooky = i % 2 == 0
        for c in range(4):
            (d / f"c{c}.txt").write_text(chapter_text(hooky), encoding="utf-8")
        (d / "novel.json").write_text(json.dumps({"title": f"Novel {i}", "followers": 5000 if hooky else 100}))
        r = runner.invoke(main, ["--data-dir", data, "import", str(d)])
        assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--data-dir", data, "analyze"])
    assert r.exit_code == 0 and "chapters_featurized" in r.output
    r = runner.invoke(main, ["--data-dir", data, "train"])
    assert r.exit_code == 0, r.output
    assert "target=log_followers" in r.output
    r = runner.invoke(main, ["--data-dir", data, "report", "--novel", "local:novel-0"])
    assert r.exit_code == 0 and "predicted" in r.output
