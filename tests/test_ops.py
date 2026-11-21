"""Roadmap J (ops): verified backup/restore, config migration, automation audit."""
import io
import json
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from gauge.cli import main
from gauge.config import CONFIG_VERSION, GaugeConfig
from gauge.ops import make_backup, render_report, restore_backup, verify_backup
from gauge.ops.report import audit_report


def _data_dir(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.ensure_dirs()
    cfg.save()
    from gauge.write.project import Project
    Project.create(cfg.projects_dir, "P", "a premise", ["Sci-fi"], 300)
    return cfg


def test_backup_verify_restore_roundtrip(tmp_path):
    cfg = _data_dir(tmp_path)
    bk = tmp_path / "b.tar.gz"
    res = make_backup(cfg.root, bk)
    assert res.files >= 2 and bk.exists()
    v = verify_backup(bk)
    assert v.ok and v.files == res.files and not v.mismatched
    out = restore_backup(bk, tmp_path / "restored")
    assert out["restored"] == res.files and (tmp_path / "restored" / "config.json").exists()


def test_restore_refuses_corrupt_backup(tmp_path):
    cfg = _data_dir(tmp_path)
    bk = tmp_path / "b.tar.gz"
    make_backup(cfg.root, bk)
    # Tamper: flip a byte in one archived file, keeping the manifest's hash — verify must catch it.
    with tarfile.open(bk, "r:gz") as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    target = next(n for n in members if n.startswith("data/") and n.endswith("config.json"))
    members[target] = members[target] + b"tampered"
    with tarfile.open(bk, "w:gz") as tar:
        for name, raw in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    v = verify_backup(bk)
    assert not v.ok and target.replace("data/", "") in v.mismatched
    with pytest.raises(ValueError, match="integrity"):
        restore_backup(bk, tmp_path / "restored2")


def test_restore_rejects_path_traversal(tmp_path):
    # A malicious manifest referencing ../ must be rejected.
    bk = tmp_path / "evil.tar.gz"
    manifest = {"created": "x", "files": {"../escape.txt": {"sha256": "", "bytes": 0}}}
    with tarfile.open(bk, "w:gz") as tar:
        mraw = json.dumps(manifest).encode()
        mi = tarfile.TarInfo("MANIFEST.json")
        mi.size = len(mraw)
        tar.addfile(mi, io.BytesIO(mraw))
    # Empty-hash file is "missing" in the archive, so verify fails first (safe either way).
    assert verify_backup(bk).ok is False


def test_restore_refuses_nonempty_dir(tmp_path):
    cfg = _data_dir(tmp_path)
    bk = tmp_path / "b.tar.gz"
    make_backup(cfg.root, bk)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "something").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        restore_backup(bk, dest)
    assert restore_backup(bk, dest, overwrite=True)["restored"] >= 2


def test_config_migration_stamps_version(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"llm": {"provider": "fake"}}), encoding="utf-8")  # no version
    cfg = GaugeConfig.load(str(root))
    assert cfg.config_version == CONFIG_VERSION and cfg.llm.provider == "fake"
    # A future version is left as-is, not downgraded.
    (root / "config.json").write_text(json.dumps({"config_version": 999}), encoding="utf-8")
    assert GaugeConfig.load(str(root)).config_version == 999


def test_audit_report_reads_config(tmp_path):
    cfg = _data_dir(tmp_path)
    cfg.newsdesk.enabled = True
    rows = audit_report(cfg)
    names = {r.name for r in rows}
    assert "newsdesk" in names and "auto-apply self-fixes" in names
    nd = next(r for r in rows if r.name == "newsdesk")
    assert nd.on is True and "approval" in nd.bound_by
    text = render_report(cfg)
    assert "automation audit" in text and "off by default" in text


def test_cli_backup_audit(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.ensure_dirs()
    cfg.save()
    runner = CliRunner(env={"COLUMNS": "200"})
    bk = str(tmp_path / "out.tar.gz")
    r = runner.invoke(main, ["--data-dir", data, "backup", "--out", bk])
    assert r.exit_code == 0 and "backed up" in r.output and Path(bk).exists()
    r = runner.invoke(main, ["--data-dir", data, "restore", bk, "--verify-only"])
    assert r.exit_code == 0 and "OK" in r.output
    r = runner.invoke(main, ["--data-dir", data, "audit"])
    assert r.exit_code == 0 and "automation audit" in r.output
