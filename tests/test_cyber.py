"""Batch D: the graduated active-defence dial and the defensive cyber-range. All inward; no outward offence."""
from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.security.audit import AuditLog
from gauge.security.defense import HONEYTOKEN_MARKER, ActiveDefense, level_for_threat, posture
from gauge.security.range import FIXTURES, run_range


def test_posture_scales_with_level():
    assert posture(0).tarpit_seconds == 0.0 and not posture(0).honeytokens
    assert posture(3).tarpit_seconds >= posture(1).tarpit_seconds
    assert posture(2).honeytokens and posture(2).decoys and posture(1).evidence
    assert level_for_threat(0) == 0 and level_for_threat(3) == 3 and level_for_threat(1, floor=2) == 2


def test_active_defense_honeytokens_and_evidence(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    ad = ActiveDefense(tmp_path / "sec", level=2, audit=audit)
    tok = ad.plant_honeytoken("api-key")
    assert tok.startswith(HONEYTOKEN_MARKER) and len(ad.honeytokens()) == 1
    # A planted token appearing in observed traffic is an exfiltration alert.
    assert ad.scan_for_leak(f"POST body: key={tok}") == [tok]
    assert ad.scan_for_leak("nothing secret here") == []
    # Evidence capture is on-disk only and recorded.
    path = ad.capture("probe", "8.8.8.8", {"tried": "oversized request"})
    assert path and path.exists() and ad.evidence()
    draft = ad.abuse_report_draft("8.8.8.8")
    assert "8.8.8.8" in draft and "No action was taken against the source" in draft
    kinds = {e["kind"] for e in audit.read(min_severity=1)}
    assert "honeytoken_tripped" in kinds and "evidence_captured" in kinds


def test_active_defense_level_0_captures_nothing(tmp_path):
    ad = ActiveDefense(tmp_path / "sec", level=0)
    assert ad.tarpit_seconds() == 0.0
    assert ad.capture("probe", "peer", "x") is None  # evidence off at level 0


def test_cyber_range_catches_all_fixtures(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    rep = run_range(max_body=256 * 1024, audit=audit)
    assert rep.total == len(FIXTURES)
    # Every fixture should be caught by its defence now that the gaps are hardened.
    assert rep.gaps == [], f"uncaught: {[g.id for g in rep.gaps]}"
    assert rep.score == 1.0
    agg = rep.by_defence()
    assert agg["sanitizer"][0] == agg["sanitizer"][1] and agg["codescan"][0] == agg["codescan"][1]
    assert any(e["kind"] == "range_run" for e in audit.read(min_severity=1))


def test_range_detects_specific_attacks():
    rep = run_range()
    by_id = {r.id: r for r in rep.results}
    assert by_id["inj-roleplay"].detected  # the jailbreak/roleplay hardening
    assert by_id["code-exfil"].detected    # the env-over-network hardening
    assert by_id["peer-public"].detected   # a genuinely public peer is refused
    assert by_id["pickle-payload"].detected  # pickle is never unpacked from the network


def test_cli_security_range_and_defense(tmp_path):
    data = str(tmp_path / "data")
    GaugeConfig(data_dir=data).save()
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", data, "security", "range"])
    assert r.exit_code == 0 and "Cyber-range" in r.output and "score 100%" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "security", "defense", "--plant", "api-key"])
    assert r.exit_code == 0 and "Active defence" in r.output and "planted" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "security", "defense"])
    assert r.exit_code == 0 and "honeytoken(s) planted" in r.output, r.output
