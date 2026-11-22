"""The global dry-run switch holds every outward-facing or hard-to-undo action (roadmap 97)."""
import os

import pytest
from click.testing import CliRunner

from gauge import dryrun
from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.db import Database
from gauge.market.email_list import DryRunMailer, Newsletter, Sender, send_newsletter
from gauge.market.posters import _Base
from gauge.market.social import OwnedAccount, SocialPost
from gauge.ops.backup import make_backup, restore_backup


@pytest.fixture
def dry(monkeypatch):
    monkeypatch.setenv(dryrun.ENV, "1")
    dryrun.reset()
    yield
    dryrun.reset()


@pytest.fixture
def db(tmp_path):
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.ensure_dirs()
    return Database(cfg.db_path)


class _LivePoster(_Base):
    def _send(self, post):
        return {"posted": True, "platform": post.account.platform}


class _LiveMailer:
    def __init__(self):
        self.sent = []

    def send(self, to, subject, body, sender):
        self.sent.append(to)
        return {"sent": True, "to": to}


def _post():
    acct = OwnedAccount(platform="mastodon", handle="@me")
    return SocialPost(account=acct, angle="update", text="New chapter is up.", day=0)


def test_switch_reads_the_environment(monkeypatch):
    monkeypatch.delenv(dryrun.ENV, raising=False)
    assert not dryrun.is_dry_run()
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv(dryrun.ENV, v)
        assert dryrun.is_dry_run()
    monkeypatch.setenv(dryrun.ENV, "0")
    assert not dryrun.is_dry_run()


def test_live_poster_sends_normally_and_is_held_in_dry_run(monkeypatch):
    monkeypatch.delenv(dryrun.ENV, raising=False)
    dryrun.reset()
    live = _LivePoster(_post().account, dry_run=False)
    assert live.post(_post())["posted"] is True          # no dry run: the real send happens
    monkeypatch.setenv(dryrun.ENV, "1")
    r = live.post(_post())
    assert r["posted"] is False and r["dry_run"] is True  # the global switch outranks dry_run=False
    assert dryrun.held()[0]["action"] == "social posting"
    dryrun.reset()


def test_newsletter_is_held_even_with_a_real_mailer(dry, db):
    db.add_subscriber("reader@example.com", "Reader", "2026-01-01T00:00:00", "signup form")
    mailer = _LiveMailer()
    res = send_newsletter(db, Newsletter("Chapter 5 is out", "Hello."), Sender("Jane", "jane@example.com", "1 Road, Town"),
                          "https://me/unsub", mailer)
    assert res["dry_run"] is True and res["sent"] == 0 and mailer.sent == []
    assert any(h["action"] == "newsletter sending" for h in dryrun.held())


def test_default_dry_mailer_is_not_reported_as_a_hold(dry, db):
    db.add_subscriber("reader@example.com", "Reader", "2026-01-01T00:00:00", "signup form")
    send_newsletter(db, Newsletter("S", "B"), Sender("Jane", "jane@example.com", "1 Road, Town"), "https://me/unsub",
                    DryRunMailer())
    assert dryrun.held() == []  # it was already a preview; nothing real was held back


def test_restore_writes_nothing_in_dry_run(dry, tmp_path):
    src = tmp_path / "data"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    os.environ.pop(dryrun.ENV)                      # make the backup for real
    archive = make_backup(src, tmp_path / "b.tar.gz").path
    os.environ[dryrun.ENV] = "1"
    dest = tmp_path / "restored"
    r = restore_backup(archive, dest)
    assert r["dry_run"] is True and r["restored"] == 0 and r["would_restore"] >= 1
    assert not dest.exists()
    assert any(h["action"] == "backup restore" for h in dryrun.held())


def test_selfmod_apply_is_held(dry, tmp_path):
    from gauge.code.selfmod import Proposal, SelfProgrammer
    sp = SelfProgrammer.__new__(SelfProgrammer)     # apply() only needs dir/root before the hold
    sp.dir, sp.root = tmp_path / "self", tmp_path / "repo"
    prop = Proposal(id="p1", instruction="x", files={"gauge/text.py": "# changed\n"}, tested=True)
    res = sp.apply(prop)
    assert res["applied"] is False and res["dry_run"] is True
    assert not (tmp_path / "repo").exists() and not (tmp_path / "self").exists()
    assert any(h["action"] == "self-modification" for h in dryrun.held())


def test_newsdesk_due_piece_stays_queued(dry, tmp_path):
    from datetime import datetime

    from gauge.newsdesk.desk import queue_for, run_cycle
    from gauge.write.project import Project
    cfg = GaugeConfig(data_dir=str(tmp_path / "data"))
    cfg.ensure_dirs()
    Project.create(cfg.projects_dir, "Site", "idea", ["Drama"], 300)
    cfg.newsdesk.project, cfg.newsdesk.byline = "Site", "Jane Doe"
    q = queue_for(cfg)
    from gauge.newsdesk.commentary import Commentary
    item = q.add(Commentary(topic="t", title="A measured take", body="Body.", byline="Jane Doe"))
    assert q.approve(item.id)
    rep = run_cycle(cfg, llm=None, topics=[], now=datetime(2026, 1, 2, 12, 0), draft=False)
    assert rep.published is None and any("dry run" in n for n in rep.notes)
    assert q.items("approved")                       # still approved, not published
    assert not (Project(cfg.projects_dir, "Site").dir / "articles").exists() or \
        not list((Project(cfg.projects_dir, "Site").dir / "articles").glob("*.json"))


def test_cli_flag_announces_summarises_and_does_not_leak(tmp_path, monkeypatch):
    monkeypatch.delenv(dryrun.ENV, raising=False)
    data = str(tmp_path / "data")
    r = CliRunner(env={"COLUMNS": "200"}).invoke(main, ["--data-dir", data, "--dry-run", "audit"])
    assert r.exit_code == 0 and "DRY RUN" in r.output and "Dry run: ACTIVE" in r.output
    assert "nothing needed holding back" in r.output
    assert dryrun.ENV not in os.environ              # restored after the command


def test_config_switch_turns_it_on(tmp_path, monkeypatch):
    monkeypatch.delenv(dryrun.ENV, raising=False)
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.dry_run = True
    cfg.save()
    r = CliRunner(env={"COLUMNS": "200"}).invoke(main, ["--data-dir", data, "audit"])
    assert r.exit_code == 0 and "DRY RUN" in r.output
