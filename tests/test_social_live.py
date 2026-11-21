"""Batch J: official-API poster adapters (owned-only, dry-run) and the opt-in email list."""
import pytest
from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.db import Database
from gauge.market.email_list import (
    EMAIL_RE,
    DryRunMailer,
    Newsletter,
    Sender,
    build_newsletter,
    import_subscribers,
    render_message,
    send_newsletter,
    unsubscribe,
)
from gauge.market.posters import MastodonPoster, RoutingPoster, make_poster
from gauge.market.social import OwnedAccount, SocialPost


def test_make_poster_without_credentials_is_dry(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("GAUGE_MASTODON") or k.startswith("GAUGE_TELEGRAM") or k.startswith("GAUGE_DISCORD") or k.startswith("GAUGE_BLUESKY") or k == "GAUGE_WEBHOOK_URL":
            monkeypatch.delenv(k, raising=False)
    p = make_poster(OwnedAccount("mastodon", "me"), dry_run=False)
    # No credentials -> a base dry-run poster that explains what is missing; never sends.
    res = p.post(SocialPost(OwnedAccount("mastodon", "me"), "teaser", "hi", 0))
    assert res.get("dry_run") is True and "credentials" in res.get("note", "")


def test_poster_refuses_unowned_account():
    p = MastodonPoster(OwnedAccount("mastodon", "notmine", i_own_this=False), dry_run=False, base_url="https://m", token="t")
    res = p.post(SocialPost(OwnedAccount("mastodon", "notmine", i_own_this=False), "teaser", "hi", 0))
    assert res["posted"] is False and "own" in res["refused"]


def test_make_poster_builds_adapter_with_env(monkeypatch):
    monkeypatch.setenv("GAUGE_MASTODON_URL", "https://mastodon.example")
    monkeypatch.setenv("GAUGE_MASTODON_TOKEN", "secret")
    p = make_poster(OwnedAccount("mastodon", "me"), dry_run=True)
    assert isinstance(p, MastodonPoster) and p.base_url == "https://mastodon.example"
    # Still dry-run: builds the adapter but sends nothing.
    assert p.post(SocialPost(OwnedAccount("mastodon", "me"), "teaser", "hi", 0))["dry_run"] is True


def test_routing_poster_dispatches_by_platform():
    rp = RoutingPoster(dry_run=True)
    r1 = rp.post(SocialPost(OwnedAccount("mastodon", "me"), "teaser", "a", 0))
    r2 = rp.post(SocialPost(OwnedAccount("telegram", "me"), "quote", "b", 0))
    assert r1["platform"] == "mastodon" and r2["platform"] == "telegram"
    assert not r1.get("posted") and not r2.get("posted")


def test_import_requires_consent():
    db = Database(":memory:")
    summary = import_subscribers(db, [
        {"email": "yes@example.com", "name": "Yes", "consent_at": "2026-01-01"},
        {"email": "flag@example.com", "opt_in": "true"},
        {"email": "nope@example.com", "name": "NoConsent"},           # refused: no consent
        {"email": "not-an-email", "consent_at": "2026-01-01"},        # refused: invalid
    ])
    assert summary["added"] == 2 and summary["no_consent"] == 1 and summary["invalid_email"] == 1
    emails = {s["email"] for s in db.subscribers()}
    assert "nope@example.com" not in emails and "yes@example.com" in emails


def test_unsubscribe_is_permanent():
    db = Database(":memory:")
    import_subscribers(db, [{"email": "a@b.com", "consent_at": "2026-01-01"}])
    assert unsubscribe(db, "a@b.com") is True
    # A later import must NOT resurrect an unsubscribed address.
    import_subscribers(db, [{"email": "a@b.com", "consent_at": "2026-02-01"}])
    assert db.subscriber_counts().get("subscribed", 0) == 0
    assert db.subscriber_counts().get("unsubscribed", 0) == 1


def test_send_only_to_subscribed_with_disclosure():
    db = Database(":memory:")
    import_subscribers(db, [{"email": "a@b.com", "consent_at": "2026-01-01"},
                            {"email": "c@d.com", "consent_at": "2026-01-02"}])
    unsubscribe(db, "c@d.com")
    nl = Newsletter("New chapter", "Chapter 5 is up.")
    mailer = DryRunMailer()
    res = send_newsletter(db, nl, Sender("Rin", "rin@me.com", "123 St"), "https://me/unsub", mailer)
    assert res["recipients"] == 1 and res["dry_run"] is True   # only the still-subscribed one
    assert mailer.sent and mailer.sent[0]["to"] == "a@b.com"
    msg = render_message(nl, "a@b.com", Sender("Rin", "rin@me.com", "123 St"), "https://me/unsub")
    assert "Unsubscribe: https://me/unsub/a@b.com" in msg and "Rin" in msg and "123 St" in msg


def test_send_refuses_without_sender_or_unsubscribe():
    db = Database(":memory:")
    import_subscribers(db, [{"email": "a@b.com", "consent_at": "2026-01-01"}])
    nl = Newsletter("s", "b")
    with pytest.raises(PermissionError):
        send_newsletter(db, nl, Sender("", ""), "https://me/unsub")       # no sender
    with pytest.raises(PermissionError):
        send_newsletter(db, nl, Sender("Rin", "rin@me.com"), "")          # no unsubscribe base


def test_email_regex():
    assert EMAIL_RE.match("a@b.com") and not EMAIL_RE.match("a@b") and not EMAIL_RE.match("nope")


def test_build_newsletter():
    class _Kit:
        title = "Ten Seconds"
        tagline = "Ten seconds is all she has."
        blurb_short = "Rin can rewind ten seconds."
        author_note = "Weekly chapters."
    nl = build_newsletter(_Kit(), url="https://x/s", chapter_title="The Seal")
    assert "Ten Seconds" in nl.subject and "The Seal" in nl.subject and "https://x/s" in nl.body


def test_cli_subscribers_and_newsletter(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.llm.provider = "fake"
    cfg.save()
    from gauge.write.project import Project
    Project.create(cfg.projects_dir, "Ten Seconds", "a courier who can rewind", ["Sci-fi"], 300)
    runner = CliRunner(env={"COLUMNS": "200"})
    csv = tmp_path / "subs.csv"
    csv.write_text("email,name,consent_at\na@b.com,Al,2026-01-01\nnope@x.com,No,\nc@d.com,Cy,2026-02-02\n", encoding="utf-8")
    r = runner.invoke(main, ["--data-dir", data, "market", "subscribers", "import", str(csv)])
    assert r.exit_code == 0 and "added 2" in r.output and "no-consent 1" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "market", "newsletter", "Ten Seconds", "--url", "https://x/s", "--chapter", "The Seal"])
    assert r.exit_code == 0 and "DRY RUN" in r.output and "Unsubscribe:" in r.output and "would go to 2" in r.output, r.output
    r = runner.invoke(main, ["--data-dir", data, "market", "social", "Ten Seconds", "--account", "mastodon:rin", "--weeks", "1", "--per-week", "1"])
    assert r.exit_code == 0 and "dry run" in r.output, r.output
