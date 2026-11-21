"""Batch I: owned-account social PR — content calendar, disclosed queue, dry run, owned-only guard."""
from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.market.launchkit import LaunchKit
from gauge.market.policy import REFUSED, check_action
from gauge.market.social import (
    ANGLES,
    DryRunPoster,
    OwnedAccount,
    SocialPost,
    content_calendar,
    publish,
)

KIT = LaunchKit(title="Ten Seconds", tagline="Ten seconds is all she has.",
                logline="A courier who can rewind ten seconds, at the cost of a memory.",
                blurb_short="Rin can rewind ten seconds. Each rewind costs a memory. The debt is coming due.",
                tags=["time loop", "heist"], author_note="New chapters weekly.")


def test_policy_refuses_engagement_and_evasion():
    for bad in ("buy_engagement", "reply_bot", "cross_community_post", "evade_spam_detection"):
        assert check_action(bad)[0] is False and bad in REFUSED
    assert check_action("schedule_own")[0] is True  # posting to your own accounts is allowed


def test_owned_account_guard():
    assert OwnedAccount("mastodon", "me").postable()[0] is True
    assert OwnedAccount("x", "notme", i_own_this=False).postable()[0] is False
    assert OwnedAccount("x", "me", official_api=False).postable()[0] is False


def test_content_calendar_is_varied_and_scheduled():
    accs = [OwnedAccount("mastodon", "rin"), OwnedAccount("bluesky", "rin")]
    cal = content_calendar(KIT, accs, weeks=2, per_week=3, url="https://x/s")
    assert len(cal) == 6
    # Angles cycle so no two consecutive posts are the same shape (variety for quality, not evasion).
    assert all(cal[i].angle != cal[i + 1].angle for i in range(len(cal) - 1))
    assert {p.account.platform for p in cal} == {"mastodon", "bluesky"}
    assert all("https://x/s" in p.text for p in cal)
    assert len({p.text for p in cal}) >= len(ANGLES)  # distinct posts


def test_calendar_refuses_unowned_account():
    bad = OwnedAccount("twitter", "someone", i_own_this=False)
    try:
        content_calendar(KIT, [bad])
        raise AssertionError("should have refused a non-owned account")
    except PermissionError as e:
        assert "own" in str(e)


def test_publish_is_dry_run_by_default_and_guards_each_post():
    accs = [OwnedAccount("mastodon", "rin")]
    cal = content_calendar(KIT, accs, weeks=1, per_week=2)
    poster = DryRunPoster()
    results = publish(cal, poster)
    assert all(r["dry_run"] for r in results) and len(poster.sent) == len(cal)
    # A non-owned post slipped into the queue is refused at publish time, not sent.
    bad_post = SocialPost(OwnedAccount("x", "notmine", i_own_this=False), "teaser", "hi", 0)
    out = publish([bad_post], poster)
    assert out[0]["posted"] is False and "refused" in out[0]


def test_cli_market_social(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.llm.provider = "fake"
    cfg.save()
    from gauge.write.project import Project
    Project.create(cfg.projects_dir, "Ten Seconds", "a courier who can rewind", ["Sci-fi"], 300)
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", data, "market", "social", "Ten Seconds",
                             "--account", "mastodon:rin_author", "--weeks", "2", "--per-week", "2", "--url", "https://x/s"])
    assert r.exit_code == 0 and "0 sent" in r.output and "planned" in r.output, r.output
    assert (cfg.projects_dir / "ten-seconds" / "social_calendar.md").exists()
    # No accounts named -> it asks, posts nothing.
    r = runner.invoke(main, ["--data-dir", data, "market", "social", "Ten Seconds"])
    assert r.exit_code == 0 and "account you own" in r.output
