"""Batch H: the legitimate launch kit — assets, channel guidance, disclosed plan, and the policy gate."""
from click.testing import CliRunner

from gauge.cli import main
from gauge.config import GaugeConfig
from gauge.llm import FakeLLM
from gauge.market import CHANNELS, build_launch_kit, channel, check_action, posting_plan
from gauge.market.launchkit import LaunchKit
from gauge.market.policy import DISCLOSURE, REFUSED, policy_text


def test_policy_gate_refuses_spam_and_allows_assets():
    for bad in ("create_account", "auto_post", "mass_dm", "scrape_contacts", "sockpuppet", "fake_reviews"):
        ok, reason = check_action(bad)
        assert ok is False and reason
    for good in ("generate_assets", "draft_post", "plan", "channel_guidance"):
        assert check_action(good)[0] is True
    assert check_action("anything_else")[0] is False  # default-deny
    assert set(REFUSED) >= {"create_account", "auto_post", "mass_dm"}
    assert "will not do" in policy_text().lower()


class _Proj:
    class state:
        name = "Ten Seconds"
        genres = ["Sci-fi"]

    def bible_markdown(self):
        return "# Bible\nA courier who can rewind ten seconds."


def test_build_launch_kit():
    kit = build_launch_kit(FakeLLM(), _Proj())
    assert kit.title == "Ten Seconds" and kit.tagline and kit.blurb_short and kit.tags
    md = kit.to_markdown()
    assert "Launch kit" in md and "Short blurb" in md


def test_posting_plan_is_disclosed_and_author_executed():
    kit = LaunchKit(title="Ten Seconds", tagline="Ten seconds is all she has.",
                    blurb_short="Rin can rewind ten seconds, at the cost of a memory.", author_note="Weekly chapters.")
    plan = posting_plan(kit, url="https://example.com/s")
    # Every venue appears, every draft carries the disclosure, and the plan is explicit that Gauge won't post.
    assert all(ch.name in plan for ch in CHANNELS)
    assert plan.count(DISCLOSURE) >= len(CHANNELS)
    assert "does not and will not post for you or create accounts" in plan
    assert "https://example.com/s" in plan


def test_channels_have_rules_and_disclosure():
    assert channel("royalroad") is not None and channel("nope") is None
    for ch in CHANNELS:
        assert ch.rules and ch.disclosure and ch.do_it_right


def test_cli_market(tmp_path):
    data = str(tmp_path / "data")
    cfg = GaugeConfig(data_dir=data)
    cfg.llm.provider = "fake"
    cfg.save()
    from gauge.write.project import Project
    Project.create(cfg.projects_dir, "Ten Seconds", "a courier who can rewind", ["Sci-fi"], 300)
    runner = CliRunner(env={"COLUMNS": "200"})
    r = runner.invoke(main, ["--data-dir", data, "market", "policy"])
    assert r.exit_code == 0 and "will not do" in r.output.lower() and "create_account" in r.output
    r = runner.invoke(main, ["--data-dir", data, "market", "channels"])
    assert r.exit_code == 0 and "Royal Road" in r.output
    r = runner.invoke(main, ["--data-dir", data, "market", "kit", "Ten Seconds"])
    assert r.exit_code == 0 and "Launch kit" in r.output and (cfg.projects_dir / "ten-seconds" / "launch_kit.md").exists() or "saved" in r.output
    r = runner.invoke(main, ["--data-dir", data, "market", "plan", "Ten Seconds", "--url", "https://example.com/s"])
    assert r.exit_code == 0 and "post them yourself" in r.output
