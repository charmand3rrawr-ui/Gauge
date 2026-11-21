"""Automated PR for accounts the author owns — a content calendar, a disclosed queue, a dry run.

This is social-media management of your *own* accounts, the way a scheduling tool works: it plans
distinct posts and, through a platform's official API, publishes them to a destination you own. The
variety between posts is about *angle* (a teaser reads differently from a milestone), because repeating
the identical post is low-quality marketing — it is explicitly NOT variation to evade a spam filter,
which the policy refuses. The guard will only ever post to an account the caller affirms it owns and
that uses an official API, and there is no code here for engagement, replies, DMs, or other people's
communities. No live network poster ships: a real one is a thin adapter the author wires with their own
credentials, and it inherits the same guard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .launchkit import LaunchKit
from .policy import check_action


@dataclass
class OwnedAccount:
    platform: str
    handle: str
    official_api: bool = True   # automation must go through the platform's sanctioned API
    i_own_this: bool = True     # the caller affirms this is the author's own account

    def postable(self) -> tuple[bool, str]:
        if not self.i_own_this:
            return False, f"{self.platform}:{self.handle} is not an account you own; Gauge posts only to your own accounts"
        if not self.official_api:
            return False, f"{self.platform}:{self.handle} would not use the official API; Gauge will not automate around a platform's rules"
        return True, "ok"


# Angle -> how to build a distinct post from the launch kit. Variety for quality, never for evasion.
def _angle_teaser(kit: LaunchKit, url: str) -> str:
    return f"{kit.tagline}\n\n{kit.blurb_short}" + (f"\n{url}" if url else "")


def _angle_quote(kit: LaunchKit, url: str) -> str:
    line = (kit.blurb_short.split(".")[0] + ".") if kit.blurb_short else kit.tagline
    return f"“{line}”\n\nFrom {kit.title}." + (f" {url}" if url else "")


def _angle_logline(kit: LaunchKit, url: str) -> str:
    return f"{kit.title}: {kit.logline}" + (f"\n{url}" if url else "")


def _angle_invitation(kit: LaunchKit, url: str) -> str:
    note = kit.author_note or "New chapters regularly."
    return f"If you like {', '.join(kit.tags[:2]) or 'web serials'}, {kit.title} might be for you. {note}" + (f"\n{url}" if url else "")


ANGLES = [("teaser", _angle_teaser), ("quote", _angle_quote), ("logline", _angle_logline), ("invitation", _angle_invitation)]


@dataclass
class SocialPost:
    account: OwnedAccount
    angle: str
    text: str
    day: int  # days from the start of the plan


def content_calendar(kit: LaunchKit, accounts: list[OwnedAccount], weeks: int = 4, per_week: int = 3,
                     url: str = "") -> list[SocialPost]:
    """A schedule of distinct posts across the author's own accounts. Angles cycle so no two posts in a
    row are the same shape. Posting to a non-owned or non-API account is refused up front."""
    for acc in accounts:
        ok, why = acc.postable()
        if not ok:
            raise PermissionError(why)
    if not accounts:
        return []
    posts: list[SocialPost] = []
    slots = max(1, weeks) * max(1, per_week)
    spacing = max(1, (max(1, weeks) * 7) // max(1, per_week * max(1, weeks)))
    for i in range(slots):
        angle_name, builder = ANGLES[i % len(ANGLES)]
        acc = accounts[i % len(accounts)]
        posts.append(SocialPost(acc, angle_name, builder(kit, url), day=i * spacing))
    return posts


class Poster(Protocol):
    def post(self, post: SocialPost) -> dict[str, Any]:  # pragma: no cover - interface
        ...


@dataclass
class DryRunPoster:
    """The default: records what WOULD be posted and sends nothing. Safe to run anywhere."""
    sent: list[SocialPost] = field(default_factory=list)

    def post(self, post: SocialPost) -> dict[str, Any]:
        self.sent.append(post)
        return {"posted": False, "dry_run": True, "platform": post.account.platform, "handle": post.account.handle,
                "angle": post.angle, "preview": post.text[:200]}


def publish(posts: list[SocialPost], poster: Optional[Poster] = None) -> list[dict[str, Any]]:
    """Run the queue through a poster (DryRunPoster by default). The owned+official guard is enforced
    per post, so even a real adapter can only reach destinations the author affirmed they own."""
    allowed, _ = check_action("schedule_own")
    if not allowed:  # pragma: no cover - schedule_own is always allowed, but fail safe
        raise PermissionError("scheduling own posts is not permitted")
    poster = poster or DryRunPoster()
    results: list[dict[str, Any]] = []
    for post in posts:
        ok, why = post.account.postable()
        if not ok:
            results.append({"posted": False, "refused": why})
            continue
        results.append(poster.post(post))
    return results


def calendar_markdown(posts: list[SocialPost]) -> str:
    out = ["# Social content calendar", "",
           "Posts to your own accounts, via each platform's official API. Review them, then publish on your "
           "own schedule. Gauge will not post to accounts you do not own or vary content to dodge spam filters.", ""]
    for p in posts:
        out += [f"## Day {p.day} — {p.account.platform} (@{p.account.handle}) — {p.angle}", "", "```", p.text, "```", ""]
    return "\n".join(out)
