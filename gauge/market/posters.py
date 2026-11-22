"""Official-API posters for the author's own social accounts — dry-run by default.

Each adapter publishes to one destination the author owns, through that platform's sanctioned API,
using credentials the author supplies in the environment (never written to config). Every adapter runs
the owned-account guard on each post and defaults to a dry run, so nothing is sent until the author
passes real credentials and explicitly asks to send. There is nothing here for other people's accounts,
engagement, replies, DMs, or evading a filter — a poster can only put the author's own words on the
author's own account.

Credentials (environment variables):
  Mastodon : GAUGE_MASTODON_URL, GAUGE_MASTODON_TOKEN
  Bluesky  : GAUGE_BLUESKY_APP_PASSWORD   (handle comes from the account)
  Telegram : GAUGE_TELEGRAM_BOT_TOKEN, GAUGE_TELEGRAM_CHAT_ID   (your own channel)
  Discord  : GAUGE_DISCORD_WEBHOOK_URL    (a webhook on your own server)
  Webhook  : GAUGE_WEBHOOK_URL            (any endpoint you control)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .social import OwnedAccount, Poster, SocialPost

_TIMEOUT = 20


@dataclass
class _Base:
    account: OwnedAccount
    dry_run: bool = True
    note: str = ""

    def post(self, post: SocialPost) -> dict[str, Any]:
        ok, why = post.account.postable()
        if not ok:
            return {"posted": False, "refused": why}
        from ..dryrun import hold, is_dry_run
        if not self.dry_run and is_dry_run():   # the global switch outranks --send
            hold("social posting", f"{post.account.platform} {post.account.handle}: {post.text[:60]}")
            return {"posted": False, "dry_run": True, "platform": post.account.platform,
                    "handle": post.account.handle, "angle": post.angle, "preview": post.text[:200],
                    "note": "held by the global dry run"}
        if self.dry_run:
            return {"posted": False, "dry_run": True, "platform": post.account.platform,
                    "handle": post.account.handle, "angle": post.angle, "preview": post.text[:200], "note": self.note}
        try:
            return self._send(post)
        except Exception as e:  # never let a network error escape as a crash
            return {"posted": False, "error": f"{type(e).__name__}: {e}", "platform": post.account.platform}

    def _send(self, post: SocialPost) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class MastodonPoster(_Base):
    base_url: str = ""
    token: str = ""

    def _send(self, post: SocialPost) -> dict[str, Any]:
        import requests
        r = requests.post(f"{self.base_url.rstrip('/')}/api/v1/statuses", data={"status": post.text},
                          headers={"Authorization": f"Bearer {self.token}"}, timeout=_TIMEOUT)
        r.raise_for_status()
        return {"posted": True, "platform": "mastodon", "id": r.json().get("id"), "handle": post.account.handle}


@dataclass
class TelegramPoster(_Base):
    bot_token: str = ""
    chat_id: str = ""

    def _send(self, post: SocialPost) -> dict[str, Any]:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                          json={"chat_id": self.chat_id, "text": post.text}, timeout=_TIMEOUT)
        r.raise_for_status()
        return {"posted": True, "platform": "telegram", "chat_id": self.chat_id}


@dataclass
class DiscordWebhookPoster(_Base):
    webhook_url: str = ""

    def _send(self, post: SocialPost) -> dict[str, Any]:
        import requests
        r = requests.post(self.webhook_url, json={"content": post.text}, timeout=_TIMEOUT)
        r.raise_for_status()
        return {"posted": True, "platform": "discord", "handle": post.account.handle}


@dataclass
class WebhookPoster(_Base):
    url: str = ""

    def _send(self, post: SocialPost) -> dict[str, Any]:
        import requests
        r = requests.post(self.url, json={"text": post.text, "platform": post.account.platform,
                                          "handle": post.account.handle, "angle": post.angle}, timeout=_TIMEOUT)
        r.raise_for_status()
        return {"posted": True, "platform": post.account.platform, "handle": post.account.handle}


@dataclass
class BlueskyPoster(_Base):
    app_password: str = ""
    pds: str = "https://bsky.social"

    def _send(self, post: SocialPost) -> dict[str, Any]:
        from datetime import datetime, timezone

        import requests
        sess = requests.post(f"{self.pds}/xrpc/com.atproto.server.createSession",
                             json={"identifier": self.account.handle, "password": self.app_password}, timeout=_TIMEOUT)
        sess.raise_for_status()
        s = sess.json()
        rec = {"repo": s["did"], "collection": "app.bsky.feed.post",
               "record": {"$type": "app.bsky.feed.post", "text": post.text[:300],
                          "createdAt": datetime.now(timezone.utc).isoformat()}}
        r = requests.post(f"{self.pds}/xrpc/com.atproto.repo.createRecord", json=rec,
                          headers={"Authorization": f"Bearer {s['accessJwt']}"}, timeout=_TIMEOUT)
        r.raise_for_status()
        return {"posted": True, "platform": "bluesky", "uri": r.json().get("uri"), "handle": post.account.handle}


# Platforms with clean official posting APIs we ship an adapter for. Others (X/Twitter, Meta, LinkedIn,
# Reddit) need the author's own registered OAuth app; a WebhookPoster to a service the author runs
# (e.g. their own automation that holds those tokens) covers them without Gauge holding the keys.
BUILT_IN = {"mastodon", "telegram", "discord", "bluesky", "webhook"}


def make_poster(account: OwnedAccount, dry_run: bool = True) -> Poster:
    """Build the adapter for an owned account from environment credentials. Missing credentials -> a
    dry-run poster that explains what is needed, so nothing is ever sent half-configured."""
    p = account.platform.lower()
    env = os.environ
    if p == "mastodon" and env.get("GAUGE_MASTODON_URL") and env.get("GAUGE_MASTODON_TOKEN"):
        return MastodonPoster(account, dry_run, base_url=env["GAUGE_MASTODON_URL"], token=env["GAUGE_MASTODON_TOKEN"])
    if p == "telegram" and env.get("GAUGE_TELEGRAM_BOT_TOKEN") and env.get("GAUGE_TELEGRAM_CHAT_ID"):
        return TelegramPoster(account, dry_run, bot_token=env["GAUGE_TELEGRAM_BOT_TOKEN"], chat_id=env["GAUGE_TELEGRAM_CHAT_ID"])
    if p == "discord" and env.get("GAUGE_DISCORD_WEBHOOK_URL"):
        return DiscordWebhookPoster(account, dry_run, webhook_url=env["GAUGE_DISCORD_WEBHOOK_URL"])
    if p == "bluesky" and env.get("GAUGE_BLUESKY_APP_PASSWORD"):
        return BlueskyPoster(account, dry_run, app_password=env["GAUGE_BLUESKY_APP_PASSWORD"])
    if p == "webhook" and env.get("GAUGE_WEBHOOK_URL"):
        return WebhookPoster(account, dry_run, url=env["GAUGE_WEBHOOK_URL"])
    note = (f"no credentials for {account.platform}; set its environment variables (see gauge.market.posters) "
            f"or use a webhook to a service you run" if p in BUILT_IN else
            f"{account.platform} has no built-in adapter; register your own app and post via a webhook you control")
    return _Base(account, dry_run=True, note=note)


@dataclass
class RoutingPoster:
    """Routes each post to the adapter for its account's platform."""
    dry_run: bool = True
    _cache: dict[str, Poster] = field(default_factory=dict)

    def post(self, post: SocialPost) -> dict[str, Any]:
        key = f"{post.account.platform}:{post.account.handle}"
        if key not in self._cache:
            self._cache[key] = make_poster(post.account, self.dry_run)
        return self._cache[key].post(post)
