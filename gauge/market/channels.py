"""Where authors legitimately share a web novel, and how to do it without getting banned.

This is static guidance, not automation: each entry names a kind of venue, summarises its
self-promotion rules, and says what disclosure it expects. Gauge never posts to any of these; it hands
the author the rules so they can post their own work correctly. Communities change their rules, so this
is a starting point, not a substitute for reading a venue's current pinned rules.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Channel:
    key: str
    name: str
    kind: str          # platform | forum | community | direct
    audience: str
    rules: str         # the self-promotion rules, summarised
    disclosure: str    # what to disclose
    do_it_right: str   # the one thing that keeps it welcome rather than spam


CHANNELS: list[Channel] = [
    Channel("royalroad", "Royal Road", "platform", "web-serial readers, LitRPG/progression/fantasy",
            "Post your own fiction; use the forums' designated self-promo threads, not general discussion. Consistent "
            "release schedule matters more than promotion.",
            "You are the author.", "Publish chapters on a steady schedule and let the platform's own rising-stars surface you; forum spam backfires."),
    Channel("reddit_genre", "Genre subreddits (r/litrpg, r/HFY, r/fantasywriters)", "community", "genre readers",
            "Most subs cap self-promotion (often a 1-in-10 ratio of self to community posts) and require flair. Read the "
            "sidebar and pinned rules first; many have a weekly self-promo thread.",
            "Flair the post as self-promo / author; say you wrote it.",
            "Contribute to the community first; post your work only where and how the sub allows, in the right thread."),
    Channel("discord", "Discord writing & reader servers", "community", "writers and readers",
            "Only in the server's promo channel, never in general chat or via DMs. One post, not repeated.",
            "Introduce yourself as the author.", "Join, take part, and share once in the promo channel — not drive-by."),
    Channel("author_newsletter", "Your own mailing list / newsletter", "direct", "readers who opted in",
            "People who chose to hear from you. No rules but your own; honour unsubscribes.",
            "It's from you; it's obvious.", "Grow it with a link in your chapters and profile; it is the one channel you own."),
    Channel("writing_forums", "Writing forums (Scribophile, Absolute Write, genre forums)", "forum", "writers, some readers",
            "Signatures and designated showcase boards are fine; thread spam is not. Give feedback to get feedback.",
            "Author, in your signature or the showcase board.", "Be a real member; let your signature link do the quiet work."),
    Channel("social", "Your own social accounts", "direct", "your followers",
            "Your own profiles. Follow each platform's rules on frequency and hashtags; no buying followers or engagement.",
            "It's your account.", "Share progress and snippets, not just 'read my book'; let people care about the story."),
]

_BY_KEY = {c.key: c for c in CHANNELS}


def channel(key: str) -> Channel | None:
    return _BY_KEY.get((key or "").strip().lower())


def channels_for(genres: list[str] | None = None) -> list[Channel]:
    """All channels (genre fit is advisory; everything here works for any serial)."""
    return list(CHANNELS)
