"""The promotion policy: what Gauge will and will not do to market a novel, enforced in code.

Marketing a book is legitimate. Spamming communities with bot-created accounts is not, and the
difference is not a matter of intensity — it is the difference between an author sharing their work and
a deception run against a platform and its readers. Gauge helps with the first and is structurally
incapable of the second: there is no account-creation tool, no third-party posting tool, and no
contact-scraping tool anywhere in this package. `check_action` is the single gate, so the boundary is
one function, visible and testable, not a promise buried in prose.
"""
from __future__ import annotations

# Things Gauge does: produce material and plans the author acts on themselves.
ALLOWED = {
    "generate_assets": "write blurbs, taglines, loglines, tags and an author's note",
    "draft_post": "draft a post for a specific community, clearly marked as the author's own promotion",
    "plan": "produce a checklist of where and how to share, with each community's rules and disclosure",
    "channel_guidance": "explain a community's self-promotion rules",
    "schedule_own": "schedule and publish posts to accounts the author owns, via the platform's official API, disclosed",
    "email_optin": "send to a mailing list of people who signed up, with sender identity and an unsubscribe link in every message",
    "publish_own_site": "publish the author's own fiction and labelled opinion to a site, under a disclosed byline",
    "opinion_labeled": "draft commentary clearly labelled as opinion, with a disclosed byline, that the author stands behind",
    "honest_headline": "write a strong, accurate headline and flag clickbait that misrepresents the content",
    "trend_discovery": "find trending topics from sources that permit it (Wikipedia pageviews, RSS you choose), honouring robots.txt",
    "translate": "translate source coverage for research",
    "original_commentary": "write original, labelled commentary on a topic that cites and links its sources",
}

# Things Gauge refuses, with the reason. These have no implementation here by design.
REFUSED = {
    "create_account": "creating accounts programmatically defeats the sign-up controls (CAPTCHA, verification) "
                      "that exist to stop spam, and violates essentially every platform's terms",
    "auto_post": "posting to third-party platforms from automation or fake identities is spam, and undisclosed "
                 "promotion is astroturfing; it gets the book banned and deceives the community",
    "mass_dm": "messaging strangers in bulk is unsolicited spam, however it is worded",
    "scrape_contacts": "harvesting people 'with interest' to target them is surveillance of private individuals",
    "sockpuppet": "posing as a reader or multiple people to praise the book deceives everyone it reaches",
    "fake_reviews": "fabricated reviews or ratings are fraud and are detectable, harming the author's reputation",
    "buy_engagement": "bought or botted likes, follows and views are fraud and violate every platform's terms",
    "reply_bot": "auto-replying into strangers' threads to inject the book is spam, however it is worded",
    "cross_community_post": "auto-posting into communities or accounts the author does not own is spam",
    "evade_spam_detection": "varying content to slip past a platform's abuse detection is defeating its controls",
    "email_nonconsent": "emailing people who did not sign up is spam and breaks anti-spam law (CAN-SPAM, GDPR, CASL)",
    "purchase_list": "bought, scraped or harvested email lists are never consented; sending to them is spam",
    "fabricate_news": "generating news-shaped content with invented facts, quotes or events is misinformation",
    "clickbait_deception": "a title or thumbnail that misrepresents the content to extract a click is deception",
    "impersonate_outlet": "writing as a real outlet or a shared pseudonym (e.g. a ZeroHedge 'Tyler Durden' byline) hides who is speaking",
    "rewrite_republish": "paraphrase-spinning other outlets' articles to republish as your own is plagiarism and, at scale, "
                         "copyright infringement — the content-farm mechanic",
    "auto_publish_unreviewed_news": "publishing checkable factual claims with no human review, on a timer, is a misinformation pipeline",
}

# Every drafted post carries this, so promotion is never passed off as something else.
DISCLOSURE = "Posted by the author. This is self-promotion of my own web novel."


def check_action(action: str) -> tuple[bool, str]:
    """The one gate. Returns (allowed, reason). Unknown actions are refused by default."""
    key = (action or "").strip().lower()
    if key in ALLOWED:
        return True, ALLOWED[key]
    if key in REFUSED:
        return False, REFUSED[key]
    return False, "unknown action; Gauge only generates assets and plans the author executes themselves"


def policy_text() -> str:
    lines = ["Gauge's promotion policy", "", "Will do:"]
    lines += [f"  - {k}: {v}" for k, v in ALLOWED.items()]
    lines += ["", "Will not do (no implementation exists for these):"]
    lines += [f"  - {k}: {v}" for k, v in REFUSED.items()]
    lines += ["", f"Every drafted post is labelled: \"{DISCLOSURE}\""]
    return "\n".join(lines)
