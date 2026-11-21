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
