"""Publish the author's own work to a website: a landing page, the serial, and a labelled-opinion blog.

Everything here carries content-integrity rails: a disclosed byline that may not impersonate a real
outlet, opinion that is labelled as opinion, headlines checked for clickbait deception, and
factual-looking claims flagged for the author to verify. It builds a static site the author hosts; it
does not manufacture news.
"""
from .integrity import (
    IMPERSONATION_DENYLIST,
    OPINION_DISCLAIMER,
    check_byline,
    check_headline,
    flag_claims,
)
from .opinion import draft_opinion
from .publisher import Article, build_site

__all__ = ["check_byline", "check_headline", "flag_claims", "OPINION_DISCLAIMER", "IMPERSONATION_DENYLIST",
           "draft_opinion", "Article", "build_site"]
