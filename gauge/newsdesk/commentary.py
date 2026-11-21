"""Original commentary on a trending topic — sourced, linked, and checked for copying.

The writer is handed the sources as research (headline, translated summary, link) and asked for original
analysis and opinion in the platform's stance. The originality check then measures how much of each
source's own wording was reproduced (shared five-word sequences as a share of the source); past the
threshold the draft is flagged as too close to the source. Every piece ends with a Sources section that
links the originals — commentary that credits its sources and adds its own thinking, not a rewritten copy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..llm.client import LLM, generate_json
from ..site.integrity import OPINION_DISCLAIMER, check_byline, check_headline, flag_claims
from ..site.stance import balance_check, stance_guidance
from .trends import SourceItem

_SCHEMA = {"type": "object",
           "properties": {"title": {"type": "string"}, "body": {"type": "string"}, "summary": {"type": "string"}},
           "required": ["title", "body"]}

_SYSTEM = ("You are an opinion columnist. You are given source coverage as RESEARCH. Write ORIGINAL analysis and "
           "opinion: your own structure, your own sentences, your own argument. Do not copy or closely paraphrase "
           "the sources; refer to them by outlet and let the Sources list carry the links. Do not invent facts, quotes "
           "or statistics. Engage the strongest opposing view. This is labelled opinion, not news reporting.")


def _domain(url: str) -> str:
    m = re.search(r"https?://([^/]+)", url or "")
    host = (m.group(1) if m else "").lower()
    return host[4:] if host.startswith("www.") else host


def source_diversity(sources: list) -> tuple[float, list[str]]:
    """How concentrated the sourcing is. Returns (share from the single most-used outlet, flags).
    A piece leaning entirely on one outlet is barely more than that outlet's take restated."""
    domains = [_domain(getattr(s, "url", "")) for s in sources]
    domains = [d for d in domains if d]
    if len(domains) < 2:
        return (1.0 if domains else 0.0), []
    from collections import Counter
    counts = Counter(domains)
    top_domain, top_n = counts.most_common(1)[0]
    share = top_n / len(domains)
    flags = []
    if share >= 0.75:
        flags.append(f"{share:.0%} of sources are from {top_domain}; find a second independent source before leaning on this")
    return round(share, 3), flags


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"[\w']+", (text or "").lower())
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def originality(text: str, sources: list[SourceItem], n: int = 5) -> tuple[float, Optional[str]]:
    """Return (originality 0..1, the most-copied source title). 1.0 = nothing reproduced. Measured as the
    largest share of any single source's five-word sequences that reappear in the draft."""
    body = _shingles(text, n)
    worst, worst_src = 0.0, None
    for s in sources:
        src = _shingles(" ".join([s.title, s.summary, s.translated]), n)
        if not src:
            continue
        reproduced = len(body & src) / len(src)
        if reproduced > worst:
            worst, worst_src = reproduced, s.title
    return round(1.0 - worst, 3), worst_src


@dataclass
class Commentary:
    topic: str
    title: str
    body: str               # labelled, bylined, with a Sources section
    byline: str
    sources: list[dict] = field(default_factory=list)
    originality: float = 1.0
    claims_to_verify: list[str] = field(default_factory=list)
    balance_flags: list[str] = field(default_factory=list)
    headline_flags: list[str] = field(default_factory=list)
    originality_flags: list[str] = field(default_factory=list)
    diversity_flags: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True only when there is nothing for a human to check — the bar for unattended publishing."""
        return not (self.claims_to_verify or self.balance_flags or self.headline_flags or self.originality_flags
                    or self.diversity_flags)


def write_commentary(llm: LLM, topic: str, sources: list[SourceItem], byline: str, stance: Any = None,
                     voice: str = "punchy", comedic: bool = True, max_overlap: float = 0.25) -> Commentary:
    ok, why = check_byline(byline)
    if not ok:
        raise PermissionError(why)
    research = "\n".join(f"- [{s.feed or s.lang or 'source'}] {s.title}: {(s.translated or s.summary)[:400]} ({s.url})"
                         for s in sources[:8]) or "(no source coverage gathered; argue from what is broadly known and say what would need checking)"
    stance_block = (stance_guidance(stance, comedic=comedic) + "\n") if stance is not None else ""
    prompt = (f"Write original commentary on the trending topic below. Voice: {voice}.\n{stance_block}"
              f"TOPIC: {topic}\nRESEARCH (do not copy; use for facts and attribution only):\n{research}\n"
              "Return JSON {title, body, summary}. The title must be accurate and eye-catching, never clickbait.")
    try:
        d: dict[str, Any] = generate_json(llm, prompt, system=_SYSTEM, schema=_SCHEMA, temperature=0.8,
                                          max_tokens=1400, purpose="newsdesk:commentary")
    except Exception:
        d = {"title": topic, "body": "", "summary": ""}
    title = str(d.get("title", "")).strip() or topic
    body = str(d.get("body", "")).strip()
    hc = check_headline(title)
    if not hc.ok:
        title = f"{topic}: what it actually means"  # replace a clickbait title with a plain, accurate one
        hc = check_headline(title)
    orig, worst_src = originality(body, sources)
    o_flags = []
    if 1.0 - orig > max_overlap:
        o_flags.append(f"too close to a source ({1.0 - orig:.0%} of “{worst_src}” reproduced); rewrite as your own analysis")
    src_list = [{"title": s.title, "url": s.url, "lang": s.lang} for s in sources if s.url]
    sources_md = ("\n\nSources:\n" + "\n".join(f"- {s['title']} — {s['url']}" for s in src_list)) if src_list else ""
    labelled = f"*{OPINION_DISCLAIMER}*\n\nBy {byline}\n\n{body}{sources_md}"
    _, div_flags = source_diversity(sources)
    return Commentary(topic=topic, title=title, body=labelled, byline=byline, sources=src_list, originality=orig,
                      claims_to_verify=flag_claims(body), balance_flags=balance_check(body),
                      headline_flags=hc.flags, originality_flags=o_flags, diversity_flags=div_flags)
