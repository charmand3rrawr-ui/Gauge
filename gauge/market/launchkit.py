"""Generate the marketing assets, then assemble a disclosed posting plan the author executes.

The assets (tagline, logline, blurbs, tags, author's note) come from the model, grounded in the
project's bible. The posting plan is built deterministically from those assets plus each channel's
rules, and every drafted post carries the disclosure line, so nothing Gauge produces can be passed off
as anything other than the author promoting their own work. Gauge writes the material; the author posts
it, under their own identity, following each community's rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..llm.client import LLM, generate_json
from .channels import Channel, channels_for
from .policy import DISCLOSURE

_SCHEMA = {
    "type": "object",
    "properties": {
        "tagline": {"type": "string"},
        "logline": {"type": "string"},
        "blurb_short": {"type": "string"},
        "blurb_long": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "author_note": {"type": "string"},
    },
    "required": ["tagline", "logline", "blurb_short", "blurb_long"],
}


@dataclass
class LaunchKit:
    title: str
    tagline: str = ""
    logline: str = ""
    blurb_short: str = ""
    blurb_long: str = ""
    tags: list[str] = field(default_factory=list)
    author_note: str = ""

    def to_markdown(self) -> str:
        lines = [f"# Launch kit — {self.title}", "", f"**Tagline:** {self.tagline}", "", f"**Logline:** {self.logline}", "",
                 "## Short blurb", self.blurb_short, "", "## Long blurb", self.blurb_long, ""]
        if self.tags:
            lines += ["## Tags", ", ".join(self.tags), ""]
        if self.author_note:
            lines += ["## Author's note", self.author_note, ""]
        return "\n".join(lines)


def build_launch_kit(llm: LLM, project: Any) -> LaunchKit:
    bible = project.bible_markdown() if hasattr(project, "bible_markdown") else ""
    name = getattr(getattr(project, "state", None), "name", "") or "Untitled"
    genres = getattr(getattr(project, "state", None), "genres", []) or []
    prompt = ("Create a marketing launch kit for a web novel. Give a tagline (<=10 words), a logline (one sentence), "
              "a short blurb (<=60 words), a long blurb (<=150 words), 5-8 tags, and a one-paragraph author's note. "
              "No spoilers, no clickbait, honest to the story.\n\n"
              f"TITLE: {name}\nGENRES: {', '.join(genres)}\nBIBLE:\n{bible[:2000]}")
    try:
        d: dict[str, Any] = generate_json(llm, prompt, schema=_SCHEMA, temperature=0.8, max_tokens=600, purpose="launch_kit")
    except Exception:
        d = {}
    return LaunchKit(
        title=name,
        tagline=str(d.get("tagline", "")).strip(),
        logline=str(d.get("logline", "")).strip(),
        blurb_short=str(d.get("blurb_short", "")).strip(),
        blurb_long=str(d.get("blurb_long", "")).strip(),
        tags=[str(t).strip() for t in d.get("tags", []) if str(t).strip()][:8],
        author_note=str(d.get("author_note", "")).strip(),
    )


def _draft_post(kit: LaunchKit, ch: Channel, url: str = "") -> str:
    """A ready-to-post draft for one channel, built from the kit and always carrying the disclosure."""
    link = f"\n\nLink: {url}" if url else ""
    if ch.kind == "direct":
        body = f"{kit.blurb_short}{link}"
    else:
        body = f"{kit.tagline}\n\n{kit.blurb_short}\n\n{kit.author_note}".strip() + link
    return f"[{DISCLOSURE}]\n\n**{kit.title}**\n{body}"


def posting_plan(kit: LaunchKit, channels: Optional[list[Channel]] = None, url: str = "") -> str:
    """A checklist the author works through themselves: per channel, the rules, disclosure, and a draft."""
    channels = channels or channels_for()
    out = [f"# Posting plan — {kit.title}", "",
           "Post each of these yourself, under your own account, following the venue's current rules. "
           "Gauge does not and will not post for you or create accounts.", ""]
    for ch in channels:
        out += [f"## {ch.name}  ({ch.kind})",
                f"- Audience: {ch.audience}",
                f"- Rules: {ch.rules}",
                f"- Disclosure: {ch.disclosure}",
                f"- Do it right: {ch.do_it_right}",
                "", "Draft to adapt and post yourself:", "", "```", _draft_post(kit, ch, url), "```", ""]
    out += ["---", "Reminder: read each venue's current pinned rules before posting; this plan is a starting point, "
            "and undisclosed or automated promotion is against the rules everywhere."]
    return "\n".join(out)
