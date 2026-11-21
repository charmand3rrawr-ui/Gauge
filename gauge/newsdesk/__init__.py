"""The newsdesk: find what the world is talking about, in any language, and write ORIGINAL commentary on it.

It discovers trending topics (Wikipedia's most-viewed pages per language, plus RSS feeds you choose),
translates source coverage for research, and drafts original, labelled opinion in the platform's declared
stance that cites and links its sources. An originality check refuses a draft that merely paraphrases a
source: this desk writes commentary, it does not rewrite other people's articles. Drafts go into a queue
published on a schedule you set, held for your approval unless a piece passes every check and you have
opted into auto-publishing.
"""
from .commentary import Commentary, originality, write_commentary
from .queue import PublishQueue, QueueItem
from .translate import detect_script, translate
from .trends import SourceItem, Topic, discover, rss_items, wikimedia_top

__all__ = ["Topic", "SourceItem", "discover", "rss_items", "wikimedia_top", "translate", "detect_script",
           "Commentary", "write_commentary", "originality", "PublishQueue", "QueueItem"]
