"""Writing: plan a serial, draft chapters with the local LLM, score them with the learned
engagement model, revise, repeat."""
from .pipeline import ChapterRejected, WritePipeline
from .project import Project

__all__ = ["ChapterRejected", "WritePipeline", "Project"]
