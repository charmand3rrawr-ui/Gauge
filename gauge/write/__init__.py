"""Writing: plan a serial, draft chapters with the local LLM, score them with the learned
engagement model, revise, repeat."""
from .pipeline import WritePipeline
from .project import Project

__all__ = ["WritePipeline", "Project"]
