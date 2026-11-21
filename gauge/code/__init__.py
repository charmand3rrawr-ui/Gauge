"""Gauge as an engineer: programming knowledge, a deterministic toolkit for bots, bot creation, and
careful self-modification."""
from . import toolkit
from .engineer import BotRegistry, BotSpec, Engineer, run_tests
from .knowledge import CARDS, Card, KnowledgeBase
from .selfmod import PROTECTED, Proposal, SelfProgrammer, repo_root

__all__ = ["toolkit", "BotRegistry", "BotSpec", "Engineer", "run_tests", "CARDS", "Card", "KnowledgeBase", "PROTECTED", "Proposal",
           "SelfProgrammer", "repo_root"]
