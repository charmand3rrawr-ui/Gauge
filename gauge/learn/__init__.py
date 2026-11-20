"""Learning: turn the ingested corpus into (1) an engagement model, (2) an exemplar library,
(3) a distilled style guide, and (4) a preference dataset for optional fine-tuning."""
from .engagement import TARGETS, EngagementModel, build_training_table
from .exemplars import ExemplarLibrary
from .trainer import Trainer

__all__ = ["EngagementModel", "TARGETS", "build_training_table", "ExemplarLibrary", "Trainer"]
