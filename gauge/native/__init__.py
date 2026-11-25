"""GaugeLM: Gauge's own language model. Pure Python + numpy; no outside model code or weights."""
from .client import NativeLLM
from .model import PRESETS, AdamW, GaugeLM, ModelConfig
from .tokenizer import BPETokenizer
from .train import NativeTrainer, TrainConfig, dedup, gather_texts, instruction_rows, preset_config

__all__ = ["NativeLLM", "PRESETS", "AdamW", "GaugeLM", "ModelConfig", "BPETokenizer", "NativeTrainer", "TrainConfig", "gather_texts", "preset_config", "instruction_rows", "dedup"]
