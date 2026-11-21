"""GaugeLM: Gauge's own language model. Pure Python + numpy; no outside model code or weights."""
from .client import NativeLLM
from .model import AdamW, GaugeLM, ModelConfig
from .tokenizer import BPETokenizer
from .train import NativeTrainer, TrainConfig, gather_texts

__all__ = ["NativeLLM", "AdamW", "GaugeLM", "ModelConfig", "BPETokenizer", "NativeTrainer", "TrainConfig", "gather_texts"]
