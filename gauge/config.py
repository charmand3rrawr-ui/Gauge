"""Configuration for Gauge.

Everything lives under a single data directory (default ``./data`` next to where you run
``gauge``). Settings are stored as JSON so they are easy to edit by hand on Windows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

DEFAULT_DATA_DIR = Path(os.environ.get("GAUGE_DATA_DIR", "data"))


class LLMConfig(BaseModel):
    """Local LLM endpoint settings. Ollama is the default; any OpenAI-compatible server works."""

    provider: str = "ollama"  # "ollama" | "openai" (LM Studio, llama.cpp server, vLLM...) | "fake"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"
    api_key: str = ""  # only needed for some OpenAI-compatible servers
    temperature: float = 0.9
    max_tokens: int = 3000
    timeout_seconds: int = 600


class IngestConfig(BaseModel):
    user_agent: str = "Gauge/0.1 (open-source web novel research tool)"
    request_delay_seconds: float = 2.5
    max_chapters_per_novel: int = 400
    max_novels_per_run: int = 25
    respect_robots_txt: bool = True


class LearnConfig(BaseModel):
    target: str = "log_followers"  # see gauge.learn.engagement.TARGETS
    ridge_alpha: float = 1.0
    exemplar_count: int = 200
    min_chapters_for_training: int = 30


class WriteConfig(BaseModel):
    target_chapter_words: int = 2200
    max_revisions: int = 2
    min_acceptable_score: float = 0.62
    exemplars_in_prompt: int = 4
    # Characters as independent agents: scene briefs before drafting, in-character review after.
    characters_enabled: bool = True
    max_active_characters: int = 5
    character_review: bool = True
    character_weight: float = 0.2  # share of the chapter score that comes from in-character consistency


class DaemonConfig(BaseModel):
    cycle_minutes: int = 180
    write_after_learning: bool = False
    active_project: str = ""


class GaugeConfig(BaseModel):
    data_dir: str = str(DEFAULT_DATA_DIR)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    learn: LearnConfig = Field(default_factory=LearnConfig)
    write: WriteConfig = Field(default_factory=WriteConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)

    # ---- paths -------------------------------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.data_dir)

    @property
    def db_path(self) -> Path:
        return self.root / "gauge.sqlite3"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def projects_dir(self) -> Path:
        return self.root / "projects"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    def ensure_dirs(self) -> None:
        for p in (self.root, self.models_dir, self.projects_dir):
            p.mkdir(parents=True, exist_ok=True)

    # ---- persistence -------------------------------------------------------
    def save(self) -> Path:
        self.ensure_dirs()
        self.config_path.write_text(json.dumps(self.model_dump(), indent=2), encoding="utf-8")
        return self.config_path

    @classmethod
    def load(cls, data_dir: Optional[str | Path] = None) -> GaugeConfig:
        root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        path = root / "config.json"
        if path.exists():
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            raw["data_dir"] = str(root)
            return cls(**raw)
        return cls(data_dir=str(root))
