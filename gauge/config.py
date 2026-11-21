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
    # Execution. num_ctx is sent on every call: Ollama's default (2-4k) silently truncates long
    # prompts. 8192 fits a full draft prompt; raise it if you have the VRAM.
    num_ctx: int = 8192
    keep_alive: str = "30m"  # keep weights loaded between calls / daemon cycles ("-1" = forever)
    fast_model: str = ""  # small model for JSON work (briefs, reviews, memory, critic); "" = main model
    parallel: int = 4  # concurrent character calls; match OLLAMA_NUM_PARALLEL
    retries: int = 3
    retry_backoff_seconds: float = 2.0
    structured_output: bool = True  # JSON-schema constrained output when the server supports it


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
    absorb_per_cycle: int = 8  # chapters turned into technique cards per daemon cycle (needs the LLM)


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
    min_mentions_for_review: int = 2  # characters named fewer times than this skip review/memory (protagonist never skips)
    checkpoint_drafts: bool = True  # save raw drafts so a crash never loses a generation
    # The World agent: physics, weather, crowds and consequences, present in every scene.
    world_enabled: bool = True
    world_agency: float = 0.6  # 0 = pure backdrop, 1 = the world constantly intervenes
    world_weight: float = 0.1  # share of the chapter score from world-consistency review
    # Creativity.
    muse_enabled: bool = True
    muse_directions: int = 3  # alternative directions proposed before each outline extension
    techniques_in_prompt: int = 3  # technique cards shown to the drafter
    novelty_weight: float = 0.15  # share of the chapter score from novelty against earlier chapters


class ResearchConfig(BaseModel):
    """The Scout's open sources. No keys, polite delays, everything cached in the database."""

    enabled: bool = True
    sources: list[str] = Field(default_factory=lambda: ["wikipedia", "wikivoyage", "gutenberg"])  # + "arxiv", "climate", "searx"
    respect_robots_txt: bool = True  # arxiv and climate APIs are robots-disallowed; set false knowingly to use them
    searx_url: str = ""  # your own SearXNG instance, e.g. http://localhost:8080 (adds general web search)
    max_queries_per_chapter: int = 3
    results_per_query: int = 3
    cache_days: int = 30
    request_delay_seconds: float = 1.5  # api.wikimedia.org rate-limits anonymous clients; be gentle
    fact_check: bool = True  # verify up to 3 real-world claims per draft
    dossier_on_create: bool = True  # look up the bible's proper nouns and trades when a project is created


class AgentsConfig(BaseModel):
    """Independent story agents (the writing room). Each may be switched off; `profile` is a shortcut."""

    profile: str = "full"  # full | lean (cheap agents only) | off
    weight: float = 0.25  # share of the chapter score that comes from the council
    reader_panel: bool = True
    showrunner: bool = True
    archivist: bool = True
    schemer: bool = True
    line_editor: bool = True
    packager: bool = True
    theme_keeper: bool = True
    sensitivity: bool = True
    pacing: bool = True
    librarian: bool = True
    comedian: bool = True  # attaches only when the genre or tone is comic
    scout: bool = True  # searches open sources for real-world detail (see research settings)
    sensitivity_rules: str = ""  # your platform's content rules, pasted
    sensitivity_categories: list[str] = Field(default_factory=list)
    block_on_sensitivity: bool = False  # severe flags force a revision instead of a warning


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
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)

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
