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
    native_dir: str = ""  # filled in by GaugeConfig.load: where the built-in GaugeLM lives
    native_device: str = "cpu"  # filled in by GaugeConfig.load from native.device


class NativeConfig(BaseModel):
    """GaugeLM, the built-in model (llm.provider = "native"). Sizes are for a desktop CPU."""

    size: str = "tiny"  # tiny | small | medium | large (see gauge native presets); overrides below when set
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_kv_head: int = 2
    block_size: int = 256
    vocab_size: int = 4096
    arch: str = "llama"  # llama (RMSNorm, RoPE, SwiGLU, GQA) | gpt (legacy)
    device: str = "auto"  # auto | cuda | cpu  (cuda needs: pip install cupy-cuda12x)
    steps: int = 2000
    batch_size: int = 8
    accum_steps: int = 2
    lr: float = 6e-4
    chat_weight: int = 3
    daemon_steps: int = 200  # continue-training steps per daemon cycle on new material
    daemon_minutes: int = 20  # time budget per cycle
    distill_rows: int = 200  # teacher answers to collect per `gauge native distill` run


class ComputeConfig(BaseModel):
    """Sensing GPUs and using the other machines you own (gauge worker)."""

    auto_device: bool = True  # re-scan each daemon cycle and switch GaugeLM to a GPU when one appears
    token: str = ""  # shared secret for your workers; `gauge compute token` generates one
    announce_port: int = 47631
    worker_port: int = 47632
    discover_seconds: float = 3.0
    distributed_training: bool = True  # use live workers for native training rounds
    pool_generation: bool = True  # spread LLM calls over workers that serve a model


class CodeConfig(BaseModel):
    """Bots and self-modification."""

    bots_enabled: bool = True  # the daemon runs scheduled bots
    max_fix_rounds: int = 2  # how many times the Engineer may rewrite a bot whose tests fail
    self_heal: bool = False  # let the daemon turn recurring errors into tested proposals
    auto_apply: bool = False  # apply proposals that pass tests without a human (off by default; keep it off unless you trust it)
    min_recurrences: int = 3  # an error must repeat this often before self-heal proposes a fix
    mutation_check: bool = False  # after a proposal passes tests, mutate the changed code to prove the tests cover it
    mutation_min_score: float = 0.5  # a proposal whose tests kill fewer than this fraction of mutants is blocked
    mutation_max_mutants: int = 8  # mutants per changed file (each re-runs the suite, so keep it small)


class SecurityConfig(BaseModel):
    """Defences. See docs/SECURITY.md."""

    injection_action: str = "scrub"  # scrub | flag | drop: what to do with instruction-shaped text in ingested content
    verify_each_cycle: bool = True  # check the sealed source tree at the start of every daemon cycle
    auto_lockdown: bool = True  # the sentinel switches risky automation off when the threat level reaches lockdown
    allow_public_workers: bool = False  # workers accept only private-network peers unless true
    max_request_bytes: int = 256 * 1024 * 1024
    rate_limit_per_minute: int = 120
    auth_lockout_failures: int = 10  # failed tokens from one peer before it is ignored for a while
    scan_generated_code: bool = True
    semantic_injection: bool = False  # a model-based second opinion on prompt injection, past what regex can catch (slower, non-deterministic)
    scan_dependencies: bool = True  # flag non-stdlib / unknown / typosquat-looking imports in generated code
    bots_network_default: bool = False  # bots get network only when their manifest says so
    active_defense_level: int = 0  # 0 passive .. 3 high: how hard Gauge pushes back on its OWN machine (tarpit, honeytokens, decoys, evidence)
    active_defense_auto: bool = True  # escalate the active-defence level with the sentinel's threat level
    drill_rounds: int = 8  # rounds of adversarial self-play in `gauge security drill` (attacks and defence params vary each round)
    locked: bool = False
    locked_reason: str = ""
    locked_previous: dict[str, bool] = Field(default_factory=dict)


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
    # Learning from your own serial.
    project_model_min_chapters: int = 30  # labelled chapters before a per-project model overrides the corpus model
    bandit_enabled: bool = True  # tune prompt settings per project by reward (Thompson sampling)
    self_preference_pairs: bool = True  # record draft-vs-accepted pairs for fine-tuning
    lessons_in_prompt: int = 5  # recurring lessons shown to the writer


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
    carry_characters: bool = False  # seed new characters from the cross-book memory (scars, habits, grudges)
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
    retention_weight: float = 0.0  # share of the chapter score from retention-curve shape (hook/mid-spike/cliff); 0 = off
    realism_weight: float = 0.0  # share of the chapter score from sensory grounding and show-dont-tell; 0 = off
    world_realism: bool = False  # ground the World agent in the real world (geography, weather, procedures) rather than inventing
    grounding: bool = False  # the Scout gathers sensory/procedural texture for each scene (needs research enabled)
    ab_titles: bool = False  # generate several chapter titles and keep the best-scoring one
    ab_hooks: bool = False  # generate several opening hooks and offer the best in the draft brief
    # Scene improv: present characters play the scene out before it is written.
    improv_enabled: bool = True
    improv_rounds: int = 2
    improv_max_cast: int = 4
    # Author persona blend: 0 = write like the data, 1 = write like the author persona.
    persona_blend: float = 0.5
    # Rebel craft mode: conventions are suggestions. Breaks rules of craft only, never rules about people.
    rebel: bool = False
    rebel_level: float = 0.6


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
    voice_coach: bool = True  # measures whether characters sound alike
    readability: bool = True  # measures burstiness, syllable load and paragraph rhythm; smooth-to-read as a number
    suitability: bool = True  # checks each chapter against the audience contract below
    arc_keeper: bool = True  # tracks the bible's long arc and flags when a serial drifts from it
    sensory: bool = True  # grounds scenes in the senses; flags told-emotion and filter words
    continuity: bool = True  # checks physical/time-of-day/season consistency
    craft_editor: bool = True  # line-craft: clichés, weak verb+adverb, on-the-nose dialogue, tense drift, anachronisms
    sensitivity_rules: str = ""  # your platform's content rules, pasted
    sensitivity_categories: list[str] = Field(default_factory=list)
    block_on_sensitivity: bool = False  # severe flags force a revision instead of a warning
    # Audience contract for the Suitability Checker.
    suitability_platform: str = ""  # free text, e.g. "Royal Road", "Wattpad"
    suitability_age_band: str = "teen"  # all | teen | adult (sets a reading-ease floor)
    suitability_heat_max: int = 1  # 0 none, 1 mild, 2 moderate, 3 explicit
    suitability_content_lines: list[str] = Field(default_factory=list)  # phrases that must never appear
    suitability_block: bool = False  # a crossed content line forces a revision instead of a warning


class NewsdeskConfig(BaseModel):
    """Trend discovery and original commentary, published on a schedule. See docs/NEWSDESK.md."""

    enabled: bool = False  # the daemon runs a newsdesk cycle each loop
    interval_minutes: int = 30  # at most one piece published per interval
    max_per_day: int = 24
    languages: list[str] = Field(default_factory=lambda: ["en", "es", "fr", "de", "ja", "pt"])  # Wikipedia editions to watch
    feeds: list[str] = Field(default_factory=list)  # RSS/Atom feeds you choose (headlines + summaries + links only)
    target_language: str = "English"  # translate research into this
    drafts_per_cycle: int = 2  # new trending topics drafted per cycle
    project: str = ""  # the site project that published pieces go into
    byline: str = ""  # your name or an owned pen name
    voice: str = "punchy"
    comedic: bool = True
    auto_publish: bool = False  # publish clean pieces without review (anything with a checkable claim still waits)
    max_source_overlap: float = 0.25  # more than this share of a source reproduced = too close; flagged


class ServerConfig(BaseModel):
    """Run Gauge headless and drive it over HTTP — on a home server, a VPS or a container. See docs/DEPLOY.md."""

    host: str = "127.0.0.1"  # loopback by default; 0.0.0.0 exposes it to the network (needs a token and a warning)
    port: int = 8800
    token: str = ""  # required for every API call; generated and printed on first start if empty
    enabled: bool = False


class DaemonConfig(BaseModel):
    cycle_minutes: int = 180
    write_after_learning: bool = False
    active_project: str = ""


CONFIG_VERSION = 1  # bump when a migration is added to GaugeConfig._migrate


class GaugeConfig(BaseModel):
    data_dir: str = str(DEFAULT_DATA_DIR)
    config_version: int = CONFIG_VERSION  # stamped on save; _migrate upgrades older files on load
    seed: Optional[int] = None  # when set, makes a run reproducible (seeds RNGs and the prompt bandit)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    learn: LearnConfig = Field(default_factory=LearnConfig)
    write: WriteConfig = Field(default_factory=WriteConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    native: NativeConfig = Field(default_factory=NativeConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    code: CodeConfig = Field(default_factory=CodeConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    newsdesk: NewsdeskConfig = Field(default_factory=NewsdeskConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @property
    def security_dir(self) -> Path:
        return self.root / "security"

    @property
    def bots_dir(self) -> Path:
        return self.root / "bots"

    @property
    def self_dir(self) -> Path:
        return self.root / "self"

    @property
    def knowledge_dir(self) -> Path:
        return self.models_dir / "knowledge"

    @property
    def native_dir(self) -> Path:
        return self.models_dir / "native"

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

    @staticmethod
    def _migrate(raw: dict[str, Any]) -> dict[str, Any]:
        """Upgrade an older config dict in place. New fields get defaults from pydantic automatically;
        this is for renames and value changes. Each step bumps config_version. Unknown future versions
        are left as-is (a newer Gauge wrote them)."""
        v = int(raw.get("config_version", 0))
        # Example of the pattern (no-op today): if v < 2: rename/convert; v = 2
        raw["config_version"] = max(v, CONFIG_VERSION)
        return raw

    @classmethod
    def load(cls, data_dir: Optional[str | Path] = None) -> GaugeConfig:
        root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        path = root / "config.json"
        if path.exists():
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            raw["data_dir"] = str(root)
            raw = cls._migrate(raw)
            cfg = cls(**raw)
        else:
            cfg = cls(data_dir=str(root))
        cfg.llm.native_dir = str(cfg.native_dir)
        cfg.llm.native_device = cfg.native.device
        return cfg
