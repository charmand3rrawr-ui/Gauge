"""The switches: every setting a person might flip, with a label, a description and a warning where
one is due. One catalog drives both the Studio's Switches panel and `gauge settings`, so they
cannot disagree. Values are validated and coerced here; the CLI and GUI only display and collect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .config import GaugeConfig


@dataclass
class Switch:
    key: str  # dotted path into GaugeConfig, e.g. "write.rebel"
    label: str
    kind: str  # bool | float | int | choice | text
    description: str
    group: str = "Writing"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: list[str] = field(default_factory=list)
    warning: str = ""  # shown in red: this affects other people, your machine, or safety


SWITCHES: list[Switch] = [
    # ---- craft ---------------------------------------------------------------------------
    Switch("write.rebel", "Rebel craft mode", "bool",
           "Conventions become suggestions: the muse proposes directions that break genre, pacing and style rules on purpose, "
           "and the judges score effect over compliance. Breaks only rules of craft, never rules about other people."),
    Switch("write.rebel_level", "How rebellious", "float", "0 = gentle nudges, 1 = every chapter should break something.", minimum=0, maximum=1),
    Switch("write.persona_blend", "Write like me vs. like the data", "float", "0 = the data-driven style guide; 1 = your author persona dominates.", minimum=0, maximum=1),
    Switch("write.novelty_weight", "Punish repetition", "float", "Share of the chapter score that comes from not repeating earlier chapters.", minimum=0, maximum=0.5),
    Switch("write.retention_weight", "Reward the retention curve", "float", "Share of the score from chapter shape: opening hook, mid-chapter spike, end cliff. 0 = off.", minimum=0, maximum=0.5),
    Switch("write.realism_weight", "Reward realism", "float", "Share of the score from sensory grounding and show-dont-tell. 0 = off. Realism can trade against pace.", minimum=0, maximum=0.5),
    Switch("write.ab_titles", "A/B chapter titles", "bool", "Generate several titles per chapter and keep the best-scoring one; choices are logged."),
    Switch("write.ab_hooks", "A/B opening hooks", "bool", "Generate several opening lines and offer the strongest in the draft brief."),
    Switch("write.min_acceptable_score", "Quality bar", "float", "Chapters below this score are revised (up to the revision limit).", minimum=0, maximum=1),
    Switch("write.max_revisions", "Revision limit", "int", "How many rewrites a chapter may get before the best version is kept.", minimum=0, maximum=5),
    # ---- the room -------------------------------------------------------------------------
    Switch("write.characters_enabled", "Characters as agents", "bool", "Each character briefs, reviews and remembers as themselves.", "The room"),
    Switch("write.carry_characters", "Carry characters between books", "bool", "Seed a new character from the cross-book memory: scars, habits and grudges survive from earlier serials.", "The room"),
    Switch("write.character_weight", "Characters' say in the score", "float", "Share of the score from in-character consistency.", "The room", 0, 0.5),
    Switch("write.improv_enabled", "Scene improv before drafting", "bool", "Present characters play the scene out first; the transcript feeds the draft.", "The room"),
    Switch("write.world_enabled", "World agent", "bool", "Physics, weather, crowds and consequences act in every scene.", "The room"),
    Switch("write.world_agency", "How much the world intervenes", "float", "0 = backdrop, 1 = the world constantly acts.", "The room", 0, 1),
    Switch("write.world_weight", "World's say in the score", "float", "Share of the score from world-rule consistency.", "The room", 0, 0.5),
    Switch("write.muse_enabled", "Muse", "bool", "Divergent directions proposed and rated before each outline extension.", "The room"),
    Switch("agents.profile", "Writing room size", "choice", "full = every agent; lean = the cheap ones; off = none.", "The room", choices=["full", "lean", "off"]),
    Switch("agents.weight", "Writing room's say in the score", "float", "Share of the chapter score the council controls.", "The room", 0, 0.6),
    Switch("agents.block_on_sensitivity", "Content flags block", "bool", "Severe content flags force a revision instead of only warning.", "The room"),
    Switch("agents.readability", "Readability coach", "bool", "Scores burstiness, syllable load and paragraph rhythm; smooth-to-read as a number.", "The room"),
    Switch("agents.suitability", "Suitability checker", "bool", "Checks each chapter against the audience contract below.", "The room"),
    Switch("agents.arc_keeper", "Arc keeper", "bool", "Tracks the bible's long arc and flags when a long serial drifts from the story it promised.", "The room"),
    Switch("agents.sensory", "Sensory editor", "bool", "Grounds scenes in the senses; flags told-emotion, filter words and generic detail.", "The room"),
    Switch("agents.continuity", "Continuity auditor", "bool", "Checks physical, time-of-day and season consistency across chapters.", "The room"),
    Switch("agents.craft_editor", "Craft editor", "bool", "Line craft: cliches, weak verb+adverb, on-the-nose dialogue, tense drift, head-hopping, anachronisms.", "The room"),
    Switch("agents.cast_tracker", "Cast tracker", "bool", "Keeps the ensemble alive: flags absent cast, flat wants/needs, frozen relationships and unused voices.", "The room"),
    Switch("agents.structure", "Structure editor", "bool", "Try-fail cost, conflict, cliffhanger variety and structural beat placement.", "The room"),
    Switch("agents.suitability_age_band", "Audience age band", "choice", "Sets a reading-ease floor: all = easiest, adult = densest allowed.", "The room", choices=["all", "teen", "adult"]),
    Switch("agents.suitability_heat_max", "Maximum heat level", "int", "0 none, 1 mild, 2 moderate, 3 explicit. Chapters above this are flagged.", "The room", 0, 3),
    Switch("agents.suitability_platform", "Target platform", "text", "Free text shown to the writer, e.g. Royal Road, Wattpad.", "The room"),
    Switch("agents.suitability_block", "Crossed content lines block", "bool", "A chapter that crosses a declared content line is revised, not just warned.", "The room"),
    # ---- learning -------------------------------------------------------------------------
    Switch("learn.bandit_enabled", "Tune prompt settings by reward", "bool", "Thompson-sampling bandit over exemplars, temperature, techniques, world agency.", "Learning"),
    Switch("learn.self_preference_pairs", "Collect fine-tuning pairs", "bool", "Save draft-vs-accepted pairs for later fine-tuning.", "Learning"),
    Switch("learn.lessons_in_prompt", "Lessons shown to the writer", "int", "How many recurring lessons go into every prompt.", "Learning", 0, 10),
    Switch("research.enabled", "Scout research", "bool", "Look up real-world detail for scenes in open sources.", "Learning"),
    Switch("research.respect_robots_txt", "Honour robots.txt", "bool",
           "When off, arXiv and Open-Meteo lookups work (their robots.txt disallow all bots even though their API terms allow identified clients).",
           "Learning", warning="Turning this off means fetching from sites whose robots.txt says no. Your call, not the software's."),
    Switch("research.fact_check", "Fact-check drafts", "bool", "Verify up to three real-world claims per chapter.", "Learning"),
    # ---- model & compute --------------------------------------------------------------------
    Switch("llm.provider", "Model provider", "choice", "ollama (default), openai-compatible server, or native GaugeLM.", "Model", choices=["ollama", "openai", "native", "fake"]),
    Switch("llm.model", "Writer model", "text", "Ollama model name for prose (e.g. llama3.1:8b, qwen2.5:14b).", "Model"),
    Switch("native.after_chapter_minutes", "Train GaugeLM after each chapter (minutes)", "int", "In Gauge Studio, train GaugeLM, Gauge's own model, for this many minutes after every chapter, whichever model wrote it. This is how GaugeLM keeps learning while the downloaded model does the writing. 0 turns it off.", "Model", 0, 120),
    Switch("llm.fast_model", "Fast model for bookkeeping", "text", "Small model for briefs, reviews, memory (e.g. llama3.2:3b). Empty = same as writer.", "Model"),
    Switch("llm.cache", "Cache repeated calls", "bool", "Reuse the answer to an identical deterministic/bookkeeping request instead of paying again. Prose is never cached.", "Model"),
    Switch("native.device", "GaugeLM device", "choice", "auto picks a GPU when one is usable.", "Model", choices=["auto", "cuda", "cpu"]),
    Switch("compute.auto_device", "Switch to a GPU when one appears", "bool", "The daemon re-scans every cycle.", "Model"),
    Switch("compute.pool_generation", "Use your workers for generation", "bool", "Spread prose calls over machines you started `gauge worker` on.", "Model"),
    Switch("compute.distributed_training", "Use your workers for training", "bool", "Model-averaging rounds across your workers.", "Model"),
    # ---- security -------------------------------------------------------------------------
    Switch("security.injection_action", "Instruction-shaped text in sources", "choice",
           "scrub = neutralise it; flag = keep but log; drop = discard suspicious passages.", "Security", choices=["scrub", "flag", "drop"]),
    Switch("security.verify_each_cycle", "Verify sealed code every cycle", "bool", "The daemon checks Gauge's own files against the seal before running automation.", "Security"),
    Switch("security.auto_lockdown", "Automatic lockdown", "bool", "When the threat level reaches lockdown, risky automation is switched off until you unlock.", "Security"),
    Switch("security.scan_generated_code", "Scan generated code", "bool", "Block bots and proposals that contain dangerous constructs.", "Security",
           warning="Turning this off lets model-written code with eval/shell/credential access run on your machine."),
    Switch("security.scan_dependencies", "Scan dependencies", "bool", "Flag or block generated code that imports unvetted or typosquat-looking packages.", "Security"),
    Switch("security.semantic_injection", "Semantic injection detector", "bool", "A model-based second opinion on prompt injection, past what regex catches (slower, non-deterministic).", "Security"),
    Switch("security.allow_public_workers", "Accept workers from public networks", "bool", "Workers accept only private-network peers unless on.", "Security",
           warning="Exposes the worker API beyond your LAN; use only behind a VPN you control."),
    Switch("security.active_defense_level", "Active defence level", "int", "How hard Gauge pushes back on its OWN machine: 0 passive, 1 tarpit+evidence, 2 +honeytokens+decoys, 3 maximum. Never touches other systems.", "Security", 0, 3),
    Switch("security.active_defense_auto", "Escalate defence with threat", "bool", "Raise the active-defence level automatically as the sentinel's threat level rises.", "Security"),
    # ---- automation -----------------------------------------------------------------------
    Switch("dry_run", "Dry run (hold everything outward)", "bool", "Hold posting, email, newsdesk publishing, self-modification, restore, daemon chapter writes and bots; report what a real run would have done. Same as gauge --dry-run.", "Automation"),
    Switch("daemon.write_after_learning", "Daemon writes chapters", "bool", "Each learning cycle also writes the next chapter of the active project.", "Automation"),
    Switch("daemon.error_budget", "Error budget (per window)", "int", "Unresolved errors the daemon may accumulate per window before it holds chapter writes, newsdesk, bots and auto-apply and slows down. 0 turns the budget off.", "Automation", 0, 1000),
    Switch("daemon.error_window_hours", "Error budget window (hours)", "int", "The rolling window the error budget is counted over.", "Automation", 1, 720),
    Switch("logs.structured", "Structured log file", "bool", "Also write every log record as a JSON line to logs/gauge.jsonl in the data directory (rotated, secrets masked).", "Automation"),
    Switch("newsdesk.enabled", "Newsdesk in the daemon", "bool", "Each daemon cycle discovers trends, drafts original commentary into the review queue, and publishes one due piece.", "Automation"),
    Switch("newsdesk.interval_minutes", "Publish interval (minutes)", "int", "At most one piece is published per interval.", "Automation", 5, 1440),
    Switch("newsdesk.max_per_day", "Pieces per day (cap)", "int", "Hard daily ceiling on published pieces.", "Automation", 1, 96),
    Switch("newsdesk.target_language", "Translate research into", "text", "Language the desk translates source coverage into.", "Automation"),
    Switch("newsdesk.byline", "Newsdesk byline", "text", "Your name or an owned pen name; outlets and shared pseudonyms are refused.", "Automation"),
    Switch("newsdesk.project", "Publish into site project", "text", "The site project that published pieces are added to.", "Automation"),
    Switch("newsdesk.auto_publish", "Auto-publish clean pieces", "bool", "Publish pieces with no flags at all without waiting for you. Anything with a checkable factual claim still waits.", "Automation",
           warning="Pieces will go live under your byline while you are not watching. Only fully clean drafts qualify; review the queue regularly."),
    Switch("code.bots_enabled", "Run scheduled bots", "bool", "The daemon runs the bots whose schedule is due.", "Automation"),
    Switch("code.self_heal", "Self-heal proposals", "bool", "Recurring errors become tested proposals to fix Gauge's own code (not applied).", "Automation",
           warning="Gauge will write proposals to change its own source. They are tested in isolation and wait for you unless auto-apply is on."),
    Switch("code.auto_apply", "Auto-apply self-fixes", "bool", "Apply proposals that pass the full test suite without asking you.", "Automation",
           warning="Gauge will commit changes to its own code while you are not watching. Protected files and tests stay out of reach; rollback exists. Keep off unless you trust it."),
    Switch("code.mutation_check", "Mutation-test self-edits", "bool", "After a proposal passes, mutate the changed code to prove the tests actually cover it; a proposal whose tests miss too many mutations is blocked.", "Automation"),
    Switch("code.mutation_min_score", "Minimum mutation score", "float", "Fraction of mutants the tests must kill for a self-edit to pass (when mutation-testing is on).", "Automation", 0, 1),
]

BY_KEY = {s.key: s for s in SWITCHES}


def get(cfg: GaugeConfig, key: str) -> Any:
    obj: Any = cfg
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def coerce(sw: Switch, value: Any) -> Any:
    if sw.kind == "bool":
        if isinstance(value, bool):
            return value
        v = str(value).strip().lower()
        if v in ("1", "true", "yes", "on", "y"):
            return True
        if v in ("0", "false", "no", "off", "n"):
            return False
        raise ValueError(f"{sw.key}: expected on/off, got {value!r}")
    if sw.kind in ("float", "int"):
        try:
            num = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{sw.key}: expected a number, got {value!r}") from e
        if sw.minimum is not None and num < sw.minimum or sw.maximum is not None and num > sw.maximum:
            raise ValueError(f"{sw.key}: {num} is outside {sw.minimum}..{sw.maximum}")
        return int(round(num)) if sw.kind == "int" else round(num, 3)
    if sw.kind == "choice":
        v = str(value).strip().lower()
        if v not in sw.choices:
            raise ValueError(f"{sw.key}: choose one of {', '.join(sw.choices)}")
        return v
    return str(value).strip()


def set_value(cfg: GaugeConfig, key: str, value: Any, save: bool = True) -> Any:
    sw = BY_KEY.get(key)
    if sw is None:
        raise KeyError(f"unknown switch {key!r}; see `gauge settings list`")
    v = coerce(sw, value)
    obj: Any = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], v)
    if key == "native.device":
        cfg.llm.native_device = v
    if save:
        cfg.save()
    return v


def state(cfg: GaugeConfig) -> list[dict[str, Any]]:
    out = []
    for sw in SWITCHES:
        out.append({"key": sw.key, "label": sw.label, "kind": sw.kind, "group": sw.group, "value": get(cfg, sw.key),
                    "description": sw.description, "warning": sw.warning, "minimum": sw.minimum, "maximum": sw.maximum, "choices": sw.choices})
    return out


def groups() -> list[str]:
    seen: list[str] = []
    for sw in SWITCHES:
        if sw.group not in seen:
            seen.append(sw.group)
    return seen


def reset(cfg: GaugeConfig, key: str, save: bool = True) -> Any:
    default = get(GaugeConfig(data_dir=cfg.data_dir), key)
    return set_value(cfg, key, default, save)
