"""GaugeLM's school: free models teach it, three ways.

A small model learns behaviour far faster from a stronger model than from raw prose. Until now one
model taught, one way (`distill.py`: the downloaded model answers Gauge's tasks). This module adds:

* **More teachers.** Every chat model installed in the local engine can teach, taking turns, so GaugeLM
  hears more than one voice. They run on this computer and cost nothing. Teachers elsewhere (any
  OpenAI-compatible address) are possible but never used unless listed in the settings with
  ``terms_ok``, because the excerpts they are shown leave this computer and because some providers do
  not allow their answers to be used for training.
* **A tutor.** GaugeLM attempts a task itself; the teacher marks the attempt and writes the answer it
  should have given. The marks are remembered per kind of task, and the kinds it is worst at are
  practised most. The marks over time are its report card.
* **Graded lessons.** The teacher writes very short, simple scenes: clean text at a level a small model
  can actually absorb, getting harder as it trains.
* **Word study.** Gauge measures which words in your library GaugeLM is most surprised by, and the
  teacher explains each one in the sentence where your library uses it, then uses it three more ways.

* **Free hosted helpers, if you connect them.** Services with a free allowance (Groq, OpenRouter's free
  models, Mistral, and others; docs/HELPERS.md) take turns with the local teachers. Each is asked only so
  many times a day, and hands over when it says "too many requests".
* **Claude, if you connect it.** Anthropic's model can lead the school (docs/CLAUDE.md). It is paid and
  outside this computer, so it teaches only after you connect a key, switch its teaching on and accept
  that decision, and only up to a daily spending limit. Then the free teachers carry on.

Everything a teacher is shown comes from text that is permitted for training (your own writing,
public-domain books, your Gauge chapters). Every row records which teacher wrote it and under what
licence that model is published, so the training set stays auditable.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..llm.client import LLM
from .distill import TASKS

log = logging.getLogger(__name__)

STATE = "school_state.json"
FILES = {"answers": "distill.jsonl", "tutor": "tutor.jsonl", "lessons": "lessons.jsonl", "words": "words.jsonl",
         "checks": "checks.jsonl"}

# Licence of the published weights, by the start of the model's name. "open" means a standard permissive
# licence (Apache-2.0 or MIT). The others have their own terms: fine for learning on your own computer,
# worth reading before you share a model trained with them. This is a guide, not legal advice.
LICENCES: list[tuple[str, str, bool]] = [
    ("qwen2.5:3b", "Qwen licence (own terms)", False), ("qwen2.5:72b", "Qwen licence (own terms)", False),
    ("qwen", "Apache-2.0", True), ("mistral", "Apache-2.0", True), ("mixtral", "Apache-2.0", True),
    ("phi", "MIT", True), ("deepseek", "MIT", True), ("smollm", "Apache-2.0", True), ("tinyllama", "Apache-2.0", True),
    ("granite", "Apache-2.0", True), ("openhermes", "Apache-2.0", True), ("zephyr", "MIT", True),
    ("llama", "Llama Community License (own terms)", False), ("gemma", "Gemma Terms of Use (own terms)", False),
    ("codellama", "Llama Community License (own terms)", False), ("command-r", "CC BY-NC (non-commercial)", False),
]
NOT_TEACHERS = ("embed", "bge-", "minilm", "nomic", "rerank", "clip", "whisper")


def licence_for(model: str) -> tuple[str, bool]:
    low = model.lower().split("/")[-1]
    for prefix, licence, is_open in LICENCES:
        if low.startswith(prefix):
            return licence, is_open
    return "unknown: check the model's page", False


@dataclass
class Teacher:
    name: str
    llm: Any
    licence: str
    open_licence: bool
    where: str = "this computer"
    lead: bool = False          # a lead teacher takes every session while it can; the others take turns when it cannot
    teach: bool = True          # its answers may be kept as lessons (false: it only checks GaugeLM's work)
    helper: str = ""            # the id of the free hosted helper it is, when it is one

    def line(self) -> str:
        return f"{self.name}  -  on {self.where}, {self.licence}"

    def ask(self, prompt: str, **kw: Any) -> str:
        return str(self.llm.generate(prompt, **kw)).strip()


def _engine_config(cfg: Any) -> Any:
    """The settings to reach the local engine, also when GaugeLM itself is the writer."""
    llm = cfg.llm
    if llm.provider in ("ollama", "openai", "lmstudio", "llamacpp", "vllm", "fake"):
        return llm
    prev: dict[str, Any] = {}
    memo = Path(cfg.root) / "studio_engine.json"         # what the writer was before GaugeLM took over
    if memo.exists():
        try:
            prev = json.loads(memo.read_text(encoding="utf-8-sig"))
        except ValueError:
            prev = {}
    return llm.model_copy(update={"provider": prev.get("provider") or "ollama", "model": prev.get("model") or "",
                                  "fast_model": "", "num_ctx": int(prev.get("num_ctx") or 8192)})


def teachers(cfg: Any, check: bool = True) -> list[Teacher]:
    """Every teacher available right now, in a steady order. With `check`, only those that answer."""
    from ..llm import make_llm

    base = _engine_config(cfg)
    out: list[Teacher] = []
    if base.provider == "fake":
        lead = claude_teacher(cfg)
        return ([lead] if lead is not None and (not check or _answers(lead)) else []) + \
            [Teacher("fake", make_llm(base, "main"), "test model", True)] + helper_teachers(cfg)
    names: list[str] = []
    if base.provider == "ollama":
        from ..llm.client import OllamaLLM

        installed = OllamaLLM(base, base.model or "x").models() if check else []      # empty when the engine is off
        usable = [m for m in installed if not any(x in m.lower() for x in NOT_TEACHERS)]
        if cfg.native.teachers:                  # the author chose who teaches
            names = [m for m in cfg.native.teachers if not installed or m in installed or f"{m}:latest" in installed]
        else:                                    # everyone installed, the writer first
            names = ([base.model] if base.model and (base.model in usable or not installed) else []) + usable
    elif base.model:
        names = [base.model]
    for name in dict.fromkeys(n for n in names if n):       # in order, no repeats
        llm = make_llm(base.model_copy(update={"model": name, "fast_model": ""}), "main")
        licence, is_open = licence_for(name)
        out.append(Teacher(name, llm, licence, is_open))
    for extra in cfg.native.outside_teachers or []:         # only ones the author listed and vouched for
        if not isinstance(extra, dict) or not extra.get("terms_ok") or not extra.get("base_url") or not extra.get("model"):
            continue
        import os

        key = os.environ.get(str(extra.get("api_key_env") or ""), "") or str(extra.get("api_key") or "")
        ecfg = base.model_copy(update={"provider": "openai", "base_url": str(extra["base_url"]), "model": str(extra["model"]),
                                       "api_key": key, "fast_model": ""})
        out.append(Teacher(str(extra.get("name") or extra["model"]), make_llm(ecfg, "main"),
                           str(extra.get("licence") or "the provider's terms"), False,
                           where=re.sub(r"^https?://", "", str(extra["base_url"])).split("/")[0]))
    out += helper_teachers(cfg)
    lead = claude_teacher(cfg)
    if lead is not None:
        out.insert(0, lead)
    if check:
        out = [t for t in out if _answers(t)]
    return out


def helper_teachers(cfg: Any, job: str = "teach") -> list[Teacher]:
    """The free hosted helpers the author connected and switched on for a job ('teach' or 'check'), that can be
    asked right now: not resting after 'too many requests', and with requests left in today's allowance."""
    if getattr(cfg, "helpers", None) is None or not cfg.helpers.providers:
        return []
    from ..llm import helpers

    out = []
    for pid, llm in helpers.working(cfg, job):
        info, mine = helpers.provider(pid, cfg), helpers.entry(cfg, pid)
        out.append(Teacher(f"{pid}:{llm.model}", llm, f"{info['name']}'s terms and the model's licence (own terms)", False,
                           where=info["name"], teach=bool(mine.get("teach")), helper=pid))
    return out


def claude_teacher(cfg: Any) -> Optional[Teacher]:
    """Claude as the lead teacher, when the author connected it, switched its teaching on and accepted that
    decision (`claude.teach` and `claude.terms_ok`), and today's spending limit has not been reached."""
    cl = getattr(cfg, "claude", None)
    if cl is None or not (cl.teach and cl.terms_ok):
        return None
    from ..llm import claude

    if not claude.installed():
        return None
    limit = float(cl.daily_usd or 0)
    if limit > 0 and claude.Ledger(cfg.root).today()["usd"] >= limit:
        return None
    llm = claude.ClaudeLLM(cfg.llm, cl.model, effort=cl.teach_effort, system=claude.TEACHER_SYSTEM)
    return Teacher(cl.model, llm, claude.LICENCE, False, where="Anthropic's computers", lead=True)


def forget_teacher(datasets_dir: Path, name: str) -> dict[str, int]:
    """Take out every row a teacher wrote (its name starts with `name`), from all the school's files.
    What GaugeLM already learned from them stays in its weights; this removes them from future training."""
    removed: dict[str, int] = {}
    for kind, fname in FILES.items():
        path = Path(datasets_dir) / fname
        if not path.exists():
            continue
        keep, gone = [], 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                teacher = str(json.loads(line).get("teacher", ""))
            except ValueError:
                teacher = ""
            if teacher.lower().startswith(name.lower()):
                gone += 1
            else:
                keep.append(line)
        if gone:
            path.write_text("".join(k + "\n" for k in keep), encoding="utf-8")
        removed[kind] = gone
    return removed


def rows_by(datasets_dir: Path, name: str) -> int:
    """How many rows in the school's files were written by a teacher whose name starts with `name`."""
    n = 0
    for fname in FILES.values():
        try:
            with (Path(datasets_dir) / fname).open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        n += str(json.loads(line).get("teacher", "")).lower().startswith(name.lower())
                    except ValueError:
                        pass
        except OSError:
            pass
    return n


def _answers(t: Teacher) -> bool:
    try:
        return bool(t.llm.available())
    except Exception:
        return False


# ---------------------------------------------------------------------------------- what it is taught from
def passages(texts: list[str], n: int, rng: random.Random, words: int = 220) -> list[str]:
    """n windows of prose from the permitted texts, spread across them."""
    long = [t.split() for t in texts if len(t.split()) >= 80]
    out: list[str] = []
    while long and len(out) < n:
        ws = rng.choice(long)
        start = rng.randrange(0, max(1, len(ws) - words))
        out.append(" ".join(ws[start:start + words]))
    return out


WHO = ["a courier", "a lighthouse keeper", "two sisters", "a tired detective", "a boy with a secret", "an old sailor",
       "a baker who cannot sleep", "a new teacher", "a thief with a conscience", "a girl who talks to crows",
       "a soldier going home", "a mapmaker", "a nurse on the night shift", "a king's cook", "a clockmaker's apprentice",
       "a fisherman's daughter", "a tailor with one customer", "a retired boxer", "a lost child", "a stranger at the inn"]
WHERE = ["on a night train", "in a flooded street", "at a market before dawn", "in a house with one lit window",
         "on a ferry in fog", "in a school after hours", "at the edge of a frozen lake", "in a crowded kitchen",
         "on a rooftop", "in a library with no doors", "at a border crossing", "in a field after the fair",
         "on a staircase that creaks", "in a hospital corridor", "under a bridge in the rain", "in a garden gone wild",
         "at a wedding nobody wanted", "in a workshop full of clocks", "on a road with no signs", "in a cellar"]
WHAT = ["finds something that is not theirs", "is followed", "must keep a promise", "hears their own name",
        "loses the one thing they came for", "is offered a bad bargain", "tells a lie that works too well",
        "opens a letter meant for someone else", "is recognised", "has to choose who to help", "waits for someone who is late",
        "breaks something precious", "is asked a question they cannot answer", "sees a door that was not there",
        "returns something stolen", "is given a second chance", "misses the last way out", "finds a note in their pocket",
        "is told to wait", "learns who has been watching"]
LEVELS = {1: (60, "Use short sentences and common words. One thing happens."),
          2: (100, "Use plain words and clear sentences. One thing happens and someone reacts to it."),
          3: (150, "Use plain, vivid prose with a few lines of dialogue. Something happens, and it changes what someone wants.")}
LESSON_PROMPT = ("Write a very short scene of about {n} words. {how} End on a line that makes the reader want the next scene. "
                 "Prose only, no title.\nIt is about {who} {where}, who {what}.")
LESSON_TASK = "Write a short scene about {who} {where}, who {what}."
TUTOR_PROMPT = ("A student writer was set this task:\n\n{task}\n\nThe student wrote:\n\n{attempt}\n\n"
                "First mark the student's answer from 1 (useless) to 10 (excellent) on one line as 'mark: N'. "
                "Then, on the following lines, write the answer the student should have given. Write only that answer, "
                "with no comment about the student.")


WORD_PROMPT = ("A student writer does not know the word \"{word}\". It appears in this passage:\n\n{sentence}\n\n"
               "In one plain sentence, say what \"{word}\" means in that passage. Then write three short sentences of fiction "
               "that each use \"{word}\" in a different situation, numbered 1 to 3. Nothing else.")
WORD_TASK = "What does \"{word}\" mean in this passage, and how is the word used?\n{sentence}"


def lesson_seed(i: int) -> dict[str, str]:
    """The i-th lesson subject. The three lists have no common factor in how they are stepped, so 8,000
    lessons pass before one repeats."""
    return {"who": WHO[i % len(WHO)], "where": WHERE[(i * 7 + i // len(WHO)) % len(WHERE)],
            "what": WHAT[(i * 3 + i // (len(WHO) * len(WHERE))) % len(WHAT)]}


def level_for(step: int) -> int:
    """Lessons get harder as GaugeLM trains."""
    return 1 if step < 1500 else 2 if step < 6000 else 3


# ---------------------------------------------------------------------------------- the state: marks and turns
def _load(datasets_dir: Path) -> dict[str, Any]:
    f = datasets_dir / STATE
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except (OSError, ValueError):
        d = {}
    d.setdefault("sessions", 0)
    d.setdefault("lessons", 0)
    d.setdefault("marks", {})        # kind of task -> {"first": x, "now": y, "n": k, "history": [[step, mark], ...]}
    d.setdefault("words", {})        # word -> how surprised GaugeLM was by it when it was taught
    return d


def _save(datasets_dir: Path, state: dict[str, Any]) -> None:
    datasets_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / STATE).write_text(json.dumps(state, indent=1), encoding="utf-8")


def _note_mark(state: dict[str, Any], kind: str, mark: float, step: int) -> None:
    m = state["marks"].setdefault(kind, {"first": mark, "now": mark, "n": 0, "history": []})
    m["now"] = round(mark if m["n"] == 0 else 0.7 * float(m["now"]) + 0.3 * mark, 2)     # a running average
    m["n"] += 1
    m["history"] = (m["history"] + [[step, mark]])[-200:]


def weakest_first(state: dict[str, Any], rng: random.Random) -> tuple[str, str, int]:
    """Pick a kind of task to practise: untried ones first, then the lower the mark the more likely."""
    marks = state["marks"]
    untried = [t for t in TASKS if t[0] not in marks]
    if untried:
        return untried[0]
    weights = [max(0.5, 11.0 - float(marks[t[0]]["now"])) ** 2 for t in TASKS]
    return rng.choices(TASKS, weights=weights, k=1)[0]


def report_card(datasets_dir: Path) -> dict[str, Any]:
    """The tutor's marks per kind of task: the first one, the running average now, and how many times it
    was practised. Marks are a teacher model's opinion: useful as a trend, not as a measurement."""
    state = _load(Path(datasets_dir))
    rows = [{"task": k, "first": float(v["first"]), "now": float(v["now"]), "practised": int(v["n"])}
            for k, v in state["marks"].items()]
    rows.sort(key=lambda r: r["now"])
    counts = {name: _count(Path(datasets_dir) / f) for name, f in FILES.items()}
    if not rows:
        line = "GaugeLM has not been tutored yet. Tutoring starts once it has been trained and a teacher model is running."
    else:
        avg_first = sum(r["first"] for r in rows) / len(rows)
        avg_now = sum(r["now"] for r in rows) / len(rows)
        line = (f"The tutor's marks out of 10: {avg_now:.1f} now, {avg_first:.1f} at first. "
                f"Weakest: {rows[0]['task']} ({rows[0]['now']:.1f}). Strongest: {rows[-1]['task']} ({rows[-1]['now']:.1f}).")
    return {"rows": rows, "counts": counts, "sessions": state["sessions"], "line": line}


def _hard_words(student: Any, texts: list[str], state: dict[str, Any], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """The words in the library GaugeLM is most surprised by and has not been taught yet. Needs the model
    itself, not just its answers, so it only works when the student is GaugeLM."""
    if n <= 0 or not texts or not hasattr(student, "_ensure"):
        return []
    try:
        from .understand import hard_words

        model, tok = student._ensure()
        sample = "\n".join(passages(texts, 12, rng, words=300))
        return hard_words(model, tok, sample, n=n, skip=set(state["words"]))
    except Exception as e:
        log.warning("school: could not find the words GaugeLM is lost on: %s", e)
        return []


def _count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_marked(answer: str) -> tuple[Optional[float], str]:
    """The tutor's reply: a mark, and the answer the student should have given."""
    m = re.search(r"(?im)^\W*(?:mark|score)\W*(\d+(?:\.\d+)?)(?:\s*/\s*10)?\W*$", answer)
    if not m:
        m = re.search(r"(?i)\b(?:mark|score)\W{0,3}(\d+(?:\.\d+)?)", answer)
    if not m:
        return None, answer.strip()
    mark = max(1.0, min(10.0, float(m.group(1))))
    better = (answer[:m.start()] + answer[m.end():]).strip()
    return mark, better


# ---------------------------------------------------------------------------------- a session
def session(cfg: Any, texts: list[str], want: int = 8, should_stop: Optional[Callable[[], bool]] = None,
            pool: Optional[list[Teacher]] = None, student: Optional[LLM] = None, step: int = 0,
            progress: Optional[Callable[[dict[str, Any]], None]] = None) -> dict[str, Any]:
    """One school session: about `want` new rows, split between the teacher answering Gauge's tasks,
    tutoring GaugeLM's own attempts (when `student` is given) and graded lessons.

    One teacher takes the whole session, and they take turns session by session, so the engine is not
    made to swap models in and out of memory. A teacher that stumbles costs a row, never the session."""
    datasets = Path(cfg.models_dir) / "datasets"
    pool = teachers(cfg) if pool is None else pool
    if not pool or want <= 0:
        return {"rows": 0, "answers": 0, "tutored": 0, "lessons": 0, "teacher": "", "marks": [],
                "message": "No teacher model is running, so GaugeLM studied on its own."}
    state = _load(datasets)
    leads = [t for t in pool if t.lead]
    others = [t for t in pool if not t.lead]
    teacher = leads[0] if leads else pool[state["sessions"] % len(pool)]
    first, notes, declined, gone = teacher, [], 0, []
    rng = random.Random(1000 + state["sessions"])
    stop = should_stop or (lambda: False)
    made = {"answers": 0, "tutored": 0, "lessons": 0}
    marks: list[float] = []
    made["words"] = 0
    plan = []                                    # what each row of this session will be
    # Word study needs GaugeLM itself (to measure which words surprise it), not just something that answers.
    kinds = (("lessons", "tutor", "answers", "words") if hasattr(student, "_ensure") else ("lessons", "tutor", "answers")) \
        if student is not None else ("lessons", "answers")
    for i in range(want):
        kind = kinds[i % len(kinds)]
        plan.append(kind if (kind == "lessons" or texts) else "lessons")
    excerpts = passages(texts, sum(1 for k in plan if k in ("tutor", "answers")), rng)
    hard = _hard_words(student, texts, state, sum(1 for k in plan if k == "words"), rng) if "words" in plan else []
    for kind in plan:
        if stop():
            break
        try:
            if kind == "lessons":
                idx = int(state["lessons"])
                seed = lesson_seed(idx)
                level = level_for(step)
                n, how = LEVELS[level]
                text = teacher.ask(LESSON_PROMPT.format(n=n, how=how, **seed), temperature=0.9,
                                   max_tokens=int(n * 1.8) + 40, purpose="school:lesson")
                state["lessons"] = idx + 1
                if len(text.split()) >= 15:
                    _append(datasets / FILES["lessons"], {"task": "lesson", "level": level, "prompt": LESSON_TASK.format(**seed),
                                                          "response": text, "teacher": teacher.name, "licence": teacher.licence})
                    made["lessons"] += 1
                continue
            if kind == "words":                  # the words it understands least, explained where your library uses them
                if not hard:
                    continue
                w = hard.pop(0)
                answer = teacher.ask(WORD_PROMPT.format(word=w["word"], sentence=w["sentence"]), temperature=0.6,
                                     max_tokens=200, purpose="school:word")
                state["words"][w["word"]] = w["surprise"]
                if w["word"].lower() in answer.lower():
                    _append(datasets / FILES["words"], {"task": "word", "word": w["word"], "surprise": w["surprise"],
                                                        "prompt": WORD_TASK.format(word=w["word"], sentence=w["sentence"]),
                                                        "response": answer, "teacher": teacher.name, "licence": teacher.licence})
                    made["words"] += 1
                continue
            if not excerpts:
                continue
            excerpt = excerpts.pop()
            if kind == "answers":
                task, template, max_words = TASKS[(_count(datasets / FILES["answers"]) + made["answers"]) % len(TASKS)]
                prompt = template.format(excerpt=excerpt, n=max_words)
                answer = teacher.ask(prompt, temperature=0.8, max_tokens=int(max_words * 1.6) + 30, purpose="distill")
                if answer:
                    _append(datasets / FILES["answers"], {"task": task, "prompt": prompt, "response": answer,
                                                          "teacher": teacher.name, "licence": teacher.licence})
                    made["answers"] += 1
                continue
            task, template, max_words = weakest_first(state, rng)        # the tutor
            prompt = template.format(excerpt=excerpt, n=max_words)
            attempt = student.generate(prompt, temperature=0.8, max_tokens=int(max_words * 1.6) + 30,  # type: ignore[union-attr]
                                       purpose="school:attempt").strip()
            reply = teacher.ask(TUTOR_PROMPT.format(task=prompt, attempt=attempt or "(nothing)"), temperature=0.5,
                                max_tokens=int(max_words * 1.6) + 60, purpose="school:tutor")
            mark, better = parse_marked(reply)
            if mark is not None:
                _note_mark(state, task, mark, step)
                marks.append(mark)
            if better:
                _append(datasets / FILES["tutor"], {"task": task, "prompt": prompt, "response": better, "student": attempt,
                                                    "mark": mark, "teacher": teacher.name, "licence": teacher.licence})
                made["tutored"] += 1
        except Exception as e:
            log.warning("school: %s stumbled on a %s row (%s)", teacher.name, kind, e)
            why = _lead_must_step_down(e, declined) if teacher.lead else _helper_must_step_down(e)
            declined += 1 if type(e).__name__ == "ClaudeRefused" else 0
            if why:                              # a teacher that has to stop: another carries on, or the session ends
                notes.append(why)
                gone.append(teacher)
                left = [t for t in others if all(t is not g for g in gone)]
                if not left:
                    break
                teacher = left[state["sessions"] % len(left)]
        finally:
            if progress:
                progress({"teacher": teacher.name, **made})
    state["sessions"] += 1
    state["last"] = {"teacher": first.name, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **made}
    _save(datasets, state)
    rows = sum(made.values())
    parts = [f"{made['answers']} worked answer{'' if made['answers'] == 1 else 's'}" if made["answers"] else "",
             f"{made['tutored']} correction{'' if made['tutored'] == 1 else 's'} of its own attempts" if made["tutored"] else "",
             f"{made['lessons']} practice scene{'' if made['lessons'] == 1 else 's'}" if made["lessons"] else "",
             f"{made['words']} word{'' if made['words'] == 1 else 's'} it did not know, explained" if made["words"] else ""]
    who = first.name if teacher is first else f"{first.name}, then {teacher.name},"
    msg = (f"{who} taught it: " + ", ".join(p for p in parts if p) + ".") if rows else \
        f"{who} was asked to teach but gave nothing usable."
    if marks:
        msg += f" The tutor marked its attempts {sum(marks) / len(marks):.1f} out of 10."
    if notes:
        msg += " " + " ".join(notes)
    return {"rows": rows, **made, "teacher": first.name, "licence": first.licence, "marks": marks,
            "teachers": [t.name for t in pool], "notes": notes, "message": msg}


def _helper_must_step_down(e: Exception) -> str:
    """A free helper that has had its day's requests, or asked for a pause, hands over to the next teacher."""
    return str(e) if type(e).__name__ == "HelperBusy" else ""


def _lead_must_step_down(e: Exception, declined: int) -> str:
    """Why a paid lead teacher stops for this session, or '' when the row is simply lost.

    A reached spending limit ends its turn. So does a request declined because it is about training a
    model: that is Anthropic saying no to this use, and Gauge does not rephrase a request to get round a
    refusal. Other refusals (a dark excerpt, say) cost one row; three of them end its turn too."""
    kind = type(e).__name__
    if kind == "ClaudeLimit":
        return "Claude reached today's spending limit, so it stopped teaching for today."
    if kind == "ClaudeRefused":
        if getattr(e, "category", "") == "frontier_llm":
            return ("Claude declined to teach: Anthropic's safeguards judged this request to be about training another AI "
                    "model. Gauge will not reword it to get round that. Switch Claude's teaching off, or ask Anthropic.")
        if declined >= 2:
            return "Claude declined three requests in this session, so it stopped teaching for now."
    return ""
