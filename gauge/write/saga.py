"""A whole novel: its length decided up front, planned in volumes, and written to the end.

Gauge plans five chapters at a time. That is how a serial stays lively, and it is also how one wanders:
with no end in sight a story neither builds toward anything nor finishes. For a novel of a set length
this module adds what is missing.

* **The plan.** The novel is divided into volumes (fifty chapters each by default). One request to the
  model gives each volume a title, what it is about, its turning point and the state it must end in,
  with the last volume resolving the promise the story made. If the model cannot be asked, a plain
  structural plan is used instead, so a run never fails for want of one.
* **Where we are.** Before every stretch is outlined, the outliner is told which volume it is in, what
  that volume must do, how many chapters are left in the volume and in the novel, and, near the end,
  that it is time to close things rather than open them.
* **The end.** Nothing is outlined past the last chapter, and the run stops there.

`write_novel` is the unattended run: chapter after chapter until the end, a stop request or repeated
failure. It keeps no state of its own; the story does. Stopping and starting again carries on from
the next unwritten chapter.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

PER_VOLUME = 50
STOP_FILE = "STOP"

PLAN_PROMPT = """Plan the whole of a serial web novel of {chapters} chapters, divided into {count} volumes of about
{per} chapters each.

BIBLE:
{bible}

STORY SO FAR ({written} chapters written):
{summary}

Each volume must escalate from the one before, change something for good, and end in a state that forces the next.
The last volume must resolve the promise the story made to its reader and bring the main character's arc to its end.
Respond with ONLY JSON: {{"volumes": [{{"title": "2-5 words", "about": "one sentence: what this volume is about",
"turn": "its turning point", "ends_with": "the state everything is in when it ends"}}, ...exactly {count} volumes]}}"""

_SHAPE = ["the promise is made and the first price is paid", "the world widens and the first real loss lands",
          "an ally is gained and a certainty is lost", "the enemy's true size becomes clear", "a victory that costs too much",
          "everything built so far is put at risk", "the lowest point: what was relied on fails",
          "the turn: a hidden strength, earned earlier, is used", "the last approach, with every thread drawn in",
          "the ending: the promise is paid in full and the cost is counted"]


def volume_count(chapters: int, per: int = PER_VOLUME) -> int:
    return max(1, -(-int(chapters) // per))


def _spans(chapters: int, count: int) -> list[tuple[int, int]]:
    """Chapter ranges for `count` volumes, as even as they can be, covering 1..chapters exactly."""
    base, extra = divmod(chapters, count)
    out, first = [], 1
    for i in range(count):
        size = base + (1 if i < extra else 0)
        out.append((first, first + size - 1))
        first += size
    return out


def plain_plan(chapters: int, per: int = PER_VOLUME) -> list[dict[str, Any]]:
    """A structural plan made without a model: the shape of a long story, stretched over the volumes."""
    count = volume_count(chapters, per)
    vols = []
    for i, (first, last) in enumerate(_spans(chapters, count)):
        beat = _SHAPE[min(len(_SHAPE) - 1, round(i * (len(_SHAPE) - 1) / max(1, count - 1)))] if count > 1 else _SHAPE[-1]
        vols.append({"number": i + 1, "first": first, "last": last, "title": f"Volume {i + 1}", "about": beat,
                     "turn": "a reversal that cannot be undone", "ends_with": "a new and harder situation" if i < count - 1
                     else "the story's promise kept and its cost counted"})
    return vols


def plan_novel(project: Any, llm: Any, chapters: int, per: int = PER_VOLUME) -> list[dict[str, Any]]:
    """Decide the novel's length and plan it in volumes. Replaces any earlier plan, keeping written chapters."""
    from ..llm import generate_json

    chapters = max(int(chapters), int(project.state.written), 1)
    count = volume_count(chapters, per)
    vols = plain_plan(chapters, per)
    try:
        d = generate_json(llm, PLAN_PROMPT.format(chapters=chapters, count=count, per=per, bible=project.bible_markdown(),
                                                  written=project.state.written, summary=project.story_so_far(400) or "(not begun)"),
                          temperature=0.7, max_tokens=min(6000, 160 * count + 400), purpose="saga:plan")
        got = [v for v in (d.get("volumes") or []) if isinstance(v, dict)]
        for v, g in zip(vols, got):                      # as many as came back; the rest keep the structural plan
            for key in ("title", "about", "turn", "ends_with"):
                if str(g.get(key) or "").strip():
                    v[key] = str(g[key]).strip()[:300]
    except Exception as e:
        log.warning("could not plan the novel with the model (%s); using a structural plan", e)
    project.state.planned_chapters = chapters
    project.state.volumes = vols
    project.save()
    return vols


def volume_at(project: Any, n: int) -> Optional[dict[str, Any]]:
    for v in project.state.volumes or []:
        if int(v["first"]) <= n <= int(v["last"]):
            return v
    return None


def where(project: Any, n: int) -> str:
    """What the outliner is told about chapter n's place in the whole novel. Empty when no length is planned."""
    total = int(project.state.planned_chapters or 0)
    if total <= 0 or n > total:
        return ""
    v = volume_at(project, n)
    left = total - n + 1
    lines = [f"THE WHOLE NOVEL is {total} chapters. This stretch starts at chapter {n}; {left} chapters remain."]
    if v:
        vols = project.state.volumes
        lines.append(f"VOLUME {v['number']} of {len(vols)} (chapters {v['first']}-{v['last']}), \"{v['title']}\": {v['about']} "
                     f"Its turning point: {v['turn']}. It must end with: {v['ends_with']}. "
                     f"{int(v['last']) - n + 1} chapters of this volume remain.")
        if int(v["number"]) < len(vols) and int(v["last"]) - n < 6:
            nxt = vols[int(v["number"])]
            lines.append(f"The volume is ending: bring it to that state, and set up the next (\"{nxt['title']}\": {nxt['about']}).")
    if left <= 5:
        lines.append("THESE ARE THE LAST CHAPTERS OF THE NOVEL. Open nothing new. Pay off every promise still standing, bring the "
                     "main character's arc to its end, and make the final chapter an ending, not a cliffhanger.")
    elif left <= 25:
        lines.append("The novel is in its final stretch: close threads rather than open them, and gather everyone for the ending.")
    return "\n".join(lines)


def stop_requested(project_dir: Path) -> bool:
    """A file called STOP beside the story asks a running novel to finish its chapter and stop."""
    return (Path(project_dir) / STOP_FILE).exists()


def write_novel(pipe: Any, project: Any, chapters: Optional[int] = None, words: Optional[int] = None,
                min_words: Optional[int] = None, should_stop: Optional[Callable[[], bool]] = None,
                progress: Optional[Callable[[dict[str, Any]], None]] = None, max_failures: int = 3,
                pause_after_failure: float = 5.0) -> dict[str, Any]:
    """Write the novel to its last chapter. Returns a summary of this run.

    `chapters` sets (or changes) the planned length, `min_words` the shortest a chapter may be, `words`
    the length to aim for. It stops when the last chapter is written, when `should_stop` says so or a
    STOP file appears (always between chapters, never mid-chapter), or after `max_failures` failures in a
    row. Whatever stops it, running it again carries on from the next unwritten chapter."""
    st = project.state
    if min_words is not None:
        st.min_chapter_words = max(0, int(min_words))
    if words:
        st.target_chapter_words = int(words)
    total = int(chapters or st.planned_chapters or 0)
    if total <= 0:
        raise ValueError("say how many chapters the novel should have")
    if total != st.planned_chapters or not st.volumes:
        plan_novel(project, pipe.fast_llm, total)
        # Chapters outlined before there was a plan are outlined again, so that they belong to it.
        st.outline = [c for c in st.outline if c.number <= st.written]
    project.save()
    stop_file = Path(project.dir) / STOP_FILE
    if stop_file.exists():
        stop_file.unlink()                               # a request left over from last time
    started, wrote, failures, short, counts, times = time.monotonic(), 0, 0, 0, [], []
    reason = "finished"
    while st.written < st.planned_chapters:
        if (should_stop is not None and should_stop()) or stop_requested(project.dir):
            reason = "stopped"
            break
        t0 = time.monotonic()
        try:
            res = pipe.write_next(project)
        except Exception as e:
            failures += 1
            log.warning("chapter %d failed (%d in a row): %s", st.written + 1, failures, e)
            try:
                pipe.db.add_journal("novel", f"chapter {st.written + 1}: {type(e).__name__}: {e}", "")
            except Exception:
                pass
            if failures >= max_failures:
                reason = f"stopped after {failures} failures in a row: {e}"
                break
            time.sleep(pause_after_failure * failures)
            continue
        failures = 0
        wrote += 1
        took = time.monotonic() - t0
        times.append(took)
        counts.append(res.words)
        if st.min_chapter_words and res.words < st.min_chapter_words:
            short += 1
        if progress:
            mean = sum(times[-20:]) / len(times[-20:])
            progress({"chapter": res.number, "of": st.planned_chapters, "words": res.words, "seconds": round(took, 1),
                      "left": st.planned_chapters - st.written, "hours_left": round(mean * (st.planned_chapters - st.written) / 3600, 1),
                      "title": res.title, "volume": (volume_at(project, res.number) or {}).get("number")})
    done = st.written >= st.planned_chapters
    out = {"wrote": wrote, "written": st.written, "of": st.planned_chapters, "finished": done, "reason": reason if not done else "finished",
           "words": sum(counts), "shortest": min(counts) if counts else 0, "below_minimum": short,
           "seconds": round(time.monotonic() - started, 1),
           "seconds_per_chapter": round(sum(times) / len(times), 1) if times else 0.0}
    if done:
        out["message"] = (f"The novel is finished: {st.written} chapters. This run wrote {wrote} of them, {sum(counts):,} words"
                          + (f", the shortest {min(counts):,} words." if counts else "."))
    else:
        left = st.planned_chapters - st.written
        out["message"] = (f"Stopped at chapter {st.written} of {st.planned_chapters} ({reason}). {left} to go; start again to carry on"
                          + (f", about {out['seconds_per_chapter'] * left / 3600:.1f} hours at this pace." if times else "."))
    if short:
        out["message"] += f" {short} chapter{'s' if short != 1 else ''} came out below the minimum of {st.min_chapter_words:,} words."
    return out


def status(project: Any) -> dict[str, Any]:
    st = project.state
    v = volume_at(project, st.written + 1) if st.planned_chapters else None
    return {"planned": st.planned_chapters, "written": st.written, "left": max(0, st.planned_chapters - st.written),
            "min_words": st.min_chapter_words, "volumes": len(st.volumes or []), "volume": v,
            "line": (f"{st.written} of {st.planned_chapters} chapters written"
                     + (f"; in volume {v['number']} of {len(st.volumes)}, \"{v['title']}\"" if v else "")
                     + (f"; no chapter under {st.min_chapter_words:,} words" if st.min_chapter_words else "") + ".")
            if st.planned_chapters else "No length is planned for this story; it is open-ended."}


def plan_markdown(project: Any) -> str:
    lines = [f"# {project.state.name}: {project.state.planned_chapters} chapters in {len(project.state.volumes or [])} volumes", ""]
    for v in project.state.volumes or []:
        lines += [f"## Volume {v['number']}: {v['title']} (chapters {v['first']}-{v['last']})", f"- About: {v['about']}",
                  f"- Turning point: {v['turn']}", f"- Ends with: {v['ends_with']}", ""]
    return "\n".join(lines)


__all__ = ["plan_novel", "plain_plan", "where", "write_novel", "status", "plan_markdown", "volume_at", "stop_requested", "json"]
