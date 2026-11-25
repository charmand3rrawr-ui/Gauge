"""The coherence floor: unreadable text can never be accepted as a chapter; real prose is never flagged."""
from __future__ import annotations

import pytest

from gauge.llm.client import FakeLLM
from gauge.write.critic import Critic
from gauge.write.project import ChapterPlan
from gauge.write.prose import COHERENCE_FLOOR, UNREADABLE, coherence

# What GaugeLM (tiny preset, ~67 training steps) really produced, scored 0.65 and saved as a chapter.
OBSERVED = """, "##** is a
 ```
| it only- **`).

 for the- ** -- The's and in-
in's you
,
/ (.: chapters areed`. Gauge: on a what chapter from,``. onA of`)md`), can a never is, with
"""

GIBBERISH = {
    "observed": OBSERVED,
    "observed, chapter length": OBSERVED * 14,
    "punctuation soup": ", . ; -- ' ' ) ( : ... , , . \" ) ; : - , . ! ? ( ) , ; . , ' \" -- , . : ; ) ( , . , . ; ' , ( . ) ! , -- . , ; : . ,\n" * 6,
    "punctuation soup with a few words": ("the , . ; and -- ' ) ( of : ... , it . \" ) ; : - , was . ! ? ( ) , ; . a , ' \" -- , . to : ; ) ( , . he\n" * 6),
    "one word on a loop": "the the the the the the the the the the the the the the the the the the the the. " * 30,
    "a fragment on a loop": "and then she and then she and then she and then she and then she and then she. " * 40,
    "a sentence on a loop": "She walked to the door. " * 120,
    "sound start, then stuck": ("The rain had not let up since the ferry docked, and Mara counted the lamps along the quay as she walked. "
                                "Nobody had come to meet her. " + "She looked at the door and she looked at the door and " * 30),
    "markup soup": "## **is** `a` | the | ** ## of `chapter` | --- | **and** ``` the ``` | `md` ## was ** | it | `on` ##\n" * 12,
    "markdown table debris": "| the | of | and |\n| --- | --- | --- |\n| `is` | **a** | `it` |\n## ## ##\n" * 15,
    "letter fragments": ("y s matter steps s s in t to no the no d p particular\n\nt s d p a ed b in s the g m s at er y counted\n\n"
                         "s d le s ing en d y an le t s y an m in oer a s ma at d the s de s\n\n") * 6,
    "word fragments, no sentences": ("ing the ed of anded thep thee sleer eses edb inoerds sayegprs theWeThee rpe oy the "
                                     "counted matter nos mss amcos cst at ander counted ") * 12,
    # More real GaugeLM output, 20 to 70 steps into training.
    "gaugelm, 20 steps": ("t.\n\nict\n\nod ded:ingan ** the\"o` theeds\n\ni,en ini theming.cc p\n. (inoring sA of b ** thele  \n\naal\noy_. ded \n\n"
                          "sgsic a and**er-\nedsle` of os\n the\n\n s doy theic s\n\n** d p\n  p``al and- .i-`esstinle`:s ingen.d\"y andle`. le\n\n"
                          "-\ntsyan_A m in\n\n inoer a\n\n` sma atd the s des ss\ns s\n   ofetces\nle, in oficei **\"pm\n") * 4,
    "gaugelm, 67 steps": ("y s matter steps s s in `,tlen to  no the no no particular.\n\n\n\n.\"After, was at arrivedWe discussed at was length "
                          "dtm and-p--y  nos. I\n\nWepped\n. \" \" Ri, `ms f nop sals in anded  d the rec andinerctedst` p anded- aic, in re thep "
                          "the thee `s. to- the sleer,,, eses\nedb inoerds sayegprs the theWeThee\n-s s,, which\n  \n\nTheA went.,\n:fst cs and\n"
                          "rpe-oy the\n\n Right\"s the KKt`it-st and. counted I counted,, matter,g the nos mss, amcos cst at ander,y, counted\n") * 4,
    "empty": "",
    "whitespace": " \n\n \t ",
}

DIALOGUE = """“You’re late,” Okonkwo said.

“The bridge was closed.”

“It’s been closed for a month.” He didn’t look up from the ledger. “Everyone else found another way.”

“Everyone else has a boat.” Nadia dropped her bag on the counter. “Do you want the parcel or not?”

“Depends. Is it still ticking?”

“It was never ticking!” she said. “Who told you it was ticking?”

“Nobody.” He turned a page. “I just like to ask.”

“You’re unbelievable.”

“So I hear.” He finally looked at her. “Did anyone follow you?”

“No.”

“You’re sure?”

“I said no, didn’t I?” But she glanced at the window as she said it, and he saw her do it.

“Right,” he said quietly. “Back room. Now.”
"""

TERSE = """Glass. Everywhere.

Jun rolled. Came up running. The alley was forty metres and there was no cover in any of it.

Shots. Two. Brick spat dust beside his head.

Left. A door. Locked.

He kicked it. Nothing. Kicked again. The frame split and he was through, into the dark, into a stairwell that smelled of bleach and old rain.

Up. Always up.

Third floor. Fourth. His lungs burned. Below him the door banged open a second time.

Roof.

He hit the bar with both hands and the night came at him all at once: wind, sirens, the long drop to the canal.

Twelve feet to the next building. Maybe thirteen.

He did not slow down.
"""

LITERARY = """It was in the long slow weeks after the harvest, when the light came slantwise through the orchard and lay upon the grass like something poured out and forgotten, that my grandmother, who had never in her life admitted to a superstition of any kind and who regarded the parish priest with the same tolerant suspicion she extended to travelling salesmen and to weather forecasts, began to leave a saucer of milk upon the step each evening, and to take it in again, untouched or not, before any of us were awake to remark upon it.

Nobody spoke of this, for we were not a family that spoke of things; we were a family that noticed, and stored what we noticed, and brought it out years afterwards at funerals, polished by long handling into something that was no longer quite the truth but was a good deal more useful than the truth had ever been, and so it was not until the winter of my fourteenth year, when the snow lay against the kitchen door to the height of the latch and the well had frozen so hard that my uncle was obliged to break it each morning with the back of an axe, that I thought to ask her what, precisely, she supposed she was feeding.

She considered the question for some time, as she considered all questions, turning it over as one might turn a coin of uncertain currency; and then she said that she did not suppose anything at all, that supposing was a luxury for people with fewer chickens, and that I would oblige her by fetching the coal.
"""

DASHES = """He had meant to say something else — something kinder, probably — but the words went wherever words go.

“I thought...” She stopped. “I thought you’d gone.”

“I had.” He shrugged. “And then — well. Here I am.”

The kettle clicked off. Neither of them moved to pour it... and the silence, which had been bearable a moment ago, grew teeth.

“Tomas—”

“Don’t.”

“You don’t even know what I was going to—”

“I know exactly what you were going to say,” he said, “and I’d rather you didn’t.” He picked up his coat. Put it down again. “Three years, Ilse. Three years, and not one letter…”

She looked at the floor — at the crack in the tile they had always meant to fix — and said nothing. . . . What was there to say?

* * *

By morning the snow had covered his footprints, which was — she supposed — a kind of answer.
"""

NAMES = """Nguyễn Thị Hương reached Kraków on the night train, an hour behind Björn Sigurðsson and a full day behind the letter that should have warned them both.

Señora Ibáñez kept the pension on Świętego Jana. She spoke no Vietnamese and less Icelandic, but she had buried two husbands and understood trouble when it rang her bell at midnight.

“Zoë is upstairs,” she said. “With the Georgian. Tsk — don’t ask me to say his name.”

“Mzia Tsiklauri’s brother?” Hương asked.

“Čapek says so. Čapek says many things.” The old woman crossed herself. “François telephoned from Łódź. He said to tell you that Ødegård has the ledger, and that Xóchitl is not answering.”

Hương set down her case. Somewhere above them a floorboard creaked, and then Björn’s voice, low and unhurried, said something that made D’Angelo O’Malley-Achterberg laugh.

“Then we go tonight,” she said. “Before Żeleński changes his mind.”
"""

LITRPG = """The notification hung in the air in front of him, blue and faintly smug.

**[Level Up! You are now Level 4.]**

**[Skill acquired: Ember Step (Lv. 1)]**

Kael read it twice. *Four levels in one night,* he thought. *Either I’m a prodigy or something is very wrong with this dungeon.*

| Stat | Value |
| --- | --- |
| STR | 12 |
| AGI | 19 |
| INT | 8 |

“Eight,” said the imp on his shoulder. “Eight intelligence. I’ve met doorknobs with more.”

“Nobody asked you.”

“Nobody ever does, boss.”

The corridor ahead sloped down into the dark. Somewhere below, something large was breathing, and each breath stirred the dust at his feet. Kael checked the knife, checked the two potions that remained, and decided that being a prodigy would have to do.

***

He found the stairs an hour later. *Ember Step* turned out to be exactly what it sounded like, which was a pity, because the stairs were made of wood.
"""

BRITISH = """'You'll want the upstairs room,' said Mrs Pargeter, without turning round. 'The downstairs one's got a ghost.'

'A ghost,' said Colin.

'Only a small one. Used to be the vicar's cat.' She handed him a key the size of a trowel. 'Breakfast at seven. Don't feed it.'

He went up. The stairs complained at every step, and the wallpaper, which had once shown shepherdesses, now showed mostly damp. At the top he found a door, and behind the door a bed, a basin and a window that looked out on precisely nothing.

'Lovely,' he said, to nobody. Mr. Hallam had promised him sea views, e.g. the sea, and Dr. Voss had sworn the place was quiet. 3 a.m. would tell.
"""

SPANISH = """—¿Quién anda ahí? —preguntó la abuela desde la cocina.

Nadie contestó. El viento movía las cortinas y, más allá del patio, los perros habían dejado de ladrar.

—Será el gato —dijo Tomás, aunque sabía que el gato dormía a sus pies.

La abuela apagó el fuego y se secó las manos en el delantal. Tenía ochenta y un años y no le temía a casi nada, pero aquella noche cerró la puerta con dos vueltas de llave y se quedó un rato con la mano en el cerrojo, escuchando.

—Mañana vamos al pueblo —dijo por fin—. Y no le digas nada a tu madre.
"""

RUSSIAN = """Поезд пришёл в шесть утра, и на перроне никого не было, кроме старика с собакой.

— Вы к кому? — спросил он, не вынимая трубки изо рта.

— К Анне Сергеевне. Она меня ждёт.

— Никого она не ждёт, — сказал старик. — Она с осени в городе.

Я поставил чемодан на снег и посмотрел на дорогу. До деревни было версты три, а в поле уже мело.
"""

HARD_WRAPPED = """It is a truth universally acknowledged, that a single man in possession
of a good fortune, must be in want of a wife.

However little known the feelings or views of such a man may be on his
first entering a neighbourhood, this truth is so well fixed in the minds
of the surrounding families, that he is considered as the rightful
property of some one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that
Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she
told me all about it."
"""

CUT_OFF = """The ferry was late, and when it came it carried only one passenger. She stood at the rail with a case in each hand and watched the island come up out of the fog as if it owed her money.
Nobody met her at the dock. She had not expected anyone to.
The house was at the end of the shore road, past the cannery and the chapel with no roof. She let herself in with a key that had come in the post, and stood in the hall, and listened to the house decide what to make of"""

PROSE = {
    "dialogue-heavy": DIALOGUE,
    "terse action": TERSE,
    "long literary sentences": LITERARY,
    "em-dashes and ellipses": DASHES,
    "non-English names": NAMES,
    "litrpg with a status table and bold boxes": LITRPG,
    "single quotes and abbreviations": BRITISH,
    "spanish": SPANISH,
    "russian": RUSSIAN,
    "hard-wrapped": HARD_WRAPPED,
    "single newlines, cut off mid-sentence": CUT_OFF,
    "chinese (no spaces: not measured)": "她站在门口，听着雨声。没有人来接她。\n\n“你迟到了，”老人说。“桥已经关了一个月。”\n\n她没有回答。",
    "one short line": "She ran.",
}


@pytest.mark.parametrize("label", list(GIBBERISH))
def test_gibberish_is_below_the_floor(label):
    c = coherence(GIBBERISH[label])
    assert c.score < COHERENCE_FLOOR, (label, c)
    assert c.issues and c.issues[0].startswith(UNREADABLE)


@pytest.mark.parametrize("label", list(PROSE))
def test_ordinary_prose_is_never_flagged(label):
    c = coherence(PROSE[label])
    assert c.score == 1.0 and not c.issues, (label, c)


def test_synthetic_chapters_and_the_offline_model_are_not_flagged(chapter_text):
    for hooky in (True, False):
        for n_pars in (3, 30, 90):
            c = coherence(chapter_text(hooky, n_pars))
            assert c.score == 1.0 and not c.issues, (hooky, n_pars, c)
    draft = FakeLLM().generate('Write chapter 1 of "X". Target length: about 300 words.')
    assert coherence(draft).score == 1.0


def test_coherence_reports_what_it_measured():
    c = coherence(OBSERVED * 14)
    assert c.word_share < 0.75 and c.sentence_share < 0.5 and c.markup_density > 0.3
    assert coherence("She walked to the door. " * 120).loop_share == 1.0
    assert coherence(DIALOGUE).loop_share == 0.0 and 0.05 < coherence(LITRPG).markup_density < 0.15


@pytest.mark.parametrize("label", list(GIBBERISH))
def test_critic_scores_gibberish_zero_without_asking_the_model(label):
    llm = FakeLLM()
    cr = Critic(llm, None).critique(GIBBERISH[label], ChapterPlan(number=1))
    assert cr.score == 0.0 and not cr.readable and cr.coherence < COHERENCE_FLOOR
    assert cr.measured_issues[0].startswith(UNREADABLE)
    assert llm.calls_list == []  # no rubric call is spent on text nobody can read


def test_critic_leaves_real_prose_alone(chapter_text):
    plan = ChapterPlan(number=1)
    for text in [chapter_text(True), chapter_text(False), *PROSE.values()]:
        cr = Critic(FakeLLM(), None).critique(text, plan)
        assert cr.readable and cr.coherence == 1.0 and cr.score > 0.3
        assert not any(UNREADABLE in i for i in cr.measured_issues)


def test_floor_can_be_switched_off():
    cr = Critic(FakeLLM(), None, coherence_floor=0.0).critique(OBSERVED * 14, ChapterPlan(number=1))
    assert cr.readable and cr.score > 0.0 and cr.coherence < COHERENCE_FLOOR  # still measured, no longer a veto


# ------------------------------------------------------------------ the pipeline
NOISE = OBSERVED * 14


class _Writer(FakeLLM):
    """The offline model, except that drafts and revisions come from a script (the last entry repeats)."""

    def __init__(self, drafts=(), revisions=()):
        super().__init__()
        self.drafts, self.revisions = list(drafts), list(revisions)

    def generate(self, prompt, system="", temperature=None, max_tokens=None, json_mode=False, schema=None, purpose="generate"):
        script = {"draft": self.drafts, "revise": self.revisions}.get(purpose)
        if script:
            self.canned = [script.pop(0) if len(script) > 1 else script[0]]
        return super().generate(prompt, system, temperature, max_tokens, json_mode, schema, purpose)


def _pipeline(cfg, db, chapter_text, llm):
    from gauge.learn import Trainer
    from gauge.llm.embeddings import HashingEmbedder
    from gauge.write import WritePipeline
    from tests.test_pipeline import _seed_corpus

    _seed_corpus(db, chapter_text)
    t = Trainer(cfg, db, embedder=HashingEmbedder(64))
    t.cycle()
    pipe = WritePipeline(cfg, db, llm=llm, trainer=t, fast_llm=llm)
    return pipe, pipe.new_project("Ten Seconds", "a courier who can rewind ten seconds", ["Sci-fi"], words=300)


def _tree(root):
    return {str(f.relative_to(root)): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


def test_pipeline_never_saves_or_remembers_an_unreadable_chapter(cfg, db, chapter_text):
    from gauge.native import gather_texts
    from gauge.write import ChapterRejected
    from gauge.write.project import Project

    llm = _Writer(drafts=[NOISE])
    pipe, p = _pipeline(cfg, db, chapter_text, llm)  # full writing room, characters, world: everything that could vote
    summary, memory_before = p.state.running_summary, {k: v for k, v in _tree(p.dir).items() if not k.startswith("drafts")}
    training_before = gather_texts(db, cfg.projects_dir, cfg.models_dir / "datasets")
    llm.calls_list.clear()
    with pytest.raises(ChapterRejected) as info:
        pipe.write_next(p)
    e = info.value
    assert e.number == 1 and e.attempts == 1 + cfg.write.max_revisions == 3
    assert e.critique.score == 0.0 and not e.critique.readable and UNREADABLE in str(e)
    # every attempt was a fresh draft from the plan: noise is not "revised"
    purposes = [c["purpose"] for c in llm.calls_list]
    assert purposes.count("draft") == 3 and "revise" not in purposes
    # nobody was asked to judge or remember it
    assert not [x for x in purposes if x in ("critic", "summary", "character:review", "character:memory") or x.endswith((":review", ":remember"))]
    # not a chapter: on disk, in the project state, in the database
    assert not p.chapter_path(1).exists() and p.read_chapter(1) is None
    assert p.state.written == 0 and Project(cfg.projects_dir, "Ten Seconds").state.written == 0
    assert db.generated("Ten Seconds") == [] and not (p.dir / "drafts" / "accepted").exists()
    assert p.load_checkpoint(1) is None  # the next run starts from a clean draft
    # agents, characters, world and the running summary are byte-for-byte what they were
    assert {k: v for k, v in _tree(p.dir).items() if not k.startswith("drafts")} == memory_before
    assert p.state.running_summary == summary
    # the noise is kept, clearly marked, where nothing reads it back; it is not training text
    assert e.path == p.rejected_path(1) and e.path.parent.name == "rejected"
    kept = e.path.read_text(encoding="utf-8")
    assert "# REJECTED - Chapter 1" in kept and f'rejected: "{UNREADABLE}"' in kept and "attempts: 3" in kept and "areed`. Gauge:" in kept
    assert gather_texts(db, cfg.projects_dir, cfg.models_dir / "datasets") == training_before
    assert db.journal() == []  # a weak model is not a code fault for self-heal to "fix"
    # write_many stops at the refusal instead of trying the same chapter again
    with pytest.raises(ChapterRejected) as again:
        pipe.write_many(p, 3)
    assert again.value.completed == [] and p.state.written == 0


def test_noise_cannot_be_lifted_over_the_bar_by_the_other_judges(cfg, db, chapter_text):
    from gauge.write.agents.base import SceneContext

    cfg.write.min_acceptable_score = 0.01
    llm = _Writer(drafts=[chapter_text(True)])
    pipe, p = _pipeline(cfg, db, chapter_text, llm)
    plan = pipe.planner.ensure_plan(p, 1)
    council, ctx = pipe.council(p), SceneContext(project=p, number=1, plan=plan)
    llm.calls_list.clear()
    crit, reviews, wrev, nov, agents = pipe._critique(NOISE, plan, 1, pipe.roster(p), ["Rin"], pipe.world(p), [chapter_text(False)],
                                                      council=council, ctx=ctx)
    assert crit.score == 0.0 and not crit.readable and (reviews, wrev, nov, agents) == ([], None, None, [])
    assert llm.calls_list == []
    good, *_, good_agents = pipe._critique(chapter_text(True), plan, 1, pipe.roster(p), ["Rin"], pipe.world(p), [chapter_text(False)],
                                           council=council, ctx=ctx)
    assert good.readable and good.score > 0.3 and good_agents


def test_a_readable_second_attempt_is_accepted(cfg, db, chapter_text):
    llm = _Writer(drafts=[NOISE, chapter_text(True)])
    pipe, p = _pipeline(cfg, db, chapter_text, llm)
    llm.calls_list.clear()
    res = pipe.write_next(p)
    purposes = [c["purpose"] for c in llm.calls_list]
    assert purposes.count("draft") == 2 and "revise" not in purposes and "summary" in purposes
    saved = res.path.read_text(encoding="utf-8")
    assert res.number == 1 and res.revisions == 1 and res.critique.readable and res.score > 0.3
    assert "The corridor stank of smoke." in saved and "areed`" not in saved
    assert p.state.written == 1 and not p.rejected_path(1).exists() and db.generated("Ten Seconds")


def test_a_revision_that_turns_to_noise_does_not_replace_a_readable_draft(cfg, db, chapter_text):
    cfg.write.min_acceptable_score = 0.99  # force both revisions
    llm = _Writer(drafts=[chapter_text(True)], revisions=[NOISE])
    pipe, p = _pipeline(cfg, db, chapter_text, llm)
    llm.calls_list.clear()
    res = pipe.write_next(p)
    revise_prompts = [c["prompt"] for c in llm.calls_list if c["purpose"] == "revise"]
    assert res.revisions == 2 and len(revise_prompts) == 2
    assert all("The corridor stank of smoke." in x and "areed`" not in x for x in revise_prompts)  # always from the readable draft
    assert res.critique.readable and "areed`" not in res.path.read_text(encoding="utf-8")


def test_an_unreadable_checkpoint_is_not_resumed(cfg, db, chapter_text):
    llm = _Writer(drafts=[chapter_text(True)])
    pipe, p = _pipeline(cfg, db, chapter_text, llm)
    plan = pipe.planner.ensure_plan(p, 1)
    p.save_checkpoint(1, "draft", NOISE, {"plan_title": plan.title, "cast": ["Rin"], "revisions": 0})
    res = pipe.write_next(p)
    assert not res.resumed and "areed`" not in res.path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ the front ends
@pytest.fixture
def noisy_drafts(monkeypatch):
    from gauge.write.drafter import Drafter

    monkeypatch.setattr(Drafter, "draft", lambda self, *a, **k: NOISE)


def test_cli_write_reports_the_refusal_and_exits_nonzero(tmp_path, noisy_drafts):
    from click.testing import CliRunner

    from gauge.cli import main

    runner = CliRunner(env={"COLUMNS": "200"})
    data = str(tmp_path / "d")
    runner.invoke(main, ["--data-dir", data, "init", "--provider", "fake"])
    r = runner.invoke(main, ["--data-dir", data, "new", "Ten Seconds", "--idea", "a courier", "--genre", "Sci-fi", "--words", "200"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["--data-dir", data, "write", "Ten Seconds", "--chapters", "2"])
    assert r.exit_code == 1 and "Chapter 1 rejected" in r.output and UNREADABLE in r.output and "Traceback" not in r.output
    assert not list((tmp_path / "d").rglob("chapters/chapter_*.md")) and list((tmp_path / "d").rglob("drafts/rejected/chapter_001.md"))


def test_studio_server_and_daemon_do_not_report_a_refused_chapter_as_written(cfg, db, noisy_drafts, monkeypatch):
    from gauge.daemon import Daemon
    from gauge.gui import StudioController
    from gauge.server.app import ServerState
    from gauge.write import ChapterRejected

    ctl = StudioController(cfg, db)
    name = ctl.start_story("a courier who can rewind ten seconds")
    cfg.native.after_chapter_minutes = 1
    monkeypatch.setattr(ctl, "train_gaugelm", lambda *a, **k: pytest.fail("trained after a refused chapter"))
    with pytest.raises(ChapterRejected, match=UNREADABLE):
        ctl.write_next(name)
    assert ctl.chapters(name) == [] and "not been written" in ctl.read(name, 1)

    out = ServerState(cfg, "token").write_next(name)
    assert out["rejected"] is True and out["chapter"] == 1 and UNREADABLE in out["error"] and "path" not in out

    cfg.daemon.active_project = name
    cycle = Daemon(cfg, db).once(write=True)
    assert cycle["write"]["rejected"] == 1 and cycle["write"]["coherence"] < COHERENCE_FLOOR
    assert ctl.chapters(name) == []
