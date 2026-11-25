# The character school: teaching Gauge to build a character

A sheet that says "name, job, brave but stubborn" is not a character, and a model asked for a
character with nothing to aim at will hand back exactly that. The character school fixes this in three
steps: it **writes down everything a newly built character needs**, it lets you **teach by example**
with finished characters of your own, and it **uses what it was taught** every time a character is
created and every time GaugeLM trains.

Code: `gauge/write/blueprint.py` (the breakdown and its checks) and `gauge/learn/character_school.py`
(the examples and the training rows). For how characters behave once they exist, see
[CHARACTERS.md](CHARACTERS.md).

## What a newly built character needs

Nine sections. Each item says what it is and the test a good answer passes. Leads (the protagonist and the antagonist) need all of it; supporting characters most of it; minor characters only what is marked.

### Identity

Who this is in one breath. If this section is vague, everything after it will be too.

- **Name** (everyone). What they are called. *Test:* Pronounceable, distinct from the rest of the cast (no two names starting the same way).
- **Other names** (leads). What different people call them: a nickname, a title, a surname. *Test:* Each alias says something about who uses it.
- **Role in the story** (everyone). Protagonist, antagonist, supporting or minor. *Test:* One of the four; it decides how much of the rest is needed.
- **Concept** (everyone). The character in one vivid line. *Test:* A specific person in a specific bind, not a job title: 'a courier who owes the wrong people'.

### Outside

What a stranger would notice. Readers remember one detail, so make it a particular one.

- **Appearance** (leads and supporting). How they look, in two concrete sentences. *Test:* Contains one detail nobody else in the cast has, and nothing a police sketch would say.
- **Habits and tells** (leads and supporting). What their body does: a ritual, a nervous tell, how they take up a room. *Test:* Could be filmed. Something a scene can show instead of naming a feeling.

### Past

Only the past that still acts on the present. Not a biography.

- **Backstory** (leads and supporting). The few events that made them who they are now. *Test:* Every sentence explains something they do today. Holds one thing they hide.
- **Wound** (leads and supporting). The specific hurt they have not got over. *Test:* An event, not an adjective: 'her mother left without a word', not 'abandonment issues'.

### Engine

What makes them act. This is the section a plot is built from; it has to hold tension.

- **Want** (leads and supporting). The concrete thing they are trying to get. *Test:* A camera could see them get it. It is what they would tell you if you asked.
- **Need** (leads and supporting). What would actually mend them. *Test:* Different from the want, and in its way. They would deny it.
- **The lie they believe** (leads). The false belief they live by. *Test:* The wound taught it; the flaw protects it; the arc is about losing it.
- **Flaw** (leads and supporting). How they get in their own way. *Test:* Causes real trouble on the page. Not a charming quirk, not a strength in disguise.
- **Stakes** (leads). What they lose if they fail. *Test:* Personal and specific. Said in words they would use.
- **Fears** (leads and supporting). What they are afraid of. *Test:* At least one they would never admit.
- **Values** (leads and supporting). What they hold to. *Test:* Two of them should be able to collide.
- **Lines they will not cross** (leads and supporting). What they refuse to do, whatever it costs. *Test:* The story should be able to push them right up to one.

### Temperament

The fifteen trait values from 0 to 1. A character with every dial at the middle is nobody.

- **Trait values** (leads and supporting). Openness, conscientiousness, extraversion and the rest, each 0 to 1. *Test:* At least three sit at an extreme (0.2 or lower, 0.8 or higher), and two of the extremes pull against each other.
- **Contradiction** (leads). The thing about them that does not fit. *Test:* Surprising and true: the enforcer who tends orchids. It makes them a person, not a type.

### Voice

How they sound. Cover the names in a page of dialogue and you should still know who is speaking.

- **Register** (leads and supporting). Formal, casual, clipped, florid, street, archaic... *Test:* One word a reader would agree with.
- **Sentence length** (leads and supporting). Short, medium or long. *Test:* Matches how they think.
- **Vocabulary** (leads and supporting). Where their words come from. *Test:* Names a source (a trade, a region, a book) and something they never say.
- **Verbal tics** (leads and supporting). Habits of speech. *Test:* Tied to a feeling: what they do when scared, lying, cornered.
- **Catchphrases** (leads). Lines that are theirs. *Test:* Rare. One is plenty.
- **Swears** (everyone). Yes or no. *Test:* Decided on purpose.
- **Humour** (leads and supporting). How they are funny, or fail to be. *Test:* Deflecting, dry, cruel, none...
- **Will not talk about** (leads and supporting). Subjects they steer away from. *Test:* At least one leads back to the wound.

### Capability

What they can do, so that problems are solved by who they are and not by luck.

- **Skills** (leads and supporting). What they are good at. *Test:* Each comes with a limit or a price. Something they are bad at is worth more than another talent.

### Situation

Where they stand when the story finds them.

- **Mood** (leads and supporting). Their state at the start. *Test:* One word, not 'fine'.
- **Goal right now** (leads and supporting). What they are doing this week. *Test:* Smaller than the want, and on the way to it.
- **Knows** (leads). Facts they hold that matter to the plot. *Test:* At least one that someone else wants.
- **Secrets** (leads and supporting). What they are hiding, and from whom. *Test:* Would change a relationship if it came out.

### Connections and change

Nobody is a character alone, and nobody interesting stays the same.

- **Relationships** (leads and supporting). How they stand with the other people in the cast. *Test:* A line of stance for each: 'older brother she resents and would die for'. At least one is unresolved.
- **Arc** (leads). Where they start, what presses on them, and where it could take them. *Test:* Names the change, and what it would cost. Could also end badly.

The same list is what Gauge checks against. `gauge character check PROJECT` prints, for each character,
which items are complete, which are **missing**, and which are **too thin** to count (an appearance of
one word, fewer than three extreme trait values, a need that only repeats the want).

## Teaching it

### From Gauge Studio

**Teach Gauge**, then **Characters...** (or the **Teach** menu, **Teach character building...**):

| button | what it does |
| --- | --- |
| **What a character needs** | opens the breakdown above to read |
| **Save a blank sheet...** | saves a fill-in sheet as a text file; open it in Notepad and complete it |
| **Add character sheets...** | adds your filled-in sheets; each one is checked, and a sheet with gaps is refused with the list of what is missing |
| **Learn from this story's characters** | takes the main characters of the story that is open and keeps the complete ones as examples |

A character sheet added with the books under **Add files...** is recognised and sent to the school
instead of the library.

### The sheet

One `key: value` per line, lists separated by `;`, lines starting with `#` ignored. A value may run on
to the next line. Trait values go from 0 to 1 (`8` and `80%` are read as 0.8); leave one out and it
stays at 0.5.

```
# Gauge character sheet
name: Marina Vale
role: antagonist
concept: a fence who launders stolen memories and has started keeping the best ones
appearance: Silver rings on every finger of the left hand and none on the right. She wears a dead man's coat, cut down to fit.
habits: turns a ring when she is pricing something; never sits with her back to a window
wound: her brother looked at her and did not know her face
want: to own every memory of the market before the Broker does
need: to be known by one person who was not paid for it
lie: anything that can be forgotten was never really yours
warmth: 0.15
cynicism: 0.9
relationship: Rin Okada = the courier she means to own and half wants to warn | trust 0.3, fear 0.2
...
```

A JSON file in the shape `gauge character show` stores is accepted too.

### From the command line

```powershell
.\gauge.bat character blueprint                      # the breakdown; --role minor for one role's checklist
.\gauge.bat character sheet --name "Marina Vale" --role antagonist --out marina.txt
.\gauge.bat character teach marina.txt --genre thriller     # files or folders of sheets
.\gauge.bat character promote "Ten Seconds"                 # learn from your own story's cast
.\gauge.bat character study "Great Expectations" "Miss Havisham" --role antagonist
.\gauge.bat character school                                # what it is learning from; --remove NAME, --export
.\gauge.bat character check "Ten Seconds" --fix             # what each character lacks; --fix fills it
.\gauge.bat character add "Ten Seconds" marina.txt          # put a hand-built character into a story
.\gauge.bat character sheet --from "Ten Seconds" "Rin Okada" --out rin.txt   # an existing character, as a sheet to edit
```

`study` reads a character out of a book in your library. It is refused for third-party copyrighted
books, the same as everywhere else in Gauge: only public-domain or openly licensed books, and files
you imported yourself, are studied (see [CORPUS.md](CORPUS.md)).

### The bar

An example is admitted when it fills at least 80% of the blueprint for its role (a thin item counts for
half). A sheet with gaps would teach Gauge to leave gaps. `--force` admits one anyway. Adding a
character with the same name from the same source replaces the earlier one. Only what belongs on a
sheet is kept: memories, mood history and chapter numbers from the story are left behind.

## What Gauge does with it

**When a character is created** (`gauge character new`, `sync`, and the cast built with a new story),
the request carries the checklist for that role and the closest examples from the school: same role
first, then shared genres, then the most complete, with your own sheets winning a tie. The model is
told to match their depth and not to reuse their names, details or wording. The result is then checked
against the blueprint, and anything missing or thin is asked for once more, by name, with its test.
Parts that passed are never overwritten, and trait values fixed with `--seed` are never changed.

**When GaugeLM trains**, the school's lessons are part of the corpus
(`data/models/datasets/characters.jsonl`, rewritten whenever the school changes and at the start of
every training session):

| rows | teach |
| --- | --- |
| the breakdown, section by section and item by item | what a character needs and the test for each part; present even before any example is added |
| a whole sheet from a name, role and one-line concept | the task Gauge actually performs |
| one item at a time for each example | the parts, which is how GaugeLM fills a sheet |
| a sheet with three items removed | spotting what is missing and saying what would fix it |

With GaugeLM as the writer the examples are not put in the request, because its context is too small
to hold them; it learns them through training instead.

## Switches

| switch | default | meaning |
| --- | --- | --- |
| `write.character_examples` | 2 | finished sheets shown when a character is created (0 to 4) |
| `write.character_repair` | on | check each new character against the blueprint and fill the gaps once |

## What this does not do

It does not make a small model invent like a large one. It gives any model a clear target and good
examples, and it catches the sheet that came back hollow. The examples are the real lever: five
finished characters of your own teach more than any setting here.
