"""A starter set of training rows written by hand for Gauge.

Gauge makes its own training data in volume: local teacher models answer its tasks, correct GaugeLM's
attempts and write practice scenes (`school.py`), and the book finder brings in novels. This file is a
small, careful seed for before any of that has run: original scenes and worked answers in exactly the
shapes the school uses, so the very first training session already has well-formed examples of every
task.

It is small on purpose (a few thousand words). It shapes behaviour; it is not a corpus.

Where it came from, plainly: these rows were written for this project by Claude, an AI model made by
Anthropic, in a working session with the project's author. They are original; nothing is copied from
any book. Anthropic's terms limit using Claude's output to train AI models that compete with its
services, and whether a small story model of your own falls under that is your decision, not Gauge's.
For that reason the set is **off unless you switch it on** (`native.seed_data`), every row is labelled
with its source, and it lives in its own file (`datasets/seed.jsonl`) so it can be left out or removed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SOURCE = "written for Gauge by Claude (Anthropic)"
LICENCE = "original text; see Anthropic's terms on training models with Claude's output"
FILE = "seed.jsonl"

# Practice scenes. The number is the lesson's place in the school's own list of subjects
# (school.lesson_seed), so each prompt is exactly the one a teacher model would be given.
SCENES: dict[int, str] = {
    11: '''Ilse had drawn every wall of the clockmaker's workshop twice. She knew where each clock hung and how many steps lay between them. So when she looked up from her paper and saw a narrow green door between the tall clock and the cuckoo, her pencil stopped. It had not been there an hour ago. All the clocks ticked on as if nothing had changed. Then, behind the green door, one more clock began to strike.''',
    148: '''Corin went in through the dark kitchen and climbed toward the one lit window. The ring was on the table beside a sleeping old woman, just as he had been told. He had it in his hand when she coughed and said, without opening her eyes, "Is that you, Peter?" He stood very still. Then he set the ring back by her cup and pulled the blanket up to her chin. On the stairs he heard her say, "Peter never was so quiet."''',
    285: '''The old sailor did not remember walking in, and there was no door to walk out of. Only shelves, and a woman at a desk who did not look up. "You lost a ship," she said. "Forty years ago." "I did." "Would you take the wheel again, on that night, knowing the rocks?" His hands began to shake. "Yes." She slid a book across the desk. On its cover was the name of his ship, and the pages were still wet.''',
    422: '''The garden had swallowed the path, so Mara and Wynn found the letter by stepping on it. It was sealed, and the name on it was their mother's. "We should take it to her," said Wynn. Mara had already broken the seal. She read it once and her face went still. "What does it say?" Mara folded it small and put it inside her shoe. "It says we have a brother," she said, "and he is coming on Sunday."''',
    559: '''The stranger had left the inn without paying and taken the night train to be far away by morning. He was almost asleep when the guard touched his shoulder. "You are to wait at the next station, sir." "Wait for whom?" The guard only shrugged and moved on. The train slowed. On the empty platform, under the one lamp, stood the innkeeper's small daughter, holding his forgotten hat in both hands.''',
    696: '''The tailor's one customer was a rich man who paid for everything in advance. They were walking the lake path together when the ice cracked twice. The rich man went through near the shore. Farther out, a boy with a sledge went through as well. "My coat!" the rich man shouted. "Pull me by the coat you made!" The tailor looked at the good grey cloth, and then at the small red hat out on the ice. He took off his own coat and ran past the rich man.''',
    833: '''Every night the king's cook carried the last tray up the back stairs, and every night the ninth step creaked under her. Tonight it creaked twice. She stopped. The stairs were silent. She went on, counting. Her foot, then nothing. Her foot, then, very softly, another. She did not turn round. She set the tray down on the landing, lifted the lid from the soup, and said, "If you are hungry, say so. If you are not, I have a ladle."''',
    970: '''At the crossing the signs had been taken down for the war and never put back. A girl sat on the wall, eating an apple. "Which way to Harrow Mill?" the soldier asked. She pointed with the apple. Then she looked at his empty sleeve and his boots. "Did we win?" she said. He opened his mouth. He thought of the river, and the men still in it. "I'll tell you when I get home," he said, and took the road she had shown him.''',
    1107: '''The new teacher stood at the rail and could not see the water. Somewhere ahead was the island and a school of nine children. Then a voice in the fog said, "Miss Arden." She turned. No one. "Miss Arden," it said again, from the other side, lower now, and pleased. The ferryman did not seem to hear. "Who knows I am coming?" she asked him. He spat over the side. "Everyone," he said. "The last one told them you would."''',
    1244: '''The boy had the wrong papers for the border and he knew it. The guard read them slowly, then looked at the boy's shoes, which were his dead brother's and two sizes too big. "Name," said the guard. The boy gave his brother's name. The guard said nothing for a long time. Then he pushed the papers back across the table, face down. "I did not hear that," he said. "Go out. Come in again. And this time tell me the name your mother calls you."''',
    1381: '''Nobody at the wedding was smiling, least of all the bride. The lighthouse keeper, who had only come for the cake, felt sorry for them all. So when the old aunt asked him if the weather would hold, he said, "No. A great storm tonight. No boat will leave for a week." It was not true. But the groom stood up, white in the face, and said, "Then I cannot sail on Friday." And the bride began, very quietly, to laugh.''',
    1518: '''The water was up to Pim's knees and every street looked the same. He put his cold hands in his coat pockets and found paper. He did not remember any paper. In his mother's quick writing it said: IF YOU ARE LOST, FIND THE BAKER WITH THE BLUE DOOR. Pim looked up. Across the flooded street was a blue door, and it was already opening.''',
    1655: '''The kitchen was full of wet men and shouting, because two boats had come in and one had not. Dagny's father sat by the fire with his hand cut open. Her uncle stood at the door, calling for someone small enough to go down into the hull of the wrecked boat on the rocks. Her father held out his bleeding hand to her. Her uncle held out a rope. "Dagny," they both said. She picked up the clean cloth, and then she put it into her father's good hand. "Hold it tight," she said, and took the rope.''',
    1792: '''At three in the morning the corridor was empty and Nessa's shoes were loud on it. Outside room twelve lay a brown glove, a man's, still warm. She picked it up. Room twelve had been empty since Tuesday. She pushed the door a little. The bed was made. The window was open. And on the pillow, laid as neatly as a gift, was the other glove.''',
    1929: '''The crows had told her not to go down, but crows say that about every cellar. In the dark her elbow met a shelf, and something fell and broke with a small bright sound. She lit her candle. It was a blue glass bird, in two pieces. Upstairs the crows had gone quite silent. Then one of them spoke through the grating, in a voice she had never heard it use. "That was his," it said. "Now he will have to come back for it."''',
    2066: '''When Oren could not sleep he walked, and tonight his feet took him to his old school. The door was open. The classroom smelled of chalk, just as it had. He sat down at the small desk by the window. On the board someone had written a list of names for tomorrow, and the last was his own. Under it, in the same careful hand: OREN TO STAY BEHIND. HE KNOWS WHY.''',
    2203: '''The fair was gone and the field was mud and paper. Hale found the boy where he had expected to, sitting on an upturned bucket. He held out the watch. "Yours?" The boy would not look at it. "I took it," he said. "From you. Yesterday." "I know. I let you." Hale put it in the boy's hand and closed his fingers over it. "It was your father's before it was mine. Now tell me where he got it."''',
    2340: '''The clockmaker would not take the parcel. "I did not order it," he said. All around them the clocks ticked, none of them agreeing. "I only carry them," said the courier. "Then carry this instead." He held out a small silver watch. "Give me one hour of your life, and I will give you this, which is worth a year's pay." She looked at the watch. Its hands were running backward. "Which hour?" she asked. He smiled. "That is the part you do not get to choose."''',
    2477: '''Brandt bought his onions before the sun was up, as he always did. When he reached for his coins he found a folded paper that had not been there at home. Four words, in pencil: THEY FIXED YOUR LAST FIGHT. He read it twice. His hands, which had never shaken in the ring, shook now. Behind the fish stall a thin man was watching him, and when their eyes met, the thin man ran.''',
    2614: '''Lio went up on the roof to fix the tower clock, because the master was afraid of heights. From there he could see the whole square. A woman in a grey travelling cloak stopped below and looked straight up at him. She put her hand to her mouth. Then she called a name. It was not Lio. It was the name he had been born with, in another town, the name nobody here had ever heard.''',
    2751: '''Rain drove Ysolde under the bridge with her maps held inside her coat. A man was already there, dry, as if he had been waiting. "You draw very well," he said. "The mill. The ford. The gap in the east wall." Those were three maps she had shown to no one. He opened his own coat. Inside, pinned in a row, were copies of every one, each marked with the day she had made it.''',
    2888: '''The case under the seat was meant to hold money. Tobin opened it in the dark of the tunnel and found a violin instead. He should have closed it. He lifted it, to see if it was worth anything, and the train lurched, and the neck snapped in his hands. When the light came back, the old man across from him was awake and looking at the pieces. He did not shout. "My wife's," he said. "Well. Now you had better learn to mend it."''',
    3025: '''He had promised his brother forty years ago, on a warm deck far from here: whoever lived longest would bring the other home. So the old sailor walked out across the frozen lake pulling a small sledge, with a box on it no bigger than a loaf. Halfway, the ice groaned. He stopped. He looked at the far shore, and at the box. "You always were heavy," he said, and went on, slower, testing each step.''',
    3162: '''The stairs in Gran's house creaked on every step but one. Sorrel knew which, and stepped over it. Tonight Nessa stepped on it by mistake, and it made no sound at all, and the wall beside it swung inward like a door. Cold air came out. The sisters looked at each other. From somewhere below, a long way down, a voice that sounded like Gran's said, "Shut that. You are letting the light in."''',
    3299: '''The stranger came from the inn to have his watch mended. The clockmaker turned it over. "I can stop it losing time," she said. "The price is that it will never again show you a happy hour. Only the others." "That is a bad bargain." "It is. Most people take it." He laughed and reached for his watch. She did not let go. "You already did," she said. "Nine years ago. You have only now noticed."''',
    3436: '''The tailor delivered the suit himself, since he had only the one customer and wished to keep him. The house was dark but for one window. He was shown up, and waited, and waited. Below, he heard the front door being locked, then the back. He tried the window. It did not open. On the bed lay the suit he had made, and beside it another, in his own size, with a note: PLEASE PUT IT ON. WE DINE AT MIDNIGHT.''',
    3573: '''She did not know how she had come to be among so many books, with a ladle still in her hand. On the nearest shelf lay a letter sealed with the king's own ring. It was addressed to the queen. The cook could not read much, but she could read enough. She read: BY THE TIME THIS REACHES YOU THE SOUP WILL HAVE DONE ITS WORK. She looked at the ladle. Then she began, quickly, to look for a way out.''',
    3710: '''His mother's garden had gone to nettles and the house was shut. He sat on the well and took off his boots. On the wall, where no rain could reach, was a row of chalk marks. One for every day. Hundreds. The last was still white and fresh. Someone had been counting the days he was gone, someone had been here this morning, and it was not his mother, for her grave was by the gate.''',
    3847: '''The school had sent word that a boy would guide her through the flood. The new teacher waited on the church steps with water sliding past below. An hour. Two. At last a small boat came round the corner with no one rowing it. It bumped the steps and stayed. In the bottom lay a slate, and on the slate, in a child's round hand: SORRY I AM LATE MISS. GET IN. I WILL EXPLAIN WHEN YOU SEE ME.''',
    3984: '''The kitchen was all steam and elbows, and nobody looked at a boy carrying plates. That was why Emeric had chosen it. But each time he passed the big table, the same cook was a little nearer the door he meant to leave by. He changed doors. She changed tables. At last she put a hot pie into his hands, as if he worked there, and said into his ear, "The letter in your shirt. Not that way. They are waiting outside. Follow me."''',
    4121: '''Quill had never been ill in his life, and the hospital confused him. He counted doors as he would count ships: nine on the left, nine on the right. He turned to go back and counted ten. The tenth was painted the same red as his lighthouse. He could smell salt. Behind it something turned, slow and steady, throwing a line of light under the door, then dark, then light.''',
    4258: '''Her mother had sent her to the next village for medicine, with one silver coin tied in a cloth. Where the roads met there were no signs, only a dog. She untied the cloth to count the coin, as she had ten times already, and the dog took cloth and coin together and was gone into the wheat. She stood with her empty hand open. Then, far off in the wheat, the dog barked twice, the way a dog barks when it wants to be followed.''',
    4395: '''Last spring Dagny had cut the wrong rope and lost her father's nets, and he had not let her on a boat since. Now she was only a passenger, on the ferry, in fog as thick as wool. The engine coughed and died. The ferryman swore at a rope caught round the screw. He looked at the grey faces of his passengers and then at her hands. "You are Aldo's girl," he said. "They say you are quick with a knife. Are you quick enough to be careful?"''',
    4532: '''After the night shift Nessa crossed the border to go home, as she did each morning. Today the guard handed her an envelope. "From the doctor. For your sister." Nessa had no sister. She said thank you and opened it on the bridge. Inside was a pass in a stranger's name, with Nessa's own face on it, and one line: THEY ARE COMING FOR THE WARD AT NOON. TAKE THE CHILDREN OUT AS HER.''',
    4669: '''The crows would not come down, because of the bells. So the girl stood outside the church in her borrowed shoes while the wedding nobody wanted went on inside. An old crow landed on the gate. "Wait," it said. "For what?" "Wait." The bells stopped. The doors stayed shut. Then, from the road behind her, came the sound of one horse being ridden much too fast.''',
    4806: '''Oren had his market stall set out by four, long before dawn, since he had been awake anyway. Every morning for eleven years the old woman had come at five for one small loaf. Five came. Half past. He kept the loaf under a cloth and told two customers it was sold. When the sun was up he shut the stall, put the loaf in his coat, and went to find out which house had no smoke.''',
}

# Passages that Gauge's writing tasks are set on, in eight different kinds of story.
EXCERPTS: dict[str, str] = {
    "bag": '''The bag had begun to hum again. Rin kept walking, because stopping was how they caught you, and counted the lamps to the corner. Seven. At the fourth she felt the seal give under her thumb, soft as wet paper. Whatever was inside had been asleep since the harbour. It was not asleep now. Behind her the footsteps that had kept pace for three streets slowed, as if they too were listening.''',
    "debt": '''The debt collector came on the first frost, as he had come for her mother and her grandmother. Ysolde met him at the gate with the ledger already open. "Three generations," he said, wiping his spectacles. "Your family is nothing if not punctual." "And yours is nothing if not patient." She turned the book toward him. Where the sum should have been, someone had written a single name, in ink that was still wet. He stopped wiping.''',
    "note": '''Hale read the note a third time under the streetlamp. Eleven words, no signature, and a smell of lamp oil. The dead man had been a careful writer; this hand slanted the wrong way. Across the square the widow's curtain moved and was still. He folded the note along its old creases and found that it wanted to fold a different way, into quarters, the way a man folds something to hide it in a shoe.''',
    "bell": '''The house had been quiet for a month when the bell in the kitchen rang. It was the bell for the blue room, which had no bell-pull any longer; Sorrel had cut it down herself. She stood with flour on her hands and watched the little brass tongue shiver to a stop. Upstairs, the floor of the blue room took one slow step toward the door.''',
    "wreck": '''They sighted the island at dawn and the wreck an hour after. It lay on the reef with its back broken, and it flew the same flag they did. Dagny put down the glass. "That is the Marigold," she said. "She left port a week behind us." Nobody on deck spoke. A week behind, and already a year of weed on her hull.''',
    "slate": '''For two years Emeric had written to the lighthouse and for two years someone had written back, in a hand that grew less careful and more kind. Now he stood on the rock with his one bag while the boat that had brought him pulled away. The door at the foot of the tower opened. The keeper who came out was older than the letters, and held a child's slate, and wrote on it: SHE SAID YOU WOULD COME.''',
    "vault": '''The vault door stood open, which was the first thing wrong. The second was the lamp, still warm. Corin counted the boxes on the shelf: twelve, as the plan had promised. He lifted the nearest and it weighed nothing at all. Behind him Mara said, very softly, "Do not turn round yet. There is a chair in the corner, and someone has been sitting in it the whole time."''',
    "list": '''On the last day of term the list went up beside the stairs. Lio found his name at the bottom, where he had expected it, and beneath his name a line drawn in red, where he had not. Perrin read over his shoulder. "That is the scholarship line," she said. "You are above it." "I am last." "You are last of the ones who are going." She was not on the list at all.''',
}

# (which passage, which of Gauge's tasks, the answer). Each of the eight tasks appears three times.
ANSWERS: list[tuple[str, str, str]] = [
    ("bag", "continue", '''She did not look back. Looking back was the second way they caught you. At the fifth lamp she turned into the dyers' lane, where the gutters ran blue, and let the hum guide her hand to the seal. It had split along one edge. Something pressed there from inside, patient, like a tongue against a tooth.

"Not yet," she told it.

The footsteps turned into the lane after her. One pair. She had thought there were two.

At the end of the lane was the wall she had climbed a hundred times as a girl. She set her foot in the old gap and went up, the bag swinging against her back, and for a moment the humming stopped. In the quiet she heard the second pair of feet. They were not behind her. They were on the other side of the wall, standing quite still, exactly where she would come down.

The seal gave way.'''),
    ("bag", "title", "What the Bag Carried"),
    ("bag", "brief", '''Wants: to reach the corner and deliver the bag before the seal fails completely.
Fears: that what is waking inside the bag already knows who is carrying it.
Would refuse: to set the bag down and run, even to save herself.'''),
    ("debt", "opening", '''The second collector did not wait for the frost. He came over the wall at midnight, dropped into the herb bed, and was halfway to the kitchen door before Ysolde's lamp found him.

"You are early," she said.

"I am not from the same office." He brushed thyme from his knees. He was young, and had the ledger's twin under his arm, bound in the same grey cloth. "Mine says your family was paid in full, forty years ago. Mine says it is we who owe you."

She did not lower the lamp. Two books, one debt, and both of them in the same hand. Behind her, in the house, the first collector's chair scraped back from the fire.'''),
    ("debt", "score", '''hook: 8
momentum: 7
cliffhanger: 9
Advice: say sooner what the family owes; the wet name lands harder once we know the price of the debt.'''),
    ("debt", "dialogue", '''"Whose name is that?"
"You tell me. You have been reading it upside down for a minute."
"Madam, this ledger has been in my keeping since your grandmother signed it."
"Then somebody keeps it better than you do."
"If this is a trick, the office will add a year for it."
"Add ten. Whoever wrote that name will not be paying in years."'''),
    ("note", "ending", '''He folded the note along its old creases and found that it wanted to fold a different way, into quarters, small enough for a shoe. Hale looked down at the dead man's boots, which the constable had set neatly by the door. The left one was laced. The right one had been laced by someone else.'''),
    ("note", "outline", '''Title: The Right Boot
Hook: Hale unlaces the dead man's right boot and finds it is the wrong size.
Turn: the boots belong to the widow's brother, who was buried last winter.
Cliffhanger: the constable admits he did not set the boots by the door; they were there when he arrived.'''),
    ("note", "continue", '''He crossed the square without hurrying, which was the only way to cross it with that curtain watching. The constable had the body laid out in the back room of the chandler's, under a sheet that was too short. Hale lifted the foot of it.

The boots were good ones. The left was scuffed at the toe. The right was not, and its lace had been tied in a bow no man ties for himself, loops to the outside, the way you tie a child's.

"Who dressed him?"

"Nobody, sir. He was found dressed."

Hale worked the bow loose and drew the boot off. Folded into the heel was a square of paper, soft with wear. He did not open it. He was looking at the stocking beneath, darned at the heel in blue wool, and remembering where he had seen that same blue an hour ago, in a basket, on the widow's windowsill, beside a curtain that moved.'''),
    ("bell", "title", "The Blue Room Bell"),
    ("bell", "brief", '''Wants: to believe the house is empty and finish the bread.
Fears: that cutting down the bell-pull only made whatever is in the blue room find another way to call.
Would refuse: to leave the house, because it is the last thing her mother left her.'''),
    ("bell", "opening", '''Sorrel took the stairs with the bread knife in her hand, since it was what she had. The second bell rang as she reached the landing: the bell for the nursery this time, sharp and impatient. She had not cut that one down. She had not needed to. Nobody had slept in the nursery for thirty years.

The blue room door stood open a hand's width. It had been locked. The key was in her apron pocket, warm against her leg, and she pressed her palm over it as if it might get out.

"I am coming," she said, because it seemed safer to be polite.

In the nursery, something small began to count.'''),
    ("wreck", "score", '''hook: 9
momentum: 6
cliffhanger: 8
Advice: let one sailor name what everyone fears before the chapter ends; the silence on deck needs a single voice to break it.'''),
    ("wreck", "dialogue", '''"A week behind us. I signed her papers myself."
"Then your papers are wrong, girl, or that reef is."
"Weed like that takes a year. You taught me so."
"I taught you to count sails, not to frighten the watch."
"Count them, then. She has ours. Every patch."
"Bring the boat round. And nobody tells the men what name is on her bow."'''),
    ("wreck", "ending", '''Nobody on deck spoke. A week behind, and already a year of weed on her hull. Dagny raised the glass again, to be sure of the name, and found a figure standing at the Marigold's broken rail with a glass of its own, looking back at her.'''),
    ("slate", "outline", '''Title: What the Slate Said
Hook: the keeper writes a second line: SHE IS NOT HERE.
Turn: the letters were written by the keeper's daughter, who cannot leave the lamp room and has never seen Emeric's replies.
Cliffhanger: Emeric climbs the tower and finds every one of his letters, unopened, nailed to the wall.'''),
    ("slate", "continue", '''He read the slate twice. The keeper waited, chalk in his fingers, as a man waits who is used to being read slowly.

"She," said Emeric. "Not you."

The keeper shook his head. He rubbed the slate clean with his sleeve and wrote again, taking care over each letter: MY DAUGHTER. THE LETTERS WERE HERS.

"Then may I see her?"

For a long moment the chalk did not move. The sea came in and went out. The child's slate looked very small in the old man's hands. At last he wrote three words and turned it round.

SHE DOES NOT KNOW.

"Does not know I have come?"

The keeper pointed up, to the lamp room, where a shape moved behind the glass. Then he wiped the slate once more and wrote the rest, and held it out at arm's length, as if he would rather not be near it.

THAT YOU EVER ANSWERED.'''),
    ("slate", "title", "The Keeper's Slate"),
    ("vault", "brief", '''Wants: to take what the plan promised and be gone before the lamp cools.
Fears: that the job was sold to someone else before he ever heard of it.
Would refuse: to leave Mara behind in that room, whatever is sitting in the chair.'''),
    ("vault", "opening", '''They ran the second job in the rain, because Mara said rain made honest men stay home. Corin went up the drainpipe with the box under his coat, empty as the day he had stolen it, to put it back.

That was the plan now. Put twelve empty boxes back on a shelf, in a vault that had been open, under the eyes of whoever liked to sit in corners. Let them think nothing had been touched. Then watch who came to check.

He was on the sill when the lamp inside was lit. Not by Mara. Mara was below him, holding the rope.

"You are late," said the chair.'''),
    ("vault", "score", '''hook: 8
momentum: 8
cliffhanger: 9
Advice: give the empty box one more beat, a rattle or a smell, so we wonder what was taken before we learn who is watching.'''),
    ("list", "dialogue", '''"You knew. You read the list before I did."
"I read it at six this morning. I wanted to see your face."
"Then where is your name, Perrin?"
"Somewhere they do not pin up."
"That is not an answer."
"No. It is a head start. Ask me again when you are on the train."'''),
    ("list", "ending", '''"You are last of the ones who are going." She was not on the list at all. He looked for her name again, slower, and found instead, in the corner, in pencil, in her own small hand: one place was given up this morning.'''),
    ("list", "outline", '''Title: The Red Line
Hook: Lio learns the list was one name longer yesterday.
Turn: Perrin gave up her place so that the line would fall below him, and has told no one why.
Cliffhanger: the headmaster asks Lio to bring Perrin to his office, and to bring her bag as well.'''),
]

# Word study: (the word, a sentence it appears in, what it means there and three more uses).
WORDS: list[tuple[str, str, str]] = [
    ("lurched", "The train lurched, and the lamp swung on its hook.",
     '''Here "lurched" means the train moved with a sudden, clumsy jolt.
1. The cart lurched over the rut and she grabbed the rail.
2. He lurched to his feet, still half asleep.
3. Her heart lurched when she saw the open door.'''),
    ("ledger", "She met him at the gate with the ledger already open.",
     '''Here "ledger" means a book in which debts and payments are written down.
1. The clerk ran his finger down the ledger and frowned.
2. Every loaf he gave away went into no ledger at all.
3. She kept a ledger of favours, and his page was long.'''),
    ("threshold", "He stopped on the threshold and would not come in.",
     '''Here "threshold" means the strip of floor in a doorway, where outside becomes inside.
1. Snow had drifted over the threshold in the night.
2. She carried the child across the threshold and set him by the fire.
3. They stood on the threshold of a war and talked about the weather.'''),
    ("reluctant", "The guard gave the papers back with a reluctant nod.",
     '''Here "reluctant" means unwilling, doing something slowly because one would rather not.
1. The mule was reluctant to cross the bridge.
2. She gave a reluctant smile and passed him the bread.
3. He was a reluctant witness, and said as little as he could.'''),
    ("gutted", "By morning the fire had gutted the mill and left the walls standing.",
     '''Here "gutted" means emptied out or destroyed on the inside.
1. The thieves had gutted the shop and left the shutters neat.
2. She gutted the fish with three quick cuts.
3. He read the letter and stood there, gutted, with the paper in his hand.'''),
    ("errand", "Her mother had sent her on an errand to the next village.",
     '''Here "errand" means a short journey made to fetch or deliver something for someone.
1. The boy ran errands for the inn and knew every door in town.
2. It was a fool's errand, and she went anyway.
3. He came back from the errand with no bread and a black eye.'''),
    ("brittle", "The ice near the reeds was thin and brittle.",
     '''Here "brittle" means hard but easily broken.
1. The old paper was brittle and cracked along the fold.
2. Her laugh was brittle, and nobody joined in.
3. Frost had made the grass brittle under his boots.'''),
    ("beckoned", "From the far door a woman in grey beckoned to him.",
     '''Here "beckoned" means signalled with a hand or a look for someone to come nearer.
1. The innkeeper beckoned her into the back room.
2. He beckoned with one finger and the dog came.
3. The lit window beckoned across the dark field.'''),
    ("glimpse", "She caught a glimpse of a red hat out on the ice.",
     '''Here "glimpse" means a very short look at something, hardly long enough to be sure.
1. He had only a glimpse of the rider's face.
2. Through the gap in the wall she glimpsed the garden.
3. That one glimpse of the sea was enough to make him homesick.'''),
    ("falter", "His voice did not falter when he gave his brother's name.",
     '''Here "falter" means to lose strength or steadiness for a moment.
1. Her steps faltered at the top of the stairs.
2. The lamp faltered and went out.
3. He began bravely, but faltered when he saw who was listening.'''),
]


def rows() -> list[dict[str, Any]]:
    """Every starter row, with its prompt built from the school's own templates so the shapes always match."""
    from .distill import TASKS
    from .school import LESSON_TASK, WORD_TASK, lesson_seed

    templates = {kind: (template, words) for kind, template, words in TASKS}
    out: list[dict[str, Any]] = []
    for i, scene in sorted(SCENES.items()):
        out.append({"task": "lesson", "level": 1, "prompt": LESSON_TASK.format(**lesson_seed(i)), "response": scene})
    for key, kind, answer in ANSWERS:
        template, words = templates[kind]
        out.append({"task": kind, "prompt": template.format(excerpt=EXCERPTS[key], n=words), "response": answer})
    for word, sentence, answer in WORDS:
        out.append({"task": "word", "word": word, "prompt": WORD_TASK.format(word=word, sentence=sentence), "response": answer})
    for r in out:
        r["teacher"], r["licence"] = SOURCE, LICENCE
    return out


def sync(datasets_dir: Path, on: bool) -> int:
    """Put the starter rows where the trainer looks for them, or take them away. Returns how many are in place."""
    f = Path(datasets_dir) / FILE
    if not on:
        if f.exists():
            f.unlink()
        return 0
    data = rows()
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
    if not f.exists() or f.read_text(encoding="utf-8") != text:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    return len(data)


def summary() -> dict[str, Any]:
    data = rows()
    kinds: dict[str, int] = {}
    for r in data:
        kinds[r["task"]] = kinds.get(r["task"], 0) + 1
    return {"rows": len(data), "words": sum(len(r["response"].split()) for r in data), "by_task": kinds, "source": SOURCE}
