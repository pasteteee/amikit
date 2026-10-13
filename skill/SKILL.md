---
name: amikit-toy-design
description: Design a high-quality amigurumi toy with the amikit engine, from a concept picture, a photo or a text idea to a spec that builds a cute, faithful, crochetable toy with an exact pattern. Use whenever an AI model (Claude, GPT, grok, ...) writes or edits an amikit spec, reworks a toy, or builds a collection.
---

# Designing toys with amikit

This is how the toys the owner approved were made: 131 of them, including 100 of 100 in the last batch. It also
covers why other attempts failed. Everything here comes from those sessions, not from theory. Read it once, then
keep [`recipes.md`](recipes.md) (constructions with numbers) and [`review.md`](review.md) (how to look, judge and
fix) open while you work. The spec format is [`../SPEC.md`](../SPEC.md). A model working inside the site's engine
service (tools `build_toy` / `edit_toy` / `finish`, no shell) also reads [`designer-card.md`](designer-card.md).

## 1. The target: what gets approved

The owner looks at the final pictures, not at metrics. Approved toys:
- **read as their subject at thumbnail size**, front and 3/4 (say what it is in one word: that word must be right);
- are **cute**: a big round head (or one round body carrying the face), big eyes set low and fairly close, a small
  low muzzle, blush, short soft limbs, a small smile;
- look like **real, well-made crochet**: nothing floating, piercing, buried, sticking out as a plank or paper-thin;
  the toy rests on its body;
- carry **one iconic detail or prop** that tells the story (a yarn ball, a witch hat, a heart, a teacup);
- use a **soft palette** (pastels and warm naturals) and a **varied face** (not the same face on every toy).

What was rejected: multi-part anatomy (seahorse, snail, caterpillar), long or spiky parts (tentacles, a hedgehog
coat), face masks that read as helmets, a snowman-stacked head, a plain blob without a prop, props that don't
read as their object (a plain ball at the feet read as "a yellow circle"), and look-alikes of known characters
(round head + round ears from behind = Mickey; a white kitten + pink bow by the ear = Hello Kitty).

**Faithful is not the same as approvable.** A batch of 20 toys tuned for silhouette IoU against busy AI concepts
got 0 approvals. The same weeks, toys built from approved constructions in the approved style got 131. Keep what
makes the character (species, palette, pose, accessory, face style). Take everything else from approved toys.

## 2. The habits that make the difference

These separate the toys that passed from the ones that didn't. A model that skips them fails, however good its
spec syntax.

1. **Start from an approved toy of the same archetype, never from scratch.** `recipes.md` §1 maps a concept to a
   construction and an example spec. Build that example unchanged first (r0) and look at it. 78 of the 100 approved
   SEO toys had a named base, and the rest borrowed a proven layout.
2. **Measure the concept before writing numbers.** Use `scripts/measure.py` (grid, proportions, width profile,
   colours) and write notes: head w/h, head height / total height, eye height from the crown, eye size and gap as
   a share of the face width, muzzle width, ear length, which parts are pieces and which are colour changes.
3. **Look at every build.** Look at the 4-view sheet next to the concept, and at the face close-up. Builders whose
   toys were recognised looked 15-35 times per toy. Builders who looked 4-5 times overrated their own toys.
4. **Change one thing per build and name it** ("r6: head taller", "r7: eyes 1 round lower + cheeks with them").
   When unsure, build 2-4 variants of that one thing and compare them in one image (`scripts/variants.py`).
   Rewriting the spec or changing ten things at once loses what was already right.
5. **Revert when a change is worse.** Keep every build (r1, r2, ...) and go back to the best one. A weaker model
   keeps iterating forward from a worse state.
6. **Numbers are guards, not goals.** IoU, the cute score, `eye_match` and the library audit were each overruled
   by eye at least once per toy:
   - a bigger head lowers IoU 0.01-0.06;
   - the cute score fell 84 -> 71 while a bunny got closer to its concept;
   - a 0.68 IoU variant was right and the 0.81 one wrong (photo shadows).

   Never optimise a face by silhouette.
7. **Identity first.** Get the 1-2 cues that make the species right before polishing anything: a hippo's huge
   muzzle reaching the eyes, a capybara's blocky snout, a fawn's sideways ears. With a wrong cue the judge names
   another animal (a pig, a ball, a chihuahua).
8. **Honest judging.** Self-scores run ~0.5 high and carried no information. Doubtful changes go to blind judges
   (`review.md`). Every report names the biggest issue left. A toy below the bar is a `draft`, said plainly.

## 3. Setup

```bash
pip install -e <amikit repo>          # or: pip install "amikit @ git+https://github.com/pasteteee/amikit.git@<sha>"
python -m amikit.build <spec> <out> [--ref picture] [--fast] [--ref-az DEG]   # the engine build
python skill/scripts/variants.py ...  # the loop helper (below); the scripts find amikit installed or via AMIKIT_LAB
```

On Windows use the venv's `python.exe` (a bare `python` can hit the Microsoft Store stub). Read and write specs as
UTF-8 (`encoding="utf-8"`, `ensure_ascii=False`). Never pass Cyrillic through a shell heredoc. Give builds a timeout
of at least 300 s, and run at most 5 in parallel. Example specs are in `toys/` of the lab's working copy (local
only, not in git) and in every site package (`spec.json`).

## 4. The loop for one toy

**Step 1. References.** `python skill/scripts/refs.py <picture> W/refs` cuts a 2x2 turnaround sheet into
`front/left/right/back.png`; a photo or drawing becomes `front.png` (`--crop` one toy out of a collage,
`--mirror` if it shows the other side). `left.png` is the toy's left side, the same as the engine's
`view_side.png`. A concept turned slightly off-front is real information: don't "correct" it. Render at the
picture's angle with `build --ref-az <deg>` when comparing with a 3/4 photo.

**Step 2. Read and measure.** Run `python skill/scripts/measure.py W/refs/front.png W/measure --at fur=0.5,0.6 ...`
and open `grid.png`. Write the notes (habit 2). Then write the **identity list**: the 1-2 cues that make this
subject, and the one iconic detail. When the text brief and the picture disagree (tail side, plate count,
freckles), the picture wins.

**Step 3. Choose the base.** Pick the archetype from `recipes.md` §1, copy the example spec to `W/<id>.json` and
set `name`, `title`, `title_ru`, `subject`. Name colours by what they are and add `colour_names_ru` for every one.
Give every part `name_ru` / `name_en`. When the subject was rejected before (`_rejected/` folders, run notes),
read why and fix exactly that. When you reuse an approved base, change the face or the accessory so it is not a
catalog duplicate.

**Step 4. r0.** `python skill/scripts/variants.py W/<id>.json W/r0 --refs W/refs` builds the base and writes
`compare.png` (concept row + model, all four views, equal height) and `compare_face.png`. Look at both. Write down
the 3-5 biggest differences, ranked.

**Step 5. Iterate.** Fix the biggest difference first, in this order:
1. silhouette and proportions (head size vs body, body shape, pose; no snowman);
2. identity cues (does it read as the subject?);
3. ears (construction, size, angle, placement);
4. face (eye size, height and spacing; muzzle; mouth; cheeks; the face must be visible);
5. colour zones and markings;
6. the accessory / prop;
7. small details.

Each round, write a variants file with 1-4 named variants of ONE thing (edit ops, `../edit.py` syntax) and run
`variants.py <current best>.json W/rN W/rN.json --refs W/refs`. Look at `compare.png` and `compare_face.png`,
pick by eye, then read `summary.json` (flags, face problems, IoU, cute) as a guard. The winner's
`W/rN/<name>.json` becomes the current best. Run the checklists in `review.md` §2 on every build.
- Move the face as a group: `{"op": "shift_face", "on": "head", "rounds": -1}` moves eyes, glints, nose, mouth,
  cheeks, muzzle and face appliques together. After re-rounding a head, the face has moved: shift it back.
- Explain every flag or fix it. "soft piece laid on ... N mm" on an ear means a bad sew direction: change `dir`
  / `front` until N ≤ ~10.
- Decide head shape and final picks on **full builds** (`--full`): fast renders make heads look wider and flatter.

**Step 6. Cute pass** (once the silhouette is right). `python -m amikit.cute W/<id>.json --ref W/refs/front.png
--out W/cute --closeups` gives a score, out-of-range features and ranked suggestions with predicted effects. It is
advice. Try a suggestion as a variant (half a move is allowed). Reject anything that removes a species or brief
trait: big ears (fennec, bat, deer), a big low muzzle (cow, hippo, T-rex), big cat eyes, short T-rex arms. On
one-piece toys (food, fish, objects) its numbers mean little: judge the face by eye.

**Step 7. Eyes.** Owners look at the eyes first. `python skill/scripts/eye_match.py W/refs W/rN/<best> <spec>`
measures safety-eye size and spacing on the concept and the model and snaps to shop sizes. Approved eyes are about
0.19-0.20 of the face width and about one eye diameter apart. Generic cute targets (0.10-0.16) are too small. The
tool cannot see embroidered eyes and is wrong on big-eared heads, so measure on `grid.png` then.

**Step 8. Finish.**
- Do a full build of the winner.
- Check the bar (`review.md` §6).
- Read `PATTERN_RU.md` once as a crocheter would. There must be no Latin ids, no mojibake and no false
  «ОШИБКИ/ERRORS» line, the ear lines must say how to press (тарелочка / ложечка), and what it says to stuff, curl
  or sew must make sense.
- After construction changes, run `python -m amikit.roundtrip <spec>`: the pattern must rebuild the same toy.

**Step 9. Report.** `{id, builds, likeness_1to10, cute_1to10, quality_1to10, reads_as, biggest_issue_left,
differences_from_concept, engine_limits, bugs, flags_explained, notes}`. Be honest: the bar is checked again by
someone else.

**Budget.** 6-20 builds per toy is normal. Toys reworked against a detailed concept took 11-36. A toy on an
approved base with a small change can stop at 3. Stop when the bar holds, not at a count.

## 5. When it won't converge

- **Five builds on one part that go nowhere** means the construction is wrong, not the numbers. Look up the part in
  `recipes.md` and change the construction (dish instead of flatten, a worn cap instead of a sunk cup, a
  one-piece body instead of a head on a body).
- **Still nothing** means it is probably an engine limit (`recipes.md` §16). Simplify honestly (a flat brim + a
  star instead of a curled brim), say so in the report, and don't fake it.
- **A defect that repeats across toys** is an engine fix, done as a separate task with the engine tests and the
  round trip, never inside a toy job.
- **When the owner complains**, put the concept next to the render at the same scale and angle first, then fix what
  differs. A fix made from close-ups alone failed ("все ещё плохо").

## 6. Rules that never bend

1. **The model is exactly what the written pattern makes.** Use only rounds, stuffing, flatten / squash / spoon /
   dish / bend / drape, appliques, motifs, worn pieces, collars, embroidery, felt, fringe, cords and flat rows. No
   render-only tricks. Glints are embroidered, or the materials list buys eyes with a glint (`highlight: true`).
2. **Original designs only.** No branded characters or look-alikes, and no copying a designer's pattern (the
   library is statistics only). Marketplace photos are learn-only references.
3. **Don't edit the engine inside a toy job.** Record gaps and bugs in the report.
4. **Approved means frozen.** Package exactly the version that was shown and approved, and never rebuild or
   "polish" it afterwards.
5. **Spec hygiene:**
   - `inc K to M` needs `M - current` to be a multiple of K, and `ch N` starts round 1 at 2N stitches (`ch 7` = 14,
     so `inc 6 to 48` fails; `ch 6` = 12 works). Check odd steps with
     `python -c "from amikit.rounds import counts; print(counts(['ch 6', 'inc 6 to 48', 'even 3']))"`.
   - Parents come before children in `assembly`, and face items name a placed instance (`eye_L`), not a part id.
   - Instance names never end in a digit. Letter suffixes skip `l` and `r`.

## 7. Collections and batches

- **Plan first.** Each toy gets a short entry: id, names, keyword, 1-3 **base specs** from approved collections,
  and a 1-2 sentence construction brief in a fixed order (pose/body -> colour zones -> ear kind -> eyes -> ONE
  iconic accessory with a construction hint, naming the recipe to reuse).
- **Group work by family.** One worker gets 5 toys of one family (cats, dogs, bears, sea, food...), reuses one
  proven body and deliberately varies the faces. Workers touch only their own ids and scratch folders. Run at most
  ~5 workers at once.
- **Spread the traps.** When one worker reports an engine trap, the lead passes it to all the others.
- **Owner review by contact sheets.** Show the 3/4 + front of each toy, its number and name, and honest weak points.
  Ask for one-line answers («одобряю всё, кроме #15»). Publish approved toys at once and freeze them.
- **Lessons go into `recipes.md`.** After a batch, add what worked or failed to the matching section (with
  numbers), not as a dated log.

## Files

| File | What |
| --- | --- |
| `recipes.md` | archetypes -> base specs; heads, bodies, faces, ears, muzzles, limbs, tails, hats, collars, props, objects, colours; copyable entries; engine gaps |
| `review.md` | what to look at per build, symptom -> fix table, defect analysis template, blind judges, owner review, the quality bar |
| `designer-card.md` | the compact version for a model with only build / edit / finish tools (the site's engine service) |
| `scripts/refs.py` | concept sheet / photo -> `front/left/right/back.png` |
| `scripts/measure.py` | grid, proportions, width profile, palette and point colours of a picture |
| `scripts/variants.py` | base + named variants (edit ops, `shift_face`) -> parallel builds -> `compare.png`, `compare_face.png`, `summary.json` |
| `scripts/eye_match.py` | safety-eye size and spacing from the concept, snapped to shop sizes |
| `scripts/ab.py` | blind A/B sheets with a hidden key, and decoding the judges' votes |
