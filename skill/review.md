# Review: how to look, what to fix, how to judge

Everything that made toys good came from looking at them carefully. This file says what to look at, how to
turn what you see into a fix, how much to trust each number, and how to judge honestly.

## 1. How to look at a build

Open `compare.png` (concept row on top, then the base and the variants, all four views at the same height) and
`compare_face.png` (face front and 3/4, larger). Go through these questions in order, and stop at the first big
"no". That one is the next fix.

1. **Silhouette and proportions.**
   - Is the head the right size against the body? (Chibi: head ≥ 1.3x the body's width; concepts: head 0.46-0.51
     of the total height, head w/h 1.1-1.25.)
   - Is the body the right shape?
   - Is it the right pose (sitting, lying, standing)?
   - Is there a neck pinch that makes a snowman?
2. **Identity.** Cover the concept and say what the model is in one word. Is it the subject? If it reads as a pig,
   a ball, a chihuahua or a monkey, fix the identity cue before anything else.
3. **Ears.**
   - Construction: cup, spoon or drape? Never a paper plate.
   - Size against the concept.
   - Angle and spread, and where they sit (not on top as Mickey ears, not edge-on, not fanned like wings).
4. **Face**, on the close-up:
   - eye size against the concept (~0.19-0.20 of the face width on approved toys);
   - eye height (measure crown to chin);
   - eye spacing (about one eye diameter on approved toys);
   - the muzzle: size, height, whether it points down slightly;
   - the mouth shape;
   - cheeks 1.5-3 rounds under the eyes;
   - glints on the same side on both eyes;
   - eyes not overlapping the muzzle rim (that reads as a frown);
   - everything visible (not under a hat, a mask or the muzzle).
5. **Colour zones and markings.** Right shape and place? Stair-stepped curves (use an applique)? A patch hiding an
   eye? Do the colours match the measured hexes (very dark and very light yarn render off)?
6. **Accessory or prop.** Does it read as its object at thumbnail size? Is it visible in the 3/4 view, not hidden
   behind the head? Is it held at the body (a loose object on the floor reads as a random disc)?
7. **Physical plausibility, in EVERY view, including side and back:**
   - nothing floating or hovering ("rests on ..." flags);
   - nothing piercing (a bow through an ear, a prop into the face);
   - no planks or paper-thin parts edge-on;
   - the far eye does not stick out of the 3/4 outline;
   - no bare holes;
   - it stands on its body.

   Several of these produce **no flag**. Only looking finds them.
8. **Dignity at thumbnail size.**
   - Nostrils that read as a second pair of eyes.
   - A door under a smile that reads as a mouth.
   - A tail stub between the legs (move it up).
   - Stray-looking threads (thin lashes, whiskers, drips).
   - Dots that read as rivets.
   - Known-character look-alikes.

When unsure about a small thing, crop and enlarge it (the face, the ears, the collar). A debug trick: recolour a
part bright red for one build to see where it really is.

## 2. Checklists

**Before each build** (write it in one line):
- Which single difference am I fixing, and which ops?
- Do all ids I touch exist?
- Do the stitch counts change by multiples of K?
- Is every parent before its children?

**After each build:**
- Did it build? Check the exit code, not a stale `flags.txt`.
- Is every flag explained?
- Is `face_problems` empty?
- Snowman?
- Is the head big enough?
- Do the ears hang or stand as in the concept?
- Is the accessory visible at 3/4?
- Is it better or worse than the previous best? If worse, revert.

## 3. Symptom -> cause -> fix

| What you see | Cause | Fix |
| --- | --- | --- |
| Two stacked balls (snowman) | body closed to a narrow neck, shallow sink | body ends **open** at ~0.8-0.85 x head count; head `sink_dir: "axis"`, sink 12-22 mm; face up 1-2 rounds; or one-piece body; or a collar/scarf over the seam |
| Small head on a long body | body too long or wide; autofit-style body growth | shorten the body, enlarge the head (scale the face with it) |
| Box / bun / flat-topped head | `inc` straight to the max then a long even run | last two increases every 2nd round, ≤ ~M/10 even rounds, first decreases every 2nd round |
| Egg head | slow increases from a small start | `mr 6`, fast to M-12, last increases every 2nd round |
| Face looks low | head sinks into the body; re-rounded head | `shift_face` up 1-2 rounds; measure the eye height on the concept |
| Frowning face, forehead fold | eyes overlapping the muzzle's rim | eyes beside or above the muzzle corners |
| "Dead" eyes | plain black beads, too small, too high | bigger (eye_match), lower, glint (`highlight` or satin glint on the same side) |
| Eyes vanish | eye in a same-colour patch, under a mask or hat, safety-eye `sink` too deep | patches off the eyes; eyes ON a face motif (`on: <motif>`); eye `sink` ≤ 0.6 |
| Muzzle is a protruding button | small round `mr` muzzle | wider chain-oval muzzle (`ch 4-8`), lower, `dir` slightly down, sink 1-2, ≥ 1 even round |
| Paper-plate ears | `flatten` (or flatten + squash) on an ear | dish (round) / spoon (pointed, long) / drape (hanging): recipes §6 |
| Mickey ears | dish ears fully on top, too big | sink ~0.55 x radius, az 65-80, 2 st narrower |
| Ears fan out like wings | hanging ears without the right `front` / `toward` | preset `floppy` + `hang: true`, entry `front [0,-1,0.45]`, `toward [0.05,-1,0]` |
| Ear "soft piece laid ... 20-30 mm" | bad sew direction | change `dir` / `front` until ≤ ~10 mm |
| Small cup ears turn into slivers | `seam: "flat"` on a cup | no flat seam; dish |
| Square ear tips | fast increases to the widest | +3 per round to the widest |
| Inner ear reads as a stripe | narrow `inner` band, or plump ear squeezes it | spoon ≥ 0.55; with `stuff: light` set `width` ~1.3, `margin_st` 0.5 |
| Arms in a T-pose / stumps at the shoulder | wide body under the shoulders; `avoid` swings them out | pear body; arms `dir` forward-down + `avoid_toward [0,0,1]`; paw-shaped arms |
| Limbs look like stuck-on patches | strong flat seam | small `pinch_rounds`, or no flat seam |
| Toy hovers / rests on a tail, prop or fin | that piece is lowest | move it up, point it back, curl it, or drop the prop |
| Bow merges into a bib / reads as a moustache | loops too close; triangles on the head | loops at az ±25 on the BODY under the head, padded, + knot |
| Scarf is a belt at mid-body | ring at the wrong round, wrong width | ring at the body round just under the chin, `around`, + a hanging end |
| Prop reads as "a yellow circle" | an untextured ball on the floor | a recipe prop (recipes §12), held at the body |
| Strawberry is a red ball | primitive instead of a recipe | pointed shape + polygon zones + seeds + green leaves/calyx |
| Hat floats or sinks | tilted `dir` with `wear`, final decrease, sunk cap | worn cap, straight dir, ~1.2x head count, no decrease at the rim |
| Hair is a swim cap / curtain / flaps | sunk cup, fringe, flat panels | `wear` cap + `pad` (+ `opening`), or head colour zones on small heads |
| Stair-stepped curved patch | colour zones are whole stitches | applique / `mask` motif / padded chain-oval applique |
| Colour patch drifts or hides | zone vs face misaligned | check it on the front view; `patch` with `eye` anchor |
| A same-colour part is invisible | no contrast and no shadow | contrast trim, or recolour to check where it is |
| Animal reads as another animal | wrong identity cue | fix the cue (recipes §5, §7) before polishing |
| Every toy has the same face | copy-pasted face | sleepy arcs, a wink, a cat "w", an open smile, a safety nose |

## 4. How much to trust each number

| Signal | Use it for | Don't use it for |
| --- | --- | --- |
| Silhouette IoU (`fitview`, `summary.json`) | catching gross proportion or pose errors; comparing variants of one size change | faces, ears, accessories, final decisions. A bigger head lowers it 0.01-0.06. Photo shadows and floors skew it. The overlay (`fit_*.png`) says more than the number |
| `cute.txt` score and suggestions | a list of things to look at; predicted effects | a target. The concept and species beat it. Its eye target (0.10-0.16 of head width) is smaller than approved eyes |
| `flags.txt` | every line must be explained or fixed | proof that the toy is fine (piercing, a hidden mouth, planks and floating details often have no flag) |
| `visibility.json` / `face_ok` | buried muzzles, eyes under ears or hats | embroidered eyes and satin mouths (not checked) |
| `library_audit.txt` | spotting a part far outside designers' range (`!!`) | lying bodies, one-piece toys, deliberate concept features |
| `eye_match.py` | safety-eye size and spacing vs the concept | embroidered eyes, big-eared heads, heads worked from the nose |
| Fast builds | iteration | head shape and final picks (do a full build) |
| Self-scores | nothing (they run ~0.5 high) | anything |

When the engine changes, rebuild the baseline before comparing. Old recorded IoUs are not reproducible.

## 5. A defect analysis that works

When a toy "isn't cute" or "isn't like the concept", write an analysis before touching numbers. The owner's own
analysis of a lop bunny is the model to follow:

```
Character recognisable, but: soft transitions, rounded part seating and the facial character are lost.
1. Ears: stiff ribbons with parallel edges, a sharp fold and a dark hole at the root (24 even rounds of 20 st;
   bend beyond the last round; soft-piece push 32.6 mm on a 92 mm head).
2. Head: flatter and wider than the concept's tall dome with full low cheeks.
3. Muzzle: a narrow protruding button instead of a wide soft oval.
4. Feet: separate balls instead of ovals turned out.
5. Collar: a wide ruffle instead of a few big round scallops.
6. Surface/light: sharp repeated stitch relief, hard dark cavities.
Priority: ears -> head & muzzle -> pose & collar -> yarn & light. Eyes already match (0.163 of head width):
don't start there. IoU 0.867 is silhouette overlap, not 86.7 % likeness.
```

Name each defect, its cause in the spec, and the priority. Then fix in that order, one group per round.

## 6. The quality bar (all must hold before packaging)

- At a glance the 3/4 view reads as the same character as the concept (or the brief): species, palette, accessory,
  face. Say it in one word and check the word.
- Cute: a round head or one round body carrying the face, big eyes low and fairly close, a small low muzzle,
  blush, short soft limbs, a varied face.
- With a concept: front IoU ≥ 0.85 and side IoU ≥ 0.80, head/body within 10 % of the concept, eyes within ~1 round
  and 5° of it. Where a deliberate deviation lowers IoU (a bigger head, a crochetable 2:1 proportion), say so.
- Nothing floating, hidden, piercing or plank-like. The toy rests on its body or legs. Every flag is explained.
  `face_ok`.
- The pattern's arithmetic is OK. PATTERN_RU has no Latin ids or mojibake and no false ERRORS line, and it reads
  like a real pattern. The round trip matches.
- The accessory is kept. If the exact shape is impossible, simplify it to an applique, felt, satin or flat rows,
  and say so in the report.

Below the bar: package it only as `"quality": "draft"` with the reason, never silently.

## 7. Blind judges

Use them for doubtful changes and for a batch before the owner sees it. One judge flips on details between rounds.
Only an overall verdict from two judges is stable.

**Protocol (A/B):**
- Rebuild the old version on the current engine with the same build settings, so only the spec differs.
- Make the sheets with `scripts/ab.py make` (random slots, hidden key, same background).
- Send each sheet to **two fresh agents** with no context. The second reads the cases in reverse order.
  Optionally use two personas ("strict visual judge", "experienced amigurumi designer").
- `ab.py decode`: keep the new version only if **both** judges prefer it. A split or a tie keeps the old one.
- Change one thing per A/B candidate. If a sweep puts the new version in the same slot everywhere, run one reversed
  round.

**A/B judge prompt:**
```
You judge crocheted amigurumi toy designs. Open ONLY these images: <paths>. Do not open any other file.
Each image shows a concept (top, when present) and two versions, A and B (3/4, front, side, face close-up).
Judge only what you see; be strict; say "tie" when you cannot see a real difference.
For each case answer: closer_to_concept (A/B/tie), cuter (A/B/tie), looks_like_real_crochet (A/B/tie),
overall (A/B/tie), strength (clear/slight), and one concrete fix for the loser ("eyes 20 % bigger and 1 round
lower", not "improve the face"). Reply with JSON {"<case>": {...}, ...} only.
```

**Absolute judge** (a batch before the owner):
```
Open ONLY these images: <paths> (read them in this order). For each toy: say in one word what it is (reads_as),
then score 1-10: cute, sellable, crochet_look (does it look like real, well-made crochet: nothing patch-like,
floating, sunk or plank-like), quality ("a shop owner insists toys must look dignified, high-quality and like real,
well-made crochet"). List problems by view and 1-3 concrete top fixes. Then rank the toys and name the 2 weakest.
Reply with JSON only.
```

Judges score what is visible. An accessory hidden behind the head counts as missing.

## 8. The owner's review

- The owner's taste is the final bar, and it differs from judges'. Toys judged 7-8 were rejected (multi-part
  anatomy) and toys judged 5-6 were approved (varied, dignified). So show honest scores, let the owner pick, and
  vary silhouettes once a recipe has been approved.
- **Contact sheets:** each toy's 3/4 and front preview, its number, its name and its keyword, three per row. Send
  them as batches finish. List what is still waiting and the weak points honestly, and suggest a one-line answer.
- **On approval:** package exactly the shown build, publish at once, and freeze it. Keep later polish aside and
  never apply it to the approved toy.
- **On a complaint:** put the concept and the render side by side at the same scale and angle, find what differs,
  fix it, and say whether the fix was measured or judged by eye.
