# Recipes: constructions that worked, with numbers

Every recipe here was built, looked at next to its concept and kept because it won. It won through owner approval,
a blind judge, or a clear before/after. Numbers are for DK yarn and the default gauge (stitch 4.0 mm, round 3.6 mm)
unless a recipe says otherwise. "Head 60" means the head's widest round has 60 stitches. Az, rounds, `dir` and
`front` follow `SPEC.md` "Frame and units": az 0 = front, +90 = the toy's left, and a `mirror: true` entry describes
the left copy. Example paths are relative to `toys/` (the lab's working copy; site packages carry `spec.json` too).

**Always start from a spec that already works.**
- Owner-approved collections: `seo100/` (100 toys, 2026-10-12), `showcase10`, `catsdogs`, `diverse8`, `diverse8b`
  (31 toys, 2026-10-09).
- Passed blind judges, but not owner-approved: `pets10`, `simple10`, `fidelity5`.
- **`new2610` and `new50` were tuned for fidelity to AI concepts and none was approved.** Use them for single part
  constructions, not as a style reference.

---

## 1. Pick the archetype first

The base construction decides more than any later tuning.

| The concept looks like | Construction | Copy |
| --- | --- | --- |
| **Small cute animal, bird or bug with the face on one round body** (the approved "bean": cats, dogs, chick, duck, penguin, bee, ladybug, dino) | ONE body piece carries the face (no neck, no snowman); tiny identical limbs; one prop held at the belly (§1a) | `catsdogs/c1_ginger_yarn`, `showcase10/07_bee_buzz`, `showcase10/09_penguin_beanie` |
| Sitting chibi animal with a big separate head (bear, bunny, puppy) | head + open body; head sunk with `sink_dir: "axis"`; body ends narrower than the head | bears `seo100/classic_teddy_bear`, `diverse8/g_panda_bamboo`; dogs `pets10/06_snowball_samoyed_puppy`, `seo100/corgi_puppy`; cats `seo100/tabby_cat`; bunnies `seo100/chubby_bunny` |
| Lying "loaf" animal (cat loaf, dachshund, fawn, seal) | closed body lying along Z (`axis [0,0,1]`), flatten 0.35-0.5, head pressed on the front-top | `seo100/cat_loaf`, `pets10/05_mocha_dachshund_loaf`, `seo100/baby_deer`, `simple10/07_seal_pup` |
| One soft mass: capybara, mochi, ghost, hamster | ONE piece; the face on the upper front | `seo100/capybara_orange`, `seo100/little_ghost`, `seo100/chubby_bunny` |
| Round creature: bat, spider, monster, pufferfish | one 60-st ball, face on the equator | `seo100/halloween_bat`, `seo100/fluffy_monster`, `seo100/pufferfish` |
| Object or food with a face (bun, egg, heart, star, moon, pickle, fruit) | one closed piece shaped by its counts + one iconic prop or detail | `simple10/*`, `seo100/puffy_heart`, `seo100/sleepy_moon`, `seo100/fried_egg` |
| Character in an object (teacup, honey pot, eggshell, basket) | the container is a separate open piece ~18 st wider than the head; the body is raised so the chest shows | `diverse8/d_mouse_teacup`, `diverse8b/b_bear_honeypot`, `seo100/kitten_yarn_basket`, `seo100/chick_in_egg` |
| Fish, whale, seal, turtle (lying, swimming) | body along +Z from the tail, face on the front pole | `seo100/clownfish`, `seo100/blue_whale`, `seo100/sea_turtle` |
| Bird | owl/chick = one round body; goose/duck = lying teardrop + neck piece | `seo100/owl`, `seo100/silly_goose`, `seo100/rubber_duck` |
| Dinosaur, dragon | bean or sitting chibi; stegosaurus = body root with the head on its upper front | `showcase10/03_dino_munch`, `seo100/t_rex`, `seo100/stegosaurus`, `diverse8b/d_dragon_egg` |
| Plant in a pot, flower, tree | pot as the root, stem and leaves as closed pieces | `seo100/sunflower_pot`, `seo100/tulip`, `seo100/christmas_tree` |
| Doll, gnome, Santa, angel, robot | body + head; hair as colour zones or a worn wig; hat as a worn cone | `seo100/doll_pigtails`, `diverse8/a_gnome_moss`, `seo100/santa_claus`, `seo100/little_robot` |

What the 131 approved toys share:
- median 8 parts and 7 colours, DK yarn and a 2.5 mm hook, height 5.5-25 cm (median 12);
- 51 of the 100 SEO toys have a separate head (40 of them `sink_dir: "axis"`, heads 54-66 st, sink median 12 mm),
  and 49 are one piece;
- 79 use safety eyes, **all with a glint** (`highlight: true`, 10-20 mm, median 13), and the rest have deliberately
  varied embroidered faces.

After one recipe was approved, the owner asked for **maximally different** toys next. Vary the silhouettes across a
collection.

### 1a. The approved bean (15 of 15 shown were approved)

```json
{"id": "body", "rounds": ["mr 6", "inc 6 to 42", "inc 6 to 54 every 2", "even 12", "dec 6 to 42 every 2", "dec 6 to 6"],
 "stuff": "firm", "colours": [{"ellipse": {"center": [9, 0], "radii": [8, 55]}, "colour": "cream"}]}
```
- The body is 31 rounds and 54 st, about 10 cm. The 60-st variant (`inc 6 to 48, inc 6 to 60 every 2, even 9`) is
  used for the ladybug, frog and octopus.
- **Eyes:** safety 13-15 mm with `highlight: true`, at round 20-22 of 31, az ±22-26.
- **Cheeks:** `mr 6, 12` open applique in a blush colour, `lift` 0.8-1.0, 1.5-3 rounds under the eyes, az ±40-46
  (about eye az + 18).
- **Nose/beak:** at az 0, 1.5-3 rounds under the eyes. Mouth: a small "u" of 5 points, ±5..7°, dipping ~0.6 round,
  about 2 rounds under the eyes (e.g. `[[19.2,-7],[18.81,-3.5],[18.6,0],[18.81,3.5],[19.2,7]]`).
- **Arms:** `mr 6, 9, even 2, 8`, light, round 12.5-13.5, az 50-55, `dir [-0.4,-0.3,1]`, `seam: "flat"` (the paws
  meet in front).
- **Feet:** `mr 6, 12, even 1, 10`, light, flatten 0.5, round 3, az 30, `dir [0.3,-0.35,1]`.
- **Prop:** a closed firm ball `mr 6, 12, 18, 24, even 3, 18, 12, 6` at round 9.5-10, az 0, `attach: "end"`, sink 5
  (+ a chain-cord thread with `curl_deg` 160).
- **Cat ears:** `{"preset": "ear", "kind": "pointed", "max": 19, "inner": "<pink>"}` at round 27.5, az 42,
  `dir [0.5,1,0.12]`, sink 1.
- **Dog ears:** `{"preset": "ear", "kind": "floppy", "max": 18, "length": 3, "hang": true}` at round 27, az 68,
  `front [0,-1,0.45]`, `toward [0.05,-1,0]`.
- **The species cue** is one crochet feature: a bill, a crown, spikes, a comb, tusks.

---

## 2. Heads

- **Round head:** `mr 6, inc 6 to M-12, inc 6 to M every 2, even ~M/10, dec 6 to M-12 every 2, dec 6 to 6`.
  - Approved bunny/panda: `mr 6, inc 6 to 42, inc 6 to 54 every 2, even 5, dec 6 to 42 every 2, dec 6 to 6`
    (24 rounds).
  - Teddy: `inc 6 to 48, inc 6 to 60 every 2, even 7, dec 6 to 48 every 2, dec 6 to 6`.
  - `inc 6 to 54` + `even 9` gave a box or a bun.
  - Note: `every 2` adds no trailing even round.
- **Cute concepts:** heads wider than tall (w/h 1.1-1.25), head 0.46-0.51 of the total height, the face in the
  lower half. An egg-shaped head (slow increases off a small start) loses the cuteness.
- **Tall dome with full low cheeks** (when the concept's head is taller): slow last increases + 7-8 even rounds,
  e.g. `mr 6, inc 6 to 42, 48, even 1, 54, even 1, 60, even 8, 54, even 1, 48, dec 6 to 6`. Decreasing the bottom
  every 2nd round down to 54 gives fuller low cheeks.
- **Wide oval head:** start from a chain (`ch 5`-`ch 9`) + `inc ... every 2`, and raise the eyes 1-2 rounds.
- **Size:** 54-66 for a sitting chibi, 72 on a long body (dachshund `even 32` at 54 st). The head is ≥ 1.3x the
  body's width.
- `preset: cute_head` works on ball heads (axolotl 60, dragon 66). It hurts frogs with eye bumps, tall domes and
  big heads (a flat crown on 66+, "a frisbee" as an oval at 78). Prefer the explicit rounds above, or
  `"crown": "round"`.
- **When the head changes, the face moves.** Re-rounding shifts every face item. Move the face back with the
  `shift_face` op (scripts/variants.py), and scale the muzzle, face patches and eyes with the head.

## 3. Head on body: never "two stacked balls"

- Head entry: `"sink_dir": "axis"`, sink 12-22 mm. The body ends **open and wide**: ~0.8-0.85 x the head count (42-48
  under 60; 36 under 60 still stacks). Then move the face 1-2 rounds up so the mouth stays above the body's rim.

  | Approved example | Body rounds | Head sink |
  |---|---|---|
  | calico | `mr 6, inc 6 to 48, blo 48, even 4, 45` | 19 mm |
  | tiger | `inc 6 to 48, blo 48, even 7` (no neck decrease) | 21 mm |
  | teddy | `mr 6, inc 6 to 48, even 5, 45, even 1, 42, even 1` under a 60-st head | 12 mm |
  | mushroom frog | 60 -> 54, open | 22 mm |
- Sitting kitten: 54-st head on a 42-st MR body with a BLO base, open at 42, sink 17 mm with `sink_dir: "axis"`,
  face +2 rounds.
- Dog base (samoyed; it carried the corgi, shiba and golden puppies): closed body + 66-st head sunk 18 mm with
  `sink_dir: "axis"`, big flat feet, stubby front legs. Swap only the ears, the head's colour zones and the tail.
- A one-teardrop concept needs **one piece**. For two masses in one piece, put a dec-then-inc neck in the rounds
  (...54, 42, 54, 60...).
- A head behind the body in profile: tilt the body root `axis [0, 1, 0.18-0.45]`. **Keep the head's `dir` upright
  or slightly back.** Tilting the head forward turns the face to the floor (cat loaf, snake).
- A collar, a scarf, a snood or a ring of fringe hides the seam. Put it on the body round just under the chin.

## 4. Bodies, bases, standing

- A seated animal's body reads best as a round ball from `mr`. A chain-oval body with `oval_axis: "front"` gives
  sitting dogs and bunnies their deeper side view, but it looks narrow from the front.
- **Pear body** under a big head (it decreases toward the neck): arms then hang instead of being pushed into a
  T-pose. Example: `inc 6 to 36, even 4, dec 6 to 30, even 2, dec 6 to 24, even 2, dec 6 to 18`.
- **Flat sitting base:** a `blo` round right after the base increases, + feet at round 3-5, keeps chibi toys
  resting on the body. Haunches: padded applique `mr 6, inc 6 to 24`, `pad 6`.
- **Lying loaf:** body root `axis [0,0,1]`, flatten ~0.35-0.5, head pressed on the front-top with a near-vertical
  `dir`. Cat loaf: 54 st, 10 even, flatten 0.35, head at body round ~22, sink ~25 mm. A horizontal belly line on a
  body worked from the nose: per round az = acos(-h/R) from a flat cut plane (a plain az band narrows to a wedge).
- **Big-headed lying animal:** the head is the root, the body is sewn to its back (sink ~24), front legs under the
  head.
- **One-piece barrel** (capybara): `ch 8` oval -> 70, blo base, slow decreases, closed top, face on the upper front.
- **Mochi bunny:** `mr 6 -> 66` flat base, blo round, 7 even, `dec 6 every 2 to 48`, close; ball feet.
- **Rubber duck in one piece:** start at the head top (mr, `axis [0,-1,-0.12]`), 6-st decrease neck, body
  increases, `blo` + `dec 6 to 6` flat base.
- A standing crocheted toy should be at most ~2:1 tall:wide. A 3:1 reference was built 2:1 so it stands (say so).
- **Gauge:** DK gives ~12 cm toys. Concepts at 16-20 cm need a thicker yarn (worsted/aran gauge in `yarn`).

## 5. Faces (owners look at the eyes first)

**Eyes**
- **Approved eyes are big and fairly close:** about 0.19-0.20 of the face width, about one eye diameter apart
  (cats up to 0.20). Concepts measured 0.15-0.20. The generic cute target (0.10-0.16) is too small. Toys with eyes
  at 0.13-0.17 and 2-3 diameters apart were not approved.
- **Sizes:** 12-15 mm on 54-60 st heads, 13-20 mm on 60-72 st heads. 24 mm on a big chinchilla head was right;
  30 mm was bug-eyed.
- **Height:** ~0.55-0.62 of the head's rounds from the crown (bunny 13.5/24, penguin 17.5/28). On the bean, round
  20-22 of 31. Our builds used to come out 1-3 rounds low. Measure crown-to-chin on the concept and move the
  **whole face group**.
- **Spacing:** az ±25-31 on round heads, ±22-26 on the bean. Too wide makes the far eye stick out of the 3/4
  outline.
- **Glints:** `highlight: true` on safety eyes (the materials list says eyes with a glint). On embroidered eyes, a
  satin glint ~0.2-0.3 x the eye at `offset_mm [-0.2w, +0.2h]`, `lift_mm` 0.9, on the **same world side on both
  eyes** (one window), not mirrored. Embroidered eye: black satin oval 7.5-8 x 9-10.5 mm, thickness 1.1.
- Safety-eye `sink` is a fraction of the eye's radius (default 0.45). 0.6 works; 1.5 buries the eye.
- Eyes must not overlap the muzzle's rim (it reads as a frown) or sit in a same-colour patch (they vanish).
- Eyes on a dark body: felt whites ~1.4x the eye, behind highlighted safety eyes. Owl: eye discs = a 4-round
  circle motif with `anchor: "eye"` + amber-iris safety eyes.

**Mouths and expressions**
- The cute default is a small "u" smile, blush 1.5-3 rounds below the eyes, and eyes just below the head's middle.
  A shallow U with more points beat a near-straight mouth in both blind votes.
- **Vary the faces across a collection:**
  - sleepy closed arcs (a 5-point stitch arc, 2 mm thick, at the eye round, az 13-35 on a 66-st head);
  - a wink;
  - a cat "w" mouth;
  - an open smile with a tongue;
  - lashes (only when thick enough not to look like stray threads);
  - a safety nose.
- **Open smile on a chain-oval muzzle:** satin `triangle` ~12x8 mm at round ~2.1, az 180, tongue oval just below.
  **On a plain face:** a wide shallow arc (az ±17, ~0.7-round dip) + a satin oval tongue.
  - A narrow V reads as a pendant.
  - A satin triangle mouth reads as a nose or a beak.
  - A small satin open mouth reads as an "o".
- **Satin details:** set `strands: 9` on noses (the default gives stripes), and use round ~1.0 on a 5-round muzzle
  (round 2 lands on the rim). Short dashes for nostrils read as a second pair of eyes, so use round satin nostrils.
- **T-rex grin:** dark red satin oval (~30x9.5 mm) on the snout front + 4 white satin triangles over its top edge.
- On a forward-pointing muzzle, az 0 is the top, so mouth stitches go at az 180.

**Masks and face patches**
- A mask that follows the eyes is a `mask` motif (chain ≈ eye spacing / stitch width + 2, rounds 3, `tall: "tr"`)
  with `anchor: "eyes"`, `drape: true`, `offset_mm [0, -7]`, wide enough to reach the head's sides. Lay the cheeks
  above it (`lift` 2.2). Siamese: chain 6-7, offset ~4.5 mm down.
  - It suits a band around the eyes only. For a cream lower face use a wide oval + side ovals, or the mask reads
    as goggles.
- Front-only face markings (husky, fennec) are `patch` zones: one per eye (`eye: "L"/"R"`), one lower-face ellipse,
  and a chin patch. A whole-round band around the head reads far worse. Check applique faces against zone faces
  with a blind A/B: on the fennec the judges preferred the zone mask.
- Monkey face: a heart motif (5 rounds) with the safety eyes ON it (`on: <motif>`).
- A patch or applique sewn over the nose or mouth hides them with no flag. Put the embroidery on the patch.

## 6. Ears (the second thing owners look at, and the feature most A/B calls turned on)

Never a plain `flatten` plate (paper-thin and tilted), never flatten + squash together (paper blades with a slit),
and never `seam: "flat"` on small cup ears (they become fins or slivers).

| Kind | Construction | Numbers that worked |
| --- | --- | --- |
| Round (bear, mouse, panda, quokka) | `dish` 0.5-0.7 + entry `attach: "edge"`, `face` ~[0.15, 0.25, 1]; close the back (`dec 6 to 6`, `end: "closed"`) so it is coat colour | teddy `mr 6, 12, 18, even 1, dec 6 to 6`, dish 0.5, `inner` zone, round 6, az 66, sink 6; small `mr 5, 10, even 1-2, 8`, dish 0.7, az ~80, sink ~8 |
| Tiny (capybara) | small dish | `mr 4, 8, even 2, 7` at az 80 |
| Pointed (cat, fox, red panda) | `spoon` 0.35 cone from mr 4-6, +3 per round to the widest (rounded tip) | bean cat: preset `pointed` max 19 at round 27.5, az 42; tall upright beats short and wide; set the `seam_line` along the head if the default pinches a funnel |
| Folded kitten ear | spoon 0.4 cone `mr 6 -> 26` + inner zone, `bend` 80 at round 4 with drape | `dir [0.8,1,0.3]`, `front [-0.2,-0.5,1]`, az 66 (dish ears made it a mouse) |
| Bat ears (French bulldog) | pointed preset `mr 6`, +3-4/round to 30, `even 5`, `stuff: light`, pink `inner` | dir [0.55, 1, 0.08] at az 60; shorter or wider reads as a fox or a chihuahua |
| Halloween bat | `mr 4, inc 2 to 10, inc 3 to 25 every 2, even 4`, spoon 0.45, light | reject `cute`'s "shrink ears" |
| Huge (fennec) | cone `mr 4, inc 3 to 13, inc 3 to 37 every 2`, spoon 0.55 + `inner` | dir ~[1.05, 1, -0.3]; reject "shrink ears" |
| Long (bunny, upright) | spoon 0.3, ~3 more even rounds than you think (the fold eats length); ears in a slight V | |
| Floppy / lop (dog, lop bunny) | preset `floppy` + `hang: true` (or kind `lop`): `bend.drape`, it hangs by its own weight | entry `front [0,-1,0.45]`, `toward [0.05,-1,0]`; ear ~25 % bigger than a plank ear; oval-leaf rounds (rounded tip, widest a third down, slow taper to ~55 %); `twist` 1 lies flat (dog), 0 shows a coloured inside (lop bunny) |
| Fawn | big light spoon ears **sideways** (`dir [1,0.25,-0.1]`) | upright reads as a chihuahua |

- **Plump ears (`"stuff": "light"`)** give thick, soft edges. They won on drop puppy ears, the chinchilla and lop
  cups, but **lost both blind judges** on a folded kitten (it lost the fold) and a fennec (narrower ears). A/B them.
  With `stuff: light` the `inner` zone shrinks to a stripe, so set its `width` ~1.3 and `margin_st` 0.5.
- **Inner ear colour:** the `inner` zone fills the spoon's hollow. Use spoon ≥ 0.55 so it reads as a hollow (the
  owner said "they must sink in"). Don't put an oval applique on a pointed ear.
- **Placement:**
  - A dish ear fully on top reads as Mickey ears. Sink ~0.55 x the dish radius, az 65-80, ~2 st narrower than
    you'd think.
  - Big flat ears (elephant, panda, mouse) read as "thin discs" from the side. Give them a thicker rim or light
    stuffing.
  - Under a hood or appliques, put small dish ears just outside the hood edge.
  - Pointed ears under a hat brim go low and to the side (round ~23.5 of 33, az 74, dir [1, 0.75, 0.05]).
- **Small concept ears:** use fewer rounds before the widest, not a smaller `sink`.

## 7. Muzzles, snouts, beaks

- A muzzle is an open, lightly stuffed piece sewn by its opening, usually a chain oval, set low and pointing
  slightly down (`dir [0, -0.1..-0.15, 1]`, sink 1-2), with at least one even round (an open muzzle with no straight
  round shows only its rim).
  - Teddy: `ch 4, 14, 20, even 2` at head round 17.5 + satin oval nose `strands` 9.
  - Wide puppy muzzle: `ch 8, 22, 28, 32, even 2`.
  - Bunny: `ch 5, inc 6 to 22, 24, even 2`.
- A snout raised to eye level reads as a disc from the front: tilt it down.
- **Identity muzzles:**
  - Hippo: huge, wide, reaching up to the eyes, with oval nostrils (a small muzzle reads as a pig).
  - Capybara: blocky tan snout.
  - Pug: padded dark chain-oval mask (pad ~8) + a small cream open muzzle.
  - Kawaii unicorn and pangolin: no snout at all (an oval snout with nostrils reads as a pig).
- Keep the species' muzzle even when `cute` says smaller or higher (cows, T-rex, tiger, fox).
- Puckered fish lips: a small dish (`mr 5, 10, 10, 8`, dish 0.7) + dark satin in the hollow under the front pole.

## 8. Limbs, feet, paws

- Designers' limbs are mostly `paw`-shaped: widest at the start (the hand or foot), narrowing to the seam. A plain
  equal-round cylinder is usually wrong. Library paws run 13-15 rounds; for chibi toys keep the shape and cut the
  length.
- Arms that worked:
  - bean `mr 6, 9, even 2, 8`;
  - teddy `mr 6, 12, 15, even 3, 12, even 4, 10`;
  - holding a heart `mr 6, 12, even 2, 10, even 4, 9`.
- Arms sewn just under the head, `seam: "flat"` with small `pinch_rounds`, `avoid: true`. On a round belly or
  holding a prop: `dir [-0.3..0.1, -0.5, 1]` + `avoid_toward [0,0,1]`, `contact` 0.6. Hanging arms on a wide body
  get pushed into a T-pose.
- Sitting legs: open tubes sewn on the front of the body pointing forward; the magic ring is the sole. Front legs of
  a sitting animal need a side-to-side `seam_line` (the default gives a peaked top).
- Toe beans read from the front only on the sole (round ~4, az 145-215 on a forward foot).
- Foot with toes: leg from `ch 6`, `oval_axis: "front"`, blo, ankle decrease; toenails = 3 satin ovals.
- Spider legs: 10-st open tubes, `bend` 100 at mid-length toward [0,-1,0], 4 per side at az 62-138, low.
- Flippers: turtle paddles = cone, flatten 0.5, `front [0,1,0]`. Seal front flippers low and angled down (or they
  read as ears). Side flippers `front [1,0,0]`, `dir [0.7,-1,0.1]`.
- `drape` only works on unstuffed open pieces. Lightly stuffed floppy limbs cannot sag.
- An open piece wider than ~24 st melts flat onto the body. Decrease it back to a ~12-st opening, or use a taller,
  narrower cone.

## 9. Tails, wings, fins

- Place tails up and back, never at the bottom rear between the legs (it reads badly on a children's toy, and a low
  tail lifts the toy).
- Curled or loop tail: an 18-24 st tube, `bend` 360 + `coil` 2 toward the side, dir up the back. Bean cat tail:
  `mr 6, 9, even 10`, light, `bend 70 at 6 toward [0.6,1,0]`, at round 6, az 165.
- Coiled snake: root tube from the tail tip (`axis [1,0,0]`, `front [0,0,-1]`), `bend` 540, coil 2, pitch_mm 16,
  head + neck piece on the last round `dir [0,1,0.3]`.
- On a tail pointing back or down, set `"front": [0,1,0]` so az 0 = up before placing spikes or crests.
- Segmented tails: sew each closed segment at `{"round": "start"}` of the previous one (sink 13-16 mm).
- **Wings:**
  - Wings that stick out: a small spoon cone (mr 4, +3/round to ~20) sewn by its opening, pointing out and up.
  - A flattened wing with `seam: "flat"` and the default front sticks out like an arm. Use `front [1,0,0]` and a
    matching `seam_line`. Owl wings pressed to the sides: `front [1,0,0.35]`.
  - Bat wings: `half_circle` 8 rows + `point` edging, attach edge az ~100, dir [1, 0.55, -0.55], face
    [0.65, 0, 1].
  - Butterfly wings: closed stuffed ovals (upper 36 st, 17 rounds), flatten 0.7, attached by the end at az ~112,
    front [0.2, 0, 1].
- **Fins:**
  - Wide fins: open cone `mr 6 -> 36` (+3/round), flatten 0.6, medium.
  - Fish fins: closed, flattened, lightly stuffed, attached by the start. A low fin under the tail points back
    (`dir [0,-0.25,-1]`).
  - Goldfish fan tail: two `half_circle` motifs (rows 9, shell edging), edge-attached, mirrored and splayed.
- Turkey fan: 7 petal motifs (chain 13, tr), edge-sewn at -72..72° behind the back.

## 10. Hair, hats, hoods (worn pieces)

- Anything with volume over the head is a **worn piece**: entry `wear: true` + `pad`, and for a face opening the
  part key `opening {from, stitches}`. A cap sewn at one point with `sink` gets laid onto the head as a swim cap.
  Fringe hair hangs like a curtain, and flat panels read as ear flaps.
- A worn cap sits only when straight and never narrowing (a tilted `dir` or a final decrease floats it). Make it
  ~1.2-1.3x the head count. To tilt it, move `at` off the pole (round N-1, az ±30).
- **Bob wig** on a 36-st head: `mr 6, inc 6 to 24, inc 6 to 42 every 2, 48, even 8`, `opening` from 11 with 16 st,
  `wear` + `pad` 3. On doll and angel heads a wig with `opening` hid the face, so their hair is head colour zones
  (rounds 1-10 + a polygon to the nape with a fringe line).
- **Witch hat:** `mr 4, 6, 8, even 1, 10, 12, even 1, 14, inc 4 to 30 every 2, even 1, flo 30, 36, 42, 48`, band
  rounds 15-16, `wear: true`, `dir [0,-1,0.12]`, pad 2. The cat's ears go below the brim.
- **Ribbed beanie:** `... even 3, rib 54 x2` worn at round end + a popcorn pompom.
- **Santa hat / nightcap:** open cone (`mr 6, +3 every 2 to 30, +3 to 48`, rib cuff) attached by its end on the
  crown (round 1.5, `dir [0,1,-0.12]`, sink 7), `bend` 115-150 to the side, white `popcorns every 2` brim, firm
  pompom at `round: start`. Back-tilted placements bury the cuff.
- **Hat on a frog with eye bumps:** a worn cap centred behind the bumps (round end-2, az 180, dir [0,-1,0.12]).
  Strawberry hat: `mr 6, inc 6 to 24, inc 6 to 48 every 2, even 2`, calyx = 6-point star applique on top.
- **Cowboy hat:** crown from `ch 7` to 40, 2 band rounds, `flo 40`, +12/round to 72, tilted back. A brim cannot
  curl yet, so add a sheriff star.
- **Pumpkin costume hat:** `mr .. 72 every 2, even 1, 66` (tucks the rim), worn, pad 3, leaf as a padded applique
  on its front.
- **Halo:** two mirrored chain cords (curl ~232, ~64 mm) from the front of the crown. A ring reads as a jar lid.
- **Hooded faces** (puffin, pangolin): head in the hood colour with cream ellipse zones.

## 11. Collars, scarves, bows, ruffles, bandanas

- **Collar / snood:** `ring <neck count>` + `around: true`. Scallops: `shells 7 every 6` where body shows below
  (puffin; `shells 9` stuck out as a ruffle there); `ring <neck>, one +6 round, shells 9 every 6` under a big head
  with little body showing (big scallops, not a cape). Don't use `fall: true` on a collar (it makes a tight band).
- **Ruffle / ghost hem:** `ring N, even 1, shells 7 every 6`, `around: true`, dir down. A +6 flare makes a saucer.
- **Mushroom skirt:** `ring <stem count>`, +12, `shells 5 every 6`, one round under the cap rim.
- **Scarf** (approved polar bear):
  ```json
  {"id": "scarf", "rounds": ["ring 48", "rib 48 x2"], "end": "open", "stuff": "none"}
  {"id": "scarf_end", "rounds": ["ch 5", "even 8"], "end": "open", "stuff": "none", "flatten": 0.9,
   "bend": {"deg": 75, "at": 6, "over": 3, "toward": [0, -1, 0]}}
  {"part": "scarf", "on": "body", "at": {"round": 17.5}, "around": true, "dir": [0, 1, 0]}
  {"part": "scarf_end", "on": "body", "at": {"round": 20.5, "az": 18}, "dir": [0.1, -0.2, 1]}
  ```
  The ring goes at the body's top round under the chin, never at mid-body (a belt). Give the body 2 more even rounds
  if the end has no chest to lie on. Chunky turtleneck: a looser ring (54 on a 48-st neck).
- **Bow at the neck:** two padded (`pad` 4-8) chain-oval loops at az ±25 + a padded knot, sewn on the BODY just
  below the head. Closer loops merge into a bib, and triangle wings on the head's lower rim read as a moustache.
  **Side bow on the head:** two loops (`ch 5, 12`; pad 5) ~20° apart, roll ±25, raised knot. Check that it doesn't
  pass through an ear (a flat bow at an ear round pierced it with no flag; on a collar it was approved).
- **Bandana / kerchief:** `triangle` motif (width 16-26), `drape: true`, `roll: 180`, ~2 rounds under the mouth.
- **Collar tag:** a 3-round gold circle motif ~2.5 rounds below a `ring` collar. Bell: a gold ball on the collar.

## 12. Held props and containers

- A prop must read as its object at thumbnail size and be **held at the body**. A plain ball lying by the feet
  read as "a yellow circle" and was removed. A prop under the toy makes it hover.
- **Upright prop** (pumpkin, egg, cheese): a closed piece `attach: "start"` (or by its end pole),
  `dir [0,1,0..0.65]`, `sink_dir [0,0,1]`, sink ~0.5 x its radius, at body round ~3-6. Arms angled in to its sides.
- **Heart held at the chest:**
  ```json
  {"id": "heart", "colour": "red", "motif": {"kind": "heart", "rounds": 3}}
  {"part": "heart", "on": "body", "at": {"round": 12, "az": 0}, "applique": true, "lift": 2, "pad": 11}
  {"part": "arm", "on": "body", "at": {"round": 14.5, "az": 54}, "dir": [-0.3, -0.5, 1], "mirror": true,
   "seam": "flat", "avoid": true, "avoid_toward": [0, 0, 1], "contact": 0.6}
  ```
- **Bean prop:** a firm ball at round 10, az 0, `attach: "end"`, sink 5, + a curled cord.
- **Bone held across:** two mirrored halves (one closed piece cannot be sewn by its middle).
- **Containers:** ~18 st wider than the head, the body raised so the chest shows, props at az ±70. Basket: sc
  tapestry checkerboard (3 st x 2 rounds) + a dark `ring` rim with `around` + `fall`.
- **Pumpkin:** `mr 7, inc 7 to 28, 28, rib 28 x1, dec 7 to 7` + 7 darker meridian lines + stem + curled tendril.
- **Strawberry:** a pointed shape, rounded polygon zones (point down), 5 satin seeds, 3 green leaf ovals or a calyx
  on top. Without the leaves it reads as a heart; a round red ball is not a strawberry.
- **Mushroom prop:** standing on a foot (`on: "foot_L"`), cap on the stem's `round: start`.

## 13. Objects, food, plants

- One iconic prop makes an object toy:
  - a bao in a bamboo steamer;
  - a chick in a zigzag eggshell (`shells 3 tr every 4`);
  - boba with raised pearl appliques (`mr 6, 11`, pad 3.5);
  - cherry and cream on a pudding.

  A plain blob reads as "an onion ball".
- **Wide mushroom cap:** a closed dome, `blo` fold, underside decreased shut, `attach: "end"`, sink ~20. Put things
  on an open stalk's tip at `{"round": "start"}`.
- **Fried egg white:** closed chain oval `ch 9, inc 6 to 84, inc 12 x2, even 1, dec 12 x2, dec 6 to 18`, unstuffed,
  root axis up. A lens reads as a bun, a dish as a hat, a popcorn rim as a doily.
- **Bread loaf:** chain oval `ch 15`, blo base, crust zone from a +3 overhang round, closed at ~33 st (21 makes an
  acorn ridge). Score marks: 3 cream satin ovals.
- **Pickle:** olive green (#7a9a35) with darker padded `mr 6` bumps off the face (bright yellow-green reads as a
  cactus). **Potato:** a smooth egg, flatten 0.15 (lumps read as ears or coins).
- **Puffy heart:** a cone body from the bottom point (+4, +3, +2 per round to 60), 5 even, flat top `dec 12`,
  flatten 0.3 + two 36-st lobes sunk 17 mm at the top corners (round 24, az 90), `sink_dir: "axis"`.
- **Puffy star:** a disc centre (axis forward, 54 st, 2 even) + 5 open cones (mr 6, +3/round to 30, flatten 0.2).
  Faces on a forward disc: round = k0 - r/3.8, az = atan2(x, y) for a point (x, y) mm from the centre.
- **Crescent moon:** one closed tube (mr 4, +2 to 32, 5 even, -2 to 4), `bend` 215 over the whole length, root
  `axis [-1,0,0]`, `front [0,-1,0]`, face at az +90. List both eyes explicitly (no `mirror` on a sideways root).
- **Lying banana:** one root, `axis [0.64,-0.77,0]`, `front [0,1,0]`, `bend` 100 over 20 rounds. A root `bend`
  ignores `toward`, so steer it with `front`.
- **Watermelon slice:** a padded half-moon (mr flat disc top + `blo` fold + rounds shrinking along a semicircle,
  root axis [0,-1,0], flatten 0.6, light). As a small bag motif it won in the lab, but lost on a tapir ("a purse at
  the hip").
- **Tiered tree:** stacked closed cones with flat BLO bases sunk with `sink_dir: "axis"`, + **white** ring-and-shells
  trims (green trims vanish).
- **Tulip:** 3 padded flattened petals low on the cup, the back one above the top; petals a deeper rose than the
  cup. **Sunflower:** a 22-round stem, leaves low facing front, a popcorn seed ring around a light face, a double
  petal row, a tall pot.
- **Upright petals / crown leaves:** leaf motifs `attach: "edge"`, `sew: "base"`, facing out (as appliques they
  read as stripes).
- **Balloon animals:** closed 18-24 st capsules sewn pole to pole (sink ~13); gloss = thin pale felt ovals.
- **Robot:** a wide chain-oval head (`ch 11` -> 52, BLO/FLO edges, flatten 0.12) on a 36-st cylinder + a dark
  screen applique with light satin eyes.
- **Jellyfish:** coil tentacles all the **same length** (it stands on them), `ring 24 -> 72` frill under the dome.
  Whale spout: 3 coiled tubes splayed out and down.

## 14. Colour zones, spots, stripes, textures

- **Colours:** sample flat regions of the concept (median of a 13x13 patch, `measure.py`), not lit highlights.
  Compare the render with the sample after the first build. Very dark yarn renders light (use ~#19191c for black,
  #0d1424 for navy). Name every custom colour in `colour_names_ru`; auto names can be wrong (#2f5a2a ->
  «тёмно-серый»).
- **Zone syntax traps:**
  - A full ring has no `az` (`[-180, 180]` colours one stitch).
  - A polygon crossing ±180 should start at the vertex nearest az 0.
  - Embroidery points from 180 to -160 go the long way round, so write -180.
- **Calico patches:** plain (round, az) ellipse zones from round 3 on, overlapping so no 1-st gaps remain. No
  `wobble` on small heads.
- **Stripes:**
  - Tiger: single-round bands on the back half (az ~100-260) + 1-2 st columns on the crown.
  - Cat loaf: partial rows az -62..62 every 3rd round.
  - Tapered wedges turn into dashes.
- **Spots on a ball:** flat `mr 6, 10` appliques (satin dots look like holes; small ellipse zones look blocky). A
  small spot under an eye must spill onto the cheek or muzzle to be seen.
- **Belly patch:** a chain-oval applique is smooth; an ellipse zone is blocky. Bluebird chest ellipse within ±48°.
- **Bobbles and fuzz:** `popcorns every 3` rounds with offsets (every 2 reads as bead bands; bands on the face read
  as warts), or padded `mr 6, 12` appliques (pad 5) in staggered rings. Poodle curls: a closed topknot + rings of
  padded appliques.
- **Fringe:** curl 200-240° gives volume (above ~300° it falls flat); `spread` ~0.15; manes only where the surface is
  roughly vertical. Unicorn mane: two back lines (az 165/195, 40 strands, 40 mm) + a side-swept forelock. Keep
  tufts short and curled (long ones read as straws).
- Drop details that look like loose yarn (thin gill lines, satin drips, lashes, toe dots). A colour zone reads better
  than thin embroidery.

## 15. Naming and the written pattern

- Every part needs `name_ru` and `name_en`, and every custom colour needs `colour_names_ru`. No Latin ids may appear
  in `PATTERN_RU.md`. Colour ids appear raw in EN, so name them with plain English words.
- **Instance names:**
  - Never end in a digit (`leg1` prints as "No 2").
  - `_a.._f` are read as sequence numbers; skip `l` and `r` (`spike_l` prints as a left copy).
  - Unknown qualifier words print as "№ N". Use words like front, back, lower, mid.
- Parents before children in `assembly`. Face items `on` a placed instance (`arm_L`), not a part id.
- After a chain start the counts must be reachable: `ch N` makes 2N, and every `inc K to M` needs M - current to be
  a multiple of K (`ch 6, inc 6 to 28` fails, 24 or 30 work).
- Read `PATTERN_RU.md` once as a crocheter. The ear lines must say how to press («тарелочка» / «ложечка»). Check
  what it says to stuff, curl and tack.

## 16. Known engine gaps: work around them or report them

- No gravity sag of stuffed parts; `drape` only on unstuffed open pieces. Curled-up poses (a head on a tail) can't
  be posed.
- No joining two pieces into one (a puffy heart shows a crease). One `bend` per piece (no S-neck).
- No brim that curls up. `bend` on a worn piece bends the brim end, not the tip.
- An unstuffed closed piece still renders ~3 cm thick. A root `bend` does not visibly curve a capsule.
- **`opening` / ring-zone pattern bugs:**
  - `opening` allows no shaping after the window. Older engines print a false ERRORS line, so check the pattern.
  - Az colour zones on `ring` / `rib` rounds are drawn but not written in the pattern. Use whole rounds or sc
    tapestry there.
- `face_ok` checks only safety eyes and noses: check embroidered eyes and satin mouths by eye.
- **`cute` limitations:**
  - It treats a tall hat as head on one-piece bodies.
  - It suggests shrinking big ears.
  - `--pick` with `scale_part head` can write irregular rounds and rename face items. Apply its idea by hand.
- `eye_match.py` cannot see embroidered eyes.
- A world-vector `sink_dir` pointing away from the parent still says "press in ~9 mm" in the text.
- Satin details render with a dark rim. The EN text says "a oval".
- `package` without `--refs` writes `quality: {}`.

Report a new gap in the toy's report. Don't edit the engine inside a toy job.
