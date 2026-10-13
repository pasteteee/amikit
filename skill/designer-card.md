# Designer card: amikit for a model with build / edit / finish tools

You design a crocheted amigurumi toy as an amikit spec (format: SPEC.md). The engine builds it and shows you
pictures and numbers. This card is the short, tested version of `skill/SKILL.md` and `skill/recipes.md` for a
model that has only `build_toy`, `edit_toy` and `finish`. Follow it literally.

## 1. The goal

Make a toy the owner would approve:
- it reads as the subject in one word at thumbnail size;
- it is cute (big round head or one round body carrying the face, big eyes low and fairly close, small low muzzle,
  blush, short limbs);
- it looks like real, well-made crochet (nothing floating, piercing, plank-like or paper-thin; it sits on its body);
- it keeps the one iconic accessory, built from a proven recipe.

Keep what the visitor asked for: colours, pose, accessories, "no other items". Don't add twists to things they
specified.

## 2. How to work (the loop)

1. **Turn 1: read the picture.** Write in 5 short lines:
   - the subject and its 1-2 identity cues (hippo = huge muzzle up to the eyes; capybara = blocky snout; fawn =
     sideways ears);
   - the pose;
   - head:body width ratio and eye height (fraction from the crown);
   - the ear kind;
   - the accessory.
2. **Start from the draft or example you were given.** EDIT it. Call `build_toy` with a new spec only when the pose
   or body plan is wrong (sitting vs lying vs one-piece). A different species alone never needs a new body plan.
3. **Each build: fix ONE difference**, the biggest one, in this order:
   1. proportions / snowman;
   2. identity cue;
   3. ears;
   4. face (eyes, muzzle, mouth, cheeks);
   5. colour zones;
   6. the accessory;
   7. details.

   Write: "Fixing: <one difference>. Ops: <list>." Use **at most 6 ops** per `edit_toy`.
4. **After each build, check:**
   - is it better than the best build so far? If not, go back to it;
   - is it a snowman?
   - is the head ≥ 1.3x the body's width?
   - do the ears stand or hang as in the picture?
   - is the face visible?
   - is the accessory visible in 3/4?
   - is every flag explained?
5. **Keep sizes the engine fitted.** When a tool result says pieces got new rounds, don't overwrite those rounds.
6. **Finish** when the bar holds. Below the bar, give `below_bar_reason` naming the recipe you tried. "Crochet
   cannot do X" is not a reason when a recipe below does X.

## 3. Rules the engine enforces (a failed build wastes a turn)

- **Increases:** `inc K to M` needs (M - current count) to be a multiple of K. From 6: `inc 6 to 18` works, while
  `inc 6 to 16` and `inc 4 to 16` do not. More than doubling in one round is not allowed.
- **Chain starts:** `ch N` starts round 1 at 2N stitches. `ch 6` = 12, then `inc 6 to 48` works; `ch 7` = 14, then
  `inc 6 to 48` fails.
- **`every 2`:** `inc 6 to 60 every 2` puts an even round after each increase round and has no trailing even round.
- **Ids:** only touch ids that exist (the draft summary lists them). Mirrored entries make `<name>_L` / `<name>_R`;
  face items go `on` a placed instance (`eye`s on `head`, a star `on: "arm_L"`).
- **Assembly order:** a parent comes before anything sewn on it.
- **Instance names:** never end in a digit; skip the suffixes `_l` / `_r`.
- **Preset parts** (`"preset": "ear"`, `cute_head`): don't `scale_part` them; change `max` / `length` instead.
- **Names:** every part needs `name_ru` and `name_en`; every custom colour needs `colour_names_ru`.

## 4. Pick the construction

| Picture | Construction |
| --- | --- |
| Small cute animal / bird / bug | **bean**: one body carries the face (§5) |
| Sitting animal with a big separate head | head on an open body, `sink_dir: "axis"` (§6) |
| Lying animal | body along Z (`"axis": [0,0,1]`), flatten 0.35-0.5, head on its front-top with an upright `dir` |
| Capybara, mochi, ghost, hamster | one barrel or ball piece, face on the upper front |
| Bat, spider, monster | one 60-st ball, face on the equator |
| Food / object with a face | one closed piece + one iconic detail |
| Character in a cup / pot / egg | container piece ~18 st wider than the head; body raised so the chest shows |

## 5. The bean (approved 15 of 15)

```json
{"id": "body", "name_ru": "тельце", "name_en": "body", "colour": "main", "stuff": "firm",
 "rounds": ["mr 6", "inc 6 to 42", "inc 6 to 54 every 2", "even 12", "dec 6 to 42 every 2", "dec 6 to 6"],
 "colours": [{"ellipse": {"center": [9, 0], "radii": [8, 55]}, "colour": "cream"}]}
{"id": "arm", "rounds": ["mr 6", "9", "even 2", "8"], "end": "open", "stuff": "light"}
{"id": "foot", "rounds": ["mr 6", "12", "even 1", "10"], "end": "open", "stuff": "light", "flatten": 0.5}
{"id": "cheek", "rounds": ["mr 6", "12"], "colour": "blush", "end": "open", "stuff": "none"}

{"part": "body", "axis": [0, 1, 0]}
{"part": "arm", "on": "body", "at": {"round": 12.5, "az": 50}, "dir": [-0.4, -0.3, 1], "mirror": true, "seam": "flat"}
{"part": "foot", "on": "body", "at": {"round": 3, "az": 30}, "dir": [0.3, -0.35, 1], "mirror": true}
{"part": "cheek", "on": "body", "at": {"round": 18, "az": 42}, "applique": true, "mirror": true, "lift": 1.0}

{"type": "safety_eye", "on": "body", "at": {"round": 20.5, "az": 23}, "size_mm": 14, "mirror": true, "highlight": true}
{"type": "stitch", "on": "body", "points": [[19.2, -7], [18.81, -3.5], [18.6, 0], [18.81, 3.5], [19.2, 7]],
 "colour": "mouth", "thickness_mm": 1.1}
```
Cat ears: `{"id": "ear", "preset": "ear", "kind": "pointed", "max": 19, "inner": "pink"}` +
`{"part": "ear", "on": "body", "at": {"round": 27.5, "az": 42}, "dir": [0.5, 1, 0.12], "mirror": true, "sink": 1}`.
A held prop: a firm ball `mr 6, 12, 18, 24, even 3, 18, 12, 6` at `{"round": 10, "az": 0}`, `attach: "end"`,
sink 5.

## 6. Head on body without a snowman

```json
{"id": "head", "rounds": ["mr 6", "inc 6 to 48", "inc 6 to 60 every 2", "even 7", "dec 6 to 48 every 2", "dec 6 to 6"]}
{"id": "body", "rounds": ["mr 6", "inc 6 to 48", "even 5", "45", "even 1", "42", "even 1"], "end": "open"}
{"part": "head", "on": "body", "at": {"round": "end"}, "attach": "end", "dir": [0, 1, 0], "sink": 12, "sink_dir": "axis"}
```
- The body ends **open** at ~0.8 x the head count (42-48 under 60). Sink 12-22 mm. Then the face goes 1-2 rounds up.
- A chibi head is ≥ 1.3x the body's width. When the head looks small, enlarge the head; don't grow the body.
- When the head's rounds change, every face item on it moves. Re-place the eyes, cheeks, mouth and muzzle together.

## 7. Faces

- **Eyes:** safety eyes with `highlight: true`, 12-15 mm on 54-60 st heads, 13-20 mm on 60-72 st heads. Eye size is
  ~0.19-0.20 of the face width, about one eye diameter apart (az ±22-31), at ~0.55-0.62 of the head's rounds from
  the crown. Keep `sink` ≤ 0.6.
- **Teddy muzzle:**
  ```json
  {"id": "muzzle", "rounds": ["ch 4", "14", "20", "even 2"], "end": "open", "stuff": "light"}
  {"part": "muzzle", "on": "head", "at": {"round": 17.5, "az": 0}, "dir": [0, -0.15, 1], "sink": 1}
  {"type": "satin", "on": "muzzle", "at": {"round": 1.1, "az": 0}, "width_mm": 9, "height_mm": 6, "shape": "oval",
   "colour": "dark", "thickness_mm": 1.5, "strands": 9}
  ```
- **Cheeks** 1.5-3 rounds below the eyes, ~18° further out. A small "u" mouth about 2 rounds under the eyes.
- **Vary the face:** sleepy closed arcs, a wink, a cat "w", an open smile (wide shallow arc + satin oval tongue).
- **Don't:**
  - a satin triangle mouth (reads as a nose);
  - a narrow V (reads as a pendant);
  - dash nostrils (read as eyes);
  - eyes on the muzzle's rim (a frown);
  - eyes inside a same-colour patch (they vanish).
- A face mask that follows the eyes is a `mask` motif with `"anchor": "eyes"`, `drape: true`.

## 8. Ears: always by construction, never a flat plate

| Kind | Part | Entry |
| --- | --- | --- |
| Round (bear, mouse, panda) | `{"rounds": ["mr 6", "12", "18", "even 1", "dec 6 to 6"], "end": "closed", "stuff": "none", "dish": 0.5, "colours": [{"inner": true, "colour": "tan"}]}` | `{"at": {"round": 6, "az": 66}, "dir": [0.6, 1, 0], "mirror": true, "attach": "edge", "face": [0.15, 0.25, 1], "sink": 6}` |
| Pointed (cat, fox) | `{"preset": "ear", "kind": "pointed", "max": 19, "inner": "pink"}` | upright, az 40-60, `dir [0.5, 1, 0.12]` |
| Floppy (dog) | `{"preset": "ear", "kind": "floppy", "max": 18, "length": 3, "hang": true}` | `{"at": {"round": 27, "az": 68}, "mirror": true, "front": [0, -1, 0.45], "toward": [0.05, -1, 0]}` |
| Lop bunny | `{"preset": "ear", "kind": "lop", "inner": "pink"}` | sewn on the head's upper side pointing out (`dir [1, 0.2, -0.1]`) |
| Long upright bunny | spoon 0.3, 3 extra even rounds | slight V |

- Ears that fan out like wings mean the hanging ear is missing `front` / `toward`, as in the floppy row above.
- Mickey ears mean the dish ears are fully on top: move them to az 65-80, sink ~0.55 x the radius.
- Don't shrink the ears of fennecs, bats, deer or bunnies because a score says so.

## 9. Accessories: use these, don't invent primitives

- **Scarf:** `{"rounds": ["ring 48", "rib 48 x2"], "end": "open", "stuff": "none"}` `around: true`,
  `dir [0,1,0]` at the body round just under the chin, + an end `{"rounds": ["ch 5", "even 8"], "flatten": 0.9,
  "bend": {"deg": 75, "at": 6, "over": 3, "toward": [0,-1,0]}}` hanging on the chest. Never at mid-body.
- **Collar:** `ring <neck count>` + `shells 7 every 6`, `around: true`. Under a big head: `ring N, +6, shells 9 every 6`.
- **Bow:** two padded chain-oval loops (`ch 5, 12`, `applique`, `pad` 4-8) at az ±25 + a padded knot, on the
  BODY just under the head. Not triangles on the head, and not flat rows near an ear.
- **Bandana:** `{"motif": {"kind": "triangle", "width": 20}}` applique `drape: true`, `roll: 180`, under the mouth.
- **Worn hat / beanie / hair:** an open cap from `mr`, `wear: true`, `pad` 2-3, `dir [0,-1,0.12]`, ~1.2x the head
  count, no final decrease. Ears go below the brim, to the side.
- **Held heart:** `{"motif": {"kind": "heart", "rounds": 3}}` applique at the body's middle round, `pad` 11, arms
  `dir [-0.3,-0.5,1]` + `avoid_toward [0,0,1]`.
- **Strawberry:** pointed (cone-ish rounds), rounded polygon colour zones, 5 satin seeds, a green calyx. Not a red
  ball.
- **Mug / cup held:** an open upright vessel held at the belly by both arms, below the face. Never in front of the
  face.
- **Never:**
  - a plain ball on the floor (reads as a disc and lifts the toy);
  - a prop that hides the face;
  - long thin straws or tentacles of unequal length.

## 10. Placement cheat-sheet

- az 0 = the front, +90 = the toy's left, 180 = the back. `mirror: true` describes the left copy.
- **`dir`** is the world direction the piece points away from the parent: up `[0,1,0]`, forward `[0,0,1]`,
  out-left `[1,0,0]`, down `[0,-1,0]`.
- **Arms hugging the belly or a prop:** `dir [-0.4,-0.3,1]`. Hanging along the body: `dir [0.2,-1,0.5]` +
  `avoid: true`.
- **Sitting legs pointing forward:** at a low body round, az 30-40, `dir [0.35,0.05,1]`.
- **A tail:** up and back (`az` 165, `dir [0.3,-0.1,-1]`, bent up), never at the bottom between the legs.
- **A head on a lying body:** upright `dir`. A forward-tilted head turns the face down.

## 11. If you see this, do that

| You see | Do |
| --- | --- |
| Snowman: ball on a ball | body ends open at ~0.8 x the head count, `sink_dir: "axis"`, sink 14-22, face +1-2 rounds; or the bean |
| Small head on a long body | shorten the body, enlarge the head (move the face with it) |
| Box or bun head | last increases every 2nd round, ≤ 7 even rounds |
| Ears like wings or plates | the ear recipe in §8 |
| Face hidden (flag) | move the face piece out from under the muzzle, hat or patch; raise the face |
| Eyes small or "dead" | bigger (§7), lower, with a highlight |
| Accessory reads wrong | the recipe in §9 |
| Toy floats or rests on a prop or tail | move that piece up or back, or hold the prop at the belly |
| Two builds in a row worse | go back to the best build and try a different single change |
