# Brief: from a picture to an amigurumi toy (agent workflow)

You get one or more pictures of a toy, a pet or a drawing. You produce `amikit/toys/<name>.json` (see `SPEC.md`):
a toy that a crocheter can make, that looks like the picture, and that is cute. The engine derives the 3D look and
the written pattern from your spec; you only decide construction and numbers.

## 1. Read the picture (write these notes down first)
- Species / character, pose (sitting, standing, lying), the pieces you see (head, body, muzzle, ears, arms, legs,
  tail, wings, patches, clothes) and which are separate pieces vs colour changes.
- Proportions from the picture, measured in the image: head width vs body width, head height vs total height,
  eye height on the head (fraction from the top), eye spacing (fraction of head width), muzzle size, ear length.
- Colours: sample medians of flat regions (e.g. with PIL); name them.
- If only one view: decide the unseen sides by the most common designer construction (`../pattern-harvest/v3/learn/KNOWLEDGE_PATTERNS.md`, `KNOWLEDGE_FACES.md`).

## 2. Choose the construction (as a designer would)
Start from the closest example in `amikit/toys/` (teddy, bunny, mouse, duck, panda, fox, penguin, beagle, corgi,
tabby) and change it; don't start from scratch. SPEC.md's last section lists constructions that work.
- Size: head max 42-60 st for a ~12-15 cm chibi; body max ~0.7x head max.
- Rounds: increases +6 per round, a few even rounds for height, decreases -6 per round (designer grammar).
- Markings as colour zones (stitch-exact), patches as appliques, embroidered details as `stitch`/`satin`.

- For every part, start from the library: `python -m amikit.library suggest <role> <animal> <head max> <kind>`
  and `stats`; adjust to the picture. Use `seam: "flat"` for arms, sitting legs, ears and tails unless the
  picture clearly shows a round seam.

## 3. Build, look, fix (the loop)
```
python -m amikit.build amikit/toys/<name>.json ../out/amikit/<name> --ref <picture>
```
Open `sheet.png` (reference first, then front, 3/4, side, back). Compare **visually** and fix the biggest
difference first: silhouette & proportions → placement of ears/muzzle/eyes → colour regions → face details →
small things. Read `flags.txt` and `library_audit.txt` every time. 6-10 iterations is normal. For a single-view picture, compare with
the 3/4 view (`view_34.png`) most; it shows the toy's left side (face toward the viewer's left), so mirror the
picture in your head (or in a comparison crop) when it faces the other way. A sitting toy may rest on its legs;
`flags.txt` warns only when another piece (a tail) lifts it.

Check the silhouette by numbers too: `python -m amikit.fitview <picture> <out>/view_front.png <out>/fit.png`
(`--crop x0,y0,x1,y1` for one view of a turntable, `--mirror` if the picture faces the other way): the reference
outline in red over ours in blue, plus IoU and width profiles. Fix where the outlines differ (head width, ear
spread, body/leg width); then check the face on a close-up — the silhouette does not see eyes or the muzzle.

## 3b. Ears and cuteness by construction
- Ears: round = `dish` on an `attach: "edge"` entry, pointed/long/floppy = `spoon` (SPEC.md); or the `ear` preset.
  Look at a head close-up (front, 35, 90 degrees) next to the concept: thickness, how they stand, size, placement.
- New toys: start the head from `"preset": "cute_head"`.
- `python -m amikit.cute <spec> --ref <concept> --out <dir> --closeups` (or `amikit_job.py cute` in the skill): the
  engine measures, scores and proposes edit ops with their predicted effect; you (or `amikit.cute_llm` with an API
  key) pick the ones the concept agrees with. Eye moves carry brows/cheeks along but not colour-zone face masks.

## 3c. Shaped flat pieces, collars, face masks (SPEC.md)
- A flat accessory with a shape (kerchief, wing, fin, plate, leaf, scale, honeycomb, slice, heart, moon) is a
  `motif`, not a disc or a flattened tube: the rows make the outline and the pattern says so row by row.
  Blind A/B vs the disc / tube / colour-zone builds (run-notes/2026-10-07-shaped-flat-pieces.md) - won: the
  watermelon-slice bag (half_circle, clear), the scalloped puffin collar (`ring` + `shells`, clear), the otter face
  mask (`mask` + `anchor: "eyes"` + `drape`), kerchief (`triangle` + `drape`), wings (half_circle + picot edging),
  heart, hexagon cells, pointed-leaf scales, the ribbed snood (all slight). Lost: plump stuffed plates (dino: keep
  stuffed cups), a shamrock of three `heart`s (satin was crisper), the lily-pad collar (a stiff ring reads as a thin
  band or a skirt; the concept's drooping plate is better as the old open cup).
- A face mask that must follow the eyes: applique with `anchor: "eyes"` (or `"eye"` per side), never a colour zone.
- Collars / snoods: `ring N` with N = the neck round's count, `around: true`; scallops = a `shells` round.

## 4. Keep it honest and cute
- Every shape comes from rounds, stuffing, flatten/squash/bend, appliques, rows or cords: things a crocheter does.
- Cute = big head, eyes low and wide, small muzzle, short soft limbs, round everything. When the picture and
  cuteness disagree, stay closer to the picture but never lose the baby-schema.
- Read the produced `PATTERN_RU.md` once at the end as a crocheter would.

## 5. Edits later
Small changes (bigger ears, another colour, move the eyes) go through `amikit/edit.py` ops (or the editor /
`llm_edit.py` from a text request); no need to rewrite the spec.
