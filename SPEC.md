# amikit toy spec (v1)

One JSON file describes a toy completely: pieces (as crochet rounds), how they are sewn together, the face.
The engine derives everything else from it: the 3D shape of every piece from its stitch counts, the yarn relief,
the renders, the GLB, the written pattern. Edit the spec (by hand, through `amikit/edit.py` ops, or in the
editor) and rebuild; never edit a mesh.

```bash
# from amigurumi-lab/, with ../.venv
python -m amikit.build amikit/toys/teddy.json ../out/amikit/teddy --ref <concept.png> [--fast]
#  -> view_front/34/side/back.png, sheet.png (reference first), flags.txt, PATTERN_RU.md / PATTERN_EN.md
```
Build takes ~4 s (`--fast` ~3 s). Read `flags.txt` after every build.

## Frame and units

- Millimetres. World: **X = the toy's left** (viewer's right in the front view), **Y = up**, **Z = front**.
- Every piece has its own axis from its **start** (magic ring / chain, round 1) to its **end** (last round).
- `az` (degrees) is a direction around a piece, measured from that piece's front reference: 0 = front,
  +90 = toward the toy's left (+X), -90 = toward its right, 180 = back. For a piece whose axis points forward
  (a muzzle) the reference is "up": az 0 = top, 180 = bottom, +90 = toy's left.
- `round: r` = the middle of round r of that piece; `12.5` = between rounds 12 and 13; `"end"` / `"start"` =
  the centre of the last / first round (a pole for closed pieces, the opening's centre for open ones).

## name, title

`"name": "teddy"` (an id), `"title": "Teddy bear"`, optional `"title_ru": "Мишка Тедди"` (the Russian pattern's
title), optional `"subject": "bear"` (the animal for the library audit; default from the name).

## yarn

```json
"yarn": {"weight": "DK", "hook_mm": 2.5, "stitch_mm": 4.0, "round_mm": 3.6, "thickness_mm": 1.9}
```
`stitch_mm` / `round_mm` are the gauge (width of one sc, height of one round). They scale the whole toy.

## colours

`"colours": {"brown": "#9c5428", "tan": "#dcb488"}`: names used everywhere else. Use the colour as it looks in a
photo (the renderer is calibrated so a lit front surface shows that colour). `black` and `white` exist by default.
Common English names (brown, tan, pink, sage, ...) are translated in the Russian pattern; for others add
`"colour_names_ru": {"tabby": "серо-коричневый"}`.

## parts

```json
{"id": "head", "rounds": ["mr 6", "inc 6 to 48", "even 8", "dec 6 to 6"],
 "colour": "brown", "end": "closed", "stuff": "firm",
 "colours": [{"rounds": [10, 14], "az": [-30, 30], "colour": "white"}],
 "flatten": 0, "squash": 0, "bend": {"deg": 60, "at": 5, "over": 2}}
```
- `name_ru` / `name_en`: the piece's name in the written pattern (otherwise taken from the id: head, ear, ...).
- `rounds`: steps, expanded to stitch counts (checked: max +100% / -50% per round):
  `mr N` magic ring (any N; more than doubling in one round is allowed only as "3 sc in one stitch") ·
  `ch N` chain-oval start: round 1 has **2N** stitches (sc along both sides of the chain,
  3 in the turns); the straight part is N-2 stitches long and runs side to side unless `oval_axis: "front"`
  (front to back: a sitting body longer front-to-back, a foot); increases go around the two turns ·
  `inc K to M` (+K per round up to M) · `inc K xT` · `dec K to M` · `even T` · a bare number (one round with
  exactly that many) · `inc 6 to 36 every 2` (an even round after each increase round) · `blo N` / `flo N`
  (back/front loops only: a crisp fold line, e.g. the edge of a flat sole) · `popcorns every K [from O]` (a round of
  sc with a popcorn - 5 dc in one stitch closed together, a bobble on the right side - every K stitches, starting O
  stitches later; the count stays the same: fleece, berry seeds, textured bands; alternate `from` between rounds).
- `end`: `closed` (finished by decreasing and pulling the hole shut) or `open` (left open and sewn on by the
  opening).
- `stuff`: `firm` | `medium` | `light` | `none` (default firm for closed, none for open). Stuffing is physics: a
  firm head bulges, an unstuffed open cup keeps its crochet shape.
- `colour` + `colours` zones (stitch-exact colour changes; later zones paint over earlier ones):
  `{"rounds": [first, last], "az": [from, to], "colour": c}` a band/rectangle;
  `{"shape": [[round, az], ...], "colour": c}` any polygon in (round, az) (V-shaped bibs, heart faces, blazes);
  vertices are literal (straight lines in round/az), so a shape across the back is written with az past 180;
  `{"ellipse": {"center": [round, az], "radii": [rounds, degrees]}, "colour": c}` (patches, eye masks).
  `{"inner": true, "colour": c}` the inside of an ear, shaped like the ear: a spoon ear's whole hollow (the front
  wall between the side folds, `margin_st` 1 stitch of rim left, `width` 0..1 default 1), a dish ear's face rounds
  minus `margin_rounds` (1), a flattened ear's middle `width` (0.62) per round; `from` (3) keeps the tip in the
  ear colour. Use it instead of an oval applique on pointed / long ears (a coloured inside wants `spoon` >= 0.55
  to read as a hollow; the ear preset's `inner` key does both);
  `{"patch": {...}, "colour": c}` a patch placed in millimetres on the surface: `at` {round, az} on this piece
  or on `on` (another piece), or `eye`: "L"/"R" (follows the eye when it moves); `size_mm` [w, h] or d;
  `offset_mm` [out, up] (out = away from the toy's middle line, up = the toy's up along the surface); `rotate`;
  `wobble` 0..0.5 + `seed` (an irregular coat patch); `mirror: true` both sides; `exact: true` = the patch stays
  where its anchor says on both copies of a pair (a spot on one side). Several pieces can name the same patch:
  each colours the stitches the ellipse reaches (a spot under the eye spreads onto the muzzle). Made by the
  `spot` / `marking` ops (amikit/custom.py);
  `"side": "L"|"R"` on any zone: only on that copy of a mirrored pair.
  Stitch centres inside get the colour - where the stitch really sits: stitch columns drift along the spiral
  (0.1 st per round), and zones take that into account, so a zone drawn at az 0 stays on the face's centre line
  in every round; the written pattern lists exactly the same stitches. On the right-hand copy of a mirrored pair,
  zones are mirrored (az flipped).
- `flatten` 0..1: the piece pressed flat along its front reference (unstuffed ears, flat arms, a tail).
  Width grows as it flattens (a flattened 12-st tube is ~6 st wide).
- `spoon` 0..1 (open, unstuffed pieces; replaces `flatten`): the cup or cone is pressed side to side only partly — the
  front wall pushed into the back wall: two sc layers (~3 yarn thicknesses apart) with rounded folds at the sides, a
  hollow front (the sag grows with the amount) and a rounded back; the opening stays a curve and is sewn following
  the parent. Pointed ears 0.35, long ears 0.3, floppy ears 0.45. The pattern says «ложечка … (spoon N %)» with the
  depth, thickness and width in mm. **Plump ears:** `"stuff": "light"` on a spoon ear puts a thin layer of stuffing
  between the walls: the ear is ~1.9x as thick (DK: ~11 mm) with thick rounded edges and still hollow in front
  (`medium` 2.6x); the pattern says «Набейте слегка — тонким слоем… Затем сформируйте ложечку». Concept-style soft ears
  (puppy drops, lop bunny cups, big fox cones, kitten cups) look right this way; a bare two-layer spoon reads as a thin
  cup. A lightly stuffed pressed ear is still a soft piece: it lies on what it meets and does not hold other pieces off.
- `dish` 0..1 (unstuffed cups, open or closed): the cup is pressed along its axis — the magic ring becomes the centre
  of a concave front face (rounds up to the first widest round lie as a disc), the rounds after it roll back into a
  thick rounded rim (at least two layers + the fold; more rounds at full width = a fatter rim) and lie behind as a
  second layer (decreases pull it in). Place it with `attach: "edge"` (stood on its rim, sewn by the lower arc,
  `face` = where the hollow looks, `sink` ~0.55 x its radius). Round bear/quokka/capybara/otter ears, 0.5. The
  pattern says «тарелочка … (dish N %)», which rounds form the face and the rim, and how many rim stitches to sew.
- `preset` (expanded before anything else into plain rounds and keys; your own keys win; `"crown": "round"` on a
  cute_head gives a domed crown — counts following a sphere, fewer straight rounds — instead of the flat-topped one):
  `{"preset": "ear", "kind": "round"|"pointed"|"long"|"floppy", "max": N, "length": L}` — designer-consensus
  rounds (`max` default from the head's widest round x the library median for that kind) + `dish` 0.5 (round, with
  the edge-placement defaults on its assembly entries) or `spoon` 0.35 / 0.3 / 0.45;
  `{"preset": "cute_head", "max": 54, "oval": false}` — a head wider than tall with full low cheeks: +6 every round
  to max-12, the last two increases every other round, max/18 straight rounds, decreases every other round to
  max-12, then +6 decreases to close (`oval: true` starts from a chain: wider still).
- `squash` 0..1: only the opening is pinched into a slit as it is sewn (a duck bill, a bear ear sewn by a pinched
  edge), ramped over the last 3 rounds.
- `bend` + `"drape": true` (unstuffed open pieces sewn by their end: lop / floppy ears): the piece hangs from its seam under its own weight — its centre line leaves along the entry's `dir`, turns toward gravity (`relax_mm` 14: how soon it gives in, `min_radius_mm` 7: the tightest curve) and lies over the stuffed pieces it meets (`clear_mm`), its length kept; `deg`/`at` only describe it in the pattern («складку не подшивайте, ушко само свисает»). Sew it pointing out from the head's upper side (`dir` ~[1, 0.2, -0.1]); preset `ear` kind `lop` (or `long`/`floppy` + `"hang": true`) writes it all, with a tapered oval-leaf ear. The centre line starts on the head and lifts off it over the fold at the seam (no arch above the head). A pressed ear (spoon, or flatten >= 0.3) folds only across its width, so below the seam it twists until its width lies across the fold: `twist` 0..1 = how far (1: flat against the head - a dog's ear; 0: it keeps facing where the entry's `front` says, its inside showing - a lop bunny's pink inside). Default: 0 when the ear has an `inner` colour zone, else 1; the pattern states it in the drape comment and, for 0, tells the maker to let the inside show.
- `bend` + `"coil": q` (q > 1, `true` = 4): a spiral instead of one arc - from round `at` to the free tip the
  curvature grows q-fold, so the piece winds inward and may turn past 360° (a seahorse / chameleon tail, `deg`
  360-540); `pitch_mm` (default ~2 yarn thicknesses) moves each turn sideways. The pattern says to roll it up from
  the tip and tack the turns.
- `bend`: `{"deg": 60, "at": 5, "over": 2, "toward": [0, 0, 1]}` the axis bends by `deg` around round `at`.
  The section between `at` and the seam keeps the entry's `dir`; the free section bends toward `toward` (world
  vector, mirrored for pairs; default the piece's front reference). A foot, an elbow, a curled tail, an ear
  tip falling forward. `bend_toward` on the assembly entry overrides it per entry.
- `front` (on the assembly entry) chooses the piece's front reference, i.e. which way `flatten` presses and
  where its az 0 is: for a flipper or a flat arm seen face-on from the front, leave the default; to see it
  edge-on, set `front` to the side.

## assembly (order matters: parents first)

```json
{"part": "body", "axis": [0, 1, 0]}                                          // the root, start at the bottom
{"part": "head", "on": "body", "at": {"round": "end"}, "attach": "end", "dir": [0, 1, 0], "sink": 4}
{"part": "muzzle", "on": "head", "at": {"round": 15, "az": 0}, "dir": [0, 0.05, 1]}
{"part": "ear", "on": "head", "at": {"round": 5.5, "az": 68}, "dir": [0.75, 1, 0.1], "mirror": true}
{"part": "inner_ear", "on": "ear", "at": {"round": 4.6, "az": 0}, "applique": true, "mirror": true}
{"part": "arm", "on": "body", "at": {"round": 15, "az": 80}, "dir": [0.15, -1, 0.15], "mirror": true,
 "avoid": true, "contact": 0.6}
```
- `on` + `at`: where on the parent it is sewn. `dir`: the direction the piece extends **away** from the parent
  (world vector; default: the parent's surface normal there). `attach`: which end touches the parent (`end`
  default: the opening or the closing pole).
- Open pieces (`end: open`) are sewn by their opening: the opening is seated on the parent and follows its
  curvature (the seam). Closed pieces are pressed in by `sink` mm (default 12% of their radius).
- `mirror: true` makes a left/right pair: the entry describes the toy's **left** (+X, positive az, positive
  dir.x); the right copy is mirrored. Instances are named `<part>_L` / `<part>_R`; a mirrored child of a mirrored
  parent (`"on": "ear"`) follows the same side.
- `applique: true`: a flat piece (e.g. `["mr 6", "inc 6 to 12"]`, `end: open`) sewn flat onto the parent's
  surface, centred at `at` (inner ears, panda eye patches, a belly patch, a heart). Its own up is the world's
  up projected onto the surface (a chain-oval applique's long axis is horizontal); `roll` rotates it (mirrored
  for pairs).
  `pad: 6` (mm) stuffs it lightly so it domes out: thighs/haunches of sitting animals, puffy cheeks.
  `along: "parent"` lays its long axis (a chain-oval applique's chain) along the parent's length — an inner ear
  follows a leaning ear (use it instead of guessing `roll`). `inset: 1.4` (mm) sews it in flush: the parent's
  fabric gives way under it, leaving a rim of the parent's colour around it, as on a real ear.
  `offset_mm: [dx, dy]` moves it in its own plane (dx toward the toy's front/side, dy up; mirrored for pairs) — e.g.
  to centre an inner ear on a flattened ear.
  `lift` (mm, default 0.6) raises it off the parent.
- `attach: "edge"`: a flat disc (two circles joined, or one folded) sewn by a short arc of its edge, standing
  out along `dir`, its face looking along `face` (default front `[0,0,1]`): round mouse ears, flat wings.
  Rendered with a back layer in the part's main colour (zones show on the front only).
- `sink_dir`: which way `sink` presses the piece in: the parent's surface normal (default), `"axis"` - back along
  the piece's own axis (a head on sloping shoulders sinks down into the body instead of sliding off backwards: use it
  against the "two stacked balls" look), or a world vector (the direction the piece moves).
- `rim_arc: [from, to]` (degrees on the piece's last round, az as everywhere): an open piece sewn by that part of its
  rim only, the rest of the edge free - a pelican's pouch under the beak, a shell opening, a pocket. The pattern
  says how many stitches to sew and which side.
- `avoid: true`: a limb swings outward about its seam until its free half stops passing through other pieces
  (`contact` = allowed press-in as a fraction of its radius, default 0.3). The flag says how far it turned.
  `avoid_toward` (world vector) swings it that way instead of outward (front legs in front of a round belly:
  `[0, 0, 1]`); `max_turn` (deg, default 90) limits the swing.
- `front`: override the piece's front reference (world vector). `roll`: spin the piece about its axis (deg).
- `seam: "flat"`: the opening is pressed flat and sewn as a line — what designers do with most arms, legs of
  sitting toys, ears and tails (arms then hang along the body instead of sticking out as stubs). `seam_line`
  (world vector) is the direction of that line (default: front-to-back for limbs hanging down or out to the side,
  side-to-side for pieces pointing up/forward; precisely: along the parent's surface, across the direction the piece
  leans); `pinch` 0..1 (default 0.8) how flat; `pinch_rounds` over how many rounds (default 2.4 for a stuffed
  piece — stuffed to just below the seam, a soft shoulder instead of a flared box — and 1.3 for an unstuffed one:
  a long pinch turns wings into flat patches).
- `name`: instance name (default the part id) for entries using the same part twice; a mirrored entry with a name
  makes `<name>` and `<name>_R` (`name_r` overrides the right one). Children use the entry name (or the part id) in
  `on`; a mirrored child of a mirrored parent goes on the matching side.
- Less common: `axis` / `pos` (root only: its axis and position), `dir_mode: "normal"` (dir = the parent's surface
  normal), `cup` (mm, edge discs: the rim comes forward), `front` (see `bend`).
- The front reference of a piece is the toy's front, turning smoothly into "up" as the piece points forward or
  back; az 0 is that reference. On the right copy of a pair every az is the mirror image of the left copy's.
- `conform` (default true for open pieces) / `conform_span` (rounds blended onto the parent, 1.6) /
  `conform_parent` (an open parent rim follows a child pressed onto it, e.g. a body's neck onto the head).

### No-sew: pieces crocheted into the parent's rounds — toy key `construction`, entry key `join`

`"construction": "nosew"` on the toy crochets every open ear, arm, leg, tail, wing, paw, fin, horn... into its
parent instead of sewing it on (`amikit/nosew.py`; by the part id's words, see `nosew.JOIN_WORDS`). Per entry:
`"join": "crochet"` joins that piece, `"join": "sew"` keeps it sewn. A joined piece is open, sewn by its opening
(`attach: end`), on a parent worked in plain rounds that is not one of a mirrored pair, at a round (not a pole);
muzzles, appliques, worn pieces, collars, closed pieces and edge-sewn discs stay sewn (the build says why).
- The piece is made first; its opening (c stitches) is laid in two layers, and in parent round j (the round
  nearest `at.round`) ⌈c/2⌉ consecutive stitches are worked through both layers of the edge and the parent's
  stitch together — one new stitch per pair; increases / decreases of the round stay where they are (a run may
  begin or end inside an increase, and may pass the round-start marker into the first stitches of round j+1).
- The run is centred nearest the entry's `az`; `at` becomes that run's centre (whole stitches), the right copy of a
  pair is the mirror image of the left copy's run (within half a stitch: the spiral drift) and its built
  direction mirrored. The seam is a fully pinched flat seam along the round (`seam_line` = the round there), the
  piece hinges about it (its axis square to the round), and the pressed edge is laid on the run end to end.
- The pattern lists the piece before its parent, says "leave it open, cut the yarn short", and in round j of the
  parent: after how many stitches, which of the round's own ops go through the edge (`<!-- join ... -->`); the
  assembly step says it is already crocheted in (`<!-- join crochet -->`; `reverse.py` reads it back).
- `python -m amikit.nosew <spec> <out>` builds the toy sewn and no-sew: `nosew.json` (stitch checks, the edge's
  distance from its stitches, length and centre errors, change from the sewn design, the pattern round trip),
  `compare.png` (sewn above, no-sew below).

### Pieces that are not worked in rounds

- **Chain cord** (`ch N`, sl st back): `{"id": "tail", "cord": {"length_mm": 60, "thickness_mm": 2.6,
  "curl_deg": 140, "slip_back": true, "link_mm": 3, "toward": [0, 1, 0]}, "colour": "pink"}` (`slip_back` only
  changes the written pattern); on the assembly entry `dir` = where it leaves the
  body, `toward` = which way it curls. Mouse tails, whiskers-free strings, hanging loops.
- **Flat rows** (ch W+1, W sc, turn; R rows): `{"id": "bow", "flat": {"width_st": 8, "rows": 5}, "colour":
  "red", "pinch": 0.85}`; placed on the parent at (round, az) facing outward, `lift` mm off it, `roll` turns
  it, `conform: true` lays it along the curve (scarf, bib, blanket) without stretching: every stitch keeps its
  distance from the centre along the surface, so a long strip wraps round as far as its length reaches (since
  2026-10-10; before, the flat rectangle was projected onto the surface and came out shorter than written);
  `pinch` gathers the middle (a bow).

### Shaped flat pieces (motifs) — opt-in part key `motif`

One layer of fabric worked flat (turned rows or rounds); the outline is made by the written rows (row length = its
stitches x stitch width, row height = sc 1 / hdc 1.5 / dc 2 / tr 2.6 x the round height, stitches worked along a
chain stand perpendicular to it, increases lengthen the outer edge). Written into the pattern row by row («Плоский
мотив — …»), read back by `reverse.py` (amikit/motif.py).
```json
{"id": "kerchief", "motif": {"kind": "triangle", "width": 26}, "colour": "sage"}         // ch 27, sc rows, dec both ends
{"id": "wing", "motif": {"kind": "triangle", "width": 8, "ends": "one", "sew": "side"}}  // right triangle, sewn by its straight side
{"id": "leaf", "motif": {"kind": "leaf", "chain": 7, "tall": "dc", "tip": "picot"}}      // pointed leaf / petal / scale
{"id": "cell", "motif": {"kind": "hexagon", "rounds": 3, "point_up": true}}              // sc rounds, 6 incs stacked on corners
{"id": "disc", "motif": {"kind": "circle", "rounds": 4, "edging": {"kind": "shell", "k": 5, "every": 4, "colour": "blue"}}}
{"id": "bag", "motif": {"kind": "half_circle", "rows": 5}}                                // D-shape: MR, turned rows, +3 per row
{"id": "moon", "motif": {"kind": "crescent", "chain": 12, "tall": "tr"}}                 // one row along a chain, incs in the middle
{"id": "heart", "motif": {"kind": "heart"}}                                               // into a MR: ch 3, 3 tr, 3 dc, ch 1, tr, ch 1, ...
{"id": "mask", "motif": {"kind": "mask", "chain": 15, "rounds": 3, "tall": "tr"}}        // chain oval + a lobe round: two lobes
{"id": "heart", "motif": {"kind": "heart", "rounds": 3}}                                  // + 3 sc rounds (2 incs per lobe, 3 sc in the point): ~2x
{"id": "star", "motif": {"kind": "star", "points": 5, "length": 5}}                       // MR 10 sc; (ch 5, sc/hdc/dc back along it, skip 1, sl st) x5
{"id": "clover", "motif": {"kind": "lobed", "lobes": 3, "tall": "dc", "stitches": 5}}    // MR: (ch 3, 5 dc, ch 3, sl st) x3, pull tight
{"id": "pad", "motif": {"kind": "lobed", "lobes": 5, "tall": "tr", "rounds": 2}}         // + sc rounds around the lobes: a bigger lobed disc
{"id": "petal", "motif": {"kind": "petal", "chain": 6, "tall": "dc"}}                     // graded up one side, 6 dc fan at the tip, back down
{"id": "wing", "motif": {"kind": "half_circle", "rows": 5, "edging": {"kind": "point", "every": 5, "length": 5, "where": "curve"}}}  // bat wing
```
Other keys: triangle `stitch` (sc/hdc/dc), `every` (decrease every k rows: taller); leaf `tall` (hdc/dc/tr);
heart `rounds` (0..8 sc rounds into the stitches only, the posts skipped: the notch stays); star `points` (3..8),
`length` (chain per point, >= 3; the stitches back along it are graded sc, hdc, dc... up to `tall`, default dc);
lobed `lobes` (3..6), `tall` (hdc/dc/tr; the posts are ch 2/3/4), `stitches` (tall stitches per lobe, 3..9),
`rounds` (sc rounds: sc in the chain spaces, sc/inc over the lobe, the sl st skipped; then a dec after each dent and 3
incs per lobe — the lobes soften as it grows); petal `chain` (>= 4), `tall` (sc/hdc/dc/tr; fan of 3/4/6/7 in the
last chain: a rounded tip, not a point);
`edging` (circle, hexagon, half_circle, mask): `{"kind": "shell", "k": 5, "every": 4, "st": "dc"}` (skip, k st in one
st, skip, sl st per `every` stitches: scallops) or `{"kind": "picot", "every": 3}` or `{"kind": "point", "every": 4,
"length": 5, "st": "dc"}` (per `every` stitches one chain point worked back with graded stitches, skip 1, the rest sl
st: bat-wing spikes) or `{"kind": "feather", "every": 5, "st": "tr"}` (sc, hdc, ... rising to 2 `st` in one st, sl
st: a saw edge of feather tips); half_circle edging `"where": "curve"` says in the pattern that the edging row goes
along the curved edge only (it always does: the straight edge is the row ends); `colours`: whole-row zones only
(`{"rounds": [a, b], "colour": c}`; other zone kinds are flagged and ignored); `sew`: which edge is sewn for
`attach: "edge"` (triangle `base` = foundation chain | `side` = the straight row ends; leaf `base` = the 3-sc end;
petal `base` = the narrow start; half_circle `base` = the straight edge; others the lowest point).
Placement: `applique: true` (lies on the parent; flat x = the piece's up, `roll` turns it), `attach: "edge"` (stands
out along `dir`, its face along `face`; rendered as one fabric layer with a rounded edge; the right copy of a pair is
the mirror image — a one-sided triangle has its decreases on the other edge), or applique + `drape: true` (laid over
the parent like cloth: lengths along the rows are kept on the curved surface — a kerchief, `roll: 180` = tip down).

### Chain-ring pieces (collars, snoods) — round steps `ring`, `rib`, `shells`; entry key `around`

- `"ring 42"` (first step): ch 42 joined with a sl st, R1 = 42 sc, one in each chain; the piece is worked outward
  from a circle (not a point). Counts decide the shape as for any piece: +6 per round lies flat (a collar), fewer
  increases make a cone, none a tube, more a ruffle.
- `"rib 40 x3"`: rounds of (fpdc, bpdc) ribbing (two sc rounds tall, vertical ribs in the relief).
- `"shells 7 every 6"` (also `"shells 5 hdc every 4"`): an edging round of shells — scallops on the edge (works on
  any piece worked in rounds).
- part `"split": true` (with `ring`): worked in turned rows, not joined: each row covers only as much of the circle
  as its stitches reach, so too few increases open a V notch at the front.
- entry `{"part": "collar", "on": "body", "at": {"round": "end"}, "around": true}`: the ring sits around the
  parent's round, the rounds grow along `dir` (default down along the parent; `[0, 1, 0]` = a standing snood); the
  fabric lies on the parent and on pieces it meets; a round longer than the circle it lies on ruffles. `fall: true`:
  the fabric hangs from the ring along the parent's surface (a soft cape). `shift` (mm) moves the ring along the
  parent, `roll`/`front` turn the round start.

### Worn pieces (hair caps, hoods, hats, helmets, shells) — entry key `wear`, part key `opening`

- entry `{"part": "wig", "on": "head", "at": {"round": "end"}, "wear": true, "dir": [0, -1, 0], "pad": 3}`: an open
  piece worked from a magic ring at its top is put ON the parent like a hat. Its magic ring is centred over `at`; it
  is pulled down along its own axis (`dir`: from its start toward its opening, default into the parent) until its
  inside rests on the parent with `pad` mm (default 1) of stuffing between; then it keeps the shape its rounds make
  (it is not laid onto the parent like a soft flap). A cap wider than the head stands off the head's sides: the
  volume of a bob, a puffy beanie, a hood. It rests where it first touches (the crown, or the shoulders of a body
  that widens below the head). The pattern says: put it on like a hat, the stuffing on top and where it rests, which
  of its rounds to sew on (where it rests), and how long the loose tacking stitches at the lower edge are.
- part `"opening": {"from": k, "stitches": g}`: after round k the piece is worked in turned rows that leave the g
  stitches at the centre front (around the round-start marker, az 0) unworked — the face opening of a hood or a bob
  (bangs = round k across the forehead, the sides hang down), the neck opening of a cape. The `rounds` keep full
  counts (the circle the rows go round); the written rows have count - g stitches. Only sc may sit in the window
  (an increase there is flagged). Mesh, samples and the materials leave the window out.
- Bob wig recipe (seal in a wig, 2026-10-10): head 36 st -> wig `mr 6, inc 6 to 24, inc 6 to 42 every 2, 48,
  even 8`, `opening` from 11 with 16 st (eyes at az +-32 stay clear), `wear` + `pad` 3: a round dome with bangs over
  the eyes and sides to the jaw, 7 mm off the head.

### Appliques anchored to the eyes — entry key `anchor`

`"anchor": "eyes"` centres an applique between the eyes on its parent (a face mask), `"anchor": "eye"` on the eye of
its own side (mirrored patches). `offset_mm` then walks it along the surface (snapped to half rounds / 0.1 st, the
position the pattern states). The piece follows the eyes when they move (cute.py eye moves, edits). Colour-zone
masks stay where they are drawn — use a `mask` motif (or any applique) with `anchor` for a face mask that must
follow the eyes.

### Can the face be seen? (`visible.py`)

Every build measures how much of each face feature is visible from the front relative to the piece it is sewn on
(eyes one by one, the nose, the mouth embroidery, a muzzle / face piece sewn on the front of the head) and writes
`visibility.json`; below eye 60 %, nose 50 %, mouth 40 %, muzzle 20 % it flags "face: ... only N % visible": a muzzle
pressed into the head, eyes under ears or cheeks. The skill's quality check reports `face_visible` / `face_ok`.
`python -m amikit.visible <spec> [--fast]`.

### Ground

The toy stands on its lowest point. `flags.txt` says when something other than the body holds it up
("the toy rests on tail; the body is 9 mm above the floor"): a sitting toy should rest on its body and legs,
so turn a low tail up/back (`dir`) or shorten the legs.

## What the written pattern shows (and what it keeps for the engine)

The pattern is for a person: positions in rounds and whole / half stitches, sizes in cm, depths only from 3 mm,
counts, colours, shop sizes of safety eyes. The exact engine values (angles, mm, %, the tenth of a stitch, the steps
of grouped copies) stay in HTML comments next to the text (`<!-- Точно: ... -->`, `<!-- st 6.9 left -->`,
`<!-- step: ... -->`); the site and the PDF drop comments and `reverse.py` reads them, so the round trip still
rebuilds the identical toy. Five or more consecutive copies of one piece on one parent (bobbles, spots, spikes) are
one written step ("K pieces, rounds a-b, pin them against the picture"). Safety eyes go in after the round that
holds them. The engine's check and flags are a comment at the end (a real arithmetic error is still shown).

## face

```json
{"type": "safety_eye", "on": "head", "at": {"round": 13.5, "az": 26}, "size_mm": 8, "mirror": true,
 "aspect": 1.0, "sink": 0.45, "iris": "green", "pupil": 0.55}      // iris/pupil optional (cat eyes)
{"type": "safety_nose", "on": "muzzle", "at": {"round": 1.6, "az": 0}, "size_mm": [9, 6.5], "shape": "oval"|"triangle"}
{"type": "stitch", "on": "muzzle", "points": [[0.9, 180], [2.8, 180]], "colour": "black", "thickness_mm": 1.2}
{"type": "satin", "on": "muzzle", "at": {"round": 1.5, "az": 0}, "width_mm": 8, "height_mm": 5,
 "shape": "triangle"|"oval"|"heart", "thickness_mm": 1.4, "rotate": 0}   // horizontal strands, toy's up = up
{"type": "strap", "on": "body", "points": [[16, 105], [12, 78], [3, 52]], "colour": "green", "thickness_mm": 3.4,
 "mirror": true}      // a crocheted chain sewn along the surface: backpack straps, headbands, collars, belts
{"type": "blush", "on": "head", "at": {"round": 16, "az": 40}, "size_mm": 9, "colour": "pink", "mirror": true}
{"type": "felt", "on": "head", "at": {"round": 13, "az": 25}, "size_mm": [12, 9], "rotate": -20, "colour": "black"}
```
{"type": "fringe", "on": "head", "points": [[2, 0], [8, 0], [14, 0]], "count": 14, "length_mm": 22,
 "colours": ["pink", "yellow", "mint"], "droop": 0.7, "curl": 0, "spread": 0.25}
     // yarn strands knotted along a line (lark's head, both ends hang): manes, cow fringe, hair; curl > 0 curls them
`points` are `[round, az]` on the piece (embroidery follows the surface, taking the short way round between
points). Mirrored items flip az (and `rotate`). Also read: eye `colour` (default black), satin `strands`
(default from the height), legacy satin `width_st` / `height_rounds`, fringe `seed` / `doubled` / `spread`,
type aliases `eye` (= safety_eye) and `embroidery` / `line` (= stitch). Undefined colour names are flagged. `blush` takes `strength` (0..1, default 0.55); `felt` takes `rotate` (deg).
A `safety_nose` with `shape: "triangle"` is a flat-topped rounded triangle pointing down.

## The designers' library (use it for every part)

~22,000 designer parts (learn-only statistics, `amikit/library.py`) tell what a part of a given animal normally
looks like relative to its head: width, length, how many rounds at full width (volume), the opening, and the
construction kind (ears: long / floppy / round / pointed; arms and legs: paw / tube / long_tube).
- Every build writes `library_audit.txt`: each part vs designers of that animal ("!!" = outside the 15-85%
  range) with a consensus suggestion.
- `python -m amikit.library stats ear bunny` (percentiles + kinds), `python -m amikit.library suggest arm bear 48 paw`
  (our own rounds synthesised from the median shape of hundreds of designers' parts; never a copy of one).
- Designers' arms and legs are mostly `paw` kind: widest at the start (the hand/foot), narrowing toward the seam,
  then a long narrower part; ears are short cups for bears/mice, slow cones from mr 4 for cats/foxes, long tubes
  with a widest point a quarter of the way for bunnies. A part that is a plain cylinder of equal rounds is
  usually wrong.
- The library is advice: the picture decides. If the picture's part is clearly longer/shorter, follow the
  picture, but keep the designer's construction kind.

## Making it look like the picture and cute (rules that held so far)

- Compare the four views with the concept every iteration; fix the biggest visual difference first
  (silhouette and proportions, then the placement of ears / muzzle / eyes, then colour, then details).
- Chibi proportions: head 1.3-1.5x the body's width; eyes on or just below the head's middle
  (~0.55-0.62 of the head's rounds from the crown), wide apart (6-8 st between them on a 42-48 st head), small
  (7-9 mm on a 48-st head); the muzzle low on the face; short limbs.
- Construction decides the look (from designers' patterns): bear ears = a `dish` stood on its rim (or the
  `ear` preset, kind round) with an applique inner ear at `{"round": "start"}` — a `flatten` ~0.85 plate reads as
  paper; cat/fox ears = a `spoon` cone; bunny ears = long open tubes, unstuffed, flattened; muzzles = open, lightly stuffed, sewn
  by the opening, often a chain oval; sitting legs = open tubes sewn on the front of the body pointing forward
  (the magic ring is the sole); arms = open tubes sewn just under the head, `avoid: true`; the body worked from
  the bottom, open at the neck, the head pressed onto it (`sink`).
- Round heads: stuffing flattens a ball made with +6 every round and a long run of equal rounds (a "box" or
  a "bun"). Designers grow the last increases every other round and shrink the first decreases the same way:
  `["mr 6", "inc 6 to 42", "inc 6 to 60 every 2", "even 5", "dec 6 to 42 every 2", "dec 6 to 6"]` (height/width
  0.83 vs 0.74 for `inc 6 to 60, even 7`).
- Measure it: every build writes `cute.txt` (amikit/cute.py: baby-schema features measured on the built geometry,
  score, out-of-range features and edit ops). `python -m amikit.cute <spec> [--ref concept.png] [--out DIR]
  [--pick 1,3]` adds the predicted effect of each op (applied, rebuilt, re-measured); `python -m amikit.cute_llm`
  lets Claude choose among them with the renders in view. The concept wins over the generic targets.
- Rounds are the only way to change a shape: more even rounds = longer, a bigger max count = wider, `ch N` for
  oval pieces. Don't fake shapes the pattern cannot make.

## Customising a toy (a user's own pet)

`amikit/custom.py` ops speak the owner's words and write everything above for you; they go through
`edit.apply` like any op, so a text command (`llm_edit.py`), a pet photo (`pet_llm.py`) or the editor can use them:
```json
{"op": "colour", "target": "all", "colour": "#b8742f"}                 // clean coat (markings removed)
{"op": "spot", "near": "eye_L", "place": "under", "colour": "#3a2a20"}   // a spot under the toy's LEFT eye
{"op": "spot", "on": "body", "where": "back", "colour": "black", "size_mm": [26, 20]}
{"op": "marking", "kind": "blaze" | "mask" | "eye_patch" | "eyebrows" | "tan_points" | "socks" | "tail_tip" |
                  "ear_tips" | "chest" | "belly" | "saddle" | "spots" | "stripes" | "muzzle" | "inner_ears" | "half_face" |
                  "collar", "colour": "white", "amount": 0.5, "side": "L" | "R" | "both"}
{"op": "ears", "kind": "pointed" | "round" | "floppy" | "long" | "folded", "size": 1.2, "inner": "pink"}
{"op": "tail", "kind": "curled" | "straight" | "up" | "stub" | "long"}
{"op": "shape", "target": "legs", "length": 0.7}                         // size / length / width factors
{"op": "eyes", "colour": "amber"}  {"op": "nose", "colour": "pink"}
```
Bases: `toys/pets/dog.json`, `toys/pets/cat.json`. Photo -> toy: `python -m amikit.pet_llm photo.jpg out
--base dog [--notes "..."]` (Claude vision, needs ANTHROPIC_API_KEY) or `--ops ops.json` (an agent wrote the ops).
Left / right are the toy's (= the pet's own): its left eye is on the viewer's right in a front photo.

