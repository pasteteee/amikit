# amikit: amigurumi from a spec

amikit is an engine for designing crochet amigurumi toys. You describe a toy as one JSON spec: every piece as
crochet rounds (`mr 6`, `inc 6 to 48`, `even 8`, ...), how the pieces are sewn together, and the face. amikit
turns that spec into:

- **a 3D toy that looks crocheted.** Every stitch is real geometry, and stuffing, seams, ears, hats and hair
  follow the stitch counts;
- **renders** (front, 3/4, side, back, a contact sheet against a reference picture, transparent previews);
- **a GLB** for web viewers;
- **the written pattern** in Russian and English (`PATTERN_RU.md`, `PATTERN_EN.md`, `pattern.json`), with the stitch
  arithmetic checked round by round;
- **quality signals**: flags (floating or hidden pieces, what the toy rests on), face visibility, a cuteness
  measure with ranked suggestions, a silhouette fit against a concept picture, and an audit against ~39,000
  designers' parts.

It is deterministic and fast: a build takes 3-5 s. The 3D view is always exactly "what this pattern makes".
Nothing is painted on or faked, so a person can crochet the toy from the pattern and get what the picture shows.

The engine is built to be driven by **AI models** (from a concept picture, a pet photo or a text request), by an
**editor** (`server.py` + `editor/`) and by people editing JSON. Models change a toy only through the spec.

> **AI models and agents designing toys:** read [`skill/SKILL.md`](skill/SKILL.md) first. It is the designer's
> workflow that produced the approved collections (process, quality bar, review). It is backed by
> [`skill/recipes.md`](skill/recipes.md), a tested construction for each kind of part and toy, and
> [`skill/review.md`](skill/review.md), how to compare a model with its concept and judge it.
> The spec format itself is [`SPEC.md`](SPEC.md).

## Quick start

```bash
pip install -e .                       # from this folder (Python >= 3.12; numpy, scipy, trimesh, moderngl, pillow)
python -m amikit.build toys/teddy.json out/teddy               # -> out/teddy/: renders, sheet, pattern, flags
python -m amikit.build toys/teddy.json out/teddy --ref concept.png --fast   # sheet with the concept first
python -m amikit.cute toys/teddy.json --ref concept.png --out out/teddy_cute --closeups
python -m amikit.fitview concept.png out/teddy/view_front.png out/teddy/fit.png   # silhouette IoU + overlay
python -m amikit.visible toys/teddy.json                       # can the eyes / nose / mouth be seen?
python -m amikit.preview toys/teddy.json out/teddy_preview     # transparent site previews
python -m amikit.server                                        # browser editor at http://127.0.0.1:8765
```

A build folder holds `view_front/34/side/back.png`, `sheet.png` (the reference first), `flags.txt`,
`library_audit.txt`, `cute.txt`, `visibility.json`, `PATTERN_RU.md`, `PATTERN_EN.md`, `pattern.json` (and
`model.glb` when exported). Rendering needs OpenGL: a desktop GPU, or headless Mesa (EGL / llvmpipe) on a server.

Good specs to start from are in `toys/`. That folder is in the lab's working copy only and is not in git (see
below). Owner-approved collections are the best examples: `seo100` (100 toys), `showcase10`, `catsdogs`,
`diverse8` and `diverse8b`. `pets10` and `simple10` passed blind judges. Single classics are `teddy`, `bunny`,
`panda`, `fox` and `penguin`. `skill/recipes.md` says which spec to copy for which kind of toy. Site packages carry
their `spec.json` too.

## How it works

```
            text request ──► llm_edit.py ──┐
  picture ─► (agent writes spec)           ├─► edit.py ops ─► spec.json ─┐
            editor UI ─► server.py ────────┘   (validated)               │
                                                                         ▼
 presets.py  opt-in construction presets (ear kinds, cute_head) expanded into plain rounds before anything else
 rounds.py   steps "mr 6", "inc 6 to 42", "even 7" → stitch counts + per-round ops (staggered like designers)
 profile.py  counts × gauge → stuffed profile (1D pressure membrane, yarn lock-up), ~0.3 s per piece
 piece.py    profile → local surface (stadium for chain ovals; flatten / spoon / dish / squash / bend / drape / coil),
             stitch V relief
 motif.py    shaped flat pieces (triangle, leaf, hexagon, circle, half circle, crescent, heart, star, lobed, petal,
             two-lobed mask; shell / picot / point / feather edgings): written rows -> geometry, pattern rows
 toy.py      placement: seam on the parent at (round, az), dir, mirror pairs, open rims follow the parent,
             appliques, worn pieces (hats, hair, hoods), collars round a neck, limb avoidance; face (safety eyes with
             dimples, noses, embroidery, blush, felt, fringe); colour charts per stitch; mesh with real stitch geometry
 render.py   moderngl: AO baked from 48-128 directions, soft key shadow, yarn sheen, glossy eyes, 4 views, every
             stitch's V relief drawn per pixel; transparent=True renders RGBA (floor = soft contact shadow only)
 export.py   GLB (vertex colours with AO and stitch grooves)
 pattern_text.py   PATTERN_RU.md / PATTERN_EN.md / pattern.json from the same data, arithmetic checked; states every
                   placement exactly (angles in degrees, depths in mm, flattening in %, stitch offsets to 0.1 st)
 build.py    CLI: spec → renders + sheet against a reference + flags + pattern + cute.txt + visibility
 preview.py  CLI: spec → transparent site previews preview_<view>.png, cropped square
 visible.py  can the face be seen? share of each eye / nose / mouth / muzzle visible from the front
 cute.py     cuteness by numbers: baby-schema features measured on the built toy, score, ranked edit ops with the
             engine's predicted effect, head close-ups; measure_image() for concept pictures
 fitview.py  silhouette of a render vs a reference picture: IoU, width profiles, red/blue overlay
 library.py  designers' parts (learn-only statistics): per-part audit in every build, `stats`, `suggest`
 custom.py   customisation in the owner's words (colour / spot / marking / ears / tail / shape / eyes / nose ops):
             spots under an eye, socks, blaze, saddle, ear kinds, curled tails
 edit.py     the one edit interface: set / add / remove / scale_part / recolour + custom ops, validated by rebuilding
 reverse.py  written pattern -> spec -> 3D: our PATTERN_EN.md exactly, designers' patterns in the usual wording
             (Rnd/R, "(sc, inc) x6", "*...* rep", sc2tog, colour changes, chain ovals, RU сбн/пр/уб); reverse_free.py
             places pieces from free-text assembly ("assumed" choices logged)
 roundtrip.py  spec -> PATTERN_EN.md -> reverse -> 3D vs the original, per piece (point-to-surface mm): a difference
               is something the written pattern does not say
 llm_edit.py / cute_llm.py / pet_llm.py   Claude API helpers: text edits, choosing cute suggestions, pet photo -> ops
```

Design choices (and why):
- **Count-derived geometry instead of a 3D fabric simulation.** Every crocheted piece is worked in the round, so its
  shape is fixed by its counts, gauge, stuffing and how it is sewn. The old spring simulator took minutes, turned
  runs of equal rounds into boxy belts and moved pieces unpredictably. Here a piece cannot look different from
  what its rounds make, and a change shows in seconds.
- **Relief is geometry.** Each stitch is two yarn legs in the mesh, with grooves between rounds. Silhouettes and AO
  show the crocheted texture instead of a painted pattern.
- **One edit interface.** `edit.apply(spec, ops)` is used by the editor, by LLMs and by the customisation ops.
  Every edit is validated by rebuilding.
- **Construction stays honest.** Shapes come only from rounds, stuffing, flatten, spoon, dish, squash, bend,
  appliques, motifs, worn pieces and embroidery, all things a crocheter does. What the engine cannot express is
  flagged, not faked.
- **The pattern is the source of truth.** Exact engine values ride along in HTML comments of the written pattern,
  and `reverse.py` reads them back. `roundtrip.py` rebuilds every toy from its own pattern (≤ ~1 mm on ~170 specs).

## Repository layout

| Path | What |
| --- | --- |
| `*.py` | the engine (modules above) |
| `SPEC.md` | the toy spec format: the reference for people and models (read at run time by the site's service) |
| `skill/` | how to design toys with the engine: `SKILL.md` (workflow), `recipes.md` (constructions), `review.md` (judging) |
| `IMAGE_TO_TOY.md` | the short picture → spec loop (superseded in detail by `skill/`) |
| `toys/` | toy specs: single classics, collections (`seo100`, `showcase10`, `pets10`, `simple10`, ...), `_v1`/`_lab` = older versions. **Local only, gitignored**: the designs stay off GitHub |
| `editor/` | the browser editor UI (`python -m amikit.server`) |
| `test_*.py` | engine tests |
| `reverse_tests/` | designers' patterns and photos for `test_reverse`: local only, never committed (third-party) |

Tests, from the folder that contains `amikit/`:

```bash
python -m unittest amikit.test_engine amikit.test_pattern_text amikit.test_reverse amikit.test_cute \
    amikit.test_shapes amikit.test_custom amikit.test_worn
python -m amikit.roundtrip --top50                  # do our patterns rebuild the same toys?
python -m amikit.regress --fast [--compare latest]  # build all toys, compare with an earlier run
```

Run the unit tests and the round trip after any engine change. Any new spec key that changes geometry must also be
written into the pattern and read back by `reverse.py`, or the round trip drifts. The tests and the round trip read
specs from `toys/` (teddy, bunny, `pets/`, `_cute_lab/`, `_shapes_lab/`, `top50/`), so they run only in a working
copy that has the toys. A fresh clone does not.

Not modelled yet: clothes worn over the body (beyond caps, hoods and collars), standing balance, gravity sag of
lightly stuffed parts (`drape` only for unstuffed open pieces), joining two pieces into one, curled-up brims,
fuzzy yarn, limbs holding objects (props are sewn on), calibration against a real crocheted sample.
`library.py` reads the designers' parts from the lab (`../../pattern-harvest/`, not in this repository). Without
that data `library suggest` / `stats` and the build's `library_audit.txt` are unavailable, and everything else
works.

## The repository and using amikit as a library

This folder is the engine's own git repository (since 2026-10-12) and an installable package. The repository root
is the package itself (`pyproject.toml`: `package-dir = {"amikit" = "."}`), so the lab keeps importing it as
`amigurumi-lab/amikit`. Engine work happens here: code, docs, the design skill and tests. The toy specs (`toys/`)
and the designers' test patterns (`reverse_tests/`) stay local and are not committed.

An install carries the code, the docs (`*.md`, `skill/*.md`; `SPEC.md` is read at run time) and the editor UI,
not the toys:

```bash
pip install "amikit @ git+https://github.com/pasteteee/amikit.git@<commit or tag>"   # a pinned version
pip install "amikit[llm] @ git+..."                                                  # + Claude API (anthropic)
pip install -e <path>/amikit                                                         # a working copy, live
```

The site (`amigurumi` repo, `engine/`) pins a commit in `engine/requirements-amikit.txt` for deploys
(`npm run engine -- update` moves it to the newest pushed commit). Locally, `AMIKIT_DIR` in its `engine/.env`
points at a working copy of this repository, so the site's engine service runs the code here (restart the service
after a change). Push engine changes before updating the site's pin.
