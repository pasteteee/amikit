# amikit: amigurumi from a spec (engine architecture)

amikit turns one JSON spec of a crochet toy into its 3D look (stitch relief, renders, GLB) and its written
pattern, deterministically and in a few seconds. A model (or a person in the editor) changes the toy only by editing
the spec; everything else is derived, so the 3D view is always "what this pattern makes".

```
            text request ──► llm_edit.py ──┐
  picture ─► (agent writes spec)           ├─► edit.py ops ─► spec.json ─┐
            editor UI ─► server.py ────────┘   (validated)               │
                                                                         ▼
 rounds.py   steps "mr 6", "inc 6 to 42", "even 7" → stitch counts + per-round ops (staggered like designers)
 profile.py  counts × gauge → stuffed profile (1D pressure membrane, yarn lock-up), ~0.3 s per piece
 piece.py    profile → local surface (stadium for chain ovals; flatten / spoon / dish / squash / bend), stitch V relief
 toy.py      placement: seam on the parent at (round, az), dir, mirror pairs, open rims follow the parent,
             appliques, limb avoidance; face (safety eyes with dimples, noses, embroidery, blush, felt);
             colour charts per stitch; mesh with real stitch geometry
 render.py   moderngl: AO baked from 48-128 directions, soft key shadow, yarn sheen, glossy eyes, 4 views, every stitch's
             V relief drawn per pixel (bump from the stitch coordinates, render.STITCH_GLSL = piece.stitch_height); transparent=True
             renders RGBA (clear backdrop, floor = soft contact shadow only)
 preview.py  CLI: spec → transparent site previews preview_<view>.png, cropped square (render_previews)
 export.py   GLB (vertex colours with AO and stitch grooves)
 pattern_text.py   PATTERN_RU.md / PATTERN_EN.md / pattern.json from the same data, arithmetic checked; states every
                   placement exactly (angles in degrees, depths in mm, flattening in %, stitch offsets to 0.1 st)
 reverse.py  written pattern -> spec -> 3D (reverse generation): our PATTERN_EN.md exactly, designers' patterns in
             the usual wording (Rnd/R/bare "12:", "(sc, inc) x6", "*...* rep", "2 sc in next st", "sc2tog", colour
             changes, chain ovals, RU сбн/пр/уб); reverse_free.py places pieces from free-text assembly and fills
             what the text leaves open with designers' defaults, logged as "assumed" in reverse_report.txt
 roundtrip.py  spec -> PATTERN_EN.md -> reverse -> 3D vs the original, per piece (point-to-surface mm); a difference =
               something the written pattern does not say
 visible.py  can the face be seen? share of each eye / nose / mouth / muzzle visible from the front (flags + JSON)
 build.py    CLI: spec → renders + sheet against a reference + flags + pattern + cute.txt
 presets.py  opt-in construction presets (ear kinds, cute_head) expanded into plain rounds before anything else
 motif.py    shaped flat pieces (triangle, leaf, hexagon, circle, half circle, crescent, heart, two-lobed mask, shell /
             picot edgings): written rows -> geometry, pattern rows, reading them back
 cute.py     cuteness by numbers: baby-schema features measured on the built toy, score, ranked edit ops with the
             engine's predicted effect, head close-ups; measure_image() for concept pictures
 cute_llm.py Claude (vision) chooses among cute.py's suggestions with the renders in view (no-op without a key)
 custom.py   customisation in the owner's words (colour / spot / marking / ears / tail / shape / eyes / nose ops
             inside edit.apply): spots under an eye, socks, blaze, saddle, ear kinds, curled tails; world-placed
             `patch` zones that span pieces, ear-shaped `inner` zones, satin / felt spots
 pet_llm.py  pet photo -> Claude (vision) -> customisation ops -> spec -> build (or --ops from an agent)
```

Design choices (and why):
- **Count-derived geometry instead of a 3D fabric simulation.** Every crocheted piece is worked in the round, so its
  shape is fixed by its counts, gauge, stuffing and how it is sewn. The old 3D spring simulator (snapfit/sim2.py)
  took minutes, turned runs of equal rounds into boxy belts and moved pieces unpredictably; here a piece cannot
  look different from what its rounds make, and a change shows in seconds.
- **Relief is geometry.** Each stitch is two yarn legs in the mesh, grooves between rounds; silhouettes and AO
  show the knitted texture instead of a painted pattern.
- **One edit interface.** `edit.apply(spec, ops)` (set / add / remove / scale_part / recolour, paths by id) is used by
  the editor and by the LLM; every edit is validated by rebuilding.
- **Construction stays honest.** Shapes come only from rounds, stuffing, flatten, spoon, dish, squash, bend, appliques — things a
  crocheter does. What the engine can't express is flagged, not faked.

Also in the package:
- `library.py`  ~22,000 designer parts (learn-only statistics): per-part audit in every build
  (`library_audit.txt`), `stats` / `suggest` (our own rounds from the median shape of hundreds of parts).
- `server.py` + `editor/`  browser editor (3D, sliders, JSON, undo, text commands via `llm_edit.py` + Claude API).
- `test_engine.py` (65 tests, ~70 s), `test_pattern_text.py`, `test_cute.py` (ears, presets, cute measure, pale masks), `regress.py` (build all toys, report, `--compare`
  with an earlier run), `calibrate.py` (renderer colour transfer curve).

```bash
../.venv/Scripts/python -m amikit.reverse pattern.txt ../out/amikit/_reverse/x --ref photo.jpg   # pattern -> 3D
../.venv/Scripts/python -m amikit.roundtrip [--top50]                     # do our patterns rebuild the same toy?
# from amigurumi-lab/
../.venv/Scripts/python -m unittest amikit.test_engine amikit.test_pattern_text amikit.test_reverse amikit.test_cute amikit.test_shapes amikit.test_custom
../.venv/Scripts/python -m amikit.cute amikit/toys/teddy.json --out ../out/amikit/teddy_cute --closeups
../.venv/Scripts/python -m amikit.regress --fast                      # -> ../out/amikit/_regress_runs/<ts>/
../.venv/Scripts/python -m amikit.regress --fast --compare latest      # after an engine change
../.venv/Scripts/python -m amikit.server                               # editor at http://127.0.0.1:8765
```

Shaped flat pieces, shell/picot edgings, chain-ring collars/snoods (ruffles from counts) and eye-anchored masks are
modelled (SPEC.md "Shaped flat pieces", run-notes/2026-10-07-shaped-flat-pieces.md); `test_shapes.py` covers them.
Not yet modelled: clothes worn over the body, standing balance, gravity sag of unstuffed parts (the author sets
directions; `fall` only for collars), bobbles / popcorn stitches, fuzzy yarn, limbs holding objects, calibration against a real
crocheted sample. Text edits through the Claude API need `ANTHROPIC_API_KEY` and have not been run here.

## The repository and using amikit as a library

This folder is the engine's own git repository (since 2026-10-12) and an installable package: the repository root
is the package itself (`pyproject.toml`: `package-dir = {"amikit" = "."}`), so the lab keeps importing it as
`amigurumi-lab/amikit`. Engine work happens here: code, toy specs (`toys/`) and the tests; designers' patterns for
`test_reverse` (`reverse_tests/`) are local only (third-party, not committed).

An install carries the code, the docs (`SPEC.md` is read at run time) and the editor UI, not the toys:

```bash
pip install "amikit @ git+https://github.com/<owner>/amikit.git@<commit or tag>"   # a pinned version
pip install "amikit[llm] @ git+..."                                                # + Claude API (anthropic)
pip install -e <path>/amigurumi-lab/amikit                                         # this working copy, live
```

The site (`amigurumi` repo, `engine/`) pins a commit in `engine/requirements-amikit.txt` for deploys
(`npm run engine -- update` moves it to the newest pushed commit); locally `AMIKIT_DIR` in its `engine/.env` points at
this folder, so the site's engine service runs the code here (restart the service after a change). Push engine
changes before updating the site's pin.
