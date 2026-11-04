# amikit video — crochet tutorial engine for Shorts

Generates 9:16 (1080×1920, 30 fps) step-by-step crochet tutorials straight from a
pattern part (`"mr 6", "inc 6 to 18", "even 1", "dec 6 to 6"` — same round notation as
amikit specs). Runs on CPU only: the simulation is plain JS, frames are rendered with
three.js in headless Chromium (SwiftShader) and encoded with ffmpeg.

```bash
npm i                                   # three.js
node make_scn.mjs /tmp/ear.json         # scenario for the bear-ear sample
./make_video.sh /tmp/ear.json ear.mp4 3 # 3 parallel render workers
node debug_fabric.cjs /tmp/f.jpg '[[0,0,55,39]]'   # look at the relaxed fabric only
```

## How the physics works

* **Yarn is one continuous, inextensible strand** from the free tail, through the magic
  ring and every stitch, onto the hook and off-screen to the left index finger
  (`G`). Every frame the strand is rebuilt and its *material coordinate* (arc length
  from a fixed point) drives the ply texture, so the yarn visibly feeds in from the
  finger and slides through loops when they tighten. Nothing is created out of thin
  air: a new loop is born at zero size on the straight working yarn
  (`presence` blending) and only grows as the hook pulls it.
* **Taut-string contacts.** Yarn under tension is a shortest path between contacts:
  loops on the hook are exact taut wraps around the shaft / throat groove
  (`curve.mjs: wrapArc`), plus contacts at the fabric edge and through the stitch hole.
* **The hook is a solid** (`hook.mjs`): the yarn centreline is pushed out of the shaft,
  head and throat groove every frame, so stitches stretch around the hook when it is
  inserted.
* **Loops leave the hook only by sliding off the tip** while the fabric holds them
  (`through2` phase), then tighten into their relaxed stitch shape.
* **Free tail** is a PBD rope with gravity that stays behind the fabric; pulling it
  closes the magic ring (ring yarn flows into the tail, length conserved).
* **Hand-like motion:** minimum-jerk easing, the hook pivots in the hole for a yarn
  over, the work is rotated so the next stitch comes to the hook (as a crocheter does).

## Files

| file | role |
| --- | --- |
| `engine/fabric.mjs` | pattern expansion, spiral surface, stitch frames, relaxed stitch paths |
| `engine/sim.mjs` | stitch phases (insert, yarn over, pull up, pull through…), strand assembly |
| `engine/hook.mjs` | hook geometry (shaft, throat, lip, thumb rest) + collisions |
| `engine/scenario.mjs` | tutorial script: captions (RU), speeds, camera shots |
| `engine/director.mjs` | camera, caption & badge tracks |
| `engine/tube.mjs`, `web/render.mjs` | yarn tube mesh with twisted-ply material, renderer |
| `render.cjs`, `make_video.sh` | headless rendering & encoding |
