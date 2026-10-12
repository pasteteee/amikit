"""Command line: toy spec -> renders (+ reference sheet), pattern, GLB.

    python -m amikit.build <spec.json> <out_dir> [--ref image.png] [--fast]
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from .toy import Toy
from .render import Renderer, sheet


def build(spec_path, out, ref=None, fast=False, views=("front", "34", "side", "back"), ref_az=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8-sig"))
    toy = Toy(spec)
    parts = toy.build(ku=4 if fast else 6, kv=3 if fast else 5)
    t1 = time.time()
    r = Renderer(size=700 if fast else 900, ssaa=2)
    r.set_scene(parts, ao_dirs=48 if fast else 128)
    ims = r.views(str(out / "view_"), which=views)
    tiles = [ims[v] for v in views]
    if ref_az is not None:
        # the reference picture's own angle (a single 3/4 photo is rarely exactly 35 degrees)
        im = r.render(float(ref_az))
        im.save(out / "view_ref.png")
        tiles = [im] + tiles
    t2 = time.time()
    refim = Image.open(ref).convert("RGB") if ref else None
    sheet(tiles, ref=refim).save(out / "sheet.png")
    try:  # can the face be seen? (a muzzle pressed into the head, eyes under ears or cheeks) -> flags
        from .visible import visibility
        vis, vflags = visibility(toy, parts)
        toy.flags.extend(vflags)
        (out / "visibility.json").write_text(json.dumps(vis, indent=1), encoding="utf8")
    except Exception as ex:  # noqa: BLE001 - a check, never a build failure
        toy.flags.append(f"face visibility check failed: {ex}")
    (out / "flags.txt").write_text("\n".join(toy.flags) or "(none)", encoding="utf8")
    try:  # every part checked against the designers' library (learn-only statistics)
        from .library import audit
        a = audit(toy.spec)                 # (presets expanded)
        lines = [f"library audit: subject {a['subject']}, head max {a['head_max']:.0f}"]
        for r in a["parts"]:
            lib = r["library"]
            lines.append(f"{'!!' if r['issues'] else 'ok'} {r['part']} ({r['role']}, {r['kind']}): {r['ours']}")
            lines.append(f"   designers ({lib['N']} parts, {lib['designers']} designers, {lib['scope']}): "
                         f"kinds {lib['kinds']}")
            lines += ["   - " + x for x in r["issues"]]
            if r["issues"] and r["suggest"]:
                lines.append(f"   consensus: {r['suggest']}")
        (out / "library_audit.txt").write_text("\n".join(lines), encoding="utf8")
    except Exception as ex:  # noqa: BLE001 - the audit is advice, never a build failure
        (out / "library_audit.txt").write_text(f"audit failed: {ex}", encoding="utf8")
    try:  # cuteness measured on the built geometry, with suggested edits (python -m amikit.cute for predictions)
        from .cute import measure, score, suggest, report_text
        feat = measure(toy, parts)
        cute = {"features": feat, "score": score(feat), "suggestions": suggest(toy.spec, feat, predict=False)}
        (out / "cute.txt").write_text(report_text(cute), encoding="utf8")
    except Exception as ex:  # noqa: BLE001 - advice, never a build failure
        (out / "cute.txt").write_text(f"cute report failed: {ex}", encoding="utf8")
    try:
        from .pattern_text import write_patterns
        write_patterns(toy, out)
    except ImportError:
        pass
    print(f"{spec_path}: geometry {t1 - t0:.1f}s, render {t2 - t1:.1f}s, {sum(len(p['F']) for p in parts)} tris;"
          f" flags: {len(toy.flags)}")
    return toy, parts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("out")
    ap.add_argument("--ref"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--ref-az", type=float, help="also render at the reference picture's angle (deg, 0 = front, "
                                                    "+ = toy's left side toward the camera)")
    a = ap.parse_args()
    build(a.spec, a.out, a.ref, a.fast, ref_az=a.ref_az)
