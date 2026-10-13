"""Measure a concept picture before writing numbers: proportions, width profile, colours, a grid to read it by.

    python measure.py <picture> <out_dir> [--at name=x,y ...] [--colours 8]

Writes
  <out>/grid.png      the picture with a labelled grid (fractions of the toy's box: 0.0 top/left .. 1.0 bottom/right)
                      so you can name points and read heights ("eyes at 0.38 of the height") instead of guessing
  <out>/measure.json  toy box in px, height/width ratio, width at every 1/20 of the height (fractions of the max
                      width: find the head, the neck dip, the body), the palette (main colours with their share,
                      as hex), and the colour at each --at point (median of a 13x13 patch; x,y as fractions of the
                      toy's box or as px when > 1)
and prints measure.json. Use the measured hexes in `colours` (sample flat, lit-but-not-highlight regions; very dark
and very light yarn may need a small correction after the first build: compare the render with the sample).
Write your notes (head w/h, head height / total, eye height from the crown, eye spacing / head width, muzzle width /
head width, ear length) BEFORE the spec; recipes.md and review.md say what the numbers usually are.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _engine as E  # noqa: E402


def mask_of(im: Image.Image) -> np.ndarray:
    a = np.asarray(im.convert("RGBA"))
    if (a[..., 3] < 250).any():
        return a[..., 3] > 64
    rgb = a[..., :3].astype(int)
    border = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3),
                             rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    return np.abs(rgb - bg).sum(2) > 40


def palette(rgb: np.ndarray, k: int):
    """Main colours of the toy's pixels: a small k-means in RGB, sorted by share."""
    px = rgb.reshape(-1, 3).astype(float)
    if len(px) > 60000:
        px = px[np.random.default_rng(0).choice(len(px), 60000, replace=False)]
    c = px[np.linspace(0, len(px) - 1, k).astype(int)].copy()
    for _ in range(20):
        lab = np.argmin(((px[:, None, :] - c[None]) ** 2).sum(2), axis=1)
        for j in range(k):
            sel = px[lab == j]
            if len(sel):
                c[j] = np.median(sel, axis=0)
    share = np.bincount(lab, minlength=k) / len(px)
    out = [{"hex": "#%02x%02x%02x" % tuple(int(v) for v in c[j]), "share": round(float(share[j]), 3)}
           for j in np.argsort(-share) if share[j] > 0.01]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="measure.py")
    ap.add_argument("picture")
    ap.add_argument("out")
    ap.add_argument("--at", action="append", default=[], help="name=x,y (fractions of the toy box, or px)")
    ap.add_argument("--colours", type=int, default=8)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    im = Image.open(a.picture).convert("RGBA")
    m = mask_of(im)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        raise SystemExit("no toy found in the picture")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    W, H = x1 - x0 + 1, y1 - y0 + 1
    rgb = np.asarray(im)[..., :3]

    rows = []
    for i in range(20):
        r0, r1 = y0 + int(H * i / 20), y0 + max(int(H * (i + 1) / 20), int(H * i / 20) + 1)
        band = m[r0:r1]
        cols = np.nonzero(band.any(0))[0]
        rows.append(round(float(cols.max() - cols.min() + 1) / W, 3) if len(cols) else 0.0)

    pts = {}
    for spec in a.at:
        name, xy = spec.split("=")
        fx, fy = (float(v) for v in xy.split(","))
        px = int(x0 + fx * W) if fx <= 1 else int(fx)
        py = int(y0 + fy * H) if fy <= 1 else int(fy)
        patch = rgb[max(0, py - 6):py + 7, max(0, px - 6):px + 7].reshape(-1, 3)
        med = np.median(patch, axis=0).astype(int)
        pts[name] = {"px": [px, py], "hex": "#%02x%02x%02x" % tuple(med)}

    res = {"picture": str(a.picture), "box_px": [x0, y0, x1, y1], "height_over_width": round(H / W, 3),
           "width_profile_top_to_bottom": rows,
           "palette": palette(rgb[m], a.colours), "points": pts,
           "note": "width_profile: toy width in each 1/20 of its height, as a fraction of its widest row"}

    g = im.convert("RGB").copy()
    d = ImageDraw.Draw(g)
    for i in range(11):
        x = x0 + W * i / 10
        y = y0 + H * i / 10
        d.line((x, y0, x, y1), fill=(255, 0, 0) if i % 5 == 0 else (255, 150, 150), width=1)
        d.line((x0, y, x1, y), fill=(255, 0, 0) if i % 5 == 0 else (255, 150, 150), width=1)
        d.text((x + 2, max(0, y0 - 12)), f"{i / 10:.1f}", fill=(200, 0, 0))
        d.text((max(0, x0 - 26), y + 2), f"{i / 10:.1f}", fill=(200, 0, 0))
    for name, p in pts.items():
        d.ellipse((p["px"][0] - 4, p["px"][1] - 4, p["px"][0] + 4, p["px"][1] + 4), outline=(0, 0, 255), width=2)
        d.text((p["px"][0] + 6, p["px"][1] - 6), name, fill=(0, 0, 255))
    g.save(out / "grid.png")
    E.write_json(out / "measure.json", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
