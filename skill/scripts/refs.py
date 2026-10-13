"""Concept picture -> reference views for the loop: front.png (+ left / right / back from a turnaround sheet).

    python refs.py <picture> <refs_dir> [--single] [--crop x0,y0,x1,y1] [--mirror]

- A transparent 2x2 turnaround sheet (top-left front, top-right left profile, bottom-left right profile, bottom-right
  back) is cut into front.png, left.png, right.png, back.png (each cropped to the toy, 768 px, on a plain backdrop).
- Any other picture (a photo, a drawing, one view) becomes front.png only (`--single` forces this). `--crop` cuts one
  toy out of a collage or a photo first (px); `--mirror` flips it (when the picture shows the toy's other side).
left.png is the toy's LEFT side (nose toward the viewer's left), the same side as the engine's view_side.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

BG = (236, 233, 229)
QUADRANTS = {"front": (0, 0), "left": (1, 0), "right": (0, 1), "back": (1, 1)}


def flatten(im: Image.Image) -> Image.Image:
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, BG + (255,))
    return Image.alpha_composite(bg, im).convert("RGB")


def crop_object(im: Image.Image, pad=0.08) -> Image.Image:
    a = im.getchannel("A").point(lambda v: 255 if v > 64 else 0)
    box = a.getbbox()
    if not box:
        return im
    x0, y0, x1, y1 = box
    s = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    out.paste(im.crop((x0, y0, x1, y1)), (s // 2 - (cx - x0), s // 2 - (cy - y0)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="refs.py")
    ap.add_argument("picture")
    ap.add_argument("refs_dir")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--crop")
    ap.add_argument("--mirror", action="store_true")
    a = ap.parse_args(argv)

    src = Image.open(a.picture)
    if a.crop:
        src = src.crop(tuple(int(v) for v in a.crop.split(",")))
    if a.mirror:
        src = ImageOps.mirror(src)
    d = Path(a.refs_dir)
    d.mkdir(parents=True, exist_ok=True)
    has_alpha = src.mode in ("RGBA", "LA") or "transparency" in src.info
    views = {}
    if a.single or not has_alpha:
        flatten(src).save(d / "front.png")
        views["front"] = "front.png"
    else:
        rgba = src.convert("RGBA")
        w, h = rgba.size
        for name, (qx, qy) in QUADRANTS.items():
            q = rgba.crop((qx * w // 2, qy * h // 2, (qx + 1) * w // 2, (qy + 1) * h // 2))
            flatten(crop_object(q)).resize((768, 768), Image.LANCZOS).save(d / f"{name}.png")
            views[name] = f"{name}.png"
    flatten(src).save(d / "sheet4.png")
    meta = {"refs_dir": str(d), "views": views,
            "note": "left.png shows the toy's LEFT side (nose to the viewer's left) = the engine's view_side.png"}
    (d / "refs.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
