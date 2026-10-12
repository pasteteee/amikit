"""Transparent previews of a toy for the site: the studio render with a clear backdrop and only the soft contact
shadow left of the floor (Renderer.render(transparent=True)), cropped to the toy + shadow on a square canvas.

    python -m amikit.preview <spec.json> <out_dir> [--views 34,front] [--size 900] [--pad 0.06]

Writes <out_dir>/preview_<view>.png (RGBA). Same geometry, colours and light as the opaque views, so a preview
always shows exactly what the pattern makes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from PIL import Image

from .render import VIEWS, Renderer
from .toy import Toy


def crop_alpha(im: Image.Image, pad: float = 0.06, threshold: int = 10) -> Image.Image:
    """Square crop around everything visible (alpha > threshold), `pad` of the side added around it."""
    a = im.getchannel("A").point(lambda v: 255 if v > threshold else 0)
    box = a.getbbox()
    if not box:
        return im
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    side = int(side * (1 + 2 * pad))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.alpha_composite(im, (int(side / 2 - cx), int(side / 2 - cy)))
    return canvas


def render_previews(spec: dict | Toy, out_dir, views=("34", "front"), size: int = 900, pad: float = 0.06,
                    fast: bool = False) -> dict[str, Path]:
    """Build the toy once and write one transparent PNG per view; returns {view: path}."""
    toy = spec if isinstance(spec, Toy) else Toy(spec)
    parts = toy.build(ku=4 if fast else 6, kv=3 if fast else 5)
    r = Renderer(size=size, ssaa=2)
    try:
        r.set_scene(parts, ao_dirs=48 if fast else 128)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        for v in views:
            im = r.render(VIEWS[v], transparent=True)
            p = out / f"preview_{v}.png"
            crop_alpha(im, pad=pad).save(p)
            paths[v] = p
        return paths
    finally:
        r.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--views", default="34,front", help=f"comma separated, of {', '.join(VIEWS)}")
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8-sig"))
    views = tuple(v.strip() for v in a.views.split(",") if v.strip())
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise SystemExit(f"unknown views: {unknown}")
    paths = render_previews(spec, a.out, views=views, size=a.size, pad=a.pad, fast=a.fast)
    print(json.dumps({"ok": True, "previews": {k: str(v) for k, v in paths.items()}}))


if __name__ == "__main__":
    main()
