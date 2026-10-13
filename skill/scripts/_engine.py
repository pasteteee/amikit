"""Shared helpers of the skill scripts: find the engine, run builds, read their results, lay out sheets."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VIEWS = ("front", "34", "side", "back")
# concept views from refs.py: left.png is the toy's LEFT side (nose to the viewer's left) = our view_side.png
CONCEPT_FOR = {"front": "front.png", "34": None, "side": "left.png", "back": "back.png"}
BG = (245, 241, 236)


def engine_path() -> str | None:
    """The folder to put on sys.path so `import amikit` works (None when amikit is installed)."""
    try:
        import amikit  # noqa: F401
        return None
    except ImportError:
        pass
    for env in ("AMIKIT_LAB", "AMIKIT_DIR"):
        v = os.environ.get(env)
        if v:
            p = Path(v).resolve()
            if (p / "amikit" / "toy.py").is_file():
                return str(p)
            if (p / "toy.py").is_file() and p.name == "amikit":
                return str(p.parent)
    here = Path(__file__).resolve()
    pkg = here.parents[2]               # <repo = package>/skill/scripts/_engine.py
    if (pkg / "toy.py").is_file() and pkg.name == "amikit":
        return str(pkg.parent)
    raise SystemExit("amikit not found: pip install it, or set AMIKIT_LAB to the folder that contains amikit/")


ENGINE = engine_path()
if ENGINE and ENGINE not in sys.path:
    sys.path.insert(0, ENGINE)


def env() -> dict:
    e = dict(os.environ)
    if ENGINE:
        e["PYTHONPATH"] = ENGINE + os.pathsep + e.get("PYTHONPATH", "")
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def build(spec: Path, out: Path, ref: Path | None = None, fast: bool = True) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "amikit.build", str(spec), str(out)]
    if ref:
        cmd += ["--ref", str(ref)]
    if fast:
        cmd.append("--fast")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env())
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def write_json(p: Path, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def flags(out: Path) -> list[str]:
    f = out / "flags.txt"
    if not f.is_file():
        return []
    lines = [x for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [] if lines == ["(none)"] else lines


def cute_score(out: Path):
    f = out / "cute.txt"
    if f.is_file():
        first = f.read_text(encoding="utf-8").splitlines()[:1]
        if first and "score" in first[0]:
            try:
                return int(first[0].split("score")[1].split("/")[0])
            except ValueError:
                return None
    return None


def iou(ref_png: Path, ours_png: Path, out_png: Path | None = None):
    from amikit.fitview import compare
    r = compare(Image.open(ref_png).convert("RGB"), Image.open(ours_png), str(out_png) if out_png else None)
    return r.get("iou") if isinstance(r, dict) else None


def toy_crop(im: Image.Image) -> Image.Image:
    """The toy only (engine toy mask, or alpha / backdrop colour), cropped to its box, RGBA."""
    im = im.convert("RGBA")
    a = np.asarray(im)
    if (a[..., 3] < 250).any():
        m = a[..., 3] > 20
    else:
        try:
            from amikit.fitview import toy_mask
            m = toy_mask(im)
        except Exception:  # noqa: BLE001
            bg = a[5, 5, :3].astype(int)
            m = np.abs(a[..., :3].astype(int) - bg).sum(2) > 30
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return im
    b = a.copy()
    b[..., 3] = np.where(m, b[..., 3], 0)
    return Image.fromarray(b).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def tile(im: Image.Image | None, S: int, H: int | None = None) -> Image.Image:
    """Scale to height H (narrower if needed), standing on the tile's floor line: equal heights compare."""
    c = Image.new("RGB", (S, S), BG)
    if im is None:
        return c
    H = H or int(S * 0.9)
    k = min(H / im.height, (S - 8) / im.width)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    c.paste(im, ((S - im.width) // 2, S - 6 - im.height), im if im.mode == "RGBA" else None)
    return c


def head_crop(im: Image.Image, top: float = 0.0, bottom: float = 0.6) -> Image.Image:
    """The upper part of the toy (face close-up): rows top..bottom of the toy's height."""
    t = toy_crop(im)
    h = t.height
    return t.crop((0, int(h * top), t.width, max(int(h * top) + 1, int(h * bottom))))


def grid(rows: list[tuple[str, list[Image.Image]]], S: int, header: list[str]) -> Image.Image:
    lab = 190
    W = lab + S * max(len(r[1]) for r in rows)
    sheet = Image.new("RGB", (W, 24 + S * len(rows)), "white")
    d = ImageDraw.Draw(sheet)
    for j, h in enumerate(header):
        d.text((lab + j * S + 6, 6), h, fill=(0, 0, 0))
    for i, (name, tiles) in enumerate(rows):
        y = 24 + i * S
        for k, line in enumerate(name.split("\n")):
            d.text((6, y + 8 + 14 * k), line, fill=(0, 0, 0))
        for j, t in enumerate(tiles):
            sheet.paste(t, (lab + j * S, y))
        d.line((0, y, W, y), fill=(200, 200, 200))
    return sheet
