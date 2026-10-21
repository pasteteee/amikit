"""Silhouette fit of a render against a reference picture (one view): masks, outline overlay, width profile.

    python -m amikit.fitview <reference.png> <render.png> <out.png> [--crop x0,y0,x1,y1] [--mirror]

The toy is separated from a plain backdrop (alpha when the picture has it; else a flood from the border through
backdrop-like pixels, stopped by the toy's edges), holes filled; both silhouettes are
scaled to the same height and aligned at the bottom centre. The overlay draws the reference outline in red and
ours in blue; the report gives the width (fraction of height) at 20 heights and the mean outline mismatch (IoU),
so a spec change can be judged by numbers as well as by eye.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _alpha_mask(im):
    """The object's mask from a real alpha channel (a cut-out picture), or None when there is none."""
    if im.mode not in ("RGBA", "LA") and "transparency" not in im.info:
        return None
    al = np.asarray(im.convert("RGBA"))[..., 3]
    clear, solid = (al < 16).mean(), (al > 240).mean()
    if clear < 0.05 or solid < 0.02:
        return None
    return al > 128


def _backdrop(a):
    """Backdrop colour at every pixel: a plane per channel fitted to the top/left/right border strips (a soft
    vignette or light gradient), refitted once without outliers (a toy touching the border)."""
    h, w, _ = a.shape
    k = max(4, min(h, w) // 60)
    ys, xs = np.mgrid[0:h, 0:w]
    sel = np.zeros((h, w), bool)
    sel[:k] = True; sel[:, :k] = True; sel[:, -k:] = True
    X = np.stack([np.ones(sel.sum()), xs[sel] / w, ys[sel] / h], 1)
    Y = a[sel]
    keep = np.ones(len(Y), bool)
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(X[keep], Y[keep], rcond=None)
        res = np.abs(Y - X @ coef).sum(1)
        keep = res < max(12.0, 3 * np.median(res))
    full = np.stack([np.ones(h * w), xs.ravel() / w, ys.ravel() / h], 1) @ coef
    return full.reshape(h, w, 3), float(np.median(res[keep]))


_MASKS: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_MASKS_MAX = 24
_MASKS_LOCK = threading.Lock()


def toy_mask(im):
    """The toy's silhouette on a plain backdrop (see _toy_mask). Remembered by the picture's content: a concept's
    pictures are measured again at every build of a job (every look, every autofit, every ranked seed), and the
    flood fill on a 768 px picture costs more than the whole render it is compared with."""
    if im.mode not in ("RGB", "RGBA", "L", "LA"):
        return _toy_mask(im)                     # palette pictures: the bytes alone do not say what the colours are
    key = (im.size, im.mode, hashlib.md5(im.tobytes()).digest())
    with _MASKS_LOCK:
        hit = _MASKS.get(key)
        if hit is not None:
            _MASKS.move_to_end(key)
    if hit is None:
        hit = _toy_mask(im)
        hit.setflags(write=False)
        with _MASKS_LOCK:
            _MASKS[key] = hit
            while len(_MASKS) > _MASKS_MAX:
                _MASKS.popitem(last=False)
    return hit.copy()


def _toy_mask(im):
    """The toy's silhouette on a plain backdrop.

    A real alpha channel is used when present. Otherwise the backdrop (a fitted colour plane) is flooded from the
    image border through pixels that look like backdrop: close to its colour, or a floor shadow (the backdrop's
    chromaticity, darker); strong local edges (the toy's outline and yarn texture) stop the flood. Everything the
    flood does not reach is the toy, so cream, white or pale-lavender toys on a light backdrop are kept even where
    their colour is near the backdrop's."""
    am = _alpha_mask(im)
    if am is not None:
        m = ndimage.binary_fill_holes(ndimage.binary_opening(am, iterations=1))
    else:
        a = np.asarray(im.convert("RGB")).astype(float)
        bg, noise = _backdrop(a)
        dist = np.abs(a - bg).sum(2)
        lum = a.mean(2); blum = np.maximum(bg.mean(2), 1.0)
        chrom = a / np.maximum(a.sum(2, keepdims=True), 1.0)
        bchrom = bg / np.maximum(bg.sum(2, keepdims=True), 1.0)
        cdist = np.abs(chrom - bchrom).sum(2)
        g = ndimage.gaussian_filter(lum, 1.0)
        grad = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1)) / 8.0
        grad = ndimage.maximum_filter(grad, 3)
        near = dist < max(14.0, 3.0 * noise)
        shadow = (cdist < 0.022) & (lum < blum * 1.01) & (lum > blum * 0.45) & (grad < 6.0)
        bgl = (near & (grad < 9.0)) | shadow
        lab, _ = ndimage.label(bgl)
        edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        reached = np.isin(lab, edge_labels[edge_labels > 0])
        m = ~reached
        m = ndimage.binary_opening(m, iterations=2)
        m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def normalise(m, H=400):
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    c = m[y0:y1 + 1, x0:x1 + 1]
    s = H / c.shape[0]
    img = Image.fromarray((c * 255).astype(np.uint8)).resize((max(1, int(c.shape[1] * s)), H))
    return np.asarray(img) > 127


def profile(m, k=20):
    H = m.shape[0]
    rows = [m[min(H - 1, int((i + 0.5) / k * H))] for i in range(k)]
    return [round(float(r.sum()) / H, 3) for r in rows]


def compare(ref_im, our_im, out_png=None, mirror=False):
    mr = normalise(toy_mask(ref_im))
    if mirror:
        mr = mr[:, ::-1]
    mo = normalise(toy_mask(our_im))
    W = max(mr.shape[1], mo.shape[1]) + 40
    H = mr.shape[0]

    def place(m):
        out = np.zeros((H, W), bool)
        cx = W // 2 - m.shape[1] // 2
        out[:, cx:cx + m.shape[1]] = m
        return out
    A, B = place(mr), place(mo)
    iou = float((A & B).sum() / max((A | B).sum(), 1))
    if out_png:
        base = np.full((H, W, 3), 255, np.uint8)
        base[A & ~B] = (255, 190, 190)
        base[B & ~A] = (190, 200, 255)
        base[A & B] = (215, 215, 215)
        img = Image.fromarray(base)
        d = ImageDraw.Draw(img)
        for m, col in ((A, (220, 30, 30)), (B, (30, 60, 220))):
            edge = m ^ ndimage.binary_erosion(m)
            ys, xs = np.where(edge)
            for x, y in zip(xs, ys):
                d.point((x, y), fill=col)
        img.save(out_png)
    return {"iou": round(iou, 3), "ref_width_profile": profile(mr), "our_width_profile": profile(mo),
            "ref_aspect": round(mr.shape[1] / mr.shape[0], 3), "our_aspect": round(mo.shape[1] / mo.shape[0], 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("ours"); ap.add_argument("out")
    ap.add_argument("--crop"); ap.add_argument("--mirror", action="store_true")
    a = ap.parse_args()
    ref = Image.open(a.ref).convert("RGB")
    if a.crop:
        ref = ref.crop(tuple(int(v) for v in a.crop.split(",")))
    print(json.dumps(compare(ref, Image.open(a.ours), a.out, a.mirror), indent=1))
