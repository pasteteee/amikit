"""Colour calibration of the renderer: a yarn ball of a known colour, the median of its lit front.

    python -m amikit.calibrate            -> prints input -> output per grey level and channel
"""
from __future__ import annotations
import json
import numpy as np

from .toy import Toy
from .render import Renderer

LEVELS = np.linspace(0.05, 1.0, 9)


def swatch_spec(rgb):
    hexc = "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)
    return {"name": "swatch", "colours": {"c": hexc},
            "parts": [{"id": "ball", "rounds": ["mr 6", "inc 6 to 36", "even 6", "dec 6 to 6"], "colour": "c"}],
            "assembly": [{"part": "ball"}]}


def measure(rgb, renderer=None, raw=False):
    """Lit-front colour of a yarn ball; raw=True measures without the inverse calibration (to refit CAL_OUT)."""
    toy = Toy(swatch_spec(rgb))
    parts = toy.build(ku=4, kv=3)
    r = renderer or Renderer(300, 1)
    r.use_calibration = raw
    r.set_scene(parts, ao_dirs=24)
    im = np.asarray(r.render(0, 8, fit=1.0)).astype(float)
    h, w, _ = im.shape
    # the front-lit centre of the ball (a disc around the middle of the frame)
    yy, xx = np.mgrid[0:h, 0:w]
    m = (xx - w * 0.5) ** 2 + (yy - h * 0.48) ** 2 < (0.12 * w) ** 2
    return np.median(im[m], axis=0) / 255.0


def curve():
    r = Renderer(300, 1)
    out = np.array([measure((g, g, g), r) for g in LEVELS])
    return LEVELS, out


if __name__ == "__main__":
    L, out = curve()
    for g, o in zip(L, out):
        print(f"{g:.3f} -> {np.round(o, 3)}")
    json.dump({"levels": L.tolist(), "out": out.tolist()}, open("calib_curve.json", "w"))
