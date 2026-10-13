"""Match the safety eyes to the concept: size and spacing measured on the front views.

    python eye_match.py <refs dir> <build dir> <spec.json> [--write]

Finds the two eyes (the largest round near-black blobs) in the concept's front.png and in the build's
view_front.png, measures eye diameter and eye spacing relative to the head width at eye height, and scales the
spec's safety-eye `size_mm` (and the eyes' az, which sets the spacing) by concept / model. Prints JSON; --write
updates the spec (eyes only). The concept wins over the generic cute targets for eyes: owners compare eyes first.
Rebuild and compare afterwards - a dark mask round the eyes can confuse the blob search ("found": false then).

It cannot see embroidered / satin eyes ("found": false), it is wrong on big-eared heads (the width at eye height
includes the ears) and on heads worked from the nose (the az fix is meaningless there): measure by hand on the
grid from measure.py then. Approved toys have eyes ~0.19-0.20 of the face width, about one eye diameter apart.
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

EYE_SIZES = (5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 20, 24, 30)     # safety eyes on sale (mm)


def measure(img, alpha=None, dark=90):
    a = np.asarray(img.convert("RGB")).astype(int)
    obj = alpha > 20 if alpha is not None else np.abs(a - a[5, 5]).sum(2) > 30
    lab, n = ndimage.label((a.max(2) < dark) & obj)
    blobs = []
    for i in range(1, n + 1):
        yy, xx = np.nonzero(lab == i)
        if len(yy) < 30:
            continue
        w, h = np.ptp(xx) + 1, np.ptp(yy) + 1
        if 0.6 < w / h < 1.6 and len(yy) / (w * h) > 0.55:       # round, filled: an eye, not a mask
            blobs.append((len(yy), max(w, h), xx.mean(), yy.mean()))
    blobs.sort(reverse=True)
    # the two eyes: a pair of similar size, side by side at the same height
    e = None
    for i in range(min(len(blobs), 5)):
        for j in range(i + 1, min(len(blobs), 5)):
            a_, b_ = blobs[i], blobs[j]
            if (abs(a_[0] - b_[0]) < 0.5 * max(a_[0], b_[0]) and abs(a_[3] - b_[3]) < 0.6 * max(a_[1], b_[1])
                    and abs(a_[2] - b_[2]) > 1.2 * max(a_[1], b_[1])):
                e = [a_, b_]
                break
        if e:
            break
    if e is None:
        return None
    cy = int((e[0][3] + e[1][3]) / 2)
    xs = np.nonzero(obj[cy])[0]
    hw = float(xs.max() - xs.min())
    return {"d": float(np.mean([b[1] for b in e])) / hw, "gap": abs(e[0][2] - e[1][2]) / hw}


def measure_any(img, alpha=None):
    """Darkest threshold that finds the pair: dark yarn (purple, brown) next to the eyes merges at higher ones."""
    for dark in (45, 60, 75, 90, 110):
        r = measure(img, alpha, dark)
        if r:
            return r
    return None


def main(refs, build, spec_path, write=False):
    con = Image.open(f"{refs}/front.png").convert("RGBA")
    alpha = np.asarray(con)[..., 3]
    # a flattened reference (refs.py) has no transparency: find the toy against the backdrop like the render
    c = measure_any(con, alpha if (alpha < 250).any() else None)
    m = measure_any(Image.open(f"{build}/view_front.png"))
    spec = json.load(open(spec_path, encoding="utf-8-sig"))
    eyes = [f for f in spec.get("face", []) if f.get("type") in ("safety_eye", "eye")]
    out = {"concept": c, "model": m, "found": bool(c and m and eyes)}
    if out["found"]:
        k = c["d"] / m["d"]
        kg = c["gap"] / m["gap"]
        out["size_factor"] = round(k, 3)
        out["spacing_factor"] = round(kg, 3)
        for f in eyes:
            # snapped to a size a shop sells (the pattern's materials list must be buyable)
            want = float(f.get("size_mm", 9)) * k
            f["size_mm"] = min(EYE_SIZES, key=lambda x: abs(x - want))
            if "az" in f.get("at", {}):
                f["at"]["az"] = round(float(f["at"]["az"]) * kg, 1)
        out["eyes"] = [{"size_mm": f["size_mm"], "at": f.get("at")} for f in eyes]
        if write:
            json.dump(spec, open(spec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*a[:3], write="--write" in a)
