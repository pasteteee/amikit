"""Cuteness by numbers: measure the baby-schema on the built toy, score it, suggest concrete edits.

    python -m amikit.cute <spec.json> [--out DIR] [--apply N] [--ref concept.png] [--json] [--no-predict]
    python -m amikit.cute targets                  # re-derive the target ranges from the concepts + judged toys

measure(toy, parts)   features from the BUILT geometry (not the spec): head width/height (the visible head, from the
                      crown to the chin above the body), head width / body width, head height share of the toy,
                      eye size / head width, eye height (fraction of the visible head from the crown), eye spacing,
                      eyes relative to the muzzle, muzzle width/height/position, limb length / body height and
                      chubbiness, ear size / head width, ear uprightness and thickness (paper-flat plates), head
                      fill (boxiness: an ellipse fills 0.785 of its box), left/right symmetry.
measure_image(im)     the same proportions from a front picture (silhouette + dark eye blobs): used on the concepts
                      and on our front render with one estimator, so the two are comparable.
score(features)       0..100 against baby-schema target ranges (TARGETS: where each range comes from is written
                      next to it: designers' library / KNOWLEDGE_FACES statistics, the toys judged cute (bunny,
                      penguin, panda), the 20 new2610 concepts measured with measure_image).
suggest(spec)         ranked edit ops (amikit/edit.py) for the features out of range; each op is applied to a copy,
                      rebuilt and re-measured, so the predicted effect is the engine's own number.
The engine owns the numbers; a person or an LLM (amikit/cute_llm.py, or the agent through
`amikit_job.py cute`) only chooses among the suggestions with the render in view.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------------------------
# targets: (low, high) of the cute range, the ideal, a weight, and where the range comes from
TARGETS = {
    # judged-cute toys = bunny, penguin, panda, teddy (amikit/toys), measured with measure(); concepts = the 20
    # new2610 concept fronts measured with measure_image() next to our renders of the same toys (2026-10-07 run note)
    "head_wh":        dict(lo=1.15, hi=1.50, ideal=1.32, w=1.0,
                           src="visible head (crown to chin above the body) width/height; judged toys 1.29-1.39"),
    "head_body":      dict(lo=1.15, hi=1.50, ideal=1.30, w=1.2,
                           src="SPEC.md chibi rule 1.3-1.5; judged toys 1.10-1.33"),
    "head_share":     dict(lo=0.44, hi=0.62, ideal=0.52, w=1.0,
                           src="visible head height / toy height without ears; judged toys 0.47-0.52"),
    "eye_d_head":     dict(lo=0.10, hi=0.16, ideal=0.13, w=1.0,
                           src="eye diameter / head width; judged toys 0.11-0.15; concepts 0.08-0.14 (median 0.115); "
                               "KNOWLEDGE_FACES: 1.3-2 st, kawaii 2.5-3 st"),
    "eye_height":     dict(lo=0.46, hi=0.64, ideal=0.56, w=1.3,
                           src="eye centre from the crown / visible head; judged toys 0.58-0.72; KNOWLEDGE_FACES "
                               "0.57 of the head rounds (~0.61 projected); concepts sit 0.13 higher than our renders "
                               "(images: concepts median 0.50, ours 0.63)"),
    "eye_spacing":    dict(lo=0.38, hi=0.54, ideal=0.46, w=1.0,
                           src="eye centres apart / head width; judged toys 0.41-0.51; concepts 0.37-0.61 "
                               "(median 0.51)"),
    "eye_above_muzzle": dict(lo=-0.9, hi=0.6, ideal=-0.1, w=0.6,
                             src="eye centre above the muzzle's top, in eye diameters; judged toys -0.93..-0.24; "
                                 "KNOWLEDGE_FACES 'one round above the snout'"),
    "muzzle_w":       dict(lo=0.28, hi=0.50, ideal=0.38, w=0.8,
                           src="muzzle width / head width; judged toys 0.34-0.47"),
    "muzzle_height":  dict(lo=0.60, hi=0.80, ideal=0.70, w=0.8,
                           src="muzzle centre from the crown / visible head; judged toys 0.66-0.74; KNOWLEDGE_FACES "
                               "0.62 of the head rounds"),
    "arm_len":        dict(lo=0.50, hi=1.10, ideal=0.80, w=0.4,
                           src="arm length / body height (a sitting chibi body is short); judged toys 0.99-1.05"),
    "leg_len":        dict(lo=0.40, hi=1.10, ideal=0.80, w=0.4,
                           src="leg length / body height; judged toys 0.42-1.01"),
    "limb_chub":      dict(lo=0.45, hi=0.90, ideal=0.65, w=0.5,
                           src="limb width / length (stubby, not sticks); judged toys 0.44-0.71"),
    "ear_w":          dict(lo=0.30, hi=0.55, ideal=0.42, w=0.6,
                           src="ear width / head width (front view); judged toys 0.46-0.54; designers' ear max_rel "
                               "0.22-0.50 by stitches"),
    "ear_tilt":       dict(lo=0.0, hi=40.0, ideal=18.0, w=0.6,
                           src="ear lean from vertical seen from the front, deg (long/floppy ears exempt); judged "
                               "panda 34, teddy 37"),
    "ear_thick":      dict(lo=5.5, hi=14.0, ideal=7.5, w=0.8,
                           src="ear thickness mm (two sc layers + the fold); flattened plates measure 4.6-5.3 "
                               "(teddy, quokka, calf) and read as paper in close-ups"),
    "head_fill":      dict(lo=0.66, hi=0.84, ideal=0.76, w=0.5,
                           src="front silhouette area / box of the visible head; judged toys 0.66-0.81; > 0.84 boxy"),
    "symmetry_mm":    dict(lo=0.0, hi=2.5, ideal=1.0, w=0.4,
                           src="mean left/right mismatch of head, ears and face, mm (stitch relief alone ~1-1.6)"),
}
EXEMPT_TILT = ("long", "floppy")


def _role(name):
    from .library import role_of_id
    return role_of_id(name)


def _parts_by_role(toy):
    roles = {}
    for name, inst in toy.inst.items():
        r = _role(inst.p.id)
        if r is None and inst.p.id.lower() in ("snout",):
            r = "muzzle"
        roles.setdefault(r, []).append(inst)
    return roles


def _mesh(parts, name):
    for m in parts:
        if m["name"] == name:
            return m
    return None


def _eyes(toy, head):
    """World centres (grounded) and diameter of the two eyes: safety eyes, or mirrored dark satin/felt eyes."""
    out = []
    for f in toy.spec.get("face", []):
        t = f.get("type")
        dark = str(f.get("colour", "black")).lower() in ("black", "brown", "dark", "darkbrown", "navy") or \
            str(f.get("colour", "")).startswith("#1") or str(f.get("colour", "")).startswith("#2")
        if t in ("safety_eye", "eye"):
            d = float(f.get("size_mm", 9.0))
        elif t in ("satin", "felt") and f.get("mirror") and dark and "at" in f:
            if t == "satin":
                d = float(np.sqrt(float(f.get("width_mm", 6)) * float(f.get("height_mm", 6))))
            else:
                sz = f.get("size_mm", [10, 7])
                d = float(np.sqrt(sz[0] * sz[1]))
            if d < 3.5:
                continue
        else:
            continue
        on = f.get("on")
        try:
            inst = toy.inst[toy._side(on, None)]
        except KeyError:
            continue
        if inst is not head:
            continue
        az = float(f["at"].get("az", 0.0))
        P1 = inst.point(f["at"]["round"], az)[0]
        pts = [P1]
        if f.get("mirror"):
            pts.append(inst.point(f["at"]["round"], -az)[0])
        out.append((pts, d))
        if len(out) and len(out[0][0]) == 2:
            break
    if not out:
        return None
    pts, d = out[0]
    if len(pts) < 2 and len(out) > 1:
        pts = pts + out[1][0]
    if len(pts) < 2:
        return None
    gs = np.array([0.0, toy.ground_shift, 0.0])
    return [p + gs for p in pts[:2]], d


def _front_fill(V):
    """Front-silhouette area of a point cloud / its bounding box, from per-row widths."""
    y = V[:, 1]
    lo, hi = y.min(), y.max()
    rows = 40
    area = 0.0
    for k in range(rows):
        sel = (y >= lo + (hi - lo) * k / rows) & (y < lo + (hi - lo) * (k + 1) / rows)
        if sel.sum() > 2:
            area += np.ptp(V[sel, 0])
    W = np.ptp(V[:, 0])
    return float(area / rows / max(W, 1e-6))


def measure(toy, parts=None):
    """Baby-schema features of a built toy (world mm, grounded). Missing pieces give None."""
    if parts is None:
        parts = toy.build(ku=2, kv=2)
    gs = np.array([0.0, toy.ground_shift, 0.0])
    roles = _parts_by_role(toy)
    f = {}
    head = (roles.get("head") or [None])[0]
    body = (roles.get("body") or [None])[0]
    one_piece = head is None
    if one_piece:
        head = body
    if head is None:
        return {"note": "no head or body piece"}
    hm = _mesh(parts, head.name)
    V = hm["V"]
    W = float(np.ptp(V[:, 0]))
    top = float(V[:, 1].max())
    cz = float(V[:, 2].mean())
    # the chin: the lowest point of the head's front half that is not inside the body
    chin = float(V[:, 1].min())
    if body is not None and body is not head:
        front = V[V[:, 2] > cz]
        sd, _, _ = body.signed_distance(front - gs)
        vis = front[sd > 0.3]
        if len(vis):
            chin = float(np.percentile(vis[:, 1], 1))
    band_body_w = None
    if one_piece:
        # head and body worked as one piece (a seal, a bao, a penguin body): the visible head is the band from the
        # crown down to the chin - a little below the lowest face feature (the muzzle, the mouth, else the eyes) -
        # and its width is measured in that band, not at the widest belly round
        floor = None
        mzs = [i for i in roles.get("muzzle", []) if i.app is None and _mesh(parts, i.name) is not None]
        if mzs:
            floor = float(_mesh(parts, mzs[0].name)["V"][:, 1].min())
        for it in toy.spec.get("face", []):
            if it.get("type") not in ("stitch", "embroidery", "line") or it.get("on") != head.p.id:
                continue
            pts = it.get("points") or []
            if pts and all(abs(((float(a) + 180) % 360) - 180) < 60 for _, a in pts):
                ys = [float(head.point(r, a)[0][1]) + float(gs[1]) for r, a in pts]
                floor = min(ys) if floor is None else min(floor, min(ys))
        if floor is None:
            ey0 = _eyes(toy, head)
            if ey0:
                cy0 = (ey0[0][0][1] + ey0[0][1][1]) / 2
                floor = cy0 - 0.75 * (top - cy0)
        if floor is not None and floor > chin:
            chin = max(chin, floor - 0.12 * (top - floor))
            band = V[V[:, 1] >= chin]
            if len(band) > 10:
                W = float(np.ptp(band[:, 0]))
            below = V[V[:, 1] < chin]
            if len(below) > 10:
                band_body_w = float(np.ptp(below[:, 0]))
    Hv = max(top - chin, 1e-6)
    f["head_w_mm"] = round(W, 1)
    f["head_h_mm"] = round(Hv, 1)
    f["head_wh"] = round(W / Hv, 3)
    hv = V[V[:, 1] >= chin]
    f["head_fill"] = round(_front_fill(hv), 3)
    loose = ("fringe@", "thread@", "satin@", "eye@", "nose@", "felt@", "strap@")
    solid = [m for m in parts if not m["name"].startswith(loose)]
    ears = roles.get("ear", [])
    ear_names = {e.name for e in ears} | {n for n in toy.inst if n.startswith("inner_ear")}
    tall = max(float(m["V"][:, 1].max()) for m in solid if m["name"] not in ear_names and "horn" not in m["name"])
    f["toy_h_mm"] = round(tall, 1)
    if one_piece and band_body_w:
        f["head_share"] = round(Hv / max(tall, 1e-6), 3)
        f["body_w_mm"] = round(band_body_w, 1)
        f["head_body"] = round(W / max(band_body_w, 1e-6), 3)
    if not one_piece:
        f["head_share"] = round(Hv / max(tall, 1e-6), 3)
        if body is not None:
            bm = _mesh(parts, body.name)
            f["body_w_mm"] = round(float(np.ptp(bm["V"][:, 0])), 1)
            f["head_body"] = round(W / max(f["body_w_mm"], 1e-6), 3)
    # eyes
    ey = _eyes(toy, head)
    if ey:
        (pL, pR), d = ey
        cy = (pL[1] + pR[1]) / 2
        f["eye_d_mm"] = round(d, 1)
        f["eye_d_head"] = round(d / W, 3)
        f["eye_height"] = round((top - cy) / Hv, 3)
        f["eye_spacing"] = round(abs(pL[0] - pR[0]) / W, 3)
        f["eye_gap_in_eyes"] = round((abs(pL[0] - pR[0]) - d) / d, 2)
    # muzzle
    mz = [i for i in roles.get("muzzle", []) if i.app is None]
    if mz:
        mm = _mesh(parts, mz[0].name)
        MV = mm["V"]
        f["muzzle_w"] = round(float(np.ptp(MV[:, 0])) / W, 3)
        f["muzzle_h"] = round(float(np.ptp(MV[:, 1])) / Hv, 3)
        f["muzzle_height"] = round((top - float(MV[:, 1].mean())) / Hv, 3)
        if ey:
            f["eye_above_muzzle"] = round((cy - float(MV[:, 1].max())) / d, 2)
    # limbs
    bh = float(np.ptp(_mesh(parts, body.name)["V"][:, 1])) if (body is not None and not one_piece) else Hv
    for role in ("arm", "leg"):
        lims = [i for i in roles.get(role, []) if i.app is None]
        if not lims:
            continue
        L = []
        C = []
        for inst in lims:
            a0 = inst.to_world(inst.p.anchor(0.0, **inst.lp)[None])[0]
            a1 = inst.to_world(inst.p.anchor(float(inst.p.M), **inst.lp)[None])[0]
            length = float(np.linalg.norm(a1 - a0)) + inst.p.max_radius() * (0.5 if inst.p.closed else 1.0)
            L.append(length)
            C.append(2 * inst.p.max_radius() / max(length, 1e-6))
        f[f"{role}_len"] = round(float(np.mean(L)) / max(bh, 1e-6), 3)
        f.setdefault("_chub", []).extend(C)
    if "_chub" in f:
        f["limb_chub"] = round(float(np.mean(f.pop("_chub"))), 3)
    # ears
    if ears:
        ws, tilts, thick, kinds = [], [], [], []
        for inst in ears:
            em = _mesh(parts, inst.name)
            EV = em["V"]
            ws.append(float(np.ptp(EV[:, 0])))
            c = EV.mean(0)
            X = EV - c
            u, s, vt = np.linalg.svd(X[::max(1, len(X) // 3000)], full_matrices=False)
            thick.append(float(np.ptp(X @ vt[2])))
            # uprightness, seen from the front: from the seam (lowest 15 % of the ear) to the ear's centre
            low = EV[EV[:, 1] <= np.percentile(EV[:, 1], 15)].mean(0)
            v = c - low
            tilts.append(abs(math.degrees(math.atan2(abs(v[0]), max(v[1], 1e-6)))))
            from .library import describe, kind_of
            H0 = max(head.p.counts)
            kinds.append(kind_of("ear", describe(inst.p.counts, H0)))
        f["ear_w"] = round(float(np.mean(ws)) / W, 3)
        f["ear_thick"] = round(float(np.mean(thick)), 1)
        f["ear_kind"] = kinds[0]
        if kinds[0] not in EXEMPT_TILT:
            f["ear_tilt"] = round(float(np.mean(tilts)), 1)
        f["ear_construction"] = ("dish" if ears[0].p.dish > 0 else "spoon" if ears[0].p.spoon > 0 else
                                 "edge" if getattr(ears[0], "edge_face", None) is not None else
                                 "flattened" if ears[0].p.flatten >= 0.5 else "tube")
    # symmetry
    face_names = {head.name} | {e.name for e in ears} | {i.name for i in roles.get("muzzle", [])}
    P = np.concatenate([m["V"][::5] for m in parts if m["name"] in face_names or m["name"].endswith("@" + head.name)])
    from scipy.spatial import cKDTree
    Pm = P * np.array([-1.0, 1.0, 1.0])
    d, _ = cKDTree(P).query(Pm)
    f["symmetry_mm"] = round(float(np.mean(d)), 2)
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in f.items()}


# ---------------------------------------------------------------------------------------------------------------
def measure_image(im):
    """Proportions from a front picture on a plain backdrop: silhouette (fitview.toy_mask) + dark eye blobs.

    The head is the biggest round blob in the upper part of the silhouette: the largest inscribed circle whose
    centre is in the top 65 % (distance transform). Its width is the silhouette's run through the circle's centre
    at the widest row near it; the crown is the top of the silhouette above the centre (ears stand to the sides);
    the chin is taken as far below the widest row as the crown is above it (heads are drawn about symmetric round
    their widest row). Eyes = the two biggest dark, round, level blobs inside the head. Only for comparing a concept
    with our render through the same estimator (the scored numbers come from the geometry)."""
    from PIL import Image
    from scipy import ndimage
    from .fitview import toy_mask
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    rgb = np.asarray(im.convert("RGB")).astype(float)
    m = toy_mask(im)
    ys, xs = np.where(m)
    if len(ys) < 100:
        return {"note": "no silhouette"}
    y0, y1 = int(ys.min()), int(ys.max())
    Htot = y1 - y0
    dt = ndimage.distance_transform_edt(m)
    dt[y0 + int(0.5 * Htot):] = 0
    dt = dt * (1.0 - 0.6 * np.clip((np.arange(dt.shape[0])[:, None] - y0) / max(Htot, 1), 0, 1))   # prefer the upper blob
    cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    rho = float(dt[cy, cx])

    def run_width(y):
        row = m[y]
        if not row[cx]:
            return 0
        l = cx
        while l > 0 and row[l - 1]:
            l -= 1
        r = cx
        while r < len(row) - 1 and row[r + 1]:
            r += 1
        return r - l + 1
    lo_y, hi_y = max(y0, int(cy - 0.6 * rho)), min(y1, int(cy + 0.6 * rho))
    wr = [run_width(y) for y in range(lo_y, hi_y + 1)]
    wy = lo_y + int(np.argmax(wr))
    head_w = float(max(wr))
    col = np.where(m[:cy, cx])[0]
    crown = int(col.min()) if len(col) else y0
    # walk up from the centre while the column stays inside (a gap = an ear above a dip)
    yy = cy
    while yy > 0 and m[yy - 1, cx]:
        yy -= 1
    crown = yy
    Hh = float(2 * (wy - crown))
    chin = crown + Hh
    f = {"head_wh": round(head_w / max(Hh, 1), 3), "head_share": round(Hh / max(y1 - crown, 1), 3)}
    body = [m[y].sum() for y in range(int(chin), y1 + 1)]
    if len(body) > 4:
        f["head_body"] = round(head_w / float(np.percentile(body, 60)), 3)
    lum = rgb.mean(2)
    dark = (lum < 55) & m
    dark[:crown] = False
    dark[int(chin):] = False
    lab, n = ndimage.label(dark)
    blobs = []
    for k in range(1, n + 1):
        yy_, xx_ = np.where(lab == k)
        A = len(yy_)
        if A < 12:
            continue
        h_, w_ = np.ptp(yy_) + 1, np.ptp(xx_) + 1
        if not (0.5 < h_ / w_ < 2.2) or A < 0.45 * h_ * w_:
            continue
        blobs.append((A, float(xx_.mean()), float(yy_.mean()), math.sqrt(4 * A / math.pi)))
    blobs.sort(reverse=True)
    pair = None
    for i in range(min(6, len(blobs))):
        for j in range(i + 1, min(6, len(blobs))):
            A1, x1, ya, d1 = blobs[i]; A2, x2, yb, d2 = blobs[j]
            if abs(ya - yb) < 0.6 * max(d1, d2) and 0.5 < A1 / A2 < 2.0 and abs(x1 - x2) > 1.5 * max(d1, d2)                     and min(x1, x2) < cx < max(x1, x2):
                pair = (blobs[i], blobs[j]); break
        if pair:
            break
    if pair:
        (A1, x1, ya, d1), (A2, x2, yb, d2) = pair
        d = (d1 + d2) / 2
        f["eye_d_head"] = round(d / head_w, 3)
        f["eye_height"] = round(((ya + yb) / 2 - crown) / max(Hh, 1), 3)
        f["eye_spacing"] = round(abs(x1 - x2) / head_w, 3)
    return {k: float(v) for k, v in f.items()}


# ---------------------------------------------------------------------------------------------------------------
def _feat_score(v, t):
    lo, hi, ideal = t["lo"], t["hi"], t["ideal"]
    if lo <= v <= hi:
        span = (hi - ideal) if v > ideal else (ideal - lo)
        return 1.0 - 0.15 * (abs(v - ideal) / max(span, 1e-9))
    span = max(hi - lo, 1e-9)
    out = (lo - v) if v < lo else (v - hi)
    return max(0.0, 0.85 - 1.6 * out / span)


def score(feat):
    """{'total': 0..100, 'items': {feature: {value, score, lo, hi, status}}}"""
    items = {}
    num = den = 0.0
    for k, t in TARGETS.items():
        v = feat.get(k)
        if v is None:
            continue
        s = _feat_score(float(v), t)
        status = "ok" if t["lo"] <= v <= t["hi"] else ("low" if v < t["lo"] else "high")
        items[k] = {"value": v, "score": round(s, 3), "lo": t["lo"], "hi": t["hi"], "ideal": t["ideal"],
                    "status": status}
        num += t["w"] * s
        den += t["w"]
    return {"total": round(100 * num / max(den, 1e-9), 1), "items": items}


# ---------------------------------------------------------------------------------------------------------------
def _find(spec, key, pred):
    for i, it in enumerate(spec.get(key, [])):
        if pred(it):
            return i, it
    return None, None


def _eye_item(spec):
    ids = [f for f in spec.get("face", []) if f.get("type") in ("safety_eye", "eye")]
    if ids:
        return ids[0]
    for f in spec.get("face", []):
        if f.get("type") in ("satin", "felt") and f.get("mirror") and f.get("on") == "head":
            return f
    return None


def _head_rounds(spec):
    from .rounds import counts
    from .presets import expand
    sp = expand(spec)
    for p in sp["parts"]:
        if p["id"] == "head" and "rounds" in p:
            return len(counts(p["rounds"])), max(counts(p["rounds"]))
    return None, None


def candidates(spec, feat):
    """Edit-op candidates for the features out of range (each a list of ops + why)."""
    from .edit import _ensure_ids
    sp = _ensure_ids(copy.deepcopy(spec))
    out = []
    sc = score(feat)["items"]
    nH, mH = _head_rounds(sp)
    eye = _eye_item(sp)

    def add(why, ops, key):
        out.append({"why": why, "ops": ops, "feature": key})

    st = lambda k: sc.get(k, {}).get("status")
    # eyes ----------------------------------------------------------------------------------------------------
    if eye and "at" in eye and nH:
        r = float(eye["at"]["round"])
        if st("eye_height") in ("low", "high"):
            want = TARGETS["eye_height"]["ideal"]
            # head rounds per visible-head fraction (~ rounds over the visible height); move toward the ideal but
            # stop at the edge of the range + half the way in (small, visible steps)
            edge = TARGETS["eye_height"]["hi"] if feat["eye_height"] > want else TARGETS["eye_height"]["lo"]
            goal = edge + 0.5 * (want - edge)
            dr = (goal - feat["eye_height"]) * nH * 0.9
            dr = float(np.clip(round(dr * 2) / 2, -3, 3)) or (0.5 if dr > 0 else -0.5)
            add(f"eyes {'too high' if dr > 0 else 'too low'} ({feat['eye_height']:.2f} of the head from the crown; "
                f"cute {TARGETS['eye_height']['lo']}-{TARGETS['eye_height']['hi']}); the eye group (brows, patches, "
                f"cheeks around the eyes) moves with them",
                _face_group_ops(sp, eye, dr), "eye_height")
        az = float(eye["at"].get("az", 30))
        if st("eye_spacing") in ("low", "high"):
            k = TARGETS["eye_spacing"]["ideal"] / max(feat["eye_spacing"], 1e-3)
            new = float(np.clip(round(az * k), 12, 60))
            add(f"eyes {'too close' if k > 1 else 'too far apart'} (spacing {feat['eye_spacing']:.2f} of the head)",
                [{"op": "set", "path": f"face.{eye['id']}.at.az", "value": new}], "eye_spacing")
        if st("eye_d_head") in ("low", "high") and eye.get("type") in ("safety_eye", "eye"):
            want = TARGETS["eye_d_head"]["ideal"] * feat["head_w_mm"]
            sizes = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 20]
            new = min(sizes, key=lambda s: abs(s - want))
            if new != eye.get("size_mm"):
                add(f"eyes {'too small' if new > eye.get('size_mm', 9) else 'too big'} "
                    f"({feat['eye_d_head']:.2f} of the head width)",
                    [{"op": "set", "path": f"face.{eye['id']}.size_mm", "value": new}], "eye_d_head")
    # muzzle --------------------------------------------------------------------------------------------------
    mi, mz = _find(sp, "parts", lambda p: p.get("id") in ("muzzle", "snout"))
    ei, me = _find(sp, "assembly", lambda e: e.get("part") in ("muzzle", "snout"))
    if mz and "rounds" in mz:
        if st("muzzle_w") in ("low", "high") and feat.get("muzzle_w") is not None:
            k = TARGETS["muzzle_w"]["ideal"] / max(feat["muzzle_w"], 1e-3)
            k = float(np.clip(k, 0.75, 1.3))
            add(f"muzzle {'small' if k > 1 else 'big'} ({feat['muzzle_w']:.2f} of the head width)",
                [{"op": "scale_part", "part": mz["id"], "factor": round(k, 2)}], "muzzle_w")
        # (one-piece head+body toys and muzzles the measure could not see have no muzzle height: nothing to move)
        if (me and isinstance(me.get("at", {}).get("round"), (int, float)) and st("muzzle_height") in ("low", "high")
                and feat.get("muzzle_height") is not None and nH):
            dr = (TARGETS["muzzle_height"]["ideal"] - feat["muzzle_height"]) * nH * 0.9
            dr = float(np.clip(round(dr * 2) / 2, -3, 3)) or (0.5 if dr > 0 else -0.5)
            add(f"muzzle {'too high' if dr > 0 else 'too low'} on the face",
                [{"op": "set", "path": f"assembly.{me.get('name', me['part'])}.at.round",
                  "value": round(float(me["at"]["round"]) + dr, 1)}], "muzzle_height")
    # head ----------------------------------------------------------------------------------------------------
    hi_, hp = _find(sp, "parts", lambda p: p.get("id") == "head")
    if hp and mH:
        if st("head_body") == "low":
            k = float(np.clip(TARGETS["head_body"]["ideal"] / max(feat["head_body"], 1e-3), 1.05, 1.25))
            add(f"head small for the body (head/body {feat['head_body']:.2f})",
                [{"op": "scale_part", "part": "head", "factor": round(k, 2)}], "head_body")
            bi, bp = _find(sp, "parts", lambda p: p.get("id") == "body")
            if bp and "rounds" in bp:
                add(f"body wide for the head (head/body {feat['head_body']:.2f})",
                    [{"op": "scale_part", "part": "body", "factor": round(1 / k, 2)}], "head_body")
        if (st("head_wh") == "low" or st("head_fill") == "high") and str(hp.get("rounds", [""])[0]).startswith("mr"):
            from .presets import head_rounds
            from .rounds import counts as _counts
            new_r = head_rounds(mH)
            k = len(_counts(new_r)) / nH
            ops = [{"op": "set", "path": "parts.head.rounds", "value": new_r}]
            ops += _remap_ops(sp, "head", k)
            add(f"head tall/boxy (w/h {feat.get('head_wh')}, fill {feat.get('head_fill')}): cute_head rounds at the "
                f"same widest round (everything sewn on the head keeps its relative place)", ops, "head_wh")
    # ears ----------------------------------------------------------------------------------------------------
    ai, ap = _find(sp, "parts", lambda p: _role(p.get("id", "")) == "ear" and "rounds" in p)
    if ap is not None:
        ek = feat.get("ear_kind", "round")
        cons = feat.get("ear_construction")
        entries = [e for e in sp.get("assembly", []) if e.get("part") == ap["id"]]
        if cons == "dish" and st("ear_thick") == "low":
            if str(ap["rounds"][0]).startswith("mr"):
                from .rounds import counts as _counts
                from .library import counts_to_steps
                c = _counts(ap["rounds"])
                i = c.index(max(c))
                add("dish ears too thin: one more round at full width rolls into a fatter rim",
                    [{"op": "set", "path": f"parts.{ap['id']}.rounds", "value": counts_to_steps(
                        [c[0]] + c[1:i + 1] + [max(c)] + c[i + 1:])}], "ear_thick")
        elif cons in ("flattened", "tube") or (cons not in ("dish", "spoon") and st("ear_thick") == "low"):
            if ek in ("round", "other"):
                ops = [{"op": "set", "path": f"parts.{ap['id']}.dish", "value": 0.5}]
                if ap.get("flatten"):
                    ops.append({"op": "remove", "path": f"parts.{ap['id']}.flatten"})
                for e in entries:
                    nm = e.get("name", e["part"])
                    ops += [{"op": "set", "path": f"assembly.{nm}.attach", "value": "edge"},
                            {"op": "set", "path": f"assembly.{nm}.face", "value": [0.1, 0.25, 1.0]},
                            {"op": "set", "path": f"assembly.{nm}.sink", "value": 5.0}]
                    for k in ("seam", "seam_line", "pinch", "pinch_rounds", "front"):
                        if k in e:
                            ops.append({"op": "remove", "path": f"assembly.{nm}.{k}"})
                add("ears are flat plates: make them dishes (ring in front, thick rolled rim, stand on the rim)",
                    ops, "ear_thick")
            else:
                ops = [{"op": "set", "path": f"parts.{ap['id']}.spoon", "value": {"pointed": 0.35, "long": 0.3}.get(ek, 0.45)}]
                if ap.get("flatten"):
                    ops.append({"op": "remove", "path": f"parts.{ap['id']}.flatten"})
                for e in entries:
                    nm = e.get("name", e["part"])
                    for k in ("seam", "seam_line", "pinch", "pinch_rounds"):
                        if k in e:
                            ops.append({"op": "remove", "path": f"assembly.{nm}.{k}"})
                add("ears are flat plates: press them into spoons (two layers, rounded folds, hollow front)",
                    ops, "ear_thick")
        if st("ear_w") in ("low", "high"):
            k = float(np.clip(TARGETS["ear_w"]["ideal"] / max(feat["ear_w"], 1e-3), 0.75, 1.35))
            add(f"ears {'small' if k > 1 else 'big'} ({feat['ear_w']:.2f} of the head width)",
                [{"op": "scale_part", "part": ap["id"], "factor": round(k, 2)}], "ear_w")
        if st("ear_tilt") == "high":
            for e in entries:
                if isinstance(e.get("dir"), list):
                    d = np.array(e["dir"], float)
                    d[0] *= 0.45
                    nm = e.get("name", e["part"])
                    add(f"ears lean out {feat['ear_tilt']:.0f} deg: stand them up",
                        [{"op": "set", "path": f"assembly.{nm}.dir", "value": [round(float(x), 2) for x in d]}],
                        "ear_tilt")
    # limbs ---------------------------------------------------------------------------------------------------
    for role in ("arm", "leg"):
        if st(f"{role}_len") == "high":
            pi, pp = _find(sp, "parts", lambda p: _role(p.get("id", "")) == role and "rounds" in p)
            if pp is not None:
                from .rounds import expand as rexp
                rr = pp["rounds"]
                evens = [i for i, s_ in enumerate(rr) if isinstance(s_, str) and s_.startswith("even ")]
                if evens:
                    i = evens[-1]
                    n = int(rr[i].split()[1])
                    new = list(rr)
                    k = max(1, n // 3)
                    if n - k > 0:
                        new[i] = f"even {n - k}"
                    else:
                        new.pop(i)
                    add(f"{role}s long ({feat[f'{role}_len']:.2f} of the body height): {k} straight round(s) fewer",
                        [{"op": "set", "path": f"parts.{pp['id']}.rounds", "value": new}], f"{role}_len")
    return out


def _face_group_ops(sp, eye, dr):
    """Move the eyes by dr rounds together with what is laid out around them on the head: face items (brows,
    highlights, blush, felt) and appliques (eye patches, cheeks) within 3.5 rounds and 35 deg of the eye; the
    muzzle and its details stay."""
    r0 = float(eye["at"]["round"]); a0 = abs(float(eye["at"].get("az", 0)))
    on = eye.get("on", "head")
    near = lambda at: (isinstance(at.get("round"), (int, float)) and abs(float(at["round"]) - r0) <= 3.5
                       and abs(abs(float(at.get("az", 0))) - a0) <= 35)
    ops = []
    for f in sp.get("face", []):
        if f.get("on") != on:
            continue
        if f is eye or ("at" in f and near(f["at"])):
            ops.append({"op": "set", "path": f"face.{f['id']}.at.round", "value": round(float(f["at"]["round"]) + dr, 1)})
        elif "points" in f and all(abs(r - r0) <= 3.5 for r, a in f["points"]) and                 any(abs(abs(a) - a0) <= 35 for r, a in f["points"]):
            ops.append({"op": "set", "path": f"face.{f['id']}.points",
                        "value": [[round(r + dr, 2), a] for r, a in f["points"]]})
    for e in sp.get("assembly", []):
        if e.get("on") == on and e.get("applique") and near(e.get("at", {})) and _role(e["part"]) is None:
            ops.append({"op": "set", "path": f"assembly.{e.get('name', e['part'])}.at.round",
                        "value": round(float(e["at"]["round"]) + dr, 1)})
    return ops


def _remap_ops(sp, part_id, k):
    """set-ops that keep things sewn on a piece in the same relative place when its rounds are rewritten (round
    numbers scaled by k = new rounds / old rounds), as edit.scale_part does."""
    names = {e.get("name", e["part"]) for e in sp.get("assembly", []) if e["part"] == part_id} | {part_id}
    ops = []
    for e in sp.get("assembly", []):
        if e.get("on") in names and isinstance(e.get("at", {}).get("round"), (int, float)):
            ops.append({"op": "set", "path": f"assembly.{e.get('name', e['part'])}.at.round",
                        "value": round(float(e["at"]["round"]) * k * 2) / 2})
    for f in sp.get("face", []):
        if f.get("on") not in names:
            continue
        if isinstance(f.get("at", {}).get("round"), (int, float)):
            ops.append({"op": "set", "path": f"face.{f['id']}.at.round", "value": round(float(f["at"]["round"]) * k, 1)})
        if "points" in f:
            ops.append({"op": "set", "path": f"face.{f['id']}.points",
                        "value": [[round(r * k, 2), a] for r, a in f["points"]]})
    return ops


def suggest(spec, feat=None, predict=True, toy=None):
    """Ranked suggestions: [{why, ops, feature, predicted: {total, delta, features}}], best predicted gain first."""
    from .toy import Toy
    from . import edit
    if feat is None:
        toy = toy or Toy(spec)
        feat = measure(toy)
    base = score(feat)["total"]
    out = []
    for c in candidates(spec, feat):
        if predict:
            try:
                new, rep = edit.apply(spec, c["ops"], validate=False)
                t2 = Toy(new)
                f2 = measure(t2)
                s2 = score(f2)["total"]
                c["predicted"] = {"total": s2, "delta": round(s2 - base, 1),
                                  "feature": {c["feature"]: [feat.get(c["feature"]), f2.get(c["feature"])]},
                                  "flags": t2.flags[:5]}
            except Exception as ex:  # noqa: BLE001 - a candidate that does not build is reported, not raised
                c["predicted"] = {"error": str(ex)[:200]}
        out.append(c)
    if predict:
        out.sort(key=lambda c: -(c.get("predicted", {}).get("delta", -99)))
    return out


def report(spec, toy=None, parts=None, predict=False):
    from .toy import Toy
    toy = toy or Toy(spec)
    feat = measure(toy, parts)
    sc = score(feat)
    sug = suggest(spec, feat, predict=predict, toy=toy)
    return {"features": feat, "score": sc, "suggestions": sug}


def report_text(rep):
    sc = rep["score"]
    L = [f"cuteness score {sc['total']:.0f}/100 (baby-schema targets; amikit/cute.py TARGETS)"]
    for k, it in sc["items"].items():
        mark = "ok" if it["status"] == "ok" else "!!"
        L.append(f"{mark} {k:<17} {it['value']!s:>7}  cute {it['lo']}-{it['hi']} (ideal {it['ideal']})"
                 + ("" if it["status"] == "ok" else f"  -> {it['status']}"))
    f = rep["features"]
    for k in ("ear_kind", "ear_construction", "eye_d_mm", "eye_gap_in_eyes", "head_w_mm", "head_h_mm"):
        if k in f:
            L.append(f"   {k}: {f[k]}")
    if rep["suggestions"]:
        L.append("suggestions (edit ops; python -m amikit.cute <spec> --apply N applies the first N):")
        for i, c in enumerate(rep["suggestions"], 1):
            pr = c.get("predicted")
            ptxt = (f" [predicted {pr['total']:.0f} ({pr['delta']:+.1f})]" if pr and "delta" in pr else
                    f" [does not build: {pr['error']}]" if pr and "error" in pr else "")
            L.append(f"{i}. {c['why']}{ptxt}")
            L.append("   " + json.dumps(c["ops"], ensure_ascii=False))
    else:
        L.append("no suggestions: every measured feature is in the cute range")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------
def head_closeups(spec_or_toy, out_dir, size=520, ref=None):
    """Head close-ups (front, 35 deg, 90 deg) + a full 3/4 view; returns the paths. With `ref` a sheet with the
    concept first is written too (closeup_sheet.png)."""
    from PIL import Image
    from .toy import Toy
    from .render import Renderer
    toy = spec_or_toy if hasattr(spec_or_toy, "inst") else Toy(spec_or_toy)
    parts = toy.build(ku=4, kv=3)
    r = Renderer(size=size, ssaa=2)
    r.set_scene(parts, ao_dirs=64)
    roles = _parts_by_role(toy)
    head = (roles.get("head") or roles.get("body") or [None])[0]
    if head is None:
        # no piece called head or body (a jellyfish dome, a star): the largest piece carries the face
        head = max(toy.inst.values(), key=lambda i: float(i.p.max_radius()))
    names = {head.name} | {i.name for i in roles.get("ear", [])} | {i.name for i in roles.get("muzzle", [])}
    V = np.concatenate([m["V"] for m in parts if m["name"] in names or m["name"].endswith("@" + head.name)])
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2
    rad = float(np.linalg.norm(hi - lo) / 2) * 0.98
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for nm, az in (("head_front", 0), ("head_35", 35), ("head_90", 90)):
        im = r.render(az, target=c, radius=rad)
        im.save(out / f"{nm}.png"); paths[nm] = str(out / f"{nm}.png")
    im = r.render(35)
    im.save(out / "full_34.png"); paths["full_34"] = str(out / "full_34.png")
    im = r.render(0)
    im.save(out / "full_front.png"); paths["full_front"] = str(out / "full_front.png")
    tiles = [Image.open(paths[k]) for k in ("head_front", "head_35", "head_90", "full_34")]
    if ref:
        rf = Image.open(ref).convert("RGB"); rf.thumbnail((size, size)); tiles = [rf] + tiles
    W = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
    sh = Image.new("RGB", (W, size), "white")
    x = 0
    for t in tiles:
        sh.paste(t, (x, (size - t.height) // 2)); x += t.width + 6
    sh.save(out / "closeup_sheet.png"); paths["sheet"] = str(out / "closeup_sheet.png")
    r.release() if hasattr(r, "release") else None
    return paths


def _targets_from_images(paths):
    feats = [measure_image(p) for p in paths]
    out = {}
    for k in ("head_wh", "head_share", "head_body", "eye_d_head", "eye_height", "eye_spacing"):
        v = [f[k] for f in feats if k in f]
        if v:
            out[k] = {"n": len(v), "p15": round(float(np.percentile(v, 15)), 3),
                      "p50": round(float(np.percentile(v, 50)), 3), "p85": round(float(np.percentile(v, 85)), 3)}
    return out, feats


def main(argv=None):
    for s_ in (sys.stdout, sys.stderr):
        try:
            s_.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="python -m amikit.cute")
    ap.add_argument("spec", help="spec.json, or 'targets' to re-derive target ranges from concept pictures")
    ap.add_argument("images", nargs="*", help="(targets) concept front pictures")
    ap.add_argument("--out"); ap.add_argument("--apply", type=int, default=0,
                                                help="apply the first N suggestions")
    ap.add_argument("--pick", help="apply these suggestions (1-based, comma separated), e.g. 1,3")
    ap.add_argument("--write", help="where to write the edited spec (default <out or spec dir>/<name>_cute.json)")
    ap.add_argument("--json", action="store_true"); ap.add_argument("--no-predict", action="store_true")
    ap.add_argument("--ref", help="concept front picture: also measured with measure_image for comparison")
    ap.add_argument("--closeups", action="store_true", help="render head close-ups into --out")
    a = ap.parse_args(argv)
    if a.spec == "targets":
        imgs = a.images or sorted(str(p) for p in (HERE.parents[1] / "output/new-toys-2026-10-07/models/_refs")
                                  .glob("*/front.png"))
        stats, _ = _targets_from_images(imgs)
        print(json.dumps(stats, indent=1))
        return 0
    from .toy import Toy
    from . import edit
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8-sig"))
    toy = Toy(spec)
    rep = report(spec, toy=toy, predict=not a.no_predict)
    if a.ref:
        rep["concept_image"] = measure_image(a.ref)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "cute.txt").write_text(report_text(rep), encoding="utf8")
        (out / "cute.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf8")
        if a.closeups:
            rep["closeups"] = head_closeups(toy, out / "closeups", ref=a.ref)
    if a.apply or a.pick:
        idx = [int(x) - 1 for x in a.pick.split(",")] if a.pick else list(range(a.apply))
        ops = [op for i in idx if 0 <= i < len(rep["suggestions"]) for op in rep["suggestions"][i]["ops"]]
        new, r2 = edit.apply(spec, ops)
        dst = Path(a.write) if a.write else (out or Path(a.spec).parent) / (Path(a.spec).stem + "_cute.json")
        dst.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf8")
        rep["applied"] = {"ops": ops, "spec": str(dst), "score_after": score(measure(Toy(new)))["total"]}
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    else:
        print(report_text(rep))
        if "concept_image" in rep:
            print("concept (front picture):", json.dumps(rep["concept_image"]))
        if "applied" in rep:
            print(f"applied {len(rep['applied']['ops'])} op(s) -> {rep['applied']['spec']} "
                  f"(score {rep['applied']['score_after']:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
