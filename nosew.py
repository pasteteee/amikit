"""No-sew construction: ears, limbs and tails crocheted into the parent while its rounds are worked.

A sewn toy finishes every piece and sews it on at the end. In the no-sew construction (the designers' "join as you
go") an open piece is made first, its opening pressed flat (two layers: k stitches against k), and when the parent
reaches the round that holds it, k stitches of that round are worked through both layers of the piece's edge and the
parent's stitches together. Nothing is sewn; the piece sits exactly on the stitches that hold it.

    spec["construction"] = "nosew"          # every eligible piece is joined (convert())
    {"part": "ear", ..., "join": "crochet"}   # one entry joined
    {"part": "ear", ..., "join": "sew"}       # one entry kept sewn in a no-sew toy

What the engine does with a joined entry (toy.py `_place` -> plan()):
- the join round j is the parent round nearest to `at.round` (2..N): the piece's edge is part of round j;
- the edge has `pairs` = opening // 2 stitch pairs; each stitch the parent makes there takes one pair, so the joined
  run of round j is the ops that produce exactly `pairs` stitches (an increase there makes two stitches, each
  through its own pair; a decrease makes one). The run is centred where the entry's `az` says and snapped to whole
  stitches; `at` becomes that snapped (round, az), so the 3D piece sits on the stitches that hold it;
- the seam is flat (fully pinched) along the round: `seam_line` = the round's direction there.
The parent's counts never change (the joined stitches replace plain ones 1:1). check() verifies it all, measure()
says how far the 3D piece is from the stitches it is joined to and how different the toy is from its sewn version.

    python -m amikit.nosew toys/bunny.json out/bunny_nosew            # sewn + no-sew builds, nosew.json, compare.png
"""
from __future__ import annotations
import copy
import math

import numpy as np

from .piece import LEAN
from .rounds import op_cp

# the pieces a no-sew pattern crochets in (by the part id's words): ears, limbs, tails, wings, horns, fins
JOIN_WORDS = ("ear", "arm", "leg", "tail", "wing", "paw", "foot", "feet", "hand", "limb", "fin", "flipper", "horn",
              "antler", "hoof", "tentacle", "claw", "thigh", "fluke")
# (a muzzle, a snout or a cheek is sewn all round its opening: it is a ring on the face, not a line in a round)
NOT_WORDS = ("inner", "muzzle", "snout", "cheek", "nose", "belly", "patch", "spot", "earring")


def _words(pid):
    return [w for w in str(pid).lower().replace("-", "_").split("_") if w]


def wants_join(pid):
    """Is this a piece a no-sew pattern crochets in (by its name)?"""
    ws = _words(pid)
    if any(w.startswith(n) for w in ws for n in NOT_WORDS):
        return False
    return any(w == j or w.startswith(j) or w.endswith(j) for w in ws for j in JOIN_WORDS)


def structural(entry, part, parent_part, mirrored_parents):
    """Why an entry cannot be crocheted in (None = it can): a joined piece is an open piece of rounds sewn by its
    opening onto a round of a piece worked in plain rounds."""
    if any(k in part for k in ("cord", "flat", "motif")):
        return "not worked in rounds"
    if entry.get("applique") or entry.get("wear") or entry.get("around") or entry.get("anchor"):
        return "an applique / worn / collar piece"
    if entry.get("attach", "end") != "end":
        return "sewn by its edge or start"
    if part.get("end", "closed") != "open":
        return "a closed piece (no opening to join)"
    if entry.get("rim_arc"):
        return "sewn by part of its rim"
    if parent_part is None or any(k in parent_part for k in ("cord", "flat", "motif")):
        return "the parent is not worked in rounds"
    if entry.get("on") in mirrored_parents:
        return "the parent is one of a mirrored pair"
    r = (entry.get("at") or {}).get("round", "end")
    if r in ("end", "start"):
        return "sewn at the parent's pole / opening"
    return None


def convert(spec, everything=False):
    """The spec with `join: "crochet"` on every entry a no-sew pattern crochets in (ears, limbs, tails, ...).

    Entries that say `join` already keep it. everything=True joins every structurally possible entry, whatever its
    name. Returns a new spec; the reasons for entries kept sewn are in spec["_nosew_kept"] (dropped by Toy)."""
    s = copy.deepcopy(spec)
    parts = {p["id"]: p for p in s.get("parts", [])}
    mirrored = set()
    for e in s.get("assembly", []):
        if e.get("mirror"):
            mirrored |= {e.get("name", e["part"]), e["part"]}
    kept = []
    for e in s.get("assembly", []):
        if "on" not in e or "join" in e:
            continue
        p = parts.get(e["part"], {})
        why = structural(e, p, parts.get(_part_of(s, e["on"])), mirrored)
        if why is None and not (everything or wants_join(e["part"])):
            why = "not an ear / limb / tail (sewn as designed)"
        if why is None:
            e["join"] = "crochet"
        else:
            kept.append([e.get("name", e["part"]), why])
    s["_nosew_kept"] = kept
    return s


def _part_of(spec, on):
    """Part id of an entry name (entries may be named; mirrored instances end in _L / _R)."""
    for e in spec.get("assembly", []):
        if e.get("name", e["part"]) == on:
            return e["part"]
    base = on[:-2] if on.endswith(("_L", "_R")) else on
    for e in spec.get("assembly", []):
        if e.get("name", e["part"]) == base or e["part"] == base:
            return e["part"]
    return on


# ---------------------------------------------------------------------------------------------------------------
# the plan: which stitches of which round hold the piece

def stitch_theta(j, n, w):
    """theta of produced stitch w (working order from the round-start marker) of round j with n stitches.

    toy.chart: chart index k sits at theta 2*pi*(k + 0.5 - LEAN*(j - 0.5)) / n and a tube is worked toward
    decreasing theta, so working-order stitch w is chart index n - 1 - w."""
    return 2 * np.pi * (n - 1 - np.asarray(w, float) + 0.5 - LEAN * (j - 0.5)) / n


def _w_of_theta(j, n, th):
    return (n - 0.5 - LEAN * (j - 0.5) - th * n / (2 * np.pi)) % n


def stitch_tokens(ops):
    """One token per produced stitch of a round: the op's own token, or for a stitch that is one of the several made
    in one stitch, 'inc_a' / 'inc_b' (first / second sc of an increase) or 'inc3_a/b/c'. A joined run may begin or end
    in the middle of an increase: its first sc goes through the piece's edge, the second does not."""
    out = []
    for o in ops:
        c, p = op_cp(o)
        if p == 1:
            out.append(o)
        elif o == "inc":
            out += ["inc_a", "inc_b"]
        elif o == "inc3":
            out += ["inc3_a", "inc3_b", "inc3_c"]
        else:
            out += [o] * p
    return out


def run_tokens(toks):
    """A run's per-stitch tokens written back as ops: a whole increase (inc_a, inc_b) is 'inc' again."""
    out, i = [], 0
    while i < len(toks):
        if toks[i] == "inc_a" and i + 1 < len(toks) and toks[i + 1] == "inc_b":
            out.append("inc"); i += 2
        elif toks[i] == "inc3_a" and toks[i + 1:i + 3] == ["inc3_b", "inc3_c"]:
            out.append("inc3"); i += 3
        else:
            out.append(toks[i]); i += 1
    return out


def plan(parent, j, az, pairs, taken=None, pair=False):
    """Where `pairs` joined stitches go in round j of the parent: the run of `pairs` stitches made in that round,
    centred nearest to `az` (one new stitch through each pair of the piece's pressed edge).

    Returns {"round", "st_start", "k", "n", "az", "ops", "wrap", "snap_st"}: st_start = the first stitch of the run
    (working order from the marker, 0-based), k = stitches made through the piece, az = the run's centre (the snapped
    placement), ops = the round's ops over the run (a split increase as 'inc_a' / 'inc_b'). `taken` = stitches of this
    round already used by other joins (avoided; a clash is reported by check())."""
    p = parent.p
    j = int(j)
    n = p.counts[j - 1]
    toks = stitch_tokens(list(p.rounds[j - 1].ops))
    # a run may pass the round-start marker: a spiral has no step there, the stitches after it are the first ones
    # of the next round, right beside the last ones of this round
    nxt = stitch_tokens(list(p.rounds[j].ops)) if j < p.N and p.rounds[j].loop in ("both", "blo", "flo") else None
    k = int(min(pairs, n))
    th = parent.theta(az) % (2 * np.pi)
    wc = _w_of_theta(j, n, th)                                    # the wanted centre, in produced stitches
    taken = taken or {}
    best = None
    for s in range(n):
        b = max(0, s + k - n)
        if b and (nxt is None or b > len(nxt)):
            continue
        cand = {j: set(range(s, min(s + k, n))), j + 1: set(range(b))}
        clash = any(cand[r] & set(taken.get(r, ())) for r in cand)
        centre = (s + (k - 1) / 2) % n
        if pair and not clash:
            # the left copy of a pair: its mirror image (the right copy) must find its own stitches
            az_m = -float(parent.az_of_theta(stitch_theta(j, n, centre)))
            wm = _w_of_theta(j, n, parent.theta(az_m) % (2 * np.pi))
            sm = int(round(wm - (k - 1) / 2)) % n
            mine = {(s + i) % n for i in range(k)}
            clash = bool(mine & {(sm + i) % n for i in range(k)})
        d = abs((centre - wc + n / 2) % n - n / 2)
        # no clash with another join first, then the nearest centre (a quarter stitch's preference for a run that
        # stays in one round: one line of the pattern instead of two)
        key = (clash, round(d + (0.25 if b else 0.0), 6), s)
        if best is None or key < best[0]:
            best = (key, s, centre, d)
    key, s, centre, d = best
    b = max(0, s + k - n)
    th_c = stitch_theta(j, n, centre)
    az_c = float((parent.az_of_theta(th_c) + 180) % 360 - 180)
    return {"round": j, "st_start": int(s), "k": k, "pairs": int(pairs), "n": int(n), "az": round(az_c, 4),
            "ops": run_tokens(toks[s:s + k - b]), "wrap": b, "ops_next": run_tokens(nxt[:b]) if b else [],
            "snap_st": float(d), "clash": bool(key[0]) or pairs > n}


def plain_round(parent, j):
    """Can stitches of round j be worked through a piece's edge? (plain sc rounds: not ribbing, shells, popcorns)"""
    return 1 <= j <= parent.p.N and parent.p.rounds[j - 1].loop in ("both", "blo", "flo")


def plan_near(parent, j, az, pairs, taken=None, pair=False, reach=1):
    """plan() in round j, or when the run does not fit there (too few free stitches, a popcorn / shell round) in the
    nearest round within `reach` that it fits (the wider one first). None when no round fits."""
    order = [j]
    for dj in range(1, reach + 1):
        cand = [j + dj, j - dj]
        cand.sort(key=lambda r: -parent.p.counts[r - 1] if 1 <= r <= parent.p.N else 0)
        order += cand
    for r in order:
        if r < 2 or r > parent.p.N or not plain_round(parent, r):
            continue
        jn = plan(parent, r, az, pairs, taken, pair=pair)
        if not jn["clash"]:
            return jn
    return None


def run_stitches(jn):
    """The parent stitches a join uses: {round: {stitch index (0-based, working order from the marker)}} - the run
    in its round and, when it passes the marker, the first stitches of the next round."""
    j, s, k, n, b = jn["round"], jn["st_start"], jn["k"], jn["n"], jn.get("wrap", 0)
    return {j: set(range(s, s + k - b)), j + 1: set(range(b))}


def join_round(parent, rnd):
    """The parent round a piece placed at `rnd` is crocheted into (its edge becomes part of that round)."""
    N = parent.p.N
    return int(min(max(int(math.floor(float(rnd) + 0.5)), 2), N))


# ---------------------------------------------------------------------------------------------------------------
# checks and measures

def joins(toy):
    """Joined instances: [(instance, parent instance, entry)]"""
    out = []
    for e in toy.entries:
        jn = e.get("_join")
        if jn and e["name"] in toy.inst:
            out.append((toy.inst[e["name"]], toy.inst[jn["parent"]], e))
    return out


def check(toy):
    """Stitch-count correctness of every join: [] when all is right, else readable problems.

    - the piece's opening is an even number of stitches (k against k when pressed flat);
    - the joined run makes exactly one stitch per pair of the edge;
    - the parent round still works its previous count and makes its own (the joined stitches replace plain ones);
    - two joins in one round never share a stitch; a join is never in a special round (ribbing, shells, popcorns) or
      in the unworked window of a hood;
    - a mirrored pair is joined in the same round, the two runs mirror images within one stitch."""
    from .rounds import check as rcheck
    errs = []
    used = {}
    for inst, parent, e in joins(toy):
        jn = e["_join"]
        c = inst.p.counts[-1]
        if c % 2:
            errs.append(f"note: {inst.name}: its opening has {c} st (odd): one stitch is worked through the single "
                        "edge stitch at the end of the fold")
        if jn["k"] != jn["pairs"]:
            errs.append(f"{inst.name}: {jn['k']} stitches made through {jn['pairs']} pairs of the edge in "
                        f"{parent.name} R{jn['round']}")
        r = parent.p.rounds[jn["round"] - 1]
        if r.loop not in ("both", "blo", "flo"):
            errs.append(f"{inst.name}: joined in a {r.loop} round of {parent.name} (R{jn['round']})")
        op = getattr(parent.p, "opening", None)
        if op is not None and jn["round"] > op["from"]:
            errs.append(f"{inst.name}: joined in a turned row of {parent.name}'s opening (R{jn['round']})")
        for r_, idx in run_stitches(jn).items():
            if not idx:
                continue
            key = (parent.name, r_)
            for other, oidx in used.get(key, []):
                if idx & oidx:
                    errs.append(f"{inst.name} and {other}: share {len(idx & oidx)} stitch(es) of {parent.name} "
                                f"R{r_}")
            used.setdefault(key, []).append((inst.name, idx))
        if jn.get("wrap") and jn["round"] + 1 > parent.p.N:
            errs.append(f"{inst.name}: the run passes the marker of {parent.name}'s last round")
    for pid in {parent.p.id for _, parent, _ in joins(toy)}:
        errs += [f"{pid}: {x}" for x in rcheck(toy.defs[pid].rounds)]
    # pairs: same round, mirrored runs
    by_name = {inst.name: (inst, e) for inst, _, e in joins(toy)}
    for a, b in getattr(toy, "pairs", {}).values():
        if a in by_name and b in by_name:
            ja, jb = by_name[a][1]["_join"], by_name[b][1]["_join"]
            if ja["round"] != jb["round"]:
                errs.append(f"{a} / {b}: joined in different rounds ({ja['round']} / {jb['round']})")
            elif abs(ja["az"] + jb["az"]) > 360.0 / ja["n"] * 1.01:
                errs.append(f"{a} / {b}: not mirror images (centres at {ja['az']:.1f} / {jb['az']:.1f} deg)")
    return errs


def _seg_dist(X, A, B):
    """Distance of points X to the polyline through A (k points) closed by the segment to B (end points)."""
    pts = np.vstack([A, B]) if B is not None else A
    d = np.full(len(X), np.inf)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        t = np.clip(((X - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(X - (a + t[:, None] * ab), axis=1))
    return d


def measure(toy):
    """Geometry of every join, in mm: how far the 3D piece's edge is from the stitches it is crocheted into.

    - pos_mean / pos_max: distance of the piece's edge (its last round, sampled all round: both layers of the pressed
      opening) from the joined run of the parent round (the line through those stitches);
    - len_piece / len_run / len_err: the pressed edge's length along the seam vs the joined run's length (both should
      be k stitches); len_err_pct relative to the run;
    - centre_err: the edge's centre vs the run's centre (along the surface, mm);
    - rounds_off: how far the edge's centre is from round j, in rounds."""
    out = {}
    for inst, parent, e in joins(toy):
        jn = e["_join"]
        j, n, s, k = jn["round"], jn["n"], jn["st_start"], jn["k"]
        ws = s - 0.5 + np.linspace(0.0, float(k), 8 * k + 1)     # the run, from its first to its last stitch's edge
        th = stitch_theta(j, n, ws)
        t = np.full(len(th), float(j) - 0.5)
        run = parent.base(t, th)
        ths = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        rim = inst.base(np.full_like(ths, float(inst.p.N)), ths)
        d = _seg_dist(rim, run, None)
        len_run = float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())
        # how much of the round the edge covers (the run and 3 st beyond each end), measured along the round, and
        # where its middle is relative to the run's middle
        wx = s - 3.5 + np.linspace(0.0, float(k + 6), 8 * (k + 6) + 1)
        line = parent.base(np.full(len(wx), float(j) - 0.5), stitch_theta(j, n, wx))
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
        arc = arc - arc[24]                                       # 0 at the run's first edge (3 st = 24 samples)
        near = np.argmin(np.linalg.norm(rim[:, None, :] - line[None, :, :], axis=2), axis=1)
        a0, a1 = float(arc[near].min()), float(arc[near].max())
        len_piece = a1 - a0
        out[inst.name] = {
            "parent": parent.name, "round": j, "stitches": [s, s + k - 1], "k": k, "pairs": jn["pairs"],
            "pos_mean": round(float(d.mean()), 2), "pos_max": round(float(d.max()), 2),
            "len_piece": round(len_piece, 1), "len_run": round(len_run, 1),
            "len_err": round(len_piece - len_run, 1),
            "len_err_pct": round(100 * (len_piece - len_run) / max(len_run, 1e-6), 1),
            "centre_err": round(abs((a0 + a1) / 2 - len_run / 2), 2),
            "snap_st": jn.get("snap_st", 0.0), "moved_rounds": jn.get("moved_rounds", 0.0),
        }
    return out


def compare(toy_sewn, toy_ns):
    """How different the no-sew toy is from its sewn design, per joined piece (mm): its tip and its centre moved,
    and the mean distance between the two surfaces (each piece sampled, symmetric nearest-point)."""
    from scipy.spatial import cKDTree
    out = {}
    for inst, _, e in joins(toy_ns):
        a = toy_sewn.inst.get(inst.name)
        if a is None:
            continue
        Pa, _, _ = a.samples(per_round=2, per_stitch=1.0)
        Pb, _, _ = inst.samples(per_round=2, per_stitch=1.0)
        da, _ = cKDTree(Pb).query(Pa)
        db, _ = cKDTree(Pa).query(Pb)
        tip_a = a.base(np.array([0.02]), np.array([0.0]))[0]
        tip_b = inst.base(np.array([0.02]), np.array([0.0]))[0]
        out[inst.name] = {"surface_mm": round(float((da.mean() + db.mean()) / 2), 2),
                          "surface_max_mm": round(float(max(da.max(), db.max())), 1),
                          "tip_mm": round(float(np.linalg.norm(tip_a - tip_b)), 1),
                          "centre_mm": round(float(np.linalg.norm(Pa.mean(0) - Pb.mean(0))), 1)}
    return out


def roundtrip(toy):
    """The written pattern rebuilt (PATTERN_EN -> reverse.py -> 3D) vs the model, per piece (mm, symmetric mean
    nearest-point distance) and per join (the same round and stitches?): the test that a crocheter reading the
    pattern makes what the picture shows."""
    from scipy.spatial import cKDTree
    from .pattern_text import pattern_model, render_markdown
    from .reverse import to_spec, build_toy, _clean_spec
    log = []
    text = render_markdown(pattern_model(toy), "en")
    s2 = to_spec(text, log)
    if isinstance(s2, tuple):
        s2 = s2[0]
    t2 = build_toy(_clean_spec(s2))
    pieces, joins_ok = {}, {}
    # (the reader names pieces from the pattern's words - a "hind foot" comes back as hind_leg - so instances are
    # matched in assembly order when the names differ)
    alias = dict(zip(toy.inst, t2.inst)) if len(toy.inst) == len(t2.inst) else {}
    for name, a in toy.inst.items():
        b = t2.inst.get(name) or t2.inst.get(alias.get(name, ""))
        if b is None:
            pieces[name] = None
            continue
        Pa, _, _ = a.samples(per_round=2, per_stitch=1.0)
        Pb, _, _ = b.samples(per_round=2, per_stitch=1.0)
        da, _ = cKDTree(Pb).query(Pa)
        db, _ = cKDTree(Pa).query(Pb)
        pieces[name] = round(float((da.mean() + db.mean()) / 2), 2)
    j2 = {e["name"]: e["_join"] for e in t2.entries if e.get("_join")}
    for inst, _, e in joins(toy):
        a, b = e["_join"], j2.get(inst.name) or j2.get(alias.get(inst.name, ""))
        joins_ok[inst.name] = bool(b) and (a["round"], a["st_start"], a["k"]) == (b["round"], b["st_start"], b["k"])
    vals = [v for v in pieces.values() if v is not None]
    return {"mean_mm": round(float(np.mean(vals)), 2) if vals else None,
            "max_mm": round(float(max(vals)), 2) if vals else None,
            "missing": [k for k, v in pieces.items() if v is None], "pieces": pieces, "joins_same": joins_ok}


def summary(meas, cmp_=None):
    """One line per toy: worst and mean edge-to-stitch distance, length error, change from the sewn design."""
    if not meas:
        return {"joins": 0}
    v = list(meas.values())
    s = {"joins": len(v),
         "pos_mean": round(float(np.mean([x["pos_mean"] for x in v])), 2),
         "pos_max": round(float(max(x["pos_max"] for x in v)), 2),
         "len_err_pct_abs": round(float(np.mean([abs(x["len_err_pct"]) for x in v])), 1),
         "centre_err": round(float(np.mean([x["centre_err"] for x in v])), 2)}
    if cmp_:
        c = list(cmp_.values())
        s["vs_sewn_surface_mm"] = round(float(np.mean([x["surface_mm"] for x in c])), 2)
        s["vs_sewn_tip_mm"] = round(float(np.mean([x["tip_mm"] for x in c])), 1)
    return s


# ---------------------------------------------------------------------------------------------------------------
# CLI

def compare_sheet(out, views=("front", "34", "side", "back"), size=360):
    """compare.png: the sewn build's views on top, the no-sew build's below (same angles)."""
    import os
    from PIL import Image, ImageDraw
    rows = []
    for label in ("sewn", "nosew"):
        ims = []
        for v in views:
            p = os.path.join(out, label, f"view_{v}.png")
            if os.path.exists(p):
                ims.append(Image.open(p).convert("RGB").resize((size, size)))
        rows.append((label, ims))
    W = size * max(len(r[1]) for r in rows)
    sheet_ = Image.new("RGB", (W, size * len(rows)), "white")
    d = ImageDraw.Draw(sheet_)
    for y, (label, ims) in enumerate(rows):
        for x, im in enumerate(ims):
            sheet_.paste(im, (x * size, y * size))
        d.text((8, y * size + 6), "sewn" if label == "sewn" else "no-sew", fill=(200, 30, 30))
    sheet_.save(os.path.join(out, "compare.png"))


def silhouette_iou(out, views=("front", "34", "side", "back")):
    """Overlap (IoU) of the sewn and no-sew silhouettes per view (same camera): 1 = the same outline."""
    import os
    from PIL import Image
    from .fitview import toy_mask
    res = {}
    for v in views:
        a, b = (os.path.join(out, x, f"view_{v}.png") for x in ("sewn", "nosew"))
        if os.path.exists(a) and os.path.exists(b):
            ma = toy_mask(Image.open(a).convert("RGB")); mb = toy_mask(Image.open(b).convert("RGB"))
            res[v] = round(float((ma & mb).sum() / max((ma | mb).sum(), 1)), 3)
    return res


def main(argv=None):
    import argparse
    import json
    import os
    from .toy import Toy

    ap = argparse.ArgumentParser(description="Build a toy sewn and no-sew; check and measure the joins")
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--all", action="store_true", help="join every structurally possible piece, not only ears / "
                                                        "limbs / tails")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    with open(a.spec, encoding="utf-8-sig") as f:
        spec = json.load(f)
    ns = convert(spec, everything=a.all)
    kept = ns.pop("_nosew_kept", [])
    t0 = Toy(spec)
    t1 = Toy(ns)
    errs = check(t1)
    meas = measure(t1)
    cmp_ = compare(t0, t1)
    rep = {"spec": a.spec, "summary": summary(meas, cmp_), "check": errs, "joins": meas, "vs_sewn": cmp_,
           "kept_sewn": kept, "flags": [f for f in t1.flags if f not in t0.flags]}
    try:
        rt = roundtrip(t1)
        rep["roundtrip"] = rt
        rep["summary"]["roundtrip_mm"] = [rt["mean_mm"], rt["max_mm"]]
        rep["summary"]["roundtrip_joins_same"] = sum(rt["joins_same"].values()), len(rt["joins_same"])
    except Exception as ex:  # noqa: BLE001 - reported, never a build failure
        rep["roundtrip"] = {"error": repr(ex)}
    with open(os.path.join(a.out, "nosew.json"), "w", encoding="utf8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    ns_path = os.path.join(a.out, "spec_nosew.json")
    with open(ns_path, "w", encoding="utf8") as f:
        json.dump(dict(ns, construction="nosew"), f, ensure_ascii=False, indent=1)
    if not a.no_render:
        from .build import build
        build(ns_path, os.path.join(a.out, "nosew"), fast=a.fast)
        build(a.spec, os.path.join(a.out, "sewn"), fast=a.fast)
        compare_sheet(a.out)
        rep["summary"]["iou"] = silhouette_iou(a.out)
        with open(os.path.join(a.out, "nosew.json"), "w", encoding="utf8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep["summary"], ensure_ascii=False))
    for x in errs:
        print("CHECK:", x)
    return rep


if __name__ == "__main__":
    main()
