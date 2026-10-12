"""Toy spec (JSON) -> placed pieces -> one scene of meshes with stitch relief, face details and colour charts.

World frame: X = the toy's left (viewer's right in a front view), Y = up, Z = front (toward the viewer).
Azimuth `az` on a piece: degrees around the piece from its front reference; +90 is toward the toy's left (+X).
A round position `round: r` means the middle of round r; `12.5` is between rounds 12 and 13.
See SPEC.md for the full format.
"""
from __future__ import annotations
import copy
import json
import numpy as np
from scipy.spatial import cKDTree

from .piece import PieceDef, stitch_height, LEAN, _smoothstep

DEFAULT_YARN = {"weight": "DK", "hook_mm": 2.5, "stitch_mm": 4.0, "round_mm": 3.6, "thickness_mm": 1.9}
MAT = {"yarn": 0, "plastic": 1, "thread": 2, "felt": 3}
WORLD_X = np.array([1.0, 0, 0]); WORLD_Y = np.array([0, 1.0, 0]); WORLD_Z = np.array([0, 0, 1.0])


def _norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def hex_rgb(c):
    if isinstance(c, (list, tuple)):
        a = np.array(c, float)
        return a / 255.0 if a.max() > 1.0 else a
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def _az_in(az, rng):
    lo, hi = rng
    if hi - lo >= 360:
        return np.ones(np.shape(az), bool)
    az = (az + 180) % 360 - 180
    lo = (lo + 180) % 360 - 180; hi = (hi + 180) % 360 - 180
    return (az >= lo) & (az <= hi) if lo <= hi else (az >= lo) | (az <= hi)


def _in_polygon(rnd, az, poly):
    """Stitch centres (round number, az array) inside a polygon of [round, az] points.

    Vertices are taken literally (no wrapping): a shape across the back seam is written with az beyond 180
    (e.g. 150..210). Each stitch is tested at az, az-360 and az+360."""
    poly = np.asarray(poly, float)
    y = float(rnd)
    az = np.asarray(az, float)
    out = np.zeros(np.shape(az), bool)
    n = len(poly)
    for shift in (-360.0, 0.0, 360.0):
        x = az + shift
        inside = np.zeros(np.shape(x), bool)
        for k in range(n):
            y1, x1 = poly[k]; y2, x2 = poly[(k + 1) % n]
            if (y1 > y) != (y2 > y):
                xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                inside ^= x < xc
        out |= inside
    return out


def _surface_project(parent, X):
    """Nearest points on a piece's surface (weighted over 8 samples: smooth) and the normals there."""
    tree, SP, SN, _ = parent.kd()
    dist, idx = tree.query(X, k=8)
    wgt = 1.0 / np.maximum(dist, 1e-3) ** 2
    wgt /= wgt.sum(1, keepdims=True)
    Nw = _norm(np.einsum("ij,ijk->ik", wgt, SN[idx]))
    Pw = np.einsum("ij,ijk->ik", wgt, SP[idx])
    sdv = np.einsum("ij,ij->i", X - Pw, Nw)
    return X - sdv[:, None] * Nw, Nw


def _surface_grid(parent, P, ex, ey, xs, ys, step=0.8):
    """A rectangle of fabric laid on a curved piece without stretching: from P walk along the surface in direction
    ex to each distance xs (signed, mm), then from each of those points along the surface in the transported
    direction ey to each distance ys. Returns points (len(xs), len(ys), 3) and normals."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)

    def walk(Q0, N0, T0, dists):
        """Walk every start point (k, 3) along its tangent T0 to each of the sorted signed `dists`."""
        k = len(Q0)
        outQ = np.zeros((k, len(dists), 3)); outN = np.zeros((k, len(dists), 3))
        outT = np.zeros((k, len(dists), 3))
        for sgn in (1.0, -1.0):
            idx = [i for i, d in enumerate(dists) if (d > 0 if sgn > 0 else d <= 0)]
            idx.sort(key=lambda i: abs(dists[i]))
            Q, N, T = Q0.copy(), N0.copy(), T0 * sgn
            gone = 0.0
            for i in idx:
                target = abs(dists[i])
                while gone < target - 1e-9:
                    h = min(step, target - gone)
                    Q, N = _surface_project(parent, Q + T * h)
                    T = _norm(T - np.einsum("ij,ij->i", T, N)[:, None] * N)
                    gone += h
                outQ[:, i] = Q; outN[:, i] = N; outT[:, i] = T * sgn
        return outQ, outN, outT

    Q0, N0 = _surface_project(parent, np.asarray(P, float)[None])
    T0 = _norm((ex - (ex @ N0[0]) * N0[0])[None])
    Qx, Nx, Tx = walk(Q0, N0, T0, xs)                       # (1, len(xs), 3)
    Qx, Nx, Tx = Qx[0], Nx[0], Tx[0]
    # the rows' direction at each column: square to the strip's length on the surface, on ey's side
    Uy = _norm(np.cross(Nx, Tx))
    flip = np.sign(np.einsum("ij,j->i", Uy, ey)); flip[flip == 0] = 1.0
    Uy = Uy * flip[:, None]
    Q, N, _ = walk(Qx, Nx, Uy, ys)                           # (len(xs), len(ys), 3)
    return Q, N


def is_soft(p):
    """A soft fabric flap that lies on other pieces rather than holding them off: unstuffed, or a pressed ear with a
    thin layer of stuffing (a plump spoon / flattened ear). Open pieces only."""
    if getattr(p, "closed", True) or getattr(p, "motif", None) is not None:
        return False
    if p.stuff == "none":
        return True
    # (a lightly stuffed flattened limb - a wing, a flipper - is not: it keeps its own shape, as it always did; the
    # rule depends on the construction only, so a piece rebuilt from the written pattern behaves the same)
    return p.stuff == "light" and getattr(p, "spoon", 0) > 0


def seam_pinch_rounds(p, e):
    """Over how many rounds a flat seam presses the opening flat. A stuffed limb (arm, leg, tail) is stuffed up
    to just below the seam, so it flattens gradually over ~2.4 rounds (a soft rounded shoulder, not a box with a
    flared lid); an unstuffed piece (wing, ear) is pressed right at the edge (1.3 rounds)."""
    if "pinch_rounds" in e:
        return float(e["pinch_rounds"])
    return 2.4 if getattr(p, "stuff", "none") != "none" else 1.3


def _inner_mask(inst, i, n, az, th, z):
    """Stitches of round i that form the inside of an ear (zone {"inner": true}), so the inner colour has the
    ear's own outline (a triangle in a pointed ear, a long drop in a bunny ear, a disc in a round ear) with a rim
    of the ear's colour all round - designers work the front part of each round in the inner colour.
    - dish ears: the front face is the rounds up to the first widest round; the inner disc is those rounds minus
      `margin_rounds` (default 1) next to the fold.
    - spoon ears (pointed / long / floppy presets): the hollow of the pressed cone - the front wall between the
      two side folds - all of it (`width` default 1), `margin_st` (default 1) stitches short of each fold: the
      sunk-in inside framed by the rim.
    - flattened or plain ears (no hollow): measured on the placed piece. In each round the stitches on the front
      face (|az| < 90) span a width across the ear; the inner colour takes the middle `width` (default 0.62) of
      it, so it widens and narrows as the ear does, and never the `margin_st` stitches next to each fold.
    `from` (default 3: the tip stays in the ear colour) is the first round that may take the inner colour."""
    p = inst.p
    if i < int(z.get("from", 3 if not getattr(p, "dish", 0) else 1)):
        return np.zeros(n, bool)
    if getattr(p, "dish", 0) > 0:
        last = int(round(p.dish_tw)) - int(z.get("margin_rounds", 1))
        return np.full(n, i <= last)
    a = np.abs((np.asarray(az, float) + 180) % 360 - 180)
    if getattr(p, "spoon", 0) > 0:
        # a spoon ear has a real hollow: its front wall between the two side folds. The inner colour fills the
        # whole hollow (`width` 1 by default) and stops `margin_st` stitches short of each fold, so it reads as
        # the sunk-in inside of the ear framed by the rim
        W, g, _ = p.spoon_dims_from_r(float(p.r_t(i - 0.5)))
        Ls = max(2 * (float(W) - float(g) / 2), 0.0)
        half = 180.0 * Ls / max(2 * Ls + np.pi * float(g), 1e-6)
        hw = half * float(z.get("width", 1.0)) - float(z.get("margin_st", 1)) * 360.0 / n
        return a <= hw if hw >= 0.5 * 360.0 / n else np.zeros(n, bool)
    front = a <= 90.0 - float(z.get("margin_st", 1)) * 360.0 / n
    if not front.any():
        return front
    P = inst.base(np.full(n, i - 0.5), np.asarray(th, float))
    y = (P - inst.origin) @ inst.R[:, 1]                 # across the ear (the front reference is local x)
    face = a <= 90.0
    lo, hi = float(y[face].min()), float(y[face].max())
    c, w = (lo + hi) / 2, (hi - lo) / 2
    return front & (np.abs(y - c) <= float(z.get("width", 0.62)) * w)


class Instance:
    """A piece placed in the world (mirrored pairs are two instances of one PieceDef)."""

    def __init__(self, pdef: PieceDef, name: str, mirror: bool = False):
        self.p = pdef
        self.name = name
        self.mirror = mirror
        self.origin = np.zeros(3)
        self.R = np.eye(3)            # columns e1, e2, e3
        self.az_sign = 1.0
        self.corrections = []         # (t_from, span, theta_grid, vectors)
        self.placed = False
        self.flags = []
        self.app = None               # (parent, lift, pad): sewn flat onto the parent (applique)
        self.lp = {}                  # per-instance local_points options (bend direction / free end)
        self.pair_sign = None         # for the right copy of a pair: the left copy's az sign

    # frame -----------------------------------------------------------------------------------------------
    def set_frame(self, e3, ref=None):
        e3 = _norm(e3)
        if ref is None:
            # front reference: the toy's front, blended smoothly into "up" as the axis turns toward the front or
            # back (a muzzle, a leg pointing forward), so a small change of direction never flips az 0
            w = float(np.clip((abs(e3 @ WORLD_Z) - 0.6) / 0.35, 0.0, 1.0))
            w = w * w * (3 - 2 * w)
            ref = (1 - w) * WORLD_Z + w * WORLD_Y
        ref = _norm(ref)
        e1 = ref - (ref @ e3) * e3
        # a reference (nearly) along the axis says nothing: fall back to up, then the toy's front, then its side
        # (an applique centred on a top pole used to get a frame from rounding noise)
        for fb in (WORLD_Y, WORLD_Z, WORLD_X):
            if np.linalg.norm(e1) >= 1e-3:
                break
            e1 = fb - (fb @ e3) * e3
        e1 = _norm(e1)
        e2 = np.cross(e3, e1)
        self.R = np.stack([e1, e2, e3], axis=1)
        side = e2 @ WORLD_X
        if abs(side) < 0.3:
            side = -(e2 @ WORLD_Y) if abs(e2 @ WORLD_Y) > 0.3 else 1.0
        self.az_sign = 1.0 if side >= 0 else -1.0
        if self.mirror and self.pair_sign is not None:
            # the right-hand copy of a pair mirrors its left copy exactly: same az -> mirror-image point
            self.az_sign = -self.pair_sign
        elif self.mirror:
            self.az_sign = -self.az_sign

    def theta(self, az_deg):
        return self.az_sign * np.radians(az_deg)

    def az_of_theta(self, th):
        return np.degrees(th) * self.az_sign

    def to_world(self, local):
        return local @ self.R.T + self.origin

    def t_of_round(self, rnd):
        if rnd == "end":
            return float(self.p.M)
        if rnd == "start":
            return 0.0
        return float(rnd) - 0.5

    # surface ---------------------------------------------------------------------------------------------
    def base(self, t, th, with_corr=True):
        t = np.asarray(t, float); th = np.asarray(th, float)
        if self.app is not None:
            parent, lift, pad = self.app
            motif = getattr(self.p, "motif", None) is not None
            if motif:
                x, y = self.p.flat_points_m(t, th, mirror=self.mirror)
            else:
                x, y = self.p.flat_points(t, th)
            if getattr(self, "drape", None) is not None:
                return self._draped(x, y, t, th, lift, pad)
            X = self.origin + x[..., None] * self.R[:, 0] + y[..., None] * self.R[:, 1]
            shp = X.shape
            Xf = X.reshape(-1, 3)
            for _ in range(2):
                sd, Q, NQ = parent.signed_distance(Xf)
                Xf = Xf - sd[:, None] * NQ
            lift_v = lift
            if pad:
                # a padded applique (a stuffed thigh, a puffy cheek): a dome over the parent
                if motif:
                    sn = self.p.pad_sn(t, th)
                else:
                    sn = np.clip(self.p._arc(np.clip(t, 0, self.p.M)) / max(float(self.p._arc(self.p.M)), 1e-6), 0, 1)
                lift_v = lift + pad * (1 - sn.reshape(-1) ** 2) ** 0.6
            Xf = Xf + NQ * np.reshape(lift_v, (-1, 1))
            return Xf.reshape(shp)
        P = self.to_world(self.p.local_points(t, th, **self.lp))
        if getattr(self, "cup", 0.0):
            rr = (self.p.r_t(t) / max(float(self.p.rb.max()), 1e-6)) ** 2
            P = P + rr[..., None] * self.cup * self.edge_face
        if with_corr and self.corrections:
            P = P + self._corr(t, th)
        if with_corr and getattr(self, "push", None) is not None:
            P = P + self._push(t, th)
        return P

    def _push(self, t, th):
        """Offsets of a piece laid over its parent (a collar resting on the shoulders, its ruffles), bilinear on a
        (t, theta) grid."""
        tg, ng, vec = self.push
        ti = np.clip(np.asarray(t, float) / tg[-1] * (len(tg) - 1), 0, len(tg) - 1.0001)
        t0 = np.floor(ti).astype(int); ft = (ti - t0)[..., None]
        if getattr(self, "push_open", False):
            # a split piece (turned rows, a C): theta 0 and 2*pi are the two edges of the gap, not one seam - the
            # grid includes both ends and is never interpolated across the gap
            ai = np.clip(np.asarray(th, float), 0, 2 * np.pi) / (2 * np.pi) * (ng - 1)
            ai = np.clip(ai, 0, ng - 1.0001)
            a0 = np.floor(ai).astype(int); fa = (ai - a0)[..., None]; a1 = a0 + 1
        else:
            ai = np.mod(np.asarray(th, float), 2 * np.pi) / (2 * np.pi) * ng
            a0 = np.floor(ai).astype(int) % ng; fa = (ai - np.floor(ai))[..., None]; a1 = (a0 + 1) % ng
        v = (vec[t0, a0] * (1 - fa) + vec[t0, a1] * fa) * (1 - ft) + (vec[t0 + 1, a0] * (1 - fa) + vec[t0 + 1, a1] * fa) * ft
        return v

    def _draped(self, x, y, t, th, lift, pad):
        """A big flat piece laid over the parent like cloth (a kerchief): its up-coordinate x is walked along the
        parent's length through the anchor, then its across-coordinate y along the parent's round reached there,
        so every row keeps its length on the curved surface (opt-in: entry key `drape`)."""
        parent = self.app[0]
        d = self.drape
        shp = np.shape(x)
        x = np.reshape(x, -1).astype(float); y = np.reshape(y, -1).astype(float)
        if "off" in d:
            y = y + d["off"][0]; x = x + d["off"][1]
        tp = np.interp(x * d["xsign"], d["arc_t"], d["tg"])
        ri = np.clip(np.interp(tp, d["tg"], np.arange(len(d["tg"]), dtype=float)), 0, len(d["tg"]) - 1.001)
        r0 = np.floor(ri).astype(int); f = ri - r0
        th_p = np.empty_like(x)
        for k in np.unique(r0):
            m = r0 == k
            a0 = np.interp(y[m] * d["ysign"], d["arc_y"][k], d["thg"])
            a1 = np.interp(y[m] * d["ysign"], d["arc_y"][k + 1], d["thg"])
            th_p[m] = a0 * (1 - f[m]) + a1 * f[m]
        P = parent.base(tp, th_p)
        N = parent.base_normal(tp, th_p)
        lift_v = lift
        if pad:
            sn = self.p.pad_sn(t, th) if getattr(self.p, "motif", None) is not None else 0.0
            lift_v = lift + pad * (1 - np.reshape(sn, -1) ** 2) ** 0.6
        P = P + N * np.reshape(lift_v, (-1, 1))
        return P.reshape(shp + (3,))

    def _setup_drape(self, parent, t0, th0):
        """Tables for _draped: arc length along the parent's length at the anchor's angle (by t), and along each
        round (by theta, one row per t)."""
        M = parent.p.M
        tg = np.linspace(0.02, M - 0.02, 160)
        P = parent.base(tg, np.full_like(tg, th0))
        arc_t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        arc_t = arc_t - np.interp(t0, tg, arc_t)
        dt = parent.base(np.array([t0 + 0.05]), np.array([th0]))[0] - parent.base(np.array([t0 - 0.05]), np.array([th0]))[0]
        xsign = 1.0 if dt @ self.R[:, 0] >= 0 else -1.0
        thg = th0 + np.linspace(-np.pi, np.pi, 721)
        T, TH = np.meshgrid(tg, thg, indexing="ij")
        Q = parent.base(T, TH)
        arc_y = np.concatenate([np.zeros((len(tg), 1)), np.cumsum(np.linalg.norm(np.diff(Q, axis=1), axis=2), axis=1)], 1)
        arc_y = arc_y - arc_y[:, 360:361]
        dth = parent.base(np.array([t0]), np.array([th0 + 0.01]))[0] - parent.base(np.array([t0]), np.array([th0 - 0.01]))[0]
        ysign = 1.0 if dth @ self.R[:, 1] >= 0 else -1.0
        self.drape = {"tg": tg, "arc_t": arc_t, "xsign": xsign, "thg": thg, "arc_y": arc_y, "ysign": ysign}

    def _corr(self, t, th):
        out = np.zeros(np.shape(t) + (3,))
        for t0, span, grid, vec in self.corrections:
            w = _smoothstep((t - t0) / span)
            if not np.any(w > 0):
                continue
            x = np.mod(th, 2 * np.pi)
            idx = x / (2 * np.pi) * len(grid)
            i0 = np.floor(idx).astype(int) % len(grid); i1 = (i0 + 1) % len(grid)
            f = (idx - np.floor(idx))[..., None]
            v = vec[i0] * (1 - f) + vec[i1] * f
            out += w[..., None] * v
        return out

    def base_normal(self, t, th):
        eps_t, eps_a = 0.02, 0.01
        tt = np.clip(t, eps_t, self.p.M - eps_t)
        Pt = self.base(tt + eps_t, th) - self.base(tt - eps_t, th)
        Pa = self.base(tt, th + eps_a) - self.base(tt, th - eps_a)
        n = np.cross(Pa, Pt)
        if self.app is not None:
            n = -n                     # a flat piece's start faces away from the parent: flip to outward
        return _norm(n)

    def point(self, rnd, az):
        """World point and outward normal at (round, az)."""
        t = self.t_of_round(rnd)
        if rnd in ("end", "start"):
            P = self.to_world(self.p.anchor(t, **self.lp)[None])[0]
            n = self.R[:, 2] * (1 if rnd == "end" else -1)
            if rnd == "end" and not self.p.closed:
                return P, n
            # closed pole: the surface point itself
            P = self.base(np.array([t]), np.array([0.0]))[0]
            return P, n
        th = self.theta(az)
        P = self.base(np.array([t]), np.array([th]))[0]
        n = self.base_normal(np.array([t]), np.array([th]))[0]
        return P, n

    def samples(self, per_round=3, per_stitch=1.5, t_range=None):
        """Dense base samples with normals, for distance queries."""
        M = self.p.M
        ts = np.linspace(0.02, M - 0.02, max(8, int(M * per_round)))
        if t_range is not None:
            ts = ts[(ts >= t_range[0]) & (ts <= t_range[1])]
        nmax = max(self.p.counts)
        ths = np.linspace(0, 2 * np.pi, max(16, int(nmax * per_stitch)), endpoint=False)
        T, TH = np.meshgrid(ts, ths, indexing="ij")
        P = self.base(T, TH).reshape(-1, 3)
        N = self.base_normal(T, TH).reshape(-1, 3)
        Tt = T.reshape(-1)
        if getattr(self.p, "opening", None) is not None:
            keep = ~self.p.in_opening(T, TH).reshape(-1)       # the unworked front window of a hood is not fabric
            P, N, Tt = P[keep], N[keep], Tt[keep]
        return P, N, Tt

    def kd(self):
        if not hasattr(self, "_kd"):
            P, N, T = self.samples()
            self._kd = (cKDTree(P), P, N, T)
        return self._kd

    def signed_distance(self, X):
        tree, P, N, _ = self.kd()
        d, i = tree.query(X)
        return np.einsum("ij,ij->i", X - P[i], N[i]), P[i], N[i]

    def penetration(self, X):
        """Depth of points inside this piece (<= 0 inside, +inf where the point is clearly outside or the
        nearest-surface test is unreliable: far away, or not behind the surface along its normal)."""
        tree, P, N, _ = self.kd()
        d, i = tree.query(X)
        v = X - P[i]
        sd = np.einsum("ij,ij->i", v, N[i])
        behind = sd < -0.6 * np.maximum(d, 1e-9)
        near = d < 1.1 * self.p.max_radius()
        return np.where(behind & near, sd, np.inf)

    def invalidate(self):
        if hasattr(self, "_kd"):
            del self._kd


class Toy:
    def __init__(self, spec: dict):
        from .presets import expand as expand_presets
        spec = expand_presets(spec)             # opt-in construction presets -> plain keys (no-op without them)
        self.spec = copy.deepcopy(spec)
        self.yarn = {**DEFAULT_YARN, **spec.get("yarn", {})}
        self.colours = {k: hex_rgb(v) for k, v in spec.get("colours", {}).items()}
        self.colours.setdefault("black", hex_rgb("#151313"))
        self.colours.setdefault("white", hex_rgb("#f4f1ea"))
        self.flags = []
        self.defs = {}
        self.cords = {}              # parts that are chain cords (no rounds): id -> definition
        self.cord_meshes = []
        self.flats = {}              # parts worked flat in turned rows: id -> definition
        self.flat_meshes = []
        for p in spec["parts"]:
            if "cord" in p:
                self.cords[p["id"]] = p
                continue
            if "flat" in p:
                self.flats[p["id"]] = p
                continue
            if "motif" in p:
                # a shaped flat piece (triangle, leaf, hexagon, ...): its rows are its geometry (amikit/motif.py)
                from .motif import MotifDef
                self.defs[p["id"]] = MotifDef(p["id"], p, self.yarn)
                self.flags.extend(self.defs[p["id"]].flags)
                continue
            self.defs[p["id"]] = PieceDef(p["id"], p, self.yarn)
            self.flags.extend(f"part {p['id']}: {m}" for m in self.defs[p["id"]].opening_errors)
            if not self.defs[p["id"]].solve_info["converged"]:
                self.flags.append(f"{p['id']}: stuffing solve did not fully converge")
        self.inst: dict[str, Instance] = {}
        self.entries = []            # resolved assembly entries (after mirroring)
        self.extras = []             # face/detail meshes
        self.tints = []              # soft blush etc.
        self.dimples = []
        self._assemble()
        self._rest_soft_pieces()
        self._check_keys()

    KNOWN = {
        "toy": {"name", "title", "title_ru", "yarn", "colours", "colour_names_ru", "parts", "assembly", "face", "notes", "version",
                "subject"},
        "part": {"id", "rounds", "colour", "colours", "end", "stuff", "flatten", "squash", "bend", "oval_axis",
                 "cord", "flat", "pinch", "notes", "name_ru", "name_en", "spoon", "dish", "motif", "split",
                 "opening"},
        "entry": {"part", "on", "at", "attach", "dir", "dir_mode", "sink", "mirror", "name", "name_r", "applique",
                  "pad", "lift", "roll", "front", "avoid", "contact", "max_turn", "conform", "conform_span",
                  "conform_parent", "axis", "pos", "face", "cup", "toward", "bend_toward", "notes", "seam", "seam_line",
                  "pinch", "pinch_rounds", "avoid_toward", "offset_mm", "along", "inset", "seat", "drape",
                  "around", "anchor", "shift", "fall", "sink_dir", "rim_arc", "wear"},
        "face": {"id", "type", "on", "at", "mirror", "size_mm", "aspect", "sink", "iris", "pupil", "colour", "shape",
                 "highlight", "offset_mm", "lift_mm", "role",
                 "colours", "count", "length_mm", "droop", "curl", "spread", "seed", "doubled", "link_mm",
                 "points", "thickness_mm", "width_mm", "height_mm", "width_st", "height_rounds", "strands",
                 "strength", "rotate", "notes"},
    }

    def _check_keys(self):
        """Flag keys the engine does not read (a typo or a key on the wrong object silently does nothing), and
        colour names that are not defined."""
        names = set(self.colours)
        used = []
        for p in self.spec.get("parts", []):
            used.append((p.get("colour"), f"part {p.get('id')}"))
            for z in p.get("colours", []):
                if "runs" in z:
                    used += [(c, f"part {p.get('id')} R{z.get('round')}") for c, _ in z["runs"]]
                    continue
                if "colour" not in z:
                    self.flags.append(f"part {p.get('id')}: a colour zone without 'colour' (ignored)")
                used.append((z.get("colour"), f"part {p.get('id')} zone"))
        for f in self.spec.get("face", []):
            used.append((f.get("colour"), f"face item {f.get('id', f.get('type'))}"))
            for c in f.get("colours", []) or []:
                used.append((c, f"face item {f.get('id', f.get('type'))}"))
        for c, where in used:
            if c is not None and c not in names and not str(c).startswith("#"):
                self.flags.append(f"colour {c!r} used by {where} is not defined in colours")
        def chk(obj, kind, label):
            extra = set(obj) - self.KNOWN[kind]
            if extra:
                # keys that belong one level down: say where they go
                from .motif import MOTIF_KEYS
                hint = ""
                if kind in ("entry", "part") and extra & set(MOTIF_KEYS):
                    hint = f"; {sorted(extra & set(MOTIF_KEYS))} belong inside the part's \"motif\""
                self.flags.append(f"unknown key(s) {sorted(extra)} on {label} (ignored){hint}")
        chk(self.spec, "toy", "the toy")
        for p in self.spec.get("parts", []):
            chk(p, "part", f"part {p.get('id')}")
        for e in self.spec.get("assembly", []):
            chk(e, "entry", f"assembly entry {e.get('name', e.get('part'))}")
        for f in self.spec.get("face", []):
            chk(f, "face", f"face item {f.get('id', f.get('type'))}")

    # assembly --------------------------------------------------------------------------------------------
    def _side(self, on, side):
        """Instance name of a parent entry `on` for a child on the left ('L') / right ('R') / single (None)."""
        if on in self.pairs:
            return self.pairs[on][0 if side != "R" else 1]
        if on in self.inst:
            return on
        if on + "_L" in self.inst:
            return on + ("_R" if side == "R" else "_L")
        raise KeyError(f"parent {on!r} is not placed yet")

    def _assemble(self):
        self.pairs = {}                # entry name (or part id) of a mirrored entry -> (left, right) instance names
        for e in self.spec.get("assembly", []):
            if e.get("mirror"):
                a = dict(e); b = copy.deepcopy(e)
                a["name"] = e.get("name", e["part"] + "_L")
                b["name"] = e.get("name_r", (e["name"] + "_R") if e.get("name") else e["part"] + "_R")
                if b["name"] in self.inst or b["name"] == a["name"]:
                    raise ValueError(f"assembly: instance name {b['name']!r} is used twice; give the entry a unique name")
                b["at"] = dict(e.get("at", {}))
                on_pair = bool(e.get("on")) and (e["on"] in self.pairs or e["on"] + "_L" in self.inst)
                if "az" in b["at"] and not on_pair:
                    b["at"]["az"] = -b["at"]["az"]
                for k in ("dir", "front", "seam_line", "bend_toward", "toward", "face", "avoid_toward", "sink_dir"):
                    if k in b and isinstance(b[k], list):
                        b[k] = [-b[k][0], b[k][1], b[k][2]]
                b["_mirror"] = True
                if "roll" in b:
                    b["roll"] = -b["roll"]
                # a mirrored child of a mirrored parent goes on the matching side
                if on_pair:
                    a["on"] = self._side(e["on"], "L"); b["on"] = self._side(e["on"], "R")
                self._place(a)
                b["_pair_left"] = a["name"]
                self._place(b)
                if e.get("avoid"):
                    self._symmetric_avoid(a, b)
                key = e.get("name", e["part"])
                self.pairs[key] = (a["name"], b["name"])
                if e["part"] != key:
                    self.pairs.setdefault(e["part"], (a["name"], b["name"]))
            else:
                e = dict(e); e.setdefault("name", e["part"])
                if e["name"] in self.inst:
                    raise ValueError(f"assembly: instance name {e['name']!r} is used twice; give the entry a unique name")
                self._place(e)

    def _symmetric_avoid(self, a, b):
        """The pattern describes the left copy of a pair and says "sew the other as its mirror image", so both copies
        must end up mirror images even when `avoid` swung them by different amounts (something sits closer to one
        of them). The copy that turned less is placed again as the mirror image of the one that turned more (both
        then clear what they had to clear)."""
        ia, ib = self.inst.get(a["name"]), self.inst.get(b["name"])
        if ia is None or ib is None:
            return
        ta, tb = getattr(ia, "avoid_turn", 0.0), getattr(ib, "avoid_turn", 0.0)
        if abs(ta - tb) < 0.25:
            return
        src, dst, e_dst = (ia, ib, b) if ta >= tb else (ib, ia, a)
        d = np.array(src.avoid_dir, float); d[0] = -d[0]
        e2 = dict(e_dst, dir=[float(x) for x in d], avoid=False)
        e2.pop("dir_mode", None)
        self.entries = [x for x in self.entries if x is not e_dst]
        del self.inst[e_dst["name"]]
        self._place(e2)
        # the replaced entry keeps its own dict for the writer (same name, avoid now resolved)
        e_dst.clear(); e_dst.update(e2)
        self.flags.append(f"{dst.name}: placed as the mirror image of {src.name} (avoid turned them "
                          f"{max(ta, tb):.0f} vs {min(ta, tb):.0f} deg; a pair is sewn symmetrically)")

    def _place(self, e):
        if e["part"] in self.flats:
            self.entries.append(e)
            self._place_flat(e)
            return
        if e["part"] in self.cords:
            self.entries.append(e)
            self._place_cord(e)
            return
        pdef = self.defs[e["part"]]
        inst = Instance(pdef, e["name"], mirror=e.get("_mirror", False))
        if e.get("_pair_left") in self.inst:
            inst.pair_sign = self.inst[e["_pair_left"]].az_sign
        self.entries.append(e)
        if "on" not in e:
            inst.set_frame(e.get("axis", [0, 1, 0]), e.get("front"))
            inst.origin = np.array(e.get("pos", [0, 0, 0]), float)
            inst.placed = True
            self.inst[inst.name] = inst
            return
        try:
            parent = self.inst[self._side(e["on"], None)]
        except KeyError as ex:
            raise KeyError(f"{e['name']}: {ex}") from ex
        at = e.get("at", {"round": "end"})
        if e.get("anchor") in ("eyes", "eye"):
            # centred between the eyes on this parent (a face mask), or on the eye of its own side (a patch around
            # one eye, mirrored pairs): it follows the eyes when they move
            at = self._eyes_at(e, parent, one=e["anchor"] == "eye")
            e["_at"] = at
        if e.get("around"):
            self._place_around(inst, parent, e)
            return
        rq = at.get("round", "end")
        if isinstance(rq, (int, float)) and not (0.5 <= float(rq) <= parent.p.N + 0.5):
            # a round the parent does not have: the point would collapse onto its pole (inside the piece)
            self.flags.append(f"{e['name']}: at round {rq} but {parent.name} has rounds 1-{parent.p.N} "
                              f"(placed at the nearest end)")
            at = dict(at, round=min(max(float(rq), 1.0), float(parent.p.N)))
            e["at"] = at                      # the pattern states the round actually used
        P, nP = parent.point(at.get("round", "end"), at.get("az", 0.0))
        if e.get("applique"):
            # a flat piece sewn on all round: lies on the parent's surface, centred at the point
            ref = e.get("front", WORLD_Y)
            if e.get("along") == "parent" and at.get("round") not in ("end", "start"):
                # its long axis (the chain of an oval piece) runs along the parent's rounds-direction: an inner ear
                # follows the ear even when the ear leans
                tp = parent.t_of_round(at.get("round"))
                thp = parent.theta(at.get("az", 0.0))
                m = parent.base(np.array([tp + 0.3]), np.array([thp]))[0] - parent.base(np.array([tp - 0.3]), np.array([thp]))[0]
                m = m - (m @ nP) * nP
                if np.linalg.norm(m) > 1e-6:
                    ref = np.cross(_norm(m), nP)
            inst.set_frame(nP, ref)
            if e.get("roll"):
                a = np.radians(e["roll"]); c, s_ = np.cos(a), np.sin(a)
                e1, e2, e3v = inst.R[:, 0], inst.R[:, 1], inst.R[:, 2]
                inst.R = np.stack([c * e1 + s_ * e2, -s_ * e1 + c * e2, e3v], axis=1)
            off = e.get("offset_mm")
            if off and (e.get("anchor") or e.get("drape")):
                # (new keys) the offset is walked onto the surface: the piece is centred on that surface point,
                # so the written position (round, stitches from the line) is the whole placement
                Q = P + float(off[0]) * inst.R[:, 1] * (1.0 if not inst.mirror else -1.0) + float(off[1]) * inst.R[:, 0]
                at = self._locate_on(parent, Q)
                # snapped to what a pattern states: half rounds, tenths of a stitch from the centre line
                r_ = min(max(round(2 * at["round"]) / 2, 1.0), float(parent.p.N))
                n_ = parent.p.counts[min(max(int(r_), 1), parent.p.N) - 1]
                a_ = at["az"]
                k_ = round((abs(a_) if abs(a_) <= 120 else 180 - abs(a_)) / 360 * n_, 1) * 360 / n_
                a_ = (k_ if abs(a_) <= 120 else 180 - k_) * (1 if a_ >= 0 else -1)
                at = {"round": r_, "az": round(a_, 4)}
                e["_at"] = at
                P, nP = parent.point(at["round"], at["az"])
                inst.set_frame(nP, ref)
                if e.get("roll"):
                    a = np.radians(e["roll"]); c, s_ = np.cos(a), np.sin(a)
                    e1, e2, e3v = inst.R[:, 0], inst.R[:, 1], inst.R[:, 2]
                    inst.R = np.stack([c * e1 + s_ * e2, -s_ * e1 + c * e2, e3v], axis=1)
                off = None
            if off:
                P = P + float(off[0]) * inst.R[:, 1] * (1.0 if not inst.mirror else -1.0) + float(off[1]) * inst.R[:, 0]
            inst.origin = P
            inset = float(e.get("inset", 0.0))
            inst.inset = inset                       # sewn in flush: the parent's fabric under it gives way
            inst.app = (parent, float(e.get("lift", 0.6)) - inset, float(e.get("pad", 0.0)))
            if e.get("drape"):
                # laid over the parent's curve like cloth (a kerchief, a big leaf over the back)
                r_ = at.get("round", "end")
                t0 = parent.t_of_round(r_) if r_ not in ("end", "start") else (parent.p.M - 0.3 if r_ == "end" else 0.3)
                th0 = parent.theta(at.get("az", 0.0))
                inst._setup_drape(parent, t0, th0)
                if off:
                    inst.drape["off"] = [float(off[0]) * (1.0 if not inst.mirror else -1.0), float(off[1])]
            inst.placed = True
            self.inst[inst.name] = inst
            return
        if e.get("wear"):
            self._place_worn(inst, parent, P, nP, e)
            return
        attach = e.get("attach", "end")
        d = _norm(e.get("dir", nP))
        if e.get("dir_mode", "world") == "normal":
            d = nP
        e["_nP"] = nP
        if attach == "edge":
            self._place_edge(inst, parent, P, nP, d, e)
            return
        self._frame(inst, d, e)
        t_anchor = float(pdef.M) if attach == "end" else 0.0
        self._seat(inst, parent, P, nP, e)
        if inst.p.bend and inst.p.bend.get("drape") and attach == "end":
            self._drape(inst, P, d)
        inst.placed = True
        if e.get("avoid"):
            self._avoid(inst, parent, P, nP, d, e)
        # an open parent rim sewn to this child (body neck onto a head): the parent's rim follows the child
        if at.get("round") == "end" and not parent.p.closed and e.get("conform_parent", True):
            self._conform(parent, inst, float(parent.p.N), e.get("conform_span", 1.6))
            parent.invalidate()
        self.inst[inst.name] = inst

    def _place_worn(self, inst, parent, P, nP, e):
        """`wear: true`: an open piece put ON its parent like a hat - a hair cap, a hood, a beanie, a helmet, a
        shell. Its magic ring is centred over `at`; it is pulled down along its own axis (`dir`: from its start
        toward its opening, default into the parent) until its inside rests on the parent with `pad` mm of
        stuffing between (default 1 mm: just the fabric). It keeps the shape its rounds make (it is not laid onto
        the parent like a soft flap); where it touches the parent it is sewn on."""
        d = _norm(e.get("dir", -nP))
        self._frame(inst, d, dict(e, attach="start"))
        pad = float(e.get("pad", 1.0))
        clear = pad + 0.5 * float(inst.p.thick)
        p = inst.p
        tg = np.linspace(0.0, p.M, int(p.M * 3) + 1)
        ng = max(48, 2 * max(p.counts))
        ths = 2 * np.pi * np.arange(ng) / ng
        T, TH = np.meshgrid(tg, ths, indexing="ij")
        keep = ~p.in_opening(T, TH).reshape(-1)
        L = float(p.zb.max() - p.zb.min()) + p.max_radius() + 2 * clear + 5.0
        inst.origin = P - inst.R @ p.anchor(0.0, **inst.lp) - d * L
        inst.invalidate()
        X0 = inst.base(T, TH).reshape(-1, 3)[keep]

        def gap(s):
            sd, _, _ = parent.signed_distance(X0 + d * s)
            return float(sd.min())
        # pull it down until it first rests on the parent: coarse steps, then halve the step
        s_max = L + 2 * parent.p.max_radius()
        s, step = 0.0, 1.0
        while s < s_max and gap(s + step) > clear:
            s += step
        lo, hi = s, s + step
        for _ in range(12):
            mid = (lo + hi) / 2
            if gap(mid) > clear:
                lo = mid
            else:
                hi = mid
        if lo >= s_max - step:
            self.flags.append(f"{e['name']}: worn piece never rests on {parent.name} (check dir / counts)")
        inst.origin = inst.origin + d * lo
        inst.invalidate()
        inst.worn = {"pad": pad, "parent": parent.name}
        inst.placed = True
        self.inst[inst.name] = inst

    def _seat(self, inst, parent, P, nP, e):
        """Put an oriented instance on its seam: anchor at P, then seat/conform an open rim or press a closed end in."""
        pdef = inst.p
        attach = e.get("attach", "end")
        t_anchor = float(pdef.M) if attach == "end" else 0.0
        inst.origin = P - inst.R @ pdef.anchor(t_anchor, **inst.lp)
        inst.corrections = []
        inst.invalidate()
        sink = e.get("sink")
        # which way the piece is pressed in: into the parent's surface (default), back along its own axis
        # (`sink_dir: "axis"` - a head on a body's sloping top sinks down into it instead of sliding back off it),
        # or a given world direction
        sd = e.get("sink_dir")
        if isinstance(sd, (list, tuple)):
            mv = _norm(np.array(sd, float))
        elif sd == "axis":
            mv = inst.R[:, 2] if attach == "end" else -inst.R[:, 2]
        else:
            mv = -nP
        inst.sink_vec = mv if sd not in (None, "normal") else None
        if attach == "end" and not pdef.closed:
            arc = e.get("rim_arc")
            self._seat_rim(inst, parent, nP, e.get("seat", "mid"), arc=arc)
            if sink:
                inst.origin = inst.origin + mv * sink
            if e.get("conform", True):
                self._conform(inst, parent, float(pdef.N), e.get("conform_span", 1.6), arc=arc)
        else:
            if sink is None:
                sink = 0.12 * pdef.max_radius()
            inst.origin = inst.origin + mv * sink
        inst.invalidate()

    def _place_cord(self, e):
        """A chain cord (ch N, often with a sl st back along it): a thin yarn tube from the seam along `dir`,
        curling by `curl_deg` toward `toward` (default up) over its length."""
        c = self.cords[e["part"]]["cord"]
        parent = self.inst[self._side(e["on"], None)]
        at = e.get("at", {"round": "end"})
        P, nP = parent.point(at.get("round", "end"), at.get("az", 0.0))
        d = _norm(e.get("dir", nP))
        tw = np.array(e.get("toward", c.get("toward", [0, 1, 0])), float)
        if e.get("_mirror") and "toward" not in e:
            tw[0] = -tw[0]                          # part-level toward; an entry's own was mirrored with the entry
        tw = _norm(tw - (tw @ d) * d) if np.linalg.norm(tw - (tw @ d) * d) > 1e-6 else _norm(np.cross(d, WORLD_X))
        L = float(c.get("length_mm", 40)); curl = np.radians(float(c.get("curl_deg", 0)))
        n = max(16, int(L / 1.0))
        s_ = np.linspace(0, 1, n)
        ang = curl * s_ ** 1.5                      # curls more toward the tip
        steps = (np.cos(ang)[:, None] * d + np.sin(ang)[:, None] * tw) * (L / (n - 1))
        pts = P - nP * 0.5 + np.vstack([[0, 0, 0], np.cumsum(steps[:-1], axis=0)])
        rad = float(c.get("thickness_mm", 2.6)) / 2
        nrm = np.tile(tw, (n, 1))
        tube = tube_mesh(pts, nrm, rad, sides=10)
        # chain links: a bump every ~stitch length along the cord
        link = float(c.get("link_mm", self.yarn["stitch_mm"] * 0.8))
        Ls = tube["UV"][:, 0]
        H = 0.5 + 0.5 * np.cos(2 * np.pi * Ls / link)
        col = self.colours.get(self.cords[e["part"]].get("colour", "black"), hex_rgb("#888888"))
        self.cord_meshes.append({"name": e.get("name", e["part"]), "V": tube["V"], "F": tube["F"],
                                 "C": np.tile(col, (len(tube["V"]), 1)), "UV": tube["UV"], "H": H,
                                 "mat": MAT["yarn"]})

    def _place_flat(self, e):
        """A piece worked flat in turned rows (ch W+1, W sc per row, R rows): a two-layer-thick rectangle.

        Placed on the parent at (round, az) facing outward, `lift` mm off the surface; `roll` turns it;
        `conform: true` lays it along the parent's curve (a scarf, a bib); the part's `pinch` 0..1 gathers its
        middle (a bow tie wrapped with yarn)."""
        pdef = self.flats[e["part"]]
        fl = pdef["flat"]
        W, R = int(fl["width_st"]), int(fl["rows"])
        w, h = float(self.yarn["stitch_mm"]), float(self.yarn["round_mm"]) * 0.95
        parent = self.inst[self._side(e["on"], None)]
        at = e.get("at", {"round": "end"})
        P, n = parent.point(at.get("round", "end"), at.get("az", 0.0))
        if abs(n @ WORLD_Y) < 0.95:
            up = _norm(WORLD_Y - (WORLD_Y @ n) * n)
        else:
            up = _norm(WORLD_Z - (WORLD_Z @ n) * n)
        side = np.cross(up, n)
        rot = np.radians(e.get("roll", 0.0))
        ex = np.cos(rot) * side + np.sin(rot) * up
        ey = -np.sin(rot) * side + np.cos(rot) * up
        ku, kv = 5, 4
        u = np.linspace(0, W, W * ku + 1)
        v = np.linspace(0, R, R * kv + 1)
        U, Vv = np.meshgrid(u, v, indexing="ij")
        X = (U - W / 2) * w
        Y = (Vv - R / 2) * h
        pinch = float(pdef.get("pinch", 0.0))
        if pinch:
            g = np.exp(-(X / (0.12 * W * w)) ** 2)
            Y = Y * (1 - pinch * g)
        fu = np.mod(U, 1.0)
        rowi = np.floor(np.clip(Vv, 0, R - 1e-6))
        fv = np.mod(Vv, 1.0)
        fv = np.where(rowi % 2 == 1, 1 - fv, fv)            # turned rows: every other row shows its back
        Hh = stitch_height(fu, fv)
        thick = 1.1 * float(self.yarn.get("thickness_mm", 1.9))
        lift = float(e.get("lift", 1.0)) + thick
        pts = P + X[..., None] * ex + Y[..., None] * ey
        if e.get("conform"):
            # laid along the parent's curve WITHOUT stretching: every stitch keeps its distance from the centre
            # along the surface (a scarf wraps round a neck as far as its length reaches; it used to be the flat
            # rectangle pressed onto the surface, i.e. shorter than the strip the pattern writes)
            y0 = (v - R / 2) * h                                  # the rows' distances before any pinch
            Q, Nw = _surface_grid(parent, P, ex, ey, X[:, 0], y0)
            if pinch:
                # a gathered middle (a bow): each column keeps its own, shorter, row distances
                Q = np.stack([np.stack([np.interp(Y[i], y0, Q[i, :, c]) for c in range(3)], -1)
                              for i in range(len(Q))])
                Nw = _norm(np.stack([np.stack([np.interp(Y[i], y0, Nw[i, :, c]) for c in range(3)], -1)
                                     for i in range(len(Nw))]))
            base = Q + Nw * lift
            nn = Nw
        else:
            bulge = pinch * 2.5 * (1 - np.exp(-(X / (0.25 * W * w)) ** 2)) if pinch else np.zeros_like(X)
            base = pts + n * (lift + bulge)[..., None]
            nn = np.broadcast_to(n, pts.shape)
        amp = 0.15 * w
        front = base + nn * (amp * (Hh - 0.45))[..., None]
        back = base - nn * (thick + amp * (Hh - 0.45))[..., None]
        nu, nv = U.shape
        ii = np.arange(nu * nv).reshape(nu, nv)
        F = np.concatenate([np.stack([ii[:-1, :-1], ii[1:, :-1], ii[1:, 1:]], -1).reshape(-1, 3),
                            np.stack([ii[:-1, :-1], ii[1:, 1:], ii[:-1, 1:]], -1).reshape(-1, 3)])
        Vf = front.reshape(-1, 3)
        Vb = back.reshape(-1, 3)
        # close the four edges between the layers
        edge = np.concatenate([ii[:, 0], ii[-1, :], ii[::-1, -1], ii[0, ::-1]])
        m = len(Vf)
        side_f = ([[edge[k], edge[k + 1], edge[k + 1] + m] for k in range(len(edge) - 1)]
                  + [[edge[k], edge[k + 1] + m, edge[k] + m] for k in range(len(edge) - 1)])
        V = np.concatenate([Vf, Vb])
        F = np.concatenate([F, F[:, ::-1] + m, np.array(side_f)])
        col = self.colours.get(pdef.get("colour", "black"), hex_rgb("#888888"))
        H2 = np.concatenate([Hh.reshape(-1), Hh.reshape(-1)])
        name = e.get("name", e["part"])
        self.flat_meshes.append({"name": name, "V": V, "F": F, "C": np.tile(col, (len(V), 1)),
                                 "UV": np.zeros((len(V), 2)), "H": H2, "mat": MAT["yarn"]})

    def _frame(self, inst, d, e):
        """Orient an instance extending along d from its seam; roll; bend direction in its own frame."""
        attach = e.get("attach", "end")
        inst.set_frame(-d if attach == "end" else d, e.get("front"))
        if e.get("roll"):
            a = np.radians(e["roll"])
            c, s = np.cos(a), np.sin(a)
            e1, e2, e3v = inst.R[:, 0], inst.R[:, 1], inst.R[:, 2]
            inst.R = np.stack([c * e1 + s * e2, -s * e1 + c * e2, e3v], axis=1)
        inst.lp = {}
        if inst.p.bend:
            tw = e.get("bend_toward", inst.p.bend.get("toward"))
            if tw is not None:
                tw = np.array(tw, float)
                if inst.mirror and "bend_toward" not in e:
                    tw[0] = -tw[0]                   # part-level toward; an entry's own is mirrored with the entry
                v = inst.R.T @ tw
                inst.lp["bend_phi"] = float(np.arctan2(v[1], v[0]))
            if inst.p.bend.get("drape") and (inst.p.spoon > 0 or inst.p.flatten >= 0.3):
                # a pressed ear is a strip of two fabric layers: it folds only across its width (about its weak
                # axis), never sideways in its own plane. Hanging from a seam that is not square to the fold, the
                # soft strip twists just below the seam until its width lies across the fold (piece.py), so the
                # hanging part shows its broad face to the head instead of standing off it at an angle
                # `twist` 0..1: how far it turns (over 2.5 rounds below the seam). 1 = the whole way: the ear lies
                # flat on the head (a dog's ear). 0 = not at all: the seam is pinched so the ear keeps facing where
                # its `front` says, its inside showing (a lop bunny's pink inside). Default: 0 when the inside has
                # its own colour (it is meant to be seen), else 1
                inner = any(z.get("inner") for z in inst.p.zones)
                inst.lp["drape_twist"] = (2.5 * inst.p.h, float(inst.p.bend.get("twist", 0.0 if inner else 1.0)))
            inst.lp["bend_free_start"] = attach == "end"
        if e.get("seam") == "flat" and attach == "end":
            # the opening is pressed flat and sewn as a line (designers' default for arms, ears, tails):
            # the seam line runs along `seam_line` (world; default front-to-back for limbs hanging down or
            # sticking out sideways, side-to-side for pieces pointing up or forward)
            e3 = inst.R[:, 2]
            sl = e.get("seam_line")
            if sl is None:
                nP = e.get("_nP")
                d_out = -e3
                cand = np.cross(nP, d_out) if nP is not None else np.zeros(3)
                if np.linalg.norm(cand) > 0.35:
                    sl = _norm(cand)                 # the piece leans away from the surface: hinge across that lean
                else:
                    sl = WORLD_X                     # sticks straight out (ears on top, legs to the front)
            sl = np.array(sl, float)
            if inst.mirror and "seam_line" in e:
                pass                      # already mirrored with the entry
            sl = sl - (sl @ e3) * e3
            if np.linalg.norm(sl) < 1e-6:
                sl = inst.R[:, 1]
            press = np.cross(e3, _norm(sl))       # squash across the seam line
            v = inst.R.T @ press
            inst.lp["squash"] = float(e.get("pinch", 0.8))
            inst.lp["squash_phi"] = float(np.arctan2(v[1], v[0]))
            inst.lp["squash_rounds"] = seam_pinch_rounds(inst.p, e)

    def _place_edge(self, inst, parent, P, nP, d, e):
        """A flat piece (two circles joined, a folded disc) sewn by a short arc of its edge, standing out along d.

        Its face looks along `face` (default: the toy's front); the edge point nearest the parent is sewn on."""
        face = _norm(e.get("face", WORLD_Z))
        if abs(float(_norm(d) @ face)) > 0.7:
            # the piece stands out almost along the way its face looks: it is seen edge-on from there
            self.flags.append(f"{e['name']}: 'face' is nearly parallel to 'dir' ({float(_norm(d) @ face):.2f}): the "
                              "piece is seen edge-on from where it should face; turn 'face' across 'dir'")
        dd = _norm(d - (d @ face) * face)
        inst.set_frame(-face, dd)                   # the magic-ring side looks along `face`; local x outward
        inst.cup = float(e.get("cup", 0.0))         # mm: the rim comes forward, a cupped ear
        M = inst.p.M
        if getattr(inst.p, "motif", None) is not None:
            # a motif is sewn by its sewn edge (foundation chain, straight side, stem): the middle of that edge is
            # its local origin; the right-hand copy of a pair is the mirror image
            if inst.mirror:
                inst.lp = {"mirror_y": True}
            rim_local = np.zeros(3)
        else:
            t_rim = inst.p.dish_rim_t() if getattr(inst.p, "dish", 0) > 0 else float(inst.p.N)  # a dish: rim middle
            rim_local = inst.p.local_points(np.array([t_rim]), np.array([np.pi]))[0]   # the edge point at -x
        inst.origin = P - inst.R @ rim_local - nP * float(e.get("sink", 0.3 * inst.p.w))
        inst.edge_face = face                       # rendered as two layers: front (with its colours) + plain back
        inst.invalidate()
        inst.placed = True
        self.inst[inst.name] = inst

    def _place_around(self, inst, parent, e):
        """A piece worked outward from a chain ring around the parent (a collar, a snood, a ruffle): the ring sits
        around the parent's round `at.round` (centred on the parent's axis), the rounds grow along `dir` (default:
        down along the parent, over the shoulders); where the fabric meets the parent (or other pieces) it lies on
        them; where a round is longer than the circle it lies on, the extra length makes ruffles."""
        at = e.get("at", {"round": "end"})
        r_ = at.get("round", "end")
        M = parent.p.M
        t = M if r_ == "end" else (0.0 if r_ == "start" else parent.t_of_round(r_))
        a = parent.to_world(parent.p.anchor(min(t + 0.2, M), **parent.lp)[None])[0]
        b = parent.to_world(parent.p.anchor(max(t - 0.2, 0), **parent.lp)[None])[0]
        up = _norm(a - b)
        C = parent.to_world(parent.p.anchor(t, **parent.lp)[None])[0]
        d = _norm(e.get("dir", -up))
        inst.set_frame(d, e.get("front"))
        if e.get("roll"):
            ang = np.radians(e["roll"]); c, s_ = np.cos(ang), np.sin(ang)
            e1, e2, e3v = inst.R[:, 0], inst.R[:, 1], inst.R[:, 2]
            inst.R = np.stack([c * e1 + s_ * e2, -s_ * e1 + c * e2, e3v], axis=1)
        inst.origin = C + up * float(e.get("shift", 0.0))
        inst.placed = True
        self.inst[inst.name] = inst
        # as worked (the counts make a flat collar, a cone or a tube), lying on whatever it meets; `fall: true`: the
        # fabric hangs from the ring and lies along the parent below it (a soft cape-like collar)
        inst.fall = (parent, t) if e.get("fall") else None
        self._lay_on(inst, e)

    def _fall(self, inst, tg, ths, lift):
        """The fabric hangs from its chain ring and lies on the parent below it: every spoke (a line of stitches
        from the ring outward) is laid along the parent's surface, downward from the ring, at its length along the
        fabric; a split (not joined) piece spreads round the parent only as far as each row's stitches reach, the
        rest stays open (the notch)."""
        p = inst.p
        parent, t0 = inst.fall
        # spoke length from the ring to each round boundary
        if getattr(p, "split", False):
            dist = tg * p.h
        else:
            dist = np.asarray(p._arc(np.clip(tg, 0, p.M)), float) - float(p._arc(0.0))
            dist = dist + np.maximum(tg - p.M, 0) * p.h
        # the parent's meridians below the ring, one per spoke direction (toward the parent's start when it was
        # worked bottom-up, toward its end when worked from the top, e.g. a head)
        a = parent.to_world(parent.p.anchor(min(t0 + 0.2, parent.p.M), **parent.lp)[None])[0]
        b = parent.to_world(parent.p.anchor(max(t0 - 0.2, 0.0), **parent.lp)[None])[0]
        if (a - b)[1] >= 0:
            tp_grid = np.linspace(min(t0, parent.p.M - 0.02), 0.02, 160)
        else:
            tp_grid = np.linspace(max(t0, 0.02), parent.p.M - 0.02, 160)

        def walk(thp):
            T_, TH_ = np.meshgrid(tp_grid, thp, indexing="ij")
            Q = parent.base(T_, TH_)
            arc = np.vstack([np.zeros((1, len(thp))), np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=2), axis=0)])
            return arc
        # spoke directions in the parent's own angle
        def parent_theta(th_local):
            v = inst.R @ np.stack([np.cos(th_local), np.sin(th_local), np.zeros_like(th_local)])
            loc = parent.R.T @ v
            return np.arctan2(loc[1], loc[0])
        thp = parent_theta(ths)
        arc = walk(thp)
        out = np.zeros((len(tg), len(ths), 3))
        for i, d in enumerate(dist):
            th_i = ths
            if getattr(p, "split", False):
                # how far round the parent this row reaches at its depth
                t_at = np.array([np.interp(d, arc[:, j], tp_grid) for j in range(len(ths))])
                ring = parent.base(t_at, thp)
                circ = float(np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1).sum())
                n = float(p.n_stitches(np.array([max(tg[i] - 1e-6, 0.0)]))[0])
                alpha = min(2 * np.pi, n * p.w / max(circ, 1e-6) * 2 * np.pi)
                thw = np.where((ths >= 0) & (ths <= 2 * np.pi), ths, np.mod(ths, 2 * np.pi))
                th_i = np.pi + (thw - np.pi) * alpha / (2 * np.pi)
                thp_i = parent_theta(th_i)
                arc_i = walk(thp_i)
            else:
                thp_i, arc_i = thp, arc
            t_i = np.array([np.interp(d, arc_i[:, j], tp_grid) for j in range(len(ths))])
            Q = parent.base(t_i, thp_i)
            N = parent.base_normal(t_i, thp_i)
            out[i] = Q + N * lift
        return out

    def _drape(self, inst, P, d):
        """`bend: {"drape": true}`: an unstuffed piece hanging from its seam (a lop ear). Its centre line is laid
        out step by step from the seam: it leaves along `dir`, turns toward gravity with a limited curvature (the
        fabric's stiffness: `relax_mm` how soon it gives in, `min_radius_mm` the tightest curve) and where the next
        step would go into a stuffed piece already placed (the head) it turns away just enough to lie on it with
        `clear_mm` between the axis and that surface (reached gradually over the fold at the seam). Arc length =
        the piece's length, so nothing stretches; the curve stays in the plane of `dir` and the bend's `toward`;
        a pressed piece twists below the seam (piece.py `drape_twist`)."""
        p = inst.p
        b = p.bend
        bphi = float(inst.lp.get("bend_phi", 0.0))
        bdir = _norm(inst.R @ np.array([np.cos(bphi), np.sin(bphi), 0.0]))
        dd = _norm(d - (d @ bdir) * bdir)
        zt = float(p.z_t(p.M))
        L = zt - float(np.min(p.zb))
        ds = 0.5
        n = int(np.ceil(L / ds)) + 1
        relax = float(b.get("relax_mm", 14.0))
        kmax = 1.0 / float(b.get("min_radius_mm", 7.0))
        clear = float(b.get("clear_mm", 0.5 * p.thick + 2.6))
        fixed = float(b.get("seam_mm", 1.2 * float(p.z_t(p.M) - p.z_t(p.M - 1))))
        g = -WORLD_Y
        phi_g = float(np.arctan2(g @ bdir, g @ dd))
        others = [o for o in self.inst.values() if o is not inst and o.placed and o.app is None
                  and o.p.stuff != "none" and not is_soft(o.p)]

        def sd_at(X):
            """Distance to the nearest stuffed piece (negative inside). The nearest sample's normal says inside
            or outside only near that piece and roughly along its normal: far away, or off to the side of it,
            the point counts as outside (a stuffed ear on the other side of the head is not under this one)."""
            if not others:
                return np.full(len(X), 1e9)
            out = []
            for o in others:
                sd, Q, _ = o.signed_distance(X)
                d = np.linalg.norm(X - Q, axis=1)
                sd = np.where((sd < 0) & (sd > -0.6 * d), d, sd)
                sd = np.where(d > 1.2 * o.p.max_radius() + 5.0, np.maximum(sd, d), sd)
                out.append(sd)
            return np.min(np.stack(out), axis=0)

        s = np.arange(n) * ds
        ramp = 2.5 * clear

        def walk(clear):
            phi = np.zeros(n); ax = np.zeros(n); az = np.zeros(n)
            cur = 0.0
            # the seam sits ON the parent: the centre line starts on its surface and lifts off it gradually as the
            # fabric folds over the seam (half the ear's thickness away within ~2.5 clearances), instead of having to
            # be `clear` away at once - which sent every ear straight out from the head before it could fall (an arch)
            for i in range(1, n):
                if s[i] > fixed:
                    step = np.clip((phi_g - cur) / relax, -kmax, kmax) * ds
                    cur = cur + step
                need = clear * min(1.0, s[i] / ramp)
                # lie on what is under the piece: try the gravity angle, else turn away the least needed
                def pos(a):
                    return (P + (az[i - 1] + ds * np.cos(a)) * dd + (ax[i - 1] + ds * np.sin(a)) * bdir)
                if s[i] > fixed and sd_at(pos(cur)[None])[0] < need:
                    cands = cur + np.radians(np.arange(1, 121, 1.5))[:, None] * np.array([1.0, -1.0])
                    cands = cands.T.reshape(-1)          # +1.5, -1.5, +3, -3 ... interleaved below
                    order = np.argsort(np.abs(cands - cur), kind="stable")
                    cands = cands[order]
                    sds = sd_at(np.stack([pos(a) for a in cands]))
                    ok = np.nonzero(sds >= need)[0]
                    if len(ok):
                        # refine between the blocked angle and the first free one: the fold angle then changes
                        # smoothly with the inputs (1.5-degree candidates made the hanging tip jump by millimetres)
                        lo_a, hi_a = cur, float(cands[ok[0]])
                        for _ in range(7):
                            mid = 0.5 * (lo_a + hi_a)
                            if sd_at(pos(mid)[None])[0] >= need:
                                hi_a = mid
                            else:
                                lo_a = mid
                        cur = hi_a
                phi[i] = cur
                ax[i] = ax[i - 1] + ds * np.sin(cur)
                az[i] = az[i - 1] + ds * np.cos(cur)
            # bending stiffness: smooth the angle along the length (the seam end stays put)
            ker = np.exp(-0.5 * (np.arange(-12, 13) / 5.0) ** 2); ker /= ker.sum()
            ph_s = np.convolve(np.pad(phi, 12, mode="edge"), ker, mode="valid")
            ph_s[s <= fixed] = 0.0
            ax = np.concatenate([[0.0], np.cumsum(ds * np.sin(ph_s[1:]))])
            az = np.concatenate([[0.0], np.cumsum(ds * np.cos(ph_s[1:]))])
            inst.lp["drape_table"] = (s, ph_s, ax, az)
            # the seam end does not move: re-seat with the table in place
            inst.origin = P - inst.R @ p.anchor(float(p.M), **inst.lp)
            inst.invalidate()

        walk(clear)

    def _rest_soft_pieces(self):
        """Unstuffed open pieces (ears, flaps) are soft fabric: where their free part would pass through another
        piece (a folded ear tip bending into the head, a hanging ear cutting through the cheek), it lies on that
        piece's surface instead. Each cross-section moves as a whole (the two layers of a pressed ear stay
        together) by the deepest penetration found in it, smoothed along the piece; nothing moves at the seam."""
        for e in self.entries:
            inst = self.inst.get(e.get("name"))
            if inst is None or not is_soft(inst.p) or inst.app is not None:
                continue
            if getattr(inst, "worn", None) is not None:       # a worn cap / hood keeps the shape its rounds make
                continue
            if getattr(inst, "push", None) is not None or getattr(inst, "edge_face", None) is not None:
                continue
            if "on" not in e or e.get("around") or getattr(inst.p, "motif", None) is not None:
                continue
            p = inst.p
            tg = np.linspace(0, p.M, int(p.M * 3) + 1)
            ng = max(48, 2 * max(p.counts))
            ths = 2 * np.pi * np.arange(ng) / ng
            T, TH = np.meshgrid(tg, ths, indexing="ij")
            P = inst.base(T, TH).reshape(-1, 3)
            lift = 0.5 * p.thick + 0.4
            # only the free part counts: the first ~2.5 rounds from the seam are sewn into / pressed against the
            # parent by construction
            attach_end = e.get("attach", "end") == "end"
            dist = (p.M - tg) if attach_end else tg
            free = np.repeat(dist > 2.5, ng)
            depth = np.zeros(len(tg))
            push_w = np.zeros((len(tg), 3))                    # need-weighted sum of the push directions per row
            for o in self.inst.values():
                if o is inst or not o.placed or o.app is not None:
                    continue
                if (o.p.stuff == "none" or is_soft(o.p)) and getattr(o, "worn", None) is None:
                    continue
                tree, SP, SN, _ = o.kd()
                dd, ii = tree.query(P)
                sd = np.einsum("ij,ij->i", P - SP[ii], SN[ii])
                # points closer than `lift` to o's surface (or inside it), where the nearest sample's normal is
                # trustworthy (the point lies along it) and o is near: the need grows from 0 at `lift`, so a
                # tiny change of the inputs changes the push a little (it used to jump in on/off)
                ok = (np.abs(sd) >= 0.6 * dd - 0.3) & (dd < 1.1 * o.p.max_radius()) & free
                need = np.where(ok & (sd < lift), lift - sd, 0.0).reshape(len(tg), ng)
                if not need.any():
                    continue
                n_g = SN[ii].reshape(len(tg), ng, 3)
                push_w += np.einsum("ij,ijk->ik", need * need, n_g)
                depth = np.maximum(depth, need.max(axis=1))
            if depth.max() < 1.0:
                continue
            # one push per cross-section (the two layers of a pressed ear stay together): as deep as its deepest
            # point, along the need-weighted mean of the normals there (continuous, not the single deepest one)
            dirs = _norm(push_w)
            shift = dirs * depth[:, None]
            # the seam end does not move (sewn on); the free part follows over ~2 rounds; smooth along the piece
            wgt = np.clip((dist - 0.6) / 2.0, 0.0, 1.0)
            wgt = wgt * wgt * (3 - 2 * wgt)
            # a cross-section that must move carries the free part beyond it along (fabric does not stretch):
            # going out from the seam the push never shrinks; its direction blends by size
            order = np.argsort(dist)
            run = np.zeros(3)
            for i in order:
                m0, m1 = float(np.linalg.norm(run)), float(depth[i])
                if m0 + m1 > 1e-9:
                    dv = run + shift[i]
                    nd = np.linalg.norm(dv)
                    run = dv / nd * max(m0, m1) if nd > 1e-9 else run
                shift[i] = run
            # a soft piece bends along its length, it cannot step sideways: the push changes over ~3 rounds
            # (sigma 1.5 rounds; 0.7 rounds left a ledge where the push turned from the head's side to its cheek)
            hw = 14
            ker = np.exp(-0.5 * (np.arange(-hw, hw + 1) / 4.5) ** 2); ker /= ker.sum()
            sm = np.stack([np.convolve(np.pad(shift[:, c], hw, mode="edge"), ker, mode="valid") for c in range(3)], 1)
            vec = (sm * wgt[:, None])[:, None, :].repeat(ng, axis=1)
            inst.push = (tg, ng, vec)
            inst.invalidate()
            self.flags.append(f"{inst.name}: soft piece laid on the pieces it would pass through "
                              f"(up to {depth.max():.1f} mm)")

    def _lay_on(self, inst, e):
        """Push a draped piece out of the pieces under it and add ruffles where its rounds are too long."""
        p = inst.p
        tg = np.linspace(0, p.M, int(p.M * 4) + 1)
        ng = max(96, 3 * max(p.counts))
        split = getattr(p, "split", False)
        ths = np.linspace(0, 2 * np.pi, ng) if split else 2 * np.pi * np.arange(ng) / ng
        inst.push_open = bool(split)
        T, TH = np.meshgrid(tg, ths, indexing="ij")
        inst.push = None
        P0 = inst.base(T, TH)
        lift = 0.6 * p.thick + 0.3
        if getattr(inst, "fall", None) is not None:
            P = self._fall(inst, tg, ths, lift).reshape(-1, 3)
        else:
            P = P0.reshape(-1, 3).copy()
        others = [o for o in self.inst.values() if o.placed and o is not inst and o.app is None
                  and getattr(o, "push", None) is None]
        for _ in range(3):
            moved = False
            for o in others:
                sd, Q, NQ = o.signed_distance(P)
                near = np.linalg.norm(P - Q, axis=1) < np.abs(sd) * 1.6 + 1.5
                hit = near & (sd < lift) & (sd > -o.p.max_radius())
                if np.any(hit):
                    P[hit] += (lift - sd[hit])[:, None] * NQ[hit]
                    moved = True
            if not moved:
                break
        P = P.reshape(P0.shape)
        # ruffles: a round longer than the circle it lies on folds into waves (amplitude from the extra length)
        ext = np.zeros(P.shape)
        circ = np.linalg.norm(np.roll(P, -1, axis=1) - P, axis=2).sum(axis=1)
        nst = p.n_stitches(np.clip(tg - 1e-6, 0, p.M)).astype(float)
        nst[0] = p.counts[0]
        if getattr(p, "shell", None) is not None and p.N >= 2:
            # an edging of shells spends its extra stitches on the scallops themselves (drawn in the fabric's own
            # plane by the shell outline), not on waves: the round lies as flat as the round under it
            nst = np.where(tg > p.M - 1, np.minimum(nst, float(p.counts[-2])), nst)
        if not getattr(p, "split", False):
            nrm = np.cross(np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1),
                           np.vstack([P[1:2] - P[0:1], P[2:] - P[:-2], P[-1:] - P[-2:-1]]))
            nrm = _norm(nrm)
            ref = P.mean(axis=(0, 1))
            sgn = np.sign(np.einsum("ijk,ijk->ij", nrm, P - ref)).mean()
            nrm = nrm * (1.0 if sgn >= 0 else -1.0)
            for i in range(len(tg)):
                ex = nst[i] * p.w / max(circ[i], 1e-6) - 1.0
                if ex > 0.04 and tg[i] > 0.5:
                    k = max(3, int(round(nst[i] / 6)))
                    lam = circ[i] / k
                    A = 0.9 * lam / np.pi * np.sqrt(ex)
                    ramp = min(1.0, (tg[i] - 0.5) / 1.5)
                    ext[i] = nrm[i] * (A * ramp * (0.5 + 0.5 * np.sin(k * ths)))[:, None]
        inst.push = (tg, ng, P - P0 + ext)
        inst.invalidate()

    def _locate_on(self, parent, X):
        """(round, az) of the point of `parent` nearest to X (two passes: coarse grid, then a fine one)."""
        M = parent.p.M
        t0, th0, dt, dth = M / 2, np.pi, M / 2, np.pi
        for k in range(3):
            ts = np.clip(np.linspace(t0 - dt, t0 + dt, 41), 0.02, M - 0.02)
            ths = np.linspace(th0 - dth, th0 + dth, 121)
            T, TH = np.meshgrid(ts, ths, indexing="ij")
            P = parent.base(T, TH)
            i = np.unravel_index(int(np.argmin(np.linalg.norm(P - X, axis=-1))), T.shape)
            t0, th0 = float(T[i]), float(TH[i])
            dt, dth = dt / 8, dth / 12
        az = float((parent.az_of_theta(th0) + 180) % 360 - 180)
        return {"round": round(t0 + 0.5, 2), "az": round(az, 1)}

    def _eyes_at(self, e, parent, one=False):
        """(round, az) between the eyes sewn on `parent` (safety eyes or embroidered eyes); one=True: the eye on
        the entry's side (the toy's left eye, the right one for the right copy of a pair)."""
        pts = []
        for f in self.spec.get("face", []):
            eye = f.get("type") in ("safety_eye", "eye") or (f.get("type") == "satin" and f.get("role") == "eye")
            if not eye or "at" not in f:
                continue
            try:
                host = self._side(f["on"], None)
            except KeyError:
                continue
            if host != parent.name and f["on"] != parent.name:
                continue
            r, a = f["at"]["round"], float(f["at"].get("az", 0.0))
            pts.append((float(r), a))
            if f.get("mirror"):
                pts.append((float(r), -a))
        if not pts:
            raise ValueError(f"{e['name']}: anchor 'eyes' but no eyes on {parent.name}")
        if one:
            left = max(pts, key=lambda q: q[1])
            return {"round": round(left[0], 3), "az": round(-left[1] if e.get("_mirror") else left[1], 3)}
        r = float(np.mean([p[0] for p in pts]))
        a = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians([p[1] for p in pts]))),
                                        np.mean(np.cos(np.radians([p[1] for p in pts]))))))
        return {"round": round(r, 3), "az": round(a, 3)}

    @staticmethod
    def _arc_weight(inst, ths, arc, ramp=20.0):
        """1 on the sewn part of the rim (`rim_arc`: [from, to] az of the piece's last round, degrees), 0 on the
        free part, a short smooth ramp between (the fabric bends where the seam ends)."""
        if not arc:
            return np.ones_like(ths)
        a0, a1 = float(arc[0]), float(arc[1])
        az = np.asarray(inst.az_of_theta(ths), float)
        mid = (a0 + a1) / 2
        half = ((a1 - a0) % 360) / 2 if a1 != a0 else 180.0
        d = np.abs((az - mid + 180) % 360 - 180) - half          # <0 inside the arc
        return np.clip(1 - d / ramp, 0.0, 1.0) ** 2 * (3 - 2 * np.clip(1 - d / ramp, 0.0, 1.0))

    def _seat_rim(self, inst, parent, nP, mode="mid", arc=None):
        """Seat an open rim on the parent. 'mid' (default): between the median and the farthest rim point (a round
        opening on a gently curved parent). 'curve': the median only — a wide flattened edge sewn following a
        strongly curved parent (a big cat ear on a round head); `conform` then lays the edge on the surface."""
        ths = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        if arc:                                  # only the sewn part of the rim sits on the parent
            ths = ths[self._arc_weight(inst, ths, arc) > 0.99]
        t = np.full_like(ths, float(inst.p.N))
        for _ in range(4):
            rim = inst.base(t, ths, with_corr=False)
            sd, _, _ = parent.signed_distance(rim)
            shift = float(np.median(sd)) if mode == "curve" else 0.5 * (np.median(sd) + np.max(sd))
            inst.origin = inst.origin - nP * shift
            if abs(shift) < 0.05:
                break

    def _conform(self, inst, target, t_rim, span, arc=None):
        """Pull the rim of `inst` onto the surface of `target`, blending over `span` rounds (with `arc`, only the
        sewn part of the rim; the rest stays free)."""
        ths = np.linspace(0, 2 * np.pi, 192, endpoint=False)
        t = np.full_like(ths, t_rim)
        rim = inst.base(t, ths, with_corr=True)
        sd, Q, NQ = target.signed_distance(rim)
        # the sewn edge tucks just under the parent's yarn so no lip floats above the seam
        vec = -(sd + 0.25 * inst.p.w)[:, None] * NQ
        if arc:
            vec = vec * self._arc_weight(inst, ths, arc)[:, None]
        inst.corrections.append((t_rim - span, span, ths, vec))
        inst.invalidate()

    def _avoid(self, inst, parent, P, nP, d, e):
        """Swing a limb away from the pieces it would pass through, keeping its attachment point.

        The limb turns about its seam toward the outward horizontal direction (an arm resting against the belly
        rather than inside it); a little contact is allowed, as stuffed pieces press into each other."""
        others = [o for o in self.inst.values() if o.placed]
        attach = e.get("attach", "end")
        if e.get("avoid_toward") is not None:
            out = _norm(np.array(e["avoid_toward"], float))
        else:
            out = nP - (nP @ WORLD_Y) * WORLD_Y
            out = _norm(out if np.linalg.norm(out) > 1e-3 else nP)
        dd = d.copy()
        tol = -float(e.get("contact", 0.3)) * inst.p.max_radius()
        k = 0
        clear = False
        limited = False
        worst = 0.0
        max_turn = np.radians(float(e.get("max_turn", 90)))
        for k in range(25):
            Pts, _, T = inst.samples(per_round=2, per_stitch=0.7)
            far = Pts[T < 0.55 * inst.p.M] if attach == "end" else Pts[T > 0.45 * inst.p.M]  # the free half
            worst = min(min(o.penetration(far).min(), 0.0) for o in others)
            if worst > tol:
                clear = True
                break
            nxt = _norm(dd + 0.06 * out)
            if np.arccos(np.clip(nxt @ _norm(d), -1, 1)) > max_turn:
                limited = True
                break
            dd = nxt
            self._frame(inst, dd, e)
            self._seat(inst, parent, P, nP, e)
        inst.avoid_dir = [float(x) for x in dd]
        inst.avoid_turn = float(np.degrees(np.arccos(np.clip(dd @ _norm(d), -1, 1))))
        if not clear:
            why = f"max_turn {np.degrees(max_turn):.0f} deg reached" if limited else "after turning away"
            self.flags.append(f"{inst.name}: presses {-worst:.1f} mm into other pieces ({why}; avoid)")
        elif k:
            ang = np.degrees(np.arccos(np.clip(dd @ _norm(d), -1, 1)))
            self.flags.append(f"{inst.name}: turned {ang:.0f} deg away from other pieces (avoid)")

    # colours ---------------------------------------------------------------------------------------------
    def chart(self, inst: Instance):
        """Colour name of every output stitch, per round (az mirrored for mirrored instances)."""
        p = inst.p
        out = []
        for i, n in enumerate(p.counts, 1):
            # where stitch k really sits: the stitch columns drift along the spiral (LEAN stitches per round, as
            # drawn by _mesh_piece), so a zone drawn at az 0 must take the stitches that end up at az 0, not the
            # ones that started there - otherwise blazes and patches slide sideways off the face as rounds go on
            th = 2 * np.pi * (np.arange(n) + 0.5 - LEAN * (i - 0.5)) / n
            az = inst.az_of_theta(th)
            az = (az + 180) % 360 - 180
            col = np.array([p.colour] * n, dtype=object)
            side = "R" if inst.mirror else "L"
            for z in p.zones:
                if "runs" in z:
                    continue
                if z.get("side", side) != side:          # a zone on one copy of a mirrored pair only
                    continue
                r0, r1 = z.get("rounds", [1, p.N])
                if not (r0 <= i <= r1):
                    continue
                m = np.ones(n, bool)
                if "patch" in z:
                    m &= self._patch_mask(inst, i, n, th, z["patch"])
                if z.get("inner"):
                    m &= _inner_mask(inst, i, n, az, th, z)
                if "az" in z:
                    m &= _az_in(az, z["az"])
                if "shape" in z:
                    m &= _in_polygon(i, az, z["shape"])
                if "ellipse" in z:
                    el = z["ellipse"]
                    (rc, ac), (rr, ar) = el["center"], el["radii"]
                    da = (az - ac + 180) % 360 - 180
                    m &= ((i - rc) / rr) ** 2 + (da / ar) ** 2 <= 1.0
                if "colour" in z:
                    col[m] = z["colour"]
            # a written multi-colour round: colour runs in working order from the round-start marker (what a
            # pattern says: "2 sc in orange, 15 sc in white, ..."); a tube is worked toward decreasing theta, a
            # flat applique is seen from its right side. `side` limits it to one copy of a mirrored pair.
            side = "R" if inst.mirror else "L"
            for z in p.zones:
                if "runs" not in z or int(z.get("round", 0)) != i or z.get("side", side) != side:
                    continue
                seq = [c for c, k in z["runs"] for _ in range(int(k))]
                if len(seq) != n:
                    raise ValueError(f"{p.id} R{i}: colour runs make {len(seq)} stitches, the round has {n}")
                col = np.array(seq if inst.app is not None else seq[::-1], dtype=object)
            out.append(list(col))
        return out

    # patches: colour zones placed by millimetres on the surface -------------------------------------------
    def _patch_anchor(self, inst, pz, flip):
        """World centre and normal of a patch, and whether it lies on the toy's left. `flip` = evaluate the
        mirror image (the right-hand copy of a pair, or the second half of a `mirror: true` patch)."""
        if "eye" in pz:
            want = str(pz["eye"]).upper()[:1]
            if flip:
                want = "R" if want == "L" else "L"
            for f in self.spec.get("face", []):
                eye = f.get("type") in ("safety_eye", "eye") or (f.get("type") == "satin" and f.get("role") == "eye")
                if not eye or "at" not in f:
                    continue
                a = float(f["at"].get("az", 0.0))
                cands = [("L" if a >= 0 else "R", a, None)]
                if f.get("mirror"):
                    cands.append(("R" if a >= 0 else "L", -a, "R"))
                for sd, az, pair_side in cands:
                    if sd != want:
                        continue
                    on = f["on"]
                    if on in self.pairs or on + "_L" in self.inst:
                        host = self.inst[self._side(on, pair_side or "L")]
                        P, n = host.point(f["at"]["round"], f["at"].get("az", 0.0))
                    else:
                        P, n = self.inst[self._side(on, None)].point(f["at"]["round"], az)
                    return P, n
            raise ValueError(f"patch: no {want} eye on this toy")
        at = pz.get("at", {})
        rnd, az = at.get("round", "end"), float(at.get("az", 0.0))
        on = pz.get("on")
        if on is None or on == inst.p.id or on == inst.name:
            host = inst
            if flip and not inst.mirror:            # a `mirror: true` patch on an unpaired piece
                az = -az
        elif on in self.pairs or on + "_L" in self.inst:
            sd = str(pz.get("anchor_side", "L")).upper()[:1]
            if flip:
                sd = "R" if sd == "L" else "L"
            host = self.inst[self._side(on, sd)]
        else:
            host = self.inst[self._side(on, None)]
            if flip:
                az = -az
        return host.point(rnd, az)

    def patch_frame(self, inst, pz, flip=False):
        """Centre (offset included), normal, 'out' and 'up' tangent directions and the side sign of a patch."""
        c, nrm = self._patch_anchor(inst, pz, flip)
        nrm = _norm(nrm)
        up = WORLD_Y - (WORLD_Y @ nrm) * nrm
        if np.linalg.norm(up) < 0.35:
            up = WORLD_Z - (WORLD_Z @ nrm) * nrm
        up = _norm(up)
        side_v = _norm(np.cross(up, nrm))       # toward +X for a surface facing front
        sgn = 1.0 if c[0] > 1.0 else (-1.0 if c[0] < -1.0 else (-1.0 if flip else 1.0))
        if side_v @ WORLD_X * sgn < 0:
            side_v = -side_v
        off = pz.get("offset_mm", [0.0, 0.0])
        c = c + float(off[0]) * side_v + float(off[1]) * up
        return c, nrm, side_v, up, sgn

    def _patch_mask(self, inst, i, n, th, pz):
        """Stitches of round i inside a patch: an ellipse (`size_mm` [width, height] or a diameter) lying on the
        surface around an anchor - (round, az) on this or another piece (`on`), or an eye (`eye`: "L"/"R") -
        moved by `offset_mm` [out, up] (out = away from the toy's middle line, up = the toy's up along the
        surface; toward the toy's front on top of the head), turned by `rotate` (deg, counter-clockwise as seen
        from outside), with an irregular edge `wobble` (0..0.5, seeded by `seed`: a real coat patch). It covers
        every piece it reaches (each piece names it in its own `colours`; `exact: true` keeps it on its own side
        of the toy on a mirrored pair); `mirror: true` adds the mirror image; `anchor_side` "R" anchors it on the
        right-hand copy of a mirrored `on` piece.
        The pattern lists the stitches it takes, round by round, like any colour zone."""
        out = np.zeros(n, bool)
        P = N = None
        # `exact`: the patch is where its anchor says on every piece it reaches (a spot on one side of the toy);
        # otherwise the right-hand copy of a pair gets the mirror image, like any zone
        own = bool(inst.mirror) and not pz.get("exact")
        for flip in ((False, True) if pz.get("mirror") else (False,)):
            flip = flip != own
            c, nrm, side_v, up, sgn = self.patch_frame(inst, pz, flip)
            size = pz.get("size_mm", 10.0)
            w, h = (float(size), float(size)) if np.isscalar(size) else (float(size[0]), float(size[1]))
            if P is None:
                T = np.full(n, i - 0.5)
                P = inst.base(T, np.asarray(th, float))
                N = inst.base_normal(T, np.asarray(th, float))
            d = P - c
            du, dv, dn = d @ side_v, d @ up, d @ nrm
            rot = np.radians(float(pz.get("rotate", 0.0))) * (-1 if sgn < 0 else 1)
            cu, sv = np.cos(rot), np.sin(rot)
            x = (cu * du + sv * dv) / (w / 2)
            y = (-sv * du + cu * dv) / (h / 2)
            rr = np.hypot(x, y)
            wob = float(pz.get("wobble", 0.0))
            if wob:
                rng = np.random.default_rng(int(pz.get("seed", 1)))
                ph = rng.uniform(0, 2 * np.pi, 3)
                phi = np.arctan2(y, x)
                rr = rr / (1 + wob * (0.55 * np.sin(2 * phi + ph[0]) + 0.3 * np.sin(3 * phi + ph[1])
                                      + 0.15 * np.sin(5 * phi + ph[2])))
            near = np.abs(dn) <= 0.5 * max(w, h) + 4.0
            out |= (rr <= 1.0) & near & ((N @ nrm) > 0.0)
        if out.any() and not pz.get("hidden"):
            # stitches covered by another piece (the head under a sewn-on muzzle) stay in the main colour: nobody
            # sees them, and the crocheter is spared the colour changes (`hidden: true` paints them anyway)
            k = np.nonzero(out)[0]
            for other in self.inst.values():
                if other is inst or other.app is not None or not getattr(other, "placed", True):
                    continue
                try:
                    pen = other.penetration(P[k])
                except Exception:      # noqa: BLE001 - a piece without a usable surface hides nothing
                    continue
                out[k[pen < -1.0]] = False
        return out

    # mesh ------------------------------------------------------------------------------------------------
    def build(self, ku=6, kv=5, relief=1.0):
        self.dimples = []
        parts = []
        for name, inst in self.inst.items():
            parts.append(self._mesh_piece(inst, ku, kv, relief))
        parts.extend(copy.deepcopy(self.cord_meshes))
        parts.extend(copy.deepcopy(self.flat_meshes))
        self._recess_appliques(parts)
        self._face(parts)
        by_name = {m["name"]: m for m in parts}
        for name, c, n_eye, rad, depth in self.dimples:
            # a safety eye's washer pulls the fabric in around the eye
            m = by_name[name]
            d = np.linalg.norm(m["V"] - c, axis=1)
            w = np.exp(-(d / rad) ** 2 * 1.5)
            m["V"] = m["V"] - n_eye * (depth * w)[:, None]
        self._seam_shading(parts)
        for m in parts:
            m["N"] = vertex_normals(m["V"], m["F"])
            if "ROFF" in m:
                # the surface under the stitch relief (dimples, insets included): the renderer lights it with the
                # relief of every stitch drawn per pixel (render.py), sharper than the mesh can carry
                m["NB"] = vertex_normals(m["V"] - m["ROFF"], m["F"])
        # ground the toy; say so when something other than the body (and its feet) holds it up.
        # Loose yarn (fringe strands, chain cords) never holds a toy up: it lies on the floor.
        cords = self._cord_parts(parts)
        loose = [m for m in parts if m["name"].startswith("fringe@") or any(m is c for c in cords)]
        solid = [m for m in parts if not any(m is l for l in loose)]
        ymin = min(m["V"][:, 1].min() for m in solid)
        for m in loose:
            m["V"][:, 1] = np.maximum(m["V"][:, 1], ymin + 0.6)
        lowest = min(solid, key=lambda m: m["V"][:, 1].min())
        root = parts[0]
        lift = root["V"][:, 1].min() - ymin
        support = ("leg", "foot", "feet", "paw", "thigh", "haunch", "sole", "base")
        if lift > 3.0 and not any(k in lowest["name"].lower() for k in support):
            self.flags.append(f"ground: the toy rests on {lowest['name']}; the {root['name']} is {lift:.0f} mm above the floor")
        for m in parts:
            m["V"][:, 1] -= ymin
        self.ground_shift = -ymin
        return parts

    def _recess_appliques(self, parts):
        """Appliques with `inset`: press the parent's fabric in under them so they sit flush, not on top."""
        by_name = {m["name"]: m for m in parts}
        for m in parts:
            inst = m.get("inst")
            if inst is None or inst.app is None or not getattr(inst, "inset", 0.0):
                continue
            parent = inst.app[0]
            host = by_name.get(parent.name)
            if host is None:
                continue
            tree = cKDTree(m["V"])
            d, _ = tree.query(host["V"], distance_upper_bound=6.0)
            d = np.minimum(d, 6.0)
            w = np.clip(1.0 - (d - 0.6) / 1.6, 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            if not np.any(w > 0):
                continue
            n0 = vertex_normals(host["V"], host["F"])
            host["V"] = host["V"] - n0 * (inst.inset * w)[:, None]

    def _cord_parts(self, parts):
        names = {c["name"] for c in self.cord_meshes}
        return [m for m in parts if m["name"] in names and m["mat"] == MAT["yarn"] and "inst" not in m]

    def _seam_shading(self, parts):
        """Darken yarn where two pieces touch (the sewn seam and the crease between pressed pieces), so pieces
        of the same colour still read as separate pieces, as in a photo."""
        # appliques lie on their parent everywhere: they get no seam shading and cast none
        yarn = [m for m in parts if m["mat"] == MAT["yarn"]
                and not (m.get("inst") is not None and m["inst"].app is not None)]
        if len(yarn) < 2:
            return
        trees = [cKDTree(m["V"][::3]) for m in yarn]
        for i, m in enumerate(yarn):
            dmin = np.full(len(m["V"]), np.inf)
            for j, t in enumerate(trees):
                if i != j:
                    d, _ = t.query(m["V"], distance_upper_bound=4.0)
                    dmin = np.minimum(dmin, d)
            f = 1.0 - 0.2 * np.exp(-(np.minimum(dmin, 50) / 1.4) ** 2)
            m["C"] = m["C"] * f[:, None]

    def _mesh_piece(self, inst, ku, kv, relief):
        p = inst.p
        w = p.w
        chart = self.chart(inst)
        verts, cols, uvs, hs, tri, ts = [], [], [], [], [], []
        amps, roffs = [], []          # per vertex: relief amplitude (mm) and relief offset, for the renderer's bump
        base_idx = 0
        bands = list(range(1, p.N + 1)) + ([p.N + 1] if p.closed else [])
        # popcorns: (centre round parameter, centre angle) of every popcorn of the piece. Each is a round ball
        # ~2 stitches across and ~1.8 rounds tall (5 dc gathered), so it spills into the neighbouring rounds
        pops = []
        for r_i, rd_ in enumerate(p.rounds, 1):
            if rd_.loop == "popcorn":
                for k_, o_ in enumerate(rd_.ops):
                    if o_ == "pop":
                        pops.append((r_i - 0.35, 2 * np.pi * (k_ + 0.5 - LEAN * (r_i - 0.5)) / rd_.n))
        pops = np.array(pops) if pops else None
        for b in bands:
            n = p.counts[min(b, p.N) - 1]
            nu = n * ku + 1
            nv = kv + 1
            u = np.linspace(0, n, nu)
            ov = 0.04 if 1 < b < bands[-1] else 0.0
            v = np.linspace(-ov, 1 + ov, nv)
            U, Vv = np.meshgrid(u, v, indexing="ij")
            T = (b - 1) + Vv
            Vv = np.clip(Vv, 0, 1)
            TH = 2 * np.pi * U / n
            B = inst.base(T, TH)
            Nn = inst.base_normal(T, TH)
            fu = np.mod(U + LEAN * T, 1.0)
            H = stitch_height(fu, Vv)
            rib = p.rounds[min(b, p.N) - 1].loop == "rib"
            if rib:
                # (fpdc, bpdc) ribbing: front-post columns stand out, back-post columns sink in between
                fr = np.mod(U, 1.0)
                odd = np.mod(np.floor(U), 2) == 1
                H = np.where(odd, 0.12 + 0.12 * np.sin(np.pi * fr), 0.55 + 0.6 * np.sin(np.pi * fr) ** 0.7)
            r_loc = p.r_t(T) + p.oval / 2
            if inst.app is not None:
                r_loc = p._arc(np.clip(T, 0, p.M))       # a flat piece: distance from its start along the fabric
            amp = 0.15 * w * relief * np.clip(r_loc / (1.4 * w), 0.25, 1.0)
            if inst.app is not None and p.oval > 0:
                amp = amp * np.clip(T / 3.0, 0.2, 1.0)   # near the chain line the rounds fold over it: keep it calm
            if b == p.N + 1:
                amp = amp * (1 - 0.5 * Vv)
            if (not p.closed) and b == p.N:
                amp = amp * np.clip((1 - Vv) * 3, 0.3, 1.0)
            if rib:
                amp = amp * 1.8
            roff = Nn * (amp * (H - 0.45))[..., None]
            X = B + roff
            j = np.mod(np.floor(U + LEAN * T).astype(int), n)    # the stitch whose V is drawn here
            rd = p.rounds[min(b, p.N) - 1]
            if pops is not None:
                near = pops[np.abs(pops[:, 0] - (b - 0.5)) < 1.6]
                if len(near):
                    bump = np.zeros_like(T)
                    for t0, th0 in near:
                        dth = (TH - th0 + np.pi) % (2 * np.pi) - np.pi
                        ds = dth * n / (2 * np.pi)                     # stitches along the round
                        d2 = (ds / 1.05) ** 2 + ((T - t0) / 0.95) ** 2
                        bump = np.maximum(bump, np.sqrt(np.clip(1.0 - d2, 0.0, 1.0)))
                    X = X + Nn * (bump * 1.25 * w * relief)[..., None]
                    roff = roff + Nn * (bump * 1.25 * w * relief)[..., None]
                    amp = np.where(bump > 0, 0.0, amp)          # (a popcorn is shaded by its own geometry)
            cr = chart[min(b, p.N) - 1]
            C = np.array([self.colours.get(cr[k], np.array([0.8, 0.2, 0.8])) for k in range(n)])[j]
            verts.append(X.reshape(-1, 3)); cols.append(C.reshape(-1, 3))
            amps.append((np.zeros_like(T) if rib else np.broadcast_to(amp, T.shape)).reshape(-1))
            roffs.append(roff.reshape(-1, 3))
            uvs.append(np.stack([U + LEAN * T, T], -1).reshape(-1, 2)); hs.append(H.reshape(-1))
            ts.append(T.reshape(-1))
            ii = np.arange(nu * nv).reshape(nu, nv) + base_idx
            a = ii[:-1, :-1]; bq = ii[1:, :-1]; c = ii[1:, 1:]; d = ii[:-1, 1:]
            if getattr(p, "opening", None) is not None and b > p.opening["from"]:
                # turned rows: the unworked front window (the face opening) has no fabric
                Tc = (T[:-1, :-1] + T[1:, 1:]) / 2; THc = (TH[:-1, :-1] + TH[1:, 1:]) / 2
                ok = ~p.in_opening(Tc, THc)
                a, bq, c, d = a[ok], bq[ok], c[ok], d[ok]
            tri.append(np.concatenate([np.stack([a, bq, c], -1).reshape(-1, 3),
                                       np.stack([a, c, d], -1).reshape(-1, 3)]))
            base_idx += nu * nv
        V = np.concatenate(verts); F = np.concatenate(tri)
        if inst.app is not None:
            F = F[:, ::-1]
        C = np.concatenate(cols); UV = np.concatenate(uvs); Hh = np.concatenate(hs); Tt = np.concatenate(ts)
        AMP = np.concatenate(amps); ROFF = np.concatenate(roffs)
        if getattr(p, "opening", None) is not None:
            # the face opening's window has no fabric: drop its vertices too (nothing may sit in the hole)
            used = np.zeros(len(V), bool); used[F.reshape(-1)] = True
            remap = np.cumsum(used) - 1
            F = remap[F]
            V, C, UV, Hh, Tt, AMP, ROFF = (x[used] for x in (V, C, UV, Hh, Tt, AMP, ROFF))
        n_front = len(V)
        face = getattr(inst, "edge_face", None)
        if face is not None and getattr(p, "motif", None) is not None:
            # a single layer of fabric standing out: its back (the wrong side, same colours) a fabric's thickness
            # behind, and the rounded edge of the fabric all round
            gap = 1.5 * p.thick
            back = V - face * gap
            ol = inst.to_world(p.outline_local(mirror_y=inst.mirror)) - face * (gap / 2)
            ol = np.vstack([ol, ol[:1]])
            tube = tube_mesh(ol, np.tile(face, (len(ol), 1)), gap / 2 * 1.08, sides=8)
            last = chart[-1]
            rim_c = self.colours.get(max(set(last), key=last.count), C[0])
            nV = len(V)
            F = np.concatenate([F, F[:, ::-1] + nV, tube["F"] + 2 * nV])
            V = np.concatenate([V, back, tube["V"]])
            C = np.concatenate([C, C, np.tile(rim_c, (len(tube["V"]), 1))])
            UV = np.concatenate([UV, UV, tube["UV"]]); Hh = np.concatenate([Hh, Hh, np.full(len(tube["V"]), 0.6)])
            Tt = np.concatenate([Tt, Tt, np.full(len(tube["V"]), float(p.M))])
        elif face is not None and not getattr(p, "dish", 0) > 0:      # (a dish has its own rim and back layer)
            # two circles joined at the edge: the back circle in the piece's main colour, a yarn's thickness apart
            gap = 1.1 * p.thick
            back = V - face * gap
            Cb = np.tile(self.colours.get(p.colour, C[0]), (len(V), 1))
            F = np.concatenate([F, F[:, ::-1] + len(V)])
            V = np.concatenate([V, back]); C = np.concatenate([C, Cb])
            UV = np.concatenate([UV, UV]); Hh = np.concatenate([Hh, Hh]); Tt = np.concatenate([Tt, Tt])
        if len(V) > n_front:          # back layers / rims: shaded by their own geometry
            AMP = np.concatenate([AMP, np.zeros(len(V) - n_front)])
            ROFF = np.concatenate([ROFF, np.zeros((len(V) - n_front, 3))])
        m = {"name": inst.name, "V": V, "F": F, "C": C, "UV": UV,
             "H": Hh, "T": Tt, "mat": MAT["yarn"], "inst": inst, "AMP": AMP, "ROFF": ROFF}
        return m

    # face ------------------------------------------------------------------------------------------------
    def _face(self, parts):
        by_name = {m["name"]: m for m in parts}
        items = []
        for f in self.spec.get("face", []):
            if f.get("mirror") and (f["on"] in self.pairs or f["on"] + "_L" in self.inst):
                a = dict(f, on=self._side(f["on"], "L")); b = dict(copy.deepcopy(f), on=self._side(f["on"], "R"))
                if "rotate" in b:
                    b["rotate"] = -b["rotate"]
                items += [a, b]
            elif f.get("mirror"):
                a = dict(f); b = copy.deepcopy(f)
                if "at" in f:
                    b["at"] = dict(f["at"]); b["at"]["az"] = -f["at"].get("az", 0)
                if "points" in b:
                    b["points"] = [[r, -az] for r, az in f["points"]]
                if "rotate" in b:
                    b["rotate"] = -b["rotate"]
                items += [a, b]
            else:
                items.append(f)
        for k_item, f in enumerate(items):
            inst = self.inst[self._side(f["on"], None)]
            host = by_name[inst.name]
            kind = f["type"]
            n_before = len(parts)
            self._face_item(parts, inst, host, f, kind)
            for m in parts[n_before:]:
                # which face item a mesh belongs to (a satin nose is many strands): visible.py groups by it
                m["face_item"] = {"index": k_item, "type": kind, "role": f.get("role"), "id": f.get("id"),
                                  "on": inst.name, "at": f.get("at")}

    def _face_item(self, parts, inst, host, f, kind):
        if kind in ("safety_eye", "eye"):
            parts.append(self._eye(inst, host, f))
        elif kind == "safety_nose":
            parts.append(self._nose(inst, host, f))
        elif kind in ("stitch", "embroidery", "line"):
            parts.append(self._thread(inst, host, f))
        elif kind == "strap":
            parts.append(self._strap(inst, host, f))
        elif kind == "satin":
            parts.extend(self._satin(inst, host, f))
        elif kind == "blush":
            self._blush(inst, host, f, parts)
        elif kind == "fringe":
            parts.extend(self._fringe(inst, host, f))
        elif kind == "felt":
            parts.append(self._felt(inst, host, f))
        else:
            self.flags.append(f"face item type {kind!r} not supported")

    def _surface_at(self, inst, host, rnd, az):
        P, n = inst.point(rnd, az)
        # snap to the actual (relief) mesh
        if "_kd" not in host:
            host["_kd"] = cKDTree(host["V"])
        d, i = host["_kd"].query(P)
        return P, n, host["V"][i]

    def _eye(self, inst, host, f):
        size = float(f.get("size_mm", 9.0))
        r = size / 2
        P, n, Pm = self._surface_at(inst, host, f["at"]["round"], f["at"].get("az", 0))
        sink = float(f.get("sink", 0.45))
        c = P + n * (r * (0.5 - sink) + 0.4)
        S = sphere(24, 16)
        aspect = f.get("aspect", 1.0)       # oval eyes: taller than wide
        up = _norm(WORLD_Y - (WORLD_Y @ n) * n)
        side = np.cross(n, up)
        Mx = np.stack([side * r, up * r * aspect, n * r * 0.85], axis=1)
        V = S["V"] @ Mx.T + c
        col = self.colours.get(f.get("colour", "black"), hex_rgb("#151313"))
        C = np.tile(col, (len(V), 1))
        if f.get("iris"):
            # a coloured safety eye: iris over the visible dome, black pupil in the middle
            zc = S["V"][:, 2]
            C[zc > 0.25] = self.colours.get(f["iris"], hex_rgb(f["iris"]) if str(f["iris"]).startswith("#") else col)
            pupil = float(f.get("pupil", 0.5))
            C[zc > np.cos(np.radians(70.0 * pupil))] = col
        F = S["F"]
        hl = f.get("highlight")
        if hl:
            # printed catchlights of "cartoon" safety eyes: a big white dot up and to one side, a small one
            # opposite. The same world side on both eyes (they are identical eyes, not mirrored): the big dot up
            # toward the toy's right (the viewer's upper left), where the studio key light also glints, so the
            # print and the reflection read as one catchlight instead of two on opposite sides
            k = 1.0 if hl is True else float(hl)
            sgn = 1.0 if side @ np.array([1.0, 0, 0]) > 0 else -1.0     # u_s < 0: the big dot toward -X
            for (u_s, u_u, rad) in ((-0.30, 0.33, 0.29 * k), (0.32, -0.28, 0.12 * k)):
                u = np.array([u_s * sgn, u_u, 0.0]); u[2] = np.sqrt(max(1 - u[0] ** 2 - u[1] ** 2, 0.05))
                p0 = c + Mx @ u
                nn = _norm(np.linalg.solve(Mx.T, u)) if abs(np.linalg.det(Mx)) > 1e-9 else n
                H_ = sphere(12, 8)
                Mh = np.stack([side * r * rad, up * r * rad, nn * r * rad * 0.35], axis=1)
                F = np.vstack([F, H_["F"] + len(V)])
                V = np.vstack([V, H_["V"] @ Mh.T + p0 + nn * 0.05])
                C = np.vstack([C, np.tile(hex_rgb("#fbfaf6"), (len(H_["V"]), 1))])
        self.dimples.append((inst.name, c, n, r * 1.25, 0.25 * r))
        return {"name": f"eye@{inst.name}", "V": V, "F": F, "C": C,
                "UV": np.zeros((len(V), 2)), "H": np.ones(len(V)), "mat": MAT["plastic"]}

    def _nose(self, inst, host, f):
        size = f.get("size_mm", [10, 7])
        if not isinstance(size, list):
            size = [size, size * 0.75]
        P, n, _ = self._surface_at(inst, host, f["at"]["round"], f["at"].get("az", 0))
        up = _norm(WORLD_Y - (WORLD_Y @ n) * n)
        side = np.cross(n, up)
        shape = f.get("shape", "oval")
        S = rounded_triangle(28, 10) if shape == "triangle" else sphere(24, 14)
        Mx = np.stack([side * size[0] / 2, up * size[1] / 2, n * min(size) * 0.38], axis=1)
        c = P + n * 0.2
        V = S["V"] @ Mx.T + c
        col = self.colours.get(f.get("colour", "black"), hex_rgb("#151313"))
        return {"name": f"nose@{inst.name}", "V": V, "F": S["F"], "C": np.tile(col, (len(V), 1)),
                "UV": np.zeros((len(V), 2)), "H": np.ones(len(V)), "mat": MAT["plastic"]}

    def _thread_path(self, inst, host, pts, lift, per_mm=1.5):
        """Polyline across the surface through (round, az) points; densified in (t, theta) space."""
        tt = np.array([inst.t_of_round(r) for r, a in pts]); aa = np.array([a for r, a in pts], float)
        P0 = []
        for k in range(len(pts) - 1):
            a0 = aa[k]
            a1 = a0 + ((aa[k + 1] - a0 + 180) % 360 - 180)      # the short way round
            n_s = 24
            s = np.linspace(0, 1, n_s, endpoint=(k == len(pts) - 2))
            T = tt[k] + (tt[k + 1] - tt[k]) * s
            A = a0 + (a1 - a0) * s
            P0.append(np.stack([T, inst.theta(A)], -1))
        TA = np.concatenate(P0)
        P = inst.base(TA[:, 0], TA[:, 1])
        if "_kd" not in host:
            host["_kd"] = cKDTree(host["V"])
        N = inst.base_normal(TA[:, 0], TA[:, 1])
        # ride on top of the relief: offset to the outermost nearby mesh point along the normal
        d, idx = host["_kd"].query(P, k=12)
        nb = host["V"][idx]
        h = np.einsum("ijk,ik->ij", nb - P[:, None, :], N).max(axis=1)
        return P + N * (np.maximum(h, 0) + lift)[:, None], N

    def _thread(self, inst, host, f):
        th = float(f.get("thickness_mm", 1.3))
        P, N = self._thread_path(inst, host, f["points"], lift=th * 0.25)
        tube = tube_mesh(P, N, th / 2, sides=8)
        col = self.colours.get(f.get("colour", "black"), hex_rgb("#151313"))
        return {"name": f"thread@{inst.name}", "V": tube["V"], "F": tube["F"], "C": np.tile(col, (len(tube["V"]), 1)),
                "UV": tube["UV"], "H": np.ones(len(tube["V"])), "mat": MAT["thread"]}

    def _project(self, host, pts, n, lift):
        """Points near the surface (a tangent-plane drawing) laid onto the relief mesh along n."""
        if "_kd" not in host:
            host["_kd"] = cKDTree(host["V"])
        for _ in range(2):
            d, idx = host["_kd"].query(pts, k=10)
            nb = host["V"][idx]
            h = np.einsum("ijk,k->ij", nb - pts[:, None, :], n).max(axis=1)
            pts = pts + n * h[:, None]
        return pts + n * lift

    def _strap(self, inst, host, f):
        """A crocheted chain (or a narrow strip) sewn along the surface: backpack straps, headbands, collars."""
        th = float(f.get("thickness_mm", self.yarn["stitch_mm"] * 0.8))
        P, N = self._thread_path(inst, host, f["points"], lift=th * 0.35)
        tube = tube_mesh(P, N, th / 2, sides=10)
        link = float(f.get("link_mm", self.yarn["stitch_mm"] * 0.8))
        H = 0.5 + 0.5 * np.cos(2 * np.pi * tube["UV"][:, 0] / link)
        col = self.colours.get(f.get("colour", "black"), hex_rgb("#888888"))
        return {"name": f"strap@{inst.name}", "V": tube["V"], "F": tube["F"], "C": np.tile(col, (len(tube["V"]), 1)),
                "UV": tube["UV"], "H": H, "mat": MAT["yarn"]}

    def _satin(self, inst, host, f):
        """Satin-stitched patch (an embroidered nose): horizontal strands filling a shape drawn in mm in the
        surface's tangent plane (up = the toy's up), laid over the stitches."""
        P, n, _ = self._surface_at(inst, host, f["at"]["round"], f["at"].get("az", 0))
        wid = float(f.get("width_mm", f.get("width_st", 3.0) * inst.p.w))
        hgt = float(f.get("height_mm", f.get("height_rounds", 2.0) * inst.p.h))
        shape = f.get("shape", "triangle")
        th = float(f.get("thickness_mm", 1.4))
        up = _norm(WORLD_Y - (WORLD_Y @ n) * n); side = np.cross(n, up)
        if f.get("rotate"):
            a = np.radians(float(f["rotate"]))
            side, up = np.cos(a) * side + np.sin(a) * up, -np.sin(a) * side + np.cos(a) * up
        nstr = max(3, int(f.get("strands", round(hgt / (th * 0.55)))))
        # offset_mm [right, up] as seen looking at that surface (a highlight on an embroidered eye);
        # lift_mm lays it over earlier embroidery (white satin on black satin)
        off = f.get("offset_mm") or [0, 0]
        P = P - float(off[0]) * side + float(off[1]) * up
        lift = float(f.get("lift_mm", 0.0))
        out = []
        col = self.colours.get(f.get("colour", "black"), hex_rgb("#151313"))
        for k in range(nstr):
            s_ = (k + 0.5) / nstr                      # 0 top .. 1 bottom
            if shape == "triangle":
                half = 0.5 * wid * (1 - s_) ** 0.9
            elif shape == "heart":
                half = 0.5 * wid * min(1.0, 1.25 * (1 - s_)) * (0.8 + 0.2 * np.sin(np.pi * s_))
            else:
                half = 0.5 * wid * np.sqrt(max(1 - (2 * s_ - 1) ** 2, 0))
            half = max(half, th * 0.4)
            y = hgt / 2 - s_ * hgt
            xs = np.linspace(-half, half, 16)
            pts = P + xs[:, None] * side + y * up
            pts = self._project(host, pts, n, th * 0.3 + lift)
            tube = tube_mesh(pts, np.tile(n, (len(pts), 1)), th / 2, sides=8)
            out.append({"name": f"satin@{inst.name}", "V": tube["V"], "F": tube["F"],
                        "C": np.tile(col, (len(tube["V"]), 1)), "UV": tube["UV"], "H": np.ones(len(tube["V"])),
                        "mat": MAT["thread"]})
        return out

    def _fringe(self, inst, host, f):
        """Yarn strands knotted along a line (a lark's head knot per strand, both ends hanging): a unicorn's mane,
        a highland cow's fringe, a lion's mane, doll hair.

        points: the line on the piece as [round, az] points; count: strands along it; length_mm: hanging length;
        droop 0..1: 0 = strands stick out along the surface normal, 1 = they fall straight down; curl (deg):
        curl along their length (curly hair, cow fringe); spread: random variation; colours: list cycled along the
        line (a rainbow mane)."""
        pts = f["points"]
        n = int(f.get("count", 12))
        L = float(f.get("length_mm", 20))
        th = float(f.get("thickness_mm", self.yarn.get("thickness_mm", 1.9) * 1.1))
        droop = float(f.get("droop", 0.6))
        curl = np.radians(float(f.get("curl", 0)))
        spread = float(f.get("spread", 0.15))
        rng = np.random.default_rng(int(f.get("seed", 7)))
        cols = f.get("colours") or [f.get("colour", "black")]
        P, N = self._thread_path(inst, host, pts, lift=0.0)
        seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        out = []
        down = -WORLD_Y
        for k in range(n):
            s_ = (k + 0.5) / n * seg[-1]
            i = int(np.clip(np.searchsorted(seg, s_) - 1, 0, len(P) - 2))
            a = (s_ - seg[i]) / max(seg[i + 1] - seg[i], 1e-9)
            p0 = P[i] * (1 - a) + P[i + 1] * a
            nrm = _norm(N[i] * (1 - a) + N[i + 1] * a)
            tang = _norm(P[i + 1] - P[i])
            for half in ((-1, 1) if f.get("doubled", True) else (1,)):
                d0 = _norm(nrm * (1 - droop) + down * droop + rng.normal(0, spread, 3) * 0.6)
                side = _norm(np.cross(d0, tang)) if np.linalg.norm(np.cross(d0, tang)) > 1e-6 else tang
                m = max(12, int(L / 1.2))
                ts = np.linspace(0, 1, m)
                Lk = L * (1 + rng.normal(0, spread * 0.4))
                ck = curl * (1 + rng.normal(0, 0.15)) * (1.0 if half > 0 else 0.85)
                # the strand leaves the surface along d0, sags toward 'down' (gravity) and curls OUTWARD (away from
                # the surface it is knotted to) by `curl` degrees over its length
                dirs = []
                for t in ts:
                    g = _norm(d0 * (1 - t * droop * 0.6) + down * t * droop * 0.6)
                    v = nrm - (nrm @ g) * g
                    v = _norm(v) if np.linalg.norm(v) > 1e-6 else side
                    ang = ck * t
                    dirs.append(_norm(g * np.cos(ang) + v * np.sin(ang)))
                dirs = np.array(dirs)
                path = p0 + nrm * th * 0.3 + half * tang * th * 0.35 + np.vstack(
                    [[0, 0, 0], np.cumsum(dirs[:-1] * (Lk / (m - 1)), axis=0)])
                # strands rest on the toy instead of passing through it
                for o in self.inst.values():
                    sd, Q, NQ = o.signed_distance(path)
                    near = np.linalg.norm(path - Q, axis=1) < 1.2 * o.p.max_radius()
                    push = np.where(near & (sd < th * 0.6), th * 0.6 - sd, 0.0)
                    path = path + NQ * push[:, None]
                tube = tube_mesh(path, np.tile(side, (m, 1)), th / 2, sides=7)
                ply = 0.5 + 0.5 * np.cos(tube["UV"][:, 0] * 2.2 + tube["UV"][:, 1] * 2)   # plied twist
                col = self.colours.get(cols[k % len(cols)], hex_rgb("#888888"))
                out.append({"name": f"fringe@{inst.name}", "V": tube["V"], "F": tube["F"],
                            "C": np.tile(col, (len(tube["V"]), 1)), "UV": tube["UV"], "H": ply,
                            "mat": MAT["yarn"]})
        return out

    def _blush(self, inst, host, f, parts):
        P, n, _ = self._surface_at(inst, host, f["at"]["round"], f["at"].get("az", 0))
        rad = float(f.get("size_mm", 8)) / 2
        col = self.colours.get(f.get("colour", "pink"), hex_rgb("#f2a0a8"))
        strength = float(f.get("strength", 0.55))
        d = np.linalg.norm(host["V"] - P, axis=1)
        w = strength * np.exp(-(d / rad) ** 2 * 1.6)
        host["C"] = host["C"] * (1 - w[:, None]) + col * w[:, None]

    def _felt(self, inst, host, f):
        """A felt or crocheted flat shape glued/sewn on (ellipse in mm, rotated), following the surface."""
        P, n, _ = self._surface_at(inst, host, f["at"]["round"], f["at"].get("az", 0))
        a, b = f.get("size_mm", [10, 7])
        rot = np.radians(f.get("rotate", 0.0))
        up = _norm(WORLD_Y - (WORLD_Y @ n) * n); side = np.cross(n, up)
        e1 = np.cos(rot) * side + np.sin(rot) * up; e2 = -np.sin(rot) * side + np.cos(rot) * up
        nr, na = 8, 48
        rr = np.linspace(0, 1, nr); aa = np.linspace(0, 2 * np.pi, na, endpoint=False)
        R_, A_ = np.meshgrid(rr, aa, indexing="ij")
        pts = P + (R_ * np.cos(A_) * a / 2)[..., None] * e1 + (R_ * np.sin(A_) * b / 2)[..., None] * e2
        pts = pts.reshape(-1, 3)
        if "_kd" not in host:
            host["_kd"] = cKDTree(host["V"])
        d, idx = host["_kd"].query(pts, k=8)
        nb = host["V"][idx]
        h = np.einsum("ijk,k->ij", nb - pts[:, None, :], n).max(axis=1)
        V = pts + n * (np.maximum(h, 0) + 0.5)[:, None]
        F = []
        for i in range(nr - 1):
            for j in range(na):
                a0 = i * na + j; a1 = i * na + (j + 1) % na; b0 = (i + 1) * na + j; b1 = (i + 1) * na + (j + 1) % na
                F += [[a0, b0, b1], [a0, b1, a1]]
        col = self.colours.get(f.get("colour", "black"), hex_rgb("#151313"))
        return {"name": f"felt@{inst.name}", "V": V, "F": np.array(F), "C": np.tile(col, (len(V), 1)),
                "UV": np.zeros((len(V), 2)), "H": np.ones(len(V)), "mat": MAT["felt"]}


# ---- mesh helpers ----------------------------------------------------------------------------------------

def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return _norm(N)


def sphere(nu, nv):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False); v = np.linspace(0, np.pi, nv)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    V = np.stack([np.sin(Vv) * np.cos(U), np.sin(Vv) * np.sin(U), np.cos(Vv)], -1).reshape(-1, 3)
    F = []
    for i in range(nu):
        for j in range(nv - 1):
            a = i * nv + j; b = ((i + 1) % nu) * nv + j
            F += [[a, b, b + 1], [a, b + 1, a + 1]]
    return {"V": V, "F": np.array(F)}


def rounded_triangle(nu, nv, rounding=0.32):
    """A dome over a rounded, flat-topped triangle pointing down (a safety nose), unit half-size."""
    A = np.array([[-1.0, 0.62], [1.0, 0.62], [0.0, -0.95]]) * (1 - rounding * 0.6)

    def sdf(p):
        d = np.full(len(p), np.inf); inside = np.ones(len(p), bool)
        for i in range(3):
            a, b = A[i], A[(i + 1) % 3]
            e = b - a; w = p - a
            t = np.clip((w @ e) / (e @ e), 0, 1)
            d = np.minimum(d, np.linalg.norm(w - np.outer(t, e), axis=1))
            inside &= (e[0] * w[:, 1] - e[1] * w[:, 0]) < 0
        return np.where(inside, -d, d) - rounding

    s = sphere(nu, nv)
    V = s["V"].copy()
    ang = np.arctan2(V[:, 1], V[:, 0])
    dirs = np.stack([np.cos(ang), np.sin(ang)], -1)
    lo = np.zeros(len(V)); hi = np.full(len(V), 3.0)
    for _ in range(30):
        mid = (lo + hi) / 2
        inside = sdf(dirs * mid[:, None]) < 0
        lo = np.where(inside, mid, lo); hi = np.where(inside, hi, mid)
    V[:, 0] *= lo; V[:, 1] *= lo
    return {"V": V, "F": s["F"]}


def tube_mesh(P, N, rad, sides=8):
    T = _norm(np.gradient(P, axis=0))
    # parallel-transport the first normal along the path (double reflection method)
    n0 = N[0] - (N[0] @ T[0]) * T[0]
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.cross(T[0], [1.0, 0, 0]) if abs(T[0][0]) < 0.9 else np.cross(T[0], [0, 1.0, 0])
    Nn = np.empty_like(P)
    Nn[0] = _norm(n0)
    for i in range(len(P) - 1):
        v1 = P[i + 1] - P[i]
        c1 = v1 @ v1
        if c1 < 1e-12:
            Nn[i + 1] = Nn[i]; continue
        rL = Nn[i] - (2 / c1) * (v1 @ Nn[i]) * v1
        tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        Nn[i + 1] = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-12 else rL
    Nn = _norm(Nn)
    B = _norm(np.cross(T, Nn))
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    V = P[:, None, :] + rad * (np.cos(a)[None, :, None] * Nn[:, None, :] + np.sin(a)[None, :, None] * B[:, None, :])
    n = len(P)
    V = V.reshape(-1, 3)
    F = []
    for i in range(n - 1):
        for j in range(sides):
            a0 = i * sides + j; a1 = i * sides + (j + 1) % sides
            b0 = a0 + sides; b1 = a1 + sides
            F += [[a0, b1, b0], [a0, a1, b1]]          # outward-facing winding
    # end caps
    c0 = len(V); c1 = c0 + 1
    V = np.vstack([V, P[0] - T[0] * rad * 0.5, P[-1] + T[-1] * rad * 0.5])
    for j in range(sides):
        F += [[c0, j, (j + 1) % sides], [c1, (n - 1) * sides + (j + 1) % sides, (n - 1) * sides + j]]
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    UV = np.stack([np.repeat(L, sides), np.tile(a, n)], -1)
    UV = np.vstack([UV, [[0, 0], [L[-1], 0]]])
    return {"V": V, "F": np.array(F), "UV": UV}


def load(path):
    with open(path, encoding="utf-8-sig") as fh:
        return Toy(json.load(fh))
