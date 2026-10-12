"""One crocheted piece: rounds -> stuffed profile -> local surface -> placed instance -> stitch-relief mesh.

Local frame of a piece: +z runs from the start (magic ring / chain) to the end (rim or closing hole); local x is
the piece's "front" reference, local y = z cross x. Deformations are expressed in that frame:
  flatten  0..1  pressed flat along local x (an unstuffed ear, a flat arm); width grows to keep the perimeter
  squash   0..1  the opening pinched into a slit as it is sewn (duck bill), ramped over the last rounds
  bend     {"deg": a, "at": round, "over": rounds}  the axis bends toward local +x (a foot, an elbow)
The mesh carries per-vertex stitch coordinates (u = stitch index along the round, t = round parameter) and the
height of the yarn relief, so every round shows its own number of V stitches as real geometry.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator

from .rounds import expand
from .profile import solve

LEAN = 0.10                 # stitch columns drift along a spiral (stitches per round)
# how far a thin layer of stuffing holds a spoon ear's two walls apart (x the unstuffed two-layer gap, added)
SPOON_PUFF = {"none": 0.0, "light": 0.9, "medium": 1.6, "firm": 2.2}


def stitch_height(fu, fv):
    """Relief of one single-crochet stitch seen from the right side, 0 (groove) .. 1 (top of a yarn leg).

    Two yarn legs form a V: they meet at the bottom centre and open toward the top corners; the groove between
    rounds is where the next round's legs sit on the previous round's top loops."""
    h = np.zeros_like(fu)
    for sgn in (-1.0, 1.0):
        p0 = np.array([0.5 + sgn * 0.08, 0.14])
        p1 = np.array([0.5 + sgn * 0.34, 0.86])
        d = p1 - p0
        L2 = d @ d
        px, py = fu - p0[0], (fv - p0[1]) * 0.95
        s = np.clip((px * d[0] + py * d[1] * 0.95) / L2, 0.0, 1.0)
        qx = px - s * d[0]; qy = py - s * d[1] * 0.95
        dist = np.sqrt(qx * qx + qy * qy)
        # plump yarn: a round leg a little wider than the gap between legs, highest in its middle (soft "u",
        # not a sharp chevron)
        leg = np.clip(1.0 - (dist / 0.285) ** 2, 0.0, 1.0)
        h = np.maximum(h, leg ** 0.42 * (0.80 + 0.20 * np.sin(np.pi * s)))
    # the band edges are the grooves between rounds: pin the relief there so neighbouring rounds (with their
    # own stitch counts) meet without cracks
    env = np.sin(np.pi * np.clip(fv, 0, 1)) ** 0.5
    return 0.36 * (1 - env) + h * env + 0.36 * env * (1 - h) * 0.6


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class PieceDef:
    """Geometry of a piece that does not depend on where it is sewn."""

    def __init__(self, pid, d, yarn):
        self.id = pid
        self.d = d
        self.w = float(yarn["stitch_mm"])
        self.h = float(yarn["round_mm"])
        self.thick = float(yarn.get("thickness_mm", 0.45 * self.w))
        # a chain-oval round counted from the centre front: on a side-to-side oval (the default) the turns are a
        # quarter of the round away; on a front-to-back oval the centre front is a turn
        turns = 0.25 if d.get("oval_axis", "side") != "front" else 0.0
        self.rounds = expand(d["rounds"], oval_turns=turns)
        self.counts = [r.n for r in self.rounds]
        self.N = len(self.rounds)
        self.closed = d.get("end", "closed") == "closed"
        self.stuff = d.get("stuff", "firm" if self.closed else "none")
        self.start = self.rounds[0].start
        self.oval = (self.rounds[0].chain - 2) * self.w if self.start == "ch" else 0.0
        loops = [r.loop for r in self.rounds]
        r, z, info = solve(self.counts, self.w, self.h, closed=self.closed, stuffing=self.stuff,
                           oval_len=self.oval, loops=loops, start=self.start)
        self.solve_info = info
        self.rb, self.zb = r, z
        self.M = len(r) - 1                    # boundaries 0..M (M = N, or N+1 when closed)
        tb = np.arange(self.M + 1, dtype=float)
        self._r = PchipInterpolator(tb, r)
        self._z = PchipInterpolator(tb, z)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(r), np.diff(z)))])
        self._arc = PchipInterpolator(tb, arc)       # distance from the start along the fabric (appliques)
        self._dr = self._r.derivative()
        self._dz = self._z.derivative()
        self.flatten = float(d.get("flatten", 0.0))
        # spoon 0..1: an unstuffed open cup pressed so its front wall nests in its back wall (a crocheted ear):
        # two fabric layers with a rounded fold at the sides, concave in front, convex behind; 0 = off
        self.spoon = float(d.get("spoon", 0.0)) if not self.closed else 0.0
        # dish 0..1: an unstuffed cup pressed along its axis: the magic ring is the centre of a concave front face,
        # the rounds after the widest roll back into a thick rounded rim and a back layer (a round bear ear)
        self.dish = float(d.get("dish", 0.0))
        if self.dish > 0:
            self._dish_setup()
        self.squash = float(d.get("squash", 0.0))
        self.bend = d.get("bend")
        # a chain-oval piece's chain runs side to side by default; "front" runs it front to back (a sitting body)
        self.oval_axis = d.get("oval_axis", "side")
        self.colour = d.get("colour", "main")
        self.zones = d.get("colours", [])
        # a chain-ring piece worked in turned rows, not joined (a collar with a V notch): it lies flat and each row
        # covers only as much of the circle as its stitches reach; the gap is at the front (az 0)
        self.split = bool(d.get("split")) and self.start == "ring"
        # a last round of shells (scallops): its stitches rise and fall along the round
        self.shell = self._shell_setup() if self.rounds[-1].loop == "shell" else None
        self.opening, self.opening_errors = self._opening_setup(d.get("opening"))

    def _opening_setup(self, op):
        """`opening: {"from": k, "stitches": g}`: after round k the piece is worked in turned rows that leave the g
        stitches at the centre front (az 0, where the rounds start) unworked - the face opening of a hood or a bob,
        the neck opening of a cape. The rounds keep their full counts in the spec (the circle the rows go round);
        each written row has count - g stitches. The unworked window must be plain sc in every such round (the
        increases and decreases of a row are worked in the part that exists)."""
        if not op:
            return None, []
        k, g = int(op.get("from", 0)), int(op.get("stitches", 0))
        errs = []
        if self.closed:
            errs.append("opening: only an open piece can have a front opening (end: open)")
        if not 1 <= k < self.N:
            errs.append(f"opening: 'from' must be a round 1-{self.N - 1} (got {k})")
        if g < 2:
            errs.append(f"opening: at least 2 stitches must be left unworked (got {g})")
        if errs:
            return None, errs
        left, right = g // 2, g - g // 2
        for i in range(k, self.N):                       # rounds k+1 .. N (0-based i)
            r = self.rounds[i]
            if g >= r.n - 3:
                errs.append(f"opening: round {i + 1} has {r.n} st, too few for {g} unworked")
                continue
            ops = list(r.ops)
            if r.loop in ("rib", "shell", "popcorn") or any(o != "sc" for o in ops[:left] + ops[len(ops) - right:]):
                errs.append(f"opening: round {i + 1} changes count or texture inside the {g} unworked front stitches "
                            f"(only sc may sit there)")
        if errs:
            return None, errs
        return {"from": k, "stitches": g, "left": left, "right": right}, []

    def in_opening(self, t, th):
        """True where (round parameter t, angle th) lies in the unworked front window of the opening rows. The
        rounds start at th = 0 (centre front) and are worked toward decreasing th: the first `left` stitches of a
        row's round sit just below th = 2*pi, the last `right` ones just above 0."""
        if self.opening is None:
            return np.zeros(np.shape(t), bool)
        t = np.asarray(t, float)
        n = self.n_stitches(np.clip(t - 1e-6, 0, self.M)).astype(float)
        u = np.mod(np.asarray(th, float), 2 * np.pi) / (2 * np.pi) * n          # stitch coordinate from az 0
        o = self.opening
        win = (u < o["right"]) | (u > n - o["left"])
        return win & (t > o["from"] + 1e-6)

    def _shell_setup(self):
        """Height (in sc rounds) of every stitch of a shell edging round: the k stitches of a shell fan into a
        rounded scallop, the sl st between shells stays low."""
        HT = {"sc": 1.0, "hdc": 1.5, "dc": 2.0, "tr": 2.6}
        hs = []
        import re as _re
        for o in self.rounds[-1].ops:
            m = _re.fullmatch(r"shell(\d+)(sc|hdc|dc|tr)", o)
            if m:
                k = int(m.group(1)); H = HT[m.group(2)]
                hs += [H * (0.35 + 0.65 * np.sin(np.pi * (i + 0.5) / k) ** 0.7) for i in range(k)]
            elif o == "sl":
                hs.append(0.3)
        hs = np.asarray(hs, float)
        n = len(hs)
        th = 2 * np.pi * (np.arange(n) + 0.5) / n
        return np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi]), np.concatenate([hs, hs, hs])

    def _shell_t(self, t, th):
        """Round parameter stretched by the shell heights in the last round (t > N-1), may exceed M."""
        if self.shell is None:
            return t
        g, hs = self.shell
        f = np.interp(np.mod(th, 2 * np.pi), g, hs)
        v = t - (self.N - 1)
        return np.where(v > 0, (self.N - 1) + v * f, t)

    def _ext(self, fn, dfn, t):
        """fn(t) continued straight past the last boundary (shell stitches taller than one round)."""
        M = float(self.M)
        tc = np.clip(t, 0, M)
        return fn(tc) + np.maximum(t - M, 0.0) * dfn(M)

    # ---- profile ------------------------------------------------------------------------------------------
    def r_t(self, t):
        return np.maximum(self._r(np.clip(t, 0, self.M)), 0.0)

    def z_t(self, t):
        return self._z(np.clip(t, 0, self.M))

    def n_stitches(self, t):
        """Stitch count of the round containing parameter t (round i spans t in [i-1, i])."""
        i = np.clip(np.floor(t).astype(int), 0, self.N - 1)
        return np.asarray(self.counts)[i]

    def _section(self, r, th):
        """Cross-section point of radius r at angle th (circle, or stadium along the oval axis)."""
        if self.oval > 0:
            if self.oval_axis == "front":
                x, y = _stadium(r, self.oval, th + np.pi / 2)
                return y, -x
            return _stadium(r, self.oval, th)
        return r * np.cos(th), r * np.sin(th)

    def flat_points(self, t, th):
        """The piece laid flat (a sewn-on applique): polar coordinates by distance along the fabric."""
        if self.shell is not None:
            te = self._shell_t(np.asarray(t, float), np.asarray(th, float))
            s = self._ext(self._arc, self._arc.derivative(), te)
        else:
            s = self._arc(np.clip(t, 0, self.M))
        if self.oval > 0:
            x, y = self._section(s, th)
        else:
            x, y = s * np.cos(th), s * np.sin(th)
        return x, y

    def anchor(self, t, **kw):
        """Centre of the round boundary at t in the local frame (bends and squashes included)."""
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return self.local_points(np.full_like(th, float(t)), th, **kw).mean(axis=0)

    def max_radius(self):
        if getattr(self, "dish", 0) > 0:
            return float(self.dish_R + self.dish_rim)
        if getattr(self, "spoon", 0) > 0:
            return float(np.max(self.spoon_dims_from_r(self.rb)[0]))
        return float(self.rb.max() + self.oval / 2)

    # ---- local surface ------------------------------------------------------------------------------------
    def local_points(self, t, th, bend_phi=0.0, bend_free_start=False, squash=None, squash_phi=0.0,
                     squash_rounds=3.0, drape_table=None, drape_twist=None):
        """Base surface (before relief) in the local frame; t, th arrays of the same shape.

        bend_phi: direction of the bend in the local xy plane (0 = toward local +x).
        bend_free_start: the piece is sewn by its end, so the part before `at` (the free part) bends and the
        sewn part keeps its direction.
        squash / squash_phi: the opening pressed flat as it is sewn (a flat seam), along the local direction
        squash_phi (0 = local x), ramped over the last `squash_rounds` rounds; overrides the part's own squash."""
        t = np.asarray(t, float); th = np.asarray(th, float)
        if self.dish > 0:
            rho, z = self.dish_rz(t)
            return np.stack([rho * np.cos(th), rho * np.sin(th), z], axis=-1)
        if self.shell is not None or self.split:
            r, z, th = self._ring_rz(t, th)
        else:
            r = self.r_t(t); z = self.z_t(t)
        if self.spoon > 0:
            x, y = self._spoon(r, th)
        else:
            x, y = self._section(r, th)
        # uniform flattening along local x (an unstuffed ear pressed flat)
        if self.flatten > 0 and self.spoon <= 0:
            x, y = self._press(x, y, self.flatten)
        sq = self.squash if squash is None else squash
        if sq > 0:
            ramp = _smoothstep((t - (self.N - squash_rounds)) / squash_rounds) * sq
            c, s_ = np.cos(squash_phi), np.sin(squash_phi)
            xr, yr = c * x + s_ * y, -s_ * x + c * y
            xr, yr = self._press(xr, yr, ramp)
            x, y = c * xr - s_ * yr, s_ * xr + c * yr
        if self.bend:
            c, s_ = np.cos(bend_phi), np.sin(bend_phi)
            xr, yr = c * x + s_ * y, -s_ * x + c * y          # bend direction -> +x
            if drape_table is not None and bend_free_start:
                # a soft piece hanging from its seam (`bend.drape`): the axis follows the centre line the toy
                # computed (gravity + lying on the pieces under it), arc length = fabric length (nothing stretches)
                zt = float(self.z_t(self.M))
                s_t, ph_t, axx_t, axz_t = drape_table
                zz = zt - z
                if drape_twist:
                    # a pressed strip folds only across its width: below the seam it twists (over `drape_twist`
                    # mm) until its weak axis (+x, or -x if nearer) lies along the bend direction
                    tgt = bend_phi if np.cos(bend_phi) >= 0 else bend_phi - np.pi
                    tgt = (tgt + np.pi) % (2 * np.pi) - np.pi
                    tw_len, tw_frac = drape_twist
                    beta = tw_frac * tgt * _smoothstep(zz / max(tw_len, 1e-6))
                    cb, sb = np.cos(beta), np.sin(beta)
                    xt, yt = cb * x - sb * y, sb * x + cb * y
                    xr, yr = c * xt + s_ * yt, -s_ * xt + c * yt
                sv = np.clip(zz, 0.0, s_t[-1])
                ph = np.interp(sv, s_t, ph_t)
                ax_x = np.interp(sv, s_t, axx_t)
                ax_z = np.interp(sv, s_t, axz_t)
                beyond = np.maximum(zz - s_t[-1], 0.0)
                ax_x = ax_x + beyond * np.sin(ph)
                ax_z = ax_z + beyond * np.cos(ph)
                x0 = xr
                xr = ax_x + x0 * np.cos(ph)
                z = zt - (ax_z - x0 * np.sin(ph))
            elif self.bend.get("coil"):
                # a spiral (a seahorse / chameleon tail): see _coil; each turn drifts sideways by pitch_mm
                zt = float(self.z_t(self.M))
                if bend_free_start:
                    xr, zz, ph = self._coil(xr, zt - z, zt)
                    z = zt - zz
                else:
                    xr, z, ph = self._coil(xr, z, None)
                yr = yr + float(self.bend.get("pitch_mm", 2.2 * self.thick)) * np.abs(ph) / (2 * np.pi)
            elif bend_free_start:
                zt = float(self.z_t(self.M))
                xr, zz = self._bend(xr, zt - z, zt)
                z = zt - zz
            else:
                xr, z = self._bend(xr, z, None)
            x, y = c * xr - s_ * yr, s_ * xr + c * yr
        return np.stack([x, y, z], axis=-1)

    def _ring_rz(self, t, th):
        """Radius, height and angle for shell-edged or split (C-shaped) pieces."""
        if self.split:
            # worked in turned rows, laid flat: row k at distance k round heights from the chain; each row covers
            # the angle its stitches reach, centred at the back, so the gap (the notch) opens at the front
            te = self._shell_t(t, th)
            r0 = float(self.rb[0])
            r = r0 + np.maximum(te, 0.0) * self.h
            n = self.n_stitches(np.clip(t - 1e-6, 0, self.M))
            alpha = np.minimum(2 * np.pi, n * self.w / np.maximum(r, 1e-6))
            thw = np.where((th >= 0) & (th <= 2 * np.pi), th, np.mod(th, 2 * np.pi))   # keep the two ends apart
            return r, np.zeros_like(r), np.pi + (thw - np.pi) * alpha / (2 * np.pi)
        te = self._shell_t(t, th)
        r = np.maximum(self._ext(self._r, self._dr, te), 0.0)
        z = self._ext(self._z, self._dz, te)
        return r, z, th

    # ---- dish: a cup pressed along its axis -----------------------------------------------------------------
    def _dish_setup(self):
        """Front face = the rounds up to the first widest round, lying as a disc (radius = distance along the
        fabric from the ring), sagging back by `dish_depth` at the centre; the rounds after it go round a fold of
        radius `dish_rim` (at least two fabric layers; more straight rounds at full width = a fatter rim) and come
        back inward as a back layer nested behind the front, its radius following the stitch counts (decreases
        pull it in; a closed piece closes at the back centre)."""
        c = np.asarray(self.counts)
        mx = c.max()
        iw = int(np.argmax(c == mx)) + 1                    # boundary at the end of the first widest round
        il = int(len(c) - np.argmax(c[::-1] == mx))         # end of the last widest round
        s = self._arc(np.arange(self.M + 1, dtype=float))
        self.dish_tw = float(iw)
        self.dish_R = float(s[iw])
        wall = float(s[il] - s[iw])
        self.dish_rim = max(2.1 * self.thick, wall / np.pi)
        self.dish_depth = 0.5 * self.dish * self.dish_R
        self.dish_r0 = max(float(self.rb[iw]), 1e-6)

    def dish_rz(self, t):
        """(radius from the axis, depth behind the front rim plane) of the dish surface at round parameter t."""
        t = np.clip(np.asarray(t, float), 0, self.M)
        R, a, D, tw = self.dish_R, self.dish_rim, self.dish_depth, self.dish_tw
        s = self._arc(t)
        front = t <= tw
        rho_f = np.minimum(s, R)
        z_f = D * (1 - (rho_f / R) ** 2)
        u = np.maximum(s - self._arc(tw), 0.0)
        phi = np.minimum(np.pi, u / a)
        rho_c = np.clip(R * self.r_t(t) / self.dish_r0, 0.0, R)
        rho_b = rho_c + a * np.sin(phi)
        z_b = a * (1 - np.cos(phi)) + D * (1 - (rho_c / R) ** 2)
        return np.where(front, rho_f, rho_b), np.where(front, z_f, z_b)

    def dish_rim_t(self):
        """Round parameter of the outermost point of the rim (the middle of the fold)."""
        ts = np.linspace(self.dish_tw, self.M, 200)
        return float(ts[int(np.argmax(self.dish_rz(ts)[0]))])

    def spoon_gap(self):
        """Distance between the two fabric layers' mid-surfaces of a spoon-pressed piece (mm): two layers of sc
        fabric (each ~1.5 yarn thicknesses: sc fabric is thicker than its yarn) lying on each other. A spoon ear
        stuffed lightly (`stuff: light`, a thin layer between the walls) keeps its walls apart: a plump ear with thick
        rounded edges that is still hollow in front (medium / firm: thicker still)."""
        return 3.0 * self.thick * (1.0 + SPOON_PUFF.get(self.stuff, 0.0))

    def spoon_dims(self, t):
        """(half-width W, layer gap g, sag D) of the spoon section at round parameter t (arrays)."""
        return self.spoon_dims_from_r(self.r_t(np.asarray(t, float)))

    def _spoon(self, r, th):
        """Cross-section of a spoon-pressed cup: a stadium of two fabric layers `g` apart (rounded folds at the
        sides, stitches spread evenly along the fabric), its mid-line bowed back by D at the centre so the front
        is concave and the back convex. th = 0 is the centre of the front wall (local +x)."""
        r = np.asarray(r, float)
        W, g, D = self.spoon_dims_from_r(r)
        Ls = np.maximum(2 * (W - g / 2), 0.0)
        x, y = _stadium_arr(g / 2, Ls, th)
        # _stadium puts th = 0 at the start of a straight side (x = +g/2, y = 0 .. L/2): centre front
        u = np.clip(y / np.maximum(W, 1e-6), -1.0, 1.0)
        x = x - D * (1 - u * u) + D / 3.0                  # bowed back in the middle, centred on the axis
        return x, y

    def spoon_dims_from_r(self, r):
        r = np.asarray(r, float)
        P = 2 * np.pi * r + 2 * self.oval                 # the round's length of fabric
        g = np.minimum(self.spoon_gap(), P / np.pi)       # near the magic ring the section is just a round tube
        Ls = np.maximum(P - np.pi * g, 0.0) / 2           # straight length of each wall before bowing
        k = 0.42 * self.spoon                             # sag of the mid-line / half-width
        for _ in range(4):                                # a bowed wall is longer than its chord: keep the length
            W = Ls / 2 + g / 2
            D = k * W
            Ls = np.maximum(P - np.pi * g, 0.0) / 2 / (1 + 8.0 / 3.0 * (D / np.maximum(Ls, 1e-6)) ** 2)
        W = Ls / 2 + g / 2
        return W, g, k * W

    def _press(self, x, y, f):
        """Press a cross-section flat along x by f (0..1); the width along y grows to keep the perimeter."""
        rr = np.maximum(np.sqrt(x * x + y * y), 1e-6)
        tmin = np.minimum(1.0, 1.1 * self.thick / rr)
        sx = np.maximum(1 - f, tmin)
        return x * sx, y * (1 + (np.pi / 2 - 1) * (1 - sx))

    def _coil(self, x, z, flip_total):
        """`bend` with `coil: q` (q > 1, default 4 when true): from the bend's start to the free tip the axis curls
        with a curvature that grows q-fold toward the tip, so it winds inward (a spiral) and can turn past 360°
        without passing through itself; the crocheter makes it by rolling the tail up from the tip and tacking
        the turns. Returns (x, z, turned angle) in the bend's frame."""
        q = float(self.bend.get("coil")) if not isinstance(self.bend.get("coil"), bool) else 4.0
        q = max(q, 1.0)
        deg = np.radians(float(self.bend.get("deg", 0.0)))
        sg = 1.0 if deg >= 0 else -1.0
        at = float(self.bend.get("at", self.N / 2)) - 0.5
        over = float(self.bend.get("over", 2.0))
        zs = float(self.z_t(at - over / 2))
        zmax = float(np.max(self.zb))
        zmin = float(np.min(self.zb))
        if flip_total is not None:
            zs = flip_total - float(self.z_t(at + over / 2))
            zend = flip_total - zmin
        else:
            zend = zmax
        L = max(zend - zs, 1e-6)
        sg_, phi_t, axx_t, axz_t = _coil_table(L, abs(deg), q)
        sv = np.clip(z - zs, 0.0, L)
        phi = np.interp(sv, sg_, phi_t)
        ax_x = np.interp(sv, sg_, axx_t)
        ax_z = zs + np.interp(sv, sg_, axz_t)
        beyond = np.maximum(z - zend, 0.0)
        ax_x = ax_x + beyond * np.sin(phi)
        ax_z = ax_z + beyond * np.cos(phi)
        below = z < zs
        bx = sg * ax_x + x * np.cos(phi)
        bz = ax_z - sg * x * np.sin(phi)
        return (np.where(below, x, bx), np.where(below, z, bz), np.where(below, 0.0, sg * phi))

    def _bend(self, x, z, flip_total):
        deg = np.radians(float(self.bend.get("deg", 0.0)))
        at = float(self.bend.get("at", self.N / 2)) - 0.5      # round r = the middle of round r
        over = float(self.bend.get("over", 2.0))
        z0 = float(self.z_t(at - over / 2)); z1 = float(self.z_t(at + over / 2))
        if flip_total is not None:
            z0, z1 = flip_total - z1, flip_total - z0
        span = max(z1 - z0, 1e-6)
        R = span / max(abs(deg), 1e-6)          # bend radius so the arc over the span turns by deg
        zc = np.clip(z - z0, 0, span)
        phi = deg * zc / span
        # point on the bent axis + offset x across it
        ax_x = R * (1 - np.cos(phi)) * np.sign(deg)
        ax_z = z0 + R * np.sin(abs(phi))
        extra = np.maximum(z - z1, 0)
        ax_x = ax_x + extra * np.sin(phi)
        ax_z = ax_z + extra * np.cos(phi)
        below = z < z0
        nx = np.cos(phi); nz = -np.sin(phi)
        bx = ax_x + x * nx
        bz = ax_z + x * nz
        return np.where(below, x, bx), np.where(below, z, bz)


def _coil_table(L, deg, q, n=600):
    """Centre line of a spiral of length L turning `deg` in total, its curvature growing linearly to q times the
    start's toward the tip: angle, x and z along it (z = 0 at the start)."""
    s = np.linspace(0.0, L, n)
    k0 = deg / max(L * (q + 1) / 2, 1e-6)
    phi = k0 * (s + (q - 1) * s * s / (2 * max(L, 1e-6)))
    ds = np.diff(s, prepend=0.0)
    ax_x = np.cumsum(np.sin(phi) * ds)
    ax_z = np.cumsum(np.cos(phi) * ds)
    return s, phi, ax_x, ax_z


def _stadium(r, L, th):
    """Point on a stadium of half-width r and straight length L (long axis along local y), by perimeter fraction.

    A chain-oval start: the rounds run along both sides of the chain and turn around its ends."""
    r = np.broadcast_to(np.asarray(r, float), np.shape(th)).copy()
    per = 2 * L + 2 * np.pi * r
    s = np.mod(th, 2 * np.pi) / (2 * np.pi) * per
    cy = L / 2
    e1 = L / 2; e2 = e1 + np.pi * r; e3 = e2 + L; e4 = e3 + np.pi * r
    x = np.empty_like(s); y = np.empty_like(s)
    safe = np.maximum(r, 1e-9)
    m = s < e1;                    x[m] = r[m];  y[m] = s[m]
    m = (s >= e1) & (s < e2);      a = (s[m] - e1[m] if np.ndim(e1) else s[m] - e1) / safe[m]
    x[m] = r[m] * np.cos(a);       y[m] = cy + r[m] * np.sin(a)
    m = (s >= e2) & (s < e3);      x[m] = -r[m]; y[m] = cy - (s[m] - e2[m])
    m = (s >= e3) & (s < e4);      a = np.pi + (s[m] - e3[m]) / safe[m]
    x[m] = r[m] * np.cos(a);       y[m] = -cy + r[m] * np.sin(a)
    m = s >= e4;                   x[m] = r[m];  y[m] = -cy + (s[m] - e4[m])
    return x, y


def _stadium_arr(r, L, th):
    """_stadium for arrays of r and L (each round its own size): same parameterisation by perimeter fraction,
    th = 0 at the centre of the +x side."""
    th = np.asarray(th, float)
    r = np.broadcast_to(np.asarray(r, float), th.shape)
    L = np.broadcast_to(np.asarray(L, float), th.shape)
    per = 2 * L + 2 * np.pi * r
    s = np.mod(th, 2 * np.pi) / (2 * np.pi) * per
    e1 = L / 2; e2 = e1 + np.pi * r; e3 = e2 + L; e4 = e3 + np.pi * r
    safe = np.maximum(r, 1e-9)
    a1 = (s - e1) / safe
    a2 = np.pi + (s - e3) / safe
    x = np.select([s < e1, s < e2, s < e3, s < e4],
                  [r, r * np.cos(a1), -r, r * np.cos(a2)], r)
    y = np.select([s < e1, s < e2, s < e3, s < e4],
                  [s, L / 2 + r * np.sin(a1), L / 2 - (s - e2), -L / 2 + r * np.sin(a2)], -L / 2 + (s - e4))
    return x, y
