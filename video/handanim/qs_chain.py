"""Crochet chain, hand-animated hook + quasi-static yarn (no dynamics, so nothing shakes or swings by itself).

The hook motion is keyframed by hand. The yarn is one strand from the tail (held by the other hand) to the
finger eyelet G, solved every sub-step as a taut string: it is pulled tight (tension), it can not go through the
hook or through itself, and it slides along itself (fed from the finger) to get the length it needs. The only
hand-authored help is the throat: while a stitch is being pulled through, the yarn that lies in the throat is held
in it (the job the crocheter's tension and the hook's lip do).

Because it is solved from the previous sub-step and moves only by small steps, the yarn can not jump, appear or pass
through anything; check_anim.py verifies the output frame by frame.

Usage: python qs_chain.py <out.json> [n_stitches=4] [skip_first=1]
"""
import json, math, sys
import numpy as np
from scipy.spatial import cKDTree

R = 0.95                      # yarn radius
RS, RNECK, XTH = 1.06, 0.51, -2.05
H_SP = 0.36                   # point spacing along the yarn
FPS = 30
SUB = 4                       # sub-steps per frame
ITERS = 14                    # solver iterations per sub-step
TAU = 0.22                    # tension (curve shortening) per iteration
NEIGH = int(4.2 / H_SP)       # points closer than this along the yarn never collide


def norm(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / (n or 1.0)


def ease(u):
    u = min(1.0, max(0.0, u)); return u * u * u * (u * (u * 6 - 15) + 10)


def rot(v, axis, ang):
    axis = norm(axis); v = np.asarray(v, float)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1 - math.cos(ang))


class Pose:
    def __init__(self, tip, a, d):
        self.tip = np.asarray(tip, float); self.a = norm(a)
        self.d = norm(np.asarray(d, float) - self.a * np.dot(d, self.a)); self.s = np.cross(self.a, self.d)

    def pt(self, x, y=0.0, z=0.0):
        return self.tip + self.a * x + self.d * y + self.s * z

    def local(self, p):
        q = np.asarray(p) - self.tip
        return np.stack([q @ self.a, q @ self.d, q @ self.s], axis=-1)

    def world(self, L):
        return self.tip + np.outer(L[:, 0], self.a) + np.outer(L[:, 1], self.d) + np.outer(L[:, 2], self.s)

    def lerp(self, o, w):
        return Pose(self.tip + (o.tip - self.tip) * w, slerp(self.a, o.a, w), slerp(self.d, o.d, w))

    def js(self):
        return dict(tip=self.tip.round(4).tolist(), a=self.a.round(5).tolist(), d=self.d.round(5).tolist())


def slerp(u, v, w):
    u, v = norm(u), norm(v); c = float(np.clip(np.dot(u, v), -1, 1)); th = math.acos(c)
    if th < 1e-6: return u
    return (u * math.sin((1 - w) * th) + v * math.sin(w * th)) / math.sin(th)


# ---------------------------------------------------------------- hook body (same as engine/hook.mjs)
def base_radius(x):
    x = np.asarray(x, float)
    u = (x + 1.15) / 1.15
    nose = RS * 0.92 * np.sqrt(np.clip(1 - u * u, 0, None)) + 0.02
    waist = 0.07 * np.exp(-(((x + 1.4) / 0.5) ** 2))
    body = np.where(x > -1.6, RS * 0.92, RS) - waist
    return np.where(x > 0, 0.0, np.where(x > -1.15, nose, body))


def groove(x):
    x = np.asarray(x, float)
    x0, xm, x1 = -4.4, -2.0, -1.45; Dg = RS - RNECK
    t1 = np.clip((x - x0) / (xm - x0), 0, 1); t2 = np.clip((x - xm) / (x1 - xm), 0, 1)
    g = np.where(x < xm, Dg * t1 * t1 * (3 - 2 * t1), Dg * (1 - t2 ** 3))
    return np.where((x <= x0) | (x >= x1), 0.0, g)


def surf(x, phi):
    c = np.clip(np.cos(phi), 0, None)
    return np.maximum(0.05, base_radius(x) - groove(x) * np.minimum(1.0, c ** 0.7 * 1.15))


def hook_push(H, P, ry=R):
    L = H.local(P); x, y, z = L[:, 0], L[:, 1], L[:, 2]
    rr = np.hypot(y, z); phi = np.arctan2(z, y)
    need = surf(x, phi) + ry * 0.98
    on_body = (x <= 0) & (x > -80) & (rr < need)
    sc = np.where(rr > 1e-6, need / np.maximum(rr, 1e-6), 1.0)
    L2 = L.copy()
    L2[on_body, 1] = np.where(rr[on_body] > 1e-6, y[on_body] * sc[on_body], -need[on_body])
    L2[on_body, 2] = np.where(rr[on_body] > 1e-6, z[on_body] * sc[on_body], 0.0)
    rt = np.linalg.norm(L, axis=1); cap = (x > 0) & (rt < ry * 0.98)
    L2[cap] = L[cap] / np.maximum(rt[cap], 1e-6)[:, None] * ry * 0.98
    moved = on_body | cap
    if moved.any(): P[moved] = H.world(L2[moved])
    return P


def throat_hold(H, P, w=0.9, sel=None):
    """Yarn caught by the hook stays at the throat: anything wrapped round the head/neck is kept in the throat's
    length (the lip stops it sliding off the tip), and on the throat side it sits down in the groove."""
    L = H.local(P); x, y, z = L[:, 0], L[:, 1], L[:, 2]
    rr = np.hypot(y, z); phi = np.arctan2(z, y)
    m = (x > -4.3) & (x < 1.2) & (rr < RS + R + 1.0)
    if sel is not None: m &= sel
    if not m.any(): return m
    L2 = L.copy()
    L2[m, 0] = x[m] + w * (np.clip(x[m], XTH - 0.5, XTH + 0.5) - x[m])
    side = m & (np.cos(phi) > 0.3)
    want = RNECK + R
    f = np.where(rr[side] > want, (rr[side] + w * (want - rr[side])) / np.maximum(rr[side], 1e-6), 1.0)
    L2[side, 1] = y[side] * f; L2[side, 2] = z[side] * f
    P[m] = H.world(L2[m])
    return m


# ---------------------------------------------------------------- yarn solver
def resample(P, h=H_SP, keep_last=True):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(round(s[-1] / h)) + 1)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, k]) for k in range(3)], axis=1)


def self_collide(P, w=None, iters=1):
    d0 = 2 * R * 1.1  # a little margin: the tension always wins back a bit
    for _ in range(iters):
        pairs = cKDTree(P).query_pairs(d0, output_type='ndarray')
        if len(pairs) == 0: return P
        pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > NEIGH]
        if len(pairs) == 0: return P
        i, j = pairs[:, 0], pairs[:, 1]
        v = P[i] - P[j]; d = np.linalg.norm(v, axis=1)
        nrm = v / np.maximum(d, 1e-6)[:, None]
        corr = (d0 - d)[:, None] * nrm
        wi = np.ones(len(i)) * 0.5 if w is None else w[i] / np.maximum(w[i] + w[j], 1e-9)
        wj = 1 - wi if w is not None else wi
        acc = np.zeros_like(P); cnt = np.zeros(len(P))
        np.add.at(acc, i, corr * wi[:, None]); np.add.at(acc, j, -corr * wj[:, None])
        np.add.at(cnt, i, 1); np.add.at(cnt, j, 1)
        k = cnt > 0
        P[k] += acc[k] / np.maximum(1, cnt[k] ** 0.5)[:, None] * 1.1
    return P


def tension(P, tau, step_max=0.15):
    """Multi-scale curve shortening: free spans straighten in a few iterations, not N^2."""
    n = len(P); disp = np.zeros_like(P)
    for k in (1, 2, 4, 8, 16):
        if 2 * k >= n: break
        i = np.arange(1, n - 1)
        lo = np.maximum(i - k, 0); hi = np.minimum(i + k, n - 1)
        wl = (i - lo) / k; wh = (hi - i) / k  # shorter arm near the ends: weight the midpoint properly
        mid = (P[lo] * wh[:, None] + P[hi] * wl[:, None]) / (wl + wh)[:, None]
        disp[1:-1] += tau * (mid - P[1:-1]) / (1 + 0.5 * math.log2(k))
    m = np.linalg.norm(disp, axis=1); sc = np.minimum(1, step_max / np.maximum(m, 1e-9))
    return P + disp * sc[:, None]


def caught_curve(n, sign):
    """The caught yarn hugging the throat (hook-local): round the neck on the throat side."""
    ph = np.radians(np.linspace(-95, 95, n)) * sign
    x = np.full(n, XTH)
    r = surf(x, ph) + R + 0.03
    return np.stack([x, r * np.cos(ph), r * np.sin(ph)], axis=1)


def seated_curve(n, sign, w, c0=0.0, c1=0.0, th=0.0):
    """Caught yarn -> loop round the shaft (w: 0..1), in world-fixed angles: the U in the throat (centred on the
    throat's world angle c0) becomes a ring hanging over the shaft (centred on c1, its legs down to the work).
    th: the hook's present roll; the loop does not turn with the hook once it is out of the throat."""
    U = np.linspace(-95, 95, n) * sign + c0
    F = np.linspace(-160, 160, n) * sign + c1
    ph = np.radians(U + (F - U) * w - th)
    x = np.full(n, XTH + (-6.0 - XTH) * w)
    r = np.maximum(surf(XTH, ph) + R + 0.03, surf(x, ph) + R + 0.05)
    return np.stack([x, r * np.cos(ph), r * np.sin(ph)], axis=1)


def roll(H):
    """World angle of the throat round the shaft (degrees, 0 = up, 90 = away from the viewer)."""
    return math.degrees(math.atan2(-H.d[2], H.d[1]))


def curve_len(C):
    return float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1)))


def resample_n(C, n):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, C[:, k]) for k in range(3)], axis=1)


def resample_free(P, fixed):
    """Resample the free spans between the fixed (authored) pieces; the fixed pieces keep their points."""
    out = []; idx = []; a = 0
    for (i0, i1) in fixed:
        F = resample(P[a:i0 + 1]) if i0 > a else P[a:a + 1]
        if out: F = F[1:]
        out.append(F); n0 = sum(len(o) for o in out) - 1
        out.append(P[i0 + 1:i1 + 1]); idx.append((n0, n0 + i1 - i0)); a = i1
    F = resample(P[a:]); out.append(F[1:] if out else F)
    return np.vstack(out), idx


def link_ring(n, sign, rho=None):
    """A finished link (hook-free): a flat ring round a vertical axis, open toward +x (the handle side); its first end
    is where the loop's first leg was, both ends lead down to the link below."""
    span = math.radians(320)
    if rho is None: rho = (n - 1) * H_SP / span
    ps = np.radians(20) + np.linspace(0, span, n)
    if sign < 0: ps = -ps
    return np.outer(rho * np.cos(ps), [1.0, 0, 0]) + np.outer(rho * np.sin(ps), LINK_B)


LINK_TILT = math.radians(55)   # links face the viewer this much (0: flat rings round a vertical axis)
LINK_B = np.array([0.0, -math.sin(LINK_TILT), math.cos(LINK_TILT)])
LINK_N = np.array([0.0, math.cos(LINK_TILT), math.sin(LINK_TILT)])


class Yarn:
    """The strand from the finished work (anchor) to the finger eyelet G. Two pieces of it can be authored:
    att (the loop / caught yarn carried by the hook, hook-local shape) and link (the old loop closing into a chain
    link, held by the other hand). Everything else is solved as a taut string."""
    def __init__(self, P, anchor, G):
        self.P = resample(np.asarray(P, float)); self.anchor = np.asarray(anchor, float); self.G = np.asarray(G, float)
        self.att = None; self.H = None; self.link = None; self.old = None
        self.frozen = np.zeros((0, 3))

    def freeze(self, f):
        self.frozen = np.vstack([self.frozen, self.P[:f]])
        self.P = self.P[f:].copy(); self.anchor = self.P[0].copy()
        for a in (self.att, self.link):
            if a is not None: a['i0'] -= f; a['i1'] -= f

    def move_work(self, delta):
        self.frozen = self.frozen + delta; self.anchor = self.anchor + delta; self.P[0] = self.anchor

    def target(self, mode, u):
        a = self.att
        if mode == 'catch':
            n = a['n0']; U = caught_curve(n, a['sign'])
            return a['L0'] + (U - a['L0']) * ease(u)
        if mode == 'hold':
            a['c0'] = roll(self.H)
            return caught_curve(a['n0'], a['sign'])
        # the loop turns with the hook (it was turned throat-up while still in the throat)
        C = seated_curve(64, a['sign'], 1.0 if mode == 'ring' else ease(u))  # 'seat' / 'ring'
        n = max(4, int(round(curve_len(C) / H_SP)) + 1)
        return resample_n(C, n)

    def link_target(self):
        k = self.link; e = ease(k['u'])
        c = k['c0'] + (k['c1'] - k['c0']) * e
        return c + k['C0'] + (k['A'] - k['C0']) * e

    def pieces(self):
        return sorted([(p['i0'], p['i1']) for p in (self.att, self.link) if p is not None])

    def step(self, H, mode=None, u=0.0, iters=ITERS, tau=TAU, stick=False, slip=False):
        P = self.P
        held_loop = None
        if slip and self.H is not None:
            # the old loop is held by the other hand: the hook slides through it, it does not travel with the hook
            Lh = self.H.local(P); rrh = np.hypot(Lh[:, 1], Lh[:, 2])
            held_loop = (Lh[:, 0] < 0.8) & (Lh[:, 0] > -12) & (rrh < RS + R + 2.8)
            if self.att is not None: held_loop[self.att['i0']:self.att['i1'] + 1] = False
            X0 = P[held_loop].copy(); ax = self.H.a.copy()
        on = None
        if stick and self.H is not None:
            L0 = self.H.local(P); rr0 = np.hypot(L0[:, 1], L0[:, 2])
            on = (L0[:, 0] < -3.6) & (rr0 < RS + R + 0.35)
            # only the loop that was on the hook sticks (the working yarn slides over the shaft freely)
            keep = np.zeros(len(P), bool)
            if self.old is not None: keep[max(0, self.old['i0'] - 3):self.old['i1'] + 4] = True
            on &= keep
            x0 = L0[on, 0]
        self.H = H
        if mode is None or (mode == 'catch' and self.att is not None and self.att.get('kind') == 'ring'):
            if self.att is not None:
                self.old = dict(i0=self.att['i0'], i1=self.att['i1'], sign=self.att['sign'])
            self.att = None  # the loop on the hook is let go: it is just yarn round the shaft now
        if mode == 'ring' and self.att is not None:
            self.att['kind'] = 'ring'
        if mode is not None and self.att is None:  # catch: the bit of yarn lying across the throat
            tb = H.pt(XTH, RNECK + R)
            K = int(round(2.7 / H_SP))
            dd = np.linalg.norm(P - tb, axis=1)
            lo = K + 2
            hi = len(P) - K - 2
            if getattr(self, 'catch_hint', None) is not None:
                lo = max(lo, self.catch_hint - 8); hi = min(hi, self.catch_hint + 9)
            c = lo + int(np.argmin(dd[lo:hi]))
            i0, i1 = c - K, c + K
            Lc = H.local(P[i0:i1 + 1])
            sign = 1.0 if np.arctan2(Lc[0, 2], Lc[0, 1]) < np.arctan2(Lc[-1, 2], Lc[-1, 1]) else -1.0
            self.att = dict(i0=i0, i1=i1, n0=i1 - i0 + 1, L0=Lc, sign=sign)
        for it in range(iters):
            P[0] = self.anchor; P[-1] = self.G
            P = tension(P, tau)
            P = hook_push(H, P)
            w = np.ones(len(P))
            for (i0, i1) in self.pieces(): w[i0:i1 + 1] = 1e-6
            nf = min(len(self.frozen), 120)
            if nf:
                Q = np.vstack([self.frozen[len(self.frozen) - nf:], P])
                wq = np.concatenate([np.full(nf, 1e-6), w])
                P = self_collide(Q, wq, iters=4)[nf:]
            else:
                P = self_collide(P, w, iters=4)
            P = hook_push(H, P)
            if held_loop is not None and held_loop.any() and len(held_loop) == len(P):
                dX = P[held_loop] - X0
                P[held_loop] = X0 + dX - np.outer(dX @ ax, ax)
            if on is not None and on.any() and len(P) == len(on):
                L = H.local(P[on]); L[:, 0] = x0; P[on] = H.world(L)
            if self.link is not None:
                k = self.link; T = self.link_target(); P[k['i0']:k['i1'] + 1] = T
            if self.att is not None:
                T = H.world(self.target(mode, u))
                i0, i1 = self.att['i0'], self.att['i1']
                if len(T) != i1 - i0 + 1:
                    P = np.vstack([P[:i0], T, P[i1 + 1:]]); self.att['i1'] = i0 + len(T) - 1
                    on = None; held_loop = None
                else:
                    P[i0:i1 + 1] = T
            P[0] = self.anchor; P[-1] = self.G
        fx = self.pieces()
        P2, idx = resample_free(P, fx)
        for (i0, i1), (j0, j1) in zip(fx, idx):
            for a in (self.att, self.link):
                if a is not None and a['i0'] == i0: a['i0'], a['i1'] = j0, j1
        if self.old is not None and fx:
            # the old loop's indices move with the free span it is in
            sh = 0
            for (i0, i1), (j0, j1) in zip(fx, idx):
                if i1 < self.old['i0']: sh = j1 - i1
            self.old['i0'] += sh; self.old['i1'] += sh
        k = np.ones(len(P2), bool)
        for (j0, j1) in idx: k[j0:j1 + 1] = False
        Q = hook_push(H, P2.copy()); P2[k] = Q[k]
        self.P = P2
        return self.P


# ---------------------------------------------------------------- scene
HOME = Pose([0, 0, 0], [-1, 0, 0], [0, 1, 0])
XL = 6.0
G0 = np.array([9.0, 13.0, -9.0])          # finger eyelet at rest: up-behind the hook (the loop stays put)
GK = dict(low=np.array([7.0, -12.0, -12.0]), rest=G0, mid=np.array([4.0, 15.0, -10.0]), left=np.array([0.5, 14.0, -9.0]))
LDROP = 6.5      # the newest link sits this far under the shaft (held there by the other hand)
PITCH = 4.0      # chain links are this far apart
RHO = 2.9        # link ring (centreline) radius: holds the two legs of the next loop
TAIL = 22.0
C_LINK = HOME.pt(-XL) + np.array([0.0, -LDROP, 0.0])


def initial_yarn():
    """A slip knot: tail up to ring0 (a finished link under the hook), from its end under it and up through it to the
    loop on the hook, round the shaft, down through ring0 again, round under its back and up to the finger."""
    c0 = C_LINK
    n0 = int(round(RHO * math.radians(320) / H_SP)) + 1
    ring0 = c0 + link_ring(n0, 1.0, RHO)
    tail = resample(np.array([ring0[0] + [0, -TAIL, 0], ring0[0] + [0, -2.0, 0]]))
    frozen = np.vstack([tail, resample(np.vstack([tail[-1], ring0[0]]))[1:-1], ring0])
    loop = HOME.world(resample_n(seated_curve(64, 1.0, 1.0), 40))
    e = ring0[-1]
    pts = [e, e - LINK_N * 1.6 + [-0.5, 0, 0], c0 - LINK_N * 2.4 + [0.6, 0, 0], c0 + [0.4, -0.4, 0.4], loop[0] + [0, -1.2, 0.2]]
    pts += list(loop)
    back = c0 + RHO * (math.cos(math.radians(-72)) * np.array([1.0, 0, 0]) + math.sin(math.radians(-72)) * LINK_B)
    pts += [loop[-1] + [0, -1.2, -0.2], c0 + [-0.4, -0.4, -0.4], c0 - LINK_N * 2.6, back - LINK_N * 2.4,
            back + norm(back - c0) * 2.4, back + [0, 4, -2.0], G0]
    Y = Yarn(resample(np.array(pts)), e, G0)
    Y.frozen = frozen
    # the loop on the hook is carried as an authored ring
    d = np.linalg.norm(Y.P - loop[0], axis=1); i0 = int(np.argmin(d))
    d = np.linalg.norm(Y.P - loop[-1], axis=1); i1 = int(np.argmin(d))
    Y.H = HOME
    Y.att = dict(i0=i0, i1=i1, n0=i1 - i0 + 1, L0=None, sign=1.0, kind='ring')
    return Y


# hook poses for one chain stitch (world mm)
TURN = Pose([1.0, -0.2, -0.6], [-1, 0.0, 0.0], [0, -0.85, 0.5])   # throat turned down-back, back at the loop
PULL = Pose([15.0, -0.6, -0.6], [-1, 0.0, 0.0], [0, -0.85, 0.5])   # pulled back through the loop
UP = Pose([15.0, 1.8, -0.5], [-1, 0.0, 0.0], [0, 1.0, 0.0])         # throat up: the new loop is on the hook


def catch_pose(Y, back=0.0):
    """Throat facing back, its groove on the working yarn where it passes behind the head (back>0: still in front)."""
    xw = HOME.pt(XTH)[0]
    P = Y.P
    start = Y.att['i1'] + 1 if Y.att is not None else len(P) // 2
    start += int(np.argmin(P[start:, 1]))  # past the turn under the last link: the working yarn
    k0 = start + int(np.argmin(np.abs(P[start:, 0] - xw) + 50 * (P[start:, 2] > -1.5) + 50 * (np.abs(P[start:, 1]) > 3.0)))
    Y.catch_hint = k0
    target = P[k0] + np.array([0, 0, back])
    a = np.array([-1.0, 0, 0]); d = np.array([0, 0.25, -1.0]); d = norm(d)
    return Pose(target - a * XTH - d * (RNECK + R), a, d)


def path(keys, u):
    if len(keys) == 1: return GK[keys[0]].copy()
    x = u * (len(keys) - 1); i = min(int(x), len(keys) - 2); w = x - i
    pts = [GK[k] for k in keys]
    p0 = pts[max(i - 1, 0)]; p1 = pts[i]; p2 = pts[i + 1]; p3 = pts[min(i + 2, len(pts) - 1)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * w + (2 * p0 - 5 * p1 + 4 * p2 - p3) * w * w + (-p0 + 3 * p1 - 3 * p2 + p3) * w ** 3)


def start_link(Y):
    """The old loop (now round the new loop's legs) becomes a link: from its present shape it is drawn into a flat ring
    under the hook by the other hand."""
    o = Y.old
    n = int(round(RHO * math.radians(320) / H_SP)) + 1
    mid = (o['i0'] + o['i1']) // 2
    i0 = max(1, mid - n // 2); i1 = i0 + n - 1
    if Y.att is not None and i1 > Y.att['i0'] - 3:
        i1 = Y.att['i0'] - 3; i0 = max(1, i1 - n + 1)
    n = i1 - i0 + 1
    C = Y.P[i0:i1 + 1].copy(); c0 = C.mean(axis=0)
    Y.link = dict(i0=i0, i1=i1, C0=C - c0, A=link_ring(n, o['sign'], RHO), c0=c0, c1=C_LINK.copy(), u=0.0)


def run(n_st, skip, out_path):
    Y = initial_yarn()
    for _ in range(60): Y.step(HOME, mode='ring', iters=20)
    frames = []
    t = 0.0
    for st in range(n_st):
        poses = dict(home=HOME, turn=TURN, pull=PULL, up=UP)
        # (hook from, hook to, finger path, seconds, throat mode, loop sticks to the shaft)
        seq = [('home', 'home', ['rest'], 0.4, 'ring', True),
               ('home', 'home', ['rest', 'mid', 'left'], 1.0, 'ring', True),  # the yarn is brought behind the head
               ('home', 'near', ['left'], 0.6, 'ring', True),                 # throat turns back, toward it
               ('near', 'catch', ['left'], 0.4, 'catch', True),               # and takes it (yarn over)
               ('catch', 'turn', ['left'], 0.6, 'hold', True),                # throat turns down with the yarn
               ('turn', 'pull', ['left'], 1.3, 'hold', False),                # pulled through the loop on the hook
               ('pull', 'up', ['left', 'mid'], 1.2, 'hold', False),           # throat up: it is the new loop
               ('up', 'home', ['mid', 'rest'], 1.1, 'seat', False),           # the hook goes into the new loop
               ('home', 'home', ['rest'], 0.3, 'ring', True)]
        T_LINK = 1.2 + 1.1; tl = 0.0
        for (p0, p1, gk, dur, mode, stick) in seq:
            if p1 == 'near':
                Y.G = path(gk, 1.0)
                poses['catch'] = catch_pose(Y); poses['near'] = catch_pose(Y, back=2.2)
            if p0 == 'pull' and Y.old is not None:
                start_link(Y)
            if p0 == 'home' and p1 == 'home' and Y.link is not None:
                Y.freeze(Y.link['i1'] + 1); Y.link = None; Y.old = None
            moved = 0.0
            A, B = poses[p0], poses[p1]
            nfr = int(round(dur * FPS))
            for k in range(nfr):
                for sbs in range(SUB):
                    u = (k + (sbs + 1) / SUB) / nfr
                    H = A.lerp(B, ease(u))
                    Y.G = path(gk, ease(u))
                    if p0 == 'turn':  # the other hand lowers the work by one link while the hook pulls through
                        Y.move_work(np.array([0.0, -PITCH, 0.0]) * (ease(u) - moved)); moved = ease(u)
                    if Y.link is not None and p0 in ('pull', 'up'):
                        tl += dur / nfr / SUB; Y.link['u'] = min(1.0, tl / T_LINK)
                    Y.step(H, mode=mode, u=u, stick=stick, slip=(p0 == 'turn'))
                t += 1 / FPS
                if st >= skip:
                    frames.append(frame_json(Y, H, t, st))
        print("stitch", st, "points", len(Y.P), "frozen", len(Y.frozen), flush=True)
    cam = dict(pos=[18, 4, 46], look=[5, -3, -2], fov=36)
    json.dump(dict(fps=FPS, r=R, frames=frames, cam=cam, G=G0.tolist()), open(out_path, "w"))
    print("frames", len(frames))


def frame_json(Y, H, t, st):
    beyond = Y.G + norm(Y.G - Y.P[-8]) * np.array([[5.0], [12.0]])
    P = np.vstack([Y.frozen, Y.P, beyond])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    nf = len(Y.frozen)
    att = None if Y.att is None else [Y.att['i0'] + nf, Y.att['i1'] + nf]
    link = None if Y.link is None else [Y.link['i0'] + nf, Y.link['i1'] + nf]
    return dict(t=round(t, 3), stitch=st, pts=P.round(3).tolist(), s=s.round(3).tolist(), hook=H.js(), att=att,
                link=link, nfrozen=nf)


if __name__ == "__main__":
    out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4; sk = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    run(n, sk, out)
